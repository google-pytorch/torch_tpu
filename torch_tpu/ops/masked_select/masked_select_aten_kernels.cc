/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "torch_tpu/ops/masked_select/masked_select_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/broadcast_tensors.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"
#include "torch_tpu/ops/gather/gather.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nonzero/nonzero.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {

namespace {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildMaskedSelectShlo(int64_t output_size,
                                                   mlir::MlirOp self,
                                                   mlir::MlirOp mask) {
  mlir::MlirBuilder& builder = self.getBuilder();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self_type.getElementType()));

  if (output_size == 0) {
    return MakeConstant(builder, 0, output_dtype, {0});
  }

  mlir::MlirOp self_flat = Flatten(self);
  mlir::MlirOp mask_flat = Flatten(mask);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp indices_2d,
                      BuildNonzeroShlo(output_size, mask_flat));

  mlir::MlirOp indices_1d = stablehlo::Reshape(indices_2d, {output_size});

  return BuildGatherShlo(self_flat, /*dim=*/0, indices_1d,
                         /*sparse_grad=*/false, output_dtype);
}

// GetMaskedSelectOutputSize returns the output size of the masked_select op by
// summing the mask.
absl::StatusOr<at::Tensor> GetMaskedSelectOutputSize(const at::Tensor& mask) {
  const c10::ScalarType scalar_dtype = c10::ScalarType::Long;
  TT_ASSIGN_OR_RETURN(
      at::Tensor result,
      ApplySumReduction(mask, std::nullopt, ReductionMode::kDropDims,
                        scalar_dtype));
  return result;
}

absl::Status PrepareOutTensor(const at::Tensor& result, at::Tensor& out) {
  TT_RET_CHECK(out.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "the out tensor is expected to be on tpu, got " << out.device().str();

  // Validate dtype prior to mutation.
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto result_dtype,
                      ConvertTo<mlir::ElementType>(result.scalar_type()));
  TT_RET_CHECK(result.scalar_type() == out.scalar_type(),
               error::kInvalidArgument)
      << "the out tensor dtype is expected to be " << ToString(result_dtype)
      << ", got " << ToString(out_dtype);

  // Resize only if shape differs to avoid unnecessary dispatch overhead.
  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, result.sizes()));
  return absl::OkStatus();
}

absl::Status CheckMaskTensorIsBool(const at::Tensor& mask) {
  TT_RET_CHECK(IsBool(mask), error::kInvalidArgument)
      << "expected the mask to be bool, got " << ToString(mask.scalar_type());
  return absl::OkStatus();
}

absl::StatusOr<at::Tensor> AtenMaskedSelectHelper(const at::Tensor& self,
                                                  const at::Tensor& mask) {
  TT_RETURN_IF_ERROR(CheckMaskTensorIsBool(mask));

  const auto broadcasted = at::broadcast_tensors({mask, self});
  const at::Tensor& mask_broadcasted = broadcasted[0];
  const at::Tensor& self_broadcasted = broadcasted[1];

  // Get the size of the mask.
  TT_ASSIGN_OR_RETURN(at::Tensor size,
                      GetMaskedSelectOutputSize(mask_broadcasted));

  // Move `size` to the CPU. This transition is necessary to
  // dispatch the op afterwards.
  int64_t size_cpu = size.cpu().item<int64_t>();

  if (size_cpu == 0) {
    return AtenEfficientZeroTensor({0}, self_broadcasted.scalar_type(),
                                   std::nullopt, self_broadcasted.device(),
                                   std::nullopt);
  }

  TT_ASSIGN_OR_RETURN(auto param_keys, TT_MAKE_OP_PARAM_CACHE_KEYS(size_cpu));

  TT_ASSIGN_OR_RETURN(
      const auto output_dtype,
      ConvertTo<mlir::ElementType>(self_broadcasted.scalar_type()));
  Dimensions output_dims = {size_cpu};

  auto builder_fn = [size_cpu](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    auto& [self_op, mask_op] = inputs;
    return BuildMaskedSelectShlo(size_cpu, self_op, mask_op);
  };

  TT_ASSIGN_OR_RETURN(
      auto result,
      DispatchOp<2>(std::move(builder_fn), {self_broadcasted, mask_broadcasted},
                    {.out_dtype = output_dtype,
                     .out_dims = output_dims,
                     .op_param_cache_keys = std::move(param_keys)}));
  return MakeTensor(std::move(result));
}

}  // namespace

at::Tensor AtenMaskedSelect(const at::Tensor& self, const at::Tensor& mask) {
  TT_KERNEL(OpName::kMaskedSelect, _, (self, mask), {
    TT_ASSIGN_OR_THROW(at::Tensor result, AtenMaskedSelectHelper(self, mask));
    return result;
  });
}

at::Tensor& AtenMaskedSelectOut(const at::Tensor& self, const at::Tensor& mask,
                                at::Tensor& out) {
  TT_KERNEL(OpName::kMaskedSelect, _, (self, mask, out), {
    TT_ASSIGN_OR_THROW(at::Tensor result, AtenMaskedSelectHelper(self, mask));
    TT_THROW_IF_ERROR(PrepareOutTensor(result, out));

    out = AtenCopyFrom(result, out, /*non_blocking=*/true);
    return out;
  });
}

}  // namespace torch_tpu
