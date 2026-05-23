/*
 * Copyright 2026 Google LLC
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

#include "torch_tpu/ops/fake_quantize/fake_quantize_aten_kernels.h"

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty_like.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::StatusOr<std::array<mlir::MlirOp, 2>>
BuildFakeQuantizePerTensorAffineCachemaskShlo(mlir::MlirOp self, double scale,
                                              int64_t zero_point,
                                              int64_t quant_min,
                                              int64_t quant_max) {
  mlir::MlirOp scale_op = MakeConstantLike(self, scale);
  mlir::MlirOp zp_op = MakeConstantLike(self, zero_point);
  mlir::MlirOp qmin_op = MakeConstantLike(self, quant_min);
  mlir::MlirOp qmax_op = MakeConstantLike(self, quant_max);

  mlir::MlirOp fake_quantized_tensor;
  fake_quantized_tensor = mlir::stablehlo::Div(self, scale_op);
  fake_quantized_tensor = mlir::stablehlo::Round(fake_quantized_tensor);
  fake_quantized_tensor = mlir::stablehlo::Add(fake_quantized_tensor, zp_op);

  // Compute the mask.
  mlir::MlirOp mask, ge_mask, le_mask;
  le_mask = mlir::stablehlo::Compare(fake_quantized_tensor, qmax_op,
                                     mlir::stablehlo::ComparisonDirection::LE);

  ge_mask = mlir::stablehlo::Compare(fake_quantized_tensor, qmin_op,
                                     mlir::stablehlo::ComparisonDirection::GE);
  mask = mlir::stablehlo::And(le_mask, ge_mask);

  fake_quantized_tensor =
      mlir::stablehlo::Clamp(qmin_op, fake_quantized_tensor, qmax_op);
  fake_quantized_tensor =
      mlir::stablehlo::Subtract(fake_quantized_tensor, zp_op);
  fake_quantized_tensor = mlir::stablehlo::Mul(fake_quantized_tensor, scale_op);

  return std::array<mlir::MlirOp, 2>{fake_quantized_tensor, mask};
}

struct FakeQuantizePerTensorAffineCachemaskResult {
  DeviceBufferRef output;
  DeviceBufferRef mask;
};

absl::StatusOr<FakeQuantizePerTensorAffineCachemaskResult>
FakeQuantizePerTensorAffineCachemaskHelper(const at::Tensor& self, double scale,
                                           int64_t zero_point,
                                           int64_t quant_min,
                                           int64_t quant_max) {
  auto op_builder = [scale, zero_point, quant_min,
                     quant_max](mlir::MlirOp input) {
    return BuildFakeQuantizePerTensorAffineCachemaskShlo(
        input, scale, zero_point, quant_min, quant_max);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  TT_ASSIGN_OR_RETURN(
      (auto [output, mask]),
      (DispatchOp<1, 2>(std::move(op_builder), {self},
                        {.out_dtypes = {output_dtype, mlir::ElementType::PRED},
                         .out_dims_list = {self.sizes(), self.sizes()},
                         .op_param_cache_keys = OpParamCacheKeys::Empty()})));

  return FakeQuantizePerTensorAffineCachemaskResult{std::move(output),
                                                    std::move(mask)};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> FakeQuantizePerTensorAffineCachemask(
    const at::Tensor& self, double scale, int64_t zero_point, int64_t quant_min,
    int64_t quant_max) {
  TT_KERNEL(
      OpName::kFakeQuantizePerTensorAffineCachemask, _,
      (self, IgnoreInCacheKey(scale, "Legacy usage"),
       IgnoreInCacheKey(zero_point, "Legacy usage"),
       IgnoreInCacheKey(quant_min, "Legacy usage"),
       IgnoreInCacheKey(quant_max, "Legacy usage")),
      {
        at::Tensor out = at::empty_like(self);
        at::Tensor mask = at::empty_like(self, at::kBool);
        TT_ASSIGN_OR_THROW((auto [output_buffer, mask_buffer]),
                           FakeQuantizePerTensorAffineCachemaskHelper(
                               self, scale, zero_point, quant_min, quant_max));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(output_buffer), out));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(mask_buffer), mask));
        return std::make_tuple(out, mask);
      });
}

}  // namespace torch_tpu
