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

#include "csrc/ops/fake_quantize/fake_quantize_aten_kernels.h"

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty_like.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

bool IsSupportedDtype(const at::ScalarType dtype) {
  return dtype == at::kFloat || dtype == at::kDouble || dtype == at::kHalf ||
         dtype == at::kBFloat16;
}

absl::StatusOr<std::array<mlir::MlirOp, 2>>
BuildFakeQuantizePerTensorAffineCachemaskShlo(
    mlir::MlirOp self, const mlir::ElementType output_dtype, const double scale,
    const int64_t zero_point, const int64_t quant_min,
    const int64_t quant_max) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      InferComputationDtype(output_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_compute,
                      CastIfNeeded(self, computation_dtype));

  mlir::MlirOp scale_op = MakeConstantLike(self_compute, scale);
  mlir::MlirOp zp_op = MakeConstantLike(self_compute, zero_point);
  mlir::MlirOp qmin_op = MakeConstantLike(self_compute, quant_min);
  mlir::MlirOp qmax_op = MakeConstantLike(self_compute, quant_max);

  mlir::MlirOp fake_quantized_tensor =
      mlir::stablehlo::Div(self_compute, scale_op);
  fake_quantized_tensor =
      mlir::stablehlo::RoundNearestEven(fake_quantized_tensor);
  fake_quantized_tensor = mlir::stablehlo::Add(fake_quantized_tensor, zp_op);

  // Compute the mask.
  mlir::MlirOp le_mask = mlir::stablehlo::Compare(
      fake_quantized_tensor, qmax_op, mlir::stablehlo::ComparisonDirection::LE);
  mlir::MlirOp ge_mask = mlir::stablehlo::Compare(
      fake_quantized_tensor, qmin_op, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp mask = mlir::stablehlo::And(le_mask, ge_mask);

  fake_quantized_tensor =
      mlir::stablehlo::Clamp(qmin_op, fake_quantized_tensor, qmax_op);
  fake_quantized_tensor =
      mlir::stablehlo::Subtract(fake_quantized_tensor, zp_op);
  fake_quantized_tensor = mlir::stablehlo::Mul(fake_quantized_tensor, scale_op);
  TT_ASSIGN_OR_RETURN(fake_quantized_tensor,
                      CastIfNeeded(fake_quantized_tensor, output_dtype));

  return std::array<mlir::MlirOp, 2>{fake_quantized_tensor, mask};
}

struct FakeQuantizePerTensorAffineCachemaskResult {
  DeviceBufferRef output;
  DeviceBufferRef mask;
};

absl::StatusOr<FakeQuantizePerTensorAffineCachemaskResult>
FakeQuantizePerTensorAffineCachemaskHelper(const at::Tensor& self,
                                           const double scale,
                                           const int64_t zero_point,
                                           const int64_t quant_min,
                                           const int64_t quant_max,
                                           OpParamCacheKeys param_keys) {
  TT_RET_CHECK(quant_min <= quant_max, error::kInvalidArgument)
      << "expected quant_min <= quant_max, got quant_min=" << quant_min
      << " and quant_max=" << quant_max;
  TT_RET_CHECK(zero_point >= quant_min && zero_point <= quant_max,
               error::kInvalidArgument)
      << "expected zero_point to be in [" << quant_min << ", " << quant_max
      << "], got " << zero_point;
  TT_RET_CHECK(IsSupportedDtype(self.scalar_type()),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());

  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  auto op_builder = [output_dtype, scale, zero_point, quant_min,
                     quant_max](mlir::MlirOp input) {
    return BuildFakeQuantizePerTensorAffineCachemaskShlo(
        input, output_dtype, scale, zero_point, quant_min, quant_max);
  };

  TT_ASSIGN_OR_RETURN(
      (auto [output, mask]),
      (DispatchOp<1, 2>(std::move(op_builder), {self},
                        {.out_dtypes = {output_dtype, mlir::ElementType::PRED},
                         .out_dims_list = {self.sizes(), self.sizes()},
                         .op_param_cache_keys = std::move(param_keys)})));

  return FakeQuantizePerTensorAffineCachemaskResult{std::move(output),
                                                    std::move(mask)};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> FakeQuantizePerTensorAffineCachemask(
    const at::Tensor& self, const double scale, const int64_t zero_point,
    const int64_t quant_min, const int64_t quant_max) {
  TT_KERNEL(
      OpName::kFakeQuantizePerTensorAffineCachemask, param_keys,
      (self, scale, zero_point, quant_min, quant_max), {
        TT_ASSIGN_OR_THROW((auto [output_buffer, mask_buffer]),
                           FakeQuantizePerTensorAffineCachemaskHelper(
                               self, scale, zero_point, quant_min, quant_max,
                               std::move(param_keys)));
        at::Tensor out = at::empty_like(self);
        at::Tensor mask = at::empty_like(self, at::kBool);
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(output_buffer), out));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(mask_buffer), mask));
        return std::make_tuple(out, mask);
      });
}

}  // namespace torch_tpu
