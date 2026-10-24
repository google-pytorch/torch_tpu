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

#include "csrc/ops/int_mm/int_mm_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::Status ValidateIntMmInputs(const at::Tensor& self,
                                 const at::Tensor& mat2) {
  TT_RETURN_IF_ERROR(ValidateIsMatrix(self, "self"));
  TT_RETURN_IF_ERROR(ValidateIsMatrix(mat2, "mat2"));
  TT_RET_CHECK(self.size(1) == mat2.size(0), error::kInvalidArgument)
      << "self.size(1) needs to match mat2.size(0), got " << self.size(1)
      << " and " << mat2.size(0);
  TT_RET_CHECK(
      self.scalar_type() == at::kChar && mat2.scalar_type() == at::kChar,
      error::kInvalidArgument)
      << "expected int8 inputs, got " << ToString(self.scalar_type()) << " and "
      << ToString(mat2.scalar_type());
  TT_RET_CHECK(self.device().type() == GetPrivateUse1DeviceType() &&
                   self.device() == mat2.device(),
               error::kInvalidArgument)
      << "expected both inputs on the same TPU device, got " << self.device()
      << " and " << mat2.device();
  return absl::OkStatus();
}

absl::Status IntMm(const at::Tensor& self, const at::Tensor& mat2,
                   at::Tensor& out, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateIntMmInputs(self, mat2));
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, at::kInt,
                                      /*allow_out_dtype_cast=*/false));
  TT_RET_CHECK(out.device() == self.device(), error::kInvalidArgument)
      << "expected out on " << self.device() << ", got " << out.device();
  TT_RETURN_IF_ERROR(ValidateIsMatrix(out, "out"));
  // Native ATen _int_mm.out requires an existing, contiguous result of the
  // correct shape. It does not resize an incorrectly shaped out tensor.
  TT_RET_CHECK(out.size(0) == self.size(0) && out.size(1) == mat2.size(1),
               error::kInvalidArgument)
      << "expected out shape [" << self.size(0) << ", " << mat2.size(1)
      << "], got " << ToString(out.sizes());
  TT_RET_CHECK(out.is_contiguous(), error::kInvalidArgument)
      << "expected out to be contiguous";

  const auto precision = GetAndAddPrecisionTo(param_keys);
  auto builder = [precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& ctx = inputs[0].getContext();
    const int64_t lhs_contracting[1] = {1};
    const int64_t rhs_contracting[1] = {0};
    const auto dimensions = mlir::stablehlo::DotDimensionNumbersAttr::get(
        &ctx, {}, {}, lhs_contracting, rhs_contracting);
    const auto precisions =
        mlir::stablehlo::PrecisionConfigAttr::get(&ctx, {precision, precision});
    // Keep i8 operands and request i32 accumulation directly.
    return mlir::stablehlo::DotGeneral(inputs[0], inputs[1], dimensions,
                                       precisions, mlir::ElementType::I32);
  };
  const int64_t shape[2] = {self.size(0), mat2.size(1)};
  return DispatchOpOut<2>(std::move(builder), {self, mat2}, out,
                          {.out_dtype = mlir::ElementType::I32,
                           .out_dims = shape,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenIntMm(const at::Tensor& self, const at::Tensor& mat2) {
  TT_KERNEL(OpName::kIntMm, param_keys, (self, mat2), {
    TT_THROW_IF_ERROR(ValidateIntMmInputs(self, mat2));
    const int64_t shape[2] = {self.size(0), mat2.size(1)};
    TT_ASSIGN_OR_THROW(at::Tensor out,
                       MakeEmptyTensor(shape, at::kInt, self.device()));
    TT_THROW_IF_ERROR(IntMm(self, mat2, out, std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenIntMmOut(const at::Tensor& self, const at::Tensor& mat2,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kIntMmOut, param_keys, (self, mat2, out), {
    TT_THROW_IF_ERROR(IntMm(self, mat2, out, std::move(param_keys)));
    return out;
  });
}

}  // namespace torch_tpu
