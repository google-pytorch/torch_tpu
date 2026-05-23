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

#include "torch_tpu/ops/logical/logical_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/binary_aten_kernels.h"
#include "torch_tpu/ops/logical/logical.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {

namespace {

// This is needed because the output dtype of the logical ops is always PRED,
// but the output dtype of BinaryOpOut is determined by the `out` tensor.
absl::Status LogicalBinaryOutImpl(const at::Tensor& self,
                                  const at::Tensor& other, at::Tensor& out,
                                  MlirBinaryOpBuilder core_op_builder) {
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));

  // adds a type conversion to match the `out` tensor's dtype.
  auto custom_op_builder =
      [out_dtype, core_op_builder = std::move(core_op_builder)](
          mlir::MlirOp self_op,
          mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                        core_op_builder(self_op, other_op));
    if (out_dtype != mlir::ElementType::PRED) {
      result = mlir::stablehlo::ConvertElementType(result, out_dtype);
    }
    return result;
  };

  TT_RETURN_IF_ERROR(
      BinaryOpOut(self, other, out, std::move(custom_op_builder),
                  {.output_dtype_override = out_dtype,
                   .op_param_cache_keys = OpParamCacheKeys::Empty()}));
  return absl::OkStatus();
}

// This helper function for unary logical operations handles the type casting
// for in-place operations, where the output tensor's dtype might not be bool.
absl::Status LogicalUnaryOutImpl(const at::Tensor& self, at::Tensor& out,
                                 MlirUnaryOpBuilder core_op_builder) {
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));

  // Wraps the core op builder to add a type conversion to match the `out`
  // tensor's dtype.
  auto custom_op_builder =
      [out_dtype, core_op_builder = std::move(core_op_builder)](
          mlir::MlirOp self_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result, core_op_builder(self_op));
    if (out_dtype != mlir::ElementType::PRED) {
      result = mlir::stablehlo::ConvertElementType(result, out_dtype);
    }
    return result;
  };

  TT_RETURN_IF_ERROR(
      ::torch_tpu::UnaryOpOut(self, out, std::move(custom_op_builder),
                              {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                               .out_dtype = out_dtype}));
  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenLogicalAndOut(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kLogicalAndOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(::torch_tpu::LogicalBinaryOutImpl(self, other, out,
                                                        BuildLogicalAndShlo));
    return out;
  });
}

at::Tensor& AtenLogicalOrOut(const at::Tensor& self, const at::Tensor& other,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kLogicalOrOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(::torch_tpu::LogicalBinaryOutImpl(self, other, out,
                                                        BuildLogicalOrShlo));
    return out;
  });
}

at::Tensor& AtenLogicalXorOut(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kLogicalXorOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(::torch_tpu::LogicalBinaryOutImpl(self, other, out,
                                                        BuildLogicalXorShlo));
    return out;
  });
}

at::Tensor& AtenLogicalNotOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kLogicalNotOut, _, (self, out), {
    TT_THROW_IF_ERROR(
        ::torch_tpu::LogicalUnaryOutImpl(self, out, BuildLogicalNotShlo));
    return out;
  });
}

}  // namespace torch_tpu
