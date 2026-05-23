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

#include "torch_tpu/ops/hardtanh/hardtanh_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/hardtanh/hardtanh.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

// Helper for building the hardtanh op.
//
// Enforces precondition type checks to remain consistent with the CPU
// implementation, converts the min and max values to MLIR constants and creates
// the MLIR builder function for the hardtanh operation along with the necessary
// cache keys based on the operation's parameters.
//
// Returns an MlirUnaryOpBuilder that builds the hardtanh op.
absl::StatusOr<MlirUnaryOpBuilder> AtenHardtanhHelper(
    const at::Tensor& self, const at::Scalar& min_val,
    const at::Scalar& max_val) {
  // Our StableHLO implementation supports these types but PT CPU kernels do
  // not, and we want to have a consistent behavior across the backends.
  TT_RET_CHECK(!c10::isComplexType(self.scalar_type()), error::kInvalidArgument)
      << "hardtanh: complex types are not supported.";
  TT_RET_CHECK(self.scalar_type() != c10::ScalarType::Bool,
               error::kInvalidArgument)
      << "hardtanh: bool type is not supported.";
  TT_RET_CHECK(c10::isSignedType(self.scalar_type()) ||
                   (min_val.toInt() >= 0 && max_val.toInt() >= 0),
               error::kInvalidArgument)
      << "hardtanh: cannot do hardtanh on an unsigned type with negative "
         "limits.";

  return [min_val,
          max_val](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirBuilder& builder = input_op.getBuilder();
    const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

    // Create constants for min_val and max_val from at::Scalar
    mlir::MlirOp min_val_op = MakeScalarConstant(builder, min_val.toDouble(),
                                                 input_type.getElementType());
    mlir::MlirOp max_val_op = MakeScalarConstant(builder, max_val.toDouble(),
                                                 input_type.getElementType());

    return BuildHardtanhShlo(input_op, min_val_op, max_val_op);
  };
}

}  // namespace

at::Tensor AtenHardtanh(const at::Tensor& self, const at::Scalar& min_val,
                        const at::Scalar& max_val) {
  TT_KERNEL(OpName::kHardtanh, param_keys, (self, min_val, max_val), {
    TT_ASSIGN_OR_THROW(auto op_builder,
                       AtenHardtanhHelper(self, min_val, max_val));
    TT_ASSIGN_OR_THROW(auto result,
                       UnaryOp(self, std::move(op_builder),
                               {.op_param_cache_keys = std::move(param_keys)}));
    return result;
  });
}

at::Tensor& AtenHardtanh_(at::Tensor& self, const at::Scalar& min_val,
                          const at::Scalar& max_val) {
  TT_KERNEL(OpName::kHardtanh_, param_keys, (self, min_val, max_val), {
    TT_ASSIGN_OR_THROW(auto op_builder,
                       AtenHardtanhHelper(self, min_val, max_val));
    TT_THROW_IF_ERROR(
        UnaryOpInPlace(self, std::move(op_builder),
                       {.op_param_cache_keys = std::move(param_keys)}));
    return self;
  });
}

at::Tensor& AtenHardtanhOut(const at::Tensor& self, const at::Scalar& min_val,
                            const at::Scalar& max_val, at::Tensor& out) {
  TT_KERNEL(OpName::kHardtanhOut, param_keys, (self, min_val, max_val, out), {
    TT_ASSIGN_OR_THROW(auto op_builder,
                       AtenHardtanhHelper(self, min_val, max_val));
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, std::move(op_builder),
                   {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

}  // namespace torch_tpu
