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

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/hardtanh/hardtanh.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

// Helper for checking hardtanh op inputs.
//
// Enforces precondition type checks to remain consistent with the CPU
// implementation.
absl::Status CheckHardtanhInputs(const at::Tensor& self,
                                 const at::Scalar& min_val,
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
  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenHardtanh(const at::Tensor& self, const at::Scalar& min_val,
                        const at::Scalar& max_val) {
  TT_KERNEL(OpName::kHardtanh, param_keys, (self, min_val, max_val), {
    TT_THROW_IF_ERROR(CheckHardtanhInputs(self, min_val, max_val));

    TT_ASSIGN_OR_THROW(at::Tensor min_tensor,
                       MakeTensor(min_val, self.scalar_type()));
    TT_ASSIGN_OR_THROW(at::Tensor max_tensor,
                       MakeTensor(max_val, self.scalar_type()));

    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [input_op, min_val_op, max_val_op] = inputs;
      return BuildHardtanhShlo(input_op, min_val_op, max_val_op);
    };

    TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(
        auto result_buffer,
        DispatchOp<3>(OpName::kHardtanh, std::move(op_builder),
                      {self, min_tensor, max_tensor},
                      {.out_dtype = out_dtype,
                       .out_dims = CopyIntVector(self.sizes()),
                       .op_param_cache_keys = std::move(param_keys)}));
    return MakeTensor(std::move(result_buffer));
  });
}

at::Tensor& AtenHardtanh_(at::Tensor& self, const at::Scalar& min_val,
                          const at::Scalar& max_val) {
  TT_KERNEL(OpName::kHardtanh_, param_keys, (self, min_val, max_val), {
    TT_THROW_IF_ERROR(CheckHardtanhInputs(self, min_val, max_val));

    TT_ASSIGN_OR_THROW(at::Tensor min_tensor,
                       MakeTensor(min_val, self.scalar_type()));
    TT_ASSIGN_OR_THROW(at::Tensor max_tensor,
                       MakeTensor(max_val, self.scalar_type()));

    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [input_op, min_val_op, max_val_op] = inputs;
      return BuildHardtanhShlo(input_op, min_val_op, max_val_op);
    };

    TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(
        auto result_buffer,
        DispatchOp<3>(OpName::kHardtanh_, std::move(op_builder),
                      {self, min_tensor, max_tensor},
                      {.out_dtype = out_dtype,
                       .out_dims = CopyIntVector(self.sizes()),
                       .op_param_cache_keys = std::move(param_keys)}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buffer), self));
    return self;
  });
}

at::Tensor& AtenHardtanhOut(const at::Tensor& self, const at::Scalar& min_val,
                            const at::Scalar& max_val, at::Tensor& out) {
  TT_KERNEL(OpName::kHardtanhOut, param_keys, (self, min_val, max_val, out), {
    TT_THROW_IF_ERROR(CheckHardtanhInputs(self, min_val, max_val));

    TT_ASSIGN_OR_THROW(at::Tensor min_tensor,
                       MakeTensor(min_val, self.scalar_type()));
    TT_ASSIGN_OR_THROW(at::Tensor max_tensor,
                       MakeTensor(max_val, self.scalar_type()));

    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [input_op, min_val_op, max_val_op] = inputs;
      return BuildHardtanhShlo(input_op, min_val_op, max_val_op);
    };

    TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    TT_ASSIGN_OR_THROW(
        auto result_buffer,
        DispatchOp<3>(OpName::kHardtanhOut, std::move(op_builder),
                      {self, min_tensor, max_tensor},
                      {.out_dtype = out_dtype,
                       .out_dims = CopyIntVector(out.sizes()),
                       .op_param_cache_keys = std::move(param_keys)}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buffer), out));
    return out;
  });
}

}  // namespace torch_tpu
