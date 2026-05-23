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

#include "torch_tpu/ops/linalg/solve_triangular/linalg_solve_triangular_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::Status CheckDimensions(const at::Tensor& a, const at::Tensor& b,
                             bool left) {
  // Solving A * X = B if left == true, X * A = B if left == false.
  TT_RET_CHECK(a.dim() >= 2, error::kInvalidArgument)
      << "expected the first argument to have at least 2 dimensions, got "
      << a.dim();
  TT_RET_CHECK(b.dim() == a.dim(), error::kInvalidArgument)
      << "expected the two inputs to have the same number of dimensions, got "
      << a.dim() << " and " << b.dim();
  TT_RET_CHECK(left || a.size(-1) == b.size(-1), error::kInvalidArgument)
      << "left == False means we are solving X * A = B; expected the two "
         "inputs to have matching last dimension, got "
      << a.size(-1) << " and " << b.size(-1);
  TT_RET_CHECK(!left || a.size(-2) == b.size(-2), error::kInvalidArgument)
      << "left == True means we are solving A * X = B; expected the two "
         "inputs to have matching second to last dimension, got "
      << a.size(-2) << " and " << b.size(-2);
  return absl::OkStatus();
}

}  // namespace

NAryMlirOpBuilder<2, 1> LinalgSolveTriangularBuilder(bool upper, bool left,
                                                     bool unitriangular,
                                                     bool adjoint = false) {
  return [upper, left, unitriangular,
          adjoint](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    auto& [a_op, b_op] = inputs;
    mlir::MlirBuilder& builder = a_op.getBuilder();
    ABSL_VLOG(1) << "triangular solve: " << "upper:" << upper
                 << " left:" << left << " unitriangular:" << unitriangular;
    auto a_type = GetTensorTypeOrDie(a_op);
    auto b_type = GetTensorTypeOrDie(b_op);
    auto a_batch_shape = a_type.getShape().slice(0, a_type.getRank() - 2);
    auto b_batch_shape = b_type.getShape().slice(0, b_type.getRank() - 2);
    TT_ASSIGN_OR_RETURN(auto broadcasted_batch_shape,
                        InferSize(a_batch_shape, b_batch_shape));
    Dimensions a_broadcasted_shape = broadcasted_batch_shape;
    a_broadcasted_shape.push_back(a_type.getShape()[a_type.getRank() - 2]);
    a_broadcasted_shape.push_back(a_type.getShape()[a_type.getRank() - 1]);

    Dimensions b_broadcasted_shape = broadcasted_batch_shape;
    b_broadcasted_shape.push_back(b_type.getShape()[b_type.getRank() - 2]);
    b_broadcasted_shape.push_back(b_type.getShape()[b_type.getRank() - 1]);
    TT_ASSIGN_OR_RETURN(a_op, BroadcastIfNeeded(a_op, a_broadcasted_shape));
    TT_ASSIGN_OR_RETURN(b_op, BroadcastIfNeeded(b_op, b_broadcasted_shape));
    auto triangular_solve_op =
        builder.create<mlir::stablehlo::TriangularSolveOp>(
            a_op.getValue(), b_op.getValue(), left, !upper, unitriangular,
            adjoint ? mlir::stablehlo::Transpose::ADJOINT
                    : mlir::stablehlo::Transpose::NO_TRANSPOSE);
    return triangular_solve_op;
  };
}

at::Tensor& AtenLinalgSolveTriangularOut(const at::Tensor& a,
                                         const at::Tensor& b, bool upper,
                                         bool left, bool unitriangular,
                                         at::Tensor& out) {
  TT_KERNEL(OpName::kLinalgSolveTriangularOut, param_keys,
            (a, b, upper, left, unitriangular, out), {
              TT_THROW_IF_ERROR(CheckDimensions(a, b, left));
              TT_CHECK_THROW(a.scalar_type() != c10::ScalarType::BFloat16 &&
                                 a.scalar_type() != c10::ScalarType::Half &&
                                 !at::isIntegralType(a.scalar_type(),
                                                     /*includeBool=*/true),
                             error::kInvalidArgument)
                  << "triangular solve not supported for dtype "
                  << ToString(a.scalar_type());

              at::ScalarType out_scalar_type =
                  c10::promoteTypes(a.scalar_type(), b.scalar_type());

              TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                                 ConvertTo<mlir::ElementType>(out_scalar_type));

              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  DispatchOp<2>(
                      LinalgSolveTriangularBuilder(upper, left, unitriangular),
                      {a, b},
                      {.out_dtype = out_dtype,
                       .out_dims = CopyIntVector(b.sizes()),
                       .op_param_cache_keys = std::move(param_keys)}));

              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buffer), out));
              return out;
            });
}

at::Tensor AtenLinalgSolveTriangular(const at::Tensor& a, const at::Tensor& b,
                                     bool upper, bool left,
                                     bool unitriangular) {
  TT_KERNEL(OpName::kLinalgSolveTriangular, param_keys,
            (a, b, upper, left, unitriangular), {
              TT_THROW_IF_ERROR(CheckDimensions(a, b, left));
              TT_CHECK_THROW(a.scalar_type() != c10::ScalarType::BFloat16 &&
                                 a.scalar_type() != c10::ScalarType::Half &&
                                 !at::isIntegralType(a.scalar_type(),
                                                     /*includeBool=*/true),
                             error::kInvalidArgument)
                  << "triangular solve not supported for dtype "
                  << ToString(a.scalar_type());

              at::ScalarType out_scalar_type =
                  c10::promoteTypes(a.scalar_type(), b.scalar_type());

              TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                                 ConvertTo<mlir::ElementType>(out_scalar_type));

              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  DispatchOp<2>(
                      LinalgSolveTriangularBuilder(upper, left, unitriangular),
                      {a, b},
                      {.out_dtype = out_dtype,
                       .out_dims = CopyIntVector(b.sizes()),
                       .op_param_cache_keys = std::move(param_keys)}));
              return MakeTensor(std::move(result_buffer));
            });
}

}  // namespace torch_tpu
