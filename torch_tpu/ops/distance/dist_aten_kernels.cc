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

#include "torch_tpu/ops/distance/dist_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
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
#include "torch_tpu/ops/index_select/index_select.h"
#include "torch_tpu/ops/linalg/vector_norm/pnorm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

namespace {

// When p=0, compute the 0-norm (Hamming distance equivalent)
// which is defined as the count of non-zero elements in the vector:
//
// 1. Compare input elements with 0.0 to get a boolean tensor.
// 2. Cast the boolean tensor to the input floating-point type (0.0 or 1.0).
// 3. Sum the values along the reduction dimension.
absl::StatusOr<mlir::MlirOp> BuildZeroNorm(mlir::MlirOp input,
                                           int64_t reduce_dim,
                                           mlir::Type element_type) {
  // Create a scalar zero and broadcast it for comparison
  mlir::MlirBuilder& builder = input.getBuilder();

  auto zero = MakeScalarConstant(builder, 0.0, element_type);
  auto input_type = GetTensorTypeOrDie(input);
  auto zero_bcast = mlir::stablehlo::BroadcastInDim(input_type, zero, {});

  auto is_non_zero = mlir::stablehlo::Compare(
      input, zero_bcast, mlir::stablehlo::ComparisonDirection::NE);

  // Convert bool to float (0.0 or 1.0)
  auto is_non_zero_casted = mlir::stablehlo::Convert(input_type, is_non_zero);

  auto reduce_op = mlir::stablehlo::Reduce(
      builder,
      /*inputs=*/{is_non_zero_casted},
      /*init_values=*/{zero},
      /*body_builder=*/
      [&](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            element_type, rb.getRegion(), rb.getOpBuilder());
      },
      /*dimensions=*/{reduce_dim});

  return reduce_op[0];
}

// Computes the p-norm distance of the difference between each pair of row
// vectors in x1 and x2.
// x1 shape: (B..., R1, C)
// x2 shape: (B..., R2, C)
// Output shape: (B..., R1, R2)
absl::StatusOr<mlir::MlirOp> BuildCdistForwardHlo(
    mlir::MlirOp x1_op, mlir::MlirOp x2_op, double p, int64_t compute_mode,
    const Dimensions common_batch_shape, int64_t r1, int64_t r2, int64_t c) {
  const mlir::RankedTensorType x1_type = GetTensorTypeOrDie(x1_op);
  const mlir::RankedTensorType x2_type = GetTensorTypeOrDie(x2_op);
  const mlir::Type element_type = x1_type.getElementType();
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_type, GetElementType(x1_op));

  // Define the target shape for the difference tensor: (B_common..., R1, R2, C)
  Dimensions target_diff_shape(common_batch_shape.begin(),
                               common_batch_shape.end());
  target_diff_shape.insert(target_diff_shape.end(), {r1, r2, c});
  const int64_t common_batch_rank = common_batch_shape.size();
  mlir::RankedTensorType bcast_type =
      mlir::RankedTensorType::get(target_diff_shape, element_type);

  // Broadcast X1 to (B_common..., R1, 1, C)
  int64_t x1_rank = x1_type.getRank();
  int64_t x1_batch_dims_count = x1_rank - 2;  // last 2 dims are R1 and C
  int64_t x1_offset = common_batch_rank - x1_batch_dims_count;

  Dimensions x1_bcast_dims;
  for (int i = 0; i < x1_batch_dims_count; ++i) {
    x1_bcast_dims.push_back(i + x1_offset);
  }
  x1_bcast_dims.push_back(common_batch_rank);      // mapped to R1 index
  x1_bcast_dims.push_back(common_batch_rank + 2);  // mapped to C index

  mlir::MlirOp x1_bcast =
      mlir::stablehlo::BroadcastInDim(bcast_type, x1_op, x1_bcast_dims);

  // Broadcast X2 to (B_common..., 1, R2, C)
  int64_t x2_rank = x2_type.getRank();
  int64_t x2_batch_dims_count = x2_rank - 2;  // last 2 dims are R2 and C
  int64_t x2_offset = common_batch_rank - x2_batch_dims_count;

  Dimensions x2_bcast_dims;
  for (int i = 0; i < x2_batch_dims_count; ++i) {
    x2_bcast_dims.push_back(i + x2_offset);
  }
  x2_bcast_dims.push_back(common_batch_rank + 1);  // mapped to R2 index
  x2_bcast_dims.push_back(common_batch_rank + 2);  // mapped to C index

  mlir::MlirOp x2_bcast =
      mlir::stablehlo::BroadcastInDim(bcast_type, x2_op, x2_bcast_dims);

  // Compute diff = X1 - X2
  mlir::MlirOp diff_op = mlir::stablehlo::Subtract(x1_bcast, x2_bcast);

  // Reduce over the C dimension.
  int64_t reduce_dim_idx = common_batch_rank + 2;

  if (p == 0.0) {
    // Special handling for 0-norm (hamming distance equivalent)
    return BuildZeroNorm(diff_op, reduce_dim_idx, element_type);
  } else {
    // General case for p-norm
    TT_ASSIGN_OR_RETURN(auto result,
                        BuildPNormShlo(diff_op, p, {reduce_dim_idx},
                                       ReductionMode::kDropDims, out_type));
    return result;
  };
}

// Computes the p-norm distance of the difference between each pair of elements
// in input.
// Input shape: (B..., R, C)
// Output shape: (B..., R(R-1)/2)
absl::StatusOr<mlir::MlirOp> BuildPdistForwardHlo(mlir::MlirOp input_op,
                                                  double p) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const mlir::Type element_type = input_type.getElementType();
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_type, GetElementType(input_op));

  const int64_t n = input_type.getDimSize(0);
  const int64_t num_pairs = n * (n - 1) / 2;

  // Create tensors of indices for pairs (i, j) such that i < j.
  Dimensions idx_left;
  Dimensions idx_right;
  idx_left.reserve(num_pairs);
  idx_right.reserve(num_pairs);

  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = i + 1; j < n; ++j) {
      idx_left.push_back(i);
      idx_right.push_back(j);
    }
  }

  auto indices_type = mlir::RankedTensorType::get(
      {num_pairs}, builder.getOpBuilder().getI64Type());
  mlir::MlirOp idx_left_op = mlir::stablehlo::Constant(
      builder, mlir::cast<mlir::DenseElementsAttr>(mlir::makeConstant(
                   llvm::ArrayRef<int64_t>(idx_left), indices_type)));
  mlir::MlirOp idx_right_op = mlir::stablehlo::Constant(
      builder, mlir::cast<mlir::DenseElementsAttr>(mlir::makeConstant(
                   llvm::ArrayRef<int64_t>(idx_right), indices_type)));

  mlir::MlirOp lhs = BuildIndexSelectShlo(input_op, /*dim=*/0, idx_left_op);
  mlir::MlirOp rhs = BuildIndexSelectShlo(input_op, /*dim=*/0, idx_right_op);

  mlir::MlirOp diff_op = mlir::stablehlo::Subtract(lhs, rhs);

  if (p == 0.0) {
    // Special handling for 0-norm (hamming distance equivalent)
    return BuildZeroNorm(diff_op, /*reduce_dim=*/1, element_type);
  } else {
    // General case for p-norm
    return BuildPNormShlo(diff_op, p, {1}, ReductionMode::kDropDims, out_type);
  }
}

bool IsBFloatOrHalf(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::BFloat16 ||
         tensor.scalar_type() == at::ScalarType::Half;
}

absl::Status CheckIsFloatingPoint(const at::Tensor& tensor,
                                  const std::string_view name) {
  TT_RET_CHECK(IsFloatingPoint(tensor), error::kInvalidArgument)
      << "expected the " << name << " dtype to be floating point, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

absl::Status CheckNotBFloatOrHalf(const at::Tensor& tensor,
                                  const std::string_view name) {
  TT_RET_CHECK(!IsBFloatOrHalf(tensor), error::kInvalidArgument)
      << "expected the " << name << " dtype not to be bfloat16 or float16, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenCdistForward(const at::Tensor& x1, const at::Tensor& x2,
                            double p, std::optional<int64_t> compute_mode) {
  TT_KERNEL(OpName::kCdistForward, param_keys, (x1, x2, p, compute_mode), {
    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(x1.scalar_type()));

    TT_THROW_IF_ERROR(CheckIsFloatingPoint(x1, "first argument's"));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(x2, "second argument's"));

    TT_CHECK_THROW(p >= 0, error::kInvalidArgument)
        << "expected the p value to be >= 0, got " << p;

    const int64_t mode = compute_mode.value_or(0);
    const int64_t c = x1.size(-1);
    const int64_t r1 = x1.size(-2);
    const int64_t r2 = x2.size(-2);

    Dimensions batch_tensor1(x1.sizes().begin(), x1.sizes().end() - 2);
    Dimensions batch_tensor2(x2.sizes().begin(), x2.sizes().end() - 2);
    TT_ASSIGN_OR_THROW(Dimensions common_batch_shape,
                       InferSize(batch_tensor1, batch_tensor2));

    // Output shape: (B_common..., R1, R2)
    Dimensions output_shape(common_batch_shape);
    output_shape.insert(output_shape.end(), {r1, r2});

    // If the output shape contains a 0 dimension, return an empty tensor
    for (int64_t dim : output_shape) {
      if (dim == 0) {
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(output_shape, x1.scalar_type(), x1.device()));
        return out;
      }
    }

    // Add check for bf16 and float16 after the empty dimension check,
    // because they are supported for empty tensors but not for non-empty ones.
    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(x1, "first argument's"));
    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(x2, "second argument's"));

    auto op_builder = [p, mode, common_batch_shape, r1, r2,
                       c](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [x1_op, x2_op] = inputs;
      return BuildCdistForwardHlo(x1_op, x2_op, p, mode, common_batch_shape, r1,
                                  r2, c);
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<2>(std::move(op_builder), {x1, x2},
                      {.out_dtype = out_dtype,
                       .out_dims = output_shape,
                       .op_param_cache_keys = std::move(param_keys)}));

    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor AtenPdistForward(const at::Tensor& self, double p) {
  TT_KERNEL(OpName::kPdistForward, param_keys, (self, p), {
    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    // If input has shape (n, m), output will have shape n(n-1)/2.
    const int64_t n = self.size(0);
    const int64_t num_pairs = n * (n - 1) / 2;
    Dimensions output_shape = {num_pairs};

    // If there are no pairs to compute the distance for, return an empty tensor
    if (num_pairs == 0) {
      TT_ASSIGN_OR_THROW(
          at::Tensor out,
          MakeEmptyTensor(output_shape, self.scalar_type(), self.device()));
      return out;
    }

    // Add check for bfloat16 and float16 after the empty dimension check,
    // because empty tensors are supported for these dtypes
    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(self, /* name= */ "input"));

    auto op_builder = [p](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildPdistForwardHlo(input, p);
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(std::move(op_builder), self,
                      {.out_dtype = out_dtype,
                       .out_dims = output_shape,
                       .op_param_cache_keys = std::move(param_keys)}));

    return MakeTensor(std::move(result_buf));
  });
}

}  // namespace torch_tpu
