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

#include "torch_tpu/ops/replication_pad/replication_pad_aten_kernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/reductions/sum.h"

namespace torch_tpu {

namespace {
absl::StatusOr<mlir::MlirOp> BuildReplicationPadShlo(mlir::MlirOp input,
                                                     Dimensions padding,
                                                     Dimensions output_shape,
                                                     int num_pad_dimensions) {
  auto broadcast_dims = Dimensions(output_shape.size());
  absl::c_iota(broadcast_dims, 0);

  auto pad_fn = [&broadcast_dims](
                    mlir::MlirOp op, int64_t dimension, int64_t left_pad,
                    int64_t right_pad) -> absl::StatusOr<mlir::MlirOp> {
    auto input_tensor_type = GetTensorTypeOrDie(op);
    Dimensions input_shape(input_tensor_type.getShape().begin(),
                           input_tensor_type.getShape().end());
    // Left and right dimensions for slicing (a vector of 0s and of max vals)
    // and a vector of stride 1.
    auto left_dim = Dimensions(input_shape.size(), 0);
    auto right_dim = input_shape;
    auto strides = Dimensions(input_shape.size(), 1);
    std::vector<mlir::MlirOp> ops;

    if (left_pad > 0) {
      // Slice the active dimension to get a 1 element strip on the left
      // then broadcast it to the padding width
      left_dim[dimension] = 0;
      right_dim[dimension] = 1;
      auto left_slice_op =
          mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
      right_dim[dimension] = left_pad;
      auto broadcast_type = mlir::RankedTensorType::get(
          right_dim, input_tensor_type.getElementType());
      ops.push_back(mlir::stablehlo::BroadcastInDim(
          broadcast_type, left_slice_op, broadcast_dims));
    }

    // Simply append or, if either pad value is negative, truncate the input
    // tensor.
    TT_ASSIGN_OR_RETURN(auto core_op,
                        BuildMaybeSlice(op, dimension, left_pad, right_pad));
    ops.push_back(core_op);

    if (right_pad > 0) {
      // Slice the active dimension to get a 1 element strip on the right
      // then broadcast it to the padding width
      left_dim[dimension] = input_shape[dimension] - 1;
      right_dim[dimension] = input_shape[dimension];
      auto right_slice_op =
          mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
      right_dim[dimension] = right_pad;
      auto broadcast_type = mlir::RankedTensorType::get(
          right_dim, input_tensor_type.getElementType());
      ops.push_back(mlir::stablehlo::BroadcastInDim(
          broadcast_type, right_slice_op, broadcast_dims));
    }
    // Concatenate the two padding strips with the original tensor between them.
    return mlir::stablehlo::Concatenate(
        op.getBuilder(), mlir::ArrayRef<mlir::MlirOp>(ops), dimension);
  };

  // Iterate over number of dimensions we need to pad
  mlir::MlirOp result = input;
  for (int i = 0; i < num_pad_dimensions; ++i) {
    TT_ASSIGN_OR_RETURN(result, pad_fn(result, output_shape.size() - i - 1,
                                       padding[i * 2], padding[i * 2 + 1]));
  }
  return result;
}

// Generate code to handle accumulation of the padding slice, or propagation of
// zeros if the padding is negative. Precondition: padding_size is non-zero.
absl::StatusOr<mlir::MlirOp> BuildReplicationPadBackwardSidePaddingShlo(
    mlir::MlirOp op, mlir::RankedTensorType input_tensor_type,
    Dimensions input_shape, int64_t padding_size, int64_t dimension,
    int64_t left_bound, int64_t right_bound) {
  // Left and right dimensions for slicing (a vector of 0s and of max vals)
  // and a vector of stride 1.
  auto left_dim = Dimensions(input_shape.size(), 0);
  auto right_dim = input_shape;
  auto strides = Dimensions(input_shape.size(), 1);
  if (padding_size > 0) {
    // Slice the active dimension to get the padding on the left, plus the 1
    // element strip that was scaled out to the padding width. Sum-Reduce the
    // slice to the 1 element strip.
    left_dim[dimension] = left_bound;
    right_dim[dimension] = right_bound;
    auto slice_op = mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
    return BuildSumShlo(slice_op, {dimension}, ReductionMode::kKeepDims);

  } else if (padding_size < 0) {
    // A negative pad value means we truncated the input.
    // So we have no gradients for those values. We construct a 0 tensor of
    // the appropriate size to concatenate.
    auto zero_dim = input_shape;
    zero_dim[dimension] = -padding_size;
    auto zero_type = mlir::RankedTensorType::get(
        zero_dim, input_tensor_type.getElementType());

    auto zero_scalar_op = mlir::stablehlo::ConvertElementType(
        mlir::stablehlo::Constant(op.getBuilder(), 0),
        input_tensor_type.getElementType());
    auto zero_tensor_op =
        mlir::stablehlo::BroadcastInDim(zero_type, zero_scalar_op, {});
    return zero_tensor_op;
  }
  return mlir::MlirOp();
}

absl::StatusOr<mlir::MlirOp> BuildReplicationPadBackwardShlo(
    mlir::MlirOp input, Dimensions padding, Dimensions output_shape,
    int num_pad_dimensions) {
  // Process padding for a single dimension.
  auto pad_fn = [](mlir::MlirOp op, int64_t dimension, int64_t left_pad,
                   int64_t right_pad) -> absl::StatusOr<mlir::MlirOp> {
    mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(op);
    Dimensions input_shape(input_tensor_type.getShape().begin(),
                           input_tensor_type.getShape().end());

    std::vector<mlir::MlirOp> ops;

    // If we had left padding (positive or negative) we need to slice and
    // accumulate or generate 0 gradients.
    if (left_pad != 0) {
      TT_ASSIGN_OR_RETURN(
          auto left_pad_op,
          BuildReplicationPadBackwardSidePaddingShlo(
              op, input_tensor_type, input_shape, left_pad, dimension,
              /*left_bound=*/0, /*right_bound=*/left_pad + 1));
      ops.push_back(left_pad_op);
    }

    // Left and right dimensions for slicing (a vector of 0s and of max vals)
    // and a vector of stride 1.
    auto left_dim = Dimensions(input_shape.size(), 0);
    auto right_dim = input_shape;
    auto strides = Dimensions(input_shape.size(), 1);
    // Slice out the middle ignoring the two reduce slices.
    // This section passes gradients through directly.
    // If padding was 0 or negative we capture the middle all the way to its
    // edge as there is no reduction to do.
    left_dim[dimension] = (left_pad > 0) ? left_pad + 1 : 0;
    right_dim[dimension] = (right_pad > 0)
                               ? (input_shape[dimension] - right_pad - 1)
                               : input_shape[dimension];
    auto middle_slice_op =
        mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
    ops.push_back(middle_slice_op);

    if (right_pad != 0) {
      TT_ASSIGN_OR_RETURN(
          auto right_pad_op,
          BuildReplicationPadBackwardSidePaddingShlo(
              op, input_tensor_type, input_shape, right_pad, dimension,
              /*left_bound=*/input_shape[dimension] - right_pad - 1,
              /*right_bound=*/input_shape[dimension]));
      ops.push_back(right_pad_op);
    }

    // Concatenate the two reduced slices with the untouched middle slice.
    return mlir::stablehlo::Concatenate(
        op.getBuilder(), mlir::ArrayRef<mlir::MlirOp>(ops), dimension);
  };

  // Iterate over number of dimensions we need to pad.
  mlir::MlirOp result = input;
  for (int i = 0; i < num_pad_dimensions; ++i) {
    TT_ASSIGN_OR_RETURN(result, pad_fn(result, output_shape.size() - i - 1,
                                       padding[i * 2], padding[i * 2 + 1]));
  }
  return result;
}

absl::Status ReplicationPadHelper(
    absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(
        mlir::MlirOp input, Dimensions padding, Dimensions output_shape,
        int num_pad_dimensions) const>
        shlo_builder_function,
    OpParamCacheKeys param_keys, const at::Tensor& self,
    at::IntArrayRef padding, at::Tensor& out, int num_pad_dimensions) {
  TT_RET_CHECK(self.scalar_type() != at::ScalarType::Bool,
               error::kInvalidArgument)
      << "not implemented for 'Bool'";

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Current usages are guaranteed to be
                 // within range.
      padding.size() == num_pad_dimensions * 2, error::kInvalidArgument)
      << "expected padding to have " << (num_pad_dimensions * 2) << " elements"
      << ", got " << padding.size() << " elements";

  Dimensions padding_vec(padding.begin(), padding.end());
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  auto op_builder = [padding_vec, out_shape = CopyIntVector(out.sizes()),
                     num_pad_dimensions,
                     shlo_builder_function = std::move(shlo_builder_function)](
                        mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(auto output,
                        shlo_builder_function(input, padding_vec, out_shape,
                                              num_pad_dimensions));
    return output;
  };

  TT_ASSIGN_OR_RETURN(
      auto out_buf,
      (DispatchOp<1>(std::move(op_builder), self,
                     {.out_dtype = element_type,
                      .out_dims = CopyIntVector(out.sizes()),
                      .op_param_cache_keys = std::move(param_keys)})));

  return AssignBufferToAtTensor(std::move(out_buf), out);
}

absl::Status Check3DPaddingIsValid(absl::Span<const int64_t> grad_output,
                                   absl::Span<const int64_t> padding) {
  constexpr size_t kNumSpatialDimensions = 3;
  constexpr std::array<std::string_view, kNumSpatialDimensions>
      spatial_dimension_names = {"depth", "height", "width"};

  // Dimension where the spatial dimensions begin.
  const size_t dimension_offset = grad_output.size() - kNumSpatialDimensions;

  for (size_t i = 0; i < kNumSpatialDimensions; i++) {
    // Padding dimensions correspond to the reversed `grad_output` dimensions.
    //
    //       Instead of being: depth-height-width
    // Padding dimensions are: width-height-depth
    const size_t padding_i = kNumSpatialDimensions - i - 1;

    const size_t beg = 2 * padding_i;
    const size_t end = 2 * padding_i + 1;
    const size_t grad_output_dimension = dimension_offset + i;

    const size_t padding_values_sum = padding[beg] + padding[end];
    const size_t grad_output_size = grad_output[grad_output_dimension];

    TT_RET_CHECK(padding_values_sum < grad_output_size, error::kInvalidArgument)
        << "expected padding at indices " << beg << " and " << end
        << " to sum to a value smaller than the grad_output "
        << spatial_dimension_names[i] << " (at dimension "
        << grad_output_dimension << ") of " << grad_output_size << ", got "
        << padding_values_sum << " (" << padding[beg] << " + " << padding[end]
        << ")";
  }

  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenReplicationPad1dOut(const at::Tensor& self,
                                    at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReplicationPad1dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(ReplicationPadHelper(
        BuildReplicationPadShlo, std::move(param_keys), self, padding, out,
        /*num_pad_dimensions=*/1));

    return out;
  });
}

at::Tensor& AtenReplicationPad2dOut(const at::Tensor& self,
                                    at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReplicationPad2dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(ReplicationPadHelper(
        BuildReplicationPadShlo, std::move(param_keys), self, padding, out,
        /*num_pad_dimensions=*/2));

    return out;
  });
}

at::Tensor& AtenReplicationPad3dOut(const at::Tensor& self,
                                    at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReplicationPad3dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(ReplicationPadHelper(
        BuildReplicationPadShlo, std::move(param_keys), self, padding, out,
        /*num_pad_dimensions=*/3));
    return out;
  });
}

at::Tensor& AtenReplicationPad1dBackwardGradInput(const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  at::IntArrayRef padding,
                                                  at::Tensor& grad_input) {
  TT_KERNEL(OpName::kReplicationPad1dBackwardGradInput, param_keys,
            (grad_output, self, padding, grad_input), {
              TT_THROW_IF_ERROR(ReplicationPadHelper(
                  BuildReplicationPadBackwardShlo, std::move(param_keys),
                  grad_output, padding, grad_input,
                  /*num_pad_dimensions=*/1));

              return grad_input;
            });
}
at::Tensor& AtenReplicationPad2dBackwardGradInput(const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  at::IntArrayRef padding,
                                                  at::Tensor& grad_input) {
  TT_KERNEL(OpName::kReplicationPad2dBackwardGradInput, param_keys,
            (grad_output, self, padding, grad_input), {
              TT_THROW_IF_ERROR(ReplicationPadHelper(
                  BuildReplicationPadBackwardShlo, std::move(param_keys),
                  grad_output, padding, grad_input,
                  /*num_pad_dimensions=*/2));

              return grad_input;
            });
}
at::Tensor& AtenReplicationPad3dBackwardGradInput(const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  at::IntArrayRef padding,
                                                  at::Tensor& grad_input) {
  TT_KERNEL(OpName::kReplicationPad3dBackwardGradInput, param_keys,
            (grad_output, self, padding, grad_input), {
              TT_THROW_IF_ERROR(ReplicationPadHelper(
                  BuildReplicationPadBackwardShlo, std::move(param_keys),
                  grad_output, padding, grad_input,
                  /*num_pad_dimensions=*/3));

              return grad_input;
            });
}
at::Tensor AtenReplicationPad2dBackward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        at::IntArrayRef padding) {
  TT_KERNEL(OpName::kReplicationPad2dBackward, _,
            (grad_output, self, IgnoreInCacheKey(padding, "Legacy usage")), {
              Dimensions gidims(grad_output.sizes().begin(),
                                grad_output.sizes().end());
              TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Current usages are
                               // guaranteed to be within range.
                  gidims.size() > 2, error::kInvalidArgument)
                  << "expected grad_output to have at least 2 dimensions"
                  << ", got " << gidims.size() << " dimensions";

              TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Current usages are
                               // guaranteed to be within range.
                  padding.size() == 4, error::kInvalidArgument)
                  << "expected padding to have " << 4 << " elements"
                  << ", got " << padding.size() << " elements";

              TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Current usages are
                               // guaranteed to be within range.
                  gidims[gidims.size() - 2] - (padding[2] + padding[3]) > 0 &&
                      gidims[gidims.size() - 1] - (padding[0] + padding[1]) > 0,
                  error::kInvalidArgument)
                  << "padding values must add up to a valid input dimension.";
              gidims[gidims.size() - 2] -= (padding[2] + padding[3]);
              gidims[gidims.size() - 1] -= (padding[0] + padding[1]);

              TT_CHECK_THROW(gidims == self.sizes(), error::kInvalidArgument)
                  << "expected the input shape to match the output (input "
                     "grad) shape "
                  << ToString(gidims)
                  << " computed by removing the padding from grad_output, "
                     "got "
                  << ToString(self.sizes());

              TT_ASSIGN_OR_THROW(
                  at::Tensor grad_input,
                  MakeEmptyTensor(gidims, self.scalar_type(), self.device()));
              AtenReplicationPad2dBackwardGradInput(grad_output, self, padding,
                                                    grad_input);
              return grad_input;
            });
}

at::Tensor AtenReplicationPad3dBackward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        at::IntArrayRef padding) {
  TT_KERNEL(
      OpName::kReplicationPad3dBackward, _,
      (grad_output, self, IgnoreInCacheKey(padding, "Legacy usage")), {
        Dimensions gidims(grad_output.sizes().begin(),
                          grad_output.sizes().end());
        TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Current usages are guaranteed
                         // to be within range.
            gidims.size() > 3, error::kInvalidArgument)
            << "expected grad_output to have at least 3 dimensions"
            << ", got " << gidims.size() << " dimensions";
        TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Current usages are guaranteed
                         // to be within range.
            padding.size() == 6, error::kInvalidArgument)
            << "expected padding to have " << 6 << " elements"
            << ", got " << padding.size() << " elements";
        TT_THROW_IF_ERROR(Check3DPaddingIsValid(grad_output.sizes(), padding));
        gidims[gidims.size() - 3] -= (padding[4] + padding[5]);
        gidims[gidims.size() - 2] -= (padding[2] + padding[3]);
        gidims[gidims.size() - 1] -= (padding[0] + padding[1]);
        TT_CHECK_THROW(gidims == self.sizes(), error::kInvalidArgument)
            << "expected the input shape to match the output (input "
               "grad) shape "
            << ToString(gidims)
            << " computed by removing the padding from the grad_output, "
               "got "
            << ToString(self.sizes());
        TT_ASSIGN_OR_THROW(
            at::Tensor grad_input,
            MakeEmptyTensor(gidims, self.scalar_type(), self.device()));
        AtenReplicationPad3dBackwardGradInput(grad_output, self, padding,
                                              grad_input);
        return grad_input;
      });
}

}  // namespace torch_tpu
