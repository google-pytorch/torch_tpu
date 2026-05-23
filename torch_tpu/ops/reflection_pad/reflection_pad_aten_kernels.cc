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

#include "torch_tpu/ops/reflection_pad/reflection_pad_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildReflectionPadShlo(mlir::MlirOp input,
                                                    Dimensions padding,
                                                    Dimensions output_shape,
                                                    size_t num_pad_dimensions) {
  auto pad_fn = [](mlir::MlirOp op, int64_t dimension, int64_t left_pad,
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
      // Slice the active dimension to get a left_pad-width strip one element
      // from the left then reflect it in the specified dimension
      left_dim[dimension] = 1;
      right_dim[dimension] = left_pad + 1;
      auto left_slice_op =
          mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
      Dimensions reverse_dims{dimension};
      ops.push_back(mlir::stablehlo::Reverse(left_slice_op, reverse_dims));
    }

    // Simply append or, if either pad value is negative, truncate the input
    // tensor.
    TT_ASSIGN_OR_RETURN(auto core_op,
                        BuildMaybeSlice(op, dimension, left_pad, right_pad));
    ops.push_back(core_op);

    if (right_pad > 0) {
      // Slice the active dimension to get a right_pad-width strip one element
      // from the right then reflect it in the specified dimension
      left_dim[dimension] = input_shape[dimension] - right_pad - 1;
      right_dim[dimension] = input_shape[dimension] - 1;
      auto right_slice_op =
          mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
      Dimensions reverse_dims(1, dimension);
      ops.push_back(mlir::stablehlo::Reverse(right_slice_op, reverse_dims));
    }
    // Concatenate the two padding strips with the original tensor between them.
    return mlir::stablehlo::Concatenate(
        op.getBuilder(), mlir::ArrayRef<mlir::MlirOp>(ops), dimension);
  };

  // Iterate over number of dimensions we need to pad
  mlir::MlirOp result = input;
  for (size_t i = 0; i < num_pad_dimensions; ++i) {
    TT_ASSIGN_OR_RETURN(result, pad_fn(result, output_shape.size() - i - 1,
                                       padding[i * 2], padding[i * 2 + 1]));
  }
  return result;
}

enum class PaddingDirection {
  kLeft,
  kRight,
};

// Generate code to handle accumulation of the padding slice, or propagation of
// zeros if the padding is negative. Precondition: padding_size is positive.
absl::StatusOr<mlir::MlirOp> BuildReflectionPadBackwardSidePositivePaddingShlo(
    mlir::MlirOp op, PaddingDirection direction, int64_t padding_size,
    int64_t dimension) {
  mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(op);
  Dimensions input_shape(input_tensor_type.getShape().begin(),
                         input_tensor_type.getShape().end());

  // Slice off the padding and an equivalent size strip from the input.
  // Reflect the padding and add to the input slice.
  // Concatenate the result with the remaining input slice.

  // Initialize the slice dimensions to cover the whole tensor.
  auto left_dim = Dimensions(input_shape.size(), 0);
  auto right_dim = input_shape;
  auto strides = Dimensions(input_shape.size(), 1);

  // Slice off the padding.
  if (direction == PaddingDirection::kLeft) {
    left_dim[dimension] = 0;
    right_dim[dimension] = padding_size;
  } else {
    left_dim[dimension] = input_shape[dimension] - padding_size;
    right_dim[dimension] = input_shape[dimension];
  }
  auto padding_slice_op =
      mlir::stablehlo::Slice(op, left_dim, right_dim, strides);

  // Slice off the edge value that was not part of the reflected padding.
  if (direction == PaddingDirection::kLeft) {
    left_dim[dimension] = padding_size;
    right_dim[dimension] = padding_size + 1;
  } else {
    left_dim[dimension] = input_shape[dimension] - padding_size - 1;
    right_dim[dimension] = input_shape[dimension] - padding_size;
  }
  auto edge_slice_op = mlir::stablehlo::Slice(op, left_dim, right_dim, strides);

  // Slice off the part of the tensor that was reflected to create the
  // padding.
  if (direction == PaddingDirection::kLeft) {
    left_dim[dimension] = padding_size + 1;
    right_dim[dimension] = padding_size * 2 + 1;
  } else {
    left_dim[dimension] = input_shape[dimension] - padding_size * 2 - 1;
    right_dim[dimension] = input_shape[dimension] - padding_size - 1;
  }
  auto input_slice_op =
      mlir::stablehlo::Slice(op, left_dim, right_dim, strides);

  // Slice off the remainder of the tensor that was not involved in padding.
  if (direction == PaddingDirection::kLeft) {
    left_dim[dimension] = padding_size * 2 + 1;
    right_dim[dimension] = input_shape[dimension];
  } else {
    left_dim[dimension] = 0;
    right_dim[dimension] = input_shape[dimension] - padding_size * 2 - 1;
  }
  auto remainder_op = mlir::stablehlo::Slice(op, left_dim, right_dim, strides);

  // Reverse the padding slice to under to reflection that created the
  // padding. Then add to the source part of the input slice to get the
  // accumulated gradients.
  Dimensions reverse_dims{dimension};
  auto flipped_padding_slice_op =
      mlir::stablehlo::Reverse(padding_slice_op, reverse_dims);
  auto accumulated_op =
      mlir::stablehlo::Add(flipped_padding_slice_op, input_slice_op);

  // Concatenate the accumulated slice with the remainder and the edge piece
  // and return the gradient tensor.
  if (direction == PaddingDirection::kLeft) {
    return mlir::stablehlo::Concatenate(
        op.getBuilder(), {edge_slice_op, accumulated_op, remainder_op},
        dimension);
  } else {
    return mlir::stablehlo::Concatenate(
        op.getBuilder(), {remainder_op, accumulated_op, edge_slice_op},
        dimension);
  }

  return mlir::MlirOp();
}

// Generate code to handle accumulation of the padding slice, or propagation of
// zeros if the padding is negative. Precondition: padding_size is negative.
absl::StatusOr<mlir::MlirOp> BuildReflectionPadBackwardSideNegativePaddingShlo(
    mlir::MlirOp op, PaddingDirection direction, int64_t padding_size,
    int64_t dimension) {
  mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(op);
  Dimensions input_shape(input_tensor_type.getShape().begin(),
                         input_tensor_type.getShape().end());

  // A negative pad value means we truncated the input.
  // So we have no gradients for those values. We construct a 0 tensor of
  // the appropriate size to concatenate.
  auto zero_dim = input_shape;
  zero_dim[dimension] = -padding_size;
  auto zero_type =
      mlir::RankedTensorType::get(zero_dim, input_tensor_type.getElementType());

  auto zero_scalar_op = mlir::stablehlo::ConvertElementType(
      mlir::stablehlo::Constant(op.getBuilder(), 0),
      input_tensor_type.getElementType());
  auto zero_tensor_op =
      mlir::stablehlo::BroadcastInDim(zero_type, zero_scalar_op, {});

  if (direction == PaddingDirection::kLeft) {
    return mlir::stablehlo::Concatenate(op.getBuilder(), {zero_tensor_op, op},
                                        dimension);
  }

  return mlir::stablehlo::Concatenate(op.getBuilder(), {op, zero_tensor_op},
                                      dimension);
}

absl::StatusOr<mlir::MlirOp> BuildReflectionPadBackwardShlo(
    mlir::MlirOp input, Dimensions padding, Dimensions output_shape,
    int num_pad_dimensions) {
  // Process padding for a single dimension.
  auto pad_fn = [](mlir::MlirOp op, int64_t dimension, int64_t left_pad,
                   int64_t right_pad) -> absl::StatusOr<mlir::MlirOp> {
    std::vector<mlir::MlirOp> ops;

    // If we had left padding (positive or negative) we need to slice and
    // accumulate or generate 0 gradients.
    if (left_pad > 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReflectionPadBackwardSidePositivePaddingShlo(
                  op, PaddingDirection::kLeft, left_pad, dimension));
    } else if (left_pad < 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReflectionPadBackwardSideNegativePaddingShlo(
                  op, PaddingDirection::kLeft, left_pad, dimension));
    }

    // If we had right padding (positive or negative) we need to slice and
    // accumulate or generate 0 gradients.
    if (right_pad != 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReflectionPadBackwardSidePositivePaddingShlo(
                  op, PaddingDirection::kRight, right_pad, dimension));
    } else if (right_pad < 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReflectionPadBackwardSideNegativePaddingShlo(
                  op, PaddingDirection::kRight, right_pad, dimension));
    }

    return op;
  };

  // Iterate over number of dimensions we need to pad.
  mlir::MlirOp result = input;
  for (int i = 0; i < num_pad_dimensions; ++i) {
    TT_ASSIGN_OR_RETURN(result, pad_fn(result, output_shape.size() - i - 1,
                                       padding[i * 2], padding[i * 2 + 1]));
  }
  return result;
}

absl::Status ReflectionPadHelper(
    absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(
        mlir::MlirOp input, Dimensions padding, Dimensions output_shape,
        int num_pad_dimensions) const>
        shlo_builder_function,
    OpParamCacheKeys param_keys, const at::Tensor& self,
    at::IntArrayRef padding, at::Tensor& out, int num_pad_dimensions) {
  TT_RET_CHECK(self.scalar_type() != at::ScalarType::Bool,
               error::kInvalidArgument)
      << "not implemented for bool";

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

}  // namespace

at::Tensor& AtenReflectionPad1dOut(const at::Tensor& self,
                                   at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReflectionPad1dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(ReflectionPadHelper(
        BuildReflectionPadShlo, std::move(param_keys), self, padding, out,
        /*num_pad_dimensions=*/1));

    return out;
  });
}

at::Tensor AtenReflectionPad2d(const at::Tensor& self,
                               at::IntArrayRef padding) {
  TT_KERNEL(
      OpName::kReflectionPad2d, _,
      (self, IgnoreInCacheKey(padding, "Legacy usage")), {
        Dimensions out_dimensions(self.sizes().begin(), self.sizes().end());
        out_dimensions[out_dimensions.size() - 2] += padding[2] + padding[3];
        out_dimensions[out_dimensions.size() - 1] += padding[0] + padding[1];

        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(out_dimensions, self.scalar_type(), self.device()));
        return AtenReflectionPad2dOut(self, padding, out);
      });
}

at::Tensor& AtenReflectionPad2dOut(const at::Tensor& self,
                                   at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReflectionPad2dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(ReflectionPadHelper(
        BuildReflectionPadShlo, std::move(param_keys), self, padding, out,
        /*num_pad_dimensions=*/2));

    return out;
  });
}

at::Tensor& AtenReflectionPad3dOut(const at::Tensor& self,
                                   at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReflectionPad3dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(ReflectionPadHelper(
        BuildReflectionPadShlo, std::move(param_keys), self, padding, out,
        /*num_pad_dimensions=*/3));
    return out;
  });
}

at::Tensor& AtenReflectionPad1dBackwardGradInput(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 at::IntArrayRef padding,
                                                 at::Tensor& grad_input) {
  TT_KERNEL(OpName::kReflectionPad1dBackwardGradInput, param_keys,
            (grad_output, self, padding, grad_input), {
              TT_THROW_IF_ERROR(ReflectionPadHelper(
                  BuildReflectionPadBackwardShlo, std::move(param_keys),
                  grad_output, padding, grad_input,
                  /*num_pad_dimensions=*/1));

              return grad_input;
            });
}

at::Tensor& AtenReflectionPad2dBackwardGradInput(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 at::IntArrayRef padding,
                                                 at::Tensor& grad_input) {
  TT_KERNEL(OpName::kReflectionPad2dBackwardGradInput, param_keys,
            (grad_output, self, padding, grad_input), {
              TT_THROW_IF_ERROR(ReflectionPadHelper(
                  BuildReflectionPadBackwardShlo, std::move(param_keys),
                  grad_output, padding, grad_input,
                  /*num_pad_dimensions=*/2));

              return grad_input;
            });
}
at::Tensor& AtenReflectionPad3dBackwardGradInput(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 at::IntArrayRef padding,
                                                 at::Tensor& grad_input) {
  TT_KERNEL(OpName::kReflectionPad3dBackwardGradInput, param_keys,
            (grad_output, self, padding, grad_input), {
              TT_THROW_IF_ERROR(ReflectionPadHelper(
                  BuildReflectionPadBackwardShlo, std::move(param_keys),
                  grad_output, padding, grad_input,
                  /*num_pad_dimensions=*/3));

              return grad_input;
            });
}
at::Tensor AtenReflectionPad2dBackward(const at::Tensor& grad_output,
                                       const at::Tensor& self,
                                       at::IntArrayRef padding) {
  TT_KERNEL(OpName::kReflectionPad2dBackward, _,
            (grad_output, self, IgnoreInCacheKey(padding, "Legacy usage")), {
              Dimensions gidims(grad_output.sizes().begin(),
                                grad_output.sizes().end());
              TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Current usages are
                               // guaranteed to be within range.
                  gidims.size() > 2, error::kInvalidArgument)
                  << "expected grad_output to have at least 2 dimensions"
                  << ", got " << gidims.size();

              TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Current usages are
                               // guaranteed to be within range.
                  padding.size() == 4, error::kInvalidArgument)
                  << "expected padding to have " << 4 << " elements"
                  << ", got " << padding.size();

              TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Current usages are
                               // guaranteed to be within range.
                  gidims[gidims.size() - 2] - (padding[2] + padding[3]) > 0 &&
                      gidims[gidims.size() - 1] - (padding[0] + padding[1]) > 0,
                  error::kInvalidArgument)
                  << "padding values must add up to a valid input dimension";
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
              AtenReflectionPad2dBackwardGradInput(grad_output, self, padding,
                                                   grad_input);
              return grad_input;
            });
}

}  // namespace torch_tpu
