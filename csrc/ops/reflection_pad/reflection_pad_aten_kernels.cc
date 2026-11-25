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

#include "csrc/ops/reflection_pad/reflection_pad_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildReflectionPadShlo(mlir::MlirOp input,
                                                    Dimensions padding,
                                                    Dimensions output_shape,
                                                    int num_pad_dimensions) {
  auto pad_fn = [](mlir::MlirOp op, const int64_t dimension,
                   const int64_t left_pad,
                   const int64_t right_pad) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(op);
    const Dimensions input_shape = CopyIntVector(input_tensor_type.getShape());
    // Left and right dimensions for slicing (a vector of 0s and of max vals)
    // and a vector of stride 1.
    Dimensions left_dim(input_shape.size(), 0);
    Dimensions right_dim = input_shape;
    const Dimensions strides(input_shape.size(), 1);
    std::vector<mlir::MlirOp> ops;

    if (left_pad > 0) {
      // Slice the active dimension to get a left_pad-width strip one element
      // from the left then reflect it in the specified dimension.
      left_dim[dimension] = 1;
      right_dim[dimension] = left_pad + 1;
      mlir::MlirOp left_slice_op =
          mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
      const Dimensions reverse_dims{dimension};
      ops.push_back(mlir::stablehlo::Reverse(left_slice_op, reverse_dims));
    }

    ops.push_back(op);

    if (right_pad > 0) {
      // Slice the active dimension to get a right_pad-width strip one element
      // from the right then reflect it in the specified dimension.
      left_dim[dimension] = input_shape[dimension] - right_pad - 1;
      right_dim[dimension] = input_shape[dimension] - 1;
      mlir::MlirOp right_slice_op =
          mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
      const Dimensions reverse_dims(1, dimension);
      ops.push_back(mlir::stablehlo::Reverse(right_slice_op, reverse_dims));
    }

    // Apply positive pads before slicing negative pads so mixed positive and
    // negative padding on the same axis never produces an empty or inverted
    // intermediate slice.
    mlir::MlirOp padded =
        ops.size() == 1 ? op
                        : mlir::stablehlo::Concatenate(
                              op.getBuilder(),
                              mlir::ArrayRef<mlir::MlirOp>(ops), dimension);
    return BuildMaybeSlice(padded, dimension, left_pad, right_pad);
  };

  // Iterate over number of dimensions we need to pad.
  mlir::MlirOp result = input;
  for (int i = 0; i < num_pad_dimensions; ++i) {
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
  Dimensions input_shape = CopyIntVector(input_tensor_type.getShape());

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

  // Reverse the padding slice to undo the reflection that created the
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
}

// Generate code to handle propagation of zeros if the padding is negative.
// Precondition: padding_size is negative.
absl::StatusOr<mlir::MlirOp> BuildReflectionPadBackwardSideNegativePaddingShlo(
    mlir::MlirOp op, PaddingDirection direction, int64_t padding_size,
    int64_t dimension) {
  mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(op);
  Dimensions input_shape = CopyIntVector(input_tensor_type.getShape());

  // A negative pad value means we truncated the input.
  // So we have no gradients for those values. We construct a 0 tensor of
  // the appropriate size to concatenate.
  auto zero_dim = input_shape;
  zero_dim[dimension] = -padding_size;
  auto zero_type =
      mlir::RankedTensorType::get(zero_dim, input_tensor_type.getElementType());

  TT_ASSIGN_OR_RETURN(mlir::MlirOp zero_scalar_op,
                      ComplexSafeConvertElementType(
                          mlir::stablehlo::Constant(op.getBuilder(), 0),
                          GetElementTypeOrDie(op)));
  mlir::MlirOp zero_tensor_op =
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
  auto pad_fn = [](mlir::MlirOp op, const int64_t dimension,
                   const int64_t left_pad,
                   const int64_t right_pad) -> absl::StatusOr<mlir::MlirOp> {
    // Apply negative pads (zero-padding truncated edges) before positive pads
    // (folding reflected edges) so that mixed positive and negative padding on
    // the same axis always has enough elements to slice the reflected region.
    if (left_pad < 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReflectionPadBackwardSideNegativePaddingShlo(
                  op, PaddingDirection::kLeft, left_pad, dimension));
    }
    if (right_pad < 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReflectionPadBackwardSideNegativePaddingShlo(
                  op, PaddingDirection::kRight, right_pad, dimension));
    }
    if (left_pad > 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReflectionPadBackwardSidePositivePaddingShlo(
                  op, PaddingDirection::kLeft, left_pad, dimension));
    }
    if (right_pad > 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReflectionPadBackwardSidePositivePaddingShlo(
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

absl::StatusOr<Dimensions> ValidateAndComputeReflectionPadOutputShape(
    const at::Tensor& self, const at::IntArrayRef padding,
    const int num_pad_dimensions) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Enforced by ATen schema SymInt[2*N].
      padding.size() == static_cast<size_t>(num_pad_dimensions * 2),
      error::kInvalidArgument)
      << "expected padding to have " << (num_pad_dimensions * 2) << " elements"
      << ", got " << padding.size() << " elements";

  const int64_t ndim = self.dim();
  const bool is_batch_mode = ndim == num_pad_dimensions + 2;
  bool valid_input = is_batch_mode || ndim == num_pad_dimensions + 1;
  if (valid_input) {
    for (int64_t d = is_batch_mode ? 1 : 0; d < ndim; ++d) {
      if (self.size(d) == 0) {
        valid_input = false;
        break;
      }
    }
  }
  TT_RET_CHECK(valid_input, error::kInvalidArgument)
      << "expected a " << (num_pad_dimensions + 1) << "D or "
      << (num_pad_dimensions + 2)
      << "D (batch mode) tensor with possibly 0 batch size and other non-zero "
         "dimensions for input, got shape "
      << ToString(self.sizes());

  Dimensions output_shape = CopyIntVector(self.sizes());
  for (int i = 0; i < num_pad_dimensions; ++i) {
    const int64_t dim = ndim - 1 - i;
    const int64_t input_size = self.size(dim);
    const int64_t left_pad = padding[2 * i];
    const int64_t right_pad = padding[2 * i + 1];
    TT_RET_CHECK(left_pad < input_size && right_pad < input_size,
                 error::kInvalidArgument)
        << "expected padding size to be less than the corresponding input "
           "dimension size "
        << input_size << " at dimension " << dim << ", got (" << left_pad
        << ", " << right_pad << ")";
    TT_ASSIGN_OR_RETURN(const int64_t pad_sum, SafeAdd(left_pad, right_pad));
    TT_ASSIGN_OR_RETURN(const int64_t out_size, SafeAdd(input_size, pad_sum));
    TT_RET_CHECK(out_size >= 1, error::kInvalidArgument)
        << "expected calculated output size at dimension " << dim
        << " to be >= 1, got " << out_size;
    output_shape[dim] = out_size;
  }
  return output_shape;
}

absl::StatusOr<at::Tensor> DispatchReflectionPad(
    absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(
        mlir::MlirOp input, Dimensions padding, Dimensions output_shape,
        int num_pad_dimensions) const>
        shlo_builder_function,
    OpParamCacheKeys param_keys, const at::Tensor& input,
    const at::IntArrayRef padding, const Dimensions& expected_out_shape,
    const int num_pad_dimensions,
    std::optional<std::reference_wrapper<at::Tensor>> out) {
  if (out.has_value()) {
    TT_RETURN_IF_ERROR(ValidateOutDtype(out->get(), input.scalar_type(),
                                        /*allow_out_dtype_cast=*/false));
  }

  if (absl::c_any_of(expected_out_shape,
                     [](const int64_t d) { return d == 0; })) {
    if (out.has_value()) {
      TT_RETURN_IF_ERROR(
          ResizeTensorIfShapeDiffers(out->get(), expected_out_shape));
      return out->get();
    }
    return MakeEmptyTensor(expected_out_shape, input.scalar_type(),
                           input.device());
  }

  TT_RET_CHECK(!IsBool(input), error::kInvalidArgument)
      << "not implemented for bool";

  Dimensions padding_vec = CopyIntVector(padding);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  auto op_builder = [padding_vec = std::move(padding_vec),
                     out_shape = expected_out_shape, num_pad_dimensions,
                     shlo_builder_function = std::move(shlo_builder_function)](
                        mlir::MlirOp op) -> absl::StatusOr<mlir::MlirOp> {
    return shlo_builder_function(op, padding_vec, out_shape,
                                 num_pad_dimensions);
  };

  if (out.has_value()) {
    TT_RETURN_IF_ERROR(
        DispatchOpOut<1>(std::move(op_builder), input, out->get(),
                         {.out_dtype = element_type,
                          .out_dims = expected_out_shape,
                          .op_param_cache_keys = std::move(param_keys)}));
    return out->get();
  }

  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef result_buf,
      DispatchOp<1>(std::move(op_builder), input,
                    {.out_dtype = element_type,
                     .out_dims = expected_out_shape,
                     .op_param_cache_keys = std::move(param_keys)}));
  return MakeTensor(std::move(result_buf));
}

absl::StatusOr<at::Tensor> ReflectionPadForwardHelper(
    OpParamCacheKeys param_keys, const at::Tensor& self,
    const at::IntArrayRef padding, const int num_pad_dimensions,
    std::optional<std::reference_wrapper<at::Tensor>> out) {
  TT_ASSIGN_OR_RETURN(const Dimensions expected_out_shape,
                      ValidateAndComputeReflectionPadOutputShape(
                          self, padding, num_pad_dimensions));
  return DispatchReflectionPad(BuildReflectionPadShlo, std::move(param_keys),
                               self, padding, expected_out_shape,
                               num_pad_dimensions, out);
}

absl::StatusOr<at::Tensor> ReflectionPadBackwardHelper(
    OpParamCacheKeys param_keys, const at::Tensor& grad_output,
    const at::Tensor& self, const at::IntArrayRef padding,
    const int num_pad_dimensions,
    std::optional<std::reference_wrapper<at::Tensor>> grad_input) {
  TT_ASSIGN_OR_RETURN(const Dimensions expected_grad_output_shape,
                      ValidateAndComputeReflectionPadOutputShape(
                          self, padding, num_pad_dimensions));
  TT_RET_CHECK(grad_output.sizes() == expected_grad_output_shape,
               error::kInvalidArgument)
      << "expected grad_output to have shape "
      << ToString(expected_grad_output_shape) << ", got "
      << ToString(grad_output.sizes());
  TT_RET_CHECK(grad_output.scalar_type() == self.scalar_type(),
               error::kInvalidArgument)
      << "expected grad_output to have dtype " << ToString(self.scalar_type())
      << ", got " << ToString(grad_output.scalar_type());
  return DispatchReflectionPad(
      BuildReflectionPadBackwardShlo, std::move(param_keys), grad_output,
      padding, CopyIntVector(self.sizes()), num_pad_dimensions, grad_input);
}

}  // namespace

at::Tensor& AtenReflectionPad1dOut(const at::Tensor& self,
                                   at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReflectionPad1dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(
        ReflectionPadForwardHelper(std::move(param_keys), self, padding,
                                   /*num_pad_dimensions=*/1, std::ref(out))
            .status());
    return out;
  });
}

at::Tensor AtenReflectionPad2d(const at::Tensor& self,
                               at::IntArrayRef padding) {
  TT_KERNEL(OpName::kReflectionPad2d, param_keys, (self, padding), {
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        ReflectionPadForwardHelper(std::move(param_keys), self, padding,
                                   /*num_pad_dimensions=*/2, std::nullopt));
    return out;
  });
}

at::Tensor& AtenReflectionPad2dOut(const at::Tensor& self,
                                   at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReflectionPad2dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(
        ReflectionPadForwardHelper(std::move(param_keys), self, padding,
                                   /*num_pad_dimensions=*/2, std::ref(out))
            .status());
    return out;
  });
}

at::Tensor& AtenReflectionPad3dOut(const at::Tensor& self,
                                   at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReflectionPad3dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(
        ReflectionPadForwardHelper(std::move(param_keys), self, padding,
                                   /*num_pad_dimensions=*/3, std::ref(out))
            .status());
    return out;
  });
}

at::Tensor& AtenReflectionPad1dBackwardGradInput(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 at::IntArrayRef padding,
                                                 at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kReflectionPad1dBackwardGradInput, param_keys,
      (grad_output, self, padding, grad_input), {
        TT_THROW_IF_ERROR(ReflectionPadBackwardHelper(
                              std::move(param_keys), grad_output, self, padding,
                              /*num_pad_dimensions=*/1, std::ref(grad_input))
                              .status());
        return grad_input;
      });
}

at::Tensor& AtenReflectionPad2dBackwardGradInput(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 at::IntArrayRef padding,
                                                 at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kReflectionPad2dBackwardGradInput, param_keys,
      (grad_output, self, padding, grad_input), {
        TT_THROW_IF_ERROR(ReflectionPadBackwardHelper(
                              std::move(param_keys), grad_output, self, padding,
                              /*num_pad_dimensions=*/2, std::ref(grad_input))
                              .status());
        return grad_input;
      });
}

at::Tensor& AtenReflectionPad3dBackwardGradInput(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 at::IntArrayRef padding,
                                                 at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kReflectionPad3dBackwardGradInput, param_keys,
      (grad_output, self, padding, grad_input), {
        TT_THROW_IF_ERROR(ReflectionPadBackwardHelper(
                              std::move(param_keys), grad_output, self, padding,
                              /*num_pad_dimensions=*/3, std::ref(grad_input))
                              .status());
        return grad_input;
      });
}

at::Tensor AtenReflectionPad2dBackward(const at::Tensor& grad_output,
                                       const at::Tensor& self,
                                       at::IntArrayRef padding) {
  TT_KERNEL(OpName::kReflectionPad2dBackward, param_keys,
            (grad_output, self, padding), {
              TT_ASSIGN_OR_THROW(
                  at::Tensor grad_input,
                  ReflectionPadBackwardHelper(
                      std::move(param_keys), grad_output, self, padding,
                      /*num_pad_dimensions=*/2, std::nullopt));
              return grad_input;
            });
}

}  // namespace torch_tpu
