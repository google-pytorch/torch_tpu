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

#include "csrc/ops/replication_pad/replication_pad_aten_kernels.h"

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
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildReplicationPadShlo(mlir::MlirOp input,
                                                     Dimensions padding,
                                                     Dimensions output_shape,
                                                     int num_pad_dimensions) {
  Dimensions broadcast_dims(output_shape.size());
  absl::c_iota(broadcast_dims, 0);

  auto pad_fn = [&broadcast_dims](
                    mlir::MlirOp op, const int64_t dimension,
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
      // Slice the active dimension to get a 1 element strip on the left
      // then broadcast it to the padding width.
      left_dim[dimension] = 0;
      right_dim[dimension] = 1;
      mlir::MlirOp left_slice_op =
          mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
      right_dim[dimension] = left_pad;
      const mlir::RankedTensorType broadcast_type = mlir::RankedTensorType::get(
          right_dim, input_tensor_type.getElementType());
      ops.push_back(mlir::stablehlo::BroadcastInDim(
          broadcast_type, left_slice_op, broadcast_dims));
    }

    ops.push_back(op);

    if (right_pad > 0) {
      // Slice the active dimension to get a 1 element strip on the right
      // then broadcast it to the padding width.
      left_dim[dimension] = input_shape[dimension] - 1;
      right_dim[dimension] = input_shape[dimension];
      mlir::MlirOp right_slice_op =
          mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
      right_dim[dimension] = right_pad;
      const mlir::RankedTensorType broadcast_type = mlir::RankedTensorType::get(
          right_dim, input_tensor_type.getElementType());
      ops.push_back(mlir::stablehlo::BroadcastInDim(
          broadcast_type, right_slice_op, broadcast_dims));
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

// Propagates zero gradients for a negatively padded (truncated) edge.
// Precondition: padding_size is negative.
absl::StatusOr<mlir::MlirOp> BuildReplicationPadBackwardSideNegativePaddingShlo(
    mlir::MlirOp op, const PaddingDirection direction,
    const int64_t padding_size, const int64_t dimension) {
  const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(op);
  Dimensions zero_dim = CopyIntVector(input_tensor_type.getShape());
  zero_dim[dimension] = -padding_size;
  const mlir::RankedTensorType zero_type =
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

// Accumulates gradients from a positively padded (replicated) edge back onto
// the boundary element. Precondition: padding_size is positive and
// padding_size < op_shape[dimension].
absl::StatusOr<mlir::MlirOp> BuildReplicationPadBackwardSidePositivePaddingShlo(
    mlir::MlirOp op, const PaddingDirection direction,
    const int64_t padding_size, const int64_t dimension) {
  const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(op);
  const Dimensions input_shape = CopyIntVector(input_tensor_type.getShape());
  Dimensions left_dim(input_shape.size(), 0);
  Dimensions right_dim = input_shape;
  const Dimensions strides(input_shape.size(), 1);

  if (direction == PaddingDirection::kLeft) {
    left_dim[dimension] = 0;
    right_dim[dimension] = padding_size + 1;
    mlir::MlirOp pad_slice =
        mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp reduced,
        BuildSumShlo(pad_slice, {dimension}, ReductionMode::kKeepDims));
    if (padding_size + 1 == input_shape[dimension]) {
      return reduced;
    }
    left_dim[dimension] = padding_size + 1;
    right_dim[dimension] = input_shape[dimension];
    mlir::MlirOp remainder =
        mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
    return mlir::stablehlo::Concatenate(op.getBuilder(), {reduced, remainder},
                                        dimension);
  }

  const int64_t split = input_shape[dimension] - padding_size - 1;
  left_dim[dimension] = split;
  right_dim[dimension] = input_shape[dimension];
  mlir::MlirOp pad_slice =
      mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp reduced,
      BuildSumShlo(pad_slice, {dimension}, ReductionMode::kKeepDims));
  if (split == 0) {
    return reduced;
  }
  left_dim[dimension] = 0;
  right_dim[dimension] = split;
  mlir::MlirOp remainder =
      mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
  return mlir::stablehlo::Concatenate(op.getBuilder(), {remainder, reduced},
                                      dimension);
}

absl::StatusOr<mlir::MlirOp> BuildReplicationPadBackwardShlo(
    mlir::MlirOp input, Dimensions padding, Dimensions output_shape,
    int num_pad_dimensions) {
  // Process padding for a single dimension.
  auto pad_fn = [](mlir::MlirOp op, const int64_t dimension,
                   const int64_t left_pad,
                   const int64_t right_pad) -> absl::StatusOr<mlir::MlirOp> {
    // Apply negative pads before positive pads so mixed positive/negative
    // padding and size-1 input dimensions always have valid slice bounds.
    if (left_pad < 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReplicationPadBackwardSideNegativePaddingShlo(
                  op, PaddingDirection::kLeft, left_pad, dimension));
    }
    if (right_pad < 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReplicationPadBackwardSideNegativePaddingShlo(
                  op, PaddingDirection::kRight, right_pad, dimension));
    }
    if (left_pad > 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReplicationPadBackwardSidePositivePaddingShlo(
                  op, PaddingDirection::kLeft, left_pad, dimension));
    }
    if (right_pad > 0) {
      TT_ASSIGN_OR_RETURN(
          op, BuildReplicationPadBackwardSidePositivePaddingShlo(
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

absl::StatusOr<Dimensions> ValidateAndComputeReplicationPadOutputShape(
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
    TT_ASSIGN_OR_RETURN(const int64_t pad_sum,
                        SafeAdd(padding[2 * i], padding[2 * i + 1]));
    TT_ASSIGN_OR_RETURN(const int64_t out_size, SafeAdd(input_size, pad_sum));
    TT_RET_CHECK(out_size >= 1, error::kInvalidArgument)
        << "expected calculated output size at dimension " << dim
        << " to be >= 1, got " << out_size;
    output_shape[dim] = out_size;
  }
  return output_shape;
}

absl::StatusOr<at::Tensor> DispatchReplicationPad(
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

absl::StatusOr<at::Tensor> ReplicationPadForwardHelper(
    OpParamCacheKeys param_keys, const at::Tensor& self,
    const at::IntArrayRef padding, const int num_pad_dimensions,
    std::optional<std::reference_wrapper<at::Tensor>> out) {
  TT_ASSIGN_OR_RETURN(const Dimensions expected_out_shape,
                      ValidateAndComputeReplicationPadOutputShape(
                          self, padding, num_pad_dimensions));
  return DispatchReplicationPad(BuildReplicationPadShlo, std::move(param_keys),
                                self, padding, expected_out_shape,
                                num_pad_dimensions, out);
}

absl::StatusOr<at::Tensor> ReplicationPadBackwardHelper(
    OpParamCacheKeys param_keys, const at::Tensor& grad_output,
    const at::Tensor& self, const at::IntArrayRef padding,
    const int num_pad_dimensions,
    std::optional<std::reference_wrapper<at::Tensor>> grad_input) {
  TT_ASSIGN_OR_RETURN(const Dimensions expected_grad_output_shape,
                      ValidateAndComputeReplicationPadOutputShape(
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
  return DispatchReplicationPad(
      BuildReplicationPadBackwardShlo, std::move(param_keys), grad_output,
      padding, CopyIntVector(self.sizes()), num_pad_dimensions, grad_input);
}

}  // namespace

at::Tensor& AtenReplicationPad1dOut(const at::Tensor& self,
                                    at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReplicationPad1dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(
        ReplicationPadForwardHelper(std::move(param_keys), self, padding,
                                    /*num_pad_dimensions=*/1, std::ref(out))
            .status());
    return out;
  });
}

at::Tensor& AtenReplicationPad2dOut(const at::Tensor& self,
                                    at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReplicationPad2dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(
        ReplicationPadForwardHelper(std::move(param_keys), self, padding,
                                    /*num_pad_dimensions=*/2, std::ref(out))
            .status());
    return out;
  });
}

at::Tensor& AtenReplicationPad3dOut(const at::Tensor& self,
                                    at::IntArrayRef padding, at::Tensor& out) {
  TT_KERNEL(OpName::kReplicationPad3dOut, param_keys, (self, padding, out), {
    TT_THROW_IF_ERROR(
        ReplicationPadForwardHelper(std::move(param_keys), self, padding,
                                    /*num_pad_dimensions=*/3, std::ref(out))
            .status());
    return out;
  });
}

at::Tensor& AtenReplicationPad1dBackwardGradInput(const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  at::IntArrayRef padding,
                                                  at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kReplicationPad1dBackwardGradInput, param_keys,
      (grad_output, self, padding, grad_input), {
        TT_THROW_IF_ERROR(ReplicationPadBackwardHelper(
                              std::move(param_keys), grad_output, self, padding,
                              /*num_pad_dimensions=*/1, std::ref(grad_input))
                              .status());
        return grad_input;
      });
}

at::Tensor& AtenReplicationPad2dBackwardGradInput(const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  at::IntArrayRef padding,
                                                  at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kReplicationPad2dBackwardGradInput, param_keys,
      (grad_output, self, padding, grad_input), {
        TT_THROW_IF_ERROR(ReplicationPadBackwardHelper(
                              std::move(param_keys), grad_output, self, padding,
                              /*num_pad_dimensions=*/2, std::ref(grad_input))
                              .status());
        return grad_input;
      });
}

at::Tensor& AtenReplicationPad3dBackwardGradInput(const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  at::IntArrayRef padding,
                                                  at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kReplicationPad3dBackwardGradInput, param_keys,
      (grad_output, self, padding, grad_input), {
        TT_THROW_IF_ERROR(ReplicationPadBackwardHelper(
                              std::move(param_keys), grad_output, self, padding,
                              /*num_pad_dimensions=*/3, std::ref(grad_input))
                              .status());
        return grad_input;
      });
}

at::Tensor AtenReplicationPad2dBackward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        at::IntArrayRef padding) {
  TT_KERNEL(OpName::kReplicationPad2dBackward, param_keys,
            (grad_output, self, padding), {
              TT_ASSIGN_OR_THROW(
                  at::Tensor grad_input,
                  ReplicationPadBackwardHelper(
                      std::move(param_keys), grad_output, self, padding,
                      /*num_pad_dimensions=*/2, std::nullopt));
              return grad_input;
            });
}

at::Tensor AtenReplicationPad3dBackward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        at::IntArrayRef padding) {
  TT_KERNEL(OpName::kReplicationPad3dBackward, param_keys,
            (grad_output, self, padding), {
              TT_ASSIGN_OR_THROW(
                  at::Tensor grad_input,
                  ReplicationPadBackwardHelper(
                      std::move(param_keys), grad_output, self, padding,
                      /*num_pad_dimensions=*/3, std::nullopt));
              return grad_input;
            });
}

}  // namespace torch_tpu
