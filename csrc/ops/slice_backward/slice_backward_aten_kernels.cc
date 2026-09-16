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

#include "csrc/ops/slice_backward/slice_backward_aten_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Builds the StableHLO graph for `aten::select_backward`.
//
// `select(dim, index)` extracts a `(rank - 1)` slice at `index` along `dim`.
// Its backward pass embeds `grad_output` back into a full `input_sizes` tensor
// filled with zeros everywhere except at `index` along `dim`.
//
// Architectural note on `DynamicUpdateSlice` vs `Concatenate`:
// In unrolled step-by-step recurrence loops (such as Mamba-3 or LSTM backward
// scans), autograd accumulates `select_backward` gradients across timesteps:
//   `acc = acc + select_backward(du_t, dim, t)`
// If `select_backward` emitted a 3-way `Concatenate([Zero_left, grad,
// Zero_right])`, XLA's `AlgebraicSimplifier` cannot fold `Add(acc,
// Concatenate(...))` into an in-place buffer update unless the FX
// slice-addition-tree pass (`cl/980341330`) is active, causing O(L) full-tensor
// materializations. By emitting `DynamicUpdateSlice(BroadcastInDim(0),
// reshaped_grad, start_indices)`, XLA's `AlgebraicSimplifier` natively folds
// `Add(acc, DUS(0, grad, idx))` into `DynamicUpdateSlice(acc,
// Add(DynamicSlice(acc, idx), grad), idx)` for in-place execution even in
// standalone eager/compiled graphs.
absl::StatusOr<mlir::MlirOp> BuildSelectBackwardShlo(
    mlir::MlirOp grad_op, const Dimensions& input_sizes, const int64_t dim,
    const int64_t index, const mlir::ElementType element_type) {
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_op);
  mlir::MlirBuilder& builder = grad_op.getBuilder();

  // Unsqueeze `grad_op` from `(rank - 1)` to `rank` with size 1 along `dim`.
  Dimensions unsqueezed_shape = input_sizes;
  unsqueezed_shape[dim] = 1;
  const auto unsqueezed_type =
      mlir::RankedTensorType::get(unsqueezed_shape, grad_type.getElementType());
  mlir::MlirOp reshaped_grad =
      mlir::stablehlo::Reshape(unsqueezed_type, grad_op);

  // If the target dimension has size 1, the unsqueezed gradient already
  // spans the entire output tensor.
  if (input_sizes[dim] == 1) {
    return reshaped_grad;
  }

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp zero_scalar,
      MakeConstant(builder, at::Scalar(0), element_type, /*shape=*/{}));
  const auto out_type =
      mlir::RankedTensorType::get(input_sizes, grad_type.getElementType());
  mlir::MlirOp zero_tensor =
      mlir::stablehlo::BroadcastInDim(out_type, zero_scalar, {});

  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(input_sizes.size());
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    const int64_t start_idx = (static_cast<int64_t>(i) == dim) ? index : 0;
    start_indices.push_back(
        MakeScalarConstant(builder, start_idx, mlir::ElementType::I64));
  }
  return mlir::stablehlo::DynamicUpdateSlice(zero_tensor, reshaped_grad,
                                             start_indices);
}

// Helper for unit-stride (`step == 1`) `slice_backward` lowerings.
// Emits 2-way `Concatenate` for boundary-anchored slices spanning at least half
// the dimension (`slice_len * 2 >= total_len`), and `DynamicUpdateSlice`
// otherwise.
mlir::MlirOp BuildUnitStrideSliceBackwardShlo(mlir::MlirOp grad_op,
                                              mlir::MlirOp zero_scalar,
                                              const Dimensions& input_sizes,
                                              const int64_t dim,
                                              const int64_t start,
                                              const int64_t end) {
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_op);
  mlir::MlirBuilder& builder = grad_op.getBuilder();
  const int64_t total_len = input_sizes[dim];
  const int64_t slice_len = end - start;

  if ((start == 0 || end == total_len) && slice_len * 2 >= total_len) {
    std::vector<mlir::MlirOp> parts;
    parts.reserve(2);
    if (start > 0) {
      Dimensions left_shape = input_sizes;
      left_shape[dim] = start;
      const auto left_type =
          mlir::RankedTensorType::get(left_shape, grad_type.getElementType());
      parts.push_back(
          mlir::stablehlo::BroadcastInDim(left_type, zero_scalar, {}));
    }
    parts.push_back(grad_op);
    if (end < total_len) {
      Dimensions right_shape = input_sizes;
      right_shape[dim] = total_len - end;
      const auto right_type =
          mlir::RankedTensorType::get(right_shape, grad_type.getElementType());
      parts.push_back(
          mlir::stablehlo::BroadcastInDim(right_type, zero_scalar, {}));
    }
    return mlir::stablehlo::Concatenate(
        builder, mlir::ArrayRef<mlir::MlirOp>(parts), dim);
  }

  const auto out_type =
      mlir::RankedTensorType::get(input_sizes, grad_type.getElementType());
  mlir::MlirOp zero_tensor =
      mlir::stablehlo::BroadcastInDim(out_type, zero_scalar, {});

  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(input_sizes.size());
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    const int64_t start_idx = (static_cast<int64_t>(i) == dim) ? start : 0;
    start_indices.push_back(
        MakeScalarConstant(builder, start_idx, mlir::ElementType::I64));
  }
  return mlir::stablehlo::DynamicUpdateSlice(zero_tensor, grad_op,
                                             start_indices);
}

// Builds the StableHLO graph for `aten::slice_backward`.
//
// Dispatches across three specialized StableHLO patterns depending on slice
// geometry and stride:
//   1. Identity (`step == 1 && slice_len == total_len`): Returns `grad_op`
//      directly without allocation.
//   2. Large boundary-anchored unit-stride slices (`step == 1`, anchored at
//      `start == 0` or `end == total_len`, spanning `slice_len >= total_len /
//      2`): Emits a 2-way `stablehlo.concatenate` joining `grad_op` with a
//      single zero-broadcast padding block. This eliminates full-tensor zero
//      broadcast
//      + DUS overhead on Causal LM shifted logits (`logits[:, :-1, :]`) and
//      RoPE half-slices (`x[..., :d//2]`, `x[..., d//2:]`).
//   3. Interior or small unit-stride slices (`step == 1`): Emits
//      `stablehlo.dynamic_update_slice` into a zero broadcast so XLA's
//      `AlgebraicSimplifier` can fold recurrence gradient accumulations
//      in-place.
//   4. Strided slices (`step > 1`): Emits `stablehlo.pad` with
//      `interior_pad = step - 1` to scatter gradients across strided positions.
absl::StatusOr<mlir::MlirOp> BuildSliceBackwardShlo(
    mlir::MlirOp grad_op, const Dimensions& input_sizes, const int64_t dim,
    const int64_t start, const int64_t end, const int64_t step,
    const mlir::ElementType element_type) {
  const int64_t total_len = input_sizes[dim];
  const int64_t slice_len = end - start;
  if (step == 1 && slice_len == total_len) {
    return grad_op;
  }

  mlir::MlirBuilder& builder = grad_op.getBuilder();
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp zero_scalar,
      MakeConstant(builder, at::Scalar(0), element_type, /*shape=*/{}));

  if (step == 1) {
    return BuildUnitStrideSliceBackwardShlo(grad_op, zero_scalar, input_sizes,
                                            dim, start, end);
  }

  // For strided slices (`step > 1`), use StableHLO Pad with interior padding
  // to interleave `(step - 1)` zeros between each element of `grad_op`.
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_op);
  const int64_t grad_len = grad_type.getShape()[dim];
  Dimensions low_pad(input_sizes.size(), 0);
  Dimensions high_pad(input_sizes.size(), 0);
  Dimensions interior_pad(input_sizes.size(), 0);
  low_pad[dim] = start;
  interior_pad[dim] = step - 1;
  high_pad[dim] = total_len - start - (grad_len - 1) * step - 1;
  return mlir::stablehlo::Pad(grad_op, zero_scalar, low_pad, high_pad,
                              interior_pad);
}

}  // namespace

at::Tensor AtenSelectBackward(const at::Tensor& grad_output,
                              at::IntArrayRef input_sizes, int64_t dim,
                              int64_t index) {
  const int64_t ndim = input_sizes.size();
  TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, ndim));
  const int64_t dim_size = input_sizes[wrapped_dim];
  if (index < 0) {
    index += dim_size;
  }
  TT_CHECK_THROW(index >= 0 && index < dim_size, error::kPythonIndexError)
      << "index out of range";

  if (std::any_of(input_sizes.begin(), input_sizes.end(),
                  [](const int64_t d) { return d == 0; })) {
    return at::zeros(input_sizes, grad_output.options());
  }

  const Dimensions input_sizes_vec = CopyIntVector(input_sizes);

  TT_KERNEL(
      OpName::kSelectBackward, param_keys,
      (grad_output, input_sizes_vec, wrapped_dim, index), {
        TT_ASSIGN_OR_THROW(
            const mlir::ElementType element_type,
            ConvertTo<mlir::ElementType>(grad_output.scalar_type()));

        auto op_builder =
            [input_sizes_vec, wrapped_dim, index,
             element_type](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildSelectBackwardShlo(input, input_sizes_vec, wrapped_dim,
                                         index, element_type);
        };

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<1>(std::move(op_builder), grad_output,
                           {.out_dtype = element_type,
                            .out_dims = input_sizes_vec,
                            .op_param_cache_keys = std::move(param_keys)})));
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(input_sizes_vec, grad_output.scalar_type(),
                            grad_output.device()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor AtenSliceBackward(const at::Tensor& grad_output,
                             at::IntArrayRef input_sizes, int64_t dim,
                             int64_t start, int64_t end, int64_t step) {
  TT_CHECK_THROW(step > 0, error::kInvalidArgument)
      << "slice step must be positive";
  const int64_t ndim = input_sizes.size();
  TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, ndim));
  const int64_t dim_size = input_sizes[wrapped_dim];

  if (start < 0) {
    start += dim_size;
  }
  start = std::clamp<int64_t>(start, 0, dim_size);
  if (end < 0) {
    end += dim_size;
  }
  end = std::clamp<int64_t>(end, start, dim_size);

  if (grad_output.numel() == 0 || start >= end ||
      std::any_of(input_sizes.begin(), input_sizes.end(),
                  [](const int64_t d) { return d == 0; })) {
    return at::zeros(input_sizes, grad_output.options());
  }

  const Dimensions input_sizes_vec = CopyIntVector(input_sizes);

  TT_KERNEL(
      OpName::kSliceBackward, param_keys,
      (grad_output, input_sizes_vec, wrapped_dim, start, end, step), {
        TT_ASSIGN_OR_THROW(
            const mlir::ElementType element_type,
            ConvertTo<mlir::ElementType>(grad_output.scalar_type()));

        auto op_builder =
            [input_sizes_vec, wrapped_dim, start, end, step,
             element_type](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildSliceBackwardShlo(input, input_sizes_vec, wrapped_dim,
                                        start, end, step, element_type);
        };

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<1>(std::move(op_builder), grad_output,
                           {.out_dtype = element_type,
                            .out_dims = input_sizes_vec,
                            .op_param_cache_keys = std::move(param_keys)})));
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(input_sizes_vec, grad_output.scalar_type(),
                            grad_output.device()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
