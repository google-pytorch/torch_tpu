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

#include "csrc/ops/slice_scatter/slice_scatter_aten_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
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

// Builds the StableHLO graph for strided `aten::slice_scatter` (`step > 1`).
//
// Lowers strided updates via `Reshape` + `DynamicUpdateSlice` (with edge-only
// high padding and slicing when `total_len` is not divisible by `step`) rather
// than `stablehlo.pad` with `interior_padding = step - 1` + `stablehlo.select`.
// On TPU, XLA cannot fuse `pad` with non-zero interior padding on the
// minor-most dimension into consumer loop fusions, materializing standalone
// `pad` kernels (e.g. +26.6 ms/step in `wan_2_2_ti2v_5b` 3D RoPE backward).
//
// Given `start = q * step + r` (`0 <= r < step`) and
// `num_blocks = ceil(total_len / step)`:
//   1. If `padded_len = num_blocks * step > total_len`, edge-pad `self_op` at
//      the high end of `dim` by `padded_len - total_len` (`interior_pad = 0`).
//   2. Reshape `dim` of `self_op` into `(num_blocks, step)` and `dim` of
//      `src_op` into `(slice_len, 1)`.
//   3. Emit unit-stride `DynamicUpdateSlice` at `(q, r)` along the reshaped
//      `(num_blocks, step)` dimensions.
//   4. Reshape `(num_blocks, step)` back to `padded_len`, and slice
//   `[0:total_len]`
//      if high padding was added.
absl::StatusOr<mlir::MlirOp> BuildStridedSliceScatterShlo(
    mlir::MlirOp self_op, mlir::MlirOp src_op, const Dimensions& input_sizes,
    const int64_t dim, const int64_t start, const int64_t step,
    const int64_t slice_len, const mlir::ElementType element_type) {
  const int64_t total_len = input_sizes[dim];
  const int64_t q = start / step;
  const int64_t r = start % step;
  const int64_t num_blocks = (total_len + step - 1) / step;
  const int64_t padded_len = num_blocks * step;
  const int64_t high_pad_amt = padded_len - total_len;

  mlir::MlirBuilder& builder = self_op.getBuilder();
  mlir::MlirOp work_self = self_op;
  if (high_pad_amt > 0) {
    const Dimensions zero_pad(input_sizes.size(), 0);
    Dimensions high_pad(input_sizes.size(), 0);
    high_pad[dim] = high_pad_amt;
    mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0, element_type);
    work_self = mlir::stablehlo::Pad(self_op, zero_scalar, zero_pad, high_pad,
                                     zero_pad);
  }

  Dimensions reshaped_self_sizes;
  reshaped_self_sizes.reserve(input_sizes.size() + 1);
  Dimensions reshaped_src_sizes;
  reshaped_src_sizes.reserve(input_sizes.size() + 1);
  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(input_sizes.size() + 1);
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (static_cast<int64_t>(i) == dim) {
      reshaped_self_sizes.push_back(num_blocks);
      reshaped_self_sizes.push_back(step);
      reshaped_src_sizes.push_back(slice_len);
      reshaped_src_sizes.push_back(1);
      start_indices.push_back(
          MakeScalarConstant(builder, q, mlir::ElementType::I64));
      start_indices.push_back(
          MakeScalarConstant(builder, r, mlir::ElementType::I64));
    } else {
      reshaped_self_sizes.push_back(input_sizes[i]);
      reshaped_src_sizes.push_back(input_sizes[i]);
      start_indices.push_back(
          MakeScalarConstant(builder, 0, mlir::ElementType::I64));
    }
  }

  mlir::MlirOp reshaped_self =
      mlir::stablehlo::Reshape(work_self, reshaped_self_sizes);
  mlir::MlirOp reshaped_src =
      mlir::stablehlo::Reshape(src_op, reshaped_src_sizes);
  mlir::MlirOp updated_reshaped = mlir::stablehlo::DynamicUpdateSlice(
      reshaped_self, reshaped_src, start_indices);

  Dimensions padded_sizes = input_sizes;
  padded_sizes[dim] = padded_len;
  mlir::MlirOp updated =
      mlir::stablehlo::Reshape(updated_reshaped, padded_sizes);
  if (high_pad_amt > 0) {
    const Dimensions slice_starts(input_sizes.size(), 0);
    const Dimensions slice_strides(input_sizes.size(), 1);
    updated = mlir::stablehlo::Slice(updated, slice_starts, input_sizes,
                                     slice_strides);
  }
  return updated;
}

mlir::MlirOp BuildBoundaryConcatenateSliceScatterShlo(
    mlir::MlirOp self_op, mlir::MlirOp src_op, const Dimensions& input_sizes,
    const int64_t dim, const int64_t start, const int64_t end,
    const int64_t total_len) {
  mlir::MlirBuilder& builder = self_op.getBuilder();
  std::vector<mlir::MlirOp> parts;
  parts.reserve(2);
  if (start > 0) {
    // Anchored at end: keep self[..., 0:start, ...], then concatenate src.
    Dimensions start_indices(input_sizes.size(), 0);
    Dimensions limit_indices = input_sizes;
    limit_indices[dim] = start;
    Dimensions strides(input_sizes.size(), 1);
    mlir::MlirOp prefix =
        mlir::stablehlo::Slice(self_op, start_indices, limit_indices, strides);
    parts.push_back(prefix);
    parts.push_back(src_op);
  } else {
    // Anchored at start: concatenate src, then keep self[..., end:total_len,
    // ...].
    parts.push_back(src_op);
    if (end < total_len) {
      Dimensions start_indices(input_sizes.size(), 0);
      start_indices[dim] = end;
      Dimensions limit_indices = input_sizes;
      Dimensions strides(input_sizes.size(), 1);
      mlir::MlirOp suffix = mlir::stablehlo::Slice(self_op, start_indices,
                                                   limit_indices, strides);
      parts.push_back(suffix);
    }
  }
  return mlir::stablehlo::Concatenate(builder,
                                      mlir::ArrayRef<mlir::MlirOp>(parts), dim);
}

// Builds the StableHLO graph for `aten::slice_scatter`.
//
// Dispatches across specialized StableHLO patterns depending on slice
// geometry and stride:
//   0. Empty slice (`slice_len == 0`): Returns `self_op` directly.
//   1. Full replacement (`step == 1 && slice_len == total_len`): Returns
//      `src_op` directly without DynamicUpdateSlice.
//   2. Strided slices (`step > 1`): Reshapes `dim` into `(num_blocks, step)`
//      and emits unit-stride `stablehlo.dynamic_update_slice` (with high-edge
//      padding/slicing only when `total_len % step != 0`).
//   3. Boundary-anchored concatenate: For unit-stride slices (`step == 1`)
//      anchored at `start == 0` or `end == total_len` where
//      `slice_len * 2 >= total_len` (e.g. Causal LM logit slicing, KV-cache
//      prefix/suffix updates), emits `stablehlo.concatenate` joining `src_op`
//      with `stablehlo.slice(self_op, ...)` along `dim`. This eliminates
//      DynamicUpdateSlice index tensor creation and enables fusion in XLA.
//   4. Interior unit-stride slices (`step == 1`): Emits
//      `stablehlo.dynamic_update_slice`.
absl::StatusOr<mlir::MlirOp> BuildSliceScatterShlo(
    mlir::MlirOp self_op, mlir::MlirOp src_op, const Dimensions& input_sizes,
    const int64_t dim, const int64_t start, const int64_t end,
    const int64_t step, const mlir::ElementType element_type) {
  const int64_t total_len = input_sizes[dim];
  const int64_t slice_len = (end - start + step - 1) / step;
  if (slice_len == 0) {
    return self_op;
  }

  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType src_type = GetTensorTypeOrDie(src_op);
  if (src_type.getElementType() != self_type.getElementType()) {
    src_op =
        mlir::stablehlo::ConvertElementType(src_op, self_type.getElementType());
  }

  if (step == 1 && slice_len == total_len) {
    return src_op;
  }

  if (step > 1) {
    return BuildStridedSliceScatterShlo(self_op, src_op, input_sizes, dim,
                                        start, step, slice_len, element_type);
  }

  // Fast-path 2: Boundary-anchored concatenate for unit-stride slices
  // spanning at least half the dimension.
  if (step == 1 && (start == 0 || end == total_len) &&
      slice_len * 2 >= total_len) {
    return BuildBoundaryConcatenateSliceScatterShlo(
        self_op, src_op, input_sizes, dim, start, end, total_len);
  }

  // Fallback for interior slices (step == 1): Emit DynamicUpdateSlice.
  mlir::MlirBuilder& builder = self_op.getBuilder();
  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(input_sizes.size());
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    const int64_t start_idx = (static_cast<int64_t>(i) == dim) ? start : 0;
    start_indices.push_back(
        MakeScalarConstant(builder, start_idx, mlir::ElementType::I64));
  }
  return mlir::stablehlo::DynamicUpdateSlice(self_op, src_op, start_indices);
}

}  // namespace

at::Tensor AtenSliceScatter(const at::Tensor& self, const at::Tensor& src,
                            int64_t dim, std::optional<int64_t> start,
                            std::optional<int64_t> end, int64_t step) {
  TT_KERNEL(
      OpName::kSliceScatter, param_keys, (self, src, dim, start, end, step), {
        TT_CHECK_THROW(step > 0, error::kInvalidArgument)
            << "expected step to be greater than 0, got " << step;
        const int64_t ndim = self.dim();
        TT_CHECK_THROW(ndim > 0, error::kPythonIndexError)
            << "expected tensor to be at least 1-dimensional, got "
               "0-dimensional";
        TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, ndim));
        const int64_t total_len = self.size(wrapped_dim);

        int64_t start_val = start.value_or(0);
        int64_t end_val = end.value_or(total_len);

        if (start_val < 0) {
          start_val += total_len;
        }
        if (end_val < 0) {
          end_val += total_len;
        }
        start_val = std::clamp<int64_t>(start_val, 0, total_len);
        end_val = std::clamp<int64_t>(end_val, start_val, total_len);

        const int64_t slice_len = (end_val - start_val + step - 1) / step;
        const Dimensions input_sizes_vec = CopyIntVector(self.sizes());

        Dimensions expected_slice_shape = input_sizes_vec;
        expected_slice_shape[wrapped_dim] = slice_len;
        TT_CHECK_THROW(src.sizes() == at::IntArrayRef(expected_slice_shape),
                       error::kInvalidArgument)
            << "expected src to have a size equal to the slice of self, got "
            << src.sizes() << " vs " << at::IntArrayRef(expected_slice_shape);
        TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        if (self.numel() == 0) {
          TT_ASSIGN_OR_THROW(auto zero_buf, CreateZeroSizeDeviceBufferRef(
                                                input_sizes_vec, element_type));
          return MakeTensor(std::move(zero_buf));
        }

        if (slice_len == 0) {
          auto noop_builder =
              [](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
            return input;
          };
          TT_ASSIGN_OR_THROW(
              auto out_buf,
              (DispatchOp<1>(std::move(noop_builder), self,
                             {.out_dtype = element_type,
                              .out_dims = input_sizes_vec,
                              .op_param_cache_keys = std::move(param_keys)})));
          return MakeTensor(std::move(out_buf));
        }

        auto op_builder = [input_sizes_vec, wrapped_dim, start_val, end_val,
                           step,
                           element_type](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildSliceScatterShlo(inputs[0], inputs[1], input_sizes_vec,
                                       wrapped_dim, start_val, end_val, step,
                                       element_type);
        };

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<2>(std::move(op_builder), {self, src},
                           {.out_dtype = element_type,
                            .out_dims = input_sizes_vec,
                            .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(std::move(out_buf));
      });
}

at::Tensor& AtenSliceScatterOut(const at::Tensor& self, const at::Tensor& src,
                                int64_t dim, std::optional<int64_t> start,
                                std::optional<int64_t> end, int64_t step,
                                at::Tensor& out) {
  TT_KERNEL(
      OpName::kSliceScatterOut, param_keys,
      (self, src, dim, start, end, step, out), {
        TT_CHECK_THROW(out.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "expected out dtype " << ToString(self.scalar_type()) << ", got "
            << ToString(out.scalar_type());
        TT_CHECK_THROW(step > 0, error::kInvalidArgument)
            << "expected step to be greater than 0, got " << step;
        const int64_t ndim = self.dim();
        TT_CHECK_THROW(ndim > 0, error::kPythonIndexError)
            << "expected tensor to be at least 1-dimensional, got "
               "0-dimensional";
        TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, ndim));
        const int64_t total_len = self.size(wrapped_dim);

        int64_t start_val = start.value_or(0);
        int64_t end_val = end.value_or(total_len);

        if (start_val < 0) {
          start_val += total_len;
        }
        if (end_val < 0) {
          end_val += total_len;
        }
        start_val = std::clamp<int64_t>(start_val, 0, total_len);
        end_val = std::clamp<int64_t>(end_val, start_val, total_len);

        const int64_t slice_len = (end_val - start_val + step - 1) / step;
        const Dimensions input_sizes_vec = CopyIntVector(self.sizes());

        Dimensions expected_slice_shape = input_sizes_vec;
        expected_slice_shape[wrapped_dim] = slice_len;
        TT_CHECK_THROW(src.sizes() == at::IntArrayRef(expected_slice_shape),
                       error::kInvalidArgument)
            << "expected src to have a size equal to the slice of self, got "
            << src.sizes() << " vs " << at::IntArrayRef(expected_slice_shape);
        TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        if (self.numel() == 0) {
          TT_ASSIGN_OR_THROW(auto zero_buf, CreateZeroSizeDeviceBufferRef(
                                                input_sizes_vec, element_type));
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, input_sizes_vec));
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(zero_buf), out));
          return out;
        }

        if (slice_len == 0) {
          auto noop_builder =
              [](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
            return input;
          };
          TT_THROW_IF_ERROR((DispatchOpOut<1>(
              std::move(noop_builder), self, out,
              {.out_dtype = element_type,
               .out_dims = input_sizes_vec,
               .op_param_cache_keys = std::move(param_keys)})));
          return out;
        }

        auto op_builder = [input_sizes_vec, wrapped_dim, start_val, end_val,
                           step,
                           element_type](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildSliceScatterShlo(inputs[0], inputs[1], input_sizes_vec,
                                       wrapped_dim, start_val, end_val, step,
                                       element_type);
        };

        TT_THROW_IF_ERROR(
            (DispatchOpOut<2>(std::move(op_builder), {self, src}, out,
                              {.out_dtype = element_type,
                               .out_dims = input_sizes_vec,
                               .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

}  // namespace torch_tpu
