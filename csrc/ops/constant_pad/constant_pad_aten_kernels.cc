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

#include "csrc/ops/constant_pad/constant_pad_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/full.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
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

// Applies constant padding (and/or negative-padding cropping) along a single
// dimension `dim` of `input`, updating `cur_shape[dim]` in place.
mlir::MlirOp PadSingleDimensionShlo(mlir::MlirOp input, mlir::MlirOp val_scalar,
                                    const mlir::RankedTensorType input_type,
                                    const int64_t dim, const int64_t left_pad,
                                    const int64_t right_pad,
                                    Dimensions& cur_shape) {
  const int64_t rank = cur_shape.size();
  const int64_t orig_dim_size = cur_shape[dim];
  mlir::MlirBuilder& builder = input.getBuilder();

  // Handle negative padding (cropping) along `dim` first via StableHLO Slice.
  mlir::MlirOp core = input;
  if (left_pad < 0 || right_pad < 0) {
    Dimensions start_indices(rank, 0);
    Dimensions limit_indices = cur_shape;
    const Dimensions strides(rank, 1);
    start_indices[dim] =
        std::min(orig_dim_size, std::max<int64_t>(0, -left_pad));
    limit_indices[dim] = std::max(
        start_indices[dim], orig_dim_size - std::max<int64_t>(0, -right_pad));
    core = mlir::stablehlo::Slice(core, start_indices, limit_indices, strides);
    cur_shape[dim] = limit_indices[dim] - start_indices[dim];
  }

  // If negative padding exceeded `orig_dim_size`, the excess crop reduces
  // the positive padding on the opposite side.
  const int64_t excess_left_crop =
      std::max<int64_t>(0, -left_pad - orig_dim_size);
  const int64_t excess_right_crop =
      std::max<int64_t>(0, -right_pad - orig_dim_size);
  const int64_t effective_left_pad =
      std::max<int64_t>(0, left_pad - excess_right_crop);
  const int64_t effective_right_pad =
      std::max<int64_t>(0, right_pad - excess_left_crop);

  // Assemble positive padding blocks around the cropped core tensor.
  std::vector<mlir::MlirOp> parts;
  parts.reserve(3);
  if (effective_left_pad > 0) {
    Dimensions left_shape = cur_shape;
    left_shape[dim] = effective_left_pad;
    const auto left_type =
        mlir::RankedTensorType::get(left_shape, input_type.getElementType());
    parts.push_back(mlir::stablehlo::BroadcastInDim(left_type, val_scalar, {}));
  }
  if (cur_shape[dim] > 0 || parts.empty()) {
    parts.push_back(core);
  }
  if (effective_right_pad > 0) {
    Dimensions right_shape = cur_shape;
    right_shape[dim] = effective_right_pad;
    const auto right_type =
        mlir::RankedTensorType::get(right_shape, input_type.getElementType());
    parts.push_back(
        mlir::stablehlo::BroadcastInDim(right_type, val_scalar, {}));
  }

  cur_shape[dim] += effective_left_pad + effective_right_pad;
  if (parts.size() == 1) {
    return parts[0];
  }
  return mlir::stablehlo::Concatenate(builder,
                                      mlir::ArrayRef<mlir::MlirOp>(parts), dim);
}

// Builds the StableHLO graph for N-dimensional constant padding.
//
// PyTorch's `pad` specification lists pairs of (left_pad, right_pad) starting
// from the innermost dimension (`rank - 1`) moving outward (`rank - 1 - i`).
// Rather than emitting a full output-sized constant broadcast followed by
// DynamicUpdateSlice (which forces an extra full-tensor allocation and prevents
// XLA fusion across trapezoidal discretization steps like `F.pad(x, (1, 0))`),
// this builder iterates over each padded dimension and constructs:
//   1. `stablehlo.slice` to crop the dimension if `left_pad < 0` or `right_pad
//   < 0`.
//   2. `stablehlo.broadcast_in_dim` for the left/right constant padding blocks
//      using the promoted scalar value `val_scalar`.
//   3. `stablehlo.concatenate` along `dim` to join `[left_block, core,
//   right_block]`.
absl::StatusOr<mlir::MlirOp> BuildConstantPadNdShlo(mlir::MlirOp input,
                                                    mlir::MlirOp val_scalar,
                                                    const Dimensions& pad_vec) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  Dimensions cur_shape = CopyIntVector(input_type.getShape());
  const int64_t rank = cur_shape.size();
  const int64_t num_pad_dims = pad_vec.size() / 2;

  mlir::MlirOp result = input;
  for (int64_t i = 0; i < num_pad_dims; ++i) {
    const int64_t dim = rank - 1 - i;
    const int64_t left_pad = pad_vec[2 * i];
    const int64_t right_pad = pad_vec[2 * i + 1];
    if (left_pad == 0 && right_pad == 0) {
      continue;
    }
    result = PadSingleDimensionShlo(result, val_scalar, input_type, dim,
                                    left_pad, right_pad, cur_shape);
  }
  return result;
}

}  // namespace

at::Tensor AtenConstantPadNd(const at::Tensor& self, at::IntArrayRef pad,
                             const at::Scalar& value) {
  TT_CHECK_THROW(pad.size() % 2 == 0, error::kInvalidArgument)
      << "length of pad must be even but instead it equals " << pad.size();
  const int64_t num_pad_dims = pad.size() / 2;
  TT_CHECK_THROW(num_pad_dims <= self.dim(), error::kInvalidArgument)
      << "pad length must be at most twice the input dimension";

  Dimensions out_dims = CopyIntVector(self.sizes());
  bool has_non_zero_pad = false;
  for (int64_t i = 0; i < num_pad_dims; ++i) {
    const int64_t dim = self.dim() - 1 - i;
    const int64_t left_pad = pad[2 * i];
    const int64_t right_pad = pad[2 * i + 1];
    if (left_pad != 0 || right_pad != 0) {
      has_non_zero_pad = true;
    }
    const int64_t cropped_size = self.size(dim) +
                                 std::min<int64_t>(0, left_pad) +
                                 std::min<int64_t>(0, right_pad);
    TT_CHECK_THROW(cropped_size >= 0, error::kInvalidArgument)
        << "length must be non-negative";
    out_dims[dim] += left_pad + right_pad;
  }

  // Fast-path no-op padding to a simple clone.
  if (!has_non_zero_pad) {
    return self.clone();
  }

  // Handle zero-volume output or zero-volume input edge cases directly.
  if (std::any_of(out_dims.begin(), out_dims.end(),
                  [](const int64_t d) { return d == 0; })) {
    return at::empty(out_dims, self.options());
  }

  if (self.numel() == 0) {
    return at::full(out_dims, value, self.options());
  }

  const Dimensions pad_vec = CopyIntVector(pad);
  // Promote the scalar padding value so that changes to `value` hit the same
  // compiled MLIR module cache key (`DispatchOp<2>`) without recompilation.
  auto promoted_value = PromoteScalar(value);

  TT_KERNEL(
      OpName::kConstantPadNd, param_keys, (self, pad_vec, promoted_value), {
        TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        TT_ASSIGN_OR_THROW(const at::Tensor val_tensor,
                           promoted_value.GetTensor(self.scalar_type()));

        auto op_builder = [pad_vec](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildConstantPadNdShlo(inputs[0], inputs[1], pad_vec);
        };

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<2>(std::move(op_builder), {self, val_tensor},
                           {.out_dtype = element_type,
                            .out_dims = out_dims,
                            .op_param_cache_keys = std::move(param_keys)})));
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(out_dims, self.scalar_type(), self.device()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
