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

#include "csrc/common/layout_utils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Contiguity.h"
#include "c10/core/MemoryFormat.h"
#include "c10/core/ScalarType.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/SmallVector.h"
#include "c10/util/strides.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

int64_t FindFp4PackedDim(const at::Tensor& tensor) {
  const int64_t rank = tensor.dim();
  if (rank == 0) {
    return -1;
  }
  if (tensor.stride(rank - 1) != 1) {
    for (int64_t d = 0; d < rank - 1; ++d) {
      if (tensor.stride(d) == 1 && tensor.size(d) > 1) {
        return d;
      }
    }
    return -1;
  }
  if (tensor.size(rank - 1) == 1) {
    return rank - 1;
  }
  for (int64_t d = 0; d < rank - 1; ++d) {
    if (tensor.size(d) == 1 && tensor.stride(d) == 1) {
      return d;
    }
  }
  return rank - 1;
}

// Applies any overrides from `shape.layout()` onto `layout`.
void ApplyCustomLayout(const Shape& shape, xla::Layout& layout) {
  if (!shape.layout().has_value()) {
    return;
  }
  const CustomLayout& custom_layout = *shape.layout();
  if (!custom_layout.minor_to_major.empty()) {
    *layout.mutable_minor_to_major() =
        xla::DimensionVector(custom_layout.minor_to_major.begin(),
                             custom_layout.minor_to_major.end());
  }
  // Dynamic on-device buffers have physical tiling determined by their upper
  // bounds rather than their runtime logical dimensions, so static custom
  // tiles do not apply.
  if (!shape.on_device_shape_is_dynamic() && !custom_layout.tiles.empty()) {
    layout.clear_tiles();
    for (const Indices& tile_dims : custom_layout.tiles) {
      layout.mutable_tiles()->emplace_back(tile_dims);
    }
  }
  if (custom_layout.element_size_in_bits != 0 &&
      custom_layout.element_size_in_bits !=
          XlaEquivalentBitwidth(shape.dtype())) {
    layout.set_element_size_in_bits(custom_layout.element_size_in_bits);
  }
}

}  // namespace

absl::StatusOr<TpuLayout> ResolveTpuLayout(const at::Tensor& tensor) {
  const auto scalar_type = tensor.scalar_type();
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  Dimensions sizes = CopyIntVector(tensor.sizes());
  Strides strides = CopyIntVector(tensor.strides());
  int64_t offset = tensor.storage_offset();

  if (scalar_type == at::kFloat4_e2m1fn_x2) {
    // FP4 values are packed 2-per-byte in PyTorch storage (represented by the
    // float4_e2m1fn_x2 dtype). For StableHLO, we need to expose the layout
    // in terms of individual unpacked FP4 elements.
    //
    // 1. Double the size of the packed (stride-1) dimension because each
    //    packed element contains 2 FP4 values.
    // 2. Double the strides of all other dimensions to maintain correct
    //    element offsets now that the packed dimension size has doubled.
    // 3. Double the storage offset to convert it from packed element offset
    //    to unpacked element offset.
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch core rejects 0D FP4 tensors
                   // generally on all devices.
        !sizes.empty(), error::kInvalidArgument)
        << "expected float4_e2m1fn_x2 tensors to be at least 1D, "
           "got 0D";
    const int64_t packed_dim = FindFp4PackedDim(tensor);
    TT_RET_CHECK(packed_dim >= 0, error::kInvalidArgument)
        << "expected packed dimension to be contiguous (stride 1) for "
           "float4_e2m1fn_x2 view, got strides "
        << ToString(strides);
    sizes[packed_dim] *= 2;
    for (size_t i = 0; i < strides.size(); ++i) {
      if (static_cast<int64_t>(i) != packed_dim) {
        strides[i] *= 2;
      }
    }
    offset *= 2;
  }
  return TpuLayout{std::move(sizes), std::move(strides), offset, dtype};
}

absl::StatusOr<std::vector<xla::Layout>> ResolveXlaLayouts(
    absl::Span<const Shape> shapes) {
  if (shapes.empty()) {
    return std::vector<xla::Layout>{};
  }
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client,
                      PjrtBackend::GetInstance().GetClient());
  std::vector<xla::Layout> layouts;
  layouts.reserve(shapes.size());
  for (const Shape& shape : shapes) {
    TT_ASSIGN_OR_RETURN(
        xla::Layout layout,
        AdaptXlaError(client->GetDefaultLayout(
                          ConvertTo<xla::PrimitiveType>(shape.dtype()),
                          shape.dimensions()),
                      /*context=*/"failed to get default layout"));
    ApplyCustomLayout(shape, layout);
    layouts.push_back(std::move(layout));
  }
  return layouts;
}

absl::StatusOr<std::optional<CustomLayout>>
ExtractCustomLayoutIfDifferentFromDefault(const Shape& shape) {
  if (!shape.layout().has_value()) {
    return std::nullopt;
  }
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client,
                      PjrtBackend::GetInstance().GetClient());
  TT_ASSIGN_OR_RETURN(
      const xla::Layout default_layout,
      AdaptXlaError(
          client->GetDefaultLayout(ConvertTo<xla::PrimitiveType>(shape.dtype()),
                                   shape.dimensions()),
          /*context=*/"failed to get default layout"));
  xla::Layout resolved_layout = default_layout;
  ApplyCustomLayout(shape, resolved_layout);
  if (resolved_layout == default_layout) {
    return std::nullopt;
  }
  ABSL_VLOG(3) << "[ExtractCustomLayoutIfDifferentFromDefault] Layout "
               << resolved_layout.ToString()
               << " differs from PjRT default layout "
               << default_layout.ToString() << " for shape ("
               << ToString(shape.dimensions()) << ")";
  CustomLayout result = *shape.layout();
  if (shape.on_device_shape_is_dynamic()) {
    // Dynamic on-device buffers (e.g. outputs of compiled models with bounded
    // dynamism) have physical tiling determined by their upper bounds rather
    // than their runtime logical dimensions. Propagate minor_to_major but omit
    // tiling so downstream eager kernels do not force static tile padding onto
    // the dynamic buffer.
    result.tiles.clear();
  }
  return result;
}

namespace {

struct BinaryOpOperand {
  Dimensions sizes;
  Strides strides;
  int64_t elem_size = 0;
  bool is_contiguous = false;
  bool is_channels_last = false;
  bool is_non_overlapping_and_dense = false;
};

BinaryOpOperand MakeBinaryOpOperand(const at::TensorBase& tensor, bool promoted,
                                    int64_t common_elem_size) {
  BinaryOpOperand op;
  op.sizes = CopyIntVector(tensor.sizes());
  if (promoted) {
    op.strides = EmptyLikeStrides(tensor);
    op.elem_size = common_elem_size;
    const at::IntArrayRef strides_ref(op.strides.data(), op.strides.size());
    op.is_contiguous = c10::_compute_contiguous<int64_t>(
        tensor.sizes(), strides_ref, tensor.numel());
    op.is_channels_last = c10::_compute_channels_last_contiguous_2d<int64_t>(
        tensor.sizes(), strides_ref);
    op.is_non_overlapping_and_dense = true;
  } else {
    op.strides = CopyIntVector(tensor.strides());
    op.elem_size = static_cast<int64_t>(tensor.element_size());
    op.is_contiguous = tensor.is_contiguous();
    op.is_channels_last = tensor.is_contiguous(at::MemoryFormat::ChannelsLast);
    op.is_non_overlapping_and_dense = tensor.is_non_overlapping_and_dense();
  }
  return op;
}

std::optional<Strides> TrySameShapeBinaryOpStrides(
    const BinaryOpOperand& self, const BinaryOpOperand& other,
    at::IntArrayRef out_sizes_ref) {
  const at::IntArrayRef self_sizes_ref(self.sizes.data(), self.sizes.size());
  const at::IntArrayRef other_sizes_ref(other.sizes.data(), other.sizes.size());
  if (self_sizes_ref != out_sizes_ref || other_sizes_ref != out_sizes_ref) {
    return std::nullopt;
  }
  if (self.is_contiguous && other.is_contiguous) {
    return CopyIntVector(c10::contiguous_strides(out_sizes_ref));
  }
  if (self.is_channels_last && other.is_channels_last) {
    return CopyIntVector(c10::get_channels_last_strides_2d(out_sizes_ref));
  }
  if (self.is_non_overlapping_and_dense && other.is_non_overlapping_and_dense &&
      self.strides == other.strides) {
    return self.strides;
  }
  return std::nullopt;
}

Strides ComputeOperandByteStrides(const BinaryOpOperand& op, size_t ndim,
                                  absl::Span<const int64_t> out_sizes) {
  Strides s(ndim, 0);
  const size_t offset = ndim - op.sizes.size();
  for (size_t i = 0; i < op.sizes.size(); ++i) {
    if (op.sizes[i] == 1 && out_sizes[offset + i] != 1) {
      s[offset + i] = 0;
    } else {
      s[offset + i] = op.strides[i] * op.elem_size;
    }
  }
  return s;
}

int CompareTensorIteratorDims(size_t dim0, size_t dim1,
                              const std::array<Strides, 2>& op_strides,
                              absl::Span<const int64_t> out_sizes) {
  for (const Strides& strides : op_strides) {
    const int64_t stride0 = strides[dim0];
    const int64_t stride1 = strides[dim1];
    if (stride0 == 0 || stride1 == 0) {
      continue;
    }
    if (stride0 < stride1) {
      return -1;
    }
    if (stride0 > stride1) {
      return 1;
    }
    if (out_sizes[dim0] > out_sizes[dim1]) {
      return 1;
    }
  }
  return 0;
}

c10::SmallVector<size_t, 5> ComputeTensorIteratorDimPerm(
    size_t ndim, const std::array<Strides, 2>& op_strides,
    absl::Span<const int64_t> out_sizes) {
  c10::SmallVector<size_t, 5> perm(ndim);
  std::iota(perm.rbegin(), perm.rend(), 0);
  for (size_t i = 1; i < ndim; ++i) {
    size_t dim1 = i;
    for (size_t j = 1; j <= i; ++j) {
      const size_t dim0 = i - j;
      const int comparison = CompareTensorIteratorDims(perm[dim0], perm[dim1],
                                                       op_strides, out_sizes);
      if (comparison > 0) {
        std::swap(perm[dim0], perm[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }
  return perm;
}

Strides StridesFromDimPerm(const c10::SmallVector<size_t, 5>& perm,
                           absl::Span<const int64_t> out_sizes,
                           at::IntArrayRef out_sizes_ref) {
  const size_t ndim = perm.size();
  bool inverted = true;
  for (size_t j = 0; j < ndim; ++j) {
    if (perm[j] != ndim - j - 1) {
      inverted = false;
      break;
    }
  }
  if (inverted) {
    return CopyIntVector(c10::contiguous_strides(out_sizes_ref));
  }

  Strides out_strides(ndim);
  int64_t next_stride = 1;
  for (size_t i = 0; i < ndim; ++i) {
    const size_t idx = perm[i];
    out_strides[idx] = next_stride;
    if (out_sizes[idx] > 1) {
      next_stride *= out_sizes[idx];
    }
  }
  return out_strides;
}

// Fast paths of `BinaryOpStrides` that mirror the dense-operand shortcuts in
// `at::TensorIterator`: empty outputs are contiguous, a scalar operand adopts
// the other operand's layout, and same-shape operands share a layout.
// Returns `std::nullopt` when the general stride-permutation algorithm is
// needed.
std::optional<Strides> TryBinaryOpStridesFastPath(
    const BinaryOpOperand& self_op, const BinaryOpOperand& other_op,
    const at::TensorBase& self, const at::TensorBase& other,
    absl::Span<const int64_t> out_sizes, at::IntArrayRef out_sizes_ref) {
  if (absl::c_contains(out_sizes, 0)) {
    return CopyIntVector(c10::contiguous_strides(out_sizes_ref));
  }
  if (other_op.sizes.empty() &&
      at::IntArrayRef(self_op.sizes.data(), self_op.sizes.size()) ==
          out_sizes_ref) {
    return EmptyLikeStrides(self);
  }
  if (self_op.sizes.empty() &&
      at::IntArrayRef(other_op.sizes.data(), other_op.sizes.size()) ==
          out_sizes_ref) {
    return EmptyLikeStrides(other);
  }
  if (const auto same_shape_strides =
          TrySameShapeBinaryOpStrides(self_op, other_op, out_sizes_ref)) {
    return same_shape_strides;
  }
  return std::nullopt;
}

}  // namespace

Strides EmptyLikeStrides(const at::TensorBase& tensor) {
  if (tensor.is_non_overlapping_and_dense()) {
    return CopyIntVector(tensor.strides());
  }
  return CopyIntVector(
      at::infer_dense_strides(tensor.sizes(), tensor.strides()));
}

Strides UnaryOpStrides(const at::TensorBase& tensor) {
  if (absl::c_contains(tensor.sizes(), 0)) {
    return CopyIntVector(c10::contiguous_strides(tensor.sizes()));
  }
  return EmptyLikeStrides(tensor);
}

Strides BinaryOpStrides(const at::TensorBase& self, const at::TensorBase& other,
                        absl::Span<const int64_t> out_sizes,
                        std::optional<c10::ScalarType> common_dtype) {
  const size_t ndim = out_sizes.size();
  if (ndim == 0) {
    return {};
  }
  if (ndim == 1) {
    return {1};
  }
  const at::IntArrayRef out_sizes_ref(out_sizes.data(), out_sizes.size());

  const c10::ScalarType effective_common_dtype = common_dtype.value_or(
      c10::promoteTypes(self.scalar_type(), other.scalar_type()));
  const bool self_promoted = (self.scalar_type() != effective_common_dtype);
  const bool other_promoted = (other.scalar_type() != effective_common_dtype);
  const int64_t common_elem_size =
      static_cast<int64_t>(c10::elementSize(effective_common_dtype));

  const BinaryOpOperand self_op =
      MakeBinaryOpOperand(self, self_promoted, common_elem_size);
  const BinaryOpOperand other_op =
      MakeBinaryOpOperand(other, other_promoted, common_elem_size);

  if (const auto fast_path = TryBinaryOpStridesFastPath(
          self_op, other_op, self, other, out_sizes, out_sizes_ref)) {
    return *fast_path;
  }

  const std::array<Strides, 2> op_strides = {
      ComputeOperandByteStrides(self_op, ndim, out_sizes),
      ComputeOperandByteStrides(other_op, ndim, out_sizes)};
  const auto perm = ComputeTensorIteratorDimPerm(ndim, op_strides, out_sizes);
  return StridesFromDimPerm(perm, out_sizes, out_sizes_ref);
}

Strides SuggestMemoryFormatStrides(const at::TensorBase& tensor) {
  switch (tensor.suggest_memory_format()) {
    case at::MemoryFormat::ChannelsLast:
      return CopyIntVector(c10::get_channels_last_strides_2d(tensor.sizes()));
    case at::MemoryFormat::ChannelsLast3d:
      return CopyIntVector(c10::get_channels_last_strides_3d(tensor.sizes()));
    default:
      return CopyIntVector(c10::contiguous_strides(tensor.sizes()));
  }
}

}  // namespace torch_tpu
