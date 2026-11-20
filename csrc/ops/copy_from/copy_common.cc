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

#include "csrc/ops/copy_from/copy_common.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/macro_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/ops/as_strided/as_strided_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::Status RestrideTensor(at::Tensor& tensor, at::IntArrayRef sizes,
                            at::IntArrayRef strides, int64_t storage_offset) {
  if (tensor.device().is_cpu()) {
    tensor = tensor.as_strided(sizes, strides, storage_offset);
    return absl::OkStatus();
  }
  TT_ASSIGN_OR_RETURN(tensor,
                      AsStrided(tensor, sizes, strides, storage_offset));
  return absl::OkStatus();
}

// Using metadata-only views (as_strided), broadcasts src to the shape of dest.
// Returns an error if this is not possible.
absl::Status BroadcastToDest(at::Tensor& src, const at::Tensor& dest) {
  if (src.sizes() == dest.sizes()) {
    // No broadcasting is needed.
    return absl::OkStatus();
  }
  // Broadcast src to match the sizes of dest.
  auto [new_src_sizes, new_src_strides] =
      at::inferExpandGeometry(src.sizes(), src.strides(), dest.sizes());
  return RestrideTensor(src, new_src_sizes, new_src_strides,
                        src.storage_offset());
}

// Using metadata-only views (as_strided), transposes dest so that it is
// row-major, and applies the same transpose to src.
// If any transformation is required, this also strips size-1 dimensions from
// dest.
absl::Status TransposeToRowMajorDest(at::Tensor& src, at::Tensor& dest) {
  ABSL_CHECK_EQ(  // CRASH_OK=precondition established by BroadcastToDest
      src.dim(), dest.dim())
      << "cannot create a matching transpose for tensors of different "
         "dimensions";
  if (dest.is_contiguous()) {
    return absl::OkStatus();
  }

  // Determine a transpose that can be applied to both src and dest to make
  // dest contiguous (even if it makes src non-contiguous).
  struct TransposeDim {
    int64_t size;
    int64_t src_stride;
    int64_t dest_stride;
  };

  std::vector<TransposeDim> transpose_dims;
  transpose_dims.reserve(dest.dim());
  for (int64_t i = 0; i < dest.dim(); ++i) {
    if (dest.size(i) == 1) {
      continue;
    }
    transpose_dims.push_back({dest.size(i), src.stride(i), dest.stride(i)});
  }
  // Sort transpose_dims by dest.stride descending.
  std::sort(transpose_dims.begin(), transpose_dims.end(),
            [](const TransposeDim& a, const TransposeDim& b) {
              return a.dest_stride > b.dest_stride;
            });
  Dimensions new_sizes;
  Strides new_src_strides;
  Strides new_dest_strides;
  new_sizes.reserve(dest.dim());
  new_src_strides.reserve(dest.dim());
  new_dest_strides.reserve(dest.dim());
  for (const auto& dim : transpose_dims) {
    new_sizes.push_back(dim.size);
    new_src_strides.push_back(dim.src_stride);
    new_dest_strides.push_back(dim.dest_stride);
  }
  TT_RETURN_IF_ERROR(
      RestrideTensor(src, new_sizes, new_src_strides, src.storage_offset()));
  TT_RETURN_IF_ERROR(
      RestrideTensor(dest, new_sizes, new_dest_strides, dest.storage_offset()));
  return absl::OkStatus();
}

[[nodiscard]] bool IsUncastableShellType(at::ScalarType t) {
  switch (t) {
#if TT_HAS_FLOAT8_E5M3FNU
    // TODO(b/579965801): Remove guard once all 2.16 pins include PR #201269.
    case at::ScalarType::Float8_e5m3fnu:
#endif
    // go/keep-sorted start
    case at::ScalarType::Bits16:
    case at::ScalarType::Bits1x8:
    case at::ScalarType::Bits2x4:
    case at::ScalarType::Bits4x2:
    case at::ScalarType::Bits8:
    case at::ScalarType::Int1:
    case at::ScalarType::Int2:
    case at::ScalarType::Int3:
    case at::ScalarType::Int4:
    case at::ScalarType::Int5:
    case at::ScalarType::Int6:
    case at::ScalarType::Int7:
    case at::ScalarType::UInt1:
    case at::ScalarType::UInt2:
    case at::ScalarType::UInt3:
    case at::ScalarType::UInt4:
    case at::ScalarType::UInt5:
    case at::ScalarType::UInt6:
    case at::ScalarType::UInt7:
      // go/keep-sorted end
      return true;
#if TT_TORCH_VERSION_GE(2, 14)
    case at::ScalarType::BComplex32:
#endif
    // go/keep-sorted start
    case at::ScalarType::BFloat16:
    case at::ScalarType::Bool:
    case at::ScalarType::Byte:
    case at::ScalarType::Char:
    case at::ScalarType::ComplexDouble:
    case at::ScalarType::ComplexFloat:
    case at::ScalarType::ComplexHalf:
    case at::ScalarType::Double:
    case at::ScalarType::Float4_e2m1fn_x2:
    case at::ScalarType::Float8_e4m3fn:
    case at::ScalarType::Float8_e4m3fnuz:
    case at::ScalarType::Float8_e5m2:
    case at::ScalarType::Float8_e5m2fnuz:
    case at::ScalarType::Float8_e8m0fnu:
    case at::ScalarType::Float:
    case at::ScalarType::Half:
    case at::ScalarType::Int:
    case at::ScalarType::Long:
    case at::ScalarType::NumOptions:
    case at::ScalarType::QInt32:
    case at::ScalarType::QInt8:
    case at::ScalarType::QUInt2x4:
    case at::ScalarType::QUInt4x2:
    case at::ScalarType::QUInt8:
    case at::ScalarType::Short:
    case at::ScalarType::UInt16:
    case at::ScalarType::UInt32:
    case at::ScalarType::UInt64:
    case at::ScalarType::Undefined:
      // go/keep-sorted end
      return false;
      // Deliberately omitting the default case. We want the compiler to force
      // us to handle all cases explicitly.
  }
}

}  // namespace

absl::Status ValidateShellDtypeCast(at::ScalarType src, at::ScalarType dst) {
  if (src == dst ||
      (!IsUncastableShellType(src) && !IsUncastableShellType(dst))) {
    return absl::OkStatus();
  }
  const at::ScalarType shell = IsUncastableShellType(src) ? src : dst;
  return TT_ERROR(error::kPythonNotImplementedError)
         << "casting from " << ToString(src) << " to " << ToString(dst)
         << " is not supported: " << ToString(shell)
         << " is a shell dtype with no defined conversions. Use .view() to "
            "reinterpret the bytes as a same-size integer dtype instead";
}

absl::Status BroadcastAndTranspose(at::Tensor& src, at::Tensor& dest) {
  TT_RETURN_IF_ERROR(BroadcastToDest(src, dest));
  TT_RETURN_IF_ERROR(TransposeToRowMajorDest(src, dest));
  return absl::OkStatus();
}

absl::Status ValidateBroadcastableToDest(const at::Tensor& src,
                                         const at::Tensor& dest) {
  // dest goes first so that the error message names the tensors in the same
  // order as CPU's copy_().
  TT_ASSIGN_OR_RETURN(const Dimensions broadcast_sizes,
                      InferSize(dest.sizes(), src.sizes()));
  TT_RET_CHECK(at::IntArrayRef(broadcast_sizes) == dest.sizes(),
               error::kInvalidArgument)
      << "output with shape " << ToString(dest.sizes())
      << " doesn't match the broadcast shape " << ToString(broadcast_sizes);
  return absl::OkStatus();
}

TransferSyncMode GetTransferSyncMode(bool non_blocking) {
  const EagerMode eager_mode = GetEagerMode();
  if (eager_mode == EagerMode::kInternalCompileFxGraph) {
    return TransferSyncMode::kIsolated;
  }
  if (eager_mode == EagerMode::kDeferNeverAndLaunchBlocking) {
    return TransferSyncMode::kEagerBlocking;
  }
  return non_blocking ? TransferSyncMode::kEagerNonBlocking
                      : TransferSyncMode::kEagerBlocking;
}

}  // namespace torch_tpu
