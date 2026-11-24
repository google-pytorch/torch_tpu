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

#include "csrc/ops/flip/flip_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for at::IntArrayRef
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
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<Dimensions> CanonicalizeFlipDims(
    const int64_t rank, const at::IntArrayRef flip_dims) {
  Dimensions canonical_dims;
  canonical_dims.reserve(flip_dims.size());
  absl::flat_hash_set<int64_t> seen_dims;
  seen_dims.reserve(flip_dims.size());
  for (const int64_t dim : flip_dims) {
    TT_ASSIGN_OR_RETURN(const int64_t wrapped_dim, SafeWrapDim(dim, rank));
    TT_RET_CHECK(seen_dims.insert(wrapped_dim).second, error::kInvalidArgument)
        << "dim " << wrapped_dim
        << " appears multiple times in the list of dims";
    // For 0D scalars, [0]/[-1] are validated and deduplicated above; keep
    // canonical_dims empty so BuildFlipShlo treats 0D flip as a no-op.
    if (rank > 0) {
      canonical_dims.push_back(wrapped_dim);
    }
  }
  return canonical_dims;
}

absl::StatusOr<mlir::MlirOp> BuildFlipShlo(
    mlir::MlirOp input_op, const absl::Span<const int64_t> flip_dims) {
  if (flip_dims.empty()) {
    return input_op;
  }
  return mlir::stablehlo::Reverse(input_op, AsArrayRef(flip_dims));
}

}  // namespace

at::Tensor AtenFlip(const at::Tensor& self, at::IntArrayRef flip_dims) {
  TT_KERNEL(OpName::kFlip, param_keys, (self, flip_dims), {
    TT_ASSIGN_OR_THROW(Dimensions flip_dims_vec,
                       CanonicalizeFlipDims(self.dim(), flip_dims));
    TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    auto op_builder = [flip_dims_vec = std::move(flip_dims_vec)](
                          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildFlipShlo(input, flip_dims_vec);
    };

    TT_ASSIGN_OR_THROW(
        DeviceBufferRef out_buf,
        (DispatchOp<1>(std::move(op_builder), self,
                       {.out_dtype = element_type,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)})));
    return MakeTensor(std::move(out_buf));
  });
}

}  // namespace torch_tpu
