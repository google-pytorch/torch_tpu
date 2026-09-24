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

#include "csrc/ops/masked_scatter/masked_scatter_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <utility>

#include "ATen/ExpandUtils.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/masked_scatter/masked_scatter.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

// Undoes the expansion of `mask` along its trailing dimensions. ATen's
// composite `masked_scatter` expands the mask to the shape of the tensor it
// scatters into, hiding that one mask element covers a whole row;
// `BuildMaskedScatterShlo` needs the stride zero dimensions back at size one.
at::Tensor NarrowBroadcastTrailingDims(const at::Tensor& mask) {
  at::Tensor narrowed = mask;
  for (int64_t dim = mask.dim() - 1; dim >= 0; --dim) {
    if (mask.size(dim) <= 1) {
      continue;
    }
    if (mask.stride(dim) != 0) {
      break;
    }
    narrowed = narrowed.slice(dim, /*start=*/0, /*end=*/1);
  }
  return narrowed;
}

}  // namespace

at::Tensor& AtenMaskedScatter_(at::Tensor& self, const at::Tensor& mask,
                               const at::Tensor& source) {
  TT_KERNEL(OpName::kMaskedScatter_, _, (self, mask, source), {
    TT_ASSIGN_OR_THROW(const auto mask_dtype,
                       ConvertTo<mlir::ElementType>(mask.scalar_type()));
    TT_CHECK_THROW(mask.scalar_type() == at::kBool, error::kInvalidArgument)
        << "expected Boolean tensor for mask, got " << ToString(mask_dtype);

    TT_ASSIGN_OR_THROW(const auto self_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const auto source_dtype,
                       ConvertTo<mlir::ElementType>(source.scalar_type()));
    TT_CHECK_THROW(self_dtype == source_dtype, error::kInvalidArgument)
        << "expected same dtype for self and source,"
        << " got self dtype " << ToString(self_dtype) << " and source dtype "
        << ToString(source_dtype);

    // `masked_scatter_` writes into `self` in place, so the mask has to fit
    // `self` rather than the two being broadcast against each other.
    TT_CHECK_THROW(at::is_expandable_to(mask.sizes(), self.sizes()),
                   error::kInvalidArgument)
        << "expected mask to be broadcastable to self shape, got "
        << mask.sizes() << " vs " << self.sizes();

    TT_CHECK_THROW(self.numel() <= std::numeric_limits<int32_t>::max(),
                   error::kPythonNotImplementedError)
        << "expected the input to contain at most "
        << std::numeric_limits<int32_t>::max()
        << " elements because masked_scatter_ uses int32 source offsets, got "
        << self.numel();

    TT_ASSIGN_OR_THROW(auto param_keys, *OpParamCacheKeysBuilder());
    Dimensions out_dims = CopyIntVector(self.sizes());

    auto shlo_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      // inputs[0]: self, inputs[1]: mask, inputs[2]: source
      return BuildMaskedScatterShlo(inputs[0], inputs[1], inputs[2]);
    };

    TT_THROW_IF_ERROR(DispatchOpOut<3>(
        std::move(shlo_builder),
        {self, NarrowBroadcastTrailingDims(mask), source}, self,
        {.out_dtype = self_dtype,
         .out_dims = out_dims,
         .op_param_cache_keys = std::move(param_keys)}));

    return self;
  });
}

}  // namespace torch_tpu
