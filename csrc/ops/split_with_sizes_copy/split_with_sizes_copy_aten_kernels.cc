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

#include "csrc/ops/split_with_sizes_copy/split_with_sizes_copy_aten_kernels.h"

#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

void AtenSplitWithSizesCopyOut(const at::Tensor& self,
                               at::IntArrayRef split_sizes, int64_t dim,
                               at::TensorList out) {
  TT_KERNEL(
      OpName::kSplitWithSizesCopyOut, _,
      (self, IgnoreInCacheKey(split_sizes, "Doesn't affect SHLO"),
       IgnoreInCacheKey(dim, "Doesn't affect SHLO"), out),
      {
        TT_ASSIGN_OR_THROW(int64_t normalized_dim,
                           SafeWrapDim(dim, self.dim()));
        for (int64_t i = 0; i < out.size(); ++i) {
          TT_THROW_IF_ERROR(ValidateOutDtype(out[i], self.scalar_type(),
                                             /*allow_out_dtype_cast=*/false));
          int64_t size = split_sizes[i];
          Dimensions expected_shape = CopyIntVector(self.sizes());
          expected_shape[normalized_dim] = size;
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out[i], expected_shape));
        }
        // split_with_sizes_copy is an undocumented op used by PyTorch's
        // fully_shard() wrapper (FSDP2). The default decomposition, found
        // in
        // https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py,
        // performs one copy per split. For FSDP2, the input is a 2-d
        // coalesced tensor of shape (fsdp_world_size, -1). The output
        // slices are reshaped into the unsharded parameters. The
        // resulting multi-dimensional reshapes are slow to compile and
        // execute. This implementation does a more performant sequence of
        // slices and concatenations in the 2-d case. The performance
        // benefits might generalize to other dimensionalities, but for
        // simplicity we choose for now to only target the known use case.
        // Each output is one op on the flattened input, so that the host
        // work doesn't grow with fsdp_world_size.
        if (normalized_dim == 1 && self.dim() == 2) {
          const int64_t num_rows = self.size(0);
          const int64_t num_cols = self.size(1);
          TT_CHECK_THROW(std::accumulate(split_sizes.begin(), split_sizes.end(),
                                         int64_t{0}) == num_cols,
                         error::kInvalidArgument)
              << "expected split_sizes to sum to " << num_cols
              << " (the input's size at dimension 1), got "
              << ToString(split_sizes);
          // With no rows, every output is empty and there is nothing to copy.
          if (num_rows == 0) {
            return;
          }
          TT_ASSIGN_OR_THROW(const mlir::ElementType dtype,
                             ConvertTo<mlir::ElementType>(self.scalar_type()));
          const at::Tensor flat_self = self.view(-1);
          int64_t start = 0;
          for (int64_t i = 0; i < out.size(); ++i) {
            const int64_t size = split_sizes[i];
            if (size > 0) {
              auto op_builder = [num_rows, num_cols, start,
                                 size](mlir::MlirOp flat) {
                const Indices stride = {1};
                std::vector<mlir::MlirOp> rows;
                rows.reserve(num_rows);
                for (int64_t row = 0; row < num_rows; ++row) {
                  const Indices begin = {row * num_cols + start};
                  const Indices limit = {row * num_cols + start + size};
                  rows.push_back(
                      mlir::stablehlo::Slice(flat, begin, limit, stride));
                }
                return mlir::stablehlo::Concatenate(flat.getBuilder(), rows, 0);
              };
              OpParamCacheKeys param_keys = OpParamCacheKeys::Empty();
              TT_THROW_IF_ERROR(param_keys.SetParam("num_rows", num_rows));
              TT_THROW_IF_ERROR(param_keys.SetParam("start", start));
              const int64_t out_dims[] = {num_rows * size};
              TT_THROW_IF_ERROR(DispatchOpOut<1>(
                  std::move(op_builder), flat_self, out[i].view(-1),
                  {.out_dtype = dtype,
                   .out_dims = out_dims,
                   .op_param_cache_keys = std::move(param_keys)}));
            }
            start += size;
          }
        } else {
          // Fallback to naive implementation: multiple copy_
          int64_t start = 0;
          for (int64_t i = 0; i < out.size(); ++i) {
            int64_t size = split_sizes[i];
            out[i].copy_(self.narrow(normalized_dim, start, size));
            start += size;
          }
        }
      });
}

}  // namespace torch_tpu
