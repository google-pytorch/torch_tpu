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
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/cat.h"
#include "ATen/ops/split_with_sizes.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"

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
        if (dim == 1 && self.dim() == 2) {
          const int64_t num_rows = self.size(0);
          const at::Tensor flat_self = self.view(-1);

          const int64_t num_params = out.size();
          std::vector<int64_t>  // INT_VEC_OK=Not tensor dimension
              all_split_sizes;
          all_split_sizes.reserve(num_params * num_rows);
          for (int64_t j = 0; j < num_rows; ++j) {
            for (int64_t size : split_sizes) {
              all_split_sizes.push_back(size);
            }
          }

          auto src_pieces = at::split_with_sizes(flat_self, all_split_sizes, 0);

          for (int64_t i = 0; i < num_params; ++i) {
            std::vector<at::Tensor> param_i_shards;
            param_i_shards.reserve(num_rows);
            for (int64_t j = 0; j < num_rows; ++j) {
              param_i_shards.push_back(src_pieces[j * num_params + i]);
            }
            at::Tensor out_i_flat = out[i].view(-1);
            at::cat_out(out_i_flat, param_i_shards, 0);
          }
        } else {
          // Fallback to naive implementation: multiple copy_
          int64_t start = 0;
          for (int64_t i = 0; i < out.size(); ++i) {
            int64_t size = split_sizes[i];
            out[i].copy_(self.narrow(dim, start, size));
            start += size;
          }
        }
      });
}

}  // namespace torch_tpu
