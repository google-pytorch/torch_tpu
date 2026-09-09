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

#include "csrc/ops/reductions/sum_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "c10/util/OptionalArrayRef.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

at::Tensor& AtenSumIntListOut(const at::Tensor& self,
                              c10::OptionalArrayRef<int64_t> dim, bool keep_dim,
                              std::optional<c10::ScalarType> dtype,
                              at::Tensor& out) {
  TT_KERNEL(
      OpName::kSumIntListOut, param_keys, (self, dim, keep_dim, dtype, out), {
        c10::ScalarType scalar_dtype = dtype.value_or(out.scalar_type());
        if (self.numel() == 0) {
          out = at::zeros({}, at::TensorOptions().dtype(scalar_dtype));
          return out;
        }
        TT_THROW_IF_ERROR(ApplySumReductionOut(
            self, out, dim,
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims,
            scalar_dtype, std::move(param_keys)));
        return out;
      });
}

}  // namespace torch_tpu
