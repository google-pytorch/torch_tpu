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
#include "torch_tpu/csrc/ops/linalg/linalg_kernels.h"

#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/IListRef_inl.h"
#include "ATen/ops/_linalg_check_errors.h"
#include "ATen/ops/diagflat.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/empty_like.h"
#include "ATen/ops/ones.h"
#include "c10/core/ScalarType.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/ops/linalg/lu/linalg_lu_kernels.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
AtenLinalgSolveExOut(const at::Tensor& a, const at::Tensor& b, bool left,
                     bool check_errors, at::Tensor& result, at::Tensor& lu,
                     at::Tensor& pivots, at::Tensor& info) {
  TT_KERNEL(
      OpName::kLinalgSolveExOut, _,
      (a, b, IgnoreInCacheKey(left, "Delegates to AtenLinalgLuSolveOut"),
       IgnoreInCacheKey(check_errors, "Doesn't affect SHLO"), result, lu,
       pivots, info),
      {
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(result, b.sizes()));
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(lu, a.sizes()));

        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(
            pivots, a.sizes().slice(0, a.dim() - 1)));
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(info, a.sizes().slice(0, a.dim() - 2)));

        if (a.numel() == 0 || b.numel() == 0) {
          info.zero_();
          return {result, lu, pivots, info};
        }

        AtenLinalgLuFactorExOut(a, /*pivot=*/true,
                                /*check_errors=*/false, lu, pivots, info);
        if (check_errors) {
          at::_linalg_check_errors(info, "solve", a.dim() == 2);
        }
        if (b.dim() < a.dim()) {
          at::Tensor b_expanded =
              left ? b.unsqueeze(b.dim()) : b.unsqueeze(b.dim() - 1);
          at::Tensor expanded_result = at::empty_like(b_expanded);
          AtenLinalgLuSolveOut(lu, pivots, b_expanded, left,
                               /*adjoint=*/false, expanded_result);
          result.copy_(expanded_result.reshape(result.sizes()));
        } else {
          AtenLinalgLuSolveOut(lu, pivots, b, left, /*adjoint=*/false, result);
        }

        return {result, lu, pivots, info};
      });
}

std::tuple<at::Tensor&, at::Tensor&> AtenLinalgInvExInverse(const at::Tensor& a,
                                                            bool check_errors,
                                                            at::Tensor& inverse,
                                                            at::Tensor& info) {
  TT_KERNEL(
      OpName::kLinalgInvExInverse, _,
      (a, IgnoreInCacheKey(check_errors, "Doesn't affect SHLO"), inverse, info),
      {
        TT_CHECK_THROW(a.dim() >= 2, error::kInvalidArgument)
            << "expected the input tensor to have at least 2 dimensions, got "
            << a.dim() << " dimensions of shape " << ToString(a.sizes());
        TT_CHECK_THROW(a.size(-1) == a.size(-2), error::kInvalidArgument)
            << "expected the input tensor's last 2 dimensions to be equal, got "
            << ToString(a.sizes());
        at::Tensor lu = at::empty_like(a);
        Dimensions pivot_dims = CopyIntVector(a.sizes());
        pivot_dims.pop_back();
        at::Tensor pivots = at::empty(pivot_dims, a.options().dtype(at::kInt));
        AtenLinalgSolveExOut(
            a, at::diagflat(at::ones({a.size(-1)}, a.options())).expand_as(a),
            /*left=*/true,
            /*check_errors=*/false, inverse, lu, pivots, info);
        if (check_errors) {
          at::_linalg_check_errors(info, "inv", a.dim() == 2);
        }
        return {inverse, info};
      });
}

}  // namespace torch_tpu
