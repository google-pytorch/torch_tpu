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

#include "torch_tpu/ops/cumprod/cumprod_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "absl/functional/bind_front.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/cumprod/cumprod.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {

at::Tensor& AtenCumprodOut(const at::Tensor& self, int64_t dim,
                           std::optional<at::ScalarType> dtype,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kCumprodOut, param_keys, (self, dim, dtype, out), {
    if (out.numel() == 0) {
      return out;
    }
    if (self.dim() == 0) {
      out.copy_(self);
      return out;
    }

    TT_ASSIGN_OR_THROW(const auto out_mlir_type,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    TT_THROW_IF_ERROR(UnaryOpOut(self, out,
                                 absl::bind_front(BuildCumprodShlo, dim, dtype),
                                 {.op_param_cache_keys = std::move(param_keys),
                                  .out_dtype = out_mlir_type}));
    return out;
  });
}

}  // namespace torch_tpu
