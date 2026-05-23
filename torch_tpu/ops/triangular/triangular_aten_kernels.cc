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

#include "torch_tpu/ops/triangular/triangular_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/functional/bind_front.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/triangular/triangular.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {

at::Tensor& AtenTriuOut(const at::Tensor& self, int64_t diagonal,
                        at::Tensor& out) {
  TT_KERNEL(OpName::kTriuOut, param_keys, (self, diagonal, out), {
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        absl::bind_front(BuildTriangularShlo, diagonal, TriangleType::kUpper),
        {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

at::Tensor& AtenTrilOut(const at::Tensor& self, int64_t diagonal,
                        at::Tensor& out) {
  TT_KERNEL(OpName::kTrilOut, param_keys, (self, diagonal, out), {
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        absl::bind_front(BuildTriangularShlo, diagonal, TriangleType::kLower),
        {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

}  // namespace torch_tpu
