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

#include "csrc/ops/cumprod/cumprod_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/functional/bind_front.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/cumprod/cumprod.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

at::Tensor& AtenCumprodOut(const at::Tensor& self, int64_t dim,
                           std::optional<at::ScalarType> dtype,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kCumprodOut, param_keys, (self, dim, dtype, out), {
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));

    // PyTorch ATen parity: when explicit dtype is provided, out tensor dtype
    // must strictly match dtype; safe-casting is only allowed when dtype is
    // omitted.
    if (dtype.has_value()) {
      TT_CHECK_THROW(out.scalar_type() == dtype.value(),
                     error::kInvalidArgument)
          << "expected the output dtype to be " << ToString(dtype.value())
          << ", got " << ToString(out.scalar_type());
    }

    // Dimension must be validated before early returns so that invalid dims
    // (e.g. dim > 0 on a 0-D scalar or empty tensor) throw IndexError.
    TT_ASSIGN_OR_THROW(const int64_t normalized_dim,
                       SafeWrapDim(dim, self.dim()));

    // Fast-path: empty tensors require no computation.
    if (out.numel() == 0) {
      return out;
    }

    // 0-D scalar tensor: cumulative scan along a scalar is an identity copy.
    // In PyTorch ATen, 0-D scalar tensors (including boolean) succeed via
    // fill_.
    if (self.dim() == 0) {
      out.copy_(self);
      return out;
    }

    // PyTorch does not support boolean cumulative scans for non-empty tensors
    // with dim > 0.
    TT_CHECK_THROW(out.scalar_type() != at::ScalarType::Bool,
                   error::kPythonNotImplementedError)
        << "not implemented for " << ToString(at::ScalarType::Bool);

    TT_ASSIGN_OR_THROW(const mlir::ElementType out_mlir_type,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));

    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        absl::bind_front(BuildCumprodShlo, normalized_dim, out_mlir_type),
        {.op_param_cache_keys = std::move(param_keys),
         .out_dtype = out_mlir_type,
         .allow_low_precision_float = !IsLowPrecisionFloat(out)}));
    return out;
  });
}

}  // namespace torch_tpu
