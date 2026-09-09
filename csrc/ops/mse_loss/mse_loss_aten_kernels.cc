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

#include "csrc/ops/mse_loss/mse_loss_aten_kernels.h"

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/Reduction.h"
#include "ATen/native/Resize.h"
#include "ATen/ops/div.h"
#include "ATen/ops/mean.h"
#include "ATen/ops/mul.h"
#include "ATen/ops/pow.h"
#include "ATen/ops/sub.h"
#include "ATen/ops/sum.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

at::Tensor& AtenMseLossOut(const at::Tensor& self, const at::Tensor& target,
                           int64_t reduction, at::Tensor& out) {
  TT_KERNEL(
      OpName::kMseLossOut, _,
      (self, target, IgnoreInCacheKey(reduction, "Doesn't affect SHLO"), out), {
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Byte &&
                           self.scalar_type() != at::ScalarType::Char &&
                           self.scalar_type() != at::ScalarType::Short &&
                           self.scalar_type() != at::ScalarType::Int &&
                           self.scalar_type() != at::ScalarType::Long &&
                           self.scalar_type() != at::ScalarType::ComplexFloat,
                       error::kInvalidArgument)
            << "expected the input dtype to be none of (uint8, int8, int16, "
               "int32, int64, complex64), got "
            << torch_tpu::ToString(self.scalar_type());
        TT_CHECK_THROW(self.scalar_type() == target.scalar_type(),
                       error::kInvalidArgument)
            << "expected input and target to have the same dtype, got "
            << torch_tpu::ToString(self.scalar_type()) << " and "
            << torch_tpu::ToString(target.scalar_type());

        {
          // All compositional ATen calls must be within a guard to prevent
          // re-dispatching to the Autograd layer, which would cause infinite
          // recursion. This ensures we are calling the primitive PrivateUse1
          // kernels.
          at::AutoDispatchBelowAutograd guard;

          at::Tensor squared_error = at::pow(at::sub(self, target), 2);
          at::Tensor
              result;  // UNINITIALIZED_TENSOR_OK=initialized in the if-else
          if (reduction == at::Reduction::None) {
            result = squared_error;
          } else if (reduction == at::Reduction::Mean) {
            result = at::mean(squared_error);
          } else if (reduction == at::Reduction::Sum) {
            result = at::sum(squared_error);
          } else {
            TT_CHECK_THROW(false, error::kInvalidArgument)
                << "unrecognized reduction mode " << reduction;
          }

          // Resize the output tensor and copy the result into it.
          // This is the standard way to handle `out=` variants.
          at::native::resize_output(out, result.sizes());
          out.copy_(result);
        }
        return out;
      });
}

at::Tensor AtenMseLossBackward(const at::Tensor& grad_output,
                               const at::Tensor& self, const at::Tensor& target,
                               int64_t reduction) {
  TT_KERNEL(
      OpName::kMseLossBackward, _,
      (grad_output, self, target,
       IgnoreInCacheKey(reduction, "Doesn't affect SHLO")),
      {
        at::Tensor grad_input;  // UNINITIALIZED_TENSOR_OK=initialized below
        {
          at::AutoDispatchBelowAutograd guard;
          at::Tensor grad_raw = at::mul(at::sub(self, target), 2);
          if (reduction == at::Reduction::None) {
            grad_input = at::mul(grad_raw, grad_output);
          } else {
            grad_input = at::mul(grad_raw, grad_output);
            if (reduction == at::Reduction::Mean) {
              grad_input = at::div(grad_input, self.numel());
            }
          }
        }
        return grad_input;
      });
}

}  // namespace torch_tpu
