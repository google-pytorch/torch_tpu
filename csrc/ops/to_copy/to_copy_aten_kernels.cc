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

#include "csrc/ops/to_copy/to_copy_aten_kernels.h"

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/empty_strided.h"
#include "c10/core/Device.h"
#include "c10/core/TensorOptions.h"
#include "csrc/common/cache_key.h"
#include "csrc/ops/copy_from/copy_from_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

at::Tensor AtenToCopy(const at::Tensor& self,
                      std::optional<at::ScalarType> dtype,
                      std::optional<at::Layout> layout,
                      std::optional<at::Device> device,
                      std::optional<bool> pin_memory, bool non_blocking,
                      std::optional<at::MemoryFormat> memory_format) {
  TT_KERNEL(
      OpName::kToCopy, _,
      // Note: these cache key arguments are ignored for the compiled param keys
      // because this kernel is a special case that is not "dispatched" since
      // it's not compiled.
      (self, IgnoreInCacheKey(dtype, "Doesn't affect SHLO"),
       IgnoreInCacheKey(layout, "Unused"),
       IgnoreInCacheKey(device, "Doesn't affect SHLO"),
       IgnoreInCacheKey(pin_memory, "Unused"),
       IgnoreInCacheKey(non_blocking, "Doesn't affect SHLO"),
       IgnoreInCacheKey(memory_format, "Doesn't affect SHLO")),
      {
        at::Device target_device = device.value_or(self.device());
        at::ScalarType target_dtype = dtype.value_or(self.scalar_type());
        at::MemoryFormat resolved_format =
            memory_format.value_or(at::MemoryFormat::Preserve);
        bool preserve_strides = resolved_format == at::MemoryFormat::Preserve;
        if (preserve_strides) {
          resolved_format = self.suggest_memory_format();
        }

        // If we are moving to CPU and the user requested non_blocking=True,
        // we want to ensure the destination is allocated in pinned memory.
        // This allows CopyTpuToCpu to use the fast async DMA path.
        if (target_device.is_cpu() && non_blocking) {
          at::TensorOptions options =
              at::TensorOptions().dtype(target_dtype).device(at::kCPU);
          // Explicitly use our pinned allocator for the intermediate.
          at::Tensor dest = at::empty(self.sizes(), options, resolved_format)
                                .pin_memory(at::Device(at::kPrivateUse1));
          // Call our TPU-specific copy_ implementation directly to avoid
          // redispatching back to this _to_copy implementation.
          ::torch_tpu::AtenCopy_(dest, self, non_blocking);
          return dest;
        }

        // Default path: let ATen handle the intermediate allocation.
        at::TensorOptions options =
            at::TensorOptions().dtype(target_dtype).device(target_device);
        at::Tensor dest;  // UNINITIALIZED_TENSOR_OK=initialized in the if-else
        if (preserve_strides && !self.is_contiguous() &&
            self.is_non_overlapping_and_dense()) {
          dest = at::empty_strided(self.sizes(), self.strides(), options);
        } else {
          dest = at::empty(self.sizes(), options, resolved_format);
        }
        ::torch_tpu::AtenCopy_(dest, self, non_blocking);
        return dest;
      });
}

}  // namespace torch_tpu
