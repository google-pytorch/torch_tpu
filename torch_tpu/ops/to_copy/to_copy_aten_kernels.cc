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

#include "torch_tpu/ops/to_copy/to_copy_aten_kernels.h"

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "c10/core/Device.h"
#include "c10/core/TensorOptions.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"

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
      (self, IgnoreInCacheKey(dtype, "Legacy usage"),
       IgnoreInCacheKey(layout, "Legacy usage"),
       IgnoreInCacheKey(device, "Legacy usage"),
       IgnoreInCacheKey(pin_memory, "Legacy usage"),
       IgnoreInCacheKey(non_blocking, "Legacy usage"),
       IgnoreInCacheKey(memory_format, "Legacy usage")),
      {
        at::Device target_device = device.value_or(self.device());
        at::ScalarType target_dtype = dtype.value_or(self.scalar_type());
        at::MemoryFormat resolved_format =
            memory_format.value_or(at::MemoryFormat::Preserve);
        if (resolved_format == at::MemoryFormat::Preserve) {
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
          AtenCopy_(dest, self, non_blocking);
          return dest;
        }

        // Default path: let ATen handle the intermediate allocation.
        at::TensorOptions options =
            at::TensorOptions().dtype(target_dtype).device(target_device);
        at::Tensor dest = at::empty(self.sizes(), options, resolved_format);
        AtenCopy_(dest, self, non_blocking);
        return dest;
      });
}

}  // namespace torch_tpu
