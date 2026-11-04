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

#include <cstdint>
#include <optional>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/empty_strided.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Optional.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/as_strided/as_strided_aten_kernels.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/copy_from_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/sparse_coo/sparse_coo_aten_kernels.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

// Creates a TPU tensor that self can be copied into, while attempting to
// preserve the strides of self.
//
// target_device must be a TPU device.
//
// If self is non-overlapping and dense, the returned tensor will have the same
// strides and storage offset as self.
//
// If self is overlapping or non-dense, then the returned tensor will have the
// same dimensions as self, but the stride pattern will be dense and
// non-overlapping, and the offset will be 0.
absl::StatusOr<at::Tensor> MakeTpuReceiverPreserveStrides(
    const at::Tensor& self, at::ScalarType target_dtype,
    std::optional<c10::DeviceIndex> tpu_device_index) {
  c10::DeviceIndex device_index =
      tpu_device_index.value_or(GetCurrentDeviceIndex());
  c10::Device target_device(GetPrivateUse1DeviceType(), device_index);

  std::vector<int64_t> owned_strides;  // INT_VEC_OK=infer_dense_strides return
  at::IntArrayRef strides = self.strides();

  if (!self.is_non_overlapping_and_dense()) {
    // Input is overlapping or non-dense (e.g., from slice/unbind/expand):
    // compute non-overlapping and dense strides that preserve the input's
    // dimension permutation.
    owned_strides = at::infer_dense_strides(self.sizes(), self.strides());
    strides = at::IntArrayRef(owned_strides);
  }

  TT_ASSIGN_OR_RETURN(at::Tensor receiver,
                      MakeEmptyMemoryFormat({self.numel()}, target_dtype,
                                            c10::Layout::Strided, target_device,
                                            /*pin_memory_opt=*/c10::nullopt,
                                            at::MemoryFormat::Contiguous));
  return AsStrided(receiver, self.sizes(), strides, /*storage_offset=*/0);
}

// Creates an empty TPU tensor that self can be copied into.
//
// target_device must be a TPU device.
//
// If memory_format is Preserve (the default), the strides and storage offset
// are preserved if self is non-overlapping and dense. If memory_format is
// Preserve but self is overlapping or non-dense, the returned tensor will have
// the same dimensions as self, but the stride pattern will be dense and
// non-overlapping, and the offset will be 0.
absl::StatusOr<at::Tensor> MakeEmptyTpuReceiver(
    const at::Tensor& self, at::ScalarType target_dtype,
    std::optional<at::MemoryFormat> format,
    std::optional<c10::DeviceIndex> tpu_device_index) {
  TT_RETURN_IF_ERROR(ValidateShellDtypeCast(self.scalar_type(), target_dtype));
  format = format.value_or(at::MemoryFormat::Preserve);
  if (format == at::MemoryFormat::Preserve) {
    return MakeTpuReceiverPreserveStrides(self, target_dtype, tpu_device_index);
  }
  // Explicit memory format requested (e.g., Contiguous, ChannelsLast):
  // allocate according to resolved_format.
  c10::DeviceIndex device_index =
      tpu_device_index.value_or(GetCurrentDeviceIndex());
  c10::Device target_device(GetPrivateUse1DeviceType(), device_index);
  return MakeEmptyMemoryFormat(self.sizes(), target_dtype, c10::Layout::Strided,
                               target_device,
                               /*pin_memory_opt=*/c10::nullopt, format);
}

}  // namespace

absl::StatusOr<at::Tensor> CopyToTpuTensor(
    const at::Tensor& self, std::optional<at::ScalarType> dtype,
    bool non_blocking, std::optional<at::MemoryFormat> memory_format,
    std::optional<c10::DeviceIndex> tpu_device_index) {
  TT_ASSIGN_OR_RETURN(
      at::Tensor dest,
      MakeEmptyTpuReceiver(self, dtype.value_or(self.scalar_type()),
                           memory_format.value_or(at::MemoryFormat::Preserve),
                           tpu_device_index));
  return CopyTensor(self, dest, non_blocking);
}

at::Tensor AtenToCopy(const at::Tensor& self,
                      std::optional<at::ScalarType> dtype,
                      std::optional<at::Layout> layout,
                      std::optional<at::Device> device,
                      std::optional<bool> pin_memory, bool non_blocking,
                      std::optional<at::MemoryFormat> memory_format) {
  if (self.is_sparse()) {
    return AtenToCopySparse(self, dtype, layout, device, pin_memory,
                            non_blocking, memory_format);
  }
  TT_KERNEL(
      OpName::kToCopy, _,
      // Note: these cache key arguments are ignored for the compiled
      // param keys because this kernel is a special case that is not
      // "dispatched" since it's not compiled.
      (self, IgnoreInCacheKey(dtype, "Doesn't affect SHLO"),
       IgnoreInCacheKey(layout, "Unused"),
       IgnoreInCacheKey(device, "Doesn't affect SHLO"),
       IgnoreInCacheKey(pin_memory, "Unused"),
       IgnoreInCacheKey(non_blocking, "Doesn't affect SHLO"),
       IgnoreInCacheKey(memory_format, "Doesn't affect SHLO")),
      {
        at::Device target_device = device.value_or(self.device());
        at::ScalarType target_dtype = dtype.value_or(self.scalar_type());
        TT_THROW_IF_ERROR(
            ValidateShellDtypeCast(self.scalar_type(), target_dtype));
        at::MemoryFormat resolved_format =
            memory_format.value_or(at::MemoryFormat::Preserve);
        bool preserve_strides = resolved_format == at::MemoryFormat::Preserve;
        if (preserve_strides) {
          resolved_format = self.suggest_memory_format();
        }

        at::Tensor dest;  // UNINITIALIZED_TENSOR_OK=initialized in the if-else
        if (!target_device.is_cpu()) {
          // Create an empty TPU receiver.
          TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=no setup to attempt a
                           // transfer between TPU to non-CPU, non-TPU device.
              target_device.is_privateuseone(), error::kInvalidArgument)
              << "cannot copy a tensor from " << self.device() << " to "
              << target_device
              << ". Only copies to and from TPU and CPU are supported";
          TT_ASSIGN_OR_THROW(
              dest, MakeEmptyTpuReceiver(self, target_dtype, memory_format,
                                         target_device.index()));
        } else if (non_blocking) {
          // If we are moving to CPU and the user requested
          // non_blocking=True, we want to ensure the destination is
          // allocated in pinned memory. This allows CopyTpuToCpu to use the
          // fast async DMA path.
          at::TensorOptions options = at::TensorOptions()
                                          .dtype(target_dtype)
                                          .device(at::kCPU)
                                          .pinned_memory(true);
          dest = at::empty(self.sizes(), options, resolved_format);
        } else {
          // dest is created on CPU so we need to redispatch to the CPU
          // implementation.
          at::TensorOptions options =
              at::TensorOptions().dtype(target_dtype).device(target_device);
          if (preserve_strides && self.is_non_overlapping_and_dense()) {
            // Input is already non-overlapping and dense: preserve its exact
            // strides.
            dest = at::empty_strided(self.sizes(), self.strides(), options);
          } else if (preserve_strides) {
            // Input is overlapping or non-dense (e.g., from
            // slice/unbind/expand): compute non-overlapping and dense strides
            // that preserve the input's dimension permutation.
            auto strides =
                at::infer_dense_strides(self.sizes(), self.strides());
            dest = at::empty_strided(self.sizes(), strides, options);
          } else {
            // Explicit memory format requested (e.g., Contiguous,
            // ChannelsLast): allocate according to resolved_format.
            dest = at::empty(self.sizes(), options, resolved_format);
          }
        }

        // Equivalent to a redispatch to AtenCopy_.
        CopyTensor(self, dest, non_blocking);
        return dest;
      });
}

}  // namespace torch_tpu
