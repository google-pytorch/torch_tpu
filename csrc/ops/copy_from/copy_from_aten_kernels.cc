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

#include "csrc/ops/copy_from/copy_from_aten_kernels.h"

#include <string>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/cpu_to_tpu.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/copy_from/tpu_to_tpu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

namespace {

enum class CopyType {
  kCpuToTpu,
  kTpuToTpu,
  kTpuToCpu,
};

absl::StatusOr<CopyType> GetCopyType(const at::Tensor& src,
                                     const at::Tensor& dest) {
  if (IsPrivateUse1Device(src)) {
    // Copy is from TPU device.
    if (IsPrivateUse1Device(dest)) {
      return CopyType::kTpuToTpu;
    } else if (IsCpuDevice(dest)) {
      return CopyType::kTpuToCpu;
    }
  } else if (IsCpuDevice(src) && IsPrivateUse1Device(dest)) {
    return CopyType::kCpuToTpu;
  }

  const std::string& device_name = GetPrivateUse1DeviceDebugName();

  return TT_ERROR(  // ERROR_COV_INFEASIBLE=Can't test with TPU and another
                    // device except for CPU.
             error::kPythonNotImplementedError)
         << "expected copy of either (i) '" << device_name << "' to '"
         << device_name << "'; (ii) '" << device_name
         << "' to 'cpu'; or (iii) 'cpu' to '" << device_name << "', got '"
         << src.device().type() << "' to '" << dest.device().type() << "'";
}

}  // namespace

at::Tensor CopyTensor(const at::Tensor& src, const at::Tensor& self_dest,
                      bool non_blocking) {
  TT_CHECK_THROW(IsPrivateUse1Device(src) || IsPrivateUse1Device(self_dest),
                 error::kInvalidArgument)
      << "expected at least one of the inputs to be on '"
      << GetPrivateUse1DeviceDebugName() << "' device, got '"
      << src.device().type() << "' (source) and '" << self_dest.device().type()
      << "' (destination)";

  if (src.is_same(self_dest)) {
    // Self-to-self copy is a no-op.
    return self_dest;
  }

  TT_ASSIGN_OR_THROW(CopyType to_copy_type, GetCopyType(src, self_dest));

  switch (to_copy_type) {
    case CopyType::kCpuToTpu:
      if (src.sizes() != self_dest.sizes()) {
        // Broadcast CPU to TPU: copy to temp TPU then broadcast TPU to TPU.
        TT_ASSIGN_OR_THROW(at::Tensor temp_tpu,
                           MakeEmptyTensor(src.sizes(), self_dest.scalar_type(),
                                           self_dest.device()));
        TT_THROW_IF_ERROR(CopyCpuToTpu(src, temp_tpu, non_blocking));
        TT_THROW_IF_ERROR(CopyTpuToTpu(temp_tpu, self_dest));
      } else {
        TT_THROW_IF_ERROR(CopyCpuToTpu(src, self_dest, non_blocking));
      }
      break;
    case CopyType::kTpuToTpu:
      TT_THROW_IF_ERROR(CopyTpuToTpu(src, self_dest));
      break;
    case CopyType::kTpuToCpu:
      // CopyTpuToCpu handles broadcasting/dtype change via fallback CPU copy.
      TT_THROW_IF_ERROR(CopyTpuToCpu(src, self_dest, non_blocking));
      break;
  }

  // Also copy the gradient if it exists on the src.
  if (src.grad().defined()) {
    at::Tensor dest_grad = self_dest.grad();
    if (!dest_grad.defined()) {
      if (to_copy_type == CopyType::kTpuToCpu) {
        // at::empty here dispatches directly to the native CPU allocator and
        // does not make the parent op composite.
        dest_grad = at::empty(self_dest.sizes(), self_dest.options());
      } else {
        TT_ASSIGN_OR_THROW(dest_grad, MakeEmptyTensor(self_dest.sizes(),
                                                      self_dest.scalar_type(),
                                                      self_dest.device()));
      }
      self_dest.mutable_grad() = dest_grad;
    }
    CopyTensor(src.grad(), dest_grad, non_blocking);
  }
  return self_dest;
}

at::Tensor AtenCopyFrom(const at::Tensor& src, const at::Tensor& self_dest,
                        bool non_blocking) {
  tsl::profiler::TraceMe trace("AtenCopyFrom");
  TT_KERNEL(
      OpName::kCopyFrom, _,
      (src, self_dest, IgnoreInCacheKey(non_blocking, "Doesn't affect SHLO")),
      { return CopyTensor(src, self_dest, non_blocking); });
}

at::Tensor AtenCopyFromAndResize(const at::Tensor& self,
                                 const at::Tensor& dst) {
  TT_KERNEL(OpName::kCopyFromAndResize, _, (self, dst), {
    if (dst.device().type() == GetPrivateUse1DeviceType()) {
      TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(dst, self.sizes()));
    } else {
      dst.resize_(self.sizes());
    }
    CopyTensor(self, dst, /*non_blocking=*/false);
    return dst;
  });
}

at::Scalar AtenLocalScalarDense(const at::Tensor& self) {
  tsl::profiler::TraceMe trace("AtenLocalScalarDense");
  TT_KERNEL(OpName::kLocalScalarDense, _, (self), {
    TT_CHECK_THROW(self.numel() == 1, error::kInvalidArgument)
        << "expected the input tensor to have 1 element, got " << self.numel();
    at::Tensor cpu_tensor =
        self.to(at::TensorOptions().device(c10::kCPU).memory_format(
            at::MemoryFormat::Contiguous));

    return cpu_tensor.item();
  });
}

// Identical to _copy_from, only registered to avoid a dispatch error in torch.
at::Tensor& AtenCopy_(at::Tensor& self, const at::Tensor& src,
                      bool non_blocking) {
  TT_KERNEL(OpName::kCopy_, _,
            (self, src, IgnoreInCacheKey(non_blocking, "Doesn't affect SHLO")),
            {
              CopyTensor(src, self, non_blocking);
              return self;
            });
}

}  // namespace torch_tpu
