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

#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"

#include <string>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Optional.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/copy_from/cpu_to_tpu.h"
#include "torch_tpu/ops/copy_from/tpu_to_cpu.h"
#include "torch_tpu/ops/copy_from/tpu_to_tpu.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"
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
             error::kUnimplemented)
         << "expected copy of either (i) '" << device_name << "' to '"
         << device_name << "'; (ii) '" << device_name
         << "' to 'cpu'; or (iii) 'cpu' to '" << device_name << "', got '"
         << src.device().type() << "' to '" << dest.device().type() << "'";
}

}  // namespace

at::Tensor AtenCopyFrom(const at::Tensor& src, const at::Tensor& self_dest,
                        bool non_blocking) {
  tsl::profiler::TraceMe trace("AtenCopyFrom");
  TT_KERNEL(
      OpName::kCopyFrom, _,
      (src, self_dest, IgnoreInCacheKey(non_blocking, "Doesn't affect SHLO")), {
        TT_CHECK_THROW(
            IsPrivateUse1Device(src) || IsPrivateUse1Device(self_dest),
            error::kInvalidArgument)
            << "expected at least one of the inputs to be on '"
            << GetPrivateUse1DeviceDebugName() << "' device, got '"
            << src.device().type() << "' (source) and '"
            << self_dest.device().type() << "' (destination)";

        if (src.is_same(self_dest)) {
          // Self-to-self copy is a no-op.
          return self_dest;
        }

        // Broadcast if necessary (on the source device).
        at::Tensor broadcasted_src = src;
        if (src.sizes() != self_dest.sizes()) {
          tsl::profiler::TraceMe trace_broadcast("AtenCopyFrom::Broadcast");
          while (broadcasted_src.dim() < self_dest.dim()) {
            broadcasted_src.unsqueeze_(0);
          }
          // This may throw if the src tensor cannot be expanded.
          broadcasted_src = broadcasted_src.expand(self_dest.sizes());
        }

        // Perform a CPU-to-TPU, TPU-to-TPU, or TPU-to-CPU copy; in any case,
        // self_dest will be updated with a new c10::StorageImpl containing the
        // same values as exist in src's c10::DataPtr.
        TT_ASSIGN_OR_THROW(CopyType to_copy_type, GetCopyType(src, self_dest));

        switch (to_copy_type) {
          case CopyType::kCpuToTpu:
            TT_THROW_IF_ERROR(
                CopyCpuToTpu(broadcasted_src, self_dest, non_blocking));
            break;
          case CopyType::kTpuToTpu:
            TT_THROW_IF_ERROR(CopyTpuToTpu(broadcasted_src, self_dest));
            break;
          case CopyType::kTpuToCpu:
            TT_THROW_IF_ERROR(
                CopyTpuToCpu(broadcasted_src, self_dest, non_blocking));
            break;
        }

        // Also copy the gradient if it exists on the src.
        if (src.grad().defined()) {
          // self_dest.grad() may not yet be defined; redispatch through to()
          // and not copy_ to avoid undefined tensor accesses.
          at::Tensor new_grad =
              src.grad().to(self_dest.options(), non_blocking, /*copy=*/true);
          self_dest.mutable_grad() = new_grad;
        }
        return self_dest;
      });
}

at::Tensor AtenCopyFromAndResize(const at::Tensor& self,
                                 const at::Tensor& dst) {
  TT_KERNEL(OpName::kCopyFromAndResize, _, (self, dst), {
    if (dst.device().type() == GetPrivateUse1DeviceType()) {
      AtenResize_(dst, self.sizes(), at::MemoryFormat::Contiguous);
    } else {
      dst.resize_(self.sizes());
    }
    return AtenCopyFrom(self, dst, /*non_blocking=*/false);
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
  TT_KERNEL(
      OpName::kCopy_, _,
      (self, src, IgnoreInCacheKey(non_blocking, "Delegates to AtenCopyFrom")),
      {
        AtenCopyFrom(src, self, non_blocking);
        return self;
      });
}

}  // namespace torch_tpu
