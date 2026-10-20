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
#include <utility>

#include "ATen/SparseTensorImpl.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/typeid.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
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

struct SparseBuffers {
  at::Tensor indices;  // UNINITIALIZED_TENSOR_OK=Struct field initialized by
                       // aggregate construction
  at::Tensor values;   // UNINITIALIZED_TENSOR_OK=Struct field initialized by
                       // aggregate construction
};

SparseBuffers CopySparseBuffers(const at::Tensor& src_sparse,
                                const at::Tensor& target_ref,
                                CopyType to_copy_type, bool non_blocking) {
  const at::Tensor src_indices = src_sparse._indices();
  const at::Tensor src_values = src_sparse._values();
  at::Tensor dest_indices;  // UNINITIALIZED_TENSOR_OK=initialized in switch
  at::Tensor dest_values;   // UNINITIALIZED_TENSOR_OK=initialized in switch
  switch (to_copy_type) {
    case CopyType::kTpuToCpu:
      dest_indices = at::empty(src_indices.sizes(),
                               src_indices.options().device(c10::kCPU));
      dest_values = at::empty(src_values.sizes(),
                              src_values.options().device(c10::kCPU).dtype(
                                  target_ref.scalar_type()));
      break;
    case CopyType::kCpuToTpu:
    case CopyType::kTpuToTpu:
      TT_ASSIGN_OR_THROW(
          dest_indices,
          MakeEmptyTensor(src_indices.sizes(), src_indices.scalar_type(),
                          target_ref.device()));
      TT_ASSIGN_OR_THROW(dest_values, MakeEmptyTensor(src_values.sizes(),
                                                      target_ref.scalar_type(),
                                                      target_ref.device()));
      break;
  }
  CopyTensor(src_indices, dest_indices, non_blocking);
  CopyTensor(src_values, dest_values, non_blocking);
  return {dest_indices, dest_values};
}

struct TpuSparseTensorImpl final : public at::SparseTensorImpl {
  TpuSparseTensorImpl(const at::DispatchKeySet key_set,
                      const caffe2::TypeMeta data_type,
                      const c10::Device device)
      : at::SparseTensorImpl(at::DispatchKeySet(c10::DispatchKey::SparseCPU),
                             data_type) {
    key_set_ = key_set;
    device_opt_ = device;
  }
};

at::Tensor CopySparseTensor(const at::Tensor& src_sparse,
                            const at::Tensor& target_ref, CopyType to_copy_type,
                            bool non_blocking) {
  auto [dest_indices, dest_values] =
      CopySparseBuffers(src_sparse, target_ref, to_copy_type, non_blocking);
  const c10::DispatchKey dispatch_key =
      dest_values.device().is_cpu() ? c10::DispatchKey::SparseCPU
                                    : c10::DispatchKey::SparsePrivateUse1;
  auto impl = c10::make_intrusive<TpuSparseTensorImpl>(
      at::DispatchKeySet(dispatch_key),
      c10::scalarTypeToTypeMeta(dest_values.scalar_type()),
      dest_values.device());
  impl->raw_resize_(src_sparse.sparse_dim(), src_sparse.dense_dim(),
                    src_sparse.sizes());
  impl->set_indices_and_values_unsafe(
      at::Tensor(dest_indices.unsafeGetTensorImpl()->shallow_copy_and_detach(
          dest_indices.unsafeGetTensorImpl()->version_counter(), true)),
      at::Tensor(dest_values.unsafeGetTensorImpl()->shallow_copy_and_detach(
          dest_values.unsafeGetTensorImpl()->version_counter(), true)));
  impl->set_coalesced(src_sparse.is_coalesced());
  return at::Tensor(std::move(impl));
}

void CopyTensorGrad(const at::Tensor& src, const at::Tensor& self_dest,
                    CopyType to_copy_type, bool non_blocking) {
  if (src.requires_grad() && !src.is_leaf() && !src.retains_grad()) {
    return;
  }
  if (!src.grad().defined()) {
    return;
  }
  if (src.grad().is_sparse()) {
    self_dest.mutable_grad() =
        CopySparseTensor(src.grad(), self_dest, to_copy_type, non_blocking);
    return;
  }
  at::Tensor dest_grad = self_dest.grad();
  if (!dest_grad.defined()) {
    switch (to_copy_type) {
      case CopyType::kTpuToCpu:
        // at::empty here dispatches directly to the native CPU allocator and
        // does not make the parent op composite.
        dest_grad = at::empty(self_dest.sizes(), self_dest.options());
        break;
      case CopyType::kCpuToTpu:
      case CopyType::kTpuToTpu:
        TT_ASSIGN_OR_THROW(dest_grad, MakeEmptyTensor(self_dest.sizes(),
                                                      self_dest.scalar_type(),
                                                      self_dest.device()));
        break;
    }
    self_dest.mutable_grad() = dest_grad;
  }
  CopyTensor(src.grad(), dest_grad, non_blocking);
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
  const TransferSyncMode transfer_sync_mode = GetTransferSyncMode(non_blocking);

  switch (to_copy_type) {
    case CopyType::kCpuToTpu:
      if (src.sizes() != self_dest.sizes()) {
        // Broadcast CPU to TPU: copy to temp TPU then broadcast TPU to TPU.
        TT_ASSIGN_OR_THROW(at::Tensor temp_tpu,
                           MakeEmptyTensor(src.sizes(), self_dest.scalar_type(),
                                           self_dest.device()));
        TT_THROW_IF_ERROR(CopyCpuToTpu(src, temp_tpu, transfer_sync_mode));
        TT_THROW_IF_ERROR(CopyTpuToTpu(temp_tpu, self_dest));
      } else {
        TT_THROW_IF_ERROR(CopyCpuToTpu(src, self_dest, transfer_sync_mode));
      }
      break;
    case CopyType::kTpuToTpu:
      TT_THROW_IF_ERROR(CopyTpuToTpu(src, self_dest));
      break;
    case CopyType::kTpuToCpu:
      // CopyTpuToCpu handles broadcasting/dtype change via fallback CPU copy.
      TT_THROW_IF_ERROR(CopyTpuToCpu(src, self_dest, transfer_sync_mode));
      break;
  }

  // Also copy the gradient if it exists on the src.
  CopyTensorGrad(src, self_dest, to_copy_type, non_blocking);
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
              // NOTE: argument order is intentionally reversed.
              // copy_ uses "self" as the first argument to be consistent with
              // the Python method: `def copy_(self, src):`.
              // CopyTensor instead is a src -> dest copy, not as a method.
              CopyTensor(src, self, non_blocking);
              return self;
            });
}

}  // namespace torch_tpu
