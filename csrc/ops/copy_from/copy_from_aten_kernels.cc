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

#include "ATen/ExpandUtils.h"
#include "ATen/SparseTensorImpl.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/graph_capture_state.h"
#include "csrc/eager/host_allocator.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/cpu_to_tpu.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/copy_from/tpu_to_tpu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/unary.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
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

// Returns true if `tensor` is a CPU tensor backed by TPU pinned memory.
bool IsPinnedCpuTensor(const at::Tensor& tensor) {
  // Query IsTpuPinnedPtr() directly instead of tensor.is_pinned(), because
  // PyTorch's Conjugate/Negative fallbacks materialize unpinned clones first.
  return tensor.is_cpu() && tensor.has_storage() &&
         IsTpuPinnedPtr(tensor.storage().data_ptr().get());
}

// Returns the error of a copy between CPU and TPU tensors that a graph capture
// cannot record because the CPU memory that the copy transfers is not pinned.
absl::Status UnpinnedCopyWhileCapturingGraphError() {
  return TT_ERROR(error::kFailedPrecondition)
         << "cannot copy between CPU and TPU tensors while capturing a graph "
            "unless the CPU tensor is pinned; use tensor.pin_memory() or "
            "allocate the tensor with pin_memory=True";
}

// Copies `src`, a CPU tensor, to `dest`, a TPU tensor, while capturing a graph
// on the current thread.
//
// Like a CUDA graph, the graph copies the contents that `src` has at replay
// time, which requires `src` to be pinned. Like on CUDA, the copy transfers a
// temporary CPU tensor, which is not pinned, instead of `src` if `src` is not
// contiguous once broadcast to `dest`, or if the copy blocks and converts the
// dtype, as blocking copies convert dtypes on the host. A blocking copy waits
// for the transfer, which invalidates the capture.
absl::Status CopyCpuToTpuWhileCapturingGraph(const at::Tensor& src,
                                             const at::Tensor& dest,
                                             const bool non_blocking) {
  const at::Tensor broadcast_src = src.expand(dest.sizes());
  if (!IsPinnedCpuTensor(src) || !broadcast_src.is_contiguous() ||
      (!non_blocking && src.scalar_type() != dest.scalar_type())) {
    return UnpinnedCopyWhileCapturingGraphError();
  }
  if (!non_blocking) {
    return InvalidateCurrentThreadGraphCapture(
        TT_ERROR(error::kFailedPrecondition)
        << "cannot copy a CPU tensor to a TPU tensor synchronously while "
           "capturing a graph; use non_blocking=True");
  }
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef host_input,
      CreateCapturedHostInput(broadcast_src, dest.device().index()));
  // HostToDeviceTransfer reads raw bytes without applying PyTorch's lazy
  // conj/neg bits, so resolve them directly via StableHLO rather than
  // calling TT_KERNEL unary wrappers inside AtenCopy_.
  if (broadcast_src.is_conj() || broadcast_src.is_neg()) {
    const bool apply_conj = broadcast_src.is_conj();
    const bool apply_neg = broadcast_src.is_neg();
    Shape resolved_shape(CopyIntVector(host_input.dimensions()),
                         host_input.element_type());
    auto op_builder = [apply_conj, apply_neg](mlir::MlirBuilder&,
                                              absl::Span<mlir::MlirOp> inputs)
        -> absl::StatusOr<DynamicMlirOpResults> {
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal builder invariant.
          inputs.size() == 1, error::kInternal)
          << "expected 1 input, got " << inputs.size();
      mlir::MlirOp result = inputs[0];
      if (apply_conj) {
        TT_ASSIGN_OR_RETURN(result, BuildConjPhysicalShlo(result));
      }
      if (apply_neg) {
        TT_ASSIGN_OR_RETURN(result, BuildNegShlo(result));
      }
      return DynamicMlirOpResults{result};
    };
    OpParamCacheKeys cache_keys = OpParamCacheKeys::Empty();
    TT_RETURN_IF_ERROR(cache_keys.SetParam("apply_conj", apply_conj));
    TT_RETURN_IF_ERROR(cache_keys.SetParam("apply_neg", apply_neg));
    TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> resolved,
                        internal::CreateDeferredDeviceBufferList({
                            .op_name = OpName::kCopy_,
                            .op_builder = std::move(op_builder),
                            .op_param_cache_keys = std::move(cache_keys),
                            .inputs = {std::move(host_input)},
                            .output_shapes = {std::move(resolved_shape)},
                        }));
    host_input = std::move(resolved[0]);
  }
  const at::Tensor tpu_src =
      MakeTensor(std::move(host_input), dest.device().index());
  // Like on CUDA, the device converts the dtype of non-blocking copies.
  return CopyTpuToTpu(tpu_src, dest);
}

// Returns the error of copying `src`, a TPU tensor, to `dest`, a CPU tensor,
// while capturing a graph on the current thread.
//
// Like on CUDA, the copy fails if the CPU memory that it transfers is not
// pinned, which is the case if `dest` is not pinned or not contiguous, or if
// the copy blocks and converts the dtype, as blocking copies convert dtypes on
// the host. A blocking copy waits for the transfer, which invalidates the
// capture. Unlike a CUDA graph, a graph cannot copy to a pinned CPU tensor.
absl::Status CopyTpuToCpuWhileCapturingGraphError(const at::Tensor& src,
                                                  const at::Tensor& dest,
                                                  const bool non_blocking) {
  (void)at::inferExpandGeometry(  // VOID_CAST_OK=validates broadcasting
      src.sizes(), src.strides(), dest.sizes());
  if (!IsPinnedCpuTensor(dest) || !dest.is_contiguous() ||
      (!non_blocking && src.scalar_type() != dest.scalar_type())) {
    return UnpinnedCopyWhileCapturingGraphError();
  }
  if (non_blocking) {
    return TT_ERROR(error::kFailedPrecondition)
           << "copying a TPU tensor to a pinned CPU tensor without blocking "
              "while capturing a graph is not supported";
  }
  return InvalidateCurrentThreadGraphCapture(
      TT_ERROR(error::kFailedPrecondition)
      << "cannot copy a TPU tensor to a CPU tensor synchronously while "
         "capturing a graph");
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

  TT_THROW_IF_ERROR(
      ValidateShellDtypeCast(src.scalar_type(), self_dest.scalar_type()));

  TT_ASSIGN_OR_THROW(CopyType to_copy_type, GetCopyType(src, self_dest));
  // Tensors copied to the device while tracing an FX graph, such as the body of
  // `associative_scan`, are constants of that subgraph, not host inputs of an
  // enclosing graph capture.
  const bool capturing = IsCurrentThreadCapturingGraph() &&
                         GetEagerMode() != EagerMode::kInternalCompileFxGraph;
  if (capturing && self_dest.numel() == 0 &&
      (to_copy_type == CopyType::kCpuToTpu ||
       to_copy_type == CopyType::kTpuToCpu)) {
    // Like on CUDA, copying no elements does not access the device, even while
    // capturing a graph, once broadcastability is validated.
    (void)at::inferExpandGeometry(  // VOID_CAST_OK=validates broadcasting
        src.sizes(), src.strides(), self_dest.sizes());
    if (to_copy_type == CopyType::kTpuToCpu && !self_dest.has_storage()) {
      at::Tensor empty_dest = at::empty(self_dest.sizes(), self_dest.options());
      self_dest.unsafeGetTensorImpl()->set_storage_keep_dtype(
          empty_dest.storage());
    }
    CopyTensorGrad(src, self_dest, to_copy_type, non_blocking);
    return self_dest;
  }
  const TransferSyncMode transfer_sync_mode = GetTransferSyncMode(non_blocking);

  switch (to_copy_type) {
    case CopyType::kCpuToTpu:
      if (capturing) {
        TT_THROW_IF_ERROR(
            CopyCpuToTpuWhileCapturingGraph(src, self_dest, non_blocking));
      } else {
        TT_THROW_IF_ERROR(CopyCpuToTpu(src, self_dest, transfer_sync_mode));
      }
      break;
    case CopyType::kTpuToTpu:
      TT_THROW_IF_ERROR(CopyTpuToTpu(src, self_dest));
      break;
    case CopyType::kTpuToCpu:
      if (capturing) {
        TT_THROW_IF_ERROR(
            CopyTpuToCpuWhileCapturingGraphError(src, self_dest, non_blocking));
      } else {
        // CopyTpuToCpu handles broadcasting/dtype change via fallback CPU copy.
        TT_THROW_IF_ERROR(CopyTpuToCpu(src, self_dest, transfer_sync_mode));
      }
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
    if (IsCurrentThreadCapturingGraph()) {
      // Like on CUDA, reading the value waits for the device, which
      // invalidates the capture.
      TT_THROW_IF_ERROR(InvalidateCurrentThreadGraphCapture(
          TT_ERROR(error::kFailedPrecondition)
          << "cannot read the value of a tensor while capturing a graph"));
    }
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
