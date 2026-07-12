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
// WARNING: Eager dispatch contains a Fast Path (op_dispatcher_fast_path.cc)
// and a Slow Path (materialize_common.cc). If you modify shape propagation,
// data type casting, or dynamic shape validation logic in the slow path,
// you MUST update the corresponding logic in the fast path helpers to prevent
// correctness mismatches or silent compilation failures.

#include "torch_tpu/eager/op_dispatcher_fast_path.h"

#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/Tensor.h"
#include "absl/log/absl_log.h"
#include "torch_tpu/eager/call_site_cache.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {
namespace internal {

bool TryUnaryOpFastPath(const at::Tensor& self, at::Tensor& out,
                        CallSiteCache& cache) {
  if (!out.is_contiguous() || out.storage_offset() != 0) {
    return false;
  }
  if (::torch_tpu::GetEagerMode() != EagerMode::kDeferNever) {
    return false;
  }
  const bool can_cache = self.is_contiguous() && self.storage_offset() == 0;
  if (!can_cache) {
    return false;
  }
  auto self_buf = ::torch_tpu::GetBufferFast(self);
  if (!self_buf.has_value() || !self_buf->is_materializing()) {
    return false;
  }
  if (!self_buf->dynamic_dimensions().empty()) {
    return false;
  }
  ::torch_tpu::CompiledKernel compiled_kernel;
  ::torch_tpu::CallSiteOutputShapes output_shapes;
  if (!cache.Lookup(self, compiled_kernel, output_shapes)) {
    return false;
  }
  ABSL_VLOG(2)
      << "TryUnaryOpFastPath: CallSiteCache Lookup succeeded for tensor "
      << self.sizes() << " (" << (int)self.scalar_type() << ")";
  auto exec_or = compiled_kernel.fixed_shape_kernel.get();
  if (!exec_or.ok()) {
    ABSL_LOG(ERROR) << "Get executable failed: " << exec_or.status().message();
    return false;
  }
  auto exec = *exec_or;
  auto memory_kinds_or = exec->GetExecutable()->GetParameterMemoryKinds();
  int num_exec_args = -1;
  if (memory_kinds_or.ok() && !memory_kinds_or->empty()) {
    num_exec_args = memory_kinds_or->front().size();
  }
  ABSL_VLOG(2) << "TryUnaryOpFastPath: Retrieved executable expects "
               << num_exec_args << " arguments";
  auto outputs_or =
      ::torch_tpu::EnqueueExecutable(exec, {*self_buf}, {output_shapes[0]});
  if (!outputs_or.ok()) {
    ABSL_LOG(ERROR) << "EnqueueExecutable failed: "
                    << outputs_or.status().message();
    return false;
  }
  if (outputs_or->empty()) {
    return false;
  }
  const at::IntArrayRef shape = self.sizes();
  if (!::torch_tpu::ResizeTensorIfShapeDiffers(out, shape).ok()) {
    return false;
  }
  if (!::torch_tpu::AssignBufferToAtTensor(std::move((*outputs_or)[0]), out)
           .ok()) {
    return false;
  }
  return true;
}

bool TryBinaryOpFastPath(const at::Tensor& self, const at::Tensor& other,
                         at::Tensor& out, CallSiteCache& cache) {
  if (!out.is_contiguous() || out.storage_offset() != 0) {
    return false;
  }
  if (::torch_tpu::GetEagerMode() != EagerMode::kDeferNever) {
    return false;
  }
  if (!(self.is_contiguous() && self.storage_offset() == 0 &&
        other.is_contiguous() && other.storage_offset() == 0)) {
    return false;
  }
  auto self_buf = ::torch_tpu::GetBufferFast(self);
  auto other_buf = ::torch_tpu::GetBufferFast(other);
  if (!self_buf.has_value() || !self_buf->is_materializing() ||
      !other_buf.has_value() || !other_buf->is_materializing()) {
    return false;
  }
  if (!self_buf->dynamic_dimensions().empty() ||
      !other_buf->dynamic_dimensions().empty()) {
    return false;
  }
  CompiledKernel compiled_kernel;
  CallSiteOutputShapes output_shapes;
  if (!cache.Lookup(self, other, compiled_kernel, output_shapes)) {
    return false;
  }
  auto exec_or = compiled_kernel.fixed_shape_kernel.get();
  if (!exec_or.ok()) {
    ABSL_LOG(ERROR) << "Get executable failed: " << exec_or.status().message();
    return false;
  }
  auto exec = *exec_or;
  auto outputs_or = ::torch_tpu::EnqueueExecutable(
      exec, {*self_buf, *other_buf}, {output_shapes[0]});
  if (!outputs_or.ok()) {
    ABSL_LOG(ERROR) << "EnqueueExecutable failed: "
                    << outputs_or.status().message();
    return false;
  }
  if (outputs_or->empty()) {
    return false;
  }
  if (!::torch_tpu::ResizeTensorIfShapeDiffers(out,
                                               output_shapes[0].dimensions())
           .ok()) {
    return false;
  }
  if (!::torch_tpu::AssignBufferToAtTensor(std::move((*outputs_or)[0]), out)
           .ok()) {
    return false;
  }
  return true;
}

}  // namespace internal
}  // namespace torch_tpu
