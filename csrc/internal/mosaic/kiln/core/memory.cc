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

#include "csrc/internal/mosaic/kiln/core/memory.h"

#include <cstdint>
#include <string_view>

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/semaphore.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace torch_tpu::kiln {
namespace {

namespace tpu = mlir::tpu;

void CheckStaticPositiveDims(llvm::ArrayRef<int64_t> dims,
                             std::string_view fn_name) {
  for (int64_t dim : dims) {
    ABSL_CHECK_GT(dim, 0)  // CRASH_OK
        << fn_name << " requires static positive dimensions";
  }
}

mlir::Value AllocaSemaphoreValue(llvm::ArrayRef<int64_t> shape,
                                 mlir::Type elem_type,
                                 std::string_view fn_name) {
  CheckStaticPositiveDims(shape, fn_name);
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::MemRefType type =
      GetSemaphoreMemRefType(b.getContext(), shape, elem_type);
  return tpu::AllocaSemaphoreOp::create(b, type);
}

// Emits a `memref.alloca` of `shape`, with the memref type built by
// `get_memref_type` (e.g. `GetVmemMemRefType`) in the memory space of `RefT`.
template <typename RefT>
RefT EmitAlloca(const Shape& shape, std::string_view fn_name,
                mlir::MemRefType (*get_memref_type)(mlir::MLIRContext*,
                                                    llvm::ArrayRef<int64_t>,
                                                    mlir::Type)) {
  CheckStaticPositiveDims(shape.dims(), fn_name);
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::MLIRContext* ctx = b.getContext();
  const mlir::MemRefType type = get_memref_type(
      ctx, shape.dims(), internal::ToMlirType(shape.dtype(), ctx));
  return internal::wrap<RefT>(mlir::memref::AllocaOp::create(b, type),
                              shape.dtype());
}

}  // namespace

VmemRef AllocaVmem(const Shape& shape) {
  const DType dtype = shape.dtype();
  ABSL_CHECK(!shape.dims().empty())  // CRASH_OK
      << "AllocaVmem() requires at least a 1D shape";
  ABSL_CHECK(!dtype.IsBool() &&  // CRASH_OK
             dtype.GetIntOrFloatBitWidth() <= 32u)
      << "AllocaVmem() does not support bool (i1) or 64-bit element types";
  return EmitAlloca<VmemRef>(shape, "AllocaVmem()", GetVmemMemRefType);
}

SmemRef AllocaSmem(const Shape& shape) {
  ABSL_CHECK_EQ(shape.dtype().GetIntOrFloatBitWidth(), 32u)  // CRASH_OK
      << "AllocaSmem() requires a 32-bit element type";
  return EmitAlloca<SmemRef>(shape, "AllocaSmem()", GetSmemMemRefType);
}

Semaphore AllocaDmaSemaphore() {
  mlir::MLIRContext* ctx = internal::CurrentBuilder().getContext();
  return internal::wrap<Semaphore>(AllocaSemaphoreValue(
      {}, tpu::DMASemaphoreType::get(ctx), "AllocaDmaSemaphore()"));
}

SemaphoreArray AllocaDmaSemaphore(Dimensions shape) {
  ABSL_CHECK(!shape.empty())  // CRASH_OK
      << "AllocaDmaSemaphore(shape) requires a non-empty shape";
  mlir::MLIRContext* ctx = internal::CurrentBuilder().getContext();
  return internal::wrap<SemaphoreArray>(AllocaSemaphoreValue(
      shape, tpu::DMASemaphoreType::get(ctx), "AllocaDmaSemaphore(shape)"));
}

Semaphore AllocaSemaphore() {
  mlir::MLIRContext* ctx = internal::CurrentBuilder().getContext();
  return internal::wrap<Semaphore>(AllocaSemaphoreValue(
      {}, tpu::SemaphoreType::get(ctx), "AllocaSemaphore()"));
}

SemaphoreArray AllocaSemaphore(Dimensions shape) {
  ABSL_CHECK(!shape.empty())  // CRASH_OK
      << "AllocaSemaphore(shape) requires a non-empty shape";
  mlir::MLIRContext* ctx = internal::CurrentBuilder().getContext();
  return internal::wrap<SemaphoreArray>(AllocaSemaphoreValue(
      shape, tpu::SemaphoreType::get(ctx), "AllocaSemaphore(shape)"));
}

Semaphore GetBarrierSemaphore() {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::MLIRContext* ctx = b.getContext();
  mlir::MemRefType type =
      GetSemaphoreMemRefType(ctx, {}, tpu::SemaphoreType::get(ctx));
  return internal::wrap<Semaphore>(tpu::GetBarrierSemaphoreOp::create(b, type));
}

}  // namespace torch_tpu::kiln
