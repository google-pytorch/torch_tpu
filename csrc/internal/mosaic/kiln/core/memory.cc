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
#include "csrc/internal/mosaic/kiln/core/specs.h"
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

mlir::Value AllocaSemaphoreValue(llvm::ArrayRef<int64_t> shape,
                                 mlir::Type elem_type,
                                 std::string_view fn_name) {
  for (int64_t dim : shape) {
    ABSL_CHECK_GT(dim, 0)  // CRASH_OK
        << fn_name << " requires static positive dimensions";
  }
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::MemRefType type =
      GetSemaphoreMemRefType(b.getContext(), shape, elem_type);
  return tpu::AllocaSemaphoreOp::create(b, type);
}

}  // namespace

Ref Alloca(const ScratchSpec& spec) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::MLIRContext* ctx = b.getContext();
  ABSL_CHECK(spec.space != MemorySpace::kHbm)  // CRASH_OK
      << "Cannot allocate HBM in a kernel";
  llvm::ArrayRef<int64_t> dims = spec.shape.dims();
  const DType dtype = spec.shape.dtype();
  for (int64_t dim : dims) {
    ABSL_CHECK_GT(dim, 0)  // CRASH_OK
        << "Alloca() requires static positive dimensions";
  }
  switch (spec.space) {
    case MemorySpace::kVmem: {
      ABSL_CHECK(!dims.empty())  // CRASH_OK
          << "VMEM Alloca() requires at least a 1D shape";
      ABSL_CHECK(!dtype.IsBool() &&  // CRASH_OK
                 dtype.GetIntOrFloatBitWidth() <= 32u)
          << "VMEM Alloca() does not support bool (i1) or 64-bit element types";
      mlir::Type elem_type = internal::ToMlirType(dtype, ctx);
      mlir::MemRefType type = GetVmemMemRefType(ctx, dims, elem_type);
      return internal::wrap<Ref>(mlir::memref::AllocaOp::create(b, type),
                                 dtype);
    }
    case MemorySpace::kSmem: {
      ABSL_CHECK_EQ(dtype.GetIntOrFloatBitWidth(), 32u)  // CRASH_OK
          << "SMEM Alloca() requires a 32-bit element type";
      mlir::Type elem_type = internal::ToMlirType(dtype, ctx);
      mlir::MemRefType type = GetSmemMemRefType(ctx, dims, elem_type);
      return internal::wrap<Ref>(mlir::memref::AllocaOp::create(b, type),
                                 dtype);
    }
    default:
      ABSL_CHECK(false)  // CRASH_OK
          << "Unsupported memory space for Alloca()";
      return internal::wrap<Ref>(mlir::Value(), dtype);
  }
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
