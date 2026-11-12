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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_VALUE_HANDLE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_VALUE_HANDLE_H_

#include <utility>

// Forward declared rather than included so that MLIR headers do not leak into
// Kiln's public headers.
namespace mlir {  // NON_TORCH_TPU_OK=mlir::Value forward declaration
class Value;
}  // namespace mlir

namespace torch_tpu::kiln::internal {

// A handle to an underlying IR value, held by every Kiln handle type (`Scalar`,
// `Tile`, `Ref`, `Semaphore`, ...) in place of a raw pointer.
//
// A handle may be null, which indicates an error. The error itself is held by
// the thread-local scope (see `ReportError()` in error.h), so that a handle
// stays the size of a pointer.
class ValueHandle {
 public:
  // Defined out of line; see the `mlir::Value` forward declaration above.
  // CHECK-fails if `value` is null; use `Null()` for a null handle.
  explicit ValueHandle(mlir::Value value);

  // Returns a null handle, which indicates an error.
  static constexpr ValueHandle Null() { return ValueHandle(); }

  // Returns true if this handle is null, which indicates an error.
  constexpr bool is_null() const { return impl_ == nullptr; }

  // Returns the held value. CHECK-fails if this handle is null.
  mlir::Value value() const;

 private:
  constexpr ValueHandle() = default;

  // Opaque pointer of the held `mlir::Value`; null for a null handle.
  const void* impl_ = nullptr;
};

static_assert(sizeof(ValueHandle) == sizeof(void*));

// Returns true if any of `values`, which are Kiln handle types (`Scalar`,
// `Tile`, `Ref`, ...), is null, i.e. stands for the value of a failed
// operation. An operation given such a value must not use it (see `unwrap()`)
// and returns a null value in turn.
template <typename... Ts>
[[nodiscard]] bool AnyNull(const Ts&... values) {
  return (values.handle().is_null() || ...);
}

// Returns a null `T`, to return in place of a value once an error has been
// reported. `args` are the constructor arguments of `T` after the handle.
template <typename T, typename... Args>
T CreateNullValue(Args&&... args) {
  return T(ValueHandle::Null(), std::forward<Args>(args)...);
}

}  // namespace torch_tpu::kiln::internal

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_VALUE_HANDLE_H_
