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

// Forward declared rather than included so that MLIR headers do not leak into
// Kiln's public headers.
namespace mlir {  // NON_TORCH_TPU_OK=mlir::Value forward declaration
class Value;
}  // namespace mlir

namespace torch_tpu::kiln::internal {

// A handle to an underlying IR value, held by every Kiln handle type (`Scalar`,
// `Tile`, `Ref`, `Semaphore`, ...) in place of a raw pointer.
class ValueHandle {
 public:
  // Defined out of line; see the `mlir::Value` forward declaration above.
  explicit ValueHandle(mlir::Value value);
  mlir::Value value() const;

 private:
  // Opaque pointer of the held `mlir::Value`.
  const void* impl_;
};

static_assert(sizeof(ValueHandle) == sizeof(void*));

}  // namespace torch_tpu::kiln::internal

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_VALUE_HANDLE_H_
