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

#ifndef TORCH_TPU_CSRC_OPS_SPARSE_COO_SPARSE_COO_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_SPARSE_COO_SPARSE_COO_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/Device.h"
#include "c10/core/Scalar.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/ArrayRef.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// Constructs a sparse COO tensor from indices and values tensors without
// delegating to at::native creation ops.
at::Tensor MakeSparseCooTensor(int64_t sparse_dim, int64_t dense_dim,
                               at::IntArrayRef sizes, const at::Tensor& indices,
                               const at::Tensor& values, bool is_coalesced);

// Arithmetic operations (add, sub, mul, neg) lowered to StableHLO.
at::Tensor& AtenAddOutSparse(const at::Tensor& self, const at::Tensor& other,
                             const c10::Scalar& alpha, at::Tensor& out);
at::Tensor AtenAddSparse(const at::Tensor& self, const at::Tensor& other,
                         const c10::Scalar& alpha);
at::Tensor& AtenAdd_Sparse(at::Tensor& self, const at::Tensor& other,
                           const c10::Scalar& alpha);

at::Tensor& AtenSubOutSparse(const at::Tensor& self, const at::Tensor& other,
                             const c10::Scalar& alpha, at::Tensor& out);
at::Tensor AtenSubSparse(const at::Tensor& self, const at::Tensor& other,
                         const c10::Scalar& alpha);
at::Tensor& AtenSub_Sparse(at::Tensor& self, const at::Tensor& other,
                           const c10::Scalar& alpha);

at::Tensor& AtenMulOutSparse(const at::Tensor& self, const at::Tensor& other,
                             at::Tensor& out);
at::Tensor AtenMulSparse(const at::Tensor& self, const at::Tensor& other);
at::Tensor& AtenMul_Sparse(at::Tensor& self, const at::Tensor& other);

at::Tensor& AtenNegOutSparse(const at::Tensor& self, at::Tensor& out);
at::Tensor AtenNegSparse(const at::Tensor& self);
at::Tensor& AtenNeg_Sparse(at::Tensor& self);

// Coalesce and conversion operations lowered to StableHLO.
at::Tensor AtenCoalesceSparse(const at::Tensor& self);
at::Tensor AtenToDenseSparse(const at::Tensor& self,
                             std::optional<c10::ScalarType> dtype,
                             std::optional<bool> masked_grad);

// Creation, copy, and resize operations using native TorchTPU buffers.
at::Tensor AtenEmptySparseSymInt(
    c10::SymIntArrayRef size, std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout, std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format);

at::Tensor AtenEmptyLikeSparseCoo(
    const at::Tensor& self, std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout, std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format);

at::Tensor AtenSparseCooTensorWithDims(int64_t sparse_dim, int64_t dense_dim,
                                       at::IntArrayRef size,
                                       std::optional<c10::ScalarType> dtype,
                                       std::optional<c10::Layout> layout,
                                       std::optional<c10::Device> device,
                                       std::optional<bool> pin_memory);

at::Tensor AtenSparseCooTensorWithDimsAndTensors(
    int64_t sparse_dim, int64_t dense_dim, c10::SymIntArrayRef size,
    const at::Tensor& indices, const at::Tensor& values,
    std::optional<c10::ScalarType> dtype, std::optional<c10::Layout> layout,
    std::optional<c10::Device> device, std::optional<bool> pin_memory,
    std::optional<bool> is_coalesced);

const at::Tensor& AtenSparseResize_(const at::Tensor& self,
                                    at::IntArrayRef size, int64_t sparse_dim,
                                    int64_t dense_dim);

const at::Tensor& AtenSparseResizeAndClear_(const at::Tensor& self,
                                            at::IntArrayRef size,
                                            int64_t sparse_dim,
                                            int64_t dense_dim);

at::Tensor& AtenZero_Sparse(at::Tensor& self);

at::Tensor AtenCloneSparse(const at::Tensor& self,
                           std::optional<c10::MemoryFormat> memory_format);

at::Tensor& AtenCopySparseToSparse_(at::Tensor& self, const at::Tensor& src,
                                    bool non_blocking);

at::Tensor& AtenCopy_Sparse(at::Tensor& self, const at::Tensor& src,
                            bool non_blocking);

at::Tensor AtenToCopySparse(const at::Tensor& self,
                            std::optional<c10::ScalarType> dtype,
                            std::optional<c10::Layout> layout,
                            std::optional<c10::Device> device,
                            std::optional<bool> pin_memory, bool non_blocking,
                            std::optional<c10::MemoryFormat> memory_format);

// Metadata accessors (direct C++ SparseTensorImpl getters).
int64_t AtenSparseDimSparse(const at::Tensor& self);
int64_t AtenDenseDimSparse(const at::Tensor& self);
int64_t AtenNnzSparse(const at::Tensor& self);
bool AtenIsCoalescedSparse(const at::Tensor& self);
at::Tensor& AtenCoalesced_Sparse(at::Tensor& self, bool coalesced);
at::Tensor AtenIndicesInternalSparse(const at::Tensor& self);
at::Tensor AtenValuesInternalSparse(const at::Tensor& self);
at::Tensor AtenIndicesSparse(const at::Tensor& self);
at::Tensor AtenValuesSparse(const at::Tensor& self);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SPARSE_COO_SPARSE_COO_ATEN_KERNELS_H_
