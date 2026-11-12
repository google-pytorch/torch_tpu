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

#ifndef TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_HELPERS_H_
#define TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_HELPERS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/TensorBase.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Batch dim (un)flattening helpers. Shapes are static sizes, or upper bounds
// for bounded dynamic dimensions; bounded dimensions are propagated.
mlir::MlirOp flatten_batch_dims(mlir::MlirOp mlir_op, int batch_size,
                                int num_batch_dims);
mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  absl::Span<const int64_t> shape);
mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  mlir::MlirOp mlir_op_with_target_dims);
int get_batch_size(absl::Span<const int64_t> shape);

mlir::MlirOp GetScaleDefaulted(mlir::MlirBuilder& builder,
                               std::optional<double> maybe_scale,
                               int64_t head_dim,
                               mlir::Type element_type = nullptr);

// The helpers below return the strides that the PyTorch meta functions of the
// SDPA ops allocate, so that the strides of eager results match the fake tensor
// strides that torch.compile traces with. Views that are valid in compiled mode
// (e.g. merging heads) are then also valid in eager mode. The strides are
// always dense, so results never inherit gaps from sliced inputs, which avoids
// memory bloat and slice-by-slice writes (dynamic-update-slice).

// Returns the strides of `torch.empty_like(tensor)` with the default
// `torch.preserve_format`: the strides of `tensor` if it is non-overlapping and
// dense, otherwise `at::infer_dense_strides`, which keeps the stride order,
// removes gaps and leaves zero-stride (expanded) dims in place. The SDPA
// backward meta functions allocate the gradients this way.
Strides EmptyLikeStrides(const at::TensorBase& tensor);

// Returns the strides of `alloc_with_matching_layout(query, result_sizes)` in
// torch/_meta_registrations.py, which the SDPA forward meta functions use to
// allocate the attention output: `EmptyLikeStrides(query)` if `result_sizes`
// equals the query sizes, otherwise dense strides that follow the query stride
// order, with zero-stride dims outermost and ties broken by making the lower
// dim innermost. The tie and zero-stride ordering follows PyTorch 2.15
// (PR #193039); older versions order these dims differently.
Strides MatchingLayoutStrides(const at::TensorBase& query,
                              absl::Span<const int64_t> result_sizes);

// Unwraps GQA `repeat_kv` head expansion (`BroadcastInDim` + `Reshape`
// repeating `H_kv` heads into `num_head_q`) for rank at least 4 `kv_op`
// (`[..., H_kv, S, D]`, where the producing broadcast operation may originate
// from a rank 3 or rank 4 source) back to `[..., H_kv, S, D]` when
// `num_head_q % H_kv == 0`. Returns `nullopt` if `kv_op` is not a pure
// head-group repetition.
std::optional<mlir::MlirOp> UnwrapRepeatedKvHead(mlir::MlirOp kv_op,
                                                 int64_t num_head_q);

// Returns true if `bias_op` evaluates to an all-zero additive attention bias
// (`0.0` at every index in its static domain).
bool IsAllZeroAttnBias(mlir::MlirOp bias_op);

// Returns true if `bias_op` is an explicit lower-triangular causal attention
// mask (`0.0` on `i >= j` and `-inf` / min finite float on `i < j`) over
// `[..., S, S]` with `q_seq_len == kv_seq_len`.
bool IsPureCausalAttnBias(mlir::MlirOp bias_op, int64_t q_seq_len,
                          int64_t kv_seq_len);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_HELPERS_H_
