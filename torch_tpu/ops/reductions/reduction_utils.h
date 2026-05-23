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

#ifndef TORCH_TPU_OPS_REDUCTIONS_REDUCTION_UTILS_H_
#define TORCH_TPU_OPS_REDUCTIONS_REDUCTION_UTILS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/OptionalArrayRef.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

// Checks that the provided dtype is either floating point or complex.
// Many reduction ops only operate on these types.
absl::Status CheckFloatOrComplex(c10::ScalarType scalar_type);

// Converts provided dimensions to canonical dimensions.
// A negative value, -i, becomes rank - i.
// If no dimensions are provided then return all: [0, 1, ..., rank - 1].
// Values outside [-rank, rank - 1] are invalid.
// Repeated values are invalid.
// For rank 0, either -1 or 0 is a valid dimension.
absl::StatusOr<Dimensions> CanonicalizeDims(const at::Tensor& tensor,
                                            c10::OptionalArrayRef<int64_t> dim);

// Returns the size array after a reduction is applied.
// Reduction dimensions must be canonicalized.
Dimensions GetSizesAfterReduction(at::IntArrayRef self_size,
                                  ReductionMode reduction_mode,
                                  at::IntArrayRef canonicalized_dims);

// Returns the input:output ratio for the provided reduction.
// When dim is provided, the ratio is the product of the sizes along the
// provided dimensions. Otherwise the reduction factor is the number of elements
// in the input tensor.
absl::StatusOr<int64_t> GetReductionFactor(const at::Tensor& input_tensor,
                                           c10::OptionalArrayRef<int64_t> dim);

// Reduces the provided tensor by applying sum.
// This foundational operation is part of many more complex reductions.
//
// @param self The input tensor.
// @param dim The dimension or dimensions to reduce.
// @param keep_dim Whether or not to retain the reduced dimensions with size 1.
// @param out_dtype The desired data type of returned tensor.
absl::StatusOr<at::Tensor> ApplySumReduction(const at::Tensor& self,
                                             c10::OptionalArrayRef<int64_t> dim,
                                             ReductionMode reduction_mode,
                                             c10::ScalarType out_dtype);

// Like ApplySumReduction but results are written to the provided `out` tensor.
absl::Status ApplySumReductionOut(const at::Tensor& self, at::Tensor& out,
                                  c10::OptionalArrayRef<int64_t> dim,
                                  ReductionMode reduction_mode,
                                  c10::ScalarType out_dtype);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_REDUCTIONS_REDUCTION_UTILS_H_
