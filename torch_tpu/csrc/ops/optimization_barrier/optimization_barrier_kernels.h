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

#ifndef TORCH_TPU_OPS_OPTIMIZATION_BARRIER_OPTIMIZATION_BARRIER_KERNELS_H_
#define TORCH_TPU_OPS_OPTIMIZATION_BARRIER_OPTIMIZATION_BARRIER_KERNELS_H_

#include <vector>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// TorchTpuOptimizationBarrier is a C++ kernel implementation for the custom
// PyTorch op torch_tpu.optimization_barrier. This operation wraps
// stablehlo.optimization_barrier, which serves as a barrier to prevent
// compiler optimizations, such as Common Subexpression Elimination (CSE),
// across it. See https://openxla.org/stablehlo/spec#optimization_barrier for
// additional details.
//
// Its primary use case is within the torch_tpu library to control
// optimizations like preventing CSE when activation checkpointing is enabled.
//
// This function is intended for internal use by the torch_tpu library and is
// not meant to be called directly by end-users.
//
// Parameters:
//   self: A list of input tensors.
//
// Returns:
//   A vector of tensors, which are the result of the optimization barrier
//   operation. The output tensors are semantically equivalent to the input
//   tensors.
std::vector<at::Tensor> TorchTpuOptimizationBarrier(at::TensorList self);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_OPTIMIZATION_BARRIER_OPTIMIZATION_BARRIER_KERNELS_H_
