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

#ifndef TORCH_TPU_INTERNAL_COMPILE_DISPATCH_SCAN_H_
#define TORCH_TPU_INTERNAL_COMPILE_DISPATCH_SCAN_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "torch_tpu/csrc/common/compilation.h"
#include "torch_tpu/csrc/ops/scan_builder.h"

namespace torch_tpu {

// Creates a loop-based scan operation on TPU.
//
// Args:
//   inits: A list of carry tensors representing the starting state of the loop.
//   inputs: A list of flat scannable sequence inputs.
//   body_module: The MLIR sub-graph representing the loop step function
//       ('combine_fn'), which takes
//       `(*carries, *scan_inputs, *additional_inputs)` and returns
//       `(*next_carries, *outputs)`.
//   scan_direction: The direction of scan execution (Forward or Reverse).
//   dummy_outputs: A list of flat zero-initialized dummy output tensors
//       representing the full stacked outputs across the entire sequence. Used
//       to statically infer the output shapes, dtypes, and memory requirements
//       of the loop accumulators.
//   num_scan_inputs: The number of active scanned inputs in the `inputs`
//       list. Any elements in `inputs` beyond this count are treated as
//       static inputs, not to be scanned over.
//
// Returns:
//   A list of tensors containing the final carries followed by the stacked loop
//   outputs.
std::vector<at::Tensor> PyCreateScanOp(
    const std::vector<at::Tensor>& inits, const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<ContextedModule>& body_module,
    ScanDirection scan_direction, const std::vector<at::Tensor>& dummy_outputs,
    int64_t num_scan_inputs);

}  // namespace torch_tpu

#endif  // TORCH_TPU_INTERNAL_COMPILE_DISPATCH_SCAN_H_
