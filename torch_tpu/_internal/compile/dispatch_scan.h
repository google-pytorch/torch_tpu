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
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/ops/scan_builder.h"

namespace torch_tpu {

// Creates a scan operation on TPU.
//
// By default (is_associative=false) the body is treated as non-associative and
// lowered to a StableHLO while loop. When is_associative=true the body is
// assumed associative and lowered to chlo.ScanOp for the native scan emitter
// (see is_associative below).
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
//   is_associative: When true, the scan is lowered to chlo.ScanOp
//       (is_associative=true) so the TPU compiler's native scan emitter
//       compiles it directly, instead of a StableHLO while loop. Only set this
//       when the step function is genuinely associative (e.g. a
//       torch.associative_scan combiner): a non-associative body would be
//       miscompiled by the associative emitter. The native emitter computes a
//       pure inclusive scan (out[0] = xs[0]); the carry inits are used for the
//       carry types but their values do not affect the per-position outputs.
//
// Returns:
//   A list of tensors containing the final carries followed by the stacked loop
//   outputs.
std::vector<at::Tensor> PyCreateScanOp(
    const std::vector<at::Tensor>& inits, const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<ContextedModule>& body_module,
    ScanDirection scan_direction, const std::vector<at::Tensor>& dummy_outputs,
    int64_t num_scan_inputs, bool is_associative = false);

}  // namespace torch_tpu

#endif  // TORCH_TPU_INTERNAL_COMPILE_DISPATCH_SCAN_H_
