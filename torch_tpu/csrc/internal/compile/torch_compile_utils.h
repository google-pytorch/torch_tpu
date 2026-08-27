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

#ifndef TORCH_TPU_INTERNAL_COMPILE_TORCH_COMPILE_UTILS_H_
#define TORCH_TPU_INTERNAL_COMPILE_TORCH_COMPILE_UTILS_H_

namespace torch_tpu {

// Returns whether to materialize collective tensors.
//
// Compiling a graph with collective ops can cause deadlocks on TPU if there are
// slight graph differences between ranks (e.g. from "if rank == 0: ..."). We
// avoid this by triggering a graph break for collective ops (materializing
// them). This behavior can be disabled by setting the environment variable
// `TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS` to `"false"` or `"0"`.
// This is useful for SPMD workloads where graph differences between ranks are
// not expected.
//
// Default is true.
[[nodiscard]] bool PyGetMaterializeCollectiveTensorsEnvVarOnce();

}  // namespace torch_tpu

#endif  // TORCH_TPU_INTERNAL_COMPILE_TORCH_COMPILE_UTILS_H_
