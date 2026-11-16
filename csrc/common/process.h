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

#ifndef TORCH_TPU_CSRC_COMMON_PROCESS_H_
#define TORCH_TPU_CSRC_COMMON_PROCESS_H_

#include <sys/types.h>

namespace torch_tpu {

// A process "has a TorchTPU ancestor process" if one of its ancestor processes
// loaded TorchTPU and the process inherited that ancestor's environment.
// Launchers that don't load TorchTPU are invisible to this mechanism.
// For example:
//
// - With `torch.multiprocessing.spawn`, `multiprocessing`, or
//   `ProcessPoolExecutor` called from a process that imported `torch_tpu`, the
//   workers have a TorchTPU ancestor process and the calling process doesn't.
// - With `torchrun`, the launcher imports `torch`, which loads TorchTPU through
//   PyTorch's device backend autoloading, so the workers have a TorchTPU
//   ancestor process. With `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, the launcher
//   doesn't load TorchTPU and no worker has a TorchTPU ancestor process.
// - With launchers that don't run TorchTPU (shell scripts, `mpirun`, `srun`,
//   Kubernetes, Borg, etc.), no worker has a TorchTPU ancestor process.
// - A process launched with an environment that doesn't include its parent's
//   environment (e.g. `subprocess.Popen(..., env={...})`) is treated as having
//   no TorchTPU ancestor process, even if its parent loaded TorchTPU.
// - An independent TorchTPU program launched as a subprocess of a TorchTPU
//   process with the inherited environment (e.g. a benchmark harness running
//   each benchmark in a subprocess) has a TorchTPU ancestor process.
//
// Therefore, this is only a best-effort heuristic. Use it only where a wrong
// answer is harmless, e.g. to deduplicate informational log messages in
// multi-process programs. Never use it for correctness, such as to decide
// which process initializes or owns a shared resource; use the distributed
// rank for that instead.

// Returns true if the current process has a TorchTPU ancestor process (see
// above).
//
// Does not depend on or trigger PjRt initialization.
[[nodiscard]] bool HasTorchTpuAncestorProcess();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_PROCESS_H_
