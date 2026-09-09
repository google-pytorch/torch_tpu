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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_AOT_ELEMENTWISE_COMPILE_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_AOT_ELEMENTWISE_COMPILE_H_

#include <string>

#include "absl/status/status.h"
#include "csrc/experimental/precompiled_kernels/bundle.pb.h"
#include "xla/pjrt/pjrt_compiler.h"

namespace torch_tpu {

// Append every supported op, dtype and rank for this target.
absl::Status AddKernels(xla::PjRtCompiler& compiler,
                        const xla::PjRtTopologyDescription& topology,
                        const std::string& dump_dir,
                        precompiled_kernels::proto::Bundle& bundle);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_AOT_ELEMENTWISE_COMPILE_H_
