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

#ifndef TORCH_TPU_DISTRIBUTED_HANDSHAKE_H_
#define TORCH_TPU_DISTRIBUTED_HANDSHAKE_H_

namespace torch_tpu {

// Options for TORCH_TPU_INTERNAL_HANDSHAKE_STAGE.
enum class HandshakeStage {
  kOff = 0,
  kCompileStage = 1,
  kDispatchStage = 2,
};

// Returns the handshake stage mode configured via
// TORCH_TPU_INTERNAL_HANDSHAKE_STAGE.
HandshakeStage GetHandshakeStageEnvVarOnce();

// Returns the handshake port.
int GetHandshakePortEnvVarOnce();

}  // namespace torch_tpu

#endif  // TORCH_TPU_DISTRIBUTED_HANDSHAKE_H_
