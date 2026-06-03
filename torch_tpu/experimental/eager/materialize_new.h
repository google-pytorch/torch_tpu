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

#ifndef TORCH_TPU_EXPERIMENTAL_EAGER_MATERIALIZE_NEW_H_
#define TORCH_TPU_EXPERIMENTAL_EAGER_MATERIALIZE_NEW_H_

#include "absl/types/span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/structured_log_buffer.h"

namespace torch_tpu {

void OnNewOpDispatch(const SharedDeviceBufferList& device_buffer_list);

void MaterializeImplNew(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    MaterializationReason reason);

void BlockOnPendingMaterializations();

void ResetNewMaterializationState();

void ShutDownNewMaterializationState();

}  // namespace torch_tpu

#endif  // TORCH_TPU_EXPERIMENTAL_EAGER_MATERIALIZE_NEW_H_
