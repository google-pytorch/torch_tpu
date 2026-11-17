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

#include "csrc/ops/record_stream/record_stream_aten_kernels.h"

#include "ATen/core/TensorBody.h"
#include "c10/core/Stream.h"
#include "csrc/common/cache_key.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"

namespace torch_tpu {

void AtenRecordStream(at::Tensor& self, c10::Stream stream) {
  TT_KERNEL(OpName::kRecordStream, _,
            (self, IgnoreInCacheKey(stream, "Doesn't affect SHLO")),
            {
                // The PJRT buffer stays alive until every pending use on any
                // stream retires, so there is nothing to record on TPU.
            });
}

}  // namespace torch_tpu
