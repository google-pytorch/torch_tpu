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

#ifndef TORCH_TPU_COMMON_NATIVE_SCAN_SUPPORT_H_
#define TORCH_TPU_COMMON_NATIVE_SCAN_SUPPORT_H_

namespace torch_tpu {

// Whether the TPU compiler (libtpu) is new enough to compile the native scan
// emitter, i.e. chlo.ScanOp with is_associative=true. Defaults to true.
//
// OSS builds pin a specific libtpu wheel that can predate the native scan
// emitter, in which case a cumulative op that emits chlo.ScanOp fails to
// compile. torch_tpu sets this from Python at import time based on the
// installed libtpu version (see torch_tpu/_internal/native_scan.py). When
// false, the scan builder falls back to the StableHLO while-loop lowering,
// which every supported libtpu can compile. Internal builds always compile
// from head, so this stays true.
bool NativeScanEmitterSupported();

// Sets the value returned by NativeScanEmitterSupported(). Thread-safe. Called
// once from Python during torch_tpu initialization.
void SetNativeScanEmitterSupported(bool supported);

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_NATIVE_SCAN_SUPPORT_H_
