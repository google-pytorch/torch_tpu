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

#include "torch_tpu/csrc/common/native_scan_support.h"

#include <atomic>

namespace torch_tpu {
namespace {

std::atomic<bool>& SupportedFlag() {
  static std::atomic<bool> supported(true);
  return supported;
}

}  // namespace

bool NativeScanEmitterSupported() {
  return SupportedFlag().load(std::memory_order_relaxed);
}

void SetNativeScanEmitterSupported(bool supported) {
  SupportedFlag().store(supported, std::memory_order_relaxed);
}

}  // namespace torch_tpu
