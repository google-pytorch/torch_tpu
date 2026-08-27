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

#ifndef TORCH_TPU_INTERNAL_PROFILER_XPROF_CALLBACK_HANDLER_H_
#define TORCH_TPU_INTERNAL_PROFILER_XPROF_CALLBACK_HANDLER_H_

#include "ATen/record_function.h"

namespace torch_tpu {

// RAII class to manage the lifetime of PyTorch RecordFunction callbacks.
// It registers callbacks on construction and unregisters them on destruction.
//
// Note on Thread-Safety:
// While PyTorch documentation states that `at::addGlobalCallback` should only
// be called during program initialization, the actual implementation in
// `at::GlobalCallbackManager` uses an internal mutex (`update_mutex_`) and an
// atomic versioning/snapshot mechanism. Therefore, dynamic registration and
// unregistration via this handler is safe in practice.
class XProfCallbackHandler {
 public:
  XProfCallbackHandler();
  ~XProfCallbackHandler() {
    if (handle_ != at::INVALID_CALLBACK_HANDLE) at::removeCallback(handle_);
  }
  XProfCallbackHandler(const XProfCallbackHandler&) = delete;
  XProfCallbackHandler& operator=(const XProfCallbackHandler&) = delete;
  XProfCallbackHandler(XProfCallbackHandler&&) = delete;
  XProfCallbackHandler& operator=(XProfCallbackHandler&&) = delete;

  // Registers the global callback handler. Should be called explicitly to avoid
  // Static Initialization Order Fiasco (SIOF) risks.
  static void Register();

 private:
  const at::CallbackHandle handle_ = at::INVALID_CALLBACK_HANDLE;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_INTERNAL_PROFILER_XPROF_CALLBACK_HANDLER_H_
