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

#ifndef TORCH_TPU_CSRC_COMMON_WARNINGS_H_
#define TORCH_TPU_CSRC_COMMON_WARNINGS_H_

#include <string_view>

#include "c10/util/Exception.h"

namespace torch_tpu {

// Canonical warning categories that map directly to subclasses of
// `torch.tpu.exceptions.TorchTpuWarning`.
//
// Enumerator names match their corresponding Python warning class names
// exactly.
enum class WarningCategory {
  kTorchTpuWarning,
  kExperimentalApiWarning,
  kDeprecatedApiWarning,
  kPerformanceWarning,
  kCorrectnessWarning,
};

// Returns the hidden delimiter token associated with the given warning
// category.
//
// We embed ASCII Record Separator (`\x1E`) control characters as the token
// delimiter (i.e., `\x1E[TT_WARN:<ClassName>]\x1E`) at the beginning of the
// warning message passed to `TORCH_WARN` / `TORCH_WARN_ONCE`.
//
// Why `\x1E` (ASCII Record Separator):
// 1. `\x1E` is a standard non-printable ASCII control character that never
//    appears in human-readable or compiler-generated warning text, guaranteeing
//    zero accidental collisions with user strings.
// 2. It is completely safe and valid inside C++ string literals, `std::string`,
//    PyTorch's internal `c10::Warning` message payloads, and Python strings.
// 3. When PyTorch flushes warnings to Python's `warnings.warn_explicit`, the
//    TorchTPU warning bridge intercepts the message, detects this prefix token,
//    maps the extracted `<ClassName>` to the corresponding subclass of
//    `torch.tpu.exceptions.TorchTpuWarning`, and completely strips the
//    delimiter token so that the user and loggers see only the clean, original
//    message.
constexpr std::string_view GetWarningToken(WarningCategory category) {
  switch (category) {
    case WarningCategory::kExperimentalApiWarning:
      return "\x1E[TT_WARN:ExperimentalApiWarning]\x1E";
    case WarningCategory::kDeprecatedApiWarning:
      return "\x1E[TT_WARN:DeprecatedApiWarning]\x1E";
    case WarningCategory::kPerformanceWarning:
      return "\x1E[TT_WARN:PerformanceWarning]\x1E";
    case WarningCategory::kCorrectnessWarning:
      return "\x1E[TT_WARN:CorrectnessWarning]\x1E";
    case WarningCategory::kTorchTpuWarning:
      return "\x1E[TT_WARN:TorchTpuWarning]\x1E";
  }
  return "\x1E[TT_WARN:TorchTpuWarning]\x1E";
}

}  // namespace torch_tpu

// Direct wrappers over public PyTorch warning macros that embed the category
// token at the front of the message.
//
// How this works:
// 1. The `category` argument (a `::torch_tpu::WarningCategory` enum value)
//    selects a prefix token via `GetWarningToken(category)`.
// 2. `TORCH_WARN` / `TORCH_WARN_ONCE` prepends this token to the variadic
//    message arguments.
// 3. PyTorch's `c10::WarningHandler` propagates the message into Python's
//    `warnings` module (under `warnings._showwarnmsg`).
// 4. `torch_tpu._internal.warnings_bridge` intercepts the warning, parses the
//    token prefix, strips it from the message, and remaps the warning category
//    to the corresponding `TorchTpuWarning` subclass before showing it.
#define TT_WARN(category, ...)                          \
  TORCH_WARN(/* TORCH_WARN_OK=implementating TT_WARN */ \
             ::torch_tpu::GetWarningToken(category), __VA_ARGS__)

#define TT_WARN_ONCE(category, ...)                               \
  TORCH_WARN_ONCE(/* TORCH_WARN_OK=implementating TT_WARN_ONCE */ \
                  ::torch_tpu::GetWarningToken(category), __VA_ARGS__)

#endif  // TORCH_TPU_CSRC_COMMON_WARNINGS_H_
