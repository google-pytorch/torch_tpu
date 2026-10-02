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

// Categories of the warnings TorchTPU emits, from both C++ and Python.
//
// TorchTPU emits its warnings with Python's built-in warning categories
// (`DeprecationWarning` for `kDeprecatedApi`, `UserWarning` for everything
// else) and tells them apart by a message prefix (`GetWarningPrefix`). See
// `torch_tpu/_internal/warnings.py` for the category hierarchy, the prefix
// design and its rationale, and how to filter by prefix. Python gets the
// prefixes from here.
//
// Enumerator names match the names of the categories in the hierarchy.
enum class WarningCategory {
  kTorchTpu,
  kPerformance,
  kUnoptimizedBuild,
  kTcMalloc,
  kCorrectness,
  kApiStage,
  kExperimentalApi,
  kDeprecatedApi,
};

// Returns the message prefix of warnings in `category`.
constexpr std::string_view GetWarningPrefix(WarningCategory category) {
  switch (category) {
    case WarningCategory::kTorchTpu:
      return "<TorchTpu/> ";
    case WarningCategory::kPerformance:
      return "<TorchTpu/Performance/> ";
    case WarningCategory::kUnoptimizedBuild:
      return "<TorchTpu/Performance/UnoptimizedBuild/> ";
    case WarningCategory::kTcMalloc:
      return "<TorchTpu/Performance/TcMalloc/> ";
    case WarningCategory::kCorrectness:
      return "<TorchTpu/Correctness/> ";
    case WarningCategory::kApiStage:
      return "<TorchTpu/ApiStage/> ";
    case WarningCategory::kExperimentalApi:
      return "<TorchTpu/ApiStage/ExperimentalApi/> ";
    case WarningCategory::kDeprecatedApi:
      return "<TorchTpu/ApiStage/DeprecatedApi/> ";
  }
  return "<TorchTpu/> ";
}

// Returns whether warnings in `category` are emitted as Python
// `DeprecationWarning`s rather than `UserWarning`s.
constexpr bool IsDeprecationWarning(WarningCategory category) {
  return category == WarningCategory::kDeprecatedApi;
}

}  // namespace torch_tpu

// Emits a warning in the given `::torch_tpu::WarningCategory` through PyTorch's
// warning machinery, i.e. `TORCH_WARN` or `TORCH_WARN_DEPRECATION` with the
// category prefix (`GetWarningPrefix`) prepended to the message. Python
// receives a plain built-in warning, so all Python warning filters apply to it
// as to any other warning from PyTorch.
//
// These macros do NOT cause TorchDynamo graph breaks.
#define TT_WARN(category, ...)                                          \
  do {                                                                  \
    if (::torch_tpu::IsDeprecationWarning(category)) {                  \
      TORCH_WARN_DEPRECATION(/* TORCH_WARN_OK=implementing TT_WARN */   \
                             ::torch_tpu::GetWarningPrefix(category),   \
                             __VA_ARGS__);                              \
    } else {                                                            \
      TORCH_WARN(/* TORCH_WARN_OK=implementing TT_WARN */               \
                 ::torch_tpu::GetWarningPrefix(category), __VA_ARGS__); \
    }                                                                   \
  } while (false)

// Like `TT_WARN`, but emits the warning only the first time it's reached in the
// process, unless `c10::WarningUtils::set_warnAlways(true)` (e.g.
// `torch.set_warn_always(True)`) is in effect, like `TORCH_WARN_ONCE`. Each
// expansion has its own block scope, so the once-state is per call site.
#define TT_WARN_ONCE(category, ...)                                \
  do {                                                             \
    if (::c10::WarningUtils::get_warnAlways()) {                   \
      TT_WARN(category, __VA_ARGS__);                              \
    } else {                                                       \
      [[maybe_unused]] static const bool tt_warn_once_done = [&] { \
        TT_WARN(category, __VA_ARGS__);                            \
        return true;                                               \
      }();                                                         \
    }                                                              \
  } while (false)

#endif  // TORCH_TPU_CSRC_COMMON_WARNINGS_H_
