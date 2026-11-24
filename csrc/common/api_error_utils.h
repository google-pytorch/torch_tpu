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

#ifndef TORCH_TPU_CSRC_COMMON_API_ERROR_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_API_ERROR_UTILS_H_

#include <string_view>
#include <utility>

#include "csrc/common/error_utils.h"
#include "csrc/ops/python_context.h"

namespace torch_tpu {

// Invokes `fn(args...)` inside a `ScopedPythonContextCapturer(PythonApiName)`
// scope and translates any thrown `TtError` into the corresponding `c10::Error`
// subclass via `TranslateTtErrorToC10Error`.
template <typename Fn, typename... Args>
decltype(auto) RunWithApiErrorHandling(std::string_view api_name, Fn&& fn,
                                       Args&&... args) {
  ScopedPythonContextCapturer capturer(PythonApiName{api_name});
  return TranslateTtErrorToC10Error(std::forward<Fn>(fn),
                                    std::forward<Args>(args)...);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_API_ERROR_UTILS_H_
