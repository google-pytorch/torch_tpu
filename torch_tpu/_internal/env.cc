/*
 * Copyright 2025 Google LLC
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
#include <string>

#include "pybind11/pybind11.h"
#include "torch/headeronly/version.h"
#include "torch_tpu/common/utils.h"

extern "C" const char* get_torch_tpu_git_commit();

static std::string GetPyTorchBuildVersion() {
#if defined(TORCH_VERSION)
  return TORCH_VERSION;
#else
  std::string version;
#ifdef TORCH_VERSION_MAJOR
  version += std::to_string(TORCH_VERSION_MAJOR);
#endif
#ifdef TORCH_VERSION_MINOR
  if (!version.empty()) {
    version += ".";
  }
  version += std::to_string(TORCH_VERSION_MINOR);
#endif
#ifdef TORCH_VERSION_PATCH
  if (!version.empty()) {
    version += ".";
  }
  version += std::to_string(TORCH_VERSION_PATCH);
#endif
  return version.empty() ? "unknown" : version;
#endif
}

namespace torch_tpu {

PYBIND11_MODULE(env, m) {
  m.attr("IS_INTERNAL_TORCH_TPU") = static_cast<bool>(TT_IS_INTERNAL_TORCH_TPU);
  m.attr("PYTORCH_BUILD_VERSION") = GetPyTorchBuildVersion();
  m.attr("TORCH_TPU_GIT_COMMIT") = get_torch_tpu_git_commit();
}

}  // namespace torch_tpu
