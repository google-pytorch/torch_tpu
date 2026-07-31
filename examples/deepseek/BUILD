# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_license//rules:license.bzl", "license")
load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library")
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary")

package(
    default_applicable_licenses = [":license"],
    default_visibility = [
        "//examples/benchmarks/e2e:__subpackages__",
        "//examples/deepseek:__subpackages__",
        "//experimental/users/yho/torch_tpu_deepseek:__subpackages__",
        "//torch_tpu/google/tools:__subpackages__",
    ],
)

license(
    name = "license",
    package_name = "torch_tpu_impl",
)

licenses(["notice"])

torch_tpu_py_library(
    name = "deepseek",
    srcs = [
        "model.py",
    ],
    deps = [
        "//shims/torch:pytorch",
    ],
)

pytype_strict_binary(
    name = "trace",
    srcs = ["trace.py"],
    exec_properties = {
        "cpp_link.mem": "20g",  # Needed for asan.
    },
    deps = [
        ":deepseek",
        "//shims/absl_py:app",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/_internal/utils:tracer_utils",
    ],
)

torch_tpu_package_end()
