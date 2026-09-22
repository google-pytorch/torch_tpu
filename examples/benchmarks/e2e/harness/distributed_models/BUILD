# Copyright 2026 Google LLC
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

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "qwen",
    srcs = ["qwen.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:compile",
        "//examples/benchmarks/e2e/harness:context",
        "//examples/benchmarks/e2e/harness:registry",
        "//examples/benchmarks/e2e/harness:step_lib",
        "//examples/benchmarks/e2e/harness:target",
        "//examples/benchmarks/e2e/harness:torch_device_ops",
        "//examples/benchmarks/e2e/harness/model_utils:ragged_moe",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//tests:module_registry",
        "@pypi//fairscale",
        "@pypi//transformers",
    ],
)

torch_tpu_py_library(
    name = "distributed_models",
    srcs = ["__init__.py"],
    deps = [
        ":qwen",  # buildcleaner: keep
    ],
)

torch_tpu_package_end()
