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
    name = "performance_utils",
    testonly = 1,
    srcs = ["performance_utils.py"],
    deps = [
        "//examples/benchmarks/e2e:benchmark_utils",
        "//examples/benchmarks/e2e:common",
        "//examples/benchmarks/e2e:device_utils",
        "//examples/benchmarks/e2e:mlcompass_utils",
        "//examples/benchmarks/e2e:performance_utils",
        "//examples/benchmarks/e2e/harness:metrics",
        "//shims/absl_py/logging",
        "//shims/torch",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/utils:log_utils",
        "@pypi//jax",
        "@pypi//numpy",
        "@pypi//optax",
        "@pypi//torchax",
    ],
)

torch_tpu_package_end()
