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

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_library(
    name = "device_utils",
    srcs = ["device_utils.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/sync",
        "@pypi//jax",
        "@pypi//psutil",
    ],
)

torch_tpu_py_test(
    name = "device_utils_test",
    srcs = ["device_utils_test.py"],
    args = ["--torch_tpu_internal_enable_compilation_container=true"],
    notest_oss = True,  # Flags don't work in OSS.
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":device_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "single_trace_trainer_adapter",
    srcs = ["single_trace_trainer_adapter.py"],
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/extras/optimizers:adamw",
        "//src/torch_tpu/extras/optimizers:muon",
        "//src/torch_tpu/extras/optimizers:optim",
        "//src/torch_tpu/extras/optimizers:sgd",
        "//src/torch_tpu/extras/optimizers:single_trace_trainer",
    ],
)

torch_tpu_package_end()
