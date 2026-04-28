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
load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary")

package(
    default_applicable_licenses = [":license"],
    default_visibility = [
        "//examples/benchmarks/e2e:__subpackages__",
        "//examples/deepseek:__subpackages__",
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

torch_tpu_py_test(
    name = "deepseek_decomposition_one_acc_test__TPU",
    srcs = ["deepseek_decomposition_one_acc_test.py"],
    args = ["--acc=tpu"],
    main = "deepseek_decomposition_one_acc_test.py",
    notap = "Currently broken. Meant to be used for finding missing pieces.",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":deepseek",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
    ],
)

torch_tpu_py_test(
    name = "deepseek_decomposition_one_acc_test__GPU",
    srcs = ["deepseek_decomposition_one_acc_test.py"],
    args = ["--acc=gpu"],
    main = "deepseek_decomposition_one_acc_test.py",
    notap = "slow test",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "ignore_for_dep=third_party.torch_tpu.api",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":deepseek",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
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
