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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ],
)

pytype_strict_library(
    name = "device_utils",
    srcs = ["device_utils.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/shims/xprof:xprof_analysis_client",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/api",
        "@pypi//psutil",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    tags = [
        # Prevent build_cleaner from adding @pypi//jax/_src/*, which is a lazy import
        # that is invisible to this package. See go/build-cleaner/customize-behavior#ignore-for-dep
        "nofixdeps",
        "ignore_for_dep=third_party.py.jax",
    ],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/device:_device_module",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/export",
        "//torch_tpu/_internal/sync",
        "@pypi//numpy",
    ],
)

pytype_strict_library(
    name = "hardware",
    srcs = ["hardware.py"],
    deps = ["@pypi//immutabledict"],
)

torch_tpu_py_test(
    name = "hardware_test",
    srcs = ["hardware_test.py"],
    deps = [
        ":hardware",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
    ],
)

torch_tpu_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":test_fixtures",
        ":utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

pytype_strict_library(
    name = "benchmarking",
    srcs = ["benchmarking.py"],
    deps = [
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "@pypi//numpy",
    ],
)

pytype_strict_library(
    name = "tracer_utils",
    srcs = ["tracer_utils.py"],
    deps = [
        ":utils",
        "//shims/torch:pytorch",
    ],
)

pytype_strict_library(
    name = "test_fixtures",
    testonly = True,
    srcs = ["test_fixtures.py"],
    deps = ["//shims/torch:pytorch"],
)

torch_tpu_py_test(
    name = "tracer_utils_test",
    srcs = ["tracer_utils_test.py"],
    strict = True,
    deps = [
        ":test_fixtures",
        ":tracer_utils",
        ":utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "tracer_utils_tpu_test",
    srcs = ["tracer_utils_tpu_test.py"],
    main = "tracer_utils_tpu_test.py",
    strict = True,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":test_fixtures",
        ":tracer_utils",
        ":utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "test_fixtures_test",
    srcs = ["test_fixtures_test.py"],
    strict = True,
    deps = [
        ":test_fixtures",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
    ],
)

pytype_strict_library(
    name = "log_utils",
    srcs = ["log_utils.py"],
    deps = [
        "//shims/absl_py/logging",
    ],
)

torch_tpu_package_end()
