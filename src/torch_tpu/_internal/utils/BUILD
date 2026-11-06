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

# Domain experts for this package:
# yilingyuan
# zhanyong-wan
# ghpvnist
# arfaian
# adivinpatel04
# haifeng-jin
# chizhang529

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ],
)

torch_tpu_py_library(
    name = "init",
    srcs = ["__init__.py"],
)

torch_tpu_py_library(
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
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/device:_device_module",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/export",
        "//src/torch_tpu/_internal/sync",
    ],
)

torch_tpu_py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    deps = [
        ":utils",
        "//shims/torch:pytorch",
        "@pypi//numpy",
    ],
)

pybind_extension(
    name = "_hardware_backend",
    srcs = ["//csrc/internal/utils:hardware_py.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:pybind_error_utils",
        "//csrc/common:tpu_hardware",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

torch_tpu_py_library(
    name = "hardware",
    srcs = ["hardware.py"],
    deps = [
        ":_hardware_backend",
        ":annotations",
    ],
)

torch_tpu_py_test(
    name = "hardware_test",
    srcs = ["hardware_test.py"],
    deps = [
        ":annotations",
        ":hardware",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:op_testing",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    tags = [
        "nomsan",  # 3p 'py/torch' error: use-of-uninitialized-value in 3 in reduced_float_type_copy
        "requires-tpu-v5lite",
    ],
    deps = [
        ":test_fixtures",
        ":test_utils",
        ":utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/dynamism",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "tracer_utils",
    srcs = ["tracer_utils.py"],
    deps = [
        ":utils",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
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
        ":test_utils",
        ":tracer_utils",
        ":utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "tracer_utils_tpu_test",
    srcs = ["tracer_utils_tpu_test.py"],
    main = "tracer_utils_tpu_test.py",
    strict = True,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":test_utils",
        ":tracer_utils",
        ":utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
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
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "log_utils",
    srcs = ["log_utils.py"],
    deps = [
        "//shims/absl_py/logging",
    ],
)

torch_tpu_py_library(
    name = "annotations",
    srcs = ["annotations.py"],
    deps = [
        ":tt_warnings",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal:warnings",
    ],
)

torch_tpu_py_test(
    name = "annotations_test",
    srcs = ["annotations_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":annotations",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/device:_device_module",
        "//src/torch_tpu/_internal/precision:precision_impl",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "tt_warnings",
    srcs = ["tt_warnings.py"],
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal:warnings",
    ],
)

torch_tpu_py_test(
    name = "tt_warnings_test",
    srcs = ["tt_warnings_test.py"],
    deps = [
        ":tt_warnings",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:warnings",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_package_end()
