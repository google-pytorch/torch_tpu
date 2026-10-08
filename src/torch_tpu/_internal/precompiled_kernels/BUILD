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
    "torch_tpu_py_test",
)
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pybind_extension(
    name = "_tpu_torch_precompiled_kernels",
    srcs = ["//csrc/internal/precompiled_kernels:precompiled_kernels_py.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:pybind_error_utils",
        "//csrc/experimental/precompiled_kernels",
    ],
)

torch_tpu_py_library(
    name = "aot_compiler",
    srcs = ["aot_compiler.py"],
    deps = [
        ":_tpu_torch_precompiled_kernels",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//src/torch_tpu/_internal:env_utils",
    ],
)

pytype_strict_binary(
    name = "aot_compiler_bin",
    srcs = ["aot_compiler.py"],
    main = "aot_compiler.py",
    deps = [
        ":aot_compiler",
    ],
)

torch_tpu_py_test(
    name = "aot_compiler_test",
    srcs = ["aot_compiler_test.py"],
    notest_oss = True,  # NOTEST_OSS_OK=Requires libtpu >= 0.0.48 with operand_to_metadata support
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":aot_compiler",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "//tests:subprocess_test_utils",
    ],
)

torch_tpu_py_test(
    name = "aot_compiler_prefill_test",
    srcs = ["aot_compiler_prefill_test.py"],
    notest_oss = True,  # NOTEST_OSS_OK=Requires libtpu >= 0.0.48 with operand_to_metadata support
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":aot_compiler",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_package_end()
