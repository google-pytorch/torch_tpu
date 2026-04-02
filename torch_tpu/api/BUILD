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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchprime:__subpackages__",
        "//shims/torchtitan:__subpackages__",
        "@pypi//helion:__subpackages__",
    ],
)

pytype_strict_library(
    name = "api",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":_device_module",
        "//shims/torch:pytorch",
        "//torch_tpu",  # buildcleaner: keep
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/sync",
    ],
)

pytype_strict_library(
    name = "_device_module",
    srcs = ["_device_module.py"],
    deps = [
        ":_device_ops_backend",
        ":streams",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu",  # buildcleaner: keep
        "//torch_tpu/_internal/precision",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

pytype_strict_library(
    name = "streams",
    srcs = ["streams.py"],
    deps = [
        ":_device_ops_backend",
    ],
)

torch_tpu_py_test(
    name = "device_module_test",
    srcs = ["device_module_test.py"],
    strict = True,
    deps = [
        ":_device_module",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
    ],
)

torch_tpu_py_test(
    name = "tpu_module_test",
    srcs = ["tpu_module_test.py"],
    strict = True,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":_device_module",
        ":api",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
    ],
)

torch_tpu_py_test(
    name = "xla_cuda_module_test",
    srcs = ["xla_cuda_module_test.py"],
    notest_oss = "Disabled during switch from opt-in to opt-out",
    strict = True,
    tags = ["requires-gpu-sm60-only"],
    deps = [
        ":_device_module",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
    ],
)

torch_tpu_py_test(
    name = "xla_cpu_module_test",
    srcs = ["xla_cpu_module_test.py"],
    strict = True,
    deps = [
        ":_device_module",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
    ],
)

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that tpu_rt.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "device_rt",
    srcs = ["device_rt.cc"],
    deps = [
        "//shims/pybind11",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:compilation_cache",
        "//torch_tpu/common:discovery",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/distributed/slicebuilder:discovery",
        "//torch_tpu/eager:device_gen_impl",
        "//torch_tpu/eager:device_types",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/eager:tpu_hooks",
        "//torch_tpu/pjrt:pjrt_init",
        "//torch_tpu/pjrt:pjrt_shutdown",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/time",
    ],
)

pybind_extension(
    name = "_device_ops_backend",
    srcs = ["device_rt.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:compilation_cache",
        "//torch_tpu/common:discovery",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/distributed/slicebuilder:discovery",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:device_gen_impl",
        "//torch_tpu/eager:device_types",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/eager:tpu_hooks",
        "//torch_tpu/pjrt:pjrt_init",
        "//torch_tpu/pjrt:pjrt_shutdown",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/time",
    ],
)

torch_tpu_package_end()
