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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

# This library is not used directly, but is needed to satisfy go/cclean
torch_tpu_cc_library(
    name = "env_build",
    srcs = ["env.cc"],
    deps = [
        "//shims/pybind11",
        "//torch_tpu/common:native_scan_support",
        "//torch_tpu/common:utils",
    ],
)

pybind_extension(
    name = "env",
    srcs = ["env.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch:torch_headers",
        "//torch_tpu/common:native_scan_support",
        "//torch_tpu/common:utils",
    ],
)

torch_tpu_py_library(
    name = "native_scan",
    srcs = ["native_scan.py"],
    visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "native_scan_test",
    srcs = ["native_scan_test.py"],
    deps = [
        ":native_scan",
        "//shims/absl_py/testing:absltest",
        "//tests:seed_test_utils",
        "//torch_tpu",
    ],
)

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that testing.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "testing_build",
    srcs = ["testing.cc"],
    deps = [
        "//shims/pybind11",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_gen_impl",
        "//torch_tpu/eager:events_queue",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:repeated_ops_heuristic",
        "//torch_tpu/eager:structured_log_buffer",
        "//torch_tpu/eager:tensor_to_buffer",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

pybind_extension(
    # Internal testing utilities.
    name = "testing",
    srcs = ["testing.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_gen_impl",
        "//torch_tpu/eager:events_queue",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:repeated_ops_heuristic",
        "//torch_tpu/eager:structured_log_buffer",
        "//torch_tpu/eager:tensor_to_buffer",
        "@com_google_absl//absl/flags:flag",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

torch_tpu_package_end()
