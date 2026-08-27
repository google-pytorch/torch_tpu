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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pybind_extension(
    name = "env",
    srcs = ["//torch_tpu/csrc/internal:env.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch:torch_headers",
        "//torch_tpu/csrc/common:native_scan_support",
        "//torch_tpu/csrc/common:utils",
    ],
)

pybind_extension(
    # Internal testing utilities.
    name = "testing",
    srcs = ["//torch_tpu/csrc/internal:testing.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
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
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/common:pybind_error_utils",
        "//torch_tpu/csrc/eager:device_gen_impl",
        "//torch_tpu/csrc/eager:events_queue",
        "//torch_tpu/csrc/eager:op_dispatcher",
        "//torch_tpu/csrc/eager:repeated_ops_heuristic",
        "//torch_tpu/csrc/eager:structured_log_buffer",
        "//torch_tpu/csrc/eager:tensor_to_buffer",
        "@com_google_absl//absl/flags:flag",
        "@xla//xla/pjrt:pjrt_client",
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

torch_tpu_package_end()
