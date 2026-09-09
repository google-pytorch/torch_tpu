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
    srcs = ["//csrc/internal:env.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:native_scan_support",
        "//csrc/common:utils",
        "//shims/torch:torch_headers",
    ],
)

pybind_extension(
    # Internal testing utilities.
    name = "testing",
    srcs = ["//csrc/internal:testing.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:error_utils",
        "//csrc/common:pybind_error_utils",
        "//csrc/eager:device_gen_impl",
        "//csrc/eager:events_queue",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:repeated_ops_heuristic",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
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
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_package_end()
