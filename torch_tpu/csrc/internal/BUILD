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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")
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
        "//torch_tpu/csrc/common:native_scan_support",
        "//torch_tpu/csrc/common:utils",
    ],
)

pybind_extension(
    name = "env",
    srcs = ["env.cc"],
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
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/eager:device_gen_impl",
        "//torch_tpu/csrc/eager:events_queue",
        "//torch_tpu/csrc/eager:op_dispatcher",
        "//torch_tpu/csrc/eager:repeated_ops_heuristic",
        "//torch_tpu/csrc/eager:structured_log_buffer",
        "//torch_tpu/csrc/eager:tensor_to_buffer",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

pybind_extension(
    # Internal testing utilities.
    name = "testing",
    srcs = ["testing.cc"],
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

torch_tpu_package_end()
