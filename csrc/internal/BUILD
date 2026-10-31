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

# Domain experts for this package:
# zhanyong-wan
# haifeng-jin

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

exports_files([
    "env.cc",
    "testing.cc",
    "warnings_impl.cc",
])

# This library is not used directly, but is needed to satisfy go/cclean
torch_tpu_cc_library(
    name = "env_build",
    srcs = ["env.cc"],
    deps = [
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:libtpu_version",
        "//csrc/common:process",
        "//csrc/common:pybind_error_utils",
        "//csrc/common:utils",
        "//shims/pybind11",
        "@com_google_absl//absl/log:absl_log",
    ],
)

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that testing.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "testing_build",
    srcs = ["testing.cc"],
    deps = [
        "//csrc/common:error_utils",
        "//csrc/common:pybind_error_utils",
        "//csrc/common:tpu_hardware",
        "//csrc/common:warnings",
        "//csrc/eager:device_gen_impl",
        "//csrc/eager:events_queue",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:python_context",
        "//csrc/pjrt:pjrt_state",
        "//shims/pybind11",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

# This library is not used directly, but is needed for go/cclean.
torch_tpu_cc_library(
    name = "warnings_impl_build",
    srcs = ["warnings_impl.cc"],
    deps = [
        "//csrc/common:pybind_error_utils",
        "//csrc/common:warnings",
        "//shims/pybind11",
    ],
)

torch_tpu_package_end()
