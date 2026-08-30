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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

exports_files(["sync_py.cc"])

torch_tpu_cc_library(
    name = "sync_cc",
    srcs = [
        "sync.cc",
    ],
    hdrs = [
        "sync.h",
    ],
    deps = [
        "//csrc/common:error_utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:events_queue",
        "//csrc/eager:materialize",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/eager:traversal",
        "//shims/torch:aten_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@xla//xla/hlo/translate:register",
    ],
)

torch_tpu_cc_library(
    name = "sync_py_cc",
    srcs = ["sync_py.cc"],
    deps = [
        ":sync_cc",
        "//csrc/common:device_type",
        "//csrc/common:error_utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:materialize",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_names",
        "//csrc/ops:python_context",
        "//shims/pybind11",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_package_end()
