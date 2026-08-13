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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

torch_tpu_cc_library(
    name = "pjrt_device_buffer",
    srcs = ["pjrt_device_buffer.cc"],
    hdrs = ["pjrt_device_buffer.h"],
    deps = [
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/runtime:backend_hdrs",
        "//torch_tpu/runtime:device_buffer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/concurrency:future",
    ],
)

torch_tpu_cc_library(
    name = "pjrt_executable",
    srcs = ["pjrt_executable.cc"],
    hdrs = ["pjrt_executable.h"],
    deps = [
        ":pjrt_device_buffer",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/runtime:device_buffer",
        "//torch_tpu/runtime:executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

torch_tpu_cc_library(
    name = "pjrt_backend_impl",
    srcs = ["pjrt_backend_impl.cc"],
    hdrs = ["pjrt_backend_impl.h"],
    deps = [
        ":pjrt_device_buffer",
        ":pjrt_executable",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/pjrt:pjrt_state",
        "//torch_tpu/runtime:backend_hdrs",
        "//torch_tpu/runtime:device_buffer",
        "//torch_tpu/runtime:executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:host_memory_allocator",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/framework:allocator",
    ],
)

torch_tpu_package_end()
