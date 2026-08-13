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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

torch_tpu_cc_library(
    name = "device_buffer",
    hdrs = ["device_buffer.h"],
    deps = [
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/concurrency:future",
    ],
)

torch_tpu_cc_library(
    name = "executable",
    hdrs = ["executable.h"],
    deps = [
        ":device_buffer",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "backend_hdrs",
    hdrs = ["backend.h"],
    deps = [
        ":device_buffer",
        ":executable",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:host_memory_allocator",
        "@xla//xla/tsl/framework:allocator",
    ],
)

torch_tpu_cc_library(
    name = "backend_config",
    srcs = ["backend_config.cc"],
    hdrs = ["backend_config.h"],
    deps = [
        ":backend_hdrs",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

torch_tpu_cc_test(
    name = "backend_config_test",
    srcs = ["backend_config_test.cc"],
    deps = [
        ":backend_config",
        ":backend_hdrs",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/status",
    ],
)

torch_tpu_cc_library(
    name = "backend_registry",
    srcs = ["backend_registry.cc"],
    hdrs = ["backend_registry.h"],
    deps = [
        ":backend_hdrs",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

torch_tpu_cc_test(
    name = "backend_registry_test",
    srcs = ["backend_registry_test.cc"],
    deps = [
        ":backend",
        ":backend_registry",
        ":mock_backend",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/status",
    ],
)

torch_tpu_cc_library(
    name = "backend",
    srcs = ["backend.cc"],
    hdrs = ["backend.h"],
    deps = [
        ":backend_config",
        ":backend_hdrs",
        ":backend_registry",
        ":device_buffer",
        ":executable",
        ":mock_backend",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/runtime/cerebro:cerebro_backend_impl",
        "//torch_tpu/runtime/pjrt:pjrt_backend_impl",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:host_memory_allocator",
        "@xla//xla/tsl/framework:allocator",
    ],
)

torch_tpu_cc_library(
    name = "mock_backend",
    srcs = ["mock_backend.cc"],
    hdrs = ["mock_backend.h"],
    deps = [
        ":backend_hdrs",
        ":device_buffer",
        ":executable",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:dimension_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:host_memory_allocator",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/framework:allocator",
    ],
)

torch_tpu_cc_test(
    name = "backend_test",
    srcs = ["backend_test.cc"],
    deps = [
        ":backend",
        ":device_buffer",
        ":executable",
        ":mock_backend",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
    ],
)

torch_tpu_cc_library(
    name = "eager_bridge",
    srcs = ["eager_bridge.cc"],
    hdrs = ["eager_bridge.h"],
    deps = [
        ":backend_hdrs",
        ":device_buffer",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

torch_tpu_cc_test(
    name = "eager_bridge_test",
    srcs = ["eager_bridge_test.cc"],
    deps = [
        ":device_buffer",
        ":eager_bridge",
        ":mock_backend",
        "//shims/torch:aten_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/status",
    ],
)

torch_tpu_cc_library(
    name = "eager_runtime_adapter",
    srcs = ["eager_runtime_adapter.cc"],
    hdrs = ["eager_runtime_adapter.h"],
    deps = [
        ":backend_hdrs",
        ":device_buffer",
        ":eager_bridge",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/framework:allocator",
    ],
)

torch_tpu_cc_test(
    name = "eager_runtime_adapter_test",
    srcs = ["eager_runtime_adapter_test.cc"],
    deps = [
        ":backend",
        ":device_buffer",
        ":eager_runtime_adapter",
        ":mock_backend",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
    ],
)

torch_tpu_package_end()
