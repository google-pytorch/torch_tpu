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
    "torch_tpu_cc_library",
    "torch_tpu_cc_test",
    "torch_tpu_package_end",
)
load(
    "//shims/build_files:build_defs.bzl",
    "TT_FRIENDS",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "tensor_buffer",
    srcs = ["tensor_buffer.cc"],
    hdrs = ["tensor_buffer.h"],
    visibility = [
        "//:__subpackages__",
    ] + TT_FRIENDS,
    deps = [
        "//csrc/common:error_utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:materialize",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//shims/torch:aten_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

torch_tpu_cc_test(
    name = "tensor_buffer_test",
    srcs = ["tensor_buffer_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    requires_libtpu = True,
    deps = [
        ":tensor_buffer",
        "//csrc/common:cache_key",
        "//csrc/common:compilation_cache",
        "//csrc/common:dimension_types",
        "//csrc/common:oss_test_main",
        "//csrc/common:shape",
        "//csrc/common:status_test_utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:materialize",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/eager:tpu_hooks",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:python_context",
        "//csrc/pjrt:pjrt_state",
        "//shims/torch:aten_headers",
        "//shims/torch/c10",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

# This must be the last line in the BUILD file.
torch_tpu_package_end()
