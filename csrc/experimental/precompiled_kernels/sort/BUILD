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

# Sorting building blocks for precompiled unbounded dynamic kernels.

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_cc_library",
    "torch_tpu_package_end",
)

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "bitonic",
    srcs = ["bitonic.cc"],
    hdrs = ["bitonic.h"],
    deps = [
        "//csrc/internal/mosaic/kiln",
    ],
)

torch_tpu_cc_library(
    name = "sort_kernels",
    srcs = ["sort_kernels.cc"],
    hdrs = ["sort_kernels.h"],
    deps = [
        ":bitonic",
        "//csrc/common:cache_key",
        "//csrc/common:compilation",
        "//csrc/common:compilation_spec",
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/common:layout_utils",
        "//csrc/common:shape",
        "//csrc/common:unbounded_dynamic_kernel_registry",
        "//csrc/experimental/precompiled_kernels:kernel_utils",
        "//csrc/experimental/precompiled_kernels:mosaic_custom_call",
        "//csrc/internal/mosaic/kiln",
        "//csrc/ops:op_names",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/client:executable_build_options",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

torch_tpu_package_end()
