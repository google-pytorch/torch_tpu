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
# tudorpip
# ryo-not-rio

# Sorting building blocks for precompiled unbounded dynamic kernels.

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_cc_library",
    "torch_tpu_cc_test",
    "torch_tpu_package_end",
)

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "sort",
    srcs = ["sort.cc"],
    hdrs = ["sort.h"],
    deps = [
        "//csrc/common:cache_key",
        "//csrc/common:compilation",
        "//csrc/common:compilation_spec",
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//csrc/common:unbounded_dynamic_kernel_registry",
        "//csrc/experimental/precompiled_kernels:kernel_utils",
        "//csrc/internal/mosaic/kiln",
        "//csrc/ops:op_names",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

torch_tpu_cc_test(
    name = "sort_test",
    srcs = ["sort_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":sort",
        "//csrc/common:dimension_types",
        "//csrc/common:oss_test_main",
        "//csrc/common:shape",
        "//csrc/common:unbounded_dynamic_kernel_registry",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
    ],
)

torch_tpu_package_end()
