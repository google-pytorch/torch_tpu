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

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "mosaic_custom_call",
    srcs = ["mosaic_custom_call.cc"],
    hdrs = ["mosaic_custom_call.h"],
    deps = [
        "//csrc/common:error_utils",
        "//csrc/internal/mosaic:op_builders",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_module_config",
    ],
)

torch_tpu_cc_library(
    name = "kernel_test_utils",
    testonly = True,
    srcs = ["kernel_test_utils.cc"],
    hdrs = ["kernel_test_utils.h"],
    deps = [
        "//csrc/common:error_utils",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_cc_test(
    name = "mosaic_custom_call_test",
    srcs = ["mosaic_custom_call_test.cc"],
    notest_oss = "Requires libtpu >= 0.0.48 with operand_to_metadata support",
    tags = [
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x",
    ],
    deps = [
        ":kernel_test_utils",
        ":mosaic_custom_call",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "//csrc/common:status_test_utils",
        "//csrc/internal/mosaic:op_builders",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_layout",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/service:hlo_runner",
    ],
)

torch_tpu_package_end()
