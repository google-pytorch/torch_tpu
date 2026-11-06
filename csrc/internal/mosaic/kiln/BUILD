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

# A Pallas-shaped C++ kernel language (Kiln) for authoring Mosaic TPU kernels.

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "kiln",
    hdrs = ["kiln.h"],
    deps = [
        "//csrc/internal/mosaic/kiln/core:control_flow",
        "//csrc/internal/mosaic/kiln/core:dma",
        "//csrc/internal/mosaic/kiln/core:dtype",
        "//csrc/internal/mosaic/kiln/core:elementwise",
        "//csrc/internal/mosaic/kiln/core:index_arg",
        "//csrc/internal/mosaic/kiln/core:index_map",
        "//csrc/internal/mosaic/kiln/core:kernel",
        "//csrc/internal/mosaic/kiln/core:literal",
        "//csrc/internal/mosaic/kiln/core:memory",
        "//csrc/internal/mosaic/kiln/core:ref",
        "//csrc/internal/mosaic/kiln/core:scalar",
        "//csrc/internal/mosaic/kiln/core:semaphore",
        "//csrc/internal/mosaic/kiln/core:shape",
        "//csrc/internal/mosaic/kiln/core:specs",
        "//csrc/internal/mosaic/kiln/core:tile",
        "//csrc/internal/mosaic/kiln/core:value_base",
        "//csrc/internal/mosaic/kiln/lib:pipeline",
    ],
)

torch_tpu_cc_test(
    name = "kiln_test",
    srcs = ["kiln_test.cc"],
    deps = [
        ":kiln",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@xla//xla/hlo/testlib:filecheck",
    ],
)

torch_tpu_cc_test(
    name = "kiln_e2e_test",
    srcs = ["kiln_e2e_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",
    },
    tags = ["requires-tpu-v5p"],
    deps = [
        ":kiln",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "//csrc/common:status_test_utils",
        "//csrc/experimental/precompiled_kernels:kernel_test_utils",
        "//csrc/internal/mosaic/kiln/mlir:stablehlo_splice",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla:error_spec",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/service:hlo_module_config",
        "@xla//xla/service:hlo_runner",
        "@xla//xla/tests:literal_test_util",
    ],
)

torch_tpu_package_end()
