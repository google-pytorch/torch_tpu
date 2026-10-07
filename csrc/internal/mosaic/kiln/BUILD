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

# A Pallas-shaped C++ kernel language (KILN) for authoring Mosaic TPU kernels.

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "kiln",
    hdrs = ["kiln.h"],
    deps = [
        "//csrc/internal/mosaic/kiln/core:cond",
        "//csrc/internal/mosaic/kiln/core:control_flow",
        "//csrc/internal/mosaic/kiln/core:dtype",
        "//csrc/internal/mosaic/kiln/core:elementwise",
        "//csrc/internal/mosaic/kiln/core:index_arg",
        "//csrc/internal/mosaic/kiln/core:index_map",
        "//csrc/internal/mosaic/kiln/core:kernel",
        "//csrc/internal/mosaic/kiln/core:literal",
        "//csrc/internal/mosaic/kiln/core:ref",
        "//csrc/internal/mosaic/kiln/core:scalar",
        "//csrc/internal/mosaic/kiln/core:specs",
        "//csrc/internal/mosaic/kiln/core:tile",
        "//csrc/internal/mosaic/kiln/core:value_base",
    ],
)

torch_tpu_cc_test(
    name = "kiln_test",
    srcs = ["kiln_test.cc"],
    deps = [
        ":kiln",
        "//csrc/common:oss_test_main",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/hlo/testlib:filecheck",
    ],
)

torch_tpu_package_end()
