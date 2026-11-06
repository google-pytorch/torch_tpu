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

# Kernel-authoring helpers built on top of the Kiln core language.

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//csrc/internal/mosaic/kiln:__subpackages__"],
)

torch_tpu_cc_library(
    name = "pipeline",
    srcs = ["pipeline.cc"],
    hdrs = ["pipeline.h"],
    deps = [
        "//csrc/internal/mosaic/kiln/core:control_flow",
        "//csrc/internal/mosaic/kiln/core:dma",
        "//csrc/internal/mosaic/kiln/core:dtype",
        "//csrc/internal/mosaic/kiln/core:elementwise",
        "//csrc/internal/mosaic/kiln/core:index_arg",
        "//csrc/internal/mosaic/kiln/core:memory",
        "//csrc/internal/mosaic/kiln/core:ref",
        "//csrc/internal/mosaic/kiln/core:scalar",
        "//csrc/internal/mosaic/kiln/core:semaphore",
        "//csrc/internal/mosaic/kiln/core:shape",
        "//csrc/internal/mosaic/kiln/core:specs",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:absl_check",
        "@llvm-project//llvm:Support",
    ],
)

torch_tpu_package_end()
