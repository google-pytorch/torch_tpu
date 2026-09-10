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

# The ahead-of-time compiler. A build tool only: it links Mosaic and the PJRT
# compiler and is never a runtime dependency of the wheel.

load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load(
    "//build_files:build_defs.bzl",
    "torch_tpu_cc_library",
    "torch_tpu_package_end",
)

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//csrc/experimental/precompiled_kernels:__subpackages__",
    ],
)

_LIBTPU = "@precompiled_libtpu//:libtpu/libtpu.so"

torch_tpu_cc_library(
    name = "elementwise_kernels",
    srcs = ["elementwise/kernels.cc"],
    hdrs = ["elementwise/kernels.h"],
    deps = [
        "//csrc/ops:op_names",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:VectorDialect",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/mosaic:tpu_dialect",
    ],
)

cc_binary(
    name = "generate_bundle",
    srcs = [
        "elementwise/compile.cc",
        "elementwise/compile.h",
        "generate.cc",
    ],
    deps = [
        ":elementwise_kernels",
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:flags",
        "//csrc/experimental/precompiled_kernels:bundle_cc_proto",
        "//csrc/ops:op_names",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@xla//xla:shape_layout",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/mosaic:tpu_dialect",
        "@xla//xla/mosaic:tpu_serde_pass",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/service:computation_placer",
    ],
)

string_flag(
    name = "chip",
    build_setting_default = "v7x",
    values = [
        "v6e",
        "v7x",
    ],
)

config_setting(
    name = "v6e",
    flag_values = {":chip": "v6e"},
)

# Compiles precompiled kernels for the target chip using the pinned libtpu.
# In Google3, uses the internal libtpu. In OSS, uses the pinned libtpu dependency.
genrule(
    name = "bundle",
    srcs = [_LIBTPU],
    outs = ["kernels.pb"],
    cmd = "TPU_SKIP_MDS_QUERY=true " +
          "TPU_LIBRARY_PATH=$(execpath %s) " % _LIBTPU +
          "$(execpath :generate_bundle) --torch_tpu_internal_output=$@ " + select({
        ":v6e": "--torch_tpu_internal_topology=v6e:2x2",
        "//conditions:default": "--torch_tpu_internal_topology=tpu7x:2x2",
    }),
    tools = [":generate_bundle"],
)

torch_tpu_package_end()
