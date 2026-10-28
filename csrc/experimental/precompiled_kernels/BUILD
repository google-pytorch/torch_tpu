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
# thcmbs
# ryo-not-rio

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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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

torch_tpu_cc_library(
    name = "kernel_utils",
    srcs = ["kernel_utils.cc"],
    hdrs = ["kernel_utils.h"],
    deps = [
        "//csrc/common:cache_key",
        "//csrc/common:compile_options_key",
    ],
)

torch_tpu_cc_library(
    name = "nonzero_indices",
    srcs = ["nonzero_indices.cc"],
    hdrs = ["nonzero_indices.h"],
    deps = [
        ":kernel_utils",
        ":mosaic_custom_call",
        "//csrc/common:cache_key",
        "//csrc/common:compilation",
        "//csrc/common:compilation_spec",
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//csrc/common:unbounded_dynamic_kernel_registry",
        "//csrc/internal/mosaic/kiln",
        "//csrc/ops:op_names",
        "@com_google_absl//absl/status",
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

# Registers every precompiled unbounded dynamic kernel with
# //csrc/common:unbounded_dynamic_kernel_registry when
# loaded. Nothing references its symbols, hence `alwayslink`. Depends on
# Mosaic through the kernels, so link it only into TPU-only targets.
torch_tpu_cc_library(
    name = "kernel_registration",
    srcs = ["kernel_registration.cc"],
    deps = [
        ":nonzero_indices",
        "//csrc/common:unbounded_dynamic_kernel_registry",
        "@com_google_absl//absl/log:absl_check",
    ],
    alwayslink = True,
)

torch_tpu_cc_test(
    name = "mosaic_custom_call_test",
    srcs = ["mosaic_custom_call_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = True,  # NOTEST_OSS_OK=Requires libtpu >= 0.0.48 with operand_to_metadata support
    tags = ["requires-tpu-v5p"],
    deps = [
        ":kernel_test_utils",
        ":mosaic_custom_call",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "//csrc/common:status_test_utils",
        "//csrc/internal/mosaic/kiln",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_layout",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/service:hlo_runner",
    ],
)

[torch_tpu_cc_test(
    name = "nonzero_indices_test_" + gen,
    srcs = ["nonzero_indices_test.cc"],
    data = ["testdata/nonzero_xla_baseline_" + gen + ".csv"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    nopresubmit = nopresubmit,
    tags = [chip],
    deps = [
        ":kernel_registration",  # buildcleaner: keep
        ":nonzero_indices",
        "//csrc/common:cache_key",
        "//csrc/common:compilation",
        "//csrc/common:compilation_cache",
        "//csrc/common:dimension_types",
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "//csrc/common:shape",
        "//csrc/common:status_test_utils",
        "//csrc/common:unbounded_dynamic_kernel_registry",
        "//csrc/ops/nonzero",
        "//csrc/pjrt:pjrt_execute",
        "//csrc/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@stablehlo//stablehlo/integrations/cpp/builder:mlir_builder",
        "@xla//xla:literal",
        "@xla//xla:shape_layout",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/service:computation_layout",
    ],
) for gen, chip, nopresubmit in [
    ("v5", "requires-tpu-v5p", None),
    ("v5e", "requires-tpu-v5lite", None),
    ("v6", "requires-tpu-v6e", "Avoid long Forge queue."),
    ("v7", "requires-tpu-v7x", "Avoid long Forge queue."),
]]

torch_tpu_package_end()
