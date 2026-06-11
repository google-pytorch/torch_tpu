# Copyright 2025 Google LLC
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

"""Common utilities for torch_tpu."""

load(
    "@xla//third_party/py/rules_pywrap:pywrap.default.bzl",
    "pywrap_binaries",
    "pywrap_library",
)
load(
    "//build_files:build_defs.bzl",
    "oss_target",
    "torch_tpu_cc_library",
    "torch_tpu_cc_test",
    "torch_tpu_package_end",
)
load(
    "//shims/build_files:build_defs.bzl",
    "add_internal_filesystem_dependencies",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "unique_file_descriptor",
    srcs = ["unique_file_descriptor.cc"],
    hdrs = ["unique_file_descriptor.h"],
)

torch_tpu_cc_library(
    name = "context_states",
    hdrs = ["context_states.h"],
    deps = [
        ":compilation_spec",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base",
        "@com_google_absl//absl/base:nullability",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:shape_util",
    ],
)

torch_tpu_cc_library(
    name = "dtype",
    srcs = ["dtype.cc"],
    hdrs = ["dtype.h"],
    deps = [
        ":dimension_types",
        ":error_utils",
        ":fingerprint_utils",
        ":to_string",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_library(
    name = "dimension_types",
    hdrs = ["dimension_types.h"],
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

torch_tpu_cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = [
        ":dimension_types",
        ":dtype",
        ":error_utils",
        ":utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:attr_type_builder_util",
        "@xla//xla:shape_util",
    ],
)

torch_tpu_cc_library(
    name = "status_builder",
    srcs = ["status_builder.cc"],
    hdrs = ["status_builder.h"],
    deps = [
        ":env_vars",
        ":utils",
        "//shims/torch",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

torch_tpu_cc_library(
    name = "error_utils",
    srcs = ["error_utils.cc"],
    hdrs = ["error_utils.h"],
    deps = [
        ":device_type",
        ":env_vars",
        ":macro_utils",
        ":status_builder",
        ":to_string",
        ":utils",
        "//shims/torch",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//torch_tpu/eager:eager_mode",
        "//torch_tpu/ops:python_context",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_library(
    name = "static_shape_check",
    srcs = ["static_shape_check.cc"],
    hdrs = ["static_shape_check.h"],
    deps = [
        ":error_utils",
        ":shape",
        ":to_string",
        "//shims/torch:aten_headers",
        "//torch_tpu/eager:device_buffer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_cc_test(
    name = "error_utils_test",
    srcs = ["error_utils_test.cc"],
    deps = [
        ":cache_key",
        ":env_vars",
        ":error_utils",
        ":utils",
        "//shims/torch/c10/util:base_headers",
        "//torch_tpu/eager:eager_mode",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops/macros:kernel",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "compilation_spec",
    hdrs = ["compilation_spec.h"],
    deps = [
        ":compile_options_key",
        "@com_google_absl//absl/base:nullability",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

torch_tpu_cc_library(
    name = "compilation",
    srcs = ["compilation.cc"],
    hdrs = ["compilation.h"],
    deps = [
        ":compilation_spec",
        ":compile_options_key",
        ":context_manager",
        ":context_states",
        ":env_vars",
        ":error_utils",
        ":fingerprint_utils",
        ":shape",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@xla//xla:shape_util",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/client:executable_build_options",
        "@xla//xla/hlo/translate:register",
        "@xla//xla/pjrt:maybe_owning_mlir_module",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/service:computation_placer_hdr",
    ],
)

torch_tpu_cc_library(
    name = "tier2_compilation_cache",
    srcs = ["tier2_compilation_cache.cc"],
    hdrs = ["tier2_compilation_cache.h"],
    deps = [
        ":cache_key",
        ":compilation",
        ":env_vars",
        ":error_utils",
        ":fingerprint_utils",
        ":unique_file_descriptor",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/tsl/platform:env",
    ],
)

torch_tpu_cc_test(
    name = "tier2_compilation_cache_test",
    srcs = ["tier2_compilation_cache_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    deps = [
        ":cache_key",
        ":compilation",
        ":compilation_spec",
        ":dimension_types",
        ":env_vars",
        ":error_utils",
        ":tier2_compilation_cache",
        ":unique_file_descriptor",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

# Tests that the tier-2 cache is disabled by default for world size 1.
torch_tpu_cc_test(
    name = "tier2_compilation_cache_enablement_test_default_one_device",
    srcs = ["tier2_compilation_cache_enablement_test.cc"],
    # Do not set the TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE environment variable.
    args = [
        "--expected_tier2_cache_name=",
    ],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    deps = [
        ":flags",
        ":tier2_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

# Tests that the tier-2 cache is enabled by default when tier-3 is enabled.
torch_tpu_cc_test(
    name = "tier2_compilation_cache_enablement_test_enabled_by_tier3",
    srcs = ["tier2_compilation_cache_enablement_test.cc"],
    args = [
        "--expected_tier2_cache_name=default",
    ],
    # Do not set the TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE environment variable.
    env = {
        # This should force enable tier-2.
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/cache",
    },
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    deps = [
        ":flags",
        ":tier2_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

# Tests that the tier-2 cache is enabled by default for world size > 1.
torch_tpu_cc_test(
    name = "tier2_compilation_cache_enablement_test_default_multi_device",
    srcs = ["tier2_compilation_cache_enablement_test.cc"],
    # Do not set the TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE environment variable.
    args = [
        "--expected_tier2_cache_name=default",
    ],
    env = {
        "WORLD_SIZE": "4",
        "TORCH_TPU_TOPOLOGY": "2,2,1",
    },
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    deps = [
        ":flags",
        ":tier2_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

# Tests that the tier-2 cache can be explicitly disabled for world size 1.
torch_tpu_cc_test(
    name = "tier2_compilation_cache_enablement_test_disabled_one_device",
    srcs = ["tier2_compilation_cache_enablement_test.cc"],
    args = [
        "--expected_tier2_cache_name=",
    ],
    env = {
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "disabled",
    },
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    deps = [
        ":flags",
        ":tier2_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

# Tests that the tier-2 cache can be explicitly disabled for world size > 1.
torch_tpu_cc_test(
    name = "tier2_compilation_cache_enablement_test_disabled_multi_device",
    srcs = ["tier2_compilation_cache_enablement_test.cc"],
    args = [
        "--expected_tier2_cache_name=",
    ],
    env = {
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "disabled",
        "WORLD_SIZE": "4",
        "TORCH_TPU_TOPOLOGY": "2,2,1",
    },
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    deps = [
        ":flags",
        ":tier2_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

# Tests that the tier-2 cache can be explicitly enabled for world size 1.
torch_tpu_cc_test(
    name = "tier2_compilation_cache_enablement_test_enabled_one_device",
    srcs = ["tier2_compilation_cache_enablement_test.cc"],
    args = [
        "--expected_tier2_cache_name=my_cache",
    ],
    env = {
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "my_cache",
    },
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    deps = [
        ":flags",
        ":tier2_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

# Tests that the tier-2 cache can be explicitly enabled for world size > 1.
torch_tpu_cc_test(
    name = "tier2_compilation_cache_enablement_test_enabled_multi_device",
    srcs = ["tier2_compilation_cache_enablement_test.cc"],
    args = [
        "--expected_tier2_cache_name=my_cache2",
    ],
    env = {
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "my_cache2",
        "WORLD_SIZE": "4",
        "TORCH_TPU_TOPOLOGY": "2,2,1",
    },
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    deps = [
        ":flags",
        ":tier2_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "tier3_compilation_cache",
    srcs = ["tier3_compilation_cache.cc"],
    hdrs = ["tier3_compilation_cache.h"],
    deps = [
        ":cache_key",
        ":compilation",
        ":env_vars",
        ":error_utils",
        ":tier2_compilation_cache",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform/cloud:gcs_file_system",  # buildcleaner: keep - for GCS support
    ] + add_internal_filesystem_dependencies(),
)

torch_tpu_cc_library(
    name = "error_utils_test_helper",
    testonly = True,
    hdrs = ["error_utils_test_helper.h"],
    # This is needed, so that the build system places this header in the build directory, changing
    # the value of `__FILE__` macro.
    #
    # For example: given a Bazel project rooted at `torch_tpu/` directory, the value of `__FILE__`
    # macro inside `error_utils_test_helper.h` would be:
    #
    #   - WITHOUT `include_prefix`:
    #       torch_tpu/common/error_utils_test_helper.h
    #
    #   - WITH `include_prefix`:
    #       bazel-out/.../torch_tpu/common/testing/error_utils_test_helper.h
    #
    # Since this header will be used to show that the normalization of paths in the build directory
    # works, we add the `include_prefix` below.
    include_prefix = "torch_tpu/common/testing",
    deps = [
        ":error_utils",
        "@com_google_absl//absl/status",
    ],
)

torch_tpu_cc_test(
    name = "error_utils_cpp_context_test",
    srcs = [
        "error_utils_cpp_context_test.cc",
    ],
    deps = [
        ":error_utils",
        ":error_utils_test_helper",
        ":macro_utils",
        ":status_builder",
        ":utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "fingerprint_utils",
    srcs = ["fingerprint_utils.cc"],
    hdrs = ["fingerprint_utils.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
    ],
)

torch_tpu_cc_test(
    name = "fingerprint_utils_test",
    srcs = ["fingerprint_utils_test.cc"],
    deps = [
        ":dtype",
        ":fingerprint_utils",
        "//torch_tpu/ops:op_names",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@stablehlo//:attr_type_builder_util",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

torch_tpu_cc_library(
    name = "macro_utils",
    srcs = ["macro_utils.cc"],
    hdrs = ["macro_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

torch_tpu_cc_library(
    name = "fixed_size_span",
    hdrs = ["fixed_size_span.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "to_string",
    srcs = ["to_string.cc"],
    hdrs = ["to_string.h"],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":dimension_types",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

torch_tpu_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "env_vars",
    srcs = ["env_vars.cc"],
    hdrs = ["env_vars.h"],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/strings",
    ],
)

torch_tpu_cc_library(
    name = "flags",
    hdrs = ["flags.h"],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/flags:flag",
    ],
)

torch_tpu_cc_test(
    name = "flags_test",
    srcs = ["flags_test.cc"],
    deps = [
        ":flags",
        "@com_google_absl//absl/flags:flag",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "aten_utils",
    srcs = ["aten_utils.cc"],
    hdrs = ["aten_utils.h"],
    deps = [
        ":device_type",
        ":dtype",
        ":error_utils",
        ":to_string",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_library(
    name = "discovery",
    srcs = ["discovery.cc"],
    hdrs = ["discovery.h"],
    deps = [
        ":env_vars",
        ":error_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

torch_tpu_cc_test(
    name = "discovery_test",
    srcs = ["discovery_test.cc"],
    deps = [
        ":discovery",
        ":env_vars",
        ":error_utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = ["environment.h"],
    deps = [
        ":env_vars",
        "//torch_tpu/distributed/slicebuilder:discovery",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

torch_tpu_cc_test(
    name = "environment_test",
    srcs = ["environment_test.cc"],
    notest_oss = "Disabled during switch from opt-in to opt-out",
    deps = [
        ":env_vars",
        ":environment",
        "//torch_tpu/distributed/slicebuilder:discovery",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@xla//xla/tsl/platform:env",
    ],
)

# Note: workaround for all pytorch .so files required by torch_tpu
torch_tpu_cc_library(
    name = "torch_rpath_flags",
    linkopts = [
        "-Wl,-rpath='$$ORIGIN/../../torch/lib'",
    ],
    visibility = ["//visibility:private"],
)

pywrap_library(
    name = "pywrap_torch_tpu",
    extra_deps = [
        ":torch_rpath_flags",
        "//shims/torch:torch_python",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/batch_transfer:batch_transfer_impl",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/compiler_options:compiler_options_impl",
        "//torch_tpu/_internal/device:_device_ops_backend",
        "//torch_tpu/_internal/device_utils:annotations_py",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/dynamism:_tpu_torch_dynamism",
        "//torch_tpu/_internal/execution_mode:execution_mode_impl",
        "//torch_tpu/_internal/pallas:tpu_torch_pallas",
        "//torch_tpu/_internal/precision:precision_impl",
        "//torch_tpu/_internal/profiler:_profiler_backend",
        "//torch_tpu/_internal/sync:_tpu_torch_sync",
        "//torch_tpu/_internal/tracing:_tpu_torch_tracing",
    ],
)

pywrap_binaries(
    name = "pywrap_torch_tpu_binaries",
    dep = ":pywrap_torch_tpu",
    visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "compile_options_key",
    hdrs = ["compile_options_key.h"],
    deps = [
        ":fingerprint_utils",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/strings:str_format",
    ],
)

torch_tpu_cc_library(
    name = "cache_key",
    srcs = ["cache_key.cc"],
    hdrs = ["cache_key.h"],
    deps = [
        ":compile_options_key",
        ":dimension_types",
        ":dtype",
        ":error_utils",
        ":fingerprint_utils",
        ":shape",
        ":to_string",
        ":utils",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/ops:op_names",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:absl_vlog_is_on",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_library(
    name = "op_name_stack",
    srcs = ["op_name_stack.cc"],
    hdrs = ["op_name_stack.h"],
    deps = [
        ":flags",
        "//torch_tpu/ops:op_names",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

torch_tpu_cc_test(
    name = "cache_key_test",
    srcs = ["cache_key_test.cc"],
    deps = [
        ":cache_key",
        ":dimension_types",
        ":error_utils",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops/macros:kernel",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_test(
    name = "op_name_stack_test",
    srcs = ["op_name_stack_test.cc"],
    deps = [
        ":op_name_stack",
        "//torch_tpu/ops:op_names",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "compilation_cache",
    srcs = ["compilation_cache.cc"],
    hdrs = ["compilation_cache.h"],
    deps = [
        ":cache_key",
        ":compilation",
        ":compilation_spec",
        ":compile_options_key",
        ":dimension_types",
        ":dtype",
        ":env_vars",
        ":error_utils",
        ":flags",
        ":shape",
        ":thread_pool",
        ":tier2_compilation_cache",
        ":tier3_compilation_cache",
        ":to_string",
        ":utils",
        "//shims/torch:torch_headers",
        "//torch_tpu/_internal/dynamism:dynamism_ops",
        "//torch_tpu/common:contain",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@tsl//tsl/platform:numbers",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/hlo/utils/concurrency:concurrency_utils",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

torch_tpu_cc_library(
    name = "contain",
    srcs = ["contain_oss.cc"],
    hdrs = ["contain.h"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
    ],
)

torch_tpu_cc_library(
    name = "compilation_test_helper",
    testonly = True,
    hdrs = ["compilation_test_helper.h"],
    deps = [
        ":compilation",
        ":compilation_spec",
        "@com_google_absl//absl/log:absl_check",
    ],
)

torch_tpu_cc_test(
    name = "compilation_test",
    srcs = ["compilation_test.cc"],
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {"cpp_link.mem": "16g"},  # multiline select
    nobuild_oss = "Disabled during switch from opt-in to opt-out",
    notest_oss = "Do not enable this in OSS, instead use `oss_compilation_test` to run tests in the OSS TPU CI.",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":compilation",
        ":compilation_spec",
        ":compilation_test_helper",
        "//shims/torch:torch_headers",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_cc_test(
    name = "compilation_cache_test",
    srcs = ["compilation_cache_test.cc"],
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {"cpp_link.mem": "16g"},  # multiline select
    notest_oss = "Environment variable TPU_LIBRARY_PATH is not set. Can't load PJRT plugin TPU.",
    deps = [
        ":cache_key",
        ":compilation",
        ":compilation_cache",
        ":compilation_spec",
        ":compile_options_key",
        ":error_utils",
        ":flags",
        ":shape",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:contain",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_cc_test(
    name = "compilation_thread_num_test",
    srcs = ["compilation_thread_num_test.cc"],
    exec_properties = {"cpp_link.mem": "16g"},  # multiline select
    notest_oss = "Environment variable TPU_LIBRARY_PATH is not set. Can't load PJRT plugin TPU.",
    tags = [
        # Needs exclusive access to machine as we are testing hardware behavior. Without this,
        # `blaze test --runs_per_test` may fail.
        "exclusive",
    ],
    deps = [
        ":compilation_cache",
        ":flags",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

exports_files([
    "contain.h",
])

# TODO: b/479574836 - Using this test as a hack to run compilation_test.cc in TPU CI.
# This test should be removed once either an equivalent C++ libtpu API is available or
# compilation_test.cc is converted to a Python test.
oss_target(
    sh_test,
    name = "oss_compilation_test",
    srcs = ["//ci/tools:run_cc_tpu_test.sh"],
    args = ["$(location :compilation_test)"],
    data = [
        ":compilation_test",
        "@pypi_libtpu//:whl",
    ],
    tags = [
        # This is normally added by the torch_tpu_... rules
        "requires-tpu",
        "requires-tpu-v5lite",
    ],
)

torch_tpu_cc_test(
    name = "compilation_parse_xla_options_test",
    srcs = ["compilation_parse_xla_options_test.cc"],
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {"cpp_link.mem": "16g"},  # multiline select
    notest_oss = "Environment variable TPU_LIBRARY_PATH is not set. Can't load PJRT plugin TPU.",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":compilation",
        ":compilation_spec",
        ":compilation_test_helper",
        "//shims/torch:torch_headers",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_cc_test(
    name = "compile_options_key_test",
    srcs = ["compile_options_key_test.cc"],
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {"cpp_link.mem": "16g"},  # multiline select
    notest_oss = "Environment variable TPU_LIBRARY_PATH is not set. Can't load PJRT plugin TPU.",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":compilation",
        ":compilation_spec",
        ":compilation_test_helper",
        ":compile_options_key",
        ":fingerprint_utils",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

torch_tpu_cc_test(
    name = "tier3_compilation_cache_test",
    srcs = ["tier3_compilation_cache_test.cc"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/my_cache",
    },
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":cache_key",
        ":compilation",
        ":compilation_spec",
        ":dimension_types",
        ":env_vars",
        ":error_utils",
        ":tier2_compilation_cache",
        ":tier3_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# Like tier3_compilation_cache_test, but uses a remote filesystem for the tier-3 cache.
torch_tpu_cc_test(
    name = "remote_tier3_compilation_cache_test",
    srcs = ["tier3_compilation_cache_test.cc"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT":
        # Alternatively, set it to a GCS directory.
        # "$DATA_ROOT/$TMP_SUBDIR/ttl=1d/remote_tier3_compilation_cache_test"
        "",
    },
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-net:external",  # Requires remote file access.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":cache_key",
        ":compilation",
        ":compilation_spec",
        ":dimension_types",
        ":env_vars",
        ":error_utils",
        ":tier2_compilation_cache",
        ":tier3_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_test(
    name = "compilation_cache_remote_filesystem_test",
    srcs = ["compilation_cache_remote_filesystem_test.cc"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT":
        # Alternatively, set it to a GCS directory.
        # "$DATA_ROOT/$TMP_SUBDIR/ttl=1d/compilation_cache_remote_filesystem_test"
        "",
    },
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-net:external",  # Requires remote file access.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":cache_key",
        ":compilation",
        ":compilation_spec",
        ":env_vars",
        ":tier2_compilation_cache",  # buildcleaner: keep
        ":tier3_compilation_cache",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
    ],
)

torch_tpu_cc_library(
    name = "dynamism_utils",
    srcs = ["dynamism_utils.cc"],
    hdrs = ["dynamism_utils.h"],
    deps = [
        ":error_utils",
        "//shims/torch:torch_headers",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:traversal",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_broadcast_lowering",
    ],
)

torch_tpu_cc_test(
    name = "dynamism_utils_test",
    srcs = ["dynamism_utils_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Environment variable TPU_LIBRARY_PATH is not set. Can't load PJRT plugin TPU.",
    deps = [
        ":cache_key",
        ":dimension_types",
        ":dynamism_utils",
        ":error_utils",
        ":shape",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:traversal",
        "//torch_tpu/ops:binary",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:register",
        "@xla//xla/tsl/platform:statusor",
    ],
)

torch_tpu_cc_library(
    name = "context_manager",
    srcs = ["context_manager.cc"],
    hdrs = ["context_manager.h"],
    deps = [
        ":context_states",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
    ],
)

torch_tpu_cc_library(
    name = "device_type",
    hdrs = ["device_type.h"],
    deps = [
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
    ],
)

torch_tpu_package_end()
