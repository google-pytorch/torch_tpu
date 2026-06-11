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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "materialize_new",
    srcs = ["materialize_new.cc"],
    hdrs = ["materialize_new.h"],
    deps = [
        "//shims/torch:torch_headers",
        "//torch_tpu/common:compilation",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:eager_mode",
        "//torch_tpu/eager:materialize_common",
        "//torch_tpu/eager:split_traversal",
        "//torch_tpu/eager:structured_log_buffer",
        "//torch_tpu/eager:traversal",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_test(
    name = "materialize_new_test",
    srcs = ["materialize_new_test.cc"],
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for ASan to avoid CppLink OOM.
    },
    notest_oss = "Environment variable TPU_LIBRARY_PATH is not set. Can't load PJRT plugin TPU.",
    deps = [
        "//shims/torch:aten_headers",
        "//torch_tpu/_internal/compile:compiled_mode",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:compilation",
        "//torch_tpu/common:compilation_cache",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:shape",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/eager:tpu_aten_kernels",
        "//torch_tpu/eager:tpu_hooks",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@xla//xla/tsl/platform:statusor",
    ],
)

torch_tpu_cc_test(
    name = "materialize_recovery_test",
    srcs = ["materialize_recovery_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Environment variable TPU_LIBRARY_PATH is not set. Can't load PJRT plugin TPU.",
    deps = [
        ":materialize_new",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:compilation_cache",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:shape",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:materialize",
        "//torch_tpu/eager:structured_log_buffer",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/eager:tpu_hooks",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@xla//xla/tsl/platform:statusor",
    ],
)

torch_tpu_package_end()
