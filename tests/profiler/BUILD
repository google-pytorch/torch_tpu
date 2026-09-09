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

load("//build_files:build_defs.bzl", "torch_tpu_cc_test", "torch_tpu_package_end", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
)

torch_tpu_py_test(
    name = "profiler_impl_test",
    srcs = ["profiler_impl_test.py"],
    main = "profiler_impl_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/profiler",
        "//torch_tpu/_internal/sync",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "profiler_api_test",
    srcs = ["profiler_api_test.py"],
    main = "profiler_api_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/profiler",
        "//torch_tpu/_internal/profiler:profiler_config",
        "//torch_tpu/_internal/sync",
        "@pypi//portpicker",
        "@tsl//tsl/profiler/protobuf:xplane_proto_py_proto",
    ],
)

torch_tpu_py_test(
    name = "profiler_backend_test",
    srcs = ["profiler_backend_test.py"],
    main = "profiler_backend_test.py",
    notest_oss = "has import errors in OSS",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/profiler:_profiler_backend",
        "@pypi//portpicker",
    ],
)

torch_tpu_cc_test(
    name = "xprof_callback_handler_test",
    srcs = ["xprof_callback_handler_test.cc"],
    deps = [
        "//csrc/common:oss_test_main",
        "//csrc/internal/profiler:xprof_callback_handler",
        "//shims/torch",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
    ],
)

torch_tpu_cc_test(
    name = "tpu_profiler_plugin_test",
    srcs = ["tpu_profiler_plugin_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    requires_libtpu = True,
    deps = [
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "//csrc/internal/profiler:tpu_profiler_plugin_cc",
        "//shims/torch:kineto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
)

# Tests that base output dir can be configured via legacy env var.
torch_tpu_cc_test(
    name = "profiler_output_dir_test_legacy",
    srcs = ["profiler_output_dir_test.cc"],
    args = [
        "--expected_output_dir=/legacy_profiler_dir",
    ],
    env = {
        "TPU_PROFILER_OUTPUT_DIR": "/legacy_profiler_dir",
    },
    deps = [
        "//csrc/common:flags",
        "//csrc/common:oss_test_main",
        "//csrc/internal/profiler:tpu_profiler_plugin_cc",
        "@com_google_absl//absl/flags:flag",
    ],
)

# Tests that base output dir can be configured via TORCH_TPU_PROFILER_OUTPUT_DIR.
torch_tpu_cc_test(
    name = "profiler_output_dir_test_public",
    srcs = ["profiler_output_dir_test.cc"],
    args = [
        "--expected_output_dir=/public_profiler_dir",
    ],
    env = {
        "TORCH_TPU_PROFILER_OUTPUT_DIR": "/public_profiler_dir",
    },
    deps = [
        "//csrc/common:flags",
        "//csrc/common:oss_test_main",
        "//csrc/internal/profiler:tpu_profiler_plugin_cc",
        "@com_google_absl//absl/flags:flag",
    ],
)

# Tests that TORCH_TPU_PROFILER_OUTPUT_DIR takes precedence over TPU_PROFILER_OUTPUT_DIR.
torch_tpu_cc_test(
    name = "profiler_output_dir_test_precedence",
    srcs = ["profiler_output_dir_test.cc"],
    args = [
        "--expected_output_dir=/public_profiler_dir",
    ],
    env = {
        "TORCH_TPU_PROFILER_OUTPUT_DIR": "/public_profiler_dir",
        "TPU_PROFILER_OUTPUT_DIR": "/legacy_profiler_dir",
    },
    deps = [
        "//csrc/common:flags",
        "//csrc/common:oss_test_main",
        "//csrc/internal/profiler:tpu_profiler_plugin_cc",
        "@com_google_absl//absl/flags:flag",
    ],
)

torch_tpu_py_test(
    name = "profiler_integration_test",
    timeout = "long",
    srcs = ["profiler_integration_test.py"],
    main = "profiler_integration_test.py",
    notest_oss = "Skipped until OSS migrates to torch 2.12 (TODO(b/499296481))",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/profiler:profiler_config",
        "//torch_tpu/_internal/sync",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_py_proto",
        "@tsl//tsl/profiler/protobuf:xplane_proto_py_proto",
    ],
)

torch_tpu_package_end()
