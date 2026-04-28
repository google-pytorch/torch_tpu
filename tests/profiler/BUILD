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
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/profiler",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "profiler_api_test",
    srcs = ["profiler_api_test.py"],
    main = "profiler_api_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/profiler",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "//torch_tpu",
        "//torch_tpu/_internal/profiler",
        "@pypi//portpicker",
    ],
)

torch_tpu_cc_test(
    name = "xprof_callback_handler_test",
    srcs = ["xprof_callback_handler_test.cc"],
    deps = [
        "//shims/torch",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/_internal/profiler:xprof_callback_handler",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_test(
    name = "tpu_profiler_plugin_test",
    srcs = ["tpu_profiler_plugin_test.cc"],
    deps = [
        "//shims/torch:kineto",
        "//torch_tpu/_internal/profiler:tpu_profiler_plugin_cc",
        "//torch_tpu/common:utils",
        "@com_google_googletest//:gtest_main",
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
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/sync",
    ],
)

torch_tpu_package_end()
