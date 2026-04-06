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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")

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
    target_compatible_with = ["@platforms//:incompatible"],  # copybara:comment(oss-only)
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/profiler",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
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
    target_compatible_with = ["@platforms//:incompatible"],  # copybara:comment(oss-only)
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/profiler",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/api",
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
        "//torch_tpu/_internal/profiler",
        "@pypi//portpicker",
    ],
)

torch_tpu_package_end()
