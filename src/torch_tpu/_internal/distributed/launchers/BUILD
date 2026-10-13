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

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "singlehost_wrapper",
    srcs = ["singlehost_wrapper.py"],
    deps = [
        "//src/torch_tpu/_internal/distributed",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_library(
    name = "multihost_wrapper",
    srcs = ["multihost_wrapper.py"],
    deps = [
        "//shims/absl_py/logging",
        "//src/torch_tpu/_internal/distributed",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_test(
    name = "singlehost_wrapper_test",
    srcs = ["singlehost_wrapper_test.py"],
    strict = True,
    deps = [
        ":singlehost_wrapper",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "multihost_wrapper_test",
    srcs = ["multihost_wrapper_test.py"],
    strict = True,
    deps = [
        ":multihost_wrapper",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_package_end()
