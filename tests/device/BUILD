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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "device_module_testing",
    testonly = True,
    srcs = ["device_module_testing.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/device:_device_module",
        "//src/torch_tpu/_internal/device:_device_ops_backend",
        "//tests:oss_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "device_module_test_tpu",
    srcs = ["device_module_tpu_test.py"],
    main = "device_module_tpu_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":device_module_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/device:_device_module",
        "//src/torch_tpu/_internal/device:_device_ops_backend",
    ],
)

test_suite(
    name = "device_module_tests",
    tests = [
        "device_module_test_tpu",
    ],
)

torch_tpu_package_end()
