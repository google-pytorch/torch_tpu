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
# yilingyuan
# zhanyong-wan
# haifeng-jin
# basioli-k

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
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/device:_device_interface",
        "//src/torch_tpu/_internal/device:_device_module",
        "//src/torch_tpu/_internal/device:_device_ops_backend",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_test(
    name = "cpu_backward_no_tpu_test",
    srcs = ["cpu_backward_no_tpu_test.py"],
    main = "cpu_backward_no_tpu_test.py",
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
    ],
)

torch_tpu_py_test(
    name = "device_module_test_no_tpu",
    srcs = ["device_module_no_tpu_test.py"],
    main = "device_module_no_tpu_test.py",
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:hardware",
    ],
)

torch_tpu_py_test(
    name = "legacy_device_apis_test",
    srcs = ["legacy_device_apis_test.py"],
    env = {
        "TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS": "1",
    },
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests:seed_test_utils",
    ],
)

test_suite(
    name = "device_module_tests",
    tests = [
        "device_module_test_no_tpu",
        "device_module_test_tpu",
        "legacy_device_apis_test",
    ],
)

torch_tpu_package_end()
