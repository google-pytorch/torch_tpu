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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_strict_library(
    name = "device_module_testing",
    testonly = True,
    srcs = ["device_module_testing.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/device:_device_module",
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
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
    ],
)

torch_tpu_py_test(
    name = "device_module_test_gpu",
    srcs = ["device_module_gpu_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND": "1",
    },
    main = "device_module_gpu_test.py",
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":device_module_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/stream_executor:cuda_platform",
    ],
)

torch_tpu_py_test(
    name = "device_module_test_cpu",
    srcs = ["device_module_cpu_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND": "1",
    },
    main = "device_module_cpu_test.py",
    tags = [],
    deps = [
        ":device_module_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
    ],
)

test_suite(
    name = "device_module_tests",
    tests = [
        "device_module_test_cpu",
        "device_module_test_gpu",
        "device_module_test_tpu",
    ],
)

torch_tpu_package_end()
