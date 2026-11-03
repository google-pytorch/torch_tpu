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

# `device_count_test.py` runs the same scenarios against `torch.cuda` and
# `torch.tpu`, so that the CUDA target documents the behavior TorchTPU should
# match.
torch_tpu_py_test(
    name = "device_count_test_cuda",
    srcs = ["device_count_test.py"],
    args = ["--device_type=cuda"],
    main = "device_count_test.py",
    nolocal = "Skipping in local runs by default as it requires --config=cuda.",
    tags = [
        "nozapfhahn",  # Requires --config=cuda; excluded from Zapfhahn coverage.
        # Requires multiple devices to tell the host's device count apart from
        # the visible device count.
        "requires-gpu-sm90-full:2",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests:accelerator_test_utils",
    ],
)

torch_tpu_py_test(
    name = "device_count_test_tpu",
    srcs = ["device_count_test.py"],
    args = ["--device_type=tpu"],
    main = "device_count_test.py",
    tags = [
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests:accelerator_test_utils",
    ],
)

# Like `device_count_test_tpu`, but on a worker that has one TPU chip. Forge
# leaves placeholder device files for the host's other chips, which TorchTPU
# must not count before the TPU runtime is initialized.
torch_tpu_py_test(
    name = "device_count_test_tpu_single_chip",
    srcs = ["device_count_test.py"],
    args = ["--device_type=tpu"],
    main = "device_count_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests:accelerator_test_utils",
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
