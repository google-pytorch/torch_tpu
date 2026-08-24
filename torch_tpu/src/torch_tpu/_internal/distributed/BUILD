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

load(
    "//build_files:build_defs.bzl",
    "oss_target",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//shims/build_files:py_relocated_library.bzl", "py_relocated_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ],
)

py_relocated_library(
    name = "gpu_env",
    srcs = ["gpu_env.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "handshake",
    srcs = ["handshake.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//portpicker",
        "@pypi//pyzmq",
    ],
)

py_relocated_library(
    name = "spmd_util",
    srcs = [
        "spmd_util.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/csrc/internal/distributed:tpu_distributed",
        "//torch_tpu/src/torch_tpu/_internal/sync",
    ],
)

py_relocated_library(
    name = "device_utils",
    srcs = ["device_utils.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "@pypi//numpy",
    ],
)

py_relocated_library(
    name = "device_mesh",
    srcs = ["device_mesh.py"],
    py_library = torch_tpu_py_library,
    deps = [
        ":device_utils",
        "//shims/torch:pytorch",
        "@pypi//numpy",
    ],
)

py_relocated_library(
    name = "from_local",
    srcs = ["from_local.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/src/torch_tpu/_internal/execution_mode",
    ],
)

py_relocated_library(
    name = "process_group_utils",
    srcs = ["process_group_utils.py"],
    py_library = torch_tpu_py_library,
    deps = [
        ":handshake",
        "//shims/torch:pytorch",
        "//torch_tpu/src/torch_tpu/_internal/compile:collective_ops",
    ],
)

py_relocated_library(
    name = "torchcomm_tpu",
    srcs = ["torchcomm_tpu.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/csrc/internal/distributed:tpu_distributed",
    ],
)

oss_target(
    torch_tpu_py_library,
    name = "multiprocessing",
    srcs = ["multiprocessing.py"],
    deps = [
        "//shims/absl_py:app",
    ],
)

torch_tpu_package_end()
