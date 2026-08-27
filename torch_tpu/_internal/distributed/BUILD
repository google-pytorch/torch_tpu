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
    "if_oss",
    "oss_target",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ],
)

torch_tpu_py_library(
    name = "gpu_env",
    srcs = ["gpu_env.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "handshake",
    srcs = ["handshake.py"],
    deps = [
        ":process_group_utils",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "@pypi//portpicker",
        "@pypi//pyzmq",
    ],
)

torch_tpu_py_library(
    name = "spmd_util",
    srcs = [
        "spmd_util.py",
    ],
    deps = [
        ":tpu_distributed",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/sync",
    ],
)

torch_tpu_py_library(
    name = "device_utils",
    srcs = ["device_utils.py"],
    deps = [
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "device_mesh",
    srcs = ["device_mesh.py"],
    deps = [
        ":device_utils",
        "//shims/torch:pytorch",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "from_local",
    srcs = ["from_local.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/execution_mode",
    ],
)

torch_tpu_py_library(
    name = "process_group_utils",
    srcs = ["process_group_utils.py"],
    deps = [
        ":collective_ops",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "torchcomm_tpu",
    srcs = ["torchcomm_tpu.py"],
    deps = if_oss(
        [],
        ["@pypi//torchcomms"],
    ) + [
        ":tpu_distributed",  # buildcleaner: keep
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
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

torch_tpu_py_library(
    name = "collective_ops",
    srcs = ["collective_ops.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

pybind_extension(
    name = "tpu_distributed",
    srcs = ["//torch_tpu/csrc/internal/distributed:tpu_distributed.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/distributed:process_group_tpu",
        "//torch_tpu/csrc/distributed:utils",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
    ],
)

torch_tpu_package_end()
