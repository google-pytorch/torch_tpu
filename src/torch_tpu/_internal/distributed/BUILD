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
    "torch_tpu_py_test",
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
    name = "distributed",
    srcs = ["distributed.py"],
    deps = [
        ":tpu_distributed",
        "//shims/absl_py/logging",
        "//src/torch_tpu/_internal:env",
        "//src/torch_tpu/_internal:env_utils",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:annotations",
        "//src/torch_tpu/_internal/utils:hardware",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "distributed_test",
    srcs = ["distributed_test.py"],
    strict = True,
    deps = [
        ":distributed",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:env",
        "//src/torch_tpu/_internal:env_utils",
        "//src/torch_tpu/_internal/utils:hardware",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "gpu_env",
    srcs = ["gpu_env.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal:env",
    ],
)

torch_tpu_py_library(
    name = "handshake",
    srcs = ["handshake.py"],
    deps = [
        ":process_group_utils",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal:env",
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
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:annotations",
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
        ":tpu_distributed",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "from_local",
    srcs = ["from_local.py"],
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal:env",
        "//src/torch_tpu/_internal:warnings",
        "//src/torch_tpu/_internal/utils:tt_warnings",
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
        "//src/torch_tpu/_internal:env",
    ],
)

oss_target(
    torch_tpu_py_library,
    name = "multiprocessing",
    srcs = ["multiprocessing.py"],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
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
    srcs = ["//csrc/internal/distributed:tpu_distributed.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:error_utils",
        "//csrc/common:pybind_error_utils",
        "//csrc/distributed:process_group_tpu",
        "//csrc/distributed:utils",
        "//csrc/pjrt:pjrt_state",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

torch_tpu_package_end()
