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
)
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "_device_module",
    srcs = ["_device_module.py"],
    deps = [
        ":_device_ops_backend",
        "//csrc/eager:amp_policies",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed:device_mesh",
        "//torch_tpu/_internal/precision",
        "//torch_tpu/_internal/stream:streams",
        "//torch_tpu/_internal/utils:annotations",
        "//torch_tpu/_internal/utils:hardware",
    ],
)

torch_tpu_py_library(
    name = "_tpu_backend_config",
    srcs = ["_tpu_backend_config.py"],
    deps = [
        ":_device_ops_backend",
        "//torch_tpu/_internal/utils:annotations",
    ],
)

pybind_extension(
    name = "_device_ops_backend",
    srcs = ["//csrc/internal/device:device_rt.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:cache_key",
        "//csrc/common:compilation",
        "//csrc/common:compilation_cache",
        "//csrc/common:device_type",
        "//csrc/common:discovery",
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:excess_precision",
        "//csrc/distributed:window_tpu",
        "//csrc/distributed/slicebuilder:discovery",
        "//csrc/eager:current_stream",
        "//csrc/eager:device_buffer",
        "//csrc/eager:device_gen_impl",
        "//csrc/eager:events_queue",
        "//csrc/eager:materialize",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/eager:tpu_hooks",
        "//csrc/internal/sync:sync_cc",
        "//csrc/pjrt:pjrt_state",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

torch_tpu_package_end()
