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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end", "torch_tpu_py_library")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pybind_extension(
    name = "batch_transfer_impl",
    srcs = ["batch_transfer_impl.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/experimental/batch_transfer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

torch_tpu_py_library(
    name = "batch_transfer",
    srcs = [
        "__init__.py",
    ],
    deps = [":batch_transfer_impl"],
)

# This library is not used directly, but is needed for go/cclean.
torch_tpu_cc_library(
    name = "batch_transfer_impl_build",
    srcs = ["batch_transfer_impl.cc"],
    deps = [
        "//shims/pybind11",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/experimental/batch_transfer",
    ],
)

torch_tpu_cc_library(
    name = "tensor_transfer_bridge",
    srcs = ["tensor_transfer_bridge.cc"],
    hdrs = ["tensor_transfer_bridge.h"],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/pjrt:pjrt_utils",
        "//torch_tpu/runtime:backend",
        "//torch_tpu/runtime:device_buffer",
        "//torch_tpu/runtime:eager_runtime_adapter",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@xla//xla/tsl/concurrency:future",
    ],
)

torch_tpu_cc_test(
    name = "tensor_transfer_bridge_test",
    srcs = ["tensor_transfer_bridge_test.cc"],
    deps = [
        ":tensor_transfer_bridge",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:oss_test_main",
        "//torch_tpu/runtime:backend",
        "//torch_tpu/runtime:backend_config",
    ],
)

torch_tpu_package_end()
