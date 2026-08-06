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
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
    ],
)

torch_tpu_py_library(
    name = "annotations",
    srcs = ["annotations.py"],
    deps = [
        ":annotations_py",
    ],
)

pybind_extension(
    name = "annotations_py",
    srcs = ["annotations_py.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/parser:hlo_parser",
    ],
)

torch_tpu_py_test(
    name = "annotations_test",
    srcs = ["annotations_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":annotations",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//torch_tpu/_internal/sync",
    ],
)

torch_tpu_package_end()
