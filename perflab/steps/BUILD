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
# lizlian
# haifeng-jin

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//shims/py_rules:py_library.bzl", "py_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

py_library(
    # PY_LIBRARY_OK=Decouple from torch_tpu to avoid C++ dependency in coverage tests.
    name = "common",
    srcs = ["common.py"],
    type_checking = True,
    deps = [
        "//perflab:target",
    ],
)

py_library(
    # PY_LIBRARY_OK=Decouple from torch_tpu to avoid C++ dependency in coverage tests.
    name = "decode",
    srcs = ["decode.py"],
    type_checking = True,
    deps = [
        ":common",
        "//perflab:step_lib",
        "//perflab:target",
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

py_library(
    # PY_LIBRARY_OK=Keep standard_train C++-free for coverage tests.
    name = "standard_train",
    srcs = ["standard_train.py"],
    type_checking = True,
    deps = [
        ":common",
        "//perflab:loss",
        "//perflab:step_lib",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "steps",
    srcs = [
        "__init__.py",
        "forward.py",
        "train.py",
    ],
    allow_multiple_srcs = True,
    deps = [
        ":common",
        ":decode",  # build_cleaner: keep
        ":standard_train",
        "//perflab:loss",
        "//perflab:single_trace_trainer_adapter",
        "//perflab:step_lib",
        "//perflab:target",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_package_end()
