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
# lukeboyer
# cbasile-g

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

# Every layer family. Like perflab/models, the family modules are srcs of one
# library: binaries discover them by importing the package at runtime, so no
# module imports them statically. A new family adds its module here.
torch_tpu_py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "linear.py",
    ],
    allow_multiple_srcs = True,
    deps = [
        "//perflab/layer_utils:builders",
        "//perflab/layer_utils:configs",
        "//perflab/layer_utils:registration",
    ],
)

torch_tpu_package_end()
