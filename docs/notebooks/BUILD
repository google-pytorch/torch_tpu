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

"""Documentation for TorchTPU."""

load("//build_files:build_defs.bzl", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/torch_tpu/docs/notebooks:__subpackages__"],
)

# Make BUILD file maintenance tools ignore all *.py files as we don't
# plan to run them as part of the build.
filegroup(
    name = "py_files",
    srcs = glob(["**/*.py"]),
    tags = ["ignore_srcs"],
    visibility = ["//visibility:private"],
)

torch_tpu_package_end()
