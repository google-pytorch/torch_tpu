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

"""Benchmark adapter for torch_tpu."""

# buildifier: disable=unused-load
load(
    "//build_files:build_defs.bzl",
    "oss_target",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

oss_target(
    torch_tpu_py_library,
    name = "benchmark_adapter",
    srcs = ["benchmark_adapter.py"],
)

oss_target(
    torch_tpu_py_library,
    name = "export_adapter",
    srcs = ["export_adapter.py"],
)

oss_target(
    torch_tpu_py_library,
    name = "xprof_adapter",
    srcs = ["xprof_adapter.py"],
)

torch_tpu_package_end()
