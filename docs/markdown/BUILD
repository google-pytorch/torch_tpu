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

"""Markdown documentation for TorchTPU."""

load("//build_files:build_defs.bzl", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

# The list of public APIs, whose generated parts `tests/api:tt_apis_test` checks and
# `tests/api:scan_tt_apis` updates.
exports_files(
    ["public_apis.md"],
    visibility = ["//tests/api:__pkg__"],
)

torch_tpu_package_end()
