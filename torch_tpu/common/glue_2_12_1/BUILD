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

# Per-version wheel pywrap_library. The PyTorch version comes from this package's
# name (glue_2_12_1 -> 2.12.1); versioned_glue_library supplies the rest. See
# //torch_tpu/common:glue.bzl and //shims/pybind11:pybind.bzl for the design.
load("//torch_tpu/common:glue.bzl", "versioned_glue_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

versioned_glue_library()
