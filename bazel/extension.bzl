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

"""Module extension to bridge legacy WORKSPACE repositories into Bzlmod."""

load("//bazel:wheel_deps.bzl", "torch_tpu_deps_repo")
load("//bazel:wheel_version.bzl", "torch_tpu_version_repo")

def _internal_deps_impl(_ctx):
    torch_tpu_deps_repo(
        name = "torch_tpu_deps",
        pyproject_toml = "//:pyproject.toml",
        requirements_txt = "//requirements:requirements.txt",
    )
    torch_tpu_version_repo(
        name = "torch_tpu_version",
        pyproject_toml = "//:pyproject.toml",
    )

internal_deps = module_extension(
    implementation = _internal_deps_impl,
)
