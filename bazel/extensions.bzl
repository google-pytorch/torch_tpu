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
"""Module extension for Torch TPU dependencies."""

load("//bazel:wheel_deps.bzl", "torch_tpu_deps_repo")

def _torch_tpu_deps_ext_impl(_ctx):
    # This is a repository rule and evaluated in the loading phase, so it cannot
    # use dynamic select() blocks. We use the Python 3.14 lock file as a static
    # baseline to parse wheel metadata requirements, which are identical across
    # Python versions. Actual compiler and test runtimes leverage version-specific
    # parse rules in MODULE.bazel.
    torch_tpu_deps_repo(
        name = "torch_tpu_deps",
        pyproject_toml = "//:pyproject.toml",
        requirements_txt = "//requirements:requirements_3_14.txt",
    )

torch_tpu_deps_ext = module_extension(
    implementation = _torch_tpu_deps_ext_impl,
)
