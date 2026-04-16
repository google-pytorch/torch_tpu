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

"""Module for single-sourcing wheel dependencies."""

def _torch_tpu_deps_repo_impl(ctx):
    """Extract dependencies from pyproject.toml."""
    pyproject_path = ctx.path(ctx.attr.pyproject_toml)
    content = ctx.read(pyproject_path)
    deps = []
    in_project_section = False
    in_deps_list = False
    for line in content.splitlines():
        sline = line.strip()

        if not sline or sline.startswith("#"):
            continue

        # Exit section if another section starts
        if sline.startswith("["):
            in_project_section = (sline == "[project]")
            in_deps_list = False
            continue

        if in_project_section and sline.startswith("dependencies"):
            in_deps_list = True
            continue

        if in_deps_list:
            if sline.startswith("]"):
                in_deps_list = False
                continue

            # This handles lines like "numpy",
            if sline.startswith('"') or sline.startswith("'"):
                dep = sline.strip(",").strip('"').strip("'")
                deps.append(dep)
            continue

    if not deps:
        fail("Dependencies not found in pyproject.toml [project] section")

    ctx.file("BUILD.bazel", "")
    ctx.file("dependencies.bzl", "WHEEL_REQUIRES = {}".format(repr(deps)))

torch_tpu_deps_repo = repository_rule(
    implementation = _torch_tpu_deps_repo_impl,
    attrs = {
        "pyproject_toml": attr.label(mandatory = True, allow_single_file = True),
    },
)
