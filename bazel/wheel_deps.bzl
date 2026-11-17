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

"""Module for parsing wheel dependencies."""

def _unquote_toml_string(line):
    s = line.strip().rstrip(",").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1].strip()
    return s

def _torch_tpu_deps_repo_impl(ctx):
    # Parse pyproject.toml groupings
    pyproject_path = ctx.path(ctx.attr.pyproject_toml)
    content = ctx.read(pyproject_path)

    dependencies = []
    optional_deps = {}
    current_section = None
    current_list_name = None

    for line in content.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        # Keep track of the current TOML table/section
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            current_list_name = None
            continue

        # Parse [project] -> dependencies
        if current_section == "project":
            if line.startswith("dependencies = ["):
                current_list_name = "dependencies"
                continue
            elif current_list_name == "dependencies":
                if line == "]":
                    current_list_name = None
                else:
                    raw_dep = _unquote_toml_string(line)
                    if raw_dep:
                        dependencies.append(raw_dep)

            # Parse [project.optional-dependencies] -> dev, benchmark, etc.
        elif current_section == "project.optional-dependencies":
            if "=" in line and line.endswith("["):
                current_list_name = line.split("=")[0].strip()
                optional_deps[current_list_name] = []
                continue
            elif current_list_name:
                if line == "]":
                    current_list_name = None
                else:
                    raw_dep = _unquote_toml_string(line)
                    if raw_dep:
                        optional_deps[current_list_name].append(raw_dep)

    if not dependencies:
        fail("Dependencies not found in pyproject.toml [project] section")

    # Export the variables
    ctx.file("BUILD.bazel", "")

    bzl_content = "WHEEL_REQUIRES = {}\n".format(dependencies)
    for opt_name, opt_deps in optional_deps.items():
        bzl_content += "WHEEL_{}_REQUIRES = {}\n".format(opt_name.upper().replace("-", "_"), opt_deps)

    ctx.file("dependencies.bzl", bzl_content)

torch_tpu_deps_repo = repository_rule(
    implementation = _torch_tpu_deps_repo_impl,
    attrs = {
        "pyproject_toml": attr.label(mandatory = True, allow_single_file = True),
        "requirements_txt": attr.label(mandatory = True, allow_single_file = True),
    },
)
