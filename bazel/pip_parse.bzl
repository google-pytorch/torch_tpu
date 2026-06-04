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

"""WORKSPACE-phase code for pip_parse setup."""

load("@python_version_repo//:py_version.bzl", "REQUIREMENTS_WITH_LOCAL_WHEELS")
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")
load(
    "@xla//third_party/py:python_init_toolchains.bzl",
    "get_toolchain_name_per_python_version",
)

_EXTRA_TORCH_BUILD = """\

# Added by //bazel:pip_parse.bzl
load("@//bazel:pip_parse_targets.bzl", "define_extra_torch_targets")

define_extra_torch_targets()

"""

def torch_tpu_pip_parse():
    # IMPORTANT: Added content should do nothing more than load from a bzl file
    # and call a macro to define additional targets. This keeps the workspace
    # and loading phases separate, which avoids having to re-process the
    # pypi repositories themselves (e.g. re-extract them, which is time
    # consuming and expensive).
    annotations = {
        "torch": package_annotation(
            additive_build_content = _EXTRA_TORCH_BUILD,
        ),
    }

    pip_parse(
        name = "pypi",
        annotations = annotations,
        # NOTE: The "python" string needs to match the name prefix used
        # in python_init_repositories
        python_interpreter_target = "@{}_host//:python".format(
            get_toolchain_name_per_python_version("python"),
        ),
        requirements_lock = REQUIREMENTS_WITH_LOCAL_WHEELS,
        extra_pip_args = [
            "--index-url",
            "https://pypi.org/simple",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        # Additional targets added via `annotations` need to listed here so
        # they're accessible via the hub `@pypi` repo targets.
        extra_hub_aliases = {
            "torch": [
                "torch_headers",
                "torch_libs",
            ],
        },
    )
