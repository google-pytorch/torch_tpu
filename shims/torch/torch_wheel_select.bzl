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

"""select() builders that route torch deps by the //:_torch_version flag.

The multi-ABI wheel builds one glue per PyTorch version (see
//shims/pybind11:pybind.bzl). For each extra (non-default) version, torch deps
must resolve to that version's wheel; the default version falls through to the
caller-supplied base rows (Python-version selection, local torch, default).

The extra versions come from //bazel:pytorch_versions.bzl, the single source of
truth, so adding a version updates every torch shim at once.
"""

load("//bazel:pytorch_versions.bzl", "EXTRA_PYTORCH_VERSIONS")
load("//bazel:supported_python_versions.bzl", "DEFAULT_PYTHON_VERSION")

# The Python version whose wheel every extra torch version currently resolves
# to: the default Python, since the extra versions are not yet built per
# Python. This is the single place to change when they are: switch to keying
# on the (torch_version, python_version) pair.
_EXTRA_TORCH_PY = DEFAULT_PYTHON_VERSION.replace(".", "")

def _torch_version_setting(version):
    return "//:torch_" + version.replace(".", "_")

def _extra_wheel(version, target):
    return "@pypi_torch_{}_{}//:{}".format(
        version.replace(".", "_"),
        _EXTRA_TORCH_PY,
        target,
    )

def torch_lib_select(target, base):
    """A select() dict choosing `target` from each extra version's wheel.

    For single-label attributes (e.g. an alias `actual`). `base` supplies the
    rows for the default torch version and is returned unchanged for it.

    Args:
      target: the wheel target name to pick, e.g. "libc10".
      base: the select() dict for the default torch version.
    """
    rows = {
        _torch_version_setting(v): _extra_wheel(v, target)
        for v in EXTRA_PYTORCH_VERSIONS
    }
    rows.update(base)
    return rows

def torch_headers_select(base, extra = []):
    """A select() dict choosing torch_headers from each extra version's wheel.

    For list-valued header `deps`. Each extra-version row is
    ["<wheel>//:torch_headers"] + `extra`; `base` supplies the default-version
    rows and is returned unchanged.

    Args:
      base: the select() dict for the default torch version.
      extra: labels appended to every extra-version row (e.g. the Python C
        headers that accompany torch_headers).
    """
    rows = {
        _torch_version_setting(v): [_extra_wheel(v, "torch_headers")] + extra
        for v in EXTRA_PYTORCH_VERSIONS
    }
    rows.update(base)
    return rows
