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

"""Stub for the pybind11 extension."""

load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", _pybind_extension = "pybind_extension")

def pybind_extension(name, **kwargs):
    """Generates a suite of PyTorch C++ bindings.

    Args:
        name: Name of the target.
        **kwargs: Additional keyword arguments to pass to the underlying pybind_extension rule.
    """

    # On OSS we have not needed any python dependencies for these. If that
    # changes we will need to incorporate these dependencies.
    kwargs.pop("py_deps", None)

    # Set wrap_py_init = False by default to ensure C++ extension code (which depends on PyTorch)
    # is kept in the versioned .so files instead of being merged into the common library.
    kwargs.setdefault("wrap_py_init", False)

    linkopts = kwargs.pop("linkopts", [])
    linkopts.append("-Wl,--allow-shlib-undefined")
    kwargs["linkopts"] = linkopts

    _pybind_extension(name = name, **kwargs)
