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

"""Stubs for Google3 Python and pytype rules."""

load("@rules_python//python:py_binary.bzl", _rules_python_py_binary = "py_binary")
load("@rules_python//python:py_library.bzl", _rules_python_py_library = "py_library")
load("@rules_python//python:py_test.bzl", _rules_python_py_test = "py_test")
load("//:visibility.bzl", "BZL_VISIBILITY")

visibility(BZL_VISIBILITY)

# Google3-only attributes on rules_python's py_library/py_binary/py_test that
# upstream OSS @rules_python does not accept.
_GOOGLE3_ONLY_ATTRS = (
    "pyi_deps",
    "pyi_srcs",
    "strict_deps",
    "type_checking",
)

def _add_src_imports(kwargs):
    package = native.package_name()
    if package.startswith("src/"):
        depth = len(package.split("/"))
        root_rel = "/".join([".." for _ in range(depth)])
        imports = list(kwargs.get("imports", []))
        if root_rel:
            imports.append(root_rel + "/src")
        else:
            imports.append("src")
        kwargs["imports"] = imports
    elif package == "examples" or package.startswith("examples/"):
        imports = list(kwargs.get("imports", []))
        if "." not in imports:
            imports.append(".")
        kwargs["imports"] = imports

def _prepare_kwargs(kwargs):
    for attr in _GOOGLE3_ONLY_ATTRS:
        kwargs.pop(attr, None)
    _add_src_imports(kwargs)

def py_binary(**kwargs):
    _prepare_kwargs(kwargs)
    _rules_python_py_binary(**kwargs)

def py_library(**kwargs):  # PY_LIBRARY_OK=OSS rules_python shim macro
    _prepare_kwargs(kwargs)
    _rules_python_py_library(**kwargs)

def py_test(**kwargs):
    _prepare_kwargs(kwargs)
    _rules_python_py_test(**kwargs)

pytype_strict_binary = py_binary
pytype_strict_library = py_library
pytype_library = py_library
pytype_strict_contrib_test = py_test
