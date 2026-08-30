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

"""Stubs for pytype rules."""

load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("//:visibility.bzl", "BZL_VISIBILITY")

visibility(BZL_VISIBILITY)

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

def pytype_strict_binary(**kwargs):
    _add_src_imports(kwargs)
    py_binary(**kwargs)

def pytype_strict_library(**kwargs):  # PY_LIBRARY_OK=OSS pytype stub macro definition
    _add_src_imports(kwargs)
    py_library(
        # PY_LIBRARY_OK=OSS pytype stub macro
        **kwargs
    )

def pytype_library(**kwargs):  # PY_LIBRARY_OK=OSS pytype stub macro definition
    _add_src_imports(kwargs)
    py_library(
        # PY_LIBRARY_OK=OSS pytype stub macro
        **kwargs
    )

def pytype_strict_contrib_test(**kwargs):
    _add_src_imports(kwargs)
    py_test(
        # PY_TEST_OK=oss
        **kwargs
    )
