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

def _adjust_kwargs(kwargs):
    kwargs.pop("type_checking", None)
    return kwargs

def pytype_strict_binary(**kwargs):
    py_binary(**_adjust_kwargs(kwargs))

def pytype_strict_library(**kwargs):
    py_library(
        # PY_LIBRARY_OK=oss
        **_adjust_kwargs(kwargs)
    )

def pytype_library(**kwargs):
    py_library(
        # PY_LIBRARY_OK=oss
        **_adjust_kwargs(kwargs)
    )

def pytype_strict_contrib_test(**kwargs):
    py_test(
        # PY_TEST_OK=oss
        **_adjust_kwargs(kwargs)
    )
