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

"""OSS implementation of py_platform_test."""

# Example wrapping standard rules_python rules.
load("@rules_python//python:defs.bzl", "py_test")

def py_platform_test(name, **kwargs):
    """OSS version of py_platform_test."""

    # Note: On OSS, we ignore platform-specific attributes or handle via select.
    # Falling back to regular py_test.
    kwargs.pop("platform", None)
    py_test(
        # PY_TEST_OK=needed for implementing py_platform_test.
        name = name,
        **kwargs
    )
