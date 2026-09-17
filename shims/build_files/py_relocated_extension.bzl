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

"""Shim for py_relocated_extension in OSS."""

load("@rules_python//python:py_extension.bzl", _default_py_extension = "py_extension")

def py_relocated_extension(
        name,
        srcs = [],
        src_dir = "src",  # @unused
        deps = [],
        visibility = None,
        tags = [],
        py_extension = None,
        module_name = None,
        **kwargs):
    """Macro adapting relocated C extensions into a py_extension in OSS.

    Args:
      name: Target name for the py_extension.
      srcs: C extension source files.
      src_dir: Source layout directory segment (ignored in OSS).
      deps: Dependencies for the extension.
      visibility: Target visibility.
      tags: Target tags.
      py_extension: Custom py_extension rule/macro function to invoke. Defaults to standard py_extension if None.
      module_name: Optional module name override. Defaults to target name if None.
      **kwargs: Additional keyword arguments passed through to py_extension.
    """
    if py_extension == None:
        py_extension = _default_py_extension

    m_name = module_name if module_name else name
    py_extension(
        name = name,
        module_name = m_name,
        srcs = srcs,
        deps = deps,
        visibility = visibility,
        tags = tags,
        **kwargs
    )
