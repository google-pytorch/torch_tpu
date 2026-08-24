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

"""Shim for py_relocated_library in OSS."""

load("@rules_python//python:py_library.bzl", _default_py_library = "py_library")

def py_relocated_library(
        name,
        srcs = [],
        src_prefix = "",
        deps = [],
        visibility = None,
        tags = [],
        py_library = None,
        **kwargs):
    """Macro adapting relocated Python packages into a py_library in OSS.

    Args:
      name: Target name for the py_library.
      srcs: Python source files to relocate.
      src_prefix: Optional prefix path under package directory to strip (ignored in OSS).
      deps: Dependencies of the Python library (including extensions).
      visibility: Target visibility.
      tags: Target tags.
      py_library: Custom py_library rule/macro function to invoke. Defaults to standard py_library if None.
      **kwargs: Additional keyword arguments passed through to py_library.
    """
    if py_library == None:
        py_library = _default_py_library

    # Use standard py_library with imports to support src/ layout in OSS.
    pkg_path = native.package_name()
    marker = "src/torch_tpu"
    if marker in pkg_path:
        index = pkg_path.find(marker)
        after_marker = pkg_path[index + len(marker):].strip("/")
        if after_marker:
            num_up = len(after_marker.split("/")) + 1
        else:
            num_up = 1
        imports = ["/".join([".."] * num_up)]
    else:
        imports = []

    imports_arg = kwargs.pop("imports", [])
    if imports:
        imports_arg.extend(imports)

    py_library(
        # PY_LIBRARY_OK=Calling injected rule parameter
        name = name,
        srcs = srcs,
        deps = deps,
        visibility = visibility,
        imports = imports_arg,
        tags = tags,
        **kwargs
    )
