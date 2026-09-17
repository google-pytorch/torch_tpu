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

"""Shim for py_relocated_pybind_extension in OSS."""

load("//shims/pybind11:pybind.bzl", _default_pybind_extension = "pybind_extension")

def py_relocated_pybind_extension(
        name,
        srcs = [],
        src_dir = "src",  # @unused
        deps = [],
        visibility = None,
        tags = [],
        pybind_extension = None,
        py_extension_module_name = None,
        **kwargs):
    """Macro adapting relocated pybind11 extensions into a pybind_extension in OSS.

    Args:
      name: Target name for the pybind_extension.
      srcs: C++ source files with pybind11 bindings.
      src_dir: Source layout directory segment (ignored in OSS).
      deps: Dependencies for the extension.
      visibility: Target visibility.
      tags: Target tags.
      pybind_extension: Custom pybind_extension rule/macro function to invoke. Defaults to standard pybind_extension if None.
      py_extension_module_name: Optional module name override. Defaults to target name if None.
      **kwargs: Additional keyword arguments passed through to pybind_extension.
    """
    if pybind_extension == None:
        pybind_extension = _default_pybind_extension

    if py_extension_module_name != None:
        kwargs["py_extension_module_name"] = py_extension_module_name

    pybind_extension(
        name = name,
        srcs = srcs,
        deps = deps,
        visibility = visibility,
        tags = tags,
        **kwargs
    )
