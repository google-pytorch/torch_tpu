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

load("@rules_python//python:py_library.bzl", _default_py_library = "py_library")
load("//shims/pybind11:pybind.bzl", _default_pybind_extension = "pybind_extension")

def py_relocated_pybind_extension(
        name,
        srcs = [],
        deps = [],
        visibility = None,
        tags = [],
        pybind_extension = None,
        py_extension_module_name = None,
        **kwargs):
    """Macro adapting relocated pybind11 extensions into a py_library in OSS.

    Args:
      name: Target name for the pybind_extension.
      srcs: C++ source files with pybind11 bindings.
      deps: Dependencies for the extension.
      visibility: Target visibility.
      tags: Target tags.
      pybind_extension: Custom pybind_extension rule/macro function to invoke. Defaults to standard pybind_extension if None.
      py_extension_module_name: Optional module name override. Defaults to target name if None.
      **kwargs: Additional keyword arguments passed through to pybind_extension.
    """
    if pybind_extension == None:
        pybind_extension = _default_pybind_extension

    m_name = py_extension_module_name if py_extension_module_name else name
    ext_impl_name = "_" + name + "_ext_impl"

    pybind_extension(
        name = ext_impl_name,
        py_extension_module_name = m_name,
        srcs = srcs,
        deps = deps,
        visibility = ["//visibility:private"],
        tags = tags,
        **kwargs
    )

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

    _default_py_library(
        name = name,
        deps = [":" + ext_impl_name],
        visibility = visibility,
        imports = imports,
        tags = tags,
    )
