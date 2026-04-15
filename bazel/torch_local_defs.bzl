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

"""Starlark macro for defining local PyTorch targets."""

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")

def py_library_source_mapping(srcs, venv_mode = True):
    """Wrapper for lists of source files coming from python libraries.

    Args:
        srcs: A string or list of strings representing the source files.
        venv_mode: A boolean indicating whether to prefix the sources with "site-packages".

    Returns:
        The prefixed string or list of strings if venv_mode is True, otherwise the original srcs.
    """
    if venv_mode:
        prefix = "site-packages"
        if type(srcs) == type([]):
            return [prefix + "/" + src for src in srcs]
        return prefix + "/" + srcs
    return srcs

def define_torch_local(name, is_source_mode = True):
    """Defines the targets for the local PyTorch repository.

    Args:
        name: The name of this macro instance (required by Bazel conventions).
        is_source_mode: Whether the repo is in source (True) or wheel (False) mode.
    """

    # Expose all header files needed for torch/include.
    # depend on the Bazel versions to avoid ODR violations and missing headers.
    cc_library(
        name = "torch_headers",
        hdrs = native.glob(
            py_library_source_mapping([
                "include/**/*.h",
                "include/**/*.hpp",
                "include/**/*.hh",
                "include/**/*.hxx",
                "torch/**/*.h",
                "torch/**/*.hpp",
                "torch/**/*.hh",
                "torch/**/*.hxx",
                "torch/**/*.cuh",
                "c10/**/*.h",
                "c10/**/*.hpp",
                "aten/**/*.h",
                "aten/**/*.hpp",
                "aten/**/*.hh",
                "aten/**/*.hxx",
                "aten/**/*.inc",
                "aten/**/*.inl",
                "aten/**/*.tcc",
                "third_party/fmt/include/**/*.h",
                "third_party/onnx/**/*.h",
                "third_party/onnx/**/*.hpp",
                "third_party/pybind11/include/**/*.h",
                "build/**/*.h",
                "build/**/*.hpp",
                "build/**/*.tcc",
                "build/**/*.inc",
            ], venv_mode = True),
        ),
        includes = py_library_source_mapping([
            "include",  # Standard install layout
            "aten/src",
            "aten/src/ATen",
            "build",
            "build/include",
            "build/aten/src",
            "torch/csrc/api/include",
            "torch/csrc",
            "third_party/pybind11/include",  # pybind11 headers
            "third_party/kineto/libkineto/include",  # kineto headers
            "third_party/fmt/include",
        ]) + ["site-packages"] + ([
            py_library_source_mapping("torch/include"),
            py_library_source_mapping("torch/include/torch/csrc/api/include"),
            py_library_source_mapping("torch/include/kineto"),
        ] if not is_source_mode else []),
        deps = [
            ":kineto_headers_internal",
            "@com_google_protobuf//:protobuf",
        ],
    )

    # Expose all header files needed for kineto/include.
    cc_library(
        name = "kineto_headers_internal",
        hdrs = native.glob(
            py_library_source_mapping([
                "third_party/kineto/libkineto/include/**/*.h",
                "third_party/kineto/libkineto/include/**/*.hpp",
            ], venv_mode = True),
        ),
        strip_include_prefix = py_library_source_mapping("third_party/kineto/libkineto/include"),
        include_prefix = "kineto",
    )

    torch_library_names = [
        "libc10",
        "libtorch_cpu",
        "libtorch_python",
        "libtorch",
        "libshm",
        "libtorch_global_deps",
    ]

    for lib_name in torch_library_names:
        cc_import(
            name = lib_name,
            shared_library = py_library_source_mapping("torch/lib/%s.so" % lib_name),
        )

        native.filegroup(
            name = lib_name + "_file",
            srcs = [py_library_source_mapping("torch/lib/%s.so" % lib_name)],
        )

    # Catch-all target for convenience
    cc_library(
        name = "torch_libs",
        deps = [":" + lib_name for lib_name in torch_library_names],
    )

    native.filegroup(
        name = "torch_py_files",
        srcs = native.glob(py_library_source_mapping([
            "torch/**/*",
            "torchgen/**/*",
            "functorch/**/*",
        ])),
    )
    py_library(
        # PY_LIBRARY_OK=Upstream source not pytype compatible
        name = "torch",
        srcs = native.glob(py_library_source_mapping([
            "torch/**/*.py",
            "torchgen/**/*.py",
            "functorch/**/*.py",
        ])),
        data = native.glob(py_library_source_mapping([
            "torch/**/*.so*",
            "torch/**/*.pyd",
            "torch/**/*.pyi",
            "torch/**/*.yaml",
            "torch/**/*.json",
            "torch/**/*.jinja",
            "torch/bin/*",
            "torch/lib/*.so*",
            "torchgen/**/*.pyi",
            "torchgen/**/*.yaml",
            "torchgen/**/*.json",
            "functorch/**/*.pyi",
        ])),
        imports = ["site-packages"],
        experimental_venvs_site_packages = "@rules_python//python/config_settings:venvs_site_packages",
        visibility = ["//visibility:public"],
        deps = [
            "@pypi//fsspec",
            "@pypi//jinja2",
            "@pypi//networkx",
            "@pypi//numpy",
            "@pypi//sympy",
            "@pypi//typing_extensions",
            "@pypi//setuptools",
            "@pypi//filelock",
        ],
    )
