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

def define_torch_local(is_source_mode = True):
    """Defines the targets for the local PyTorch repository.

    Args:
        is_source_mode: Whether the repo is in source (True) or wheel (False) mode.
    """

    # Expose all header files in torch/include.
    # We exclude conflicting third-party deps and explicitly
    # depend on the Bazel versions to avoid ODR violations and missing headers.
    cc_library(
        name = "torch_headers",
        hdrs = native.glob(
            [
                "site-packages/include/**",
                "site-packages/torch/**",
                "site-packages/c10/**",
                "site-packages/aten/**",
                "site-packages/third_party/fmt/include/**",
                "site-packages/third_party/onnx/**",
                "site-packages/third_party/pybind11/include/**",
                "site-packages/build/**",
            ],
            exclude = [
                # Use torch_tpu's protobuf, not PyTorch's bundled one to avoid ODR
                "site-packages/include/google/**",
                "site-packages/torch/include/google/**",  # Exclude Protobuf (Conflict)
                "**/*.cpp",
                "**/*.cc",
                "**/*.cu",
                "**/*.py",
                "**/*.md",
            ] + (["site-packages/torch/include/**", "site-packages/build/torch/include/**"] if is_source_mode else []),
        ),
        includes = [
            "site-packages/include",  # Standard install layout
            "site-packages/aten/src",
            "site-packages/aten/src/ATen",
            "site-packages/build",
            "site-packages/build/include",
            "site-packages/build/aten/src",
            "site-packages/torch/csrc/api/include",
            "site-packages/torch/csrc",
            "site-packages/third_party/pybind11/include",  # pybind11 headers
            "site-packages",  # Main root for #include "c10/..." and #include "torch/..."
        ] + (["site-packages/torch/include", "site-packages/torch/include/torch/csrc/api/include"] if not is_source_mode else []),
    )

    # Use cc_import for prebuilt libraries to properly handle RPATH and DT_NEEDED
    cc_import(
        name = "libc10",
        shared_library = "site-packages/torch/lib/libc10.so",
    )

    cc_import(
        name = "libtorch_cpu",
        shared_library = "site-packages/torch/lib/libtorch_cpu.so",
    )

    cc_import(
        name = "libtorch_python",
        shared_library = "site-packages/torch/lib/libtorch_python.so",
    )

    # Catch-all target for convenience
    cc_library(
        name = "torch_libs",
        deps = [
            ":libc10",
            ":libtorch_cpu",
            ":libtorch_python",
        ],
    )

    py_library(
        # PY_LIBRARY_OK=Upstream source not pytype compatible
        name = "torch",
        srcs = native.glob([
            "site-packages/torch/**/*.py",
            "site-packages/torchgen/**/*.py",
        ]),
        data = native.glob([
            "site-packages/torch/**/*.so*",
            "site-packages/torch/**/*.pyd",
            "site-packages/torch/**/*.pyi",
            "site-packages/torch/**/*.yaml",
            "site-packages/torch/**/*.json",
            "site-packages/torch/bin/*",
            "site-packages/torch/lib/*.so*",
            "site-packages/torchgen/**/*.pyi",
            "site-packages/torchgen/**/*.yaml",
            "site-packages/torchgen/**/*.json",
        ]),
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
        ],
    )
