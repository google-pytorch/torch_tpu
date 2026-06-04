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

"""Macro phase bzl code for pip_parse customizations."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

def define_extra_torch_targets():
    """Defines additional targets for the PyTorch distribution inside site-packages."""
    cc_library(
        name = "torch_headers",
        hdrs = native.glob(
            include = [
                "site-packages/torch/include/**/*.h",
                "site-packages/torch/include/**/*.hpp",
                "site-packages/torch/include/*.h",
                "site-packages/torch/include/*.hpp",
            ],
            exclude = [],
        ),
        # NOTE: There's a Bazel bug with strip_include_prefix where it
        # incorrectly thinks files provided by another target using
        # a strip_include_prefix of a sub-directory aren't being correctly
        # provided as dependencies. Using the `includes` attribute
        # seems to work around this.
        includes = [
            "site-packages/torch/include",
            "site-packages/torch/include/torch/csrc/api/include",
        ],
        deps = [
            "@com_google_protobuf//:protobuf",
        ],
    )

    lib_targets = []
    for lib in native.glob(["site-packages/torch/lib/lib*.so*"]):
        basename = paths.basename(lib)
        so_offset = basename.find(".so")
        target_name = basename[:so_offset]

        # NOTE: Some of the libs depend on each other, however, which depend
        # on which depends on the torch version and build. Unfortunately, we
        # can't inspect them during this build phase, so can't automatically
        # construct the DAG of dependencies.
        cc_import(
            name = target_name,
            shared_library = lib,
        )
        lib_targets.append(target_name)

    cc_library(
        name = "torch_libs",
        deps = lib_targets,
        visibility = ["//visibility:public"],
    )
