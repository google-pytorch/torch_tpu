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

    # This is a subset of all the `.so` files bundled into the PyTorch wheel.
    # More libraries can be added here as necessary.
    standard_libs = [
        "libc10",
        "libtorch_cpu",
        "libtorch_python",
        "libshm",
    ]

    for lib_name in standard_libs:
        cc_import(
            name = lib_name,
            shared_library = "site-packages/torch/lib/{}.so".format(lib_name),
        )

    cc_library(
        name = "libtorch",
        srcs = native.glob([
            "site-packages/torch/lib/libtorch.so",
            "site-packages/torch/lib/libtorch.dylib",
        ]),
        deps = [
            ":libc10",
            ":libtorch_cpu",
            ":libshm",
        ],
        visibility = ["//visibility:public"],
    )

    cc_library(
        name = "libtorch_global_deps",
        deps = [
            ":libc10",
            ":libtorch_cpu",
        ],
        visibility = ["//visibility:public"],
    )

    # TODO: figure out why torch_tpu is linking libpython at all and stop it
    cc_library(
        name = "libpython",  # ALLOW_CC_TARGETS=External wheel repo target cannot use custom macros
        srcs = native.glob(["site-packages/torch/lib/libpython3.*.so*"]),
        visibility = ["//visibility:public"],
    )

    cc_library(
        name = "torch_libs",
        deps = [
            ":libtorch",
            ":libtorch_python",
            ":libtorch_global_deps",
            ":libpython",
        ],
        visibility = ["//visibility:public"],
    )
