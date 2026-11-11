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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

def define_extra_torch_targets():
    """Defines additional targets for the PyTorch distribution inside site-packages."""
    cc_library(
        # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
        name = "torch_headers",
        hdrs = native.glob(
            include = [
                "site-packages/torch/**/*.h",
                "site-packages/torch/**/*.hpp",
                "site-packages/torch/**/*.hh",
                "site-packages/torch/**/*.hxx",
                "site-packages/torch/**/*.cuh",
                "site-packages/torch/**/*.inc",
                "site-packages/torch/**/*.inl",
                "site-packages/torch/**/*.tcc",
            ],
            # Exclude PyTorch's bundled google/protobuf headers from declared
            # inputs so they do not shadow @com_google_protobuf in
            # sandboxed/remote builds.
            exclude = [
                "site-packages/torch/include/google/protobuf/**",
            ],
        ),
        # NOTE: There's a Bazel bug with strip_include_prefix where it
        # incorrectly thinks files provided by another target using
        # a strip_include_prefix of a sub-directory aren't being correctly
        # provided as dependencies. Using the `includes` attribute
        # seems to work around this.
        includes = [
            "site-packages/torch/include",
            "site-packages/torch/include/torch/csrc/api/include",
            "site-packages/torch/include/kineto",
        ],
        visibility = ["//visibility:public"],
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
        cc_library(
            # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
            name = lib_name,
            srcs = native.glob([
                "site-packages/torch/lib/{}.so".format(lib_name),
            ]),
            visibility = ["//visibility:public"],
        )

    cc_library(
        # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
        name = "libtorch",
        srcs = native.glob([
            "site-packages/torch/lib/libtorch.so",
        ]),
        deps = [
            ":libc10",
            ":libtorch_cpu",
            ":libshm",
        ],
        visibility = ["//visibility:public"],
    )

    cc_library(
        # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
        name = "libtorch_global_deps",
        deps = [
            ":libc10",
            ":libtorch_cpu",
        ],
        visibility = ["//visibility:public"],
    )

    cc_library(
        # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
        name = "torch_libs",
        deps = [
            ":libtorch",
            ":libtorch_python",
            ":libtorch_global_deps",
        ],
        visibility = ["//visibility:public"],
    )
