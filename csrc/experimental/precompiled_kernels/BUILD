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

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load(
    "//build_files:build_defs.bzl",
    "if_oss",
    "torch_tpu_cc_library",
    "torch_tpu_package_end",
)
load("//shims/build_files:torch_version.bzl", "reset_torch_config")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

bool_flag(
    name = "enabled",
    build_setting_default = False,
)

config_setting(
    name = "enabled_config",
    flag_values = {":enabled": "true"},
)

# Keep the pybind dependency list plain for its version-pinning transition.
# The default target contributes no object code or compiler dependencies.
torch_tpu_cc_library(name = "disabled")

alias(
    name = "optional_plugin",
    actual = select({
        ":enabled_config": "//csrc/experimental/" +
                           "precompiled_kernels/runtime:plugin",
        "//conditions:default": ":disabled",
    }),
)

proto_library(
    name = "bundle_proto",
    srcs = ["bundle.proto"],
    deps = [
        "@xla//xla:xla_data_proto",
    ],
)

cc_proto_library(
    name = "_bundle_cc_proto",
    deps = [":bundle_proto"],
)

reset_torch_config(
    name = "_bundle_cc_proto_fixed",
    dep = ":_bundle_cc_proto",
)

alias(
    name = "bundle_cc_proto",
    actual = if_oss(
        select({
            "//:wheel_build_enabled": ":_bundle_cc_proto_fixed",
            "//conditions:default": ":_bundle_cc_proto",
        }),
        ":_bundle_cc_proto",
    ),
)

torch_tpu_package_end()
