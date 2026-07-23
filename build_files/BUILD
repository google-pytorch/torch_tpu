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

"""Utilities for building torch_tpu."""

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
)
load("//build_files:build_defs_test.bzl", "build_defs_test_suite")
load("//shims/build_defs_testing:bzl_library.bzl", "bzl_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

exports_files([
    "lsan_suppressions.txt",
    "tsan_suppressions.txt",
])

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        "//bazel:supported_python_versions_bzl",
        "//shims/build_cleaner:build_defs_lib",
        "//shims/build_files:build_defs_bzl",
        "//shims/build_files:torch_version_bzl",
        "//shims/py_platform_test:py_platform_test_bzl",
        "//shims/py_rules:pytype_bzl",
        "@bazel_skylib//rules:build_test",
        "@rules_cc//cc:core_rules",
        "@rules_python//python:py_test_bzl",
        "@xla//third_party/py/rules_pywrap:pywrap.default.bzl",
    ],
)

bzl_library(
    name = "py_dist_info_bzl",
    srcs = ["py_dist_info.bzl"],
    deps = [
        "//shims/build_cleaner:build_defs_lib",
    ],
)

# This library just holds the ENTRY_POINTS variable, so this target is unused;
# it only exists because cclean wants something to point to entry_points.bzl
bzl_library(
    name = "entry_points_bzl",
    srcs = ["entry_points.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

# Creates a test suite with the given name. The individual tests in
# the suite are named build_defs_test_<suffix> and defined in build_defs_test.bzl.
build_defs_test_suite(
    name = "build_defs_test",
)

torch_tpu_package_end()
