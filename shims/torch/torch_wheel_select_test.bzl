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

"""Analysis tests for //shims/torch:torch_wheel_select.bzl."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("@rules_testing//lib:test_suite.bzl", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
load("@rules_testing//lib:util.bzl", "util")
load(
    "//shims/torch:torch_wheel_select.bzl",
    "torch_whl_cc_headers_for_testing",
    "torch_whl_cc_lib_for_testing",
)

def _fake_whl_files_impl(ctx):
    files = []
    for rel_path in ctx.attr.paths:
        f = ctx.actions.declare_file("{}/{}".format(ctx.label.name, rel_path))
        ctx.actions.write(f, "")
        files.append(f)
    return [DefaultInfo(files = depset(files))]

_fake_whl_files = rule(
    implementation = _fake_whl_files_impl,
    attrs = {
        "paths": attr.string_list(mandatory = True),
    },
)

def _test_headers_filters_and_system_includes(name):
    util.helper_target(
        _fake_whl_files,
        name = name + "_whl",
        paths = [
            "site-packages/torch/include/ATen/ATen.h",
            "site-packages/torch/include/c10/macros/Macros.hpp",
            "site-packages/torch/include/ATen/cuda/Empty.cuh",
            "site-packages/torch/include/google/protobuf/arena.h",
            "site-packages/torch/include/google/protobuf/port_def.inc",
            "site-packages/torch/__init__.py",
            "site-packages/torch/lib/libc10.so",
        ],
    )
    util.helper_target(
        torch_whl_cc_headers_for_testing,
        name = name + "_subject",
        whl_files = ":" + name + "_whl",
    )
    analysis_test(
        name = name,
        impl = _test_headers_filters_and_system_includes_impl,
        target = ":" + name + "_subject",
    )

def _test_headers_filters_and_system_includes_impl(env, target):
    cc_ctx = target[CcInfo].compilation_context
    header_short_paths = [f.short_path for f in cc_ctx.headers.to_list()]
    prefix = "shims/torch/" + env.ctx.label.name + "_whl/site-packages/torch/"
    env.expect.that_collection(header_short_paths).contains_exactly([
        prefix + "include/ATen/ATen.h",
        prefix + "include/c10/macros/Macros.hpp",
        prefix + "include/ATen/cuda/Empty.cuh",
    ])

    repo_root = "{}/shims/torch/{}_whl".format(
        env.ctx.bin_dir.path,
        env.ctx.label.name,
    )
    env.expect.that_collection(cc_ctx.system_includes.to_list()).contains_exactly([
        repo_root + "/site-packages/torch/include",
        repo_root + "/site-packages/torch/include/torch/csrc/api/include",
        repo_root + "/site-packages/torch/include/kineto",
    ])

# `bazel-diff` and `ci/tools/run_bazel_diff.py` run `bazel cquery //...` over
# the workspace without `--allow_analysis_failures`, and `cquery //...`
# evaluates `tags = ["manual"]` targets. Gate the failure-inducing inputs on
# `allow_analysis_failures=true` (which `analysis_test(expect_failure = True)`
# enables via its incoming configuration transition) so the helper `_subject`
# targets analyze cleanly under top-level `cquery //...` while still failing as
# expected when analyzed by `analysis_test`.
_ALLOW_ANALYSIS_FAILURES = ":_allow_analysis_failures"

def _test_headers_fails_when_empty(name):
    util.helper_target(
        _fake_whl_files,
        name = name + "_whl",
        paths = select({
            _ALLOW_ANALYSIS_FAILURES: [
                "site-packages/torch/__init__.py",
                "site-packages/torch/include/google/protobuf/arena.h",
            ],
            "//conditions:default": [
                "site-packages/torch/include/ATen/ATen.h",
            ],
        }),
    )
    util.helper_target(
        torch_whl_cc_headers_for_testing,
        name = name + "_subject",
        whl_files = ":" + name + "_whl",
    )
    analysis_test(
        name = name,
        impl = _test_headers_fails_when_empty_impl,
        target = ":" + name + "_subject",
        expect_failure = True,
    )

def _test_headers_fails_when_empty_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.str_matches("No PyTorch header files found under site-packages/torch/*"),
    )

def _test_cc_lib_links_single_so(name):
    util.helper_target(
        _fake_whl_files,
        name = name + "_whl",
        paths = [
            "site-packages/torch/lib/libc10.so",
            "site-packages/torch/lib/libtorch_cpu.so",
        ],
    )
    util.helper_target(
        torch_whl_cc_lib_for_testing,
        name = name + "_subject",
        lib_name = "libc10",
        whl_files = ":" + name + "_whl",
    )
    analysis_test(
        name = name,
        impl = _test_cc_lib_links_single_so_impl,
        target = ":" + name + "_subject",
    )

def _test_cc_lib_links_single_so_impl(env, target):
    files = [f.short_path for f in target[DefaultInfo].files.to_list()]
    env.expect.that_collection(files).contains_exactly([
        "shims/torch/" + env.ctx.label.name + "_whl/site-packages/torch/lib/libc10.so",
    ])
    linker_inputs = target[CcInfo].linking_context.linker_inputs.to_list()
    env.expect.that_collection(linker_inputs).has_size(1)
    libs = linker_inputs[0].libraries
    env.expect.that_collection(libs).has_size(1)
    env.expect.that_str(libs[0].dynamic_library.basename).equals("libc10.so")

def _test_cc_lib_fails_on_missing_so(name):
    util.helper_target(
        _fake_whl_files,
        name = name + "_whl",
        paths = select({
            _ALLOW_ANALYSIS_FAILURES: ["site-packages/torch/lib/libtorch_cpu.so"],
            "//conditions:default": ["site-packages/torch/lib/libc10.so"],
        }),
    )
    util.helper_target(
        torch_whl_cc_lib_for_testing,
        name = name + "_subject",
        lib_name = "libc10",
        whl_files = ":" + name + "_whl",
    )
    analysis_test(
        name = name,
        impl = _test_cc_lib_fails_on_missing_so_impl,
        target = ":" + name + "_subject",
        expect_failure = True,
    )

def _test_cc_lib_fails_on_missing_so_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.str_matches("Expected exactly 1 shared library for libc10*"),
    )

def _test_cc_lib_fails_on_duplicate_so(name):
    util.helper_target(
        _fake_whl_files,
        name = name + "_whl",
        paths = select({
            _ALLOW_ANALYSIS_FAILURES: [
                "a/site-packages/torch/lib/libc10.so",
                "b/site-packages/torch/lib/libc10.so",
            ],
            "//conditions:default": [
                "site-packages/torch/lib/libc10.so",
            ],
        }),
    )
    util.helper_target(
        torch_whl_cc_lib_for_testing,
        name = name + "_subject",
        lib_name = "libc10",
        whl_files = ":" + name + "_whl",
    )
    analysis_test(
        name = name,
        impl = _test_cc_lib_fails_on_duplicate_so_impl,
        target = ":" + name + "_subject",
        expect_failure = True,
    )

def _test_cc_lib_fails_on_duplicate_so_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.str_matches("Expected exactly 1 shared library for libc10*"),
    )

def torch_wheel_select_test_suite(name):
    util.helper_target(
        native.config_setting,
        name = "_allow_analysis_failures",
        values = {"allow_analysis_failures": "true"},
    )
    test_suite(
        name = name,
        tests = [
            _test_headers_filters_and_system_includes,
            _test_headers_fails_when_empty,
            _test_cc_lib_links_single_so,
            _test_cc_lib_fails_on_missing_so,
            _test_cc_lib_fails_on_duplicate_so,
        ],
    )
