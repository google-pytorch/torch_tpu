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

"""Unit tests for build_defs.bzl."""

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//build_files:build_defs.bzl", "check_and_adjust_test_tags_for_testing")

def _test_nobuild(ctx):
    """Tests the nobuild parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nobuild = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nobuild"], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_notap(ctx):
    """Tests the notap parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    asserts.true(
        env,
        tags == ["manual", "notap"],  # NOTAP_OK=tests
        "tags: %s" % tags,
    )
    asserts.true(env, result.create_build_test)
    asserts.true(
        env,
        result.build_test_tags == [],
        "build_test_tags: %s" % result.build_test_tags,
    )
    return unittest.end(env)

def _test_nopresubmit(ctx):
    """Tests the nopresubmit parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["manual", "nofastbuild"], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_nolocal(ctx):
    """Tests the nolocal parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nolocal = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["manual"], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_internal_nobuild_oss(ctx):
    """Tests the nobuild_oss parameter in internal builds."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nobuild_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == [], "tags: %s" % tags)

    # In an internal build, nobuild_oss has no effect and thus does not make
    # it necessary to generate a build_test.
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_oss_nobuild_oss(ctx):
    """Tests the nobuild_oss parameter in OSS."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nobuild_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nobuild", "notap"], "tags: %s" % tags)  # NOTAP_OK=tests

    # In an OSS build, nobuild_oss should disable generating the build_test.
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_internal_notest_oss(ctx):
    """Tests the notest_oss parameter in internal builds."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notest_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == [], "tags: %s" % tags)

    # In an internal build, notest_oss has no effect and thus does not make
    # it necessary to generate a build_test.
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_oss_notest_oss(ctx):
    """Tests the notest_oss parameter in OSS."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        notest_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["notap"], "tags: %s" % tags)  # NOTAP_OK=tests

    # In an OSS build, notest_oss should enable generating the build_test.
    asserts.true(env, result.create_build_test)
    asserts.true(
        env,
        result.build_test_tags == [],
        "build_test_tags: %s" % result.build_test_tags,
    )
    return unittest.end(env)

def _test_cuda_build_test(ctx):
    """Tests the notap parameter for CUDA tests."""

    env = unittest.begin(ctx)
    tags = ["requires-gpu-a100"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    asserts.true(
        env,
        tags == ["manual", "notap", "requires-gpu-a100"],  # NOTAP_OK=tests
        "tags: %s" % tags,
    )
    asserts.true(env, result.create_build_test)
    asserts.true(
        env,
        result.build_test_tags == ["requires-gpu-nvidia"],
        "result.build_test_tags: %s" % result.build_test_tags,
    )
    return unittest.end(env)

def _test_internal_notap_nobuild(ctx):
    """Tests using both notap and nobuild in internal builds."""
    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        nobuild = "reason",
        tags = tags,
    )

    asserts.true(
        env,
        tags == ["manual", "nobuild", "notap"],  # NOTAP_OK=tests
        "tags: %s" % tags,
    )
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_oss_notest_oss_nobuild_oss(ctx):
    """Tests using both notest_oss and nobuild_oss in OSS builds."""
    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        notest_oss = "reason1",
        nobuild_oss = "reason2",
        tags = tags,
    )

    asserts.true(
        env,
        tags == ["nobuild", "notap"],  # NOTAP_OK=tests
        "tags: %s" % tags,
    )
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

# go/keep-sorted start
cuda_build_test = unittest.make(_test_cuda_build_test)
internal_nobuild_oss_test = unittest.make(_test_internal_nobuild_oss)
internal_notap_nobuild_test = unittest.make(_test_internal_notap_nobuild)
internal_notest_oss_test = unittest.make(_test_internal_notest_oss)
nobuild_test = unittest.make(_test_nobuild)
nolocal_test = unittest.make(_test_nolocal)
nopresubmit_test = unittest.make(_test_nopresubmit)
notap_test = unittest.make(_test_notap)
oss_nobuild_oss_test = unittest.make(_test_oss_nobuild_oss)
oss_notest_oss_nobuild_oss_test = unittest.make(_test_oss_notest_oss_nobuild_oss)
oss_notest_oss_test = unittest.make(_test_oss_notest_oss)
# go/keep-sorted end

def build_defs_test_suite(name):
    """Creates a test suite for build_defs.bzl, which will run all tests in this file.

    Args:
        name: The name of the test suite. All tests in this suite will be prefixed with this name.
    """

    def add_test(tests, rule_func, name):
        rule_func(name = name)
        tests.append(":" + name)

    tests = []

    # go/keep-sorted start
    add_test(tests, cuda_build_test, name + "_cuda_build_test")
    add_test(tests, internal_nobuild_oss_test, name + "_internal_nobuild_oss")
    add_test(tests, internal_notap_nobuild_test, name + "_internal_notap_nobuild")
    add_test(tests, internal_notest_oss_test, name + "_internal_notest_oss")
    add_test(tests, nobuild_test, name + "_nobuild")
    add_test(tests, nolocal_test, name + "_nolocal")
    add_test(tests, nopresubmit_test, name + "_nopresubmit")
    add_test(tests, notap_test, name + "_notap")
    add_test(tests, oss_nobuild_oss_test, name + "_oss_nobuild_oss")
    add_test(tests, oss_notest_oss_nobuild_oss_test, name + "_oss_notest_oss_nobuild_oss")
    add_test(tests, oss_notest_oss_test, name + "_oss_notest_oss")
    # go/keep-sorted end

    native.test_suite(
        name = name,
        tests = tests,
    )
