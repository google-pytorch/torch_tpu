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
load("//build_files:build_defs.bzl", "check_and_adjust_test_tags_for_testing", "tpu_gen")

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
        tags == ["manual", "notap", "notest"],  # NOTAP_OK=tests
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

    asserts.true(env, tags == ["manual", "nopresubmit"], "tags: %s" % tags)
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

    asserts.true(env, tags == ["nobuild", "notest"], "tags: %s" % tags)

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

    asserts.true(env, tags == ["notest"], "tags: %s" % tags)

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
        tags == ["manual", "notap", "notest", "requires-gpu-a100"],  # NOTAP_OK=tests
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
        tags == ["manual", "nobuild", "notap", "notest"],  # NOTAP_OK=tests
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
        tags == ["nobuild", "notest"],
        "tags: %s" % tags,
    )
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_internal_nopresubmit_oss(ctx):
    """Tests the nopresubmit_oss parameter in internal builds."""
    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == [], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_oss_nopresubmit_oss(ctx):
    """Tests the nopresubmit_oss parameter in OSS builds."""
    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nopresubmit"], "tags: %s" % tags)
    asserts.true(env, result.create_build_test)
    return unittest.end(env)

def _test_internal_manual_nopresubmit_oss_tag(ctx):
    """Tests that the nopresubmit_oss param appends correctly even if 'nopresubmit_oss' is in tags in internal builds."""
    env = unittest.begin(ctx)
    tags = ["nopresubmit_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nopresubmit_oss"], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_oss_manual_nopresubmit_oss_tag(ctx):
    """Tests that the nopresubmit_oss param appends correctly even if 'nopresubmit_oss' is in tags in OSS builds."""
    env = unittest.begin(ctx)
    tags = ["nopresubmit_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nopresubmit", "nopresubmit_oss"], "tags: %s" % tags)
    asserts.true(env, result.create_build_test)
    return unittest.end(env)

def _test_internal_nonightly_oss(ctx):
    """Tests the nonightly_oss parameter in internal builds."""
    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nonightly_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == [], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_oss_nonightly_oss(ctx):
    """Tests the nonightly_oss parameter in OSS builds."""
    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nonightly_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nonightly"], "tags: %s" % tags)
    asserts.true(env, result.create_build_test)
    return unittest.end(env)

def _test_internal_manual_nonightly_oss_tag(ctx):
    """Tests that the nonightly_oss param appends correctly even if 'nonightly_oss' is in tags in internal builds."""
    env = unittest.begin(ctx)
    tags = ["nonightly_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nonightly_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nonightly_oss"], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_oss_manual_nonightly_oss_tag(ctx):
    """Tests that the nonightly_oss param appends correctly even if 'nonightly_oss' is in tags in OSS builds."""
    env = unittest.begin(ctx)
    tags = ["nonightly_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nonightly_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nonightly", "nonightly_oss"], "tags: %s" % tags)
    asserts.true(env, result.create_build_test)
    return unittest.end(env)

def _test_oss_presubmit_tpu_generation_explicit(ctx):
    env = unittest.begin(ctx)
    tags = []
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        oss_presubmit_tpu_generation = tpu_gen("v7", reason = "Requires specific hardware for feature testing"),
        tags = tags,
    )
    asserts.true(env, "presubmit-v7" in tags, "tags: %s" % tags)
    return unittest.end(env)

def _test_oss_presubmit_tpu_generation_implicit(ctx):
    env = unittest.begin(ctx)
    tags = ["requires-tpu"]
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        incompatible_tpu_generations = {"v5": "reason"},
        tags = tags,
    )
    asserts.false(env, "presubmit-v5" in tags, "tags: %s" % tags)
    asserts.true(env, "presubmit-v7" in tags, "tags: %s" % tags)
    return unittest.end(env)

def _test_oss_presubmit_tpu_generation_implicit_v6(ctx):
    env = unittest.begin(ctx)
    tags = ["requires-tpu"]
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        incompatible_tpu_generations = {"v5": "reason", "v7": "reason"},
        tags = tags,
    )
    asserts.true(env, "presubmit-v6" in tags, "tags: %s" % tags)
    return unittest.end(env)

def _test_oss_presubmit_tpu_generation_default(ctx):
    env = unittest.begin(ctx)
    tags = ["requires-tpu"]
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags,
    )

    # It should select one of the valid generations: v5, v6, or v7.
    asserts.true(
        env,
        "presubmit-v5" in tags or "presubmit-v6" in tags or "presubmit-v7" in tags,
        "tags: %s" % tags,
    )
    return unittest.end(env)

def _test_oss_presubmit_tpu_generation_incompatible_all(ctx):
    env = unittest.begin(ctx)
    tags = ["requires-tpu"]
    errors = []

    def mock_fail(msg):
        errors.append(msg)

    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        incompatible_tpu_generations = {"v5": "r", "v6": "r", "v7": "r"},
        fail_func = mock_fail,
        tags = tags,
    )
    asserts.true(env, len(errors) == 1, "Expected 1 error, got %s" % errors)
    asserts.true(env, "TPU tests failing on all generations" in errors[0], "Unexpected error: %s" % errors[0])
    return unittest.end(env)

def _test_incompatible_tpu_generations_validation_fails(ctx):
    env = unittest.begin(ctx)

    # Test cases for validation failures: (incompatible_tpu_generations, expected_error_substring)
    test_cases = [
        ("not a dict", "must be a dict"),
        ({1: "reason"}, "keys must be strings"),
        ({"v5": 123}, "values must be non-empty strings"),
        ({"v5": ""}, "values must be non-empty strings"),
        ({"unknown": "reason"}, "Unknown TPU generation"),
    ]

    for incompat, expected_err in test_cases:
        errors = []

        def mock_fail(msg):
            errors.append(msg)

        check_and_adjust_test_tags_for_testing(
            is_oss = True,
            incompatible_tpu_generations = incompat,
            fail_func = mock_fail,
            tags = ["requires-tpu"],
        )
        asserts.true(env, len(errors) == 1, "Expected 1 error for %s, got %s" % (incompat, errors))
        asserts.true(env, expected_err in errors[0], "Expected error containing '%s' for %s, got '%s'" % (expected_err, incompat, errors[0]))

    return unittest.end(env)

# go/keep-sorted start
cuda_build_test = unittest.make(_test_cuda_build_test)
incompatible_tpu_generations_validation_fails_test = unittest.make(_test_incompatible_tpu_generations_validation_fails)
internal_manual_nonightly_oss_tag_test = unittest.make(_test_internal_manual_nonightly_oss_tag)
internal_manual_nopresubmit_oss_tag_test = unittest.make(_test_internal_manual_nopresubmit_oss_tag)
internal_nobuild_oss_test = unittest.make(_test_internal_nobuild_oss)
internal_nonightly_oss_test = unittest.make(_test_internal_nonightly_oss)
internal_nopresubmit_oss_test = unittest.make(_test_internal_nopresubmit_oss)
internal_notap_nobuild_test = unittest.make(_test_internal_notap_nobuild)
internal_notest_oss_test = unittest.make(_test_internal_notest_oss)
nobuild_test = unittest.make(_test_nobuild)
nolocal_test = unittest.make(_test_nolocal)
nopresubmit_test = unittest.make(_test_nopresubmit)
notap_test = unittest.make(_test_notap)
oss_manual_nonightly_oss_tag_test = unittest.make(_test_oss_manual_nonightly_oss_tag)
oss_manual_nopresubmit_oss_tag_test = unittest.make(_test_oss_manual_nopresubmit_oss_tag)
oss_nobuild_oss_test = unittest.make(_test_oss_nobuild_oss)
oss_nonightly_oss_test = unittest.make(_test_oss_nonightly_oss)
oss_nopresubmit_oss_test = unittest.make(_test_oss_nopresubmit_oss)
oss_notest_oss_nobuild_oss_test = unittest.make(_test_oss_notest_oss_nobuild_oss)
oss_notest_oss_test = unittest.make(_test_oss_notest_oss)
oss_presubmit_tpu_generation_default_test = unittest.make(_test_oss_presubmit_tpu_generation_default)
oss_presubmit_tpu_generation_explicit_test = unittest.make(_test_oss_presubmit_tpu_generation_explicit)
oss_presubmit_tpu_generation_implicit_test = unittest.make(_test_oss_presubmit_tpu_generation_implicit)
oss_presubmit_tpu_generation_implicit_v6_test = unittest.make(_test_oss_presubmit_tpu_generation_implicit_v6)
oss_presubmit_tpu_generation_incompatible_all_test = unittest.make(_test_oss_presubmit_tpu_generation_incompatible_all)
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
    add_test(tests, incompatible_tpu_generations_validation_fails_test, name + "_incompatible_tpu_generations_validation_fails")
    add_test(tests, internal_manual_nonightly_oss_tag_test, name + "_internal_manual_nonightly_oss_tag")
    add_test(tests, internal_manual_nopresubmit_oss_tag_test, name + "_internal_manual_nopresubmit_oss_tag")
    add_test(tests, internal_nobuild_oss_test, name + "_internal_nobuild_oss")
    add_test(tests, internal_nonightly_oss_test, name + "_internal_nonightly_oss")
    add_test(tests, internal_nopresubmit_oss_test, name + "_internal_nopresubmit_oss")
    add_test(tests, internal_notap_nobuild_test, name + "_internal_notap_nobuild")
    add_test(tests, internal_notest_oss_test, name + "_internal_notest_oss")
    add_test(tests, nobuild_test, name + "_nobuild")
    add_test(tests, nolocal_test, name + "_nolocal")
    add_test(tests, nopresubmit_test, name + "_nopresubmit")
    add_test(tests, notap_test, name + "_notap")
    add_test(tests, oss_manual_nonightly_oss_tag_test, name + "_oss_manual_nonightly_oss_tag")
    add_test(tests, oss_manual_nopresubmit_oss_tag_test, name + "_oss_manual_nopresubmit_oss_tag")
    add_test(tests, oss_nobuild_oss_test, name + "_oss_nobuild_oss")
    add_test(tests, oss_nonightly_oss_test, name + "_oss_nonightly_oss")
    add_test(tests, oss_nopresubmit_oss_test, name + "_oss_nopresubmit_oss")
    add_test(tests, oss_notest_oss_nobuild_oss_test, name + "_oss_notest_oss_nobuild_oss")
    add_test(tests, oss_notest_oss_test, name + "_oss_notest_oss")
    add_test(tests, oss_presubmit_tpu_generation_default_test, name + "_oss_presubmit_tpu_generation_default")
    add_test(tests, oss_presubmit_tpu_generation_explicit_test, name + "_oss_presubmit_tpu_generation_explicit")
    add_test(tests, oss_presubmit_tpu_generation_implicit_test, name + "_oss_presubmit_tpu_generation_implicit")
    add_test(tests, oss_presubmit_tpu_generation_implicit_v6_test, name + "_oss_presubmit_tpu_generation_implicit_v6")
    add_test(tests, oss_presubmit_tpu_generation_incompatible_all_test, name + "_oss_presubmit_tpu_generation_incompatible_all")
    # go/keep-sorted end

    native.test_suite(
        name = name,
        tests = tests,
    )
