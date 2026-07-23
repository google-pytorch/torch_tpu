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

load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("@rules_testing//lib:test_suite.bzl", "test_suite")
load("//build_files:build_defs.bzl", "check_and_adjust_test_tags_for_testing", "is_backend_dep_for_testing", "is_oss", "torch_tpu_cc_test", "tpu_gen")

_TagsInfo = provider(
    "Provider for extracting rule attributes during analysis tests.",
    fields = ["tags", "rule_kind", "env", "exec_properties"],
)

def _tags_aspect_impl(_target, ctx):
    return [_TagsInfo(
        tags = ctx.rule.attr.tags,
        rule_kind = ctx.rule.kind,
        env = getattr(ctx.rule.attr, "env", {}),
        exec_properties = getattr(ctx.rule.attr, "exec_properties", {}),
    )]

tags_aspect = aspect(
    implementation = _tags_aspect_impl,
)

def _test_macro_tags_impl(env, targets):
    tags = targets.subject[_TagsInfo].tags
    env.expect.that_collection(tags).contains("nobuild")

def _test_macro_tags(name):
    torch_tpu_cc_test(
        name = name + "_subject",
        srcs = [],
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    analysis_test(
        name = name,
        impl = _test_macro_tags_impl,
        targets = {"subject": name + "_subject"},
        attrs = {"subject": {"aspects": [tags_aspect]}},
    )

# --- requires_libtpu macro tests ---
#
# When torch_tpu_cc_test is invoked with requires_libtpu = True (or when requires_libtpu is
# omitted and automatically inferred from a "requires-tpu" tag in OSS), the macro wraps the C++
# binary inside an sh_test rule (run_cc_tpu_test.sh) to configure TPU_LIBRARY_PATH in OSS.
# In internal Google3 builds, or when requires_libtpu is explicitly set to False, no sh_test
# wrapper is created, and the top-level rule emitted is directly a cc_test.
#
# We verify this behavior using analysis tests that inspect the target's rule_kind via tags_aspect.

def _test_requires_libtpu_inferred_impl(env, targets):
    """Verifies that requires_libtpu defaults to True in OSS when 'requires-tpu' tag is present."""

    # In OSS, because the target has 'requires-tpu' in its tags, requires_libtpu is inferred as True,
    # emitting an sh_test top-level rule. In internal builds, it emits a cc_test.
    rule_kind = targets.subject[_TagsInfo].rule_kind
    env.expect.that_str(rule_kind).equals("sh_test" if is_oss() else "cc_test")

def _test_requires_libtpu_inferred(name):
    torch_tpu_cc_test(
        name = name + "_subject",
        srcs = [],
        tags = ["requires-tpu"],
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    analysis_test(
        name = name,
        impl = _test_requires_libtpu_inferred_impl,
        targets = {"subject": name + "_subject"},
        attrs = {"subject": {"aspects": [tags_aspect]}},
    )

def _test_requires_libtpu_explicit_true_impl(env, targets):
    """Verifies that explicitly passing requires_libtpu = True wraps the test in OSS and forwards env/exec_properties."""

    # When requires_libtpu = True is explicitly provided, an sh_test wrapper is emitted in OSS
    # even when no 'requires-tpu' hardware tags are present on the target.
    info = targets.subject[_TagsInfo]
    env.expect.that_str(info.rule_kind).equals("sh_test" if is_oss() else "cc_test")
    env.expect.that_dict(info.env).contains_exactly({"TEST_ENV": "1"})

    # In OSS (when rule_kind is sh_test), cpp_link.mem is filtered out from exec_properties
    # so that sh_test does not crash with non-existent exec group errors.
    expected_exec_properties = (
        {"test.pool": "tpu-pool"} if is_oss() else {"cpp_link.mem": "20g", "test.pool": "tpu-pool"}
    )
    env.expect.that_dict(info.exec_properties).contains_exactly(expected_exec_properties)

def _test_requires_libtpu_explicit_true(name):
    torch_tpu_cc_test(
        name = name + "_subject",
        srcs = [],
        requires_libtpu = True,
        env = {"TEST_ENV": "1"},
        exec_properties = {"cpp_link.mem": "20g", "test.pool": "tpu-pool"},
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    analysis_test(
        name = name,
        impl = _test_requires_libtpu_explicit_true_impl,
        targets = {"subject": name + "_subject"},
        attrs = {"subject": {"aspects": [tags_aspect]}},
    )

def _test_requires_libtpu_explicit_false_impl(env, targets):
    """Verifies that explicitly passing requires_libtpu = False disables wrapping even when 'requires-tpu' is tagged."""

    # When requires_libtpu = False is explicitly passed, it overrides any 'requires-tpu' tag inference,
    # ensuring the top-level rule remains a cc_test across both OSS and internal builds.
    rule_kind = targets.subject[_TagsInfo].rule_kind
    env.expect.that_str(rule_kind).equals("cc_test")

def _test_requires_libtpu_explicit_false(name):
    torch_tpu_cc_test(
        name = name + "_subject",
        srcs = [],
        requires_libtpu = False,
        tags = ["requires-tpu"],
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    analysis_test(
        name = name,
        impl = _test_requires_libtpu_explicit_false_impl,
        targets = {"subject": name + "_subject"},
        attrs = {"subject": {"aspects": [tags_aspect]}},
    )

def _test_is_backend_dep(env):
    """Tests classification of version-independent backend deps vs PyTorch/internal deps."""
    if not is_oss():
        return

    # External, non-PyTorch backend deps are pinned to a single config.
    for dep in [
        "@xla//xla:shape_util",
        "@llvm-project//mlir:IR",
        "@stablehlo//:stablehlo_ops",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen3",
        "@com_googlesource_code_re2//:re2",
        # Non-torch package in the shared @pypi hub -> still backend.
        "@pypi//numpy:numpy_headers",
        # Classification keys on the known torch sources, not on the substring
        # "torch": a backend repo whose label happens to contain it is not torch.
        "@backend_with_torch_in_name//:lib",
    ]:
        env.expect.where(dep = dep).that_bool(is_backend_dep_for_testing(dep)).equals(True)

    # Our PyTorch sources (which must vary per version) and torch_tpu's own code
    # are not; keep in step with _TORCH_DEP_PREFIXES in build_defs.bzl.
    for dep in [
        "//shims/torch:torch_headers",
        "@local_torch//:libc10",
        "@pypi_torch_2_12_1_312//:libc10",
        "@pypi_torch_312//:libtorch_cpu",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:dtype",
        ":a_local_target",
    ]:
        env.expect.where(dep = dep).that_bool(is_backend_dep_for_testing(dep)).equals(False)

def _test_nobuild(env):
    """Tests the nobuild parameter."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nobuild = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nobuild"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_notap(env):
    """Tests the notap parameter."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["manual", "notap", "notest"])  # NOTAP_OK=Testing notap tagging
    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly([])

def _test_nopresubmit(env):
    """Tests the nopresubmit parameter."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["manual", "nopresubmit"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_nolocal(env):
    """Tests the nolocal parameter."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nolocal = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["manual"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_internal_nobuild_oss(env):
    """Tests the nobuild_oss parameter in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nobuild_oss = "reason",
        tags = tags,
    )

    # In an internal build, nobuild_oss has no effect and thus does not make
    # it necessary to generate a build_test.
    env.expect.that_collection(tags).contains_exactly([])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_nobuild_oss(env):
    """Tests the nobuild_oss parameter in OSS."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nobuild_oss = "reason",
        tags = tags,
    )

    # In an OSS build, nobuild_oss should disable generating the build_test.
    env.expect.that_collection(tags).contains_exactly(["nobuild", "notest"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_internal_notest_oss(env):
    """Tests the notest_oss parameter in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notest_oss = "reason",
        tags = tags,
    )

    # In an internal build, notest_oss has no effect and thus does not make
    # it necessary to generate a build_test.
    env.expect.that_collection(tags).contains_exactly([])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_notest_oss(env):
    """Tests the notest_oss parameter in OSS."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        notest_oss = "reason",
        tags = tags,
    )

    # In an OSS build, notest_oss should enable generating the build_test.
    env.expect.that_collection(tags).contains_exactly(["notest"])
    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly([])

def _test_cuda_build_test(env):
    """Tests the notap parameter for CUDA tests."""
    tags = ["requires-gpu-a100"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    expected_tags = ["manual", "notap", "notest", "requires-gpu", "requires-gpu-a100"] if is_oss() else ["manual", "notap", "notest", "requires-gpu-a100"]  # NOTAP_OK=Testing notap tagging
    env.expect.that_collection(tags).contains_exactly(expected_tags)
    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly(["requires-gpu-nvidia"])

def _test_internal_notap_nobuild(env):
    """Tests using both notap and nobuild in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        nobuild = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["manual", "nobuild", "notap", "notest"])  # NOTAP_OK=Testing notap tagging
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_notest_oss_nobuild_oss(env):
    """Tests using both notest_oss and nobuild_oss in OSS builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        notest_oss = "reason1",
        nobuild_oss = "reason2",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nobuild", "notest"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_internal_nopresubmit_oss(env):
    """Tests the nopresubmit_oss parameter in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly([])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_nopresubmit_oss(env):
    """Tests the nopresubmit_oss parameter in OSS builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nopresubmit"])
    env.expect.that_bool(result.create_build_test).equals(True)

def _test_internal_manual_nopresubmit_oss_tag(env):
    """Tests that the nopresubmit_oss param appends correctly even if 'nopresubmit_oss' is in tags in internal builds."""
    tags = ["nopresubmit_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nopresubmit_oss"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_manual_nopresubmit_oss_tag(env):
    """Tests that the nopresubmit_oss param appends correctly even if 'nopresubmit_oss' is in tags in OSS builds."""
    tags = ["nopresubmit_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nopresubmit", "nopresubmit_oss"])
    env.expect.that_bool(result.create_build_test).equals(True)

def _test_internal_nonightly_oss(env):
    """Tests the nonightly_oss parameter in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nonightly_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly([])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_nonightly_oss(env):
    """Tests the nonightly_oss parameter in OSS builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nonightly_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nonightly"])
    env.expect.that_bool(result.create_build_test).equals(True)

def _test_internal_manual_nonightly_oss_tag(env):
    """Tests that the nonightly_oss param appends correctly even if 'nonightly_oss' is in tags in internal builds."""
    tags = ["nonightly_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nonightly_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nonightly_oss"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_manual_nonightly_oss_tag(env):
    """Tests that the nonightly_oss param appends correctly even if 'nonightly_oss' is in tags in OSS builds."""
    tags = ["nonightly_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nonightly_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nonightly", "nonightly_oss"])
    env.expect.that_bool(result.create_build_test).equals(True)

def _test_oss_presubmit_tpu_generation_explicit(env):
    tags = []
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        oss_presubmit_tpu_generation = tpu_gen("v7", reason = "Requires specific hardware for feature testing"),
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v7")

def _test_oss_presubmit_tpu_generation_implicit(env):
    tags = ["requires-tpu", "fails-on-tpu-v5"]
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v7")

def _test_oss_presubmit_tpu_generation_implicit_v6(env):
    tags = ["requires-tpu", "fails-on-tpu-v5", "fails-on-tpu-v7"]
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v6")

def build_defs_test_suite(name):
    """Creates a test suite for build_defs.bzl, which will run all tests in this file.

    Args:
        name: The name of the test suite. All tests in this suite will be prefixed with this name.
    """
    test_suite(
        name = name + "_rules_testing",
        tests = [
            _test_macro_tags,
            _test_requires_libtpu_inferred,
            _test_requires_libtpu_explicit_true,
            _test_requires_libtpu_explicit_false,
        ],
        basic_tests = [
            # go/keep-sorted start
            _test_cuda_build_test,
            _test_internal_manual_nonightly_oss_tag,
            _test_internal_manual_nopresubmit_oss_tag,
            _test_internal_nobuild_oss,
            _test_internal_nonightly_oss,
            _test_internal_nopresubmit_oss,
            _test_internal_notap_nobuild,
            _test_internal_notest_oss,
            _test_is_backend_dep,
            _test_nobuild,
            _test_nolocal,
            _test_nopresubmit,
            _test_notap,
            _test_oss_manual_nonightly_oss_tag,
            _test_oss_manual_nopresubmit_oss_tag,
            _test_oss_nobuild_oss,
            _test_oss_nonightly_oss,
            _test_oss_nopresubmit_oss,
            _test_oss_notest_oss,
            _test_oss_notest_oss_nobuild_oss,
            _test_oss_presubmit_tpu_generation_explicit,
            _test_oss_presubmit_tpu_generation_implicit,
            _test_oss_presubmit_tpu_generation_implicit_v6,
            # go/keep-sorted end
        ],
    )
    native.test_suite(
        name = name,
        tests = [":" + name + "_rules_testing"],
    )
