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
load(
    "//build_files:build_defs.bzl",
    # go/keep-sorted start
    "add_rbe_exec_properties_for_testing",
    "check_and_adjust_test_tags_for_testing",
    "clamp_shard_count_for_testing",
    "get_tpu_chip_count_for_testing",
    "get_tpu_version_for_testing",
    "is_backend_dep_for_testing",
    "is_oss",
    "torch_tpu_cc_test",
    "torch_tpu_py_test",
    "tpu_gen",
    # go/keep-sorted end
)

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

def _test_py_test_select_env_impl(env, targets):
    """Verifies that torch_tpu_py_test produces a py_test rule when env uses select()."""
    info = targets.subject[_TagsInfo]
    env.expect.that_str(info.rule_kind).equals("py_test")

def _test_py_test_select_env(name):
    torch_tpu_py_test(
        name = name + "_subject",
        srcs = ["build_defs_test.py"],
        main = "build_defs_test.py",
        is_wheel_test = True,
        env = select({
            "//conditions:default": {"TEST_VAR": "value"},
        }),
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    analysis_test(
        name = name,
        impl = _test_py_test_select_env_impl,
        targets = {"subject": name + "_subject"},
        attrs = {"subject": {"aspects": [tags_aspect]}},
    )

def _test_py_test_multi_tpu_targets_impl(env, targets):
    """Verifies that torch_tpu_py_test expands multiple requires-TPU tags into per-TPU targets with tag isolation."""
    target_v5_tags = targets.target_v5[_TagsInfo].tags
    target_v6_tags = targets.target_v6[_TagsInfo].tags
    target_v7_tags = targets.target_v7[_TagsInfo].tags

    # Target 1 (v5)
    env.expect.that_collection(target_v5_tags).contains("requires-tpu-v5lite")
    env.expect.that_collection(target_v5_tags).contains("custom_tag")
    env.expect.that_collection(target_v5_tags).not_contains("requires-tpu-v6e")
    env.expect.that_collection(target_v5_tags).not_contains("requires-tpu-v7x")
    env.expect.that_collection(target_v5_tags).not_contains("nopresubmit")

    # Checking fails-on-tpu tags tests an implementation detail rather than the
    # behavior we care about (a blaze query skipping mismatched TPU runners).
    # Update or replace this check if the exclusion mechanism changes.
    env.expect.that_collection(target_v5_tags).contains("fails-on-tpu-v6")
    env.expect.that_collection(target_v5_tags).contains("fails-on-tpu-v7")
    env.expect.that_collection(target_v5_tags).not_contains("fails-on-tpu-v5")

    # Target 2 (v6)
    env.expect.that_collection(target_v6_tags).contains("requires-tpu-v6e")
    env.expect.that_collection(target_v6_tags).contains("custom_tag")
    env.expect.that_collection(target_v6_tags).not_contains("requires-tpu-v5lite")
    env.expect.that_collection(target_v6_tags).not_contains("requires-tpu-v7x")
    env.expect.that_collection(target_v6_tags).contains("nopresubmit")
    env.expect.that_collection(target_v6_tags).contains("fails-on-tpu-v5")
    env.expect.that_collection(target_v6_tags).contains("fails-on-tpu-v7")
    env.expect.that_collection(target_v6_tags).not_contains("fails-on-tpu-v6")

    # Target 3 (v7)
    env.expect.that_collection(target_v7_tags).contains("requires-tpu-v7x")
    env.expect.that_collection(target_v7_tags).contains("custom_tag")
    env.expect.that_collection(target_v7_tags).not_contains("requires-tpu-v5lite")
    env.expect.that_collection(target_v7_tags).not_contains("requires-tpu-v6e")
    env.expect.that_collection(target_v7_tags).contains("nopresubmit")
    env.expect.that_collection(target_v7_tags).contains("fails-on-tpu-v5")
    env.expect.that_collection(target_v7_tags).contains("fails-on-tpu-v6")
    env.expect.that_collection(target_v7_tags).not_contains("fails-on-tpu-v7")

    # In OSS, the presubmit-v<N> tag picks the runner, so each target must have
    # the one matching its accelerator.
    if is_oss():
        env.expect.that_collection(target_v5_tags).contains("presubmit-v5")
        env.expect.that_collection(target_v5_tags).not_contains("presubmit-v6")
        env.expect.that_collection(target_v5_tags).not_contains("presubmit-v7")

        env.expect.that_collection(target_v6_tags).contains("presubmit-v6")
        env.expect.that_collection(target_v6_tags).not_contains("presubmit-v5")
        env.expect.that_collection(target_v6_tags).not_contains("presubmit-v7")

        env.expect.that_collection(target_v7_tags).contains("presubmit-v7")
        env.expect.that_collection(target_v7_tags).not_contains("presubmit-v5")
        env.expect.that_collection(target_v7_tags).not_contains("presubmit-v6")

        env.expect.that_dict(targets.target_v5[_TagsInfo].exec_properties).contains_exactly(
            _V5_RBE_EXEC_PROPERTIES,
        )

        # TODO: b/567928058 - Expect `ci_tpu_v6_full_rbe` and `ci_tpu_v7_full_rbe`
        # once those RBE pools are added to _RBE_TPU_POOLS.
        env.expect.that_dict(targets.target_v6[_TagsInfo].exec_properties).contains_exactly({})
        env.expect.that_dict(targets.target_v7[_TagsInfo].exec_properties).contains_exactly({})

def _test_py_test_multi_tpu_targets(name):
    torch_tpu_py_test(
        name = name + "_subject",
        srcs = ["build_defs_test.py"],
        main = "build_defs_test.py",
        is_wheel_test = True,
        tags = [
            "custom_tag",
            "nofastbuild",
            "noopt",
            "notest",
        ],
        run_on_accelerators = [
            "requires-tpu-v5lite",
            "requires-tpu-v6e",
            "requires-tpu-v7x",
        ],
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
    )
    analysis_test(
        name = name,
        impl = _test_py_test_multi_tpu_targets_impl,
        targets = {
            "target_v5": name + "_subject_" + get_tpu_version_for_testing("requires-tpu-v5lite"),
            "target_v6": name + "_subject_" + get_tpu_version_for_testing("requires-tpu-v6e"),
            "target_v7": name + "_subject_" + get_tpu_version_for_testing("requires-tpu-v7x"),
        },
        attrs = {
            "target_v5": {"aspects": [tags_aspect]},
            "target_v6": {"aspects": [tags_aspect]},
            "target_v7": {"aspects": [tags_aspect]},
        },
    )

def _test_py_test_multi_tpu_targets_existing_fails_on_impl(env, targets):
    """Verifies that pre-existing fails-on-tpu tags are not duplicated when run_on_accelerators expands."""
    target_v5_tags = targets.target_v5[_TagsInfo].tags
    target_v6_tags = targets.target_v6[_TagsInfo].tags

    v7_count_v5 = [t for t in target_v5_tags if t == "fails-on-tpu-v7"]
    env.expect.that_int(len(v7_count_v5)).equals(1)

    v7_count_v6 = [t for t in target_v6_tags if t == "fails-on-tpu-v7"]
    env.expect.that_int(len(v7_count_v6)).equals(1)
    env.expect.that_collection(target_v6_tags).not_contains("nopresubmit")

def _test_py_test_multi_tpu_targets_existing_fails_on(name):
    torch_tpu_py_test(
        name = name + "_subject",
        srcs = ["build_defs_test.py"],
        main = "build_defs_test.py",
        is_wheel_test = True,
        tags = [
            "custom_tag",
            "fails-on-tpu-v7",
        ],
        run_on_accelerators = [
            "requires-tpu-v5lite",
            "requires-tpu-v6e",
        ],
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    analysis_test(
        name = name,
        impl = _test_py_test_multi_tpu_targets_existing_fails_on_impl,
        targets = {
            "target_v5": name + "_subject_" + get_tpu_version_for_testing("requires-tpu-v5lite"),
            "target_v6": name + "_subject_" + get_tpu_version_for_testing("requires-tpu-v6e"),
        },
        attrs = {
            "target_v5": {"aspects": [tags_aspect]},
            "target_v6": {"aspects": [tags_aspect]},
        },
    )

def _test_py_test_single_tpu_target_impl(env, targets):
    """Verifies that torch_tpu_py_test with a single TPU tag creates target with original name."""
    tags = targets.subject[_TagsInfo].tags
    env.expect.that_collection(tags).contains("requires-tpu-v5lite")
    env.expect.that_collection(tags).contains("custom_tag")

def _test_py_test_single_tpu_target(name):
    torch_tpu_py_test(
        name = name + "_subject",
        srcs = ["build_defs_test.py"],
        main = "build_defs_test.py",
        is_wheel_test = True,
        tags = [
            "custom_tag",
            "requires-tpu-v5lite",
        ],
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    analysis_test(
        name = name,
        impl = _test_py_test_single_tpu_target_impl,
        targets = {
            "subject": name + "_subject",
        },
        attrs = {
            "subject": {"aspects": [tags_aspect]},
        },
    )

def _test_py_test_non_tpu_requires_tags_impl(env, targets):
    """Verifies that non-TPU requires-* tags do not trigger multi-target splitting."""
    tags = targets.subject[_TagsInfo].tags
    env.expect.that_collection(tags).contains("requires-net:external")
    env.expect.that_collection(tags).contains("requires-mem:16g")

def _test_py_test_non_tpu_requires_tags(name):
    torch_tpu_py_test(
        name = name + "_subject",
        srcs = ["build_defs_test.py"],
        main = "build_defs_test.py",
        is_wheel_test = True,
        tags = [
            "requires-net:external",
            "requires-mem:16g",
        ],
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    analysis_test(
        name = name,
        impl = _test_py_test_non_tpu_requires_tags_impl,
        targets = {
            "subject": name + "_subject",
        },
        attrs = {
            "subject": {"aspects": [tags_aspect]},
        },
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
    """Verifies that requires_libtpu defaults to True in OSS when a TPU tag is present."""

    # In OSS, because the target has 'requires-tpu-*', requires_libtpu is inferred as True,
    # emitting an sh_test top-level rule. In internal builds, it emits a cc_test.
    rule_kind = targets.subject[_TagsInfo].rule_kind
    env.expect.that_str(rule_kind).equals("sh_test" if is_oss() else "cc_test")

def _test_requires_libtpu_inferred(name):
    torch_tpu_cc_test(
        name = name + "_subject",
        srcs = [],
        tags = ["requires-tpu-v5lite"],
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
    """Verifies that explicitly passing requires_libtpu = False disables wrapping even when a TPU tag is present."""

    # When requires_libtpu = False is explicitly passed, it overrides any 'requires-tpu' tag inference,
    # ensuring the top-level rule remains a cc_test across both OSS and internal builds.
    rule_kind = targets.subject[_TagsInfo].rule_kind
    env.expect.that_str(rule_kind).equals("cc_test")

def _test_requires_libtpu_explicit_false(name):
    torch_tpu_cc_test(
        name = name + "_subject",
        srcs = [],
        requires_libtpu = False,
        tags = ["requires-tpu-v5lite"],
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
        "//csrc/common:dtype",
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
        nobuild_oss = True,  # reason
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
        nobuild_oss = True,  # reason
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
        notest_oss = True,  # reason
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
        notest_oss = True,  # reason
        tags = tags,
    )

    # In an OSS build, notest_oss should enable generating the build_test.
    env.expect.that_collection(tags).contains_exactly(["notest"])
    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly([])

def _test_build_test_excludes_cpu_tags(env):
    """Tests that cpu:* reservation tags are excluded from build_test_tags."""
    tags = ["custom_tag", "cpu:8", "cpu:16"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly(["custom_tag"])

def _test_build_test_excludes_tpu_tags(env):
    """Tests that requires-tpu* and TPU generation tags are excluded from build_test_tags."""
    tags = ["custom_tag", "requires-accel-hbm", "requires-tpu", "requires-tpu-v5p:2", "requires-tpu-v5lite"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly(["custom_tag"])

def _test_ops_test_build_test(env):
    """Tests that ops-test generates a build_test in OSS with ops-test and TPU tags excluded."""
    tags = ["custom_tag", "ops-test", "requires-tpu-v5lite"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags,
    )

    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly(["custom_tag"])

def _test_cuda_build_test(env):
    """Tests the notap parameter for CUDA tests."""
    tags = ["requires-gpu-sm90"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    expected_tags = ["manual", "notap", "notest", "requires-gpu", "requires-gpu-sm90"] if is_oss() else ["manual", "notap", "notest", "requires-gpu-sm90"]  # NOTAP_OK=Testing notap tagging
    env.expect.that_collection(tags).contains_exactly(expected_tags)
    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly(["cuda-build-test"])

def _test_cuda_nopresubmit(env):
    """Tests the nopresubmit parameter for CUDA tests."""
    tags = ["requires-gpu-sm90"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit = "reason",
        tags = tags,
    )

    expected_tags = ["manual", "nopresubmit", "requires-gpu", "requires-gpu-sm90"] if is_oss() else ["manual", "nopresubmit", "requires-gpu-sm90"]
    env.expect.that_collection(tags).contains_exactly(expected_tags)
    env.expect.that_bool(result.create_build_test).equals(False)

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
        notest_oss = True,  # reason1
        nobuild_oss = True,  # reason2
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nobuild", "notest"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_notap_notest_oss(env):
    """Tests that notest_oss is a no-op on a target that already sets notap."""
    notap_only_tags = []
    notap_only = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        notap = "reason",
        tags = notap_only_tags,
    )

    both_tags = []
    both = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        notap = "reason",
        notest_oss = True,  # NOTEST_OSS_OK=for testing notest_oss
        tags = both_tags,
    )

    # notap already implies notest, so passing notest_oss as well changes
    # nothing. BUILD files should therefore not set both.
    env.expect.that_collection(notap_only_tags).contains_exactly(["manual", "notap", "notest"])  # NOTAP_OK=Testing notap tagging
    env.expect.that_collection(both_tags).contains_exactly(notap_only_tags)
    env.expect.that_bool(both.create_build_test).equals(notap_only.create_build_test)
    env.expect.that_collection(both.build_test_tags).contains_exactly(notap_only.build_test_tags)

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
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        oss_presubmit_tpu_generation = tpu_gen("v7", reason = "Requires specific hardware for feature testing"),
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v7")
    env.expect.that_str(result.tpu_generation).equals("v7")

def _test_oss_presubmit_tpu_generation_implicit(env):
    tags = ["requires-tpu", "fails-on-tpu-v5"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v7")
    env.expect.that_str(result.tpu_generation).equals("v7")

def _test_oss_presubmit_tpu_generation_implicit_v6(env):
    tags = ["requires-tpu", "fails-on-tpu-v5", "fails-on-tpu-v7"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v6")
    env.expect.that_str(result.tpu_generation).equals("v6")

def _test_oss_presubmit_tpu_generation_implicit_default_v5(env):
    tags = ["requires-tpu"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v5")
    env.expect.that_str(result.tpu_generation).equals("v5")

def _test_get_tpu_version(env):
    env.expect.that_str(get_tpu_version_for_testing("requires-custom_hw")).equals("custom_hw")
    env.expect.that_str(get_tpu_version_for_testing("requires-custom_hw:8")).equals("custom_hw_8")
    env.expect.that_str(get_tpu_version_for_testing("requires-accelerator-v1:4")).equals("accelerator-v1_4")

def _test_get_tpu_chip_count(env):
    """Tests that _get_tpu_chip_count extracts chip counts from multi-chip TPU tags."""
    env.expect.that_int(get_tpu_chip_count_for_testing("requires-tpu-v5lite:8")).equals(8)
    env.expect.that_int(get_tpu_chip_count_for_testing("requires-tpu-v7x:4")).equals(4)
    env.expect.that_bool(get_tpu_chip_count_for_testing("requires-tpu-v5lite") == None).equals(True)
    env.expect.that_bool(get_tpu_chip_count_for_testing("cpu:8") == None).equals(True)

def _test_shard_count_clamping(env):
    """Tests that shard_count is automatically clamped in OSS to the maximum concurrent shards per test."""

    # 1. Exclusive tag in OSS clamps shard_count > 1 down to 1.
    tags_exclusive = ["exclusive", "requires-tpu-v5lite:8"]
    res_exclusive = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags_exclusive,
        shard_count = 8,
    )
    env.expect.that_int(res_exclusive.shard_count).equals(1)

    # 2. Multi-chip tag on OSS runner without 'exclusive' clamps to runner_chips // required_chips.
    clamped_v5_8chips = clamp_shard_count_for_testing(
        is_oss = True,
        tags = ["requires-tpu-v5lite:8"],
        shard_count = 12,
    )
    env.expect.that_int(clamped_v5_8chips).equals(1)

    clamped_v5_4chips = clamp_shard_count_for_testing(
        is_oss = True,
        tags = ["requires-tpu-v5p:4"],
        shard_count = 6,
    )
    env.expect.that_int(clamped_v5_4chips).equals(2)

    clamped_v7_4chips = clamp_shard_count_for_testing(
        is_oss = True,
        tags = ["requires-tpu-v7x:4"],
        shard_count = 4,
    )
    env.expect.that_int(clamped_v7_4chips).equals(1)

    # 3. Single-chip test in OSS preserves shard_count.
    clamped_single_chip = clamp_shard_count_for_testing(
        is_oss = True,
        tags = ["requires-tpu-v5lite"],
        shard_count = 8,
    )
    env.expect.that_int(clamped_single_chip).equals(8)

    # 4. In internal builds (Forge), shard_count is never clamped.
    clamped_internal_forge = clamp_shard_count_for_testing(
        is_oss = False,
        tags = ["exclusive", "requires-tpu-v5lite:8"],
        shard_count = 12,
    )
    env.expect.that_int(clamped_internal_forge).equals(12)

# --- OSS RBE routing tests ---

_V5_RBE_TAGS = ["requires-tpu"]

_V5_RBE_EXEC_PROPERTIES = {
    "test.Pool": "ci_tpu_v5_full_rbe",
    "test.dockerPrivileged": "true",
    "test.dockerRunAsRoot": "true",
}

def _rbe_exec_properties(tpu_generation, tags = _V5_RBE_TAGS, exec_properties = None):
    kwargs = {} if exec_properties == None else {"exec_properties": exec_properties}
    add_rbe_exec_properties_for_testing(kwargs, tags, tpu_generation)
    return kwargs.get("exec_properties", {})

def _test_rbe_oss_v5_test_gets_pool(env):
    """Tests that an OSS single-chip v5 test goes to the v5e RBE pool."""
    env.expect.that_dict(_rbe_exec_properties("v5")).contains_exactly(
        _V5_RBE_EXEC_PROPERTIES,
    )

def _test_rbe_internal_test_gets_nothing(env):
    """Tests that internal builds (tpu_generation = None) never get RBE exec_properties."""
    env.expect.that_dict(_rbe_exec_properties(None)).contains_exactly({})

def _test_rbe_test_without_pool_gets_nothing(env):
    """Tests that CPU, v7 and multi-chip v5 tests don't get a TPU pool."""
    for tpu_generation, tags in [
        (None, []),
        # TODO: b/567928058 - Once v7x and 8-chip v5e RBE worker pools are added
        # to _RBE_TPU_POOLS, move these two cases into positive pool tests
        # expecting `_V5_RBE_EXEC_PROPERTIES | {"test.Pool": "ci_tpu_v7_full_rbe"}`
        # and `_V5_RBE_EXEC_PROPERTIES | {"test.Pool": "ci_tpu_v5_8chip_full_rbe"}`.
        ("v7", ["requires-tpu-v7x", "requires-tpu"]),
        ("v5", ["requires-tpu", "requires-tpu-v5lite:8"]),
    ]:
        env.expect.where(tpu_generation = tpu_generation, tags = tags).that_dict(
            _rbe_exec_properties(tpu_generation, tags),
        ).contains_exactly({})

def _test_rbe_caller_exec_properties_win(env):
    """Tests that exec_properties the caller set win over the RBE ones."""
    exec_properties = _rbe_exec_properties(
        "v5",
        exec_properties = {"test.Pool": "custom_pool", "test.mem": "32g"},
    )

    env.expect.that_dict(exec_properties).contains_exactly({
        "test.Pool": "custom_pool",
        "test.dockerPrivileged": "true",
        "test.dockerRunAsRoot": "true",
        "test.mem": "32g",
    })

def _test_rbe_macro_exec_properties_impl(env, targets):
    """Verifies exec_properties emitted by torch_tpu_py_test and torch_tpu_cc_test."""
    expected_v5 = _V5_RBE_EXEC_PROPERTIES if is_oss() else {}
    expected_override = (
        _V5_RBE_EXEC_PROPERTIES | {"test.Pool": "custom_pool", "test.mem": "32g"}
    ) if is_oss() else {"test.Pool": "custom_pool", "test.mem": "32g"}
    expected_cc_override = (
        _V5_RBE_EXEC_PROPERTIES | {"test.Pool": "custom_pool", "test.mem": "32g"}
    ) if is_oss() else {"cpp_link.mem": "20g", "test.Pool": "custom_pool", "test.mem": "32g"}

    env.expect.that_dict(targets.py_v5[_TagsInfo].exec_properties).contains_exactly(expected_v5)
    env.expect.that_dict(targets.py_override[_TagsInfo].exec_properties).contains_exactly(
        expected_override,
    )
    env.expect.that_dict(targets.py_cpu[_TagsInfo].exec_properties).contains_exactly({})

    # TODO: b/567928058 - Expect `_V5_RBE_EXEC_PROPERTIES | {"test.Pool": "ci_tpu_v5_8chip_full_rbe"}`
    # in OSS once the 8-chip v5e RBE pool is added to _RBE_TPU_POOLS.
    env.expect.that_dict(targets.py_multichip[_TagsInfo].exec_properties).contains_exactly({})

    # TODO: b/567928058 - Expect `_V5_RBE_EXEC_PROPERTIES | {"test.Pool": "ci_tpu_v7_full_rbe"}`
    # in OSS once the v7x RBE pool is added to _RBE_TPU_POOLS.
    env.expect.that_dict(targets.py_v7[_TagsInfo].exec_properties).contains_exactly({})

    env.expect.that_dict(targets.cc_v5[_TagsInfo].exec_properties).contains_exactly(expected_v5)
    env.expect.that_dict(targets.cc_override[_TagsInfo].exec_properties).contains_exactly(
        expected_cc_override,
    )

def _test_rbe_macro_exec_properties(name):
    common_py_kwargs = {
        "srcs": ["build_defs_test.py"],
        "main": "build_defs_test.py",
        "is_wheel_test": True,
        "nobuild": "Analysis test subject",
        "nolocal": "Analysis test subject",
        "notap": "Analysis test subject",  # NOTAP_OK=Analysis test subject
    }
    torch_tpu_py_test(
        name = name + "_py_v5",
        tags = ["requires-tpu-v5lite"],
        **common_py_kwargs
    )
    torch_tpu_py_test(
        name = name + "_py_override",
        tags = ["requires-tpu-v5lite"],
        exec_properties = {"test.Pool": "custom_pool", "test.mem": "32g"},
        **common_py_kwargs
    )
    torch_tpu_py_test(
        name = name + "_py_cpu",
        **common_py_kwargs
    )
    torch_tpu_py_test(
        name = name + "_py_multichip",
        tags = ["requires-tpu-v5lite:8"],
        **common_py_kwargs
    )
    torch_tpu_py_test(
        name = name + "_py_v7",
        tags = ["requires-tpu-v7x"],
        oss_presubmit_tpu_generation = tpu_gen("v7", reason = "Analysis test subject"),
        **common_py_kwargs
    )
    torch_tpu_cc_test(
        name = name + "_cc_v5",
        srcs = [],
        tags = ["requires-tpu-v5lite"],
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    torch_tpu_cc_test(
        name = name + "_cc_override",
        srcs = [],
        tags = ["requires-tpu-v5lite"],
        exec_properties = {"cpp_link.mem": "20g", "test.Pool": "custom_pool", "test.mem": "32g"},
        nobuild = "Analysis test subject",
        nolocal = "Analysis test subject",
        notap = "Analysis test subject",
    )
    subject_names = [
        "py_v5",
        "py_override",
        "py_cpu",
        "py_multichip",
        "py_v7",
        "cc_v5",
        "cc_override",
    ]
    analysis_test(
        name = name,
        impl = _test_rbe_macro_exec_properties_impl,
        targets = {k: name + "_" + k for k in subject_names},
        attrs = {k: {"aspects": [tags_aspect]} for k in subject_names},
    )

def build_defs_test_suite(name):
    """Creates a test suite for build_defs.bzl, which will run all tests in this file.

    Args:
        name: The name of the test suite. All tests in this suite will be prefixed with this name.
    """
    test_suite(
        name = name + "_rules_testing",
        tests = [
            # go/keep-sorted start
            _test_macro_tags,
            _test_py_test_multi_tpu_targets,
            _test_py_test_multi_tpu_targets_existing_fails_on,
            _test_py_test_non_tpu_requires_tags,
            _test_py_test_select_env,
            _test_py_test_single_tpu_target,
            _test_rbe_macro_exec_properties,
            _test_requires_libtpu_explicit_false,
            _test_requires_libtpu_explicit_true,
            _test_requires_libtpu_inferred,
            # go/keep-sorted end
        ],
        basic_tests = [
            # go/keep-sorted start
            _test_build_test_excludes_cpu_tags,
            _test_build_test_excludes_tpu_tags,
            _test_cuda_build_test,
            _test_cuda_nopresubmit,
            _test_get_tpu_chip_count,
            _test_get_tpu_version,
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
            _test_ops_test_build_test,
            _test_oss_manual_nonightly_oss_tag,
            _test_oss_manual_nopresubmit_oss_tag,
            _test_oss_nobuild_oss,
            _test_oss_nonightly_oss,
            _test_oss_nopresubmit_oss,
            _test_oss_notap_notest_oss,
            _test_oss_notest_oss,
            _test_oss_notest_oss_nobuild_oss,
            _test_oss_presubmit_tpu_generation_explicit,
            _test_oss_presubmit_tpu_generation_implicit,
            _test_oss_presubmit_tpu_generation_implicit_default_v5,
            _test_oss_presubmit_tpu_generation_implicit_v6,
            _test_rbe_caller_exec_properties_win,
            _test_rbe_internal_test_gets_nothing,
            _test_rbe_oss_v5_test_gets_pool,
            _test_rbe_test_without_pool_gets_nothing,
            _test_shard_count_clamping,
            # go/keep-sorted end
        ],
    )
    native.test_suite(
        name = name,
        tests = [":" + name + "_rules_testing"],
    )
