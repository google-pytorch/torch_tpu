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

"""Unit tests for accelerator_specification.bzl."""

load("@rules_testing//lib:test_suite.bzl", "test_suite")
load(
    "//build_files:accelerator_specification.bzl",
    # go/keep-sorted start
    "ALL_TPUS",
    "ANY_TPU",
    "OSS_TPU_RUNNER_CHIPS",
    "SPARSECORE_TPUS",
    "TPU_GENERATION_PREFERENCES",
    "TPU_VERSION_TO_GENERATION",
    "resolve_legacy_accelerator_config",
    "run_on_config",
    "tpu_config",
    "tpus",
    # go/keep-sorted end
)
load(
    "//build_files:accelerator_specification_internal.bzl",
    # go/keep-sorted start
    "OSS_TPU_MODELS",
    "TPU_MODELS",
    "TPU_MODEL_PREFERENCES",
    "oss_availability_error",
    # go/keep-sorted end
)

visibility("private")

def _test_tpu_catalog_invariants(env):
    """Verifies each TPU model's generation, hardware tags, and derived OSS runner tables."""

    # Expect to be updated when new models are added.
    env.expect.that_collection([
        (
            model,
            info["generation"],
            info["requires_tag"],
            info["max_chips"],
            info["ok_to_wait_tag"],
            info["ok_to_wait_min_chips"],
        )
        for model, info in TPU_MODELS.items()
    ]).contains_exactly([
        ("v5e", 5, "requires-tpu-v5lite", 8, None, None),
        ("v5p", 5, "requires-tpu-v5p", 4, "ok-to-wait-for-v5p-in-presubmit", 2),
        ("v6e", 6, "requires-tpu-v6e", 8, "ok-to-wait-for-v6e-in-presubmit", 1),
        ("v7x", 7, "requires-tpu-v7x", 4, "ok-to-wait-for-v7x-in-presubmit", 1),
    ])
    env.expect.that_collection(TPU_MODEL_PREFERENCES).contains_exactly(
        TPU_MODELS.keys(),
    )

    env.expect.that_dict(OSS_TPU_MODELS).contains_exactly({
        "v5": "v5e",
        "v6": "v6e",
        "v7": "v7x",
    })
    env.expect.that_collection(TPU_GENERATION_PREFERENCES).contains_exactly([
        "v5",
        "v7",
        "v6",
    ]).in_order()
    env.expect.that_dict(TPU_VERSION_TO_GENERATION).contains_exactly({
        "requires-tpu-v5lite": "v5",
        "requires-tpu-v5p": "v5",
        "requires-tpu-v6e": "v6",
        "requires-tpu-v7x": "v7",
    })
    env.expect.that_dict(OSS_TPU_RUNNER_CHIPS).contains_exactly({
        "v5": 8,
        "v6": 8,
        "v7": 4,
    })

def _test_tpus_default_returns_all_models_and_min_chips_one(env):
    """Verifies default tpus() returns all registered TPU models and min_chips = 1."""
    result = tpus()

    env.expect.that_collection(result["models"]).contains_exactly(["v5e", "v5p", "v6e", "v7x"])
    env.expect.that_int(result["min_chips"]).equals(1)

def _test_tpus_since_filters_older_generations(env):
    """Verifies tpus(since=...) excludes earlier generations while keeping matching and newer ones."""
    result = tpus(since = "v6")

    # Expect to be updated when new models are added.
    env.expect.that_collection(result["models"]).contains_exactly(["v6e", "v7x"])
    env.expect.that_int(result["min_chips"]).equals(1)

def _test_tpus_only_selects_models_and_generations(env):
    """Verifies tpus(only=...) restricts results to the specified models or generations."""
    by_models = tpus(only = ["v5e", "v7x"])
    by_v5 = tpus(only = ["v5"])
    by_v6 = tpus(only = ["v6"])
    by_v7 = tpus(only = ["v7"])

    env.expect.that_collection(by_models["models"]).contains_exactly(["v5e", "v7x"])
    env.expect.that_collection(by_v5["models"]).contains_at_least(["v5e", "v5p"])
    env.expect.that_collection(by_v5["models"]).contains_none_of(["v6e", "v7x"])
    env.expect.that_collection(by_v6["models"]).contains("v6e")
    env.expect.that_collection(by_v6["models"]).contains_none_of(["v5e", "v5p", "v7x"])
    env.expect.that_collection(by_v7["models"]).contains("v7x")
    env.expect.that_collection(by_v7["models"]).contains_none_of(["v5e", "v5p", "v6e"])

def _test_tpus_exclude_removes_models_and_generations(env):
    """Verifies tpus(exclude=...) omits excluded models or generations without affecting others."""
    without_v7x = tpus(exclude = ["v7x"])
    without_v5 = tpus(exclude = ["v5"])
    since_v6_without_v7x = tpus(since = "v6", exclude = ["v7x"])

    env.expect.that_collection(without_v7x["models"]).contains_at_least(["v5e", "v5p", "v6e"])
    env.expect.that_collection(without_v7x["models"]).not_contains("v7x")

    env.expect.that_collection(without_v5["models"]).contains_at_least(["v6e", "v7x"])
    env.expect.that_collection(without_v5["models"]).contains_none_of(["v5e", "v5p"])

    env.expect.that_collection(since_v6_without_v7x["models"]).contains("v6e")
    env.expect.that_collection(since_v6_without_v7x["models"]).contains_none_of(["v5e", "v5p", "v7x"])

def _test_tpus_min_chips_filters_and_preserves_chip_count(env):
    """Verifies tpus(min_chips=...) filters by host topology and preserves min_chips."""
    eight_chips = tpus(min_chips = 8)
    four_chips = tpus(min_chips = 4)

    env.expect.that_int(eight_chips["min_chips"]).equals(8)
    env.expect.that_collection(eight_chips["models"]).contains_at_least(["v5e", "v6e"])
    env.expect.that_collection(eight_chips["models"]).contains_none_of(["v5p", "v7x"])

    env.expect.that_int(four_chips["min_chips"]).equals(4)
    env.expect.that_collection(four_chips["models"]).contains_at_least(
        ["v5e", "v5p", "v6e", "v7x"],
    )

def _test_sparsecore_tpus_excludes_v5e(env):
    """Verifies SPARSECORE_TPUS includes SparseCore-capable TPUs with min_chips = 1."""
    env.expect.that_collection(SPARSECORE_TPUS["models"]).contains_at_least(
        ["v5p", "v6e", "v7x"],
    )
    env.expect.that_collection(SPARSECORE_TPUS["models"]).not_contains("v5e")
    env.expect.that_int(SPARSECORE_TPUS["min_chips"]).equals(1)

def _test_tpu_config_stores_model_or_generation(env):
    """Verifies tpu_config stores exact models and bare generations for lazy resolution."""
    exact_v5p = tpu_config("v5p")
    gen_v5 = tpu_config("v5")
    gen_v6 = tpu_config("v6")
    gen_v7 = tpu_config("v7")

    env.expect.that_dict(exact_v5p).contains_exactly({"type": "tpu", "model": "v5p"})
    env.expect.that_dict(gen_v5).contains_exactly({"type": "tpu", "model": "v5"})
    env.expect.that_dict(gen_v6).contains_exactly({"type": "tpu", "model": "v6"})
    env.expect.that_dict(gen_v7).contains_exactly({"type": "tpu", "model": "v7"})

def _test_resolve_default_run_on_single_target_mapping(env):
    """Verifies default run_on (omitted or explicit run_on_config()) maps to a single target."""
    g3_result = resolve_legacy_accelerator_config(
        name = "my_test",
        compatible_accelerators = tpus(),
        is_oss = False,
    )
    g3_explicit_default = resolve_legacy_accelerator_config(
        name = "my_test",
        compatible_accelerators = tpus(),
        run_on = run_on_config(),
        is_oss = False,
    )
    oss_result = resolve_legacy_accelerator_config(
        name = "my_test",
        compatible_accelerators = tpus(),
        is_oss = True,
    )
    oss_explicit_default = resolve_legacy_accelerator_config(
        name = "my_test",
        compatible_accelerators = tpus(),
        run_on = run_on_config(),
        is_oss = True,
    )

    env.expect.that_collection(g3_result["tags"]).contains_exactly(["requires-tpu-v5lite"])
    env.expect.that_bool(g3_result["run_on_accelerators"] == None).equals(True)
    env.expect.that_bool(g3_result["nopresubmit"] == None).equals(True)
    env.expect.that_dict(g3_explicit_default).contains_exactly(g3_result)

    env.expect.that_collection(oss_result["tags"]).contains_exactly(["requires-tpu-v5lite"])
    env.expect.that_bool(oss_result["run_on_accelerators"] == None).equals(True)
    env.expect.that_bool(oss_result["oss_presubmit_tpu_generation"] == None).equals(True)
    env.expect.that_dict(oss_explicit_default).contains_exactly(oss_result)

def _test_resolve_compatibility_exclusions_and_sparsecore_fallback(env):
    """Verifies OSS fails-on-tpu-* tags and ANY_TPU fallback from v5p to v7x on SPARSECORE_TPUS."""
    no_v7_g3 = resolve_legacy_accelerator_config(
        name = "no_v7",
        compatible_accelerators = tpus(exclude = ["v7"]),
        is_oss = False,
    )
    no_v7_oss = resolve_legacy_accelerator_config(
        name = "no_v7",
        compatible_accelerators = tpus(exclude = ["v7"]),
        is_oss = True,
    )
    sparse_g3 = resolve_legacy_accelerator_config(
        name = "sparse",
        compatible_accelerators = SPARSECORE_TPUS,
        is_oss = False,
    )
    sparse_oss = resolve_legacy_accelerator_config(
        name = "sparse",
        compatible_accelerators = SPARSECORE_TPUS,
        is_oss = True,
    )

    env.expect.that_collection(no_v7_g3["tags"]).contains_exactly(["requires-tpu-v5lite"])
    env.expect.that_collection(no_v7_oss["tags"]).contains_exactly([
        "requires-tpu-v5lite",
        "fails-on-tpu-v7",
    ])

    env.expect.that_collection(sparse_g3["tags"]).contains_exactly(["requires-tpu-v5p"])
    env.expect.that_collection(sparse_oss["tags"]).contains_exactly([
        "requires-tpu-v7x",
        "fails-on-tpu-v5",
    ])
    env.expect.that_bool(sparse_oss["oss_presubmit_tpu_generation"] == None).equals(True)

def _test_resolve_explicit_presubmit_and_ok_to_wait_tags(env):
    """Verifies explicit presubmit sets OSS presubmit generation and google3 ok-to-wait tags."""
    explicit_v7_run_on = run_on_config(
        reason = "Load balancing onto v7",
        presubmit = [tpu_config("v7")],
    )

    g3_v7 = resolve_legacy_accelerator_config(
        name = "v7_pre",
        compatible_accelerators = tpus(),
        run_on = explicit_v7_run_on,
        is_oss = False,
    )
    oss_v7 = resolve_legacy_accelerator_config(
        name = "v7_pre",
        compatible_accelerators = tpus(),
        run_on = explicit_v7_run_on,
        is_oss = True,
    )

    # Default ANY_TPU on v7-only compatibility with break-glass nopresubmit does NOT auto-add
    # ok-to-wait-for-v7x-in-presubmit.
    g3_any_v7 = resolve_legacy_accelerator_config(
        name = "v7_any",
        compatible_accelerators = tpus(since = "v7"),
        is_oss = False,
        has_nopresubmit = True,
    )

    # Single-chip v5p does not need ok-to-wait, while multi-chip v5p does when explicitly requested.
    g3_single_v5p = resolve_legacy_accelerator_config(
        name = "v5p_single",
        compatible_accelerators = tpus(),
        run_on = run_on_config(reason = "Run on v5p", presubmit = [tpu_config("v5p")]),
        is_oss = False,
    )
    g3_multi_v5p = resolve_legacy_accelerator_config(
        name = "v5p_multi",
        compatible_accelerators = tpus(min_chips = 4),
        run_on = run_on_config(reason = "Run multi-chip v5p", presubmit = [tpu_config("v5p")]),
        is_oss = False,
    )

    env.expect.that_collection(g3_v7["tags"]).contains_exactly([
        "requires-tpu-v7x",
        "ok-to-wait-for-v7x-in-presubmit",
    ])
    env.expect.that_collection(oss_v7["tags"]).contains_exactly(["requires-tpu-v7x"])
    env.expect.that_dict(oss_v7["oss_presubmit_tpu_generation"]).contains_exactly({
        "generation": "v7",
        "reason": "Load balancing onto v7",
    })

    env.expect.that_collection(g3_any_v7["tags"]).contains_exactly(["requires-tpu-v7x"])
    env.expect.that_collection(g3_single_v5p["tags"]).contains_exactly(["requires-tpu-v5p"])
    env.expect.that_collection(g3_multi_v5p["tags"]).contains_exactly([
        "requires-tpu-v5p:4",
        "ok-to-wait-for-v5p-in-presubmit",
    ])

def _test_resolve_multichip_tags_and_oss_exclusive(env):
    """Verifies min_chips > 1 maps to Forge full-host tags and OSS :4/:8 tags plus exclusive."""
    four_g3 = resolve_legacy_accelerator_config(
        name = "four_chips",
        compatible_accelerators = tpus(min_chips = 4),
        is_oss = False,
    )
    four_oss = resolve_legacy_accelerator_config(
        name = "four_chips",
        compatible_accelerators = tpus(min_chips = 4),
        is_oss = True,
    )
    eight_g3 = resolve_legacy_accelerator_config(
        name = "eight_chips",
        compatible_accelerators = tpus(min_chips = 8),
        is_oss = False,
    )
    eight_oss = resolve_legacy_accelerator_config(
        name = "eight_chips",
        compatible_accelerators = tpus(min_chips = 8),
        is_oss = True,
    )

    env.expect.that_collection(four_g3["tags"]).contains_exactly(["requires-tpu-v5lite:8"])
    env.expect.that_collection(four_oss["tags"]).contains_exactly([
        "requires-tpu-v5lite:4",
        "exclusive",
    ])

    env.expect.that_collection(eight_g3["tags"]).contains_exactly(["requires-tpu-v5lite:8"])
    env.expect.that_collection(eight_oss["tags"]).contains_exactly([
        "requires-tpu-v5lite:8",
        "exclusive",
        "fails-on-tpu-v7",
    ])

def _test_resolve_empty_workflows_and_single_target_narrowing(env):
    """Verifies empty workflows map to nopresubmit / nopresubmit_oss / nonightly_oss and single-target narrowing."""
    skip_pre_run_on = run_on_config(
        reason = "Skip presubmit",
        presubmit = [],
        nightly = [],
    )

    g3_skip_pre = resolve_legacy_accelerator_config(
        name = "skip_pre",
        compatible_accelerators = tpus(),
        run_on = skip_pre_run_on,
        is_oss = False,
    )
    oss_skip_pre = resolve_legacy_accelerator_config(
        name = "skip_pre",
        compatible_accelerators = tpus(),
        run_on = skip_pre_run_on,
        is_oss = True,
    )

    oss_nightly_only = resolve_legacy_accelerator_config(
        name = "nightly_only",
        compatible_accelerators = tpus(),
        run_on = run_on_config(
            reason = "Nightly only",
            presubmit = [],
            postsubmit = [],
            nightly = ALL_TPUS,
        ),
        is_oss = True,
    )

    oss_narrow_all = resolve_legacy_accelerator_config(
        name = "narrow_all",
        compatible_accelerators = tpus(),
        run_on = run_on_config(
            reason = "Only one TPU needed in postsubmit and nightly",
            postsubmit = ANY_TPU,
            nightly = ANY_TPU,
        ),
        is_oss = True,
    )

    env.expect.that_str(g3_skip_pre["nopresubmit"]).equals("Skip presubmit")
    env.expect.that_bool(oss_skip_pre["nopresubmit"] == None).equals(True)
    env.expect.that_str(oss_skip_pre["nopresubmit_oss"]).equals("Skip presubmit")
    env.expect.that_str(oss_skip_pre["nonightly_oss"]).equals("Skip presubmit")

    env.expect.that_str(oss_nightly_only["nopresubmit"]).equals("Nightly only")
    env.expect.that_bool(oss_nightly_only["nopresubmit_oss"] == None).equals(True)
    env.expect.that_bool(oss_nightly_only["nonightly_oss"] == None).equals(True)

    env.expect.that_collection(oss_narrow_all["tags"]).contains_exactly([
        "requires-tpu-v5lite",
        "fails-on-tpu-v6",
        "fails-on-tpu-v7",
    ])

def _test_resolve_multi_model_delegates_to_run_on_accelerators(env):
    """Verifies multi-model postsubmit translates to run_on_accelerators in google3 and OSS."""
    ops_run_on = run_on_config(
        reason = "Cover v5e and v6e",
        postsubmit = [tpu_config("v5e"), tpu_config("v6e")],
    )

    g3_ops = resolve_legacy_accelerator_config(
        name = "ops_test",
        compatible_accelerators = tpus(only = ["v5e", "v6e"]),
        run_on = ops_run_on,
        is_oss = False,
    )
    oss_ops = resolve_legacy_accelerator_config(
        name = "ops_test",
        compatible_accelerators = tpus(only = ["v5e", "v6e"]),
        run_on = ops_run_on,
        is_oss = True,
    )
    oss_post_only_multi = resolve_legacy_accelerator_config(
        name = "ops_oss_post_only",
        compatible_accelerators = tpus(only = ["v5e", "v6e"]),
        run_on = run_on_config(
            reason = "Postsubmit across v5e and v6e",
            presubmit = [],
            postsubmit = [tpu_config("v5e"), tpu_config("v6e")],
        ),
        is_oss = True,
    )

    g3_post_only_multi = resolve_legacy_accelerator_config(
        name = "ops_post_only",
        compatible_accelerators = SPARSECORE_TPUS,
        run_on = run_on_config(
            reason = "Postsubmit across all SparseCore TPUs",
            presubmit = [],
            postsubmit = ALL_TPUS,
        ),
        is_oss = False,
    )

    env.expect.that_collection(g3_ops["run_on_accelerators"]).contains_exactly([
        "requires-tpu-v5lite",
        "requires-tpu-v6e",
    ])
    env.expect.that_bool(g3_ops["nopresubmit"] == None).equals(True)

    env.expect.that_collection(oss_ops["run_on_accelerators"]).contains_exactly([
        "requires-tpu-v5lite",
        "requires-tpu-v6e",
    ])
    env.expect.that_bool(oss_post_only_multi["nopresubmit"] == None).equals(True)
    env.expect.that_str(oss_post_only_multi["nopresubmit_oss"]).equals(
        "Postsubmit across v5e and v6e",
    )

    env.expect.that_collection(g3_post_only_multi["run_on_accelerators"]).contains_exactly([
        "requires-tpu-v5p",
        "requires-tpu-v6e",
        "requires-tpu-v7x",
    ])
    env.expect.that_str(g3_post_only_multi["nopresubmit"]).equals(
        "Postsubmit across all SparseCore TPUs",
    )

def _test_oss_availability_and_notest_oss_fallback(env):
    """Verifies OSS availability errors for v5p and clean fallback when runs_in_oss is False."""
    default_run_on = run_on_config()
    v5p_presubmit = run_on_config(
        reason = "Explicit v5p presubmit",
        presubmit = [tpu_config("v5p")],
    )
    v5_presubmit = run_on_config(
        reason = "Bare v5 presubmit",
        presubmit = [tpu_config("v5")],
    )

    env.expect.that_str(
        oss_availability_error(["v5p"], default_run_on),
    ).equals("none of the TPU models in compatible_accelerators (v5p) have hardware in OSS")
    env.expect.that_str(
        oss_availability_error(SPARSECORE_TPUS["models"], v5p_presubmit),
    ).equals("none of the compatible TPU models for \"v5p\" (v5p), requested in presubmit, have hardware in OSS")
    env.expect.that_str(
        oss_availability_error(SPARSECORE_TPUS["models"], v5_presubmit),
    ).equals("none of the compatible TPU models for \"v5\" (v5p), requested in presubmit, have hardware in OSS")
    env.expect.that_bool(
        oss_availability_error(SPARSECORE_TPUS["models"], default_run_on) == None,
    ).equals(True)

    oss_notest = resolve_legacy_accelerator_config(
        name = "v5p_only",
        compatible_accelerators = tpus(only = ["v5p"]),
        is_oss = True,
        runs_in_oss = False,
    )

    env.expect.that_collection(oss_notest["tags"]).contains_exactly(["requires-tpu-v5p"])
    env.expect.that_bool(oss_notest["run_on_accelerators"] == None).equals(True)

def accelerator_specification_test_suite(name):
    """Creates a test suite for accelerator_specification.bzl.

    Args:
        name: The name of the test suite.
    """
    test_suite(
        name = name,
        basic_tests = [
            # go/keep-sorted start
            _test_oss_availability_and_notest_oss_fallback,
            _test_resolve_compatibility_exclusions_and_sparsecore_fallback,
            _test_resolve_default_run_on_single_target_mapping,
            _test_resolve_empty_workflows_and_single_target_narrowing,
            _test_resolve_explicit_presubmit_and_ok_to_wait_tags,
            _test_resolve_multi_model_delegates_to_run_on_accelerators,
            _test_resolve_multichip_tags_and_oss_exclusive,
            _test_sparsecore_tpus_excludes_v5e,
            _test_tpu_catalog_invariants,
            _test_tpu_config_stores_model_or_generation,
            _test_tpus_default_returns_all_models_and_min_chips_one,
            _test_tpus_exclude_removes_models_and_generations,
            _test_tpus_min_chips_filters_and_preserves_chip_count,
            _test_tpus_only_selects_models_and_generations,
            _test_tpus_since_filters_older_generations,
            # go/keep-sorted end
        ],
    )
