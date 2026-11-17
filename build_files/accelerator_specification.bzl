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

"""Target accelerator compatibility and run_on configuration specifications."""

load(
    "//build_files:accelerator_specification_internal.bzl",
    # go/keep-sorted start
    _ALL_TPUS = "ALL_TPUS",
    _ALL_TPU_MODELS = "ALL_TPU_MODELS",
    _ANY_TPU = "ANY_TPU",
    _OSS_TPU_MODELS = "OSS_TPU_MODELS",
    _TPU_GENERATION_TO_MODELS = "TPU_GENERATION_TO_MODELS",
    _TPU_GENERATION_TO_NUMBER = "TPU_GENERATION_TO_NUMBER",
    _TPU_MODELS = "TPU_MODELS",
    _TPU_MODEL_PREFERENCES = "TPU_MODEL_PREFERENCES",
    _TPU_MODEL_TO_GENERATION = "TPU_MODEL_TO_GENERATION",
    _accelerator_config = "accelerator_config",
    _expand_tpu_names = "expand_tpu_names",
    _has_hardware = "has_hardware",
    _resolve_google3_legacy_config = "resolve_google3_legacy_config",
    _resolve_oss_legacy_config = "resolve_oss_legacy_config",
    _validate_specification = "validate_specification",
    _validate_workflow_configuration = "validate_workflow_configuration",
    # go/keep-sorted end
)

visibility(
    [
        "//...",
        # Visibility is not ignored for .bzl loads in experimental
        # unlike regular targets.
        # copybara:uncomment "//experimental/users/...",
    ],
)

# Sentinel constants for run_on_config() (e.g. `run_on_config(presubmit = ANY_TPU, ...)`).
# `ANY_TPU`: Select a single TPU model based on priority order.
# `ALL_TPUS`: Run workflow on all compatible TPU models.
ANY_TPU = _ANY_TPU
ALL_TPUS = _ALL_TPUS

# Default ranking of which hardware generations to prioritize when automatically
# assigning an OSS presubmit runner to a test.
TPU_GENERATION_PREFERENCES = list({
    _TPU_MODEL_TO_GENERATION[model]: True
    for model in _TPU_MODEL_PREFERENCES
    if _has_hardware(model, is_oss = True)
}.keys())

# Maps a `requires-tpu-<version>` tag to the TPU generation it corresponds to.
# Hardware in OSS CI is selected by the `presubmit-v<N>` tag filters in .bazelrc,
# not by `requires-tpu-<version>`, so a test asking for a specific accelerator
# must be given the matching generation.
TPU_VERSION_TO_GENERATION = {
    info["requires_tag"]: _TPU_MODEL_TO_GENERATION[model]
    for model, info in _TPU_MODELS.items()
}

# Number of TPU chips available on a single OSS CI runner VM per generation.
OSS_TPU_RUNNER_CHIPS = {
    gen: _TPU_MODELS[model]["max_chips"]
    for gen, model in _OSS_TPU_MODELS.items()
}

def tpus(
        *,
        since = None,
        only = None,
        exclude = None,
        min_chips = 1):
    """Returns a TPU compatibility specification matching the given constraints.

    Called with no arguments, `tpus()` matches every TPU model with `min_chips = 1`,
    and each argument narrows that set further (logical AND).

    Args:
        since: Optional bare TPU generation string (e.g. "v5", "v6", "v7")
            specifying the minimum generation (inclusive). Cannot be combined
            with `only`, and cannot be a specific variant like "v5e".
        only: Optional non-empty list of TPU model or generation names to
            include (e.g. `["v5e", "v7x"]` or `["v5"]`). Cannot be combined
            with `since` or `exclude`.
        exclude: Optional list of TPU model or generation names to exclude
            (e.g. `["v7x"]` or `["v5"]`). Cannot be combined with `only`.
        min_chips: Minimum number of TPU chips required on a single host
            (defaults to 1). Narrows the set to models whose host topology
            supports at least `min_chips` chips (`v5e` and `v6e` support up to
            8 chips; `v5p` and `v7x` support up to 4 chips).

    Returns:
        A dict with `models` (non-empty list of TPU model strings) and `min_chips`.
    """
    if only != None and (since != None or exclude != None):
        fail("Cannot combine 'only' with 'since' or 'exclude' in tpus().")

    if type(min_chips) != "int" or min_chips < 1:
        fail("'min_chips' in tpus() must be a positive integer, got: %s" % repr(min_chips))

    models = [m for m in _ALL_TPU_MODELS if _TPU_MODELS[m]["max_chips"] >= min_chips]

    if since != None:
        if since not in _TPU_GENERATION_TO_NUMBER:
            fail("Invalid 'since' value %s in tpus(). 'since' must be a bare TPU generation (%s), not a variant." % (
                repr(since),
                ", ".join(_TPU_GENERATION_TO_NUMBER.keys()),
            ))
        min_gen = _TPU_GENERATION_TO_NUMBER[since]
        models = [m for m in models if _TPU_MODELS[m]["generation"] >= min_gen]

    if only != None:
        only_set = _expand_tpu_names(only, "only")
        models = [m for m in models if m in only_set]

    if exclude != None:
        exclude_set = _expand_tpu_names(exclude, "exclude")
        models = [m for m in models if m not in exclude_set]

    if not models:
        fail("tpus(since=%s, only=%s, exclude=%s, min_chips=%s) matched no compatible TPU models." % (
            repr(since),
            repr(only),
            repr(exclude),
            repr(min_chips),
        ))

    return {
        "models": models,
        "min_chips": min_chips,
    }

# Shared constant for single-chip tests that require SparseCore.
SPARSECORE_TPUS = tpus(only = ["v5p", "v6e", "v7x"])

def tpu_config(model):
    """Creates a TPU accelerator configuration for a TPU model or generation.

    Args:
        model: Exact TPU model (e.g. "v5p") or bare generation (e.g. "v5"). If a
            bare generation is specified, it is resolved lazily against
            `compatible_accelerators` by default TPU model priority order into a
            single compatible TPU model.

    Returns:
        An accelerator configuration dict for the TPU model or generation.
    """
    if model in _TPU_MODELS or model in _TPU_GENERATION_TO_MODELS:
        return _accelerator_config(accelerator_type = "tpu", model = model)
    fail("Unknown TPU model or generation %s in tpu_config(). Valid models: %s; valid generations: %s." % (
        repr(model),
        _ALL_TPU_MODELS,
        list(_TPU_GENERATION_TO_NUMBER.keys()),
    ))

def run_on_config(
        *,
        reason = None,
        presubmit = ANY_TPU,
        postsubmit = None,
        nightly = None):
    """Creates a `run_on` configuration mapping CI workflows to target accelerators.

    When `postsubmit` or `nightly` is omitted (`None`), environment-specific
    defaults matching existing CI behavior are used during resolution:
    - google3: `postsubmit` defaults to `ANY_TPU` (reusing the `presubmit` TPU
      when `presubmit` selects a single model). `nightly` is validated against
      `compatible_accelerators` when set and otherwise ignored (TAP has no
      separate nightly workflow).
    - OSS: `postsubmit` and `nightly` both default to `ALL_TPUS`.

    Args:
        reason: A string explaining why a non-default `run_on` configuration is
            used, or None when using the default configuration.
        presubmit: Accelerator selection for presubmit (`ANY_TPU`, `ALL_TPUS`,
            or a list of `tpu_config()` dicts, including `[]`). Defaults to
            `ANY_TPU`.
        postsubmit: Optional accelerator selection for postsubmit (`ANY_TPU`,
            `ALL_TPUS`, or a list of `tpu_config()` dicts, including `[]`).
        nightly: Optional accelerator selection for nightly (`ANY_TPU`,
            `ALL_TPUS`, or a list of `tpu_config()` dicts, including `[]`).

    Returns:
        A dict representing the `run_on` configuration.
    """
    if reason != None and (type(reason) != "string" or not reason):
        fail("'reason' in run_on_config() must be a non-empty string when provided.")
    _validate_workflow_configuration(presubmit, "presubmit")
    if postsubmit != None:
        _validate_workflow_configuration(postsubmit, "postsubmit")
    if nightly != None:
        _validate_workflow_configuration(nightly, "nightly")

    if presubmit == [] and postsubmit == [] and nightly == []:
        fail("At least one workflow in run_on_config() must be non-empty.")

    has_non_default = (
        presubmit != ANY_TPU or
        postsubmit != None or
        nightly != None
    )
    if has_non_default and not reason:
        fail("A non-empty 'reason' must be provided in run_on_config() when overriding default run_on settings.")

    return {
        "reason": reason,
        "presubmit": presubmit,
        "postsubmit": postsubmit,
        "nightly": nightly,
    }

def resolve_legacy_accelerator_config(
        name,
        compatible_accelerators,
        *,
        run_on = None,
        is_oss = False,
        runs_in_oss = True,
        has_notap = False,
        has_nopresubmit = False,
        supports_multi_target = True):
    """Maps `compatible_accelerators` and `run_on` onto existing tag and macro attributes.

    Evaluates hardware availability and workflow representability only for the
    active `is_oss` environment so `if_oss(...)` branches in `BUILD` files work
    transparently. Fails fast if a requested workflow combination cannot be
    faithfully executed by the current tag / `run_on_accelerators` infrastructure.

    Args:
        name: Base name of the test target.
        compatible_accelerators: Dict returned by `tpus()`.
        run_on: Optional dict returned by `run_on_config()` (or `None` for defaults).
        is_oss: Whether to resolve for OSS CI (`True`) or internal google3 (`False`).
        runs_in_oss: Whether the test executes in OSS (`False` when `notest_oss`,
            `nobuild_oss`, `nobuild`, or `notap` disables OSS test execution).
        has_notap: Whether `notap` is set on the target.
        has_nopresubmit: Whether break-glass `nopresubmit` is set on the target.
        supports_multi_target: Whether the calling rule supports `run_on_accelerators`
            expansion (`True` for `torch_tpu_py_test`, `False` for `torch_tpu_cc_test`).

    Returns:
        A dict of resolved legacy macro attributes; see `resolve_google3_legacy_config` for keys.
    """
    compat, effective_run_on = _validate_specification(compatible_accelerators, run_on)

    if is_oss:
        return _resolve_oss_legacy_config(
            name = name,
            compat_models = compat["models"],
            min_chips = compat["min_chips"],
            effective_run_on = effective_run_on,
            runs_in_oss = runs_in_oss,
            supports_multi_target = supports_multi_target,
        )
    return _resolve_google3_legacy_config(
        name = name,
        compat_models = compat["models"],
        min_chips = compat["min_chips"],
        effective_run_on = effective_run_on,
        has_notap = has_notap,
        has_nopresubmit = has_nopresubmit,
        supports_multi_target = supports_multi_target,
    )
