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

"""Implementation helpers behind `accelerator_specification.bzl`.

Holds the TPU catalog and the private validation and legacy tag-mapping logic. Load
`accelerator_specification.bzl` instead; only it (and its test) should load this file.
"""

visibility("private")

# Sentinel constants exported by `accelerator_specification.bzl`, defined here so internal helpers
# can compare against them without creating a load cycle.
ANY_TPU = "any_tpu"
ALL_TPUS = "all_tpus"

_DEFAULT_RUN_ON = {
    # go/keep-sorted start
    "nightly": None,
    "postsubmit": None,
    "presubmit": ANY_TPU,
    "reason": None,
    # go/keep-sorted end
}

# Hardware properties of each TPU model. `requires_tag` and `ok_to_wait_tag` are google3 tags that
# Copybara rewrites to the matching OSS tags, so they must stay complete string literals.
TPU_MODELS = {
    # go/keep-sorted start block=yes numeric=yes
    "v5e": {
        # go/keep-sorted start
        "generation": 5,
        "max_chips": 8,
        "ok_to_wait_min_chips": None,
        "ok_to_wait_tag": None,
        "oss_partial_chips": 4,
        "requires_tag": "requires-tpu-v5lite",
        # go/keep-sorted end
    },
    "v5p": {
        # go/keep-sorted start
        "generation": 5,
        "max_chips": 4,
        "ok_to_wait_min_chips": 2,
        "ok_to_wait_tag": "ok-to-wait-for-v5p-in-presubmit",
        "oss_partial_chips": None,
        "requires_tag": "requires-tpu-v5p",
        # go/keep-sorted end
    },
    "v6e": {
        # go/keep-sorted start
        "generation": 6,
        "max_chips": 8,
        "ok_to_wait_min_chips": 1,
        "ok_to_wait_tag": "ok-to-wait-for-v6e-in-presubmit",
        "oss_partial_chips": None,
        "requires_tag": "requires-tpu-v6e",
        # go/keep-sorted end
    },
    "v7x": {
        # go/keep-sorted start
        "generation": 7,
        "max_chips": 4,
        "ok_to_wait_min_chips": 1,
        "ok_to_wait_tag": "ok-to-wait-for-v7x-in-presubmit",
        "oss_partial_chips": None,
        "requires_tag": "requires-tpu-v7x",
        # go/keep-sorted end
    },
    # go/keep-sorted end
}

# Default ranking of TPU models to prioritize when automatically assigning a
# single runner to a workflow (`ANY_TPU`).
TPU_MODEL_PREFERENCES = ["v5e", "v5p", "v7x", "v6e"]

# TPU model of the OSS CI runners for each TPU generation (each OSS runner VM is a full single-host
# VM of that model). TPU models missing here (e.g. v5p) have no hardware in OSS.
OSS_TPU_MODELS = {
    # go/keep-sorted start numeric=yes
    "v5": "v5e",
    "v6": "v6e",
    "v7": "v7x",
    # go/keep-sorted end
}

def _generation_name(gen_num):
    """Formats an integer TPU generation number (e.g. 5) as a generation name ('v5')."""
    return "v%d" % gen_num

def _build_generation_to_models_mapping():
    """Constructs a dictionary mapping TPU generation names (e.g. 'v5') to lists of TPU model names."""
    gen_to_models = {}
    for model, info in TPU_MODELS.items():
        gen = _generation_name(info["generation"])
        if gen not in gen_to_models:
            gen_to_models[gen] = []
        gen_to_models[gen].append(model)
    return gen_to_models

ALL_TPU_MODELS = list(TPU_MODELS.keys())
TPU_MODEL_TO_GENERATION = {
    model: _generation_name(info["generation"])
    for model, info in TPU_MODELS.items()
}
TPU_GENERATION_TO_NUMBER = {
    _generation_name(info["generation"]): info["generation"]
    for info in TPU_MODELS.values()
}
TPU_GENERATION_TO_MODELS = _build_generation_to_models_mapping()

def has_hardware(model, is_oss):
    """Returns whether the environment has CI hardware for `model`.

    Args:
        model: TPU model name from `TPU_MODELS`.
        is_oss: Whether to check OSS CI hardware (`True`) or internal google3 (`False`).

    Returns:
        `True` if the target environment has hardware for `model`, else `False`.
    """
    return not is_oss or OSS_TPU_MODELS.get(TPU_MODEL_TO_GENERATION[model]) == model

def expand_tpu_names(names, arg_name):
    """Validates and expands a list of model or bare generation names into a set of models.

    Args:
        names: List of TPU model or bare generation strings.
        arg_name: Parameter name in `tpus()` ("only" or "exclude") for error messages.

    Returns:
        A dict mapping each expanded TPU model name to `True`.
    """
    if type(names) != "list":
        fail("'%s' in tpus() must be a list of TPU model or generation names, got: %s" % (
            arg_name,
            type(names),
        ))
    if arg_name == "only" and not names:
        fail("'only' in tpus() cannot be empty.")

    expanded = {}
    for item in names:
        if item in TPU_MODELS:
            expanded[item] = True
        elif item in TPU_GENERATION_TO_MODELS:
            for model in TPU_GENERATION_TO_MODELS[item]:
                expanded[model] = True
        else:
            fail("Unknown TPU model or generation %s in tpus(%s=...). Valid models: %s; valid generations: %s." % (
                repr(item),
                arg_name,
                ALL_TPU_MODELS,
                list(TPU_GENERATION_TO_NUMBER.keys()),
            ))
    return expanded

def accelerator_config(accelerator_type, model):
    """Creates an accelerator configuration dictionary for use in `run_on_config`.

    Args:
        accelerator_type: The general accelerator type (e.g. "tpu").
        model: Specific accelerator model or bare TPU generation (e.g. "v5e", "v5").

    Returns:
        A dict representing the accelerator configuration.
    """
    return {
        "type": accelerator_type,
        "model": model,
    }

def validate_workflow_configuration(workflow_config, workflow_name):
    """Validates a workflow configuration (presubmit, postsubmit, or nightly).

    Args:
        workflow_config: `ANY_TPU`, `ALL_TPUS`, or a list of `tpu_config()` dicts.
        workflow_name: Workflow parameter name ("presubmit", "postsubmit", or "nightly").
    """
    if workflow_config == ANY_TPU or workflow_config == ALL_TPUS:
        return
    if type(workflow_config) != "list":
        fail("'%s' in run_on_config() must be ANY_TPU, ALL_TPUS, or a list of tpu_config() dicts, got: %s" % (
            workflow_name,
            repr(workflow_config),
        ))
    for item in workflow_config:
        if (
            type(item) != "dict" or
            item.get("type") != "tpu" or
            (item.get("model") not in TPU_MODELS and item.get("model") not in TPU_GENERATION_TO_MODELS)
        ):
            fail("Elements of '%s' in run_on_config() must be created via tpu_config(), got: %s" % (
                workflow_name,
                repr(item),
            ))

def _select_preferred_tpu_model(candidates):
    """Returns the highest-priority model in `TPU_MODEL_PREFERENCES` from `candidates`."""
    for model in TPU_MODEL_PREFERENCES:
        if model in candidates:
            return model
    fail("No preferred TPU model found in %s (expected models from %s)." % (
        candidates,
        ALL_TPU_MODELS,
    ))

def _requires_tag_for_model(model, min_chips, is_oss):
    """Returns the `requires-*` hardware tag for `model` and `min_chips`."""
    info = TPU_MODELS[model]
    base_tag = info["requires_tag"]
    if min_chips == 1:
        return base_tag
    if is_oss and info["oss_partial_chips"] != None and min_chips <= info["oss_partial_chips"]:
        return "%s:%d" % (base_tag, info["oss_partial_chips"])
    return "%s:%d" % (base_tag, info["max_chips"])

def _ok_to_wait_tag_for_model(model, min_chips):
    """Returns the google3 presubmit `ok-to-wait-for-*` tag needed for `(model, min_chips)`, or None."""
    info = TPU_MODELS[model]
    if info["ok_to_wait_tag"] != None and min_chips >= info["ok_to_wait_min_chips"]:
        return info["ok_to_wait_tag"]
    return None

def _resolve_generation_model(generation, workflow_name, compat_models):
    """Selects the highest-priority model of `generation` in `compat_models`."""
    matching = [
        model
        for model in compat_models
        if TPU_MODEL_TO_GENERATION[model] == generation
    ]
    if not matching:
        fail("No compatible TPU model found for generation %s in %s (%s)." % (
            repr(generation),
            workflow_name,
            compat_models,
        ))
    return _select_preferred_tpu_model(matching)

def _resolve_workflow_models(workflow_config, workflow_name, compat_models):
    """Resolves a single workflow's configuration into a list of compatible TPU models."""
    if workflow_config == ANY_TPU:
        return [_select_preferred_tpu_model(compat_models)]

    if workflow_config == ALL_TPUS:
        return [m for m in ALL_TPU_MODELS if m in compat_models]

    models = []
    for cfg in workflow_config:
        requested = cfg["model"]
        if requested in TPU_MODELS:
            if requested not in compat_models:
                fail("Requested TPU model %s in %s is not in compatible_accelerators (%s)." % (
                    repr(requested),
                    workflow_name,
                    compat_models,
                ))
            resolved_model = requested
        else:
            resolved_model = _resolve_generation_model(
                requested,
                workflow_name,
                compat_models,
            )
        if resolved_model not in models:
            models.append(resolved_model)
    return models

def validate_specification(compatible_accelerators, run_on):
    """Validates `compatible_accelerators` and `run_on`.

    Args:
        compatible_accelerators: Dict created by `tpus()`.
        run_on: Optional dict created by `run_on_config()`, or `None` for defaults.

    Returns:
        A `(compat, effective_run_on)` tuple.
    """
    if compatible_accelerators == None and run_on != None:
        fail("'run_on' cannot be specified without 'compatible_accelerators'.")
    if (
        type(compatible_accelerators) != "dict" or
        "models" not in compatible_accelerators or
        "min_chips" not in compatible_accelerators
    ):
        fail("'compatible_accelerators' must be a dict created by tpus().")

    if run_on == None:
        return compatible_accelerators, _DEFAULT_RUN_ON
    if (
        type(run_on) != "dict" or
        "presubmit" not in run_on or
        "postsubmit" not in run_on or
        "nightly" not in run_on or
        "reason" not in run_on
    ):
        fail("'run_on' must be a dict created by run_on_config().")
    return compatible_accelerators, run_on

def _models_matching(model_or_generation, models):
    """Returns the models in `models` that are `model_or_generation` or belong to that generation."""
    return [
        m
        for m in models
        if m == model_or_generation or TPU_MODEL_TO_GENERATION[m] == model_or_generation
    ]

def oss_availability_error(compat_models, effective_run_on):
    """Checks whether a validated specification has hardware in OSS CI.

    Args:
        compat_models: List of compatible TPU model strings.
        effective_run_on: Validated `run_on` dict.

    Returns:
        An error reason string if OSS hardware is unavailable, or `None`.
    """
    available_models = [m for m in compat_models if has_hardware(m, is_oss = True)]
    if not available_models:
        return "none of the TPU models in compatible_accelerators (%s) have hardware in OSS" % (
            ", ".join(compat_models),
        )

    for workflow_name in ("presubmit", "postsubmit", "nightly"):
        workflow_config = effective_run_on[workflow_name]
        if workflow_config == None or workflow_config == ANY_TPU or workflow_config == ALL_TPUS:
            continue
        for cfg in workflow_config:
            requested = cfg["model"]
            matching_compat = _models_matching(requested, compat_models)
            if matching_compat and not _models_matching(requested, available_models):
                return "none of the compatible TPU models for %s (%s), requested in %s, have hardware in OSS" % (
                    repr(requested),
                    ", ".join(matching_compat),
                    workflow_name,
                )
    return None

# Temporary helper used only while mapping `compatible_accelerators` and `run_on`
# onto legacy `run_on_accelerators` configurations.
def _validate_multi_target_presubmit(
        *,
        name,
        pre_models,
        post_models,
        presubmit_cfg,
        min_chips,
        is_oss,
        skips_google3_presubmit = False):
    """Validates that `(pre_models, post_models)` can be executed by `run_on_accelerators`.

    Returns:
        `(mapped_nopresubmit_needed, extra_tags)` when representable, otherwise calls `fail(...)`.
    """
    if not pre_models:
        return True, []

    for model in pre_models:
        if model not in post_models:
            fail(("%s: presubmit model %s is not in postsubmit (%s); 'run_on_accelerators' " +
                  "runs every generated target in postsubmit.") % (
                name,
                repr(model),
                post_models,
            ))
        if model in ("v6e", "v7x"):
            fail(("%s: running %s in presubmit alongside multiple postsubmit accelerators (%s) " +
                  "is not yet supported ('run_on_accelerators' skips v6e/v7x in presubmit).") % (
                name,
                repr(model),
                post_models,
            ))

    # In `run_on_accelerators`, every v5 target (`v5e` and `v5p`) runs in presubmit
    # unless test-wide `nopresubmit` is set, while `v6e` and `v7x` get `nopresubmit`.
    roa_presubmit_models = [m for m in post_models if m in ("v5e", "v5p")]
    if sorted(pre_models) != sorted(roa_presubmit_models):
        fail(("%s: presubmit models %s with multi-model postsubmit %s cannot be represented by " +
              "'run_on_accelerators', which runs every v5 model in postsubmit (%s) in presubmit.") % (
            name,
            pre_models,
            post_models,
            roa_presubmit_models,
        ))

    if is_oss:
        return False, []

    extra_tags = []
    for model in pre_models:
        wait_tag = _ok_to_wait_tag_for_model(model, min_chips)
        if wait_tag == None:
            continue
        if type(presubmit_cfg) == "list":
            if wait_tag not in extra_tags:
                extra_tags.append(wait_tag)
        elif not skips_google3_presubmit:
            fail(("%s: running %s (min_chips=%d) in google3 presubmit requires explicitly " +
                  "listing it in 'presubmit' via run_on_config() (or setting presubmit = []).") % (
                name,
                model,
                min_chips,
            ))

    return False, extra_tags

def resolve_google3_legacy_config(
        *,
        name,
        compat_models,
        min_chips,
        effective_run_on,
        has_notap,
        has_nopresubmit,
        supports_multi_target):
    """Maps a validated specification onto google3 legacy tags and `run_on_accelerators`.

    Args:
        name: Base name of the test target.
        compat_models: List of compatible TPU model strings.
        min_chips: Minimum TPU chip count required on a single host.
        effective_run_on: Validated `run_on` dict.
        has_notap: Whether `notap` is set on the target.
        has_nopresubmit: Whether break-glass `nopresubmit` is set on the target.
        supports_multi_target: Whether the calling rule supports `run_on_accelerators`.

    Returns:
        A dict of resolved legacy macro attributes; see the implementation for its keys.
    """
    presubmit_cfg = effective_run_on["presubmit"]
    postsubmit_cfg = effective_run_on["postsubmit"]
    nightly_cfg = effective_run_on["nightly"]
    skips_google3_presubmit = has_notap or has_nopresubmit

    pre_models = _resolve_workflow_models(presubmit_cfg, "presubmit", compat_models)

    # Validate explicit nightly entries against compatible_accelerators even though TAP has no
    # nightly workflow, so typos or incompatible models fail immediately in google3.
    if nightly_cfg != None:
        _resolve_workflow_models(nightly_cfg, "nightly", compat_models)

    if postsubmit_cfg == None or postsubmit_cfg == ANY_TPU:
        # Legacy google3 runs the same target in TAP presubmit and postsubmit, so an
        # omitted/`ANY_TPU` postsubmit reuses a single `presubmit` model rather than
        # resolving `ANY_TPU` independently to the top-priority compatible model.
        if len(pre_models) == 1:
            post_models = list(pre_models)
        else:
            post_models = _resolve_workflow_models(ANY_TPU, "postsubmit", compat_models)
    else:
        post_models = _resolve_workflow_models(postsubmit_cfg, "postsubmit", compat_models)

    if not post_models:
        if not has_notap:
            fail(("%s: postsubmit = [] is not supported in google3 without 'notap' " +
                  "(TAP continuous builds run all non-notap targets); use if_oss(...) " +
                  "if only OSS postsubmit should be skipped.") % name)
        if len(pre_models) > 1:
            fail("%s: multiple presubmit models (%s) with postsubmit = [] cannot be represented." % (
                name,
                pre_models,
            ))
        fallback_model = pre_models[0] if pre_models else _select_preferred_tpu_model(compat_models)
        return {
            "nonightly_oss": None,
            "nopresubmit": None,
            "nopresubmit_oss": None,
            "oss_presubmit_tpu_generation": None,
            "run_on_accelerators": None,
            "tags": [_requires_tag_for_model(fallback_model, min_chips, is_oss = False)],
        }

    if len(post_models) == 1:
        if pre_models and pre_models != post_models:
            fail(("%s: google3 cannot run different TPU models in presubmit (%s) and " +
                  "postsubmit (%s) on a single target.") % (
                name,
                pre_models,
                post_models,
            ))
        model = post_models[0]
        tags = [_requires_tag_for_model(model, min_chips, is_oss = False)]
        mapped_nopresubmit = None
        if not pre_models:
            mapped_nopresubmit = effective_run_on["reason"]
        else:
            wait_tag = _ok_to_wait_tag_for_model(model, min_chips)
            if wait_tag != None:
                if type(presubmit_cfg) == "list":
                    tags.append(wait_tag)
                elif not skips_google3_presubmit:
                    fail(("%s: running %s in google3 presubmit requires explicitly listing it in " +
                          "'presubmit' via run_on_config() (or setting presubmit = []).") % (
                        name,
                        model,
                    ))
        return {
            # str | None: Reason string for skipping OSS nightly.
            "nonightly_oss": None,
            # str | None: Reason string for skipping presubmit.
            "nopresubmit": mapped_nopresubmit,
            # str | None: Reason string for skipping OSS presubmit only.
            "nopresubmit_oss": None,
            # dict[str, str] | None: `tpu_gen`-shaped dict for OSS presubmit generation.
            "oss_presubmit_tpu_generation": None,
            # list[str] | None: `requires-*` tags for multi-target expansion.
            "run_on_accelerators": None,
            # list[str]: Hardware and scheduling tags to append to the test.
            "tags": tags,
        }

    if not supports_multi_target:
        fail("%s: multi-accelerator configurations (%s) are not yet supported by this test rule." % (
            name,
            post_models,
        ))

    needs_nopresubmit, extra_tags = _validate_multi_target_presubmit(
        name = name,
        pre_models = pre_models,
        post_models = post_models,
        presubmit_cfg = presubmit_cfg,
        min_chips = min_chips,
        is_oss = False,
        skips_google3_presubmit = skips_google3_presubmit,
    )
    return {
        "nonightly_oss": None,
        "nopresubmit": effective_run_on["reason"] if needs_nopresubmit else None,
        "nopresubmit_oss": None,
        "oss_presubmit_tpu_generation": None,
        "run_on_accelerators": [
            _requires_tag_for_model(m, min_chips, is_oss = False)
            for m in post_models
        ],
        "tags": extra_tags,
    }

def resolve_oss_legacy_config(
        *,
        name,
        compat_models,
        min_chips,
        effective_run_on,
        runs_in_oss,
        supports_multi_target):
    """Maps a validated specification onto OSS legacy tags and `run_on_accelerators`.

    Args:
        name: Base name of the test target.
        compat_models: List of compatible TPU model strings.
        min_chips: Minimum TPU chip count required on a single host.
        effective_run_on: Validated `run_on` dict.
        runs_in_oss: Whether the test executes in OSS CI.
        supports_multi_target: Whether the calling rule supports `run_on_accelerators`.

    Returns:
        A dict of resolved legacy macro attributes; see the implementation for its keys.
    """

    # Validate against compatible_accelerators first so unknown/incompatible models always fail,
    # even when `runs_in_oss` is False.
    for workflow_name in ("presubmit", "postsubmit", "nightly"):
        workflow_cfg = effective_run_on[workflow_name]
        if workflow_cfg != None:
            _resolve_workflow_models(workflow_cfg, workflow_name, compat_models)

    oss_error = oss_availability_error(compat_models, effective_run_on)
    if oss_error:
        if runs_in_oss:
            fail(("%s: %s. Use if_oss(...) to give OSS its own 'compatible_accelerators' or " +
                  "'run_on', or set notest_oss to True if this test should not run in OSS.") % (
                name,
                oss_error,
            ))

        # The test is not executed in OSS (e.g. `notest_oss` or `notap`), so emit the hardware
        # tag of its highest-priority compatible model without `fails-on-tpu-*` tags so
        # `_build_test` and `requires_libtpu` still build cleanly.
        tags = [_requires_tag_for_model(_select_preferred_tpu_model(compat_models), min_chips, is_oss = True)]
        if min_chips > 1:
            tags.append("exclusive")
        return {
            "nonightly_oss": None,
            "nopresubmit": None,
            "nopresubmit_oss": None,
            "oss_presubmit_tpu_generation": None,
            "run_on_accelerators": None,
            "tags": tags,
        }

    oss_compat_models = [m for m in compat_models if has_hardware(m, is_oss = True)]
    presubmit_cfg = effective_run_on["presubmit"]
    postsubmit_cfg = effective_run_on["postsubmit"] if effective_run_on["postsubmit"] != None else ALL_TPUS
    nightly_cfg = effective_run_on["nightly"] if effective_run_on["nightly"] != None else ALL_TPUS

    pre_models = _resolve_workflow_models(presubmit_cfg, "presubmit", oss_compat_models)
    post_models = _resolve_workflow_models(postsubmit_cfg, "postsubmit", oss_compat_models)
    night_models = _resolve_workflow_models(nightly_cfg, "nightly", oss_compat_models)

    if not pre_models and not post_models and not night_models:
        fail("%s: at least one workflow in 'run_on' must resolve to a TPU model in OSS." % name)

    # Explicit multi-model `postsubmit = [...]` translates to `run_on_accelerators`, matching
    # google3 so target names (`<name>_tpu-v5lite`, `<name>_tpu-v6e`, ...) stay consistent.
    if type(effective_run_on["postsubmit"]) == "list" and len(post_models) > 1:
        if not supports_multi_target:
            fail("%s: multi-accelerator configurations (%s) are not yet supported by this test rule." % (
                name,
                post_models,
            ))
        if night_models and sorted(night_models) != sorted(post_models):
            fail(("%s: OSS 'nightly' models (%s) must match explicit multi-model 'postsubmit' " +
                  "(%s) or be [] when delegating to 'run_on_accelerators'.") % (
                name,
                night_models,
                post_models,
            ))
        needs_nopresubmit, _ = _validate_multi_target_presubmit(
            name = name,
            pre_models = pre_models,
            post_models = post_models,
            presubmit_cfg = presubmit_cfg,
            min_chips = min_chips,
            is_oss = True,
        )
        extra_tags = ["exclusive"] if min_chips > 1 else []
        return {
            "nonightly_oss": effective_run_on["reason"] if not night_models else None,
            "nopresubmit": None,
            "nopresubmit_oss": effective_run_on["reason"] if needs_nopresubmit else None,
            "oss_presubmit_tpu_generation": None,
            "run_on_accelerators": [
                _requires_tag_for_model(m, min_chips, is_oss = True)
                for m in post_models
            ],
            "tags": extra_tags,
        }

    # Single-target OSS tag mapping.
    if len(pre_models) > 1:
        fail(("%s: running multiple TPU models in OSS presubmit (%s) is not yet supported " +
              "without per-model RBE pools.") % (
            name,
            pre_models,
        ))
    if pre_models and not post_models:
        fail(("%s: skipping OSS postsubmit while running OSS presubmit (%s) is not yet " +
              "supported by tag-based OSS CI.") % (
            name,
            pre_models,
        ))
    if post_models and night_models and sorted(post_models) != sorted(night_models):
        fail(("%s: different non-empty OSS 'postsubmit' (%s) and 'nightly' (%s) model sets " +
              "cannot be represented on a single target (both share 'fails-on-tpu-*' tags).") % (
            name,
            post_models,
            night_models,
        ))

    # Non-empty `postsubmit` and `nightly` are guaranteed equal above because both
    # OSS workflows share `fails-on-tpu-*` tags on a single target.
    postsubmit_or_nightly_models = post_models if post_models else night_models
    if pre_models and pre_models[0] not in postsubmit_or_nightly_models:
        fail(("%s: OSS presubmit model %s must also be included in OSS postsubmit/nightly (%s) " +
              "on a single target.") % (
            name,
            repr(pre_models[0]),
            postsubmit_or_nightly_models,
        ))

    preferred_post_or_nightly_model = _select_preferred_tpu_model(postsubmit_or_nightly_models)
    primary_model = pre_models[0] if pre_models else preferred_post_or_nightly_model

    tags = [_requires_tag_for_model(primary_model, min_chips, is_oss = True)]
    if min_chips > 1:
        tags.append("exclusive")
    for gen, oss_model in OSS_TPU_MODELS.items():
        if oss_model not in postsubmit_or_nightly_models:
            tags.append("fails-on-tpu-" + gen)

    oss_presubmit_tpu_generation = None
    mapped_nopresubmit = None
    mapped_nopresubmit_oss = None
    if pre_models:
        chosen_gen = TPU_MODEL_TO_GENERATION[pre_models[0]]
        default_gen = TPU_MODEL_TO_GENERATION[preferred_post_or_nightly_model]
        if chosen_gen != default_gen:
            oss_presubmit_tpu_generation = {
                "generation": chosen_gen,
                "reason": effective_run_on["reason"],
            }
    elif post_models:
        mapped_nopresubmit_oss = effective_run_on["reason"]
    else:
        mapped_nopresubmit = effective_run_on["reason"]

    mapped_nonightly_oss = effective_run_on["reason"] if not night_models else None

    return {
        "nonightly_oss": mapped_nonightly_oss,
        "nopresubmit": mapped_nopresubmit,
        "nopresubmit_oss": mapped_nopresubmit_oss,
        "oss_presubmit_tpu_generation": oss_presubmit_tpu_generation,
        "run_on_accelerators": None,
        "tags": tags,
    }
