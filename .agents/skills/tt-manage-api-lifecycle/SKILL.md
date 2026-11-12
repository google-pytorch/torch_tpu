---
name: tt-manage-api-lifecycle
description: >-
  Guides designing, proposing (Design Doc / RFC), implementing, exposing, graduating, and deprecating TorchTPU public and internal APIs.
  Use when adding new public or internal APIs, modifying existing APIs, proposing major architectural changes or design documents, changing API stability stages (from Internal to Experimental to Stable to Deprecated), registering environment variables, updating tt_apis.yaml baselines, or handling API deprecations.
  Don't use for general code reviews (use tt-review-code), formatting pull request descriptions (use tt-create-change), or implementing op kernels that don't change the public or internal API surface (use tt-implement-ops).
---

# TorchTPU API Lifecycle

The API design principles and lifecycle stages are defined in these documents,
which are the single source of truth. Read them before adding or changing an
API:

*   **TorchTPU API Design Guidelines**: `docs/developers/api_design_guide.md`
*   **TorchTPU API Policy**: `docs/developers/api_policy.md`

Many of their rules are enforced by tests, so follow the failure messages rather
than re-checking them by hand:

*   `tests/api:tt_apis_test`: the `tt_apis.yaml` and `public_apis.md` baselines,
    stage annotations, public namespaces, one public path per API, private
    instance attributes, and no "experimental" in API names.
*   `tests/device:device_module_test_tpu`: `torch.tpu` exports match the device
    module.
*   `csrc/common/env_vars.h`: every env var read or written through its helpers
    is registered in `kEnvVarInfo`, and read as its registered type
    (compile-time checks).

This skill covers what isn't enforced automatically: when to write a design doc,
and the steps to implement and register an API.

## 1. Design Docs & Approvals

*   Non-trivial architectural changes or new public APIs need a design doc
    reviewed by the API owners (the domain experts listed in `tests/api/BUILD`)
    before implementation.
*   Any change to `tests/api/tt_apis.yaml` requires approval from the API
    owners.
*   The pull request description must state the user-facing rationale and API
    impact.

## 2. Implementation Workflow

### Step 1: Implement in `_internal/` (or `csrc/`)

Keep the implementation under `torch_tpu._internal/` (e.g.
`src/torch_tpu/_internal/ops/`) or `csrc/`, so that its imports don't leak onto
the public namespace.

### Step 2: Mark the Lifecycle Stage

*   **Python classes / functions**
    (`src/torch_tpu/_internal/utils/annotations.py`):

    ```python
    from torch_tpu._internal.utils import annotations

    @annotations.experimental("New graph capture API.")  # or stable(reason)
    def capture_graph(...):
      ...

    @annotations.deprecated(version="0.2.1", reason="Use ... instead.")
    def get_device_name(...):
      ...
    ```

    If an API is incompatible with `torch.compile`, apply
    `@annotations.forbid_in_torch_compile("...")` as the outermost decorator.

*   **Non-callable attributes / dataclass fields / module attributes**: add them
    to `__tt_api_stages__` with
    `annotations.ApiStageInfo(stage=..., reason=...)` (plus `version=...` for
    deprecated ones).

*   **Environment variables** (`csrc/common/env_vars.h`): define an
    `inline constexpr char kFooEnvVar[] = "TORCH_TPU_FOO";` and register it in
    `kEnvVarInfo`, keeping both lists sorted. Use `SymbolStage::InternalApi()`
    for `TORCH_TPU_INTERNAL_*`, `Experimental()`, `Stable()`, or
    `Deprecated("<version>")`; `InternalImplementation()` is only for env vars
    that TorchTPU writes but never reads. Also give its `EnvVarType` (`kString`,
    `kBoolean`, `kInt` or `kInt64`), which must match how it is read:
    `GetEnvOnce()`, `GetBooleanEnvOnce()` or `GetIntegerEnvOnce()`.

*   **C++ custom ops**: register the kernel as an internal op named `_foo` with
    `ImplInternal<OpName::kFoo>` in the
    `TORCH_LIBRARY_IMPL(tpu, PrivateUse1, m)` block of
    `csrc/eager/tpu_aten_kernels.cc`. Then expose it through a Python wrapper in
    `src/torch_tpu/_internal/ops/ops.py` that calls `torch.ops.tpu._foo` and
    carries the stage decorator (e.g. `ragged_dot`), and export the wrapper in
    `torch.tpu.ops` (Step 3).

### Step 3: Register the Public Path

All `torch.tpu.*` modules are built from `_PUBLIC_MODULES` in
`src/torch_tpu/_internal/public_api.py`.

1.  **Submodules** (`torch.tpu.ops`, `torch.tpu.pallas`, etc.): map the public
    name to its `"torch_tpu._internal.pkg.mod:qualname"` target in the
    submodule's `ModuleSpec.exports` (keep keys sorted; wrap module-level
    properties in `Property(...)`):

    ```python
    "torch.tpu.pallas": ModuleSpec(
        exports={
            "get_pallas_mesh": (
                "torch_tpu._internal.pallas.mesh:get_pallas_mesh"
            ),
        },
    )
    ```

2.  **Root `torch.tpu` attributes** (`torch.tpu.<name>`):

    *   Add `"<name>": _tpu_attr("<name>")` to `_PUBLIC_MODULES["torch.tpu"]`.
        `_tpu_attr` only builds the lookup string
        `"...:TpuDeviceModule.<name>"`; it doesn't define the attribute.
    *   Define the attribute on `_DeviceModule` (or `TpuDeviceModule`) in
        `src/torch_tpu/_internal/device/_device_module.py`:
        *   Functions: a `@classmethod` with its stage decorator, or a class
            attribute assigned an already-decorated `_internal` function.
        *   Classes / values that must warn on access (e.g. `TpuLayout`): don't
            define them on the class; register them only in the module-level
            `__tt_api_stages__` with `annotations.ApiStageInfo(..., value=...)`,
            so `_DeviceModuleMeta.__getattr__` resolves them and emits the
            warning.
        *   Read/write global knobs: a `@property` on `_DeviceModuleMeta`,
            exported with `Property(...)` instead of `_tpu_attr`.

3.  **BUILD, wheel, and registry test**:

    *   If the target lives in a new internal library, add it to `deps` of
        `:public_api` in `src/torch_tpu/_internal/BUILD` (with
        `# build_cleaner: keep`, like the existing entries) and to
        `WHEEL_PY_TARGETS` in `build_files/wheel.bzl`. For root `torch.tpu`
        attributes, add it to the deps of
        `//src/torch_tpu/_internal/device:_device_module` instead.
    *   Add a test case to `tests/api_registry_test.py` for the symbol's
        exposure, identity, and `TT_API_STAGE`.

### Step 4: Update `tt_apis.yaml` & `public_apis.md`

If new APIs or stage transitions were introduced, update
`tests/api/tt_apis.yaml` and `docs/public_apis.md` in the same pull request.

`tt_apis_test` regenerates both files and compares them with the checked-in
ones. It runs only in the "Presubmit on TPU v5 (newest deps)" GitHub Actions job
(on TPU v5e with `requirements/requirements_3_14_latest.txt`), because the API
surface depends on the PyTorch version and hardware.

**Do not run `scan_tt_apis` on a host without a TPU or without the newest
dependencies.** The result won't match the baseline: without a TPU, the
`torch.accelerator.*` probes and device-dependent values get dropped; and a
PyTorch version other than the newest supported release changes the API surface.

**Let CI tell you**: push your pull request; when it adds or changes APIs, the
first newest-deps presubmit run is expected to fail, and the `tt_apis_test`
failure lists the exact `Added APIs` / `Removed APIs` / `Modified APIs` plus
update instructions. Then do one of:

*   Apply the listed changes by hand.
*   Run `bazel run //tests/api:scan_tt_apis` on a TPU VM with the newest
    dependencies (`cp requirements/requirements_3_14_latest.txt
    requirements/requirements_3_14.txt`).
*   Download `tt_apis.yaml` and `public_apis.md` from `//tests/api:tt_apis_test`
    → `test.outputs/` under **Invocation Artifacts** in the Build/Test Explorer
    link (`Streaming build results to: ...`) in the failed job's log. The GitHub
    `bazel-testlogs-*` artifact mentioned in the failure message is currently
    not uploaded.

If the newest-deps presubmit passes, the checked-in files already match the
generated ones and nothing else needs to change.

## 3. Pre-Pull-Request Checklist

*   [ ] **PyTorch-native first**: no existing `torch`, `torch.accelerator`, or
    `torch.distributed` mechanism already covers this use case.
*   [ ] **Knob scope**: global knobs are `torch.tpu.*` properties (with an
    optional `TORCH_TPU_*` env var); local knobs are context managers; no Python
    `flags.DEFINE_*`.
*   [ ] **Registration**: `src/torch_tpu/_internal/public_api.py` and
    `tests/api_registry_test.py` updated, plus the BUILD deps and `wheel.bzl` if
    a new library is needed.
*   [ ] **Baselines**: `tests/api/tt_apis.yaml` and `docs/public_apis.md`
    updated (never from a `scan_tt_apis` run without a TPU or the newest
    dependencies).
*   [ ] **Tests**: unit tests cover the API behavior and its stage warnings.
