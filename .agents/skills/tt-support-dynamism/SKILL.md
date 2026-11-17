---
name: tt-support-dynamism
description: >-
  Investigates, implements, and fixes bounded dynamic shape support for PyTorch
  operations (ops) on TorchTPU. Use when implementing a new op (together with
  tt-implement-ops, which covers op scaffolding) or writing or debugging a
  StableHLO op lowering that must handle bounded shapes, fixing failing ops in
  DYNAMISM_DENY_LISTS (COMPILE, COMPILE_GRAD, or EAGER for compile-skipped ops),
  extending the torch.compile dynamic FX passes, adding reusable dynamism
  utilities to op_builder_utils, or orchestrating dynamism fixes across a batch
  of ops. Don't use for static-shape dtype exclusions (use tt-support-dtypes).
---

# TorchTPU Bounded Dynamism Support

Playbook for supporting bounded dynamism in TorchTPU op lowerings and
`torch.compile` FX passes, and for burning down `DYNAMISM_DENY_LISTS` in
`tests/dynamism_test_utils.py`. All paths are relative to the repository root.

## Core Principles

*   **Keep op files free of ad-hoc dynamism**: Ops under `csrc/ops/` must not
    contain `GetDimensionSize` / `SetDimensionSize`, bounds plumbing, or manual
    pad/slice sequences. Call an existing utility from `op_builder_utils.h` (see
    [op_builder_utils.md](references/op_builder_utils.md)); if none fits, add a
    general one with tests in `csrc/ops/op_builder_utils_test.cc`.
*   **Fix where the root cause lives**: StableHLO lowering failures (§3 Cases
    A–C) are fixed via `op_builder_utils` utilities; `torch.compile` graph
    failures (Cases D–E) in the FX passes under
    `src/torch_tpu/_internal/compile/dynamic/`.
*   **Target `COMPILE` and `COMPILE_GRAD`, not `EAGER`**: Eager dynamism is not
    a goal (e.g., view ops cannot support it). *Exception*: for ops skipped in
    `COMPILE` mode (`compiled_deny_list` in `_compiled_supports_op` in
    `tests/op_testing.py`, or `exclude_dtypes` / `skip_if` in
    `tests/ops_test.py`), reproduce C++ lowering fixes in `EAGER` mode, or
    enable `COMPILE` mode first if feasible.
*   **Never query static sizes on dynamic MLIR types or symbolic ATen tensors**:
    *   MLIR: `RankedTensorType::getDimSize(i)` and `getShape()` return
        `ShapedType::kDynamic` (`INT64_MIN`, `-9223372036854775808`) on bounded
        dims, and `getNumElements()` asserts `hasStaticShape()` (garbage in opt
        builds). Use `GetDimensions`, `GetStaticShape`, or `GetNumElements`.
    *   ATen C++ kernels: `sizes()` / `strides()` guard on and silently
        specialize symbolic `FakeTensor` dims during tracing (recompiling for
        every new shape), and `numel()` / `storage_offset()` throw. Accept
        `ConstTensor` (`csrc/common/const_tensor.h`) and use its `dim()`,
        `sym_sizes()`, `sym_size()`, and `sym_numel()` (they return
        `absl::StatusOr`). Comparing the returned `SymInt`s with `==` / `<` also
        guards. `ops_test_symbolic_shapes` detects these specializations.

## Implementing a New Op

Use this skill alongside tt-implement-ops: tt-implement-ops covers scaffolding
the op, and new lowerings must support bounded shapes from the start:

1.  Follow the Core Principles and build on
    [op_builder_utils.md](references/op_builder_utils.md); use §3 Cases A–C as a
    checklist of patterns to avoid.
2.  An OpInfo-based `test_<op>` in `tests/ops_test.py` is automatically run by
    the dynamism targets in §5. Run them for the new op (skip §1).
3.  If a mode fails and can't be fixed in the same change, add a diagnosed
    `DYNAMISM_DENY_LISTS` entry (with a GitHub issue where possible) per §5's
    filtering policy instead of disabling dynamism in `ops_test.py`.

## 1. Identify the Deny-List Entry

Find the op in `DYNAMISM_DENY_LISTS` and note its failing modes and triage
reason. In `tests/ops_test.py`, find `test_<op>` and check whether `COMPILE` is
filtered there or in `_compiled_supports_op`.

## 2. Run the Reproducer (TDD)

Temporarily remove the target mode(s) for the op from `DYNAMISM_DENY_LISTS`,
then run the scoped target:

```bash
# Use ops_test_dynamic_compiled_grad_tpu-v5lite for COMPILE_GRAD, and
# ops_test_dynamic_tpu-v5lite for EAGER (only for compile-skipped ops).
bazel test -c opt //tests:ops_test_dynamic_compiled_tpu-v5lite \
  --test_filter=TestOps.test_{op_name} --test_output=all --test_sharding_strategy=disabled
```

*   `--test_filter` is a substring match (`$` anchors match nothing), so
    `test_sin` also runs `test_sinh`.
*   Narrow with `--test_arg=--dtypes=float32` and
    `--test_arg=--dynamism_plan_filter=t0d0_t0d1`.
*   Dump StableHLO/HLO modules to a directory with
    `--test_env=XLA_FLAGS="--xla_dump_to=/absolute/dump/dir"` to inspect
    `#stablehlo.bounds`, `get_dimension_size` / `set_dimension_size`, and where
    bounds are dropped.

## 3. Diagnose Common Failure Modes

*   **Case A: Static shape query** (`INT64_MIN` / `hasStaticShape` assertion).
    The builder calls `getDimSize`, `getShape`, `getNumElements`, static
    `MakeConstant`, or static `IotaOp` on a bounded operand. Replace with
    `MakeConstantLike`, `Iota`, `GetNumElements`, `GetStaticShape`, or a new
    `op_builder_utils` helper. Don't paper over it with
    `TT_RET_CHECK(type.hasStaticShape(), ...)` unless bounded dynamism is
    genuinely impossible for the op.
*   **Case B: Multi-dim `broadcast_in_dim` / `reshape` failure in `COMPILE`**.
    `COMPILE` marks several dims dynamic at once (e.g., `t0d0_t0d1`), and raw
    `stablehlo.broadcast_in_dim` / `stablehlo.reshape` reject multiple bounded
    dims or cross-bound reassociations. Route through `ApplyBroadcastIfNeeded` /
    `BroadcastIfNeeded` and `Unsqueeze` / `Squeeze` / `Flatten` /
    `ReshapeToStaticDimensions`, or extend those utilities.
*   **Case C: Missing bounded support or dropped `#stablehlo.bounds`** (garbage
    values, buffer/bound mismatch, MLIR bounds-rank assertion). A builder needs
    a static result type, type inference drops `TypeExtensionsAttr`, or XLA
    rejects bounded operands. Propagate bounds or restore runtime sizes in a
    shared `op_builder_utils` helper (see "Adding a New Utility" in
    [op_builder_utils.md](references/op_builder_utils.md)).
*   **Case D: Unhandled `SymInt` / `SymInt[]` args** (`DetectSymIntUsagesPass`
    raises `NotImplementedError`, or `local_scalar_dense` during tracing).
    `DynamicCompiler` hands static upper-bound tensors to `StaticCompiler`, so
    no `SymInt` may survive the FX passes. Fix in
    `src/torch_tpu/_internal/compile/dynamic/`:
    *   *`SymInt` as a data value* (not a size/dim/offset):
        `HandleSymIntUsagesPass` (`graph_transformations.py`) converts top-level
        positional `SymInt` args to runtime tensors via
        `transform_context.ensure_tensor`. Extend it for kwargs or nested lists,
        and switch `.Scalar` targets to their `.Tensor` overload (verify the
        overload exists and the result dtype is unchanged) so tracing doesn't
        call `.item()` (e.g., `aten.div.Scalar` in `mean` backward).
    *   *Shape-determining `SymInt` / `SymInt[]`* (sizes, dims, offsets):
        inspect the traced FX graph first, since the op may already be
        decomposed into ops an existing pass handles. Prefer extending an
        existing pass's `_op_handlers` map over a new pass. Using
        `transform_context.get_upper_bound(...)`, choose the simplest correct
        rewrite:
        1.  **Compute at the upper bound, then
            `torch.ops.tpu._set_dimension_logical_size`**: only if the in-bounds
            elements of the upper-bound result equal the runtime result.
        2.  **Compose existing TPU custom ops** (the `_dynamic_*` family,
            `_set_dimension_logical_size`, registered in
            `csrc/eager/tpu_aten_kernels.cc`) with ATen ops on runtime size
            tensors (e.g., computing offsets).
        3.  **Add a new `_dynamic_*` custom op**: last resort; follow
            [new_dynamic_custom_op.md](references/new_dynamic_custom_op.md).
    *   Ops whose output sizes depend on tensor *values* are not bounded
        dynamism; filter them with a diagnosed reason.
*   **Case E: Dynamic view limitations**. Non-contiguous view placeholders
    (`InputDynamicViewPass` in `dynamic_view_pass.py`, `view_decomposition.py`)
    or in-graph view ops (`view_ops_passes.py`) hit an unsupported pattern
    (including views in `COMPILE_GRAD` backward graphs). Extend those passes to
    use runtime shape/slice tensors and static upper bounds, with the same order
    of preference as Case D.

## 4. Implement Fix and Add Focused Tests

1.  **Implement where the root cause lives**: for Cases A–C, call an existing
    `op_builder_utils` utility from the op; only if none fits, add or extend a
    utility (with its test) first. For Cases D–E, extend the relevant pass. Pass
    rewrites that reuse existing ops need no `op_builder_utils` change unless
    the new graph hits a lowering failure; a new `_dynamic_*` op needs its C++
    lowering as for Cases A–C.
2.  **Add Python tests only for net-new coverage**: `ops_test_dynamic_compiled*`
    already test `OpInfo` samples across valid marking plans. Add a dedicated
    test only for scenarios it misses (e.g., symbolic `x.shape[i]` arguments or
    specific multi-dim combinations).

    *   **`COMPILE`**: add to `tests/dynamism/compile_test.py`, or group related
        ops in a category file under `tests/dynamism/` (like
        `generative_ops_test.py` or `dynamic_view_test.py`). For new
        `tests/dynamism/BUILD` targets, follow
        [tt-write-tests](../tt-write-tests/SKILL.md).
    *   Keep setup minimal and compute `expected` on CPU, so TPU eager bugs
        don't mask `torch.compile` bugs:

        ```python
        def test_example_multi_dynamic_dims(self):
          class Model(torch.nn.Module):

            def forward(self, x):
              return torch.ones_like(x)

          model = Model()
          compiled = torch.compile(model, backend="tpu")
          x_cpu = torch.zeros(4, 6)
          expected = model(x_cpu)
          x_tpu = x_cpu.to(self.device)
          torch._dynamo.mark_dynamic(x_tpu, 0, min=2, max=16)
          torch._dynamo.mark_dynamic(x_tpu, 1, min=2, max=16)
          utils.assert_close(compiled(x_tpu), expected)
        ```

    *   **`EAGER`** (compile-skipped ops only): add a minimal test in
        `tests/dynamism_test.py` if `ops_test_dynamic` is not enough.

## 5. Verify and Update `DYNAMISM_DENY_LISTS`

1.  Before removing deny-list entries or landing a new lowering, run every
    applicable target (use `--test_filter=TestOps.test_{op_name}` for the
    `ops_test_*` targets):
    *   `//csrc/ops:op_builder_utils_test` if `op_builder_utils` changed.
    *   `//tests:ops_test_dynamic_compiled_tpu-v5lite` and
        `//tests:ops_test_dynamic_compiled_grad_tpu-v5lite`. These are
        `nopresubmit` (presubmit CI doesn't run them), so run them manually; a
        green presubmit does not cover them.
    *   Any `//tests/dynamism:<target>` added or changed in §4.
    *   `//tests:ops_test_symbolic_shapes_tpu-v5lite`; also remove the op from
        `SYMBOLIC_SHAPES_KNOWN_FAILURES` in `ops_test.py` if fixed.
    *   `//tests:ops_test_dynamic_tpu-v5lite`, which runs in presubmit. If a C++
        fix makes `EAGER` pass too, remove the op from `DYNAMISM_DENY_LISTS`
        entirely; if the fix is `torch.compile`-only (e.g., view ops), narrow
        its modes to `{EAGER}`.
    *   `//tests:op_testing_test` to validate deny-list integrity.
2.  **Filtering policy** when an op or mode must stay filtered:
    *   `DYNAMISM_DENY_LISTS` maps a diagnosed root cause (with a GitHub issue
        where possible) to `{op_name: modes}`, where `modes` is a subset of
        `{EAGER, COMPILE, COMPILE_GRAD}` or `ALL_MODES`. Use `untriaged: ` only
        when the root cause is unknown.
    *   `DYNAMISM_NOT_APPLICABLE_OPS` is for ops where bounded dynamism is out
        of scope by design, with explicit justification.
    *   Never disable dynamism inside `ops_test.py`.

## 6. Orchestrating Batch Fixes

For a batch of `DYNAMISM_DENY_LISTS` ops, group ops by shared root cause (an
`op_builder_utils` helper or an FX pass), review that no dynamism logic leaked
into individual op kernels, and verify each fix with §2–§5.
