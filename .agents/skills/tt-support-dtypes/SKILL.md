---
name: tt-support-dtypes
description: >-
  Investigates and fixes unsupported or excluded datatypes (dtypes) for PyTorch
  operations (ops) on TorchTPU. Use when an op fails for a specific datatype,
  when a datatype is excluded in ops_test.py, when implementing support for a
  new datatype on an existing op, or when orchestrating a
  plan-review-implement-verify loop across a batch of ops. Don't use for generic
  PyTorch/XLA issues (use standard xla tools) or for adding new operations from
  scratch.
---

# Torch TPU Dtype Support

Instructions for identifying, reproducing, diagnosing, and fixing unsupported
datatypes on TorchTPU operations.

## 1. Identify Dtype Exclusions

Exclusions are defined in the test file `tests/ops_test.py`. Search for the test
method for your target operation (e.g., `test_addmm`) to see if it has
`exclude_dtypes` or `exclude_inplace_dtypes` defined.

Example from `ops_test.py`:

```python
  def test_addmm(self):
    self.do_test_op(
        "addmm",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
    )
```

> [!IMPORTANT] **GPU Comparison Test Exclusions:** Stress that newly supported
> dtypes shouldn't be excluded from GPU comparison tests. When enabling a dtype,
> ensure you remove it from `exclude_dtypes` and `exclude_inplace_dtypes`.

## 2. Run Reproducer

To see the exact failure, run the specific test case for the op and target
dtype. Use the `--dtypes` test argument to override and run only the target
dtype.

```bash
bazel test -c opt //tests:ops_test_tpu-v5lite \
  --test_filter=TestOps.test_{op_name}$ --test_arg=--dtypes={dtype} \
  --test_output=all --test_sharding_strategy=disabled
```

*   `{op_name}`: The name of the op as defined in `ops_test.py` (e.g. `addmm`,
    `arange`).
*   `{dtype}`: The PyTorch dtype to test (e.g. `float16`, `bfloat16`,
    `complex64`, `int64`, `bool`).

> [!NOTE] If the dtype is currently excluded in `ops_test.py`, you must
> temporarily comment out or modify the exclusion in `ops_test.py` to allow the
> test to run for that dtype.

## 3. Diagnose the Failure

Analyze the test failure traceback to determine the root cause:

*   **Case A: Unimplemented stablehlo conversion / JAX restriction** If the
    error message contains `UNIMPLEMENTED: Converting from type ...` or
    references compilation failures in `stablehlo` lowerings, it means the
    StableHLO builder or compiler doesn't support the operation on that type.
*   **Case B: Explicit check or restriction in TorchTPU C++ code** Look for
    error trace lines pointing to `csrc/ops/` C++ files.
    -   Check if there are helper validation functions (like
        `CheckAlphaTypeSupported`) that throw an error for the target dtype.
    -   Check if the op implementation enforces type constraints using
        `ToElementType` or results in `TT_THROW_IF_ERROR`.
*   **Case C: Type promotion issues** If the error indicates shape or type
    mismatches (e.g., `promoted_scalar_type` is incorrect), check how the op
    determines its output type in C++ (look for `at::result_type` usage).
*   **Case D: Precision Loss / Wrong Values due to casting** If the output dtype
    is integer/boolean (e.g., `uint8`, `int64`, `bool`) but inputs or
    intermediate values are floating point, check if casting inputs directly to
    the output dtype causes precision loss (e.g., step size `1.5` becomes `1`).
    Fix by:
    1.  Determining the promoted `computation_dtype` using `c10::promoteTypes`.
    2.  Keeping the inputs and intermediate computations in the promoted dtype
        (or `run_dtype` if additional promotion with output dtype is needed).
    3.  Building the graph on TPU in `run_dtype`, and then casting the final
        result to `output_dtype` using `mlir::stablehlo::ConvertElementType`.
*   **Case E: Overflow/Underflow in dynamic length/shape calculation** For
    operations with dynamic shape or length calculation (like `arange`), narrow
    integer types (e.g., `uint8`) can overflow or underflow during intermediate
    arithmetic (e.g., `end - start + step` when calculating length).
    Furthermore, if input bounds are cast to the narrow target dtype *before*
    the length calculation, bounds exceeding the maximum value (e.g., `260` for
    `uint8`) will wrap around (e.g., `260` -> `4`), leading to incorrect output
    sizes. Fix by:
    1.  Performing the length/shape calculation using wider types (`int64` for
        integers, `double` or `float32` for floats) directly on the original
        unconverted inputs.
    2.  Only casting the final generated sequence/tensor to the narrow target
        dtype (e.g., via `mlir::stablehlo::ConvertElementType` on the output) at
        the very end of the graph.
    3.  Ensuring dynamic length calculations use signed `int64` to prevent
        unsigned overflow/underflow wrap-around (e.g., when `end < start`).
    4.  **Case F: Identity Mapping for Integer Ops.** If a floating-point
        operation (such as `ceil`, `floor`, `trunc`, `round`) is mathematically
        an identity operation for integer inputs, and StableHLO does not support
        it on integer types, implement the builder in C++ by checking
        `input_type.getElementType().isInteger()` and returning the input
        operand directly, bypassing the StableHLO op call.

> [!IMPORTANT] **TT_KERNEL Gotcha:** In TorchTPU, the `TT_KERNEL` macro enforces
> in debug mode that `GetTensor()` is called on all `PromotedScalar` arguments
> declared in the kernel to ensure promoted tensors are used in the computation
> graph. Do not artificially call `GetTensor()` on unused arguments just to
> bypass the runtime check; bypassing the runtime check hides potential bugs
> where promoted tensors are not properly wired into SHLO building. If an
> argument is truly unused, remove the `PromotedScalar` declaration or handle it
> properly.

## 4. Implement the Fix

1.  Locate the operator implementation in `csrc/ops/` (e.g.,
    `binary_aten_kernels.cc`, `softmax/softmax_aten_kernels.cc`).
2.  If the dtype was blocked by an explicit type check (e.g.
    `CheckAlphaTypeSupported`), relax the check if the underlying StableHLO
    builder supports it.
3.  If type conversion is missing, update the C++ helper methods to correctly
    cast inputs or handle the datatype.
4.  Rebuild and run the test:

    ```bash
    bazel test -c opt //tests:ops_test_tpu-v5lite \
      --test_filter=TestOps.test_{op_name}$ --test_arg=--dtypes={dtype} \
      --test_output=all --test_sharding_strategy=disabled
    ```

## 5. Verify and Clean Up

1.  Make sure the test passes successfully for the target dtype.
2.  Remove the target dtype from `exclude_dtypes` or `exclude_inplace_dtypes` in
    `tests/ops_test.py`. Remember that newly supported dtypes shouldn't be
    excluded from GPU comparison tests.
3.  **Regenerate GPU Golden Files & Tolerances:** If you enabled dtypes that
    were previously excluded, follow
    [tt-update-tolerances](../tt-update-tolerances/SKILL.md) to verify and
    adjust error bounds (if new GPU golden artifacts are needed, maintainers
    regenerate them during review).

    > [!IMPORTANT] **Tolerance Overrides Policy:** Tolerance overrides in
    > `ops_test.py` (`ACCURACY_OVERRIDES_VS_GPU` and
    > `ACCURACY_OVERRIDES_VS_GPU_COMPILED`) should **ONLY** be added or updated
    > when absolutely necessary. Default tolerances must be preferred unless
    > there is a known acceptable floating-point deviation between platforms
    > (TPU vs. GPU/CPU). For exact operations or integer datatypes, tolerance
    > overrides should NOT be used.

4.  **Add Explicit Unit Tests:** Write simple, explicit unit tests for the newly
    supported dtypes in `tests/ops_unit_test.py` to prevent regressions.

    > [!IMPORTANT] **CPU Dtype Support Gotcha:** When writing unit tests using
    > `assert_close_tpu_vs_cpu`, ensure PyTorch CPU implements the op for that
    > dtype. For example, PyTorch CPU's `ceil` and `floor` do not support `bool`
    > tensors and will raise a `RuntimeError` on CPU. You must exclude such
    > dtypes from CPU-comparison tests.

5.  Run the full test suite for the op (without `--dtypes` override) to verify
    no regressions:

    ```bash
    bazel test -c opt //tests:ops_test_tpu-v5lite \
      --test_filter=TestOps.test_{op_name}$ --test_sharding_strategy=disabled
    ```

## 6. Orchestrating a Larger Dtype Effort

For a batch of ops, or when a fix is risky enough to warrant a design review,
drive the work through subagents.

1.  **Plan.** Spawn a `planner` subagent with a high-reasoning model, pointing
    it at this skill. It should locate the implementation and its `ops_test.py`
    entry, run the reproducer (§2), diagnose the root cause (§3), and return a
    plan naming the C++ kernels, tests, and BUILD targets to modify.
2.  **Review.** Spawn three `reviewer` subagents concurrently on the draft plan:

    *   *Static kernel C++ logic*: host-side size and shape math, division by
        zero, precision truncation on inputs, type promotion helpers, and
        `TT_KERNEL` debug-mode assertions.
    *   *Dynamic kernel MLIR logic*: where intermediate casts happen, dynamic
        length calculations overflowing narrow dtypes, signed arithmetic for
        subtraction, and clamp-before-cast to prevent wrap-around.
    *   *Test and verification*: coverage in `ops_unit_test.py`, error tests in
        `errors_test.py`, and golden-file requirements.

    Iterate until all three approve. When a reviewer surfaces a new
    generalizable gotcha, add it to §3 so future agents inherit it.
3.  **Implement** per §4, under TDD: remove the exclusion and confirm the test
    fails first, then fix the kernel, then add the unit tests.
4.  **Verify** per §5. On any failure, return to step 3.
