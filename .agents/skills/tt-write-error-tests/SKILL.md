---
name: tt-write-error-tests
description: >-
  Guides formatting C++ error messages and authoring, structuring, and placing
  error tests in TorchTPU across errors_test.py, tpu_errors_test.py, and
  distributed/collective_errors_test.py. Use when writing or reviewing
  user-facing or internal C++ error messages, writing tests for C++ error checks
  and other user-facing errors reachable when using TorchTPU, verifying error
  message quality and GPU parity, structuring et.assert_raises_message calls,
  applying @et.gpu_crashes_or_raises or @et.why_tpu_only decorators, or deciding
  between triple-quoted strings and regex. Don't use for functional/behavioral
  op tests (use tt-write-tests or ops_unit_test.py) or internal harness
  utilities.
---

# TorchTPU Error Testing

Guidelines and instructions for formatting C++ error messages and authoring,
structuring, and placing error tests in TorchTPU.

--------------------------------------------------------------------------------

## Error Message Format

*   **End-user error messages**: Apply these rules to all user-facing error
    messages, including when adapting upstream PyTorch messages. Start with a
    lowercase letter and omit trailing punctuation. Do not start with an op or
    function name (e.g., `"op(): ..."` or `"op() ..."`); for op kernels,
    `TT_KERNEL` prepends `"op(): "` automatically. Prefer
    `expected <condition>, got <actual>` (required when using `"expected"` or
    `"got"`), and write `>= 0` instead of `"non-negative"`. Use PyTorch concepts
    rather than C++ internals, and format values and PyTorch dtype names (e.g.,
    `float32`, not `f32`) with `ToString(value)`. Write tensor dimensionality as
    `<N>D` (e.g., `0D`, `2D`) or symbolic `<n>-D`.
*   **Internal error messages**: Error messages for crashing macros (e.g.,
    `ABSL_CHECK`) should be complete sentences with proper capitalization and
    ending punctuation; include a `". This is a TorchTPU bug."` suffix to make
    it clear it's not the user's fault.

--------------------------------------------------------------------------------

## 0. Error Test Criteria

Error test suites (`*errors_test.py`) verify user-facing error message contracts
and assert the exact exception type and message.

Do **not** place tests in `*errors_test.py` if they are:

-   **Functional/behavioral tests**: Verifying core op logic, shape propagation,
    or numerical accuracy.
-   **Incidental errors**: Framework mechanisms where an error happens to be
    raised.
-   **Internal utility tests**: Test harness helpers, private data structures,
    or mocks.

Place these in `tests/ops_unit_test.py` or component unit tests instead.

--------------------------------------------------------------------------------

## 1. Test Placement

Default rule: Always try `errors_test.py` first, then `tpu_errors_test.py`,
unless the error condition is known to be TPU-only.

1.  **`tests/distributed/collective_errors_test.py`**
    -   For errors in distributed communication primitives (`all_reduce`,
        `all_gather`, `reduce_scatter`, `all_to_all`, `ProcessGroupTPU`) that
        require a multi-rank world.
2.  **`tests/errors_test.py`**
    -   For all ops/APIs where GPU also raises an error or crashes.
    -   Ensures error quality and behavioral parity between TPU and GPU.
    -   Tests run in both `--test_mode=tpu` and `--test_mode=gpu` modes.
    -   If GPU crashes rather than raising a clean Python exception, use
        `@et.gpu_crashes_or_raises`.
3.  **`tests/tpu_errors_test.py`**
    -   For errors that occur only on TPU with no GPU counterpart (e.g.,
        unsupported TPU-only data types, TPU-specific compile flags, device
        buffer donation).
    -   Every test method must be decorated with `@et.why_tpu_only("reason")`.
    -   Never use `gpu=...` kwargs here.

--------------------------------------------------------------------------------

## 2. Test Structure

```python
  def test_op_name_error_condition(self):
    # Setup outside context manager
    lhs = torch.tensor([1.0, 2.0], device=et.device())
    rhs = torch.tensor([3.0, 4.0], device=et.device())

    # Assertion
    with et.assert_raises_message(
        RuntimeError,
        tpu="""sub(): the dtype of the first argument cannot be bool""",
        gpu="""Subtraction, the `-` operator, with a bool tensor is not supported.""",
    ):
      # Trigger
      torch.sub(lhs.to(torch.bool), rhs)
```

-   **Setup outside `with`**: Perform all allocations and inputs before
    `with et.assert_raises_message(...)`.
-   **Use `et.assert_raises_message`**: Never use `assertRaisesRegex` (banned).
-   **Kwargs**:
    -   `errors_test.py`: Specify both `tpu="""..."""` and `gpu="""..."""`.
    -   `tpu_errors_test.py`: Specify only `tpu="""..."""`.
-   **Device**: In `errors_test.py`, always use `device=et.device()`. In
    `tpu_errors_test.py`, use `device=et.device()` or `"tpu"`.

--------------------------------------------------------------------------------

## 3. Strings vs. Regular Expressions

-   **Triple-quoted strings (`"""..."""`)**: Default for exact full matching
    without backslash escapes.
-   **Regex (`re.compile(...)`)**: Use only for dynamic message content
    (unstable line numbers, file paths, memory addresses, or async wrapper
    headers).
-   Matches use `fullmatch`. Use `.*` and `re.DOTALL` when matching substrings
    across lines.

--------------------------------------------------------------------------------

## 4. Decorators

-   **`@et.gpu_crashes_or_raises("reason")`**: Only in `errors_test.py` when GPU
    crashes (e.g. CUDA assert, SIGSEGV) or raises an unstructured error. Do not
    use if GPU raises a clean exception.
-   **`@et.why_tpu_only("reason")`**: Mandatory on every test in
    `tpu_errors_test.py`. Incompatible with `@et.gpu_crashes_or_raises`.
-   **`@parameterized.named_parameters(...)`**: For testing multiple dtypes or
    parameter variants.
-   **`@et.skip_if(predicate, "reason")`**: Use instead of `unittest.skipIf` for
    conditions depending on absl flags.
