---
name: tt-analyze-tpu-only-error-test-to-gpu-movement
description: >-
  Adapts TPU error tests to errors_test.py and analyzes them on GPU to evaluate
  whether they can be moved permanently for cross-device parity.
  Use when investigating whether specific TPU error tests would also pass on GPU
  with a few changes.
  Don't use for general error test authoring (use tt-write-error-tests) or
  standard unit testing (use tt-write-tests).
---

# Analyze TPU-Only Error Test Movement to GPU

This skill evaluates whether an error test currently residing in
`tests/tpu_errors_test.py` (or already moved to `tests/errors_test.py`) can be
permanently moved to `tests/errors_test.py` for cross-device TPU vs. GPU parity
validation.

## Input

*   `<TEST_NAME>` (required): Error test method name (e.g.,
    `test_jagged_offsets_not_int64`).
*   `<OUTPUT_DIR>` (optional): Directory where logs and the analysis report are
    written. If omitted, defaults to `/tmp/tpu-only-test-analysis/<TEST_NAME>/`
    (outside the repository tree). Always report the resolved `<OUTPUT_DIR>`
    path (and explicitly note when the default directory was used) to the user.

## Step-by-Step Workflow

### Step 1: Locate Candidate Test & Check Location

Check where `<TEST_NAME>` is currently defined:

1.  **Search Locations**:
    *   **(a) `tpu_errors_test.py`**: Search for `<TEST_NAME>` in
        `tests/tpu_errors_test.py`.
    *   **(b) `errors_test.py`**: Search for `<TEST_NAME>` in
        `tests/errors_test.py`.
2.  **Branch by Location**:
    *   **If on (a) `tpu_errors_test.py`**: Determine the start and end line
        numbers of the entire test method in `tests/tpu_errors_test.py`
        (including any docstrings or decorators like `@et.why_tpu_only(...)`).
        Proceed to **Step 2** to adapt and move/copy the test to
        `tests/errors_test.py`.
    *   **If on (b) `errors_test.py`**: The test is already present in
        `tests/errors_test.py`. Skip **Step 2** and proceed directly to **Step
        3: Verify on TPU**.

### Step 2: Adapt and Copy Test to `errors_test.py`

Copy and adapt the test method to the bottom of the `TpuVsGpuErrorTest` class in
`tests/errors_test.py`:

1.  **Adapt and Append**:
    *   Copy the test from `tests/tpu_errors_test.py`.
    *   Append it to the bottom of the `TpuVsGpuErrorTest` class in
        `tests/errors_test.py`.
    *   Remove `@et.why_tpu_only(...)`: TPU-only decorators are banned in
        `tests/errors_test.py`.
    *   Make device-agnostic: replace hard-coded `"tpu"` strings with
        `device=et.device()`, replace `torch.tpu` references with
        `torch.get_device_module(device)`, and find an analogous
        `torch.cuda.<CUDAFN>` for any `torch.tpu.<TPUFN>`.
    *   Add `gpu="",` kwarg: in the `et.assert_raises_message(...)` call,
        explicitly add `gpu="",` alongside `tpu=...`.
2.  **Verify Edit**: Inspect the bottom of `TpuVsGpuErrorTest` in
    `tests/errors_test.py` to ensure clean indentation and syntax.

### Step 3: Verify on TPU

Confirm the adapted test executes and passes on TPU:

```bash
bazel test //tests:errors_test_tpu \
  --test_filter="*<TEST_NAME>*" \
  --test_sharding_strategy=disabled
```

*   The Bazel command must return exit code 0.
*   If it fails on TPU, inspect the failure, fix any syntax or parameter issues,
    and re-verify before proceeding.

### Step 4: Run on GPU & Capture Logs

Execute the test against the GPU target and capture the complete raw log:

1.  **Create output directory**: Resolve `<OUTPUT_DIR>` (defaulting to
    `/tmp/tpu-only-test-analysis/<TEST_NAME>/` when not specified; never write
    inside the repository tree) and ensure the directory exists.
2.  **Run with Bazel and capture output**:

    ```bash
    bazel test --config=cuda //tests:errors_test_gpu \
      --test_filter="*<TEST_NAME>*" \
      --test_sharding_strategy=disabled \
      --test_output=all > "<OUTPUT_DIR>/1-output.txt" 2>&1
    ```
3.  **Wait asynchronously**: Wait for the background task to complete (do not
    busy-poll).

### Step 5: Classify Diagnostic Case

Inspect `<OUTPUT_DIR>/1-output.txt` and identify which diagnostic case occurred
on GPU:

*   **`Passing`**: GPU raised an error matching `gpu="..."` (or the expected
    error).
*   **`NoExceptionRaised`**: PyTorch on CUDA executed the operation successfully
    without raising any exception (the op ran to completion).
*   **`ExceptionMismatch`**: GPU raised an exception, but with a different
    exception type or error message than TPU.
*   **`Crash`**: GPU crashed during execution (e.g., CUDA assert failure,
    illegal memory access, or SIGSEGV).

### Step 6: CUDA Kernel Location

Locate the PyTorch C++ kernel declaration and dispatch implementations for the
op being tested in the PyTorch source repository:

1.  **Kernel function name and available dispatch options**:
    *   Search for the operator in PyTorch's native function registry:
        `aten/src/ATen/native/native_functions.yaml`.
    *   Identify the kernel function name and the dispatch entries registered
        for it (e.g., `CUDA`, `CompositeExplicitAutograd`,
        `CompositeImplicitAutograd`).
2.  **CUDA dispatch implementation**:
    *   If registered for CUDA, find the kernel implementation file (usually
        under `aten/src/ATen/native/cuda/`).
3.  **Composite dispatch implementation**:
    *   If registered as composite, find the dispatch implementation file
        (usually under `aten/src/ATen/native/`).

### Step 7: What's Wrong About the Test Setup, i.e. Inputs?

Analyze the test method and its inputs to answer two questions:

1.  **What is the op supposed to do?**: Understand the op's intended behavior,
    expected tensor arguments, shapes, and data types.
2.  **What's wrong about the inputs? Or, what are we testing?**: Identify the
    specific invalid condition or invariant the test intends to exercise (e.g.,
    mismatched dimensions, out-of-bounds indices, unsupported dtype, negative
    sizes, or invalid stride).

### Step 8: Class-Specific Analysis & Outcome

Perform root cause analysis specific to the diagnostic case identified in Step
5:

#### Class: `Passing`

*   **Analysis**: No root cause analysis needed because the test already passes
    or raises the expected error on GPU.
*   **Outcome**: `Movable`. Proceed to Step 9.

#### Class: `NoExceptionRaised`

Analysis should answer:

1.  **Is the current test setup supported by CUDA?**
    *   If so: Look for proof that the test setup is supported by CUDA (e.g.,
        `AT_DISPATCH*` macros, execution paths specific to this case, etc.).
    *   **Outcome**: `Not-Movable`. Proceed to Step 9.
2.  **Otherwise, we assume the input is invalid for the given op**:
    *   **Is it an internal op that assumes its inputs are already validated?**
        (e.g., a specialized version of a public op that uses cuBLAS, which is
        expected to be called only inside the kernel of the public op that will
        dispatch to the appropriate kernel).
    *   **What aspect of the output computation is wrong?** (e.g., out-of-bounds
        reads/writes, silent data corruption, input data left incorrectly
        unused, or wrong output values). Collect code location/snippets as
        proof.
    *   **Outcome**: `Not-Movable`. Proceed to Step 9.

#### Class: `ExceptionMismatch`

Analysis should answer:

1.  **Is the GPU error message related to the TPU one?**
    *   Check if both errors enforce the same invariant (e.g., dimension
        mismatch or unsupported type), even if wording or exception class
        differs slightly.
    *   If so:
        *   **Outcome**: `Movable`. Proceed to Step 9.
2.  **Is it even possible to get a related GPU error calling this op?**
    *   If so:
        *   Find the inputs' characteristics (shapes, dtypes, values, etc.) that
            would be necessary so that execution gets to that error check, and
            that error check condition fails (i.e. triggers).
        *   List all characteristics that the current inputs don't satisfy, and
            the corresponding diff necessary to fix that.
        *   If such a list is satisfiable (i.e., not an impossible condition):
            *   **Outcome**: `Movable`. Proceed to Step 9.
    *   **Outcome**: `Not-Movable`. Proceed to Step 9.

#### Class: `Crash`

Analysis should answer:

1.  **Was it an OOB read/write? Where (provide file and line)?**
    *   Locate the exact crash site in the CUDA kernel or ATen C++ source.
2.  **List all other locations the execution could crash due to an OOB read.**
    *   Use them as "taint sources" in a data-dependency graph.
3.  **Do any of those OOB data get to an error check condition?**
    *   I.e. are those error check conditions reachable from the tainted
        sources?
    *   Check whether the out-of-bounds data would be compared against some
        value, which could lead to an unrelated error being raised instead of a
        crash.
    *   If there is at least one location that satisfies this property:
        *   **Outcome**: `Movable`. Proceed to Step 9.
4.  **Outcome**: If none satisfy this property, `Not-Movable`. Proceed to
    Step 9.

### Step 9: Author Report, Clean Up, & Summary of Findings

1.  **Write Report**: Write the findings from the earlier steps to
    `<OUTPUT_DIR>/2-analysis.md` using the structure in
    [Analysis Report Template](assets/analysis_report_template.md).

2.  **Revert Temporary Workspace Edit**: If `<TEST_NAME>` came from
    `tests/tpu_errors_test.py` (Step 1 branch (a)), revert the temporary Step 2
    addition by removing the copied `<TEST_NAME>` test method from
    `tests/errors_test.py` so the repository stays clean and `errors_test_gpu`
    is not broken. If `<TEST_NAME>` was already in `tests/errors_test.py` (Step
    1 branch (b)), leave `tests/errors_test.py` untouched.

#### Summary of Findings to the User

Present a concise final summary to the user:

*   State the test name and source location.
*   State the diagnostic case (`Passing`, `NoExceptionRaised`,
    `ExceptionMismatch`, or `Crash`).
*   State the classified `Outcome` (`Movable` or `Not-Movable`) along with a
    one-sentence core technical justification.
*   Report the resolved `<OUTPUT_DIR>` path (and note if the default directory
    was used) and point the user to `<OUTPUT_DIR>/2-analysis.md` for full
    details.
