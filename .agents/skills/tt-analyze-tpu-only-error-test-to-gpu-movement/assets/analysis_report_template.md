# Investigation Report: <TEST_NAME>

## 1. Summary

*   **Test Name**: `<TEST_NAME>`
*   **Source Location**: `tpu_errors_test.py` or `errors_test.py`
*   **Diagnostic Case**: `Passing` / `NoExceptionRaised` / `ExceptionMismatch` /
    `Crash`
*   **Outcome**: `Movable` or `Not-Movable`

## 2. Outcome

(State the final outcome and technical rationale from Step 8)

## 3. Test Implementation

(Code snippet of the adapted test method)

## 4. TPU Execution Result

(Command, Bazel exit code, and observed behavior from Step 3)

## 5. GPU Execution Result

(Command, log path `<OUTPUT_DIR>/1-output.txt`, diagnostic case, and error
message or traceback from Step 4 and Step 5)

## 6. CUDA Kernel Location

(Findings from Step 6: `native_functions.yaml` entries, dispatch options, and
CUDA / Composite implementation source file paths and lines)

## 7. Test Setup & Input Analysis

(Findings from Step 7: what the op is supposed to do, and what is wrong about
the test setup/inputs or what is being tested)

## 8. Root Cause & Class-Specific Analysis

(Detailed findings from Step 8 for `Passing`, `NoExceptionRaised`,
`ExceptionMismatch`, or `Crash`, including any incremental input transformation
steps with reasons and errors)
