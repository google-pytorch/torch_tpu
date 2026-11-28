# Migration Final Report: <TEST_NAME>

## 1. Summary

*   **Test Name**: `<TEST_NAME>`
*   **Outcome**: `Movable` or `Not-Movable`
*   **Diagnostic Case**: `ExceptionMismatch` / `Crash`
*   **Overall Status**: `SUCCESS` or `FAILED: <REASON>`
*   **Final Test Placement**: `errors_test.py` (removed from
    `tpu_errors_test.py`) or remains in `tpu_errors_test.py`

## 2. Input Transformation Steps

(List each input adjustment applied in sequence — each with an explanation under
80 characters and a small diff showing the actual code change — or state
`None required (only error message/type adaptation in assert_raises_message).`)

## 3. Verification Commands & Logs

(Record the TPU and GPU verification commands executed in STEP 2 and STEP 3,
along with log paths `<OUTPUT_DIR>/3-output-tpu-<SEQ>.txt` and
`<OUTPUT_DIR>/3-output-gpu-<SEQ>.txt` and observed results.)

## 4. Results

### Final Test Diff

(Unified diff of the final test changes in `errors_test.py` and
`tpu_errors_test.py`.)

### Failure Reason (if FAILED)

(If Overall Status is `FAILED`, state what was expected and what was actually
observed; otherwise `N/A`.)
