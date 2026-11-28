TASK: apply the fix in the analysis report `<OUTPUT_DIR>/2-analysis.md` and
verify it passes on both platforms.

STEP 1: ensure the adapted test is in `tests/errors_test.py` and apply input
fixes.

*   Read the analysis report `<OUTPUT_DIR>/2-analysis.md`.
*   Find `<TEST_NAME>` in `tests/errors_test.py`. If it is not present yet,
    insert the adapted test from `## 3. Test Implementation` of
    `<OUTPUT_DIR>/2-analysis.md` at the bottom of `TpuVsGpuErrorTest` in
    `tests/errors_test.py`.
*   Apply the changes identified in the analysis report, so that execution
    reaches and triggers the desired error check (or, in case of Crash
    diagnostic, to steer out-of-bounds execution into an error check condition).

STEP 2: run the test on TPU and try to make it pass.

*   Run the test on TPU (SEQ: 1). If it fails, inspect the log:

    *   Only if it failed due to a confirmed error-message mismatch, replace the
        `tpu=` kwarg with the observed error message (using `re.compile(...)`
        only for unstable dynamic addresses, file paths, or line numbers per
        `tt-write-error-tests`) and run the test on TPU (SEQ: 2) again.
    *   If it failed due to `NoExceptionRaisedError`, a wrong exception type, or
        a setup/syntax error, do not overwrite `tpu=`; proceed to STEP 5 and
        record `FAILED: <REASON>`.

    ```bash
    bazel test //tests:errors_test_tpu \
        --test_filter="*<TEST_NAME>*" \
        --test_sharding_strategy=disabled \
        --test_output=all > "<OUTPUT_DIR>/3-output-tpu-<SEQ>.txt" 2>&1
    ```

STEP 3: run the test on GPU and try to make it pass.

*   Run the test on GPU (SEQ: 1). If it fails, inspect the log:

    *   Only if it failed due to a confirmed error-message mismatch, replace the
        `gpu=` kwarg with the observed error message (using `re.compile(...)`
        only for unstable dynamic addresses, file paths, or line numbers per
        `tt-write-error-tests`) and run the test on GPU (SEQ: 2) again.
    *   If it failed due to `NoExceptionRaisedError`, a wrong exception type, or
        a setup/syntax error, do not overwrite `gpu=`; proceed to STEP 5 and
        record `FAILED: <REASON>`.

    ```bash
    bazel test --config=cuda //tests:errors_test_gpu \
        --test_filter="*<TEST_NAME>*" \
        --test_sharding_strategy=disabled \
        --test_output=all > "<OUTPUT_DIR>/3-output-gpu-<SEQ>.txt" 2>&1
    ```

STEP 4: read the last output from both TPU and GPU, and confirm they both
passed.

*   If they both passed:
    *   Delete the entire method (including any `@et.why_tpu_only` decorator)
        from `tests/tpu_errors_test.py`.

STEP 5: write report to `<OUTPUT_DIR>/4-final-report.md` using the structure in
[Final Report Template](final_report_template.md):

<FINAL_REPORT_TEMPLATE>
