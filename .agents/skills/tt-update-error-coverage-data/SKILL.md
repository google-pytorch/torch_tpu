---
name: tt-update-error-coverage-data
description: >-
  Updates the error coverage data file for TorchTPU.
  Use when the user wishes to update the TorchTPU error coverage data.
---

# Update TorchTPU Error Coverage Data

This skill guides you through the process of updating the error coverage
baseline for TorchTPU, `tests/error_coverage_data.yaml`, which
`//tests:error_coverage_test` checks.

The test fails if the baseline doesn't match the coverage in a way that could
hide a regression, and its failure message lists what to fix:

-   **Regression**: a file has more uncovered error macros than its baseline
    `remaining` count (or has any while it has no entry). Add error tests
    instead of updating the baseline, unless others caused the regression.
-   **Outdated baseline**: a file has fewer uncovered error macros than its
    `remaining` count, or the baseline has an entry for a file that no longer
    has error macros to cover (e.g. a deleted file). Update the baseline.
-   **Outdated `ERROR_COV_INFEASIBLE` comment**: an error macro marked as
    infeasible to test is covered. Remove the comment, then update the baseline.

Changes in the `total` and `infeasible` counts, and fully covered files missing
from the baseline, are only warnings. The update command still records them.

## Instructions

### 1. Run the update command

Run the command that the test failure message prints to generate the updated
coverage data:

```bash
cp requirements/requirements_3_14_latest.txt \
  requirements/requirements_3_14.txt && \
  bazel test --config=ci_tpu_v5_newest_deps \
  --run_under="$(pwd)/ci/tools/parallel_accelerator_execute.sh" \
  //tests:error_coverage_test --test_arg=--print_coverage_data --test_output=all
```

Then paste the data between `BEGIN_YAML` and `END_YAML` into
`tests/error_coverage_data.yaml`, below the license header.

### 2. Commit the changes

Follow [tt-create-change](../tt-create-change/SKILL.md) to commit the changes to
the YAML file and open a pull request. Use the following summary line:

```
Update the `error_coverage_data.yaml` file.
```

No further description is needed.
