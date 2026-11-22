Branch `missing-ops-batch` tested on a v6e-4 (us-east5-b, 2026-11-01), wheel built from the branch on the 4-chip host:

- `bazel test //tests:ops_test_tpu-v6e --test_filter="TestOps.test_logaddexp TestOps.test_nansum TestOps.test_std_mean TestOps.test_trace TestOps.test_addbmm TestOps.test_smooth_l1_loss TestOps.test_huber_loss" --local_test_jobs=1`: PASSED.

Raw logs: https://github.com/joinalahmed/captum-on-torchtpu/tree/main/results/v6e-4/branches
