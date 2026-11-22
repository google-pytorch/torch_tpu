Branch `dynamism-denylist-triage` tested on a v6e-4 (us-east5-b, 2026-11-01), wheel built from the branch on the 4-chip host, filter = the 11 ops removed from the deny list:

- `bazel test //tests:ops_test_dynamic_tpu-v6e --local_test_jobs=1`: PASSED (45.8 s).
- `bazel test //tests:ops_test_dynamic_compiled_tpu-v6e --local_test_jobs=1`: PASSED (28.9 s).

Raw logs: https://github.com/joinalahmed/captum-on-torchtpu/tree/main/results/v6e-4/branches
