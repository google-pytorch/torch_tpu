Branch `loader-import-error-context` tested on a v6e-4 (us-east5-b, 2026-11-01), wheel built from the branch on the 4-chip host:

- `bazel test //tests:_versioned_so_loader_test`: PASSED.

Note for anyone reproducing: running this test file with pytest against an installed wheel fails 16 cases, because it inspects the repo package layout; the bazel target is the valid run.
