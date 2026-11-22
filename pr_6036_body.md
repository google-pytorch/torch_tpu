## Description

Bound the per-thread cache of device buffers that `MakeBuffer` in
`csrc/eager/op_dispatcher.cc` keeps for Python scalars.

In eager modes every distinct `(scalar value, dtype)` is materialised once as a
0-d device buffer and stored in a `thread_local absl::flat_hash_map` that is
never pruned. Workloads that produce a stream of new scalars (learning-rate
schedules, step counters, attribution path coefficients) therefore leak one
HBM allocation per value; `torch.accelerator.empty_cache()` cannot release
them. Reported in #6036 (+19,999 live allocations for 20,000 distinct scalars).

This change adds `kMaxCachedScalarBuffers = 1024` and clears the map when the
bound is reached before inserting. Clearing is safe because entries are
ref-counted `DeviceBufferRef` handles: a caller or an in-flight transfer that
still holds a buffer keeps it alive. Clearing is chosen over LRU because scalar
buffers are cheap to recreate and the map is on the eager dispatch path.

Adds `tests/scalar_buffer_cache_test.py`, which asserts that 5,000 distinct
scalars add fewer than 1,088 live allocations in `DEFER_NEVER` and
`DEFER_AND_FUSE`, and that a repeated scalar adds at most 2.

Fixes #6036.

## Test Plan

Environment: Cloud TPU v6e-1 (us-east5-b), Python 3.14, torch 2.11.0+cpu (the
version pinned in `requirements/requirements_3_14.txt`), wheel built from this
branch with
`bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe`.
Rebuilt and rerun after merging `main` (ba59861, on top of the
`ScheduleHostToDeviceTransfer` change in 7bcf25a); numbers below are from that
build.

- `PYTHONPATH=. python -m pytest tests/scalar_buffer_cache_test.py`
  -> 3 passed in 6.49s (`DEFER_NEVER`, `DEFER_AND_FUSE`, repeated scalar).
- Repro from #6036 on the released 0.2.0 wheel vs. this build
  (`x.add_(i / 1024)` x 5000, `allocation.all.current` delta):
  -> released 0.2.0: **+4999** live allocations (none released by
  `empty_cache()`); this build: **+903** (5000 distinct values, cache cleared
  4 times, 904 resident). Repeated scalar: +2 in both.
- Real workload that triggers the leak: Captum `IntegratedGradients` on a small
  MLP with `n_steps` swept 10..109 (each step count yields new path
  coefficients). 0.2.0: +6361 live allocations; this build: +280.
- Current `main` without this change (wheel built from fa636e8's base): the
  `DEFER_NEVER` case of the new test already passes, `DEFER_AND_FUSE` still
  leaks (+5001 live allocations for 5000 scalars). The bound applies to both
  paths since they share `MakeBuffer`.
- `bazel test //tests:scalar_buffer_cache_test`: not run (no RBE access; the
  wheel build used `--config=no_rbe`).

Assisted-by: Claude Fable 5.1
