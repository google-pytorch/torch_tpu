## Description

Follow-up to #6037 (landed as 1e71c2b51) and its duplicate #6102, which bound the per-thread scalar device-buffer cache in `MakeBuffer` by clearing the whole map once it holds 1024 entries.

#6037's description measures the cost of a full clear: a workload whose working set of distinct scalars is larger than the cache (a 1500-value cycle) misses on every lookup, 1.19x per op in `DEFER_NEVER` and 1.45x in `DEFER_AND_FUSE`. The review also noted that in `DEFER_AND_FUSE` a scalar used before and after the clear becomes two graph inputs, which can break deduplication and add a compile.

This keeps the cache in two generations instead. Misses go into a young map; after 1024 misses it becomes the old map and the previous old map is dropped. A hit in the old map moves the entry back to young. So a scalar used within the last 1024 misses always survives, a working set of up to 2048 distinct values keeps hitting after one warm-up pass, and stale values (a learning-rate schedule) are still freed after two generations. The common hit path stays a single hash lookup. The memory bound becomes 2048 scalar buffers per thread (about 2 MiB) instead of 1024.

What does not change: cache hits, and loops whose scalars are new every step (an optimizer with a schedule) run at the same speed. Only the 1024-to-2048 working-set regime and fused regions that straddle an eviction benefit.

One question for reviewers: if the 1024 bound matters more than the larger working set, this can be two generations of 512 (`kMaxCachedScalarBuffers = 512`), which keeps the total at 1024 and still gives the hot entries a second generation.

## Test Plan

TPU v6e-1. Built on 988b22f12 with 1e71c2b51 cherry-picked (`clear`) and this change on top (`twogen`); the standalone build of current main is blocked by the torch-bundled protobuf headers shadowing `@com_google_protobuf` (same error as the local-torch wheel workflow on main; separate issue). 4000 to 6000 ops per scenario, `torch.accelerator.synchronize()` per op in `DEFER_NEVER` and per 50 ops in `DEFER_AND_FUSE`. "Dispatch" is the host-side time of the Python call.

| scenario | mode | clear | twogen |
|---|---|---|---|
| hit, same scalar | DEFER_NEVER | 53.7 us/op, dispatch p50 9.4 us | 55.0 us/op, p50 12.8 us |
| hit, same scalar | DEFER_AND_FUSE | 27.0 us/op, p50 8.7 us | 27.3 us/op, p50 10.4 us |
| cycle of 1500 distinct scalars | DEFER_NEVER | 60.1 us/op, p50 31.0 us | 57.5 us/op, p50 10.6 us |
| cycle of 1500 distinct scalars | DEFER_AND_FUSE | 48.6 us/op, p50 32.3 us | 33.5 us/op, p50 9.4 us |
| cycle of 512 distinct scalars | DEFER_AND_FUSE | 27.3 us/op, p99 39.2 us | 29.1 us/op, p99 14.3 us |
| fused region reusing 0.9 across the eviction | DEFER_AND_FUSE | 417 us/step, +1 compile | 390 us/step, +0 compiles |
| AdamW + cosine LR, small MLP, 1500 steps | DEFER_AND_FUSE | 1.01 ms/step | 1.02 ms/step |

Device memory for a stream of 4000 distinct scalars tops out at two generations: `allocation.all.current` +1952 vs +928.

Tests in `tests/accelerator_api_test.py` (pytest on the TPU, 5 passed):
- `test_distinct_scalars_do_not_accumulate_device_buffers`: bound updated to two generations.
- `test_queued_op_keeps_scalar_buffer_after_cache_eviction`: turns over both generations before checking the evicted scalar.
- new `test_scalar_cycle_longer_than_one_generation_stays_cached` (both eager modes): two warm-up passes over 1500 scalars, then a third pass that creates no scalar buffers.
