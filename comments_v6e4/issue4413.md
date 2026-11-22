Branch `docs-memory-apis` checked on a v6e-4 (us-east5-b, 2026-11-01) with the wheel built from the branch, after a 1024x1024 fp32 allocation:

```
memory_allocated 4296704
max_memory_allocated 4296704
memory_reserved 0
get_memory_info (33541746176, 33546042880)
hbm_bytes_per_device 34359738368
```

The values match what the documented semantics describe (per-process, per-device view; `memory_reserved` 0 under the default allocator).
