# Draft comment for google-pytorch/torch_tpu#6037 (or a follow-up issue)

Thanks for landing this (1e71c2b51). Two follow-up suggestions from running the same
fix on a v6e-1 / v6e-4 with the #6036 repro and with Captum Integrated Gradients:

**1. Keep the hot entries across the boundary with a two-generation map.**
A full `clear()` has two costs the PR body measures: a workload that cycles
through more than 1024 distinct scalars misses on every lookup (1.19x per op in
DEFER_NEVER, 1.45x in DEFER_AND_FUSE for a 1500-value cycle), and in
DEFER_AND_FUSE a scalar used before and after the clear becomes two graph inputs,
which @benawilson noted can break deduplication and add a compile. Both go away
with two maps instead of one, at no per-lookup cost:

```cpp
static thread_local absl::flat_hash_map<HashableScalar, DeviceBufferRef> young, old;
if (auto it = young.find(k); it != young.end()) return it->second;
if (auto it = old.find(k); it != old.end()) {
  DeviceBufferRef ref = it->second;
  old.erase(it);
  young.insert({k, ref});          // promote: still in use
  return ref;
}
... create buf_ref ...
if (young.size() >= kMaxCachedScalarBuffers) { old = std::move(young); young.clear(); }
young.insert({k, buf_ref});
```

Anything used in the last 1024 misses survives one eviction, so a 1500-value cycle
fits (bound becomes 2048 entries, about 2 MiB), a learning-rate schedule still
evicts its stale values after two generations, and a scalar reused inside one fused
region keeps a single `DeviceBufferRef`. It is the same O(1) hit path and about
ten more lines than the clear; a real LRU is not needed.

**2. Measure the test against a warm baseline instead of a fresh thread.**
`test_distinct_scalars_do_not_accumulate_device_buffers` starts a worker thread
to get an empty cache and then polls `/proc/self/task/<tid>` so the thread's
`thread_local` destructor runs before the next test. Taking the
`allocation.all.current` snapshot after a warm-up op and asserting on the delta
needs neither: the bound shows up as "fewer than ~1.1x kMaxCachedScalarBuffers new
live allocations for N >> 1024 distinct scalars" whatever the cache held before,
and the test stays Linux-independent. That is how the (duplicate) #6102 test did
it; happy to send it as a small PR if useful.
