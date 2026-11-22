## Description

#4413 reports that `torch.tpu.memory_allocated()` and
`torch.tpu.max_memory_allocated()` do not exist and asks for the supported way
to read peak HBM to be documented. The supported path is `torch.accelerator`
(the `torch.tpu` device APIs that duplicate it are internal on PyTorch 2.11+),
and `docs/api_reference.md` had no memory section.

This adds a **Memory** section after **Core Device Queries** with a
`torch.cuda` -> `torch.accelerator` table and what each call returns on TPU:

- `memory_allocated`, `max_memory_allocated`, `reset_peak_memory_stats`,
  `get_memory_info`, `memory_stats`: implemented.
- `memory_reserved` / `max_memory_reserved`: always 0 (PJRT owns device
  memory); `empty_cache`: no-op; `reset_accumulated_memory_stats`: warns, not
  implemented. These match `tensor_to_buffer.cc`.
- A note that operations are asynchronous in every execution mode, so
  `synchronize()` first if the value must include the work issued so far, and a
  pointer to `hbm_bytes_per_device` and to the XProf Memory Profile.

Only `docs/api_reference.md` changes. The other items in #4413 (log noise,
experimental-API warnings) are not touched.

## Test Plan

Ran the snippet and the claims on a Cloud TPU v6e-1 (wheel built from `main`,
torch 2.11.0+cpu):

- `memory_allocated` / `max_memory_allocated` after `randn(1024, 1024)` and
  `synchronize()`: 8.6 MiB; `get_memory_info()`: `(31.2, 31.2)` GiB;
  `hbm_bytes_per_device`: 32 GiB.
- `memory_reserved()` and `max_memory_reserved()`: 0; `empty_cache()` returns;
  `reset_accumulated_memory_stats()` logs `not implemented for TPU` and
  returns; `memory_stats()` has the CUDA key set.
- Asynchrony: `y = x @ x` then `memory_allocated()` before `synchronize()`
  shows no change in strict eager and in `TORCH_TPU_DEFER_AND_FUSE=1`; after
  `synchronize()` it grows by 4 MiB in both.
- Markdown-only change, no code paths affected.

Assisted-by: Claude Fable 5.1

🤖 Generated with [Claude Code](https://claude.com/claude-code)
