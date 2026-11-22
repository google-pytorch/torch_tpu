The memory-API part of this issue is addressed on a branch: a **Memory** section in `docs/api_reference.md` with a `torch.cuda` -> `torch.accelerator` table stating what each call returns on TPU (`memory_allocated`, `max_memory_allocated`, `reset_peak_memory_stats`, `get_memory_info`, `memory_stats` implemented; `memory_reserved` always 0 because PJRT owns device memory; `empty_cache` no-op; `reset_accumulated_memory_stats` warns), plus a note that ops are asynchronous in every execution mode so `synchronize()` first, and a pointer to `hbm_bytes_per_device` and the XProf Memory Profile. Values were checked on a v6e-1.

https://github.com/joinalahmed/torch_tpu/tree/docs-memory-apis
(diff: https://github.com/google-pytorch/torch_tpu/compare/main...joinalahmed:torch_tpu:docs-memory-apis)

I cannot open the PR yet because the repository allows users without write access 3 open PRs and #6102, #6105 and #6106 fill that quota; I will open it when a slot frees. The log-noise and experimental-API-warning items in this issue are not touched.
