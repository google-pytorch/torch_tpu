# TorchTPU capabilities relevant to explainability/attribution libraries (as of 2026-10-30)

Note: The docs say nothing explicit about module hooks, torch.autograd.grad, higher-order grads, torch.func, or custom autograd.Function. All "works/doesn't" conclusions on those are inferences or gaps. Research limited to fetched pages; GitHub issue search API was rate-limited, so issue searches used github.com web search pages (titles only).

## Is autograd (backward, torch.autograd.grad, higher-order) supported? Module forward/backward hooks? Custom autograd Functions? torch.func?

### Takeaway
Backward is officially supported at the ATen-op level (~450 ops, forward and backward) and a device-generic `test_autograd` suite was enabled on TPU on Oct 27, 2026 (90 tests pass, with a list of exclusions). No official doc mentions hooks, `torch.autograd.grad`, double backward, custom `autograd.Function`, or `torch.func`; their status is unverified.

### Cited Findings
- Blog: "Added support for ~450 Aten ops (skipping ops with structural or composite delegates), including forward and backward pass for all Core ATen ops required for the major deep learning architectures" — [Introducing TorchTPU](https://pytorch.org/blog/introducing-torchtpu/)
- Intro page: "Supports 450 core ATen operators (both forward and backward passes)"; no hooks content; no explicit list of unsupported ops — [Intro to TorchTPU](https://torchtpu.readthedocs.io/en/latest/intro_to_torchtpu/)
- Google Cloud page: "ATen operators in forward and backward passes, continuously tested against popular Hugging Face models" — [PyTorch on TPU](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu)
- Feature set page lists "FW + BW" under Supported (Validated) basics; lists nothing on hooks, torch.func, custom autograd Function, in-place, views — [Feature set](https://torchtpu.readthedocs.io/en/latest/feature_set/)
- PR #5887 "Enable `test_autograd` device-generic tests with TorchTPU", merged Oct 27 2026; 90 tests pass. Excluded: `index_reduce` lowering crash, nested-tensor NotImplementedError escaping as std::terminate, multi-device tests, sparse COO inference-tensor mismatch, `resize_` bumping version counter twice, device memory stats after failed backward, missing `triu_indices`, `median`, `_to_sparse`, complex128 `fill_`, float64 range, CUDA-specific DataParallel. The PR page does not say whether hook/double-backward/functorch tests are included (list is in "Files changed", not seen) — [PR 5887](https://github.com/google-pytorch/torch_tpu/pull/5887)
- Open issue #4409: backward of a `torch.compile(backend="tpu")` function whose inputs are views (e.g. `x[0]`, `x[1]`, `x[2]` of a requires_grad tensor, or row slices) fails with `RuntimeProgramInputMismatch` in `as_strided`; Python only shows "a required argument is in a failed state". Reproduced on v6e-8 and v5e, torch_tpu 0.2.0. Workarounds: pass base tensor and index inside, clone view inputs, or compile whole fwd+bwd step. Open, awaiting assignee, community fix in progress — [Issue 4409](https://github.com/google-pytorch/torch_tpu/issues/4409)
- Autograd-related PRs: #5953 "Record autograd for avg_pool1d" (closed Oct 29), #6050 "Enable gradient checks for avg_pool1d in ops_test" (open) — [GitHub issue/PR search](https://github.com/google-pytorch/torch_tpu/issues?q=autograd+OR+hook+OR+captum+OR+torch.func+OR+colab) (indicates per-op autograd registration gaps still being fixed)
- Blog notes FSDP2 injects collectives "through module hooks, which break torch.compile into per-op dispatches" (hooks work in eager for FSDP2; hooks + compile are a performance problem, SimpleFSDP preferred) — [Introducing TorchTPU](https://pytorch.org/blog/introducing-torchtpu/)
- Also: FSDP2 (`fully_shard`) issue #5830 shows module-hook-based FSDP2 runs on TPU (closed Oct 30) — [GitHub search](https://github.com/google-pytorch/torch_tpu/issues?q=is%3Aissue+colab+OR+hooks+OR+%22register_hook%22+OR+%22torch.func%22+OR+vmap+OR+%22custom+Function%22)
- Closed PR #4682 "Count Gemma4 checkpoint recomputes with forward pre-hooks" suggests forward pre-hooks used in TorchTPU examples/tests — [PR 4682](https://github.com/google-pytorch/torch_tpu/pull/4682) (title only)
- Issue #5667 (open): BatchNorm never updates running_mean/running_var in training mode (eager and compile) — relevant if attribution runs models in train mode — [GitHub search](https://github.com/google-pytorch/torch_tpu/issues?q=is%3Aissue+colab+OR+hooks+OR+%22register_hook%22+OR+%22torch.func%22+OR+vmap+OR+%22custom+Function%22)
- Open op-gap issues: #6070 `torch.linalg.cross`/`aten::linalg_cross.out` not implemented; #5725 `torch.gcd` unsupported; #5826 `aten::_int_mm` — [GitHub search](https://github.com/google-pytorch/torch_tpu/issues?q=autograd+OR+hook+OR+captum+OR+torch.func+OR+colab)
- Public API doc: any API not listed is "undefined behavior and unsupported"; doc has no mention of autograd/hooks/torch.func — [public_apis.md](https://github.com/google-pytorch/torch_tpu/blob/main/docs/public_apis.md)
- Blog mentions `torch._higher_order_ops.scan` used under compile (a higher-order op, but not autograd higher-order grads) — [Introducing TorchTPU](https://pytorch.org/blog/introducing-torchtpu/)

### Inferences
- Because eager mode dispatches at the ATen level and the standard autograd engine is PyTorch's, Python-level `register_forward_hook`/`register_full_backward_hook`/`Tensor.register_hook` and custom `autograd.Function` (Gradient x Input, Integrated Gradients, Grad-CAM, DeepLIFT in Captum) very likely work in eager mode, but no official source confirms it. Needs a smoke test.
- `torch.autograd.grad` with `create_graph=True` (double backward, needed by e.g. smoothgrad variants, Hessian-based/influence methods) depends on every op's backward being itself differentiable; unverified. `torch.func`/`vmap`/`functorch` is entirely undocumented; likely weak.
- Hooks inside `torch.compile(backend="tpu")` are untested in docs; the blog treats module hooks as compile-breaking for FSDP2.
- Op gaps (e.g. cross, gcd, index_reduce, median, triu_indices) show coverage is not complete; attribution methods using unusual ops might fail or need CPU fallback (docs do not describe CPU fallback).

### Gaps
- No doc on hooks, `torch.autograd.grad`, `create_graph`, `retain_graph`, `torch.func`, custom `autograd.Function`, or CPU fallback for unsupported ops.
- Which test_autograd tests are included/skipped in PR 5887 (file diff not read).
- No complete list of unsupported ops.

## In-place ops, views, dynamic shapes, memory

### Takeaway
In-place ops and views have PyTorch-consistent semantics via functionalization and view decomposition/inversion, but views are costly and overlapping/broadcast writes error. Dynamic shapes: only `dynamic=None`/`mark_dynamic` with bounded buckets works under compile; data-dependent shapes unsupported.

### Cited Findings
- StableHLO is functional SSA; TorchTPU automatically functionalizes underscore ops and `out=`, swaps new buffer into `UntypedStorage` metadata; partial writes via view inversion using `stablehlo.dynamic_update_slice`; works only for non-overlapping views; writing to a broadcast result "always fails explicitly" — [Memory model](https://torchtpu.readthedocs.io/en/latest/torchtpu_memory_model/)
- Views use "view decomposition"; logically match CUDA, not free in performance; prefer contiguous. Clone is copy-on-write. Out-of-place updates can double peak memory; small write to large buffer may copy whole base storage — [Memory model](https://torchtpu.readthedocs.io/en/latest/torchtpu_memory_model/)
- Compiled executables require contiguous, zero-offset inputs; non-trivial views trigger temporary copy from eager — [Execution model](https://torchtpu.readthedocs.io/en/latest/torchtpu_execution_model/)
- Data-dependent shapes (e.g. `torch.masked_select`) force blocking sync; `if my_tensor.all()` causes sync, and graph break under compile — [Execution model](https://torchtpu.readthedocs.io/en/latest/torchtpu_execution_model/)
- Dynamic shapes: "Bounded Dynamism"; `torch.compile(model, dynamic=None)` default in that page, `torch._dynamo.mark_dynamic(tensor, dim, min=..., max=...)`; `bucket_size` default 128, `num_dynamic_buckets` default 1, max 8; unsupported: `dynamic=True`, `torch.nonzero`/data-dependent shapes, Pallas/`jax_op` custom kernels, limited view inputs, not all ops; bounded-dynamic executables not persisted in AOTAutogradCache — [Dynamic shapes](https://torchtpu.readthedocs.io/en/latest/dynamic_shapes/)
- Conflict: Intro page says `torch.compile` defaults to `dynamic=False` and `dynamic=None`/`True` are "not yet supported for production"; the dynamic shapes page presents `dynamic=None` as the default mode — [Intro](https://torchtpu.readthedocs.io/en/latest/intro_to_torchtpu/) vs [Dynamic shapes](https://torchtpu.readthedocs.io/en/latest/dynamic_shapes/)
- Feature set: "Dynamic ops (recompilation)" validated; "SymInts", "Jagged tensors" experimental; "Dynamic ops (no recompilation)", "Streams", "Graph capture" listed as future — [Feature set](https://torchtpu.readthedocs.io/en/latest/feature_set/)
- Other limits: no native FlexAttention; matmul accumulation below fp32 (tf32) unsupported; only one process can access a TPU device — [Intro](https://torchtpu.readthedocs.io/en/latest/intro_to_torchtpu/)

### Inferences
- Attribution loops over varying batch sizes/perturbation counts (SHAP, LIME, occlusion) will recompile per new shape unless buckets/padding are used; eager mode avoids whole-graph recompiles but compiles per-op.
- Slicing/view inputs plus backward under compile (issue 4409) is a direct risk for per-sample attribution done by slicing a batch.

### Gaps
- No doc on autograd + views interaction (saved tensors, version counters beyond the test exclusion list).

## torch.compile(backend=...) and compile cache env vars

### Takeaway
Confirmed: `torch.compile(model, backend="tpu")`. Cache tiers: tier 1 in-memory (default), tier 2 `TORCH_TPU_TIER2_COMPILATION_CACHE`, tier 3 `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`.

### Cited Findings
- Blog code: `compiled_model = torch.compile(model, backend="tpu")`; Dynamo passes `torch.fx.GraphModule` + example inputs; compiles a single StableHLO program via XLA; doesn't use TorchInductor — [Introducing TorchTPU](https://pytorch.org/blog/introducing-torchtpu/); [How TorchTPU works](https://torchtpu.readthedocs.io/en/latest/how_torchtpu_works/); [Google Cloud](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu)
- `TORCH_TPU_TIER2_COMPILATION_CACHE`: a directory name (not path), placed under `/dev/shm/torch_tpu_cache/<name>`, host-local, `"disabled"` turns off, no default stated. `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`: `gs://` URI or filesystem path, shared across hosts/runs. Both must be set before `import torch` — [API reference](https://torchtpu.readthedocs.io/en/latest/api_reference/)
- Tier 2 disabled for single-device programs; Tier 3 off by default until storage location specified — [How TorchTPU works](https://torchtpu.readthedocs.io/en/latest/how_torchtpu_works/)
- Blog: Tier 3 "useful when running TPU programs in execution environments like Colab and data centers" — [Introducing TorchTPU](https://pytorch.org/blog/introducing-torchtpu/)
- Eager modes: strict (default); `TORCH_TPU_LAUNCH_BLOCKING=1` (debug, sync after each op); `TORCH_TPU_DEFER_AND_FUSE=1`; API `torch.tpu.EagerMode` (`DEFER_AND_FUSE`, `DEFER_NEVER`, `DEFER_NEVER_AND_LAUNCH_BLOCKING`), `torch.tpu.set_eager_mode(mode)` — [API reference](https://torchtpu.readthedocs.io/en/latest/api_reference/); [Execution model](https://torchtpu.readthedocs.io/en/latest/torchtpu_execution_model/)
- Other env vars: `TPU_VISIBLE_DEVICES`, `TORCH_DEVICE_BACKEND_AUTOLOAD`, `LIBTPU_INIT_ARGS`, `XLA_FLAGS`, `TORCH_TPU_ENABLE_DEBUG_CHECKS=1`, `TORCH_TRACE`, `TORCH_TPU_PROFILER_OUTPUT_DIR`, `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS`, `TORCH_TPU_TOPOLOGY` — [API reference](https://torchtpu.readthedocs.io/en/latest/api_reference/)
- Other: `torch.amp.autocast("tpu")` defaults bfloat16; distributed backend `"tpu_dist"` (alias `"tpu"`); `torch.tpu.optimizers.SingleTraceTrainer` fuses fwd+loss+bwd+step — [API reference](https://torchtpu.readthedocs.io/en/latest/api_reference/)

### Inferences
- Only fully stated default for cache: tier 1 on; tier 2 and 3 defaults are as above.

### Gaps
- Docs say the persistent cache (tier 2) is "device-local"; API doc says host-local `/dev/shm`. Minor wording difference, not a real conflict.

## Are Captum, SHAP or other explainability tools mentioned?

### Takeaway
No. Zero mentions found in docs, blog, or searched issues/PRs/forum.

### Cited Findings
- GitHub search for captum and shap in repo issues/PRs returned 0 results (API query, before rate limit) — GitHub search API `repo:google-pytorch/torch_tpu captum` / `shap` (no URL page; API https://api.github.com/search/issues)
- Forum category has only two topics: "About the TPU category" (Oct 7 2026) and "TorchTPU DevLabs, Nov 9-10, 2026" (Oct 28) — [discuss.pytorch.org TPU](https://discuss.pytorch.org/c/tpu/45)
- No GitHub Discussions (has_discussions false) — GitHub repo API
- Docs ecosystem list: Hugging Face, Ray, vLLM/SGLang, TorchTitan, TorchRec, llm-d; nothing on interpretability — [Google Cloud](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu)

### Inferences
- Captum/SHAP are untested by the TorchTPU team; compatibility rests on generic PyTorch autograd behavior.

### Gaps
- Did not complete body-text search of issues for "explainab" / "interpretab" (rate limit); issue-search page listing showed no relevant titles.

## Running TorchTPU on free Colab TPU; Python/torch versions

### Takeaway
Docs claim Colab is supported (feature set "Colab" under validated; Intro page has an "Open In Colab" badge, "enable the TPU runtime"), but no Colab-specific install steps, Python or torch version were found. Official VM path: Python 3.12 + `pip install torch_tpu`.

### Cited Findings
- Feature set lists "Colab" under Community, Supported (Validated) — [Feature set](https://torchtpu.readthedocs.io/en/latest/feature_set/)
- Intro page: "Open In Colab" badge; run quickstart by enabling TPU runtime — [Intro](https://torchtpu.readthedocs.io/en/latest/intro_to_torchtpu/)
- VM guide: `sudo apt install -y python3.12 python3.12-venv`, `python3.12 -m venv env`, `pip install torch_tpu`; says install order matters; no torch version stated; Colab badge links to `get_started_on_a_vm.ipynb` on branch `0.2.0-docs` — [Get started on a VM](https://torchtpu.readthedocs.io/en/latest/get_started_on_a_vm/)
- README: latest public release 0.2.0 on PyPI: `pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu`; Python 3.11+ per PyTorch compatibility matrix; Apache 2.0; notebooks are Marimo files needing TPU VM (v6e easiest) — [README](https://github.com/google-pytorch/torch_tpu)
- Conflict: Python 3.12 (VM guide) vs 3.11+ (README). Python 3.12 is the Kaggle environment in issue 3047.
- API doc: `torch.accelerator` APIs need PyTorch 2.11+, `torch.accelerator.random` needs 2.13+ (older use `torch.tpu`), implying a range of torch versions are supported — [API reference](https://torchtpu.readthedocs.io/en/latest/api_reference/)
- Issue #3047 (closed "not planned" Oct 30 by stale bot): Kaggle TPU install failed because earlier nightlies needed an Artifact Registry token; a maintainer-side comment says the Colab team was given a single wheel file; no official Kaggle/Colab route delivered in that thread — [Issue 3047](https://github.com/google-pytorch/torch_tpu/issues/3047)
- Supported hardware v5e, v5p, v6e, v7/7x — [FAQ](https://torchtpu.readthedocs.io/en/latest/faq/), [Feature set](https://torchtpu.readthedocs.io/en/latest/feature_set/). Colab TPU runtime hardware (v5e-1/v6e-1) was not verified.

### Inferences
- Pip-installing 0.2.0 on Colab is plausible if the Colab runtime Python is 3.12 and libtpu/TPU VM access works, but no source confirms free-tier Colab works end to end. Free Colab TPU hardware generation is not specified in any source seen.

### Gaps
- Could not read the Colab notebook (sign-in required), so no Colab install cell, torch version, or runtime type known. Test needed.
- Pip wheel metadata (requires_python, requires_dist) not retrievable from PyPI JSON (null).
