# Captum on TorchTPU

TorchTPU support for [Captum](https://github.com/meta-pytorch/captum), PyTorch's explainability library.
Everything here runs on `torch.device("tpu")` via [TorchTPU](https://github.com/google-pytorch/torch_tpu):
the model (`device_map="tpu"`), the attribution, and the tests. Built for the #TorchTPUSprint.

## What this adds

1. **A verified compatibility matrix** of every `captum.attr` method on TorchTPU, CPU vs TPU numerics
   included (`scripts/captum_tpu_matrix.py`).
2. **A device-generic test suite for Captum** (`upstream/captum/tests/attr/test_accelerator.py`, branch
   `tpu-accelerator-tests`): Captum's existing 78 attribution configs, run on CPU and on any accelerator
   (`CAPTUM_TEST_DEVICE=tpu`, or `torch.accelerator`, or CUDA) and compared. Upstream PR: meta-pytorch/captum#1951.
3. **Fixes** for whatever the matrix and suite turn up, upstreamed where possible.
4. **A "Captum on TPUs" tutorial** (notebook + blog) with the one setting every TPU user needs:
   `torch.set_float32_matmul_precision("highest")` — see FINDINGS.md.

See [FINDINGS.md](FINDINGS.md) for measured results and [reports/](reports/) for the prior-art research. The
write-up is published separately.

## Upstream work in torch_tpu

Focus shifted to contributing to TorchTPU itself (sprint priority topic: execution modes, XProf, precision).
- **Fix for google-pytorch/torch_tpu#6036** (eager device buffers for distinct Python scalars never evicted):
  bounded scalar-buffer cache in `csrc/eager/op_dispatcher.cc` plus `tests/scalar_buffer_cache_test.py`.
  PR google-pytorch/torch_tpu#6102; built and verified on the v6e VM (+4999 -> +903 live allocations, regression test passes).
- **Issue google-pytorch/torch_tpu#6103 + PR google-pytorch/torch_tpu#6105**: `aten::max_unpool2d/3d` were missing (found by the
  Captum suite; broke DeepLift on CNNs); implemented as a StableHLO batched scatter with tests, verified on the v6e VM.
- **PR google-pytorch/torch_tpu#6106**: compiled TPU partitions read CPU view arguments past their buffer
  (`x * cpu_tensor[1].to(tpu)` under `torch.compile` returned 0). Root cause via StableHLO dump; fix in
  `split_compiler.py` compiles against the contiguous layout the argument is copied with. Test added, verified on v6e.
- **PR meta-pytorch/captum#1952**: LayerIntegratedGradients captured a CPU `cumsum` tensor as a compiled-graph
  input (the "expected tensor to be on tpu, got cpu" failure); replaced with Python ints. 12 LIG tests pass on TPU.
- Branch `docs-memory-apis`: `torch.accelerator` memory API reference section (#4413); measured on v6e.
- Branch `loader-import-error-context`: the versioned extension loader names the glue and the installed PyTorch
  when the `.so` fails to import (was a bare `undefined symbol`). Test added.
- Branch `missing-ops-batch`: `smooth_l1_loss`, `huber_loss` (+backward), `logaddexp`, `logaddexp2`, `nansum`,
  `std_mean`, `trace`, `addbmm` lowerings with unit tests; build and TPU test in progress.
- Branch `explainability-example`: `examples/explainability/captum_attribution.py` plus docs case study
  "Explainability as a performance lens" (precision, strict/DeferAndFuse/compile comparison, XProf, memory,
  convergence). Verified on v6e: Qwen3-0.6B LIG 64 steps warm 0.80 s strict / 0.36 s DeferAndFuse / 0.16 s compile.
- Branch `dynamism-denylist-triage`: 11 stale entries removed from the bounded-dynamism deny list (verified in eager and
  compiled mode on v6e), `clamp` moved to a diagnosed reason.
- Triage comment on google-pytorch/torch_tpu#4408 (compiled SDPA backward NaN does not reproduce on 0.2.0).
- The docs, loader, ops, example and dynamism branches await PR creation: the repository allows users without write
  access at most 3 open PRs, and #6102, #6105, #6106 fill that quota.
  Each is claimed by an issue in the meantime: google-pytorch/torch_tpu#6111 (ops), #6112 (dynamism), #6113 (loader),
  #6114 (example), and a comment on #4413 (memory docs).

## Layout

| Path | Purpose |
|---|---|
| `scripts/tpu_vm.sh`, `scripts/env.sh` | create / set up / sync / run / delete the Cloud TPU v6e-1 VM |
| `scripts/tpu_smoke.py` | 12 checks: hooks, autograd.grad, double backward, custom Function, torch.func, compile+backward, Captum IG |
| `scripts/tpu_correctness.py` | CPU vs TPU agreement for 6 Captum methods, default vs "highest" matmul precision |
| `scripts/captum_tpu_matrix.py` | the full method matrix |
| `scripts/hf_captum_bench.py`, `scripts/hf_ig_diag_bench.py` | Captum on Qwen3-0.6B: agreement, IG scaling, eager vs compile vs DeferAndFuse |
| `scripts/demo_captum_qwen_tpu.py` | end-to-end demo: `device_map="tpu"`, LayerIntegratedGradients, LLMAttribution, HTML heatmap |
| `torchtpu_xai/` | small helpers (HTML heatmap renderer, a reference IG with convergence check). Not the product. |
| `upstream/captum/` | shallow clone of Captum; PR work lives on branch `tpu-accelerator-tests` |
| `hf_tpu_bench.py`, `build_notebook.py`, `hf_on_torchtpu.ipynb` | earlier "HF on TorchTPU" scaffold; superseded |

## Running on a TPU VM

```bash
source scripts/env.sh                      # gcloud on PATH; set TPU_PROJECT / TPU_ZONE
scripts/tpu_vm.sh create && scripts/tpu_vm.sh setup && scripts/tpu_vm.sh sync
scripts/tpu_vm.sh run '~/env/bin/python tpu_smoke.py'
scripts/tpu_vm.sh run '~/env/bin/python captum_tpu_matrix.py'
scripts/tpu_vm.sh run 'CAPTUM_TEST_DEVICE=tpu ~/env/bin/python -m pytest -q test_accelerator.py'
scripts/tpu_vm.sh delete                   # the VM bills while it exists
```

Environment used so far: Cloud TPU v6e-1 (us-east5-b), Python 3.12, torch 2.15.0, torch_tpu 0.2.0,
Captum 0.9.0, transformers 5.22.1.
