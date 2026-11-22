## Description

Adds an explainability example and a case study page.

`examples/explainability/captum_attribution.py` runs Captum Integrated Gradients on a TPU in the current execution mode, compares the attributions with the same computation on CPU, reports the warm time and peak HBM, and optionally writes an XProf trace with `torch.profiler`. It covers a small MLP (no downloads) and Hugging Face causal LMs through `LayerIntegratedGradients` with `logits_to_keep=1`.

`docs/source/case_studies_explainability.py` (Marimo source) and its Markdown export `docs/case_studies_explainability.md` cover:

- why attributions need `torch.set_float32_matmul_precision("highest")` (1e-3 vs 1e-7 relative error against CPU),
- strict eager vs DeferAndFuse vs `torch.compile(backend="tpu")` on Qwen3-0.6B, with the fixes the compiled path needs (meta-pytorch/captum#1952, #6106),
- the `aten::_to_copy` pattern in the XProf operator table and what it says about strict eager mode,
- bounding peak HBM with `logits_to_keep=1` and `internal_batch_size`,
- checking the convergence delta before trusting an attribution.

The page is registered in `mkdocs.yml`, `docs/source/index.py`, `docs/index.md` and `examples/README.md`. Attribution workloads are a compact stress test for the execution modes (many small ops with host scalars in the loop), which is why they make a useful case study beyond explainability itself.

## Test Plan

On a Cloud TPU v6e-1 (torch_tpu 0.2.0, Captum 0.9.0, Transformers 5.22.1), with the script as committed:

| Command | Result |
| --- | --- |
| `python3 captum_attribution.py` | warm 0.065 s, max rel err vs CPU 2.0e-07 |
| `TORCH_TPU_DEFER_AND_FUSE=1 python3 captum_attribution.py` | warm 0.039 s, 2.0e-07 |
| `python3 captum_attribution.py --compile` | warm 0.062 s, 2.0e-07 |
| `python3 captum_attribution.py --model Qwen/Qwen3-0.6B` | warm 0.80 s, 9.2e-05, peak 10.6 GiB |
| `TORCH_TPU_DEFER_AND_FUSE=1 ... --model Qwen/Qwen3-0.6B` | warm 0.36 s, 9.3e-05, peak 11.1 GiB |
| `... --model Qwen/Qwen3-0.6B --compile` (Captum with #1952, torch_tpu with #6106) | warm 0.16 s, 7.0e-05, peak 15.3 GiB |
| `... --model Qwen/Qwen3-0.6B --steps 128 --internal-batch-size 64` | warm 1.42 s, 5.5e-04 |
| `python3 captum_attribution.py --profile-dir ~/prof_example` | trace written, operator table printed (`aten::_to_copy` 82% of self CPU) |

`mkdocs build --strict` passes with the new page in the nav.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
