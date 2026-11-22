Branch `explainability-example` run on a v6e-4 (us-east5-b, 2026-11-01), wheel built from the branch on the 4-chip host, `examples/explainability/captum_attribution.py` in the three execution modes:

| mode | warm median of 3 | max rel err vs CPU |
|---|---|---|
| strict eager | 0.031 s | 2.0e-7 |
| `TORCH_TPU_DEFER_AND_FUSE=1` | 0.022 s | 2.0e-7 |
| `--compile` | 0.029 s | 2.0e-7 |

Multi-chip note for the case study: libtpu is exclusive per process, so multi-chip attribution is one torchrun worker per chip. Sharding the Integrated Gradients grid across the 4 chips with a `tpu_dist` all_reduce reproduces Captum's single-chip result to 5.6e-8 (equal batch size); at 256 steps, which does not fit one chip in a single batch, it is 1.67x faster than Captum's `internal_batch_size=64` chunking on one chip. Details: https://github.com/joinalahmed/captum-on-torchtpu/blob/main/results/v6e-4/multichip_ig.md
