# Integrated Gradients sharded across 4 TPU chips (v6e-4, us-east5-b, 2026-11-01)

Script: `scripts/captum_multichip_ig.py`, launched with `torchrun --nproc_per_node=4` after
`eval $(python -m torch.tpu.distributed | sed 's/^/export /')`. Released wheel torch_tpu 0.2.0,
torch 2.15.0, Captum 0.9.0, transformers 5.x, Qwen3-0.6B fp32, 53-token prompt, strict eager,
float32 matmul precision "highest", riemann_trapezoid.

Each rank evaluates a disjoint quarter of the Riemann grid in one batched forward/backward on its
own chip, then `dist.all_reduce(SUM)` over the `tpu_dist` process group assembles the
attribution. Rank 0 also runs Captum's `LayerIntegratedGradients` on one chip as the reference.
Warm times are the median of 3 runs after a warm-up; the sharded time includes the all-reduce.

| n_steps | 4 chips, sharded (s) | 1 chip, Captum (s) | 1-chip batching | rel. error vs Captum | speedup |
|---|---|---|---|---|---|
| 64 | 0.355 | 0.219 | all 64 at once | 1.2e-4 | 0.62x |
| 256 | 0.327 | 0.547 | internal_batch_size=64 | 5.6e-8 | 1.67x |
| 256 | 0.357 | OOM | all 256 at once | n/a | n/a |

Notes

- Process model: libtpu is exclusive per process and each torchrun worker owns exactly one chip
  (`torch.accelerator.device_count()` is 1 inside a worker, 4 in the parent). A second process
  that touches the TPU while one is running fails with "The TPU is already in use by process
  with pid N". `TPU_VISIBLE_CHIPS` style variables are not how chips are assigned.
- 64 steps is too small a problem to split: one chip finishes the whole 64-step batch in 0.22 s,
  and the 4-way split pays an all-reduce plus four separate dispatches for a 16-step batch each.
- 256 steps does not fit on one v6e chip in a single batch (the same OOM as on the v6e-1:
  `tt_jit_scaled_dot_product_cudnn_attention` fails to reserve 864 MB with 860 MB free). Captum's
  `internal_batch_size=64` works around it in 4 sequential chunks; the 4-chip version runs the
  same 4 chunks in parallel, hence 1.67x rather than 4x (per-chunk fixed costs and the all-reduce
  are not amortised at this size).
- Agreement with Captum is at float32 round-off when the two sides use the same batch size
  (5.6e-8); at different batch sizes the matmul reduction order differs and the gap is 1e-4,
  consistent with the single-chip precision findings.
- An unbatched variant (one grid point per forward, 16 dispatches per rank) ran at 5.86 s: the
  strict-eager per-dispatch cost dominates, as in the single-chip execution-mode results.
