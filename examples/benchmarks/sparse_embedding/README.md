# sparse embedding benchmarks

Apples-to-apples sparse embedding-lookup benchmarks across accelerators (GPU vs TPU),
to characterize achieved HBM bandwidth for the recsys embedding path.

## Goal

Run a single, matched workload on both backends and report achieved HBM bandwidth
(GB/s/chip) across a pool-factor and batch-size sweep, so GPU and TPU numbers are
directly comparable.

Reference workload:
`N tables x rows x dim`, f32, fwd + bwd + fused RowWiseAdagrad, pool sweep 1..100,
batch sweep, CUDA-event / device-synced timing of a steady-state loop after warmup.

## Backends

- **GPU (FBGEMM TBE)** — `SplitTableBatchedEmbeddingBagsCodegen`, single chip,
  RowWiseAdagrad fused into the backward. Local A100 (or B200 where available).
- **TPU SparseCore** — `jax_tpu_embedding` `tpu_sparse_dense_matmul` (+ fused Adagrad),
  pooled, with minibatching where the per-partition id count exceeds the TileSpmem
  (~512 KB/SC) `max_ids_per_partition` budget.
- **TPU SparseCore via TorchTPU ops** — `bench_tpu_torchtpu.py` uses JTE only for
  table stacking / CSR preprocessing, then calls `torch.ops.tpu.sparse_dense_matmul`
  and the experimental TorchTPU optimizer-update ops directly. It can run full
  Adagrad (`--optimizer adagrad`, matching JTE's slot shape) or row-wise Adagrad
  (`--optimizer rowwise_adagrad`, matching the FBGEMM optimizer state shape).
- **TPU TensorCore gather** — plain `jnp.take` / `index_select` baseline (no SparseCore).

## Script inventory

All scripts in this directory are intentionally standalone and are run directly,
not through Bazel targets:

| file | purpose |
|---|---|
| `bench_gpu_tbe.py` | GPU FBGEMM TBE embedding-bag forward and forward+backward benchmark. |
| `bench_gpu_gather.py` | GPU raw `torch.index_select` and explicit pooled-gather baseline. |
| `bench_gpu_gather_vs_tbe.py` | One-process comparison of GPU raw gather, pooled gather, and FBGEMM TBE. |
| `bench_tpu_gather.py` | TPU TensorCore/JAX `jnp.take` pooled-gather baseline, no SparseCore. |
| `bench_tpu_jte.py` | TPU JTE SparseCore embedding benchmark with JTE preprocessing/minibatching. |
| `bench_tpu_torchtpu.py` | Single-process TorchTPU direct SparseCore op benchmark using JTE CSR preprocessing. |
| `bench_tpu_torchtpu_dist.py` | Distributed 8-process TorchTPU direct SparseCore benchmark for the target high-pool case. |
| `bench_tpu_pallas_sc_gather.py` | Pallas SparseCore raw-gather diagnostic probe, no pooling or optimizer. |
| `csr_utils.py` | Shared CSR cleanup helper for the TorchTPU direct-op benchmarks. |

## Key findings

- The gap is dominated by **pooling factor**, not silicon: at pool=1 the TPU
  TensorCore gather (~1.3 TB/s/chip) is comparable to a B200 (~1.3 TB/s @ pool=1 dim=256).
  GPUs pull ahead only at high pool (B200 ~5.5 TB/s = 88% HBM @ pool=100 dim=256),
  where arithmetic intensity amortizes launch/addressing overhead.
- So a fair comparison must sweep pool 1..100 on both sides with the same op
  (fwd-only vs fwd+bwd) and the same tables/dim/batch.

## Environment (GPU)

Fresh conda env (`embbench`), torch 2.11 + FBGEMM 1.6 (matches the torchtpu stack):
```bash
conda create -y -n embbench python=3.11
conda activate embbench
HTTPS_PROXY=http://fwdproxy:8080 pip install -U torch fbgemm-gpu \
    --index-url https://download.pytorch.org/whl/cu128
```
Note: `fbgemm-gpu==1.7.0` is not yet on the cu128/cu130 pip wheel index (only ≤1.6.0);
1.6.0 is the newest installable and is fine for these kernels.

## A100 results (bench_gpu_tbe.py)

16 tables × 1M rows × 128 dim f32 (8.2 GB), FBGEMM TBE, RowWiseAdagrad fused, single
A100 80GB (HBM peak ~2039 GB/s). All 16 tables are fused into ONE TBE (one weight
buffer + one kernel) — the GPU-native form of the "fuse N tables → one-table problem".

| batch | pool | fwd GB/s (%peak) | fwd+bwd GB/s (%peak) |
|------:|-----:|-----------------:|---------------------:|
| 4096  |   1  |   220 (11%)      |   254 (12%)          |
| 4096  |  16  |  1430 (70%)      |  1024 (50%)          |
| 4096  |  64  |  1601 (78%)      |  1136 (56%)          |
| 4096  | 100  |  1614 (79%)      |  1196 (59%)          |
| 512   | 100  |  1382 (68%)      |   996 (49%)          |

Takeaway: **pooling factor is the dominant variable.** pool=1 is launch/overhead-bound
(~11% HBM); pool=100 + big batch saturates ~79% HBM on forward, ~59% on fwd+bwd. Same
roofline shape as high-pool GPU embedding-bag studies. Run: `CUDA_VISIBLE_DEVICES=0 python examples/benchmarks/sparse_embedding/bench_gpu_tbe.py`.

## GPU raw-gather results (bench_gpu_gather.py)

`bench_gpu_gather.py` is the GPU analogue of a plain `gather`/`take`, not TBE and
not a fused embedding-bag kernel. It uses the same fused `[16M, 128]` table and
times:

```text
raw gather:    output = table.index_select(0, flat_indices)
pooled gather: output = table.index_select(...).view(nbag, pool, dim).sum(1)
```

The raw-gather BW counts one row read plus one output row write per id. The
pooled path uses the same forward byte numerator as the embedding-bag benchmarks
(`K` row reads plus one pooled output write per bag). On the currently visible
GPU (`NVIDIA PG509-210`; run with default A100 peak only for raw GB/s, not %peak),
random indices measured:

| batch | pool | raw gather ms | raw gather GB/s | pooled gather ms | pooled gather GB/s |
|------:|-----:|--------------:|----------------:|-----------------:|-------------------:|
| 4096  |   1  | 0.05 | 1231 | 0.10 | 643 |
| 4096  |  16  | 0.78 | 1372 | 1.14 | 502 |
| 4096  |  64  | 3.11 | 1379 | 4.39 | 497 |
| 4096  | 100  | 4.87 | 1379 | 6.84 | 496 |
| 512   |   1  | 0.02 | 462  | 0.04 | 236 |
| 512   |  16  | 0.10 | 1316 | 0.16 | 446 |
| 512   |  64  | 0.39 | 1365 | 0.57 | 480 |
| 512   | 100  | 0.61 | 1371 | 0.87 | 485 |

Takeaway: raw `index_select` reaches ~1.37 TB/s on random gathers once the lookup
is large enough, but adding the explicit pooling reduction drops the effective
embedding-bag forward numerator to ~0.5 TB/s. This is still not equivalent to
FBGEMM TBE, which fuses gather and pooling in a specialized embedding kernel.

`bench_gpu_gather_vs_tbe.py` runs the direct comparison in one process with
matching table-local IDs for TBE and table-offset IDs for the fused raw-gather
tensor:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n embbench python \
  examples/benchmarks/sparse_embedding/bench_gpu_gather_vs_tbe.py --batch 4096 --pool 100
```

Measured on the same visible `NVIDIA PG509-210`:

| path | ms | GB/s | note |
|---|---:|---:|---|
| raw `index_select` | 4.87 | 1379 | row read + materialized row write per id |
| pooled `index_select + sum` | 6.83 | 496 | explicit reduction after materializing `[6.55M,128]` |
| FBGEMM TBE fwd | 2.11 | 1604 | fused gather + pooling, writes only pooled output |
| FBGEMM TBE fwd+bwd | 8.51 | 1197 | fused RowWiseAdagrad backward |

For this target point, TBE forward is 2.3x faster than raw `index_select` by wall
time, and 3.2x faster than explicit pooled `index_select + sum`.

## TPU SparseCore status

`bench_tpu_gather.py` is the TPU TensorCore/JAX pooled-gather baseline. It uses
`jnp.take(...).sum(axis=1)` on the same fused `[16M, 128]` table shape, but it is
not a SparseCore path and it does not use the TorchTPU custom op. The exact
`batch=4096, pool=100` point was not rerun for this summary snapshot. Earlier raw
TensorCore gather probes reached roughly 0.85-1.0 TB/s on one TPU device at
`dim >= 128`, but that is a raw gather ceiling and is not directly comparable to
the pooled embedding-bag rows below.

`bench_tpu_jte.py` is the SparseCore/JTE path. Forward uses `enable_minibatching=True`
on both preprocessing and `tpu_sparse_dense_matmul`, which is required for high-pool
inputs to run. Backward follows the production `scripts/yuzhenhuang/tpu` pattern:
CSR inputs are lifted to a mesh and `tpu_sparse_dense_matmul_grad` runs inside
`jax.shard_map`, so the grad op receives local embedding shards instead of the full
logical table.

TPU7x HBM peak is treated as 7370 GB/s per physical chip; the pod has 8 JAX devices
across 4 physical chips, so per-chip BW is aggregate BW / 4.

`--layout fused-table` (the default) is the one-feature 16M-row workaround that
runs without dropping ids at high pool. On an 8-device TPU7x VM (8 devices =
4 chips), short `--iters 3` runs produced:

| batch | pool | fwd GB/s/chip (%HBM) | fwd+bwd GB/s/chip (%HBM, full Adagrad bytes) |
|------:|-----:|---------------------:|-------------------------:|
| 512   |   1  |   0.0 (0.00%)        |   0.1 (0.00%)            |
| 512   |  16  |   0.3 (0.00%)        |   1.1 (0.02%)            |
| 512   |  64  |   1.2 (0.02%)        |   4.5 (0.06%)            |
| 512   | 100  |   1.9 (0.03%)        |   6.9 (0.09%)            |
| 4096  |   1  |   0.3 (0.00%)        |   0.8 (0.01%)            |
| 4096  |  16  |   2.5 (0.03%)        |   8.8 (0.12%)            |
| 4096  |  64  |   8.7 (0.12%)        |  29.9 (0.41%)            |
| 4096  | 100  |  13.0 (0.18%)        |  44.3 (0.60%)            |

`--layout stacked-tables` keeps 16 real feature/table specs and stacks them into
one SC stack, which is closer to production JTE. JTE's `stack_tables` ignores the
per-table `max_ids_per_partition` and uses stacked default 256. Observed
per-partition ids at `batch=4096 pool=100` are ~25-26k, so JTE minibatches the
input unless `--max-ids` is raised enough to exceed SparseCore scratchpad limits.
The harness now patches the resulting `StackedTableSpec` directly with `--max-ids`
(default 1024, tested from 512 to 4096), without using
`update_preprocessing_parameters`. This makes the true stacked layout run without
id dropping. It still does not improve performance:
`batch=4096 pool=100 --layout stacked-tables --max-ids 512 --no-autotune-sc-limits
--no-allow-id-dropping` measured fwd 67.23 ms = 12.6 GB/s/chip and fwd+bwd
101.67 ms = 25.0 GB/s/chip. Running `--autotune-sc-limits` is harmful here: it
raises the COO buffer enough to fail the SparseCore scratchpad check (`Row bytes:
~13 MB`, `TileSpmem: 524288`).

Takeaway: the backward sharding bug is fixed, but the standalone SparseCore numbers
are still overhead/scratchpad limited and nowhere near GPU-class for this exact
uniform high-pool microbench.

Production `scripts/yuzhenhuang/tpu/sparse_bench` was also checked on pod 2. Its
default-style low-pool case (`dlrm_small`, 26 tables x 100k x 64, global
batch 4096, pool=1, stacked tables) runs with SC fwd+bwd around 11.17 ms, but it
does not provide an apples-to-apples high-pool result for this target. The path
does not pass `enable_minibatching=True`; with `pool=100`, observed per-partition
ids are far above the default limit, and autotuning the limits high enough runs
into the SparseCore TileSpmem row-size failure. The standalone `bench_tpu_jte.py`
is therefore still the JTE high-pool path for this workload.

## TorchTPU SparseCore direct-op status

`bench_tpu_torchtpu.py` and `bench_tpu_torchtpu_dist.py` are the direct
TorchTPU-op paths. They construct JTE table specs and use JTE for SparseCore CSR
preprocessing, but the timed region calls TorchTPU SparseCore ops directly:

```bash
python3 bench_tpu_torchtpu.py --optimizer adagrad --bwd
python3 bench_tpu_torchtpu.py --optimizer rowwise_adagrad --bwd
```

`--optimizer adagrad` matches JTE's full elementwise Adagrad accumulator.
`--optimizer rowwise_adagrad` uses a 1-D scalar accumulator per row, making
fwd+bwd optimizer traffic comparable to FBGEMM `EXACT_ROWWISE_ADAGRAD`.

The pod has been tested with a source-built TorchTPU wheel that exposes the
minibatching custom calls:

```text
a source-built TorchTPU wheel
```

Use the distributed harness for the target high-pool case. It launches one
process per JAX/TorchTPU device and uses `--sharded-table`, matching the physical
JTE table shard shape (`16M / 8 = 2M` rows per JAX device):

```bash
JAX_PLATFORMS=cpu torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  examples/benchmarks/sparse_embedding/bench_tpu_torchtpu_dist.py \
  --batch 4096 --pool 100 --sc-limit 12288 --sharded-table --bwd \
  --optimizer rowwise_adagrad
```

Corrected distributed numbers for `16 x 1M x 128, batch=4096, pool=100`
on pod 2 (8 JAX devices = 4 physical TPU7x chips):

| path | optimizer | fwd ms | fwd GB/s/chip (%HBM) | fwd+bwd ms | fwd+bwd GB/s/chip (%HBM) |
|---|---|---:|---:|---:|---:|
| JTE SparseCore | full Adagrad | 65.2 | 13.0 (0.18%) | 95.3 | 44.2 (0.60%) |
| TorchTPU direct distributed | full Adagrad | 6.40 | 132 (1.80%) | 17.82 | 236 (3.21%) |
| TorchTPU direct distributed | RowWiseAdagrad | 6.41 | 132 (1.79%) | 17.27 | 147 (2.00%) |

Notes:
- The TorchTPU aggregate BW accounting was corrected after an earlier version
  accidentally multiplied by world size twice. Per-chip numbers above divide
  aggregate BW by 4 physical chips, not 8 JAX devices.
- `--sharded-table` now uses the actual per-rank table shape in the timed path.
  An earlier harness revision accidentally allocated the full 16M-row table in
  every rank outside smoke tests, which understated the direct-op result.
- `--sc-limit 12288` is the best tested point after the sharded-table fix.
  `--sc-limit 16384` still fails with the SparseCore TileSpmem row-size check
  (`Row bytes: 8454144`, `TileSpmem: 524288`), so the limit sweep is bounded by
  scratchpad layout rather than table size.
- Full-Adagrad fwd+bwd uses full `[rows, dim]` accumulator traffic. RowWiseAdagrad
  uses `[rows]` accumulator traffic and is the optimizer-equivalent comparison to
  FBGEMM TBE.
- The full-Adagrad direct op was numerically checked against JTE for cases where
  both fit: forward output, updated table, and updated accumulator matched exactly
  in the tested one-rank case.
- The target-size 8-rank sharded RowWiseAdagrad path was smoke-checked with a
  ones table: every rank reported `smoke_fwd_ok=True`, and backward returned
  local shard-shaped updates `(2000000, 128)` plus rowwise accumulators
  `(2000000,)`.

## Pallas SparseCore raw-gather probe

`bench_tpu_pallas_sc_gather.py` is not an embedding-bag benchmark. It is a
diagnostic SparseCore Pallas kernel for:

```text
output[i, :] = table[indices[i], :]
```

It has no pooling reduction and no optimizer update. It is useful only to probe
how much raw row-copy/gather traffic this Pallas SparseCore path can issue.

Corrected best result after explicitly sharding the replicated inputs across all
8 JAX devices:

| shape | mode | block_rows | ms | aggregate GB/s | GB/s/chip (%HBM) |
|---|---|---:|---:|---:|---:|
| rows/device=2M, ids/device=819200, dim=128 | random gather | 400 | 3.66 | 1832 | 458 (6.21%) |

Random, sequential, and contiguous-copy variants were close once sharding was
fixed. Larger blocks improved throughput until SparseCore tile SPMEM failed at
`block_rows=512`. This suggests the Pallas probe is limited by the exposed
SparseCore copy/gather execution path and SPMEM tiling, not by random ID locality.

## Current summary

Target high-pool snapshot for `16 x 1M x 128`, `batch=4096`, `pool=100`:

| backend | source script | fwd GB/s/chip (%HBM) | fwd+bwd GB/s/chip (%HBM) | caveat |
|---|---|---:|---:|---|
| A100 FBGEMM TBE | `bench_gpu_tbe.py` / `bench_gpu_gather_vs_tbe.py` | 1614 GB/s on one GPU (79%) | 1196 GB/s on one GPU (59%) | RowWiseAdagrad, mature GPU TBE kernel |
| GPU raw `index_select` | `bench_gpu_gather.py` / `bench_gpu_gather_vs_tbe.py` | 1379 GB/s on one GPU (68%) | n/a | materializes `[6.55M, 128]`, no pooling/optimizer |
| GPU pooled `index_select + sum` | `bench_gpu_gather.py` / `bench_gpu_gather_vs_tbe.py` | 496 GB/s on one GPU (24%) | n/a | explicit pooling after materializing gather output |
| TPU JAX TensorCore pooled gather | `bench_tpu_gather.py` | not recorded in this snapshot | not recorded in this snapshot | not SparseCore |
| JTE SparseCore | `bench_tpu_jte.py` | 13.0 (0.18%) | 44.2 (0.60%) | full Adagrad only; standalone JTE path overhead-bound |
| TorchTPU direct SparseCore | `bench_tpu_torchtpu_dist.py` | 132 (1.80%) | 236 (3.21%) | full Adagrad; sharded-table distributed direct op, optimizer-equivalent to JTE |
| TorchTPU direct SparseCore | `bench_tpu_torchtpu_dist.py` | 132 (1.79%) | 147 (2.00%) | RowWiseAdagrad; sharded-table distributed direct op, optimizer-equivalent to FBGEMM |
| Pallas SparseCore raw gather | `bench_tpu_pallas_sc_gather.py` | 458 (6.21%) | n/a | raw gather only, no pooling or optimizer |

The strongest current conclusion is that the standalone JTE path is not
representative of the best SparseCore throughput we can get from this pod. The
direct TorchTPU path is much faster than JTE but still far below A100 TBE and far
below TPU7x HBM peak. The Pallas raw-gather probe reaches higher throughput again,
but it is not yet an apples-to-apples pooled embedding benchmark.
