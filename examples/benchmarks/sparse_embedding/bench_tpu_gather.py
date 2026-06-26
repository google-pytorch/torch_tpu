# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

"""TPU TensorCore pooled-gather bench — matched to bench_gpu_tbe.py.

Same workload (num_tables x rows x dim, pool/batch sweep, fwd + fwd+bwd) and the
SAME bytes/sample accounting as the GPU FBGEMM TBE bench, but implemented as a
plain TensorCore gather + segment-sum (pooled embedding) in JAX. This is the
RELIABLE TPU path (the jax_tpu_embedding SparseCore path is overhead-bound /
crashes at high pool on this setup, so it is benched separately).

Tables are fused into one [num_tables*rows, dim] table (one gather), mirroring
the GPU's fused TBE. Single TPU device (= 1 TensorCore = half a chip); a full
chip has 2 -> ~2x. Per-iter forced execution via block_until_ready.

Run (on the pod): python bench_tpu_gather.py  [--batches ...] [--pools ...]
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

# v7x HBM peak per chip ~7400 GB/s; one device = 1 TensorCore (half a chip).
_V7X_HBM_PEAK_GBPS = 7400.0


def bytes_per_sample(num_tables: int, dim: int, pool: int) -> tuple[int, int]:
    fwd = (pool + 1) * dim * 4
    bwd = dim * 4 + pool * (2 * dim * 4 + 2 * 4)
    return num_tables * fwd, num_tables * (fwd + bwd)


def bench(
    num_tables: int, rows: int, dim: int, batch: int, pool: int, peak: float, iters: int
) -> None:
    V = num_tables * rows  # fused table
    table = jnp.asarray(np.random.randn(V, dim).astype(np.float32))
    table = jax.device_put(table)
    jax.block_until_ready(table)
    # one bag per (sample, table); B*num_tables bags of `pool` ids each
    nbag = batch * num_tables
    idxs = [
        jax.device_put(np.random.randint(0, V, size=(nbag, pool)).astype(np.int32))
        for _ in range(iters + 3)
    ]
    jax.block_until_ready(idxs[0])

    @jax.jit
    def fwd(t, i):
        # gather [nbag, pool, dim] -> sum over pool -> [nbag, dim]
        return jnp.take(t, i, axis=0).sum(axis=1)

    @jax.jit
    def fwd_bwd(t, i):
        # grad of pooled-sum loss wrt table (scatter-add) — fwd + bwd
        return jax.grad(lambda tt: jnp.take(tt, i, axis=0).sum(axis=1).sum())(t)

    def time_fn(fn, j0):
        for j in range(3):
            r = fn(table, idxs[j])
        jax.block_until_ready(r)
        t0 = time.perf_counter()
        for j in range(iters):
            r = fn(table, idxs[j0 + j])
        jax.block_until_ready(r)
        return (time.perf_counter() - t0) / iters

    b_fwd, b_fb = bytes_per_sample(num_tables, dim, pool)
    t_fwd = time_fn(fwd, 3)
    t_fb = time_fn(fwd_bwd, 3)
    bw_fwd = b_fwd * batch / t_fwd / 1e9
    bw_fb = b_fb * batch / t_fb / 1e9
    print(
        f"  T={num_tables} rows={rows // 1000}K D={dim} bs={batch:5d} pool={pool:3d} | "
        f"fwd {t_fwd * 1e3:6.2f}ms {bw_fwd:7.0f} GB/s/dev ({100 * bw_fwd / peak:4.1f}%) | "
        f"fwd+bwd {t_fb * 1e3:6.2f}ms {bw_fb:7.0f} GB/s/dev ({100 * bw_fb / peak:4.1f}%)",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-tables", type=int, default=16)
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--batches", type=int, nargs="+", default=[512, 4096])
    p.add_argument("--pools", type=int, nargs="+", default=[1, 16, 64, 100])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--peak-gbps", type=float, default=_V7X_HBM_PEAK_GBPS)
    args = p.parse_args()
    dev = jax.devices()[0]
    print(
        f"device: {dev.device_kind} (1 of {jax.device_count()}); "
        f"workload {args.num_tables}x{args.rows}x{args.dim} fused = "
        f"{args.num_tables * args.rows * args.dim * 4 / 1e9:.1f} GB; peak {args.peak_gbps:.0f} GB/s/chip "
        f"(1 device = 1 TensorCore = half a chip)",
        flush=True,
    )
    print("== TPU TensorCore pooled-gather fwd / fwd+bwd (per device) ==")
    for batch in args.batches:
        for pool in args.pools:
            bench(
                args.num_tables,
                args.rows,
                args.dim,
                batch,
                pool,
                args.peak_gbps,
                args.iters,
            )


if __name__ == "__main__":
    main()
