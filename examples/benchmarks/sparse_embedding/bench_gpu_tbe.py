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

"""Single-GPU sparse embedding forward/backward bandwidth bench (FBGEMM TBE).

Measures achieved HBM bandwidth of the recsys embedding path on ONE GPU using
FBGEMM's SplitTableBatchedEmbeddingBagsCodegen (TBE), with RowWiseAdagrad fused
into the backward (the GPU equivalent of TPU SparseCore's fused-Adagrad path).

Methodology:
  - all inputs (indices/offsets/grad) pre-built on GPU before the timed loop
  - tight CUDA-event-timed loop of `iters` steps after `warmup`
  - sweep pooling factor (the dominant variable for achieved BW) and batch size
  - bytes/sample accounting:
      fwd = (K+1)*D*4                          (read K rows + write 1 pooled row)
      bwd = D*4 + K*(2*D*4 + 2*4)              (rowwise-adagrad slot = 4B scalar/row)
    summed over all tables; BW = bytes/sample * batch / step_time.

Run:
  CUDA_VISIBLE_DEVICES=0 python examples/benchmarks/sparse_embedding/bench_gpu_tbe.py
  (set CUDA_VISIBLE_DEVICES to an idle GPU)
"""

import argparse
from typing import List, Tuple

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)

# A100 80GB SXM4 HBM2e peak (GB/s); override with --peak-gbps.
_A100_PEAK_GBPS = 2039.0


def build_tbe(
    num_tables: int, rows: int, dim: int, optimizer: EmbOptimType, device: torch.device
) -> SplitTableBatchedEmbeddingBagsCodegen:
    specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]] = [
        (rows, dim, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
        for _ in range(num_tables)
    ]
    return SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=specs,
        optimizer=optimizer,
        learning_rate=0.1,
        eps=0.1,
        pooling_mode=PoolingMode.SUM,
        weights_precision=SparseType.FP32,
        output_dtype=SparseType.FP32,
        device=device,
    )


def make_inputs(
    num_tables: int, rows: int, batch: int, pool: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    total = batch * num_tables * pool
    indices = torch.randint(0, rows, (total,), dtype=torch.long, device=device)
    offsets = torch.arange(
        0, total + 1, pool, dtype=torch.long, device=device
    )  # length batch*num_tables + 1, uniform bag size = pool
    return indices, offsets


def bytes_per_sample(
    num_tables: int, dim: int, pool: int, optimizer: str
) -> Tuple[int, int]:
    fwd = (pool + 1) * dim * 4
    if optimizer == "adagrad":
        # Full Adagrad: per-element accumulator row read/write.
        bwd = dim * 4 + pool * (4 * dim * 4)
    else:
        # RowWiseAdagrad: scalar accumulator read/write per row.
        bwd = dim * 4 + pool * (2 * dim * 4 + 2 * 4)
    return num_tables * fwd, num_tables * (fwd + bwd)


def _time_loop(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3  # seconds/iter


def bench(
    num_tables: int,
    rows: int,
    dim: int,
    batch: int,
    pool: int,
    optimizer_name: str,
    optimizer: EmbOptimType,
    iters: int,
    warmup: int,
    peak: float,
    device: torch.device,
) -> None:
    emb = build_tbe(num_tables, rows, dim, optimizer, device)
    indices, offsets = make_inputs(num_tables, rows, batch, pool, device)
    out = emb(indices, offsets)
    grad = torch.randn_like(out)

    def fwd() -> None:
        emb(indices, offsets)

    def fwd_bwd() -> None:
        o = emb(indices, offsets)
        o.backward(grad)

    t_fwd = _time_loop(fwd, iters, warmup)
    t_fb = _time_loop(fwd_bwd, iters, warmup)
    b_fwd, b_fb = bytes_per_sample(num_tables, dim, pool, optimizer_name)
    bw_fwd = b_fwd * batch / t_fwd / 1e9
    bw_fb = b_fb * batch / t_fb / 1e9
    print(
        f"  opt={optimizer_name:7s} T={num_tables} rows={rows // 1000}K "
        f"D={dim} bs={batch:5d} pool={pool:3d} | "
        f"fwd {t_fwd * 1e3:6.2f}ms {bw_fwd:7.0f} GB/s ({100 * bw_fwd / peak:4.1f}%) | "
        f"fwd+bwd {t_fb * 1e3:6.2f}ms {bw_fb:7.0f} GB/s ({100 * bw_fb / peak:4.1f}%)",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-tables", type=int, default=16)
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--batches", type=int, nargs="+", default=[512, 4096])
    p.add_argument("--pools", type=int, nargs="+", default=[1, 16, 64, 100])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--peak-gbps", type=float, default=_A100_PEAK_GBPS)
    p.add_argument(
        "--optimizer",
        choices=["rowwise", "adagrad"],
        default="rowwise",
        help=(
            "rowwise uses FBGEMM EXACT_ROWWISE_ADAGRAD; adagrad uses "
            "FBGEMM EXACT_ADAGRAD with a full per-element accumulator."
        ),
    )
    args = p.parse_args()
    optimizer = (
        EmbOptimType.EXACT_ADAGRAD
        if args.optimizer == "adagrad"
        else EmbOptimType.EXACT_ROWWISE_ADAGRAD
    )

    assert torch.cuda.is_available(), "no CUDA device"
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    total_gb = args.num_tables * args.rows * args.dim * 4 / 1e9
    print(f"GPU: {name} | peak {args.peak_gbps:.0f} GB/s")
    print(
        f"workload: {args.num_tables} tables x {args.rows} rows x {args.dim} dim "
        f"f32 = {total_gb:.1f} GB embeddings, {optimizer.value} fused"
    )
    print("== FBGEMM TBE single-GPU embedding fwd / fwd+bwd BW ==")
    for batch in args.batches:
        for pool in args.pools:
            bench(
                args.num_tables,
                args.rows,
                args.dim,
                batch,
                pool,
                args.optimizer,
                optimizer,
                args.iters,
                args.warmup,
                args.peak_gbps,
                device,
            )


if __name__ == "__main__":
    main()
