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

"""Compare plain GPU gather against FBGEMM TBE on the same logical workload.

This is a small diagnostic companion to `bench_gpu_gather.py` and
`bench_gpu_tbe.py`. It is intentionally not a new kernel benchmark: it answers
one question directly for a fixed workload:

  how much faster is FBGEMM's fused embedding-bag path than plain
  `torch.index_select`, with and without an explicit pooling reduction?

The raw/pooled gather path uses one fused `[num_tables * rows, dim]` tensor.
The TBE path uses `num_tables` separate FBGEMM tables. Indices are generated as
table-local IDs and offset into the fused tensor for the raw/pooled path, so the
row distribution matches the TBE table layout.

Run:
  CUDA_VISIBLE_DEVICES=0 python bench_gpu_gather_vs_tbe.py --batch 4096 --pool 100
"""

import argparse
from typing import Callable, List, Tuple

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

_A100_PEAK_GBPS = 2039.0


def build_tbe(
    num_tables: int,
    rows: int,
    dim: int,
    device: torch.device,
) -> SplitTableBatchedEmbeddingBagsCodegen:
    specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]] = [
        (rows, dim, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
        for _ in range(num_tables)
    ]
    return SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=specs,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        eps=0.1,
        pooling_mode=PoolingMode.SUM,
        weights_precision=SparseType.FP32,
        output_dtype=SparseType.FP32,
        device=device,
    )


def make_matching_indices(
    num_tables: int,
    rows: int,
    batch: int,
    pool: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    table_local = torch.randint(
        0,
        rows,
        (num_tables, batch, pool),
        dtype=torch.long,
        device=device,
    )
    table_offsets = (
        torch.arange(num_tables, dtype=torch.long, device=device).view(num_tables, 1, 1)
        * rows
    )
    fused_indices = (table_local + table_offsets).reshape(-1)
    tbe_indices = table_local.reshape(-1)
    offsets = torch.arange(
        0,
        tbe_indices.numel() + 1,
        pool,
        dtype=torch.long,
        device=device,
    )
    return fused_indices, tbe_indices, offsets


def raw_gather_bytes_per_sample(num_tables: int, dim: int, pool: int) -> int:
    return num_tables * pool * 2 * dim * 4


def pooled_fwd_bytes_per_sample(num_tables: int, dim: int, pool: int) -> int:
    return num_tables * (pool + 1) * dim * 4


def tbe_fwd_bwd_bytes_per_sample(num_tables: int, dim: int, pool: int) -> int:
    fwd = (pool + 1) * dim * 4
    bwd = dim * 4 + pool * (2 * dim * 4 + 2 * 4)
    return num_tables * (fwd + bwd)


def time_loop(fn: Callable[[], torch.Tensor], iters: int, warmup: int) -> float:
    result = fn()
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    result.record_stream(torch.cuda.current_stream())
    return start.elapsed_time(end) / iters / 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tables", type=int, default=16)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--pool", type=int, default=100)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--peak-gbps", type=float, default=_A100_PEAK_GBPS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    vocab_size = args.num_tables * args.rows
    fused_table = torch.empty(
        (vocab_size, args.dim), dtype=torch.float32, device=device
    )
    fused_table.normal_(mean=0.0, std=1.0 / args.dim**0.5)
    fused_indices, tbe_indices, offsets = make_matching_indices(
        args.num_tables,
        args.rows,
        args.batch,
        args.pool,
        device,
    )
    nbag = args.batch * args.num_tables

    tbe = build_tbe(args.num_tables, args.rows, args.dim, device)
    tbe_out = tbe(tbe_indices, offsets)
    grad = torch.randn_like(tbe_out)

    def raw_gather() -> torch.Tensor:
        return torch.index_select(fused_table, 0, fused_indices)

    def pooled_gather() -> torch.Tensor:
        return (
            torch.index_select(fused_table, 0, fused_indices)
            .view(nbag, args.pool, args.dim)
            .sum(dim=1)
        )

    def tbe_fwd() -> torch.Tensor:
        return tbe(tbe_indices, offsets)

    def tbe_fwd_bwd() -> torch.Tensor:
        output = tbe(tbe_indices, offsets)
        output.backward(grad)
        return output

    raw_time = time_loop(raw_gather, args.iters, args.warmup)
    pooled_time = time_loop(pooled_gather, args.iters, args.warmup)
    tbe_fwd_time = time_loop(tbe_fwd, args.iters, args.warmup)
    tbe_fb_time = time_loop(tbe_fwd_bwd, args.iters, args.warmup)

    raw_bw = (
        raw_gather_bytes_per_sample(args.num_tables, args.dim, args.pool)
        * args.batch
        / raw_time
        / 1e9
    )
    pooled_bw = (
        pooled_fwd_bytes_per_sample(args.num_tables, args.dim, args.pool)
        * args.batch
        / pooled_time
        / 1e9
    )
    tbe_fwd_bw = (
        pooled_fwd_bytes_per_sample(args.num_tables, args.dim, args.pool)
        * args.batch
        / tbe_fwd_time
        / 1e9
    )
    tbe_fb_bw = (
        tbe_fwd_bwd_bytes_per_sample(args.num_tables, args.dim, args.pool)
        * args.batch
        / tbe_fb_time
        / 1e9
    )

    print(f"GPU: {name} | peak {args.peak_gbps:.0f} GB/s")
    print(
        f"workload: {args.num_tables}x{args.rows}x{args.dim}, "
        f"bs={args.batch}, pool={args.pool}, ids={fused_indices.numel()}"
    )
    print("== gather vs FBGEMM TBE ==")
    print(
        f"raw index_select       {raw_time * 1e3:7.2f} ms  {raw_bw:8.1f} GB/s  "
        f"({100 * raw_bw / args.peak_gbps:5.1f}% peak)"
    )
    print(
        f"pooled index_select    {pooled_time * 1e3:7.2f} ms  {pooled_bw:8.1f} GB/s  "
        f"({100 * pooled_bw / args.peak_gbps:5.1f}% peak)"
    )
    print(
        f"FBGEMM TBE fwd         {tbe_fwd_time * 1e3:7.2f} ms  "
        f"{tbe_fwd_bw:8.1f} GB/s  ({100 * tbe_fwd_bw / args.peak_gbps:5.1f}% peak)"
    )
    print(
        f"FBGEMM TBE fwd+bwd     {tbe_fb_time * 1e3:7.2f} ms  "
        f"{tbe_fb_bw:8.1f} GB/s  ({100 * tbe_fb_bw / args.peak_gbps:5.1f}% peak)"
    )
    print(f"TBE fwd speedup vs raw gather: {raw_time / tbe_fwd_time:.2f}x wall-time")
    print(
        "TBE fwd speedup vs pooled index_select: "
        f"{pooled_time / tbe_fwd_time:.2f}x wall-time"
    )


if __name__ == "__main__":
    main()
