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

"""Single-GPU raw gather bandwidth bench.

This is the GPU analogue of a plain JAX `take`/gather, not FBGEMM TBE and not a
fused embedding-bag kernel. It fuses the benchmark's logical tables into one
`[num_tables * rows, dim]` table and times:

  raw gather:    output = table.index_select(0, flat_indices)
  pooled gather: output = table.index_select(...).view(nbag, pool, dim).sum(1)

The raw gather BW accounts for one D-wide row read and one D-wide output write
per id. The pooled path uses the same fwd bytes as the embedding-bag benchmarks:
K row reads plus one pooled output write per bag.

Run:
  CUDA_VISIBLE_DEVICES=0 python bench_gpu_gather.py --batches 4096 --pools 100
"""

import argparse
from typing import Callable

import torch

_A100_PEAK_GBPS = 2039.0


def raw_gather_bytes_per_sample(num_tables: int, dim: int, pool: int) -> int:
    return num_tables * pool * 2 * dim * 4


def pooled_gather_bytes_per_sample(num_tables: int, dim: int, pool: int) -> int:
    return num_tables * (pool + 1) * dim * 4


def _time_loop(fn: Callable[[], torch.Tensor], iters: int, warmup: int) -> float:
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
    # Touch the result after timing so the function cannot be treated as dead.
    result.record_stream(torch.cuda.current_stream())
    return start.elapsed_time(end) / iters / 1e3


def make_indices(
    vocab_size: int,
    num_tables: int,
    batch: int,
    pool: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    total = batch * num_tables * pool
    if mode == "random":
        return torch.randint(0, vocab_size, (total,), dtype=torch.long, device=device)
    if mode == "sequential":
        return torch.arange(total, dtype=torch.long, device=device) % vocab_size
    if mode == "strided":
        return (torch.arange(total, dtype=torch.long, device=device) * 257) % vocab_size
    raise ValueError(f"unknown indices mode: {mode}")


def bench(
    num_tables: int,
    rows: int,
    dim: int,
    batch: int,
    pool: int,
    indices_mode: str,
    iters: int,
    warmup: int,
    peak: float,
    device: torch.device,
) -> None:
    vocab_size = num_tables * rows
    table = torch.empty((vocab_size, dim), dtype=torch.float32, device=device)
    table.normal_(mean=0.0, std=1.0 / dim**0.5)
    indices = make_indices(vocab_size, num_tables, batch, pool, indices_mode, device)
    nbag = batch * num_tables

    def raw_gather() -> torch.Tensor:
        return torch.index_select(table, 0, indices)

    def pooled_gather() -> torch.Tensor:
        return torch.index_select(table, 0, indices).view(nbag, pool, dim).sum(dim=1)

    raw_time = _time_loop(raw_gather, iters, warmup)
    pooled_time = _time_loop(pooled_gather, iters, warmup)
    raw_bytes = raw_gather_bytes_per_sample(num_tables, dim, pool)
    pooled_bytes = pooled_gather_bytes_per_sample(num_tables, dim, pool)
    raw_bw = raw_bytes * batch / raw_time / 1e9
    pooled_bw = pooled_bytes * batch / pooled_time / 1e9
    print(
        f"  T={num_tables} rows={rows // 1000}K D={dim} bs={batch:5d} "
        f"pool={pool:3d} idx={indices_mode:10s} | "
        f"raw {raw_time * 1e3:6.2f}ms {raw_bw:7.0f} GB/s "
        f"({100 * raw_bw / peak:4.1f}%) | "
        f"pooled {pooled_time * 1e3:6.2f}ms {pooled_bw:7.0f} GB/s "
        f"({100 * pooled_bw / peak:4.1f}%)",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tables", type=int, default=16)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batches", type=int, nargs="+", default=[512, 4096])
    parser.add_argument("--pools", type=int, nargs="+", default=[1, 16, 64, 100])
    parser.add_argument(
        "--indices",
        choices=["random", "sequential", "strided"],
        default="random",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--peak-gbps", type=float, default=_A100_PEAK_GBPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    total_gb = args.num_tables * args.rows * args.dim * 4 / 1e9
    print(f"GPU: {name} | peak {args.peak_gbps:.0f} GB/s")
    print(
        f"workload: fused table {args.num_tables}x{args.rows}x{args.dim} "
        f"f32 = {total_gb:.1f} GB; raw gather is not pooled embedding/TBE"
    )
    print("== torch.index_select raw gather / pooled gather ==")
    for batch in args.batches:
        for pool in args.pools:
            bench(
                args.num_tables,
                args.rows,
                args.dim,
                batch,
                pool,
                args.indices,
                args.iters,
                args.warmup,
                args.peak_gbps,
                device,
            )


if __name__ == "__main__":
    main()
