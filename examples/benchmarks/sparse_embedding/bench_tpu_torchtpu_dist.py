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

"""Distributed TorchTPU SparseCore benchmark with CPU-only JTE preprocessing.

Launch with one TorchTPU process per TPU core. JTE is used only on CPU to build
the SparseCore CSR inputs; the TPU path is the TorchTPU direct SparseCore op.
"""

import argparse
import dataclasses
import os
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import torch
import torch.distributed as dist
from jax_tpu_embedding.sparsecore.lib.nn import (
    embedding as E,
    embedding_spec as S,
    table_stacking as TS,
)
from csr_utils import sanitize_minibatched_row_pointers

_V7X_HBM_PEAK_GBPS = 7400.0


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def bytes_per_sample(
    num_tables: int, dim: int, pool: int, optimizer: str
) -> tuple[int, int]:
    fwd = (pool + 1) * dim * 4
    if optimizer == "adagrad":
        bwd = dim * 4 + pool * (4 * dim * 4)
    else:
        bwd = dim * 4 + pool * (2 * dim * 4 + 2 * 4)
    return num_tables * fwd, num_tables * (fwd + bwd)


def build_specs(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    vocab_size = args.num_tables * args.rows
    effective_batch = args.num_tables * args.batch
    table_spec = S.TableSpec(
        vocabulary_size=vocab_size,
        embedding_dim=args.dim,
        initializer=jax.nn.initializers.ones,
        optimizer=S.AdagradOptimizerSpec(
            learning_rate=args.learning_rate,
            initial_accumulator_value=args.initial_accumulator,
        ),
        combiner="sum",
        name="t0",
        max_ids_per_partition=args.sc_limit,
        max_unique_ids_per_partition=args.sc_limit,
    )
    specs = {
        "f0": S.FeatureSpec(
            name="f0",
            table_spec=table_spec,
            input_shape=(effective_batch, args.pool),
            output_shape=(effective_batch, args.dim),
        )
    }
    TS.stack_tables(
        specs,
        table_names=["t0"],
        global_device_count=world_size,
        num_sc_per_device=args.num_sc_per_device,
        stack_name="t0",
    )
    override_stacked_limits(specs, args.sc_limit)
    return specs


def override_stacked_limits(specs: dict[str, Any], sc_limit: int) -> None:
    for feature_spec in specs.values():
        stacked = feature_spec.table_spec.stacked_table_spec
        feature_spec.table_spec.stacked_table_spec = dataclasses.replace(
            stacked,
            max_ids_per_partition=sc_limit,
            max_unique_ids_per_partition=sc_limit,
        )


def make_rank_csr(
    args: argparse.Namespace,
    specs: dict[str, Any],
    rank: int,
    world_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    effective_batch = args.num_tables * args.batch
    vocab_size = args.num_tables * args.rows
    ids = (
        np.random.randint(
            0,
            vocab_size,
            size=(effective_batch, args.pool),
            dtype=np.int32,
        )
        if not args.deterministic_ids
        else (
            np.arange(effective_batch * args.pool, dtype=np.int32).reshape(
                effective_batch, args.pool
            )
            * 7
        )
        % vocab_size
    )
    csr, stats = E.preprocess_sparse_dense_matmul_input(
        {"f0": ids},
        None,
        specs,
        world_size,
        world_size,
        num_sc_per_device=args.num_sc_per_device,
        sharding_strategy="MOD",
        enable_minibatching=True,
        allow_id_dropping=args.allow_id_dropping,
    )
    stack_name = "t0"

    def chunk(array: Any) -> np.ndarray:
        arr = np.asarray(array)
        per_rank = arr.shape[0] // world_size
        return arr[rank * per_rank : (rank + 1) * per_rank].copy()

    row_pointers = chunk(csr.lhs_row_pointers[stack_name])
    num_minibatches = int(np.max(np.asarray(csr.num_minibatches)))
    row_pointers = sanitize_minibatched_row_pointers(
        row_pointers,
        np.asarray(stats.max_ids_per_partition[stack_name])[
            rank * args.num_sc_per_device : (rank + 1) * args.num_sc_per_device
        ],
        np.array([num_minibatches], dtype=np.int32),
        row_pointers_per_sparsecore_per_minibatch=16,
    )
    return (
        row_pointers,
        chunk(csr.lhs_embedding_ids[stack_name]),
        chunk(csr.lhs_sample_ids[stack_name]),
        chunk(csr.lhs_gains[stack_name]),
        num_minibatches,
        args.sc_limit,
        args.sc_limit,
    )


def sync() -> None:
    tpu_mod = getattr(torch, "tpu", None)
    if tpu_mod is not None and hasattr(tpu_mod, "synchronize"):
        tpu_mod.synchronize()
    else:
        torch.empty((), device=torch.device("tpu")).cpu()


def time_loop(fn: Any, iters: int, warmup: int) -> float:
    fn()
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tables", type=int, default=16)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--pool", type=int, default=100)
    parser.add_argument("--sc-limit", type=int, default=1024)
    parser.add_argument("--num-sc-per-device", type=int, default=2)
    parser.add_argument(
        "--optimizer",
        choices=["adagrad", "rowwise_adagrad"],
        default="adagrad",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--initial-accumulator", type=float, default=0.1)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Full Adagrad default matches JTE's AdagradOptimizerSpec.",
    )
    parser.add_argument("--allow-id-dropping", action="store_true")
    parser.add_argument("--deterministic-ids", action="store_true")
    parser.add_argument(
        "--sharded-table",
        action="store_true",
        help="Use the per-rank table shard shape that JTE uses instead of a replicated global table.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bwd", action="store_true")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--peak-gbps", type=float, default=_V7X_HBM_PEAK_GBPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    np.random.seed(args.seed)

    specs = build_specs(args, world_size)
    (
        row_pointers_np,
        embedding_ids_np,
        sample_ids_np,
        gains_np,
        num_minibatches_value,
        max_ids,
        max_unique_ids,
    ) = make_rank_csr(args, specs, rank, world_size)

    import torch_tpu  # noqa: F401

    device = torch.device("tpu")
    row_pointers = torch.tensor(row_pointers_np, device=device)
    embedding_ids = torch.tensor(embedding_ids_np, device=device)
    sample_ids = torch.tensor(sample_ids_np, device=device)
    gains = torch.tensor(gains_np, device=device)
    num_minibatches = torch.tensor(
        num_minibatches_value, dtype=torch.int32, device=device
    )

    effective_batch = args.num_tables * args.batch
    local_batch = _ceil_div(effective_batch, world_size)
    vocab_size = args.num_tables * args.rows
    local_vocab_size = _ceil_div(vocab_size, world_size)
    table_rows = local_vocab_size if args.sharded_table else vocab_size
    table = torch.ones((table_rows, args.dim), dtype=torch.float32, device=device)
    if not args.smoke:
        table = torch.empty((table_rows, args.dim), dtype=torch.float32, device=device)
    activations_grad = torch.ones(
        (local_batch, args.dim), dtype=torch.float32, device=device
    )
    learning_rate = torch.tensor(args.learning_rate, dtype=torch.float32, device=device)
    epsilon = torch.tensor(args.epsilon, dtype=torch.float32, device=device)

    def fwd() -> torch.Tensor:
        return torch.ops.tpu.sparse_dense_matmul_with_minibatching(
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            num_minibatches,
            table,
            device_batch_size=local_batch,
            max_ids_per_partition=max_ids,
            max_unique_ids_per_partition=max_unique_ids,
        )

    if args.smoke:
        out = fwd().cpu().numpy()
        ok = bool(np.allclose(out, np.full_like(out, args.pool)))
        print(f"RANK {rank} smoke_fwd_ok={ok}", flush=True)

    grad_op = getattr(
        torch.ops.tpu,
        f"sparse_dense_matmul_grad_with_{args.optimizer}_with_minibatching",
    )
    accumulator = (
        torch.full_like(table, args.initial_accumulator)
        if args.optimizer == "adagrad"
        else torch.full(
            (table_rows,),
            args.initial_accumulator,
            dtype=torch.float32,
            device=device,
        )
    )

    def fwd_bwd() -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        forward_output = fwd()
        if args.optimizer == "adagrad":
            updated = grad_op(
                row_pointers,
                embedding_ids,
                sample_ids,
                gains,
                num_minibatches,
                table,
                accumulator,
                activations_grad,
                learning_rate,
                device_batch_size=local_batch,
                max_ids_per_partition=max_ids,
                max_unique_ids_per_partition=max_unique_ids,
            )
        else:
            updated = grad_op(
                row_pointers,
                embedding_ids,
                sample_ids,
                gains,
                num_minibatches,
                table,
                accumulator,
                activations_grad,
                learning_rate,
                epsilon,
                device_batch_size=local_batch,
                max_ids_per_partition=max_ids,
                max_unique_ids_per_partition=max_unique_ids,
            )
        return forward_output, updated

    if args.smoke and args.bwd:
        _, (updated_table, updated_accumulator) = fwd_bwd()
        print(
            f"RANK {rank} smoke_bwd_shapes="
            f"{tuple(updated_table.shape)},{tuple(updated_accumulator.shape)}",
            flush=True,
        )

    dist.barrier()
    fwd_time = time_loop(fwd, args.iters, args.warmup)
    b_fwd, b_fb = bytes_per_sample(args.num_tables, args.dim, args.pool, args.optimizer)
    local_batch_for_bytes = args.batch / world_size
    local_bw = b_fwd * local_batch_for_bytes / fwd_time / 1e9
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, fwd_time)

    line = ""
    if rank == 0:
        mean_fwd = float(np.mean(np.asarray(gathered, dtype=np.float64)))
        agg_bw = b_fwd * args.batch / mean_fwd / 1e9
        line = (
            "TorchTPU SC direct distributed | "
            f"optimizer={args.optimizer} world={world_size} "
            f"bs={args.batch} effective_batch={effective_batch} pool={args.pool} | "
            f"fwd {mean_fwd * 1e3:.2f}ms agg {agg_bw:.1f} "
            f"/rank {local_bw:.1f} GB/s"
        )

    if args.bwd:
        fb_time = time_loop(fwd_bwd, args.iters, args.warmup)
        gathered_fb: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_fb, fb_time)
        if rank == 0:
            mean_fb = float(np.mean(np.asarray(gathered_fb, dtype=np.float64)))
            agg_fb = b_fb * args.batch / mean_fb / 1e9
            line += f" | fwd+bwd {mean_fb * 1e3:.2f}ms agg {agg_fb:.1f} GB/s"

    if rank == 0:
        print(line, flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
