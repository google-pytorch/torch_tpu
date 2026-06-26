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

"""TPU SparseCore embedding bench using the experimental TorchTPU SC ops.

This is the third path next to:
  - bench_gpu_tbe.py: FBGEMM TBE, fused RowWiseAdagrad
  - bench_tpu_jte.py: JTE SparseCore, fused full Adagrad

JTE is used only for table stacking and CSR input preprocessing. The timed path
calls torch.ops.tpu.sparse_dense_matmul directly, plus either the TorchTPU full
Adagrad or row-wise Adagrad optimizer-update op.

The default path uses the non-minibatching custom-call variants. With
--enable-minibatching, JTE's CSR preprocessing emits minibatched SparseCore
inputs and this benchmark calls the TorchTPU minibatching custom-call variants.
"""

import argparse
import collections
import dataclasses
import math
import time
from typing import Any

import jax
import numpy as np
import torch
from jax_tpu_embedding.sparsecore.lib.nn import (
    embedding as E,
    embedding_spec as S,
    table_stacking as TS,
)
from jax_tpu_embedding.sparsecore.utils import utils as U
from csr_utils import sanitize_minibatched_row_pointers

_V7X_HBM_PEAK_GBPS = 7400.0


def _num_sc() -> int:
    try:
        return U.num_sparsecores_per_device()
    except Exception:
        return 2


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def bytes_per_sample(
    num_tables: int, dim: int, pool: int, optimizer: str
) -> tuple[int, int]:
    fwd = (pool + 1) * dim * 4
    if optimizer == "adagrad":
        # Full Adagrad: read/write the D-wide weight row and D-wide accumulator.
        bwd = dim * 4 + pool * (4 * dim * 4)
    else:
        # RowWiseAdagrad: read/write the D-wide weight row and scalar slot.
        bwd = dim * 4 + pool * (2 * dim * 4 + 2 * 4)
    return num_tables * fwd, num_tables * (fwd + bwd)


def _to_torch_device(array: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), device=device)


def _sync() -> None:
    tpu_mod = getattr(torch, "tpu", None)
    if tpu_mod is not None and hasattr(tpu_mod, "synchronize"):
        tpu_mod.synchronize()
        return
    # Fallback for older torch_tpu builds: a tiny device-to-host copy blocks.
    torch.empty((), device=torch.device("tpu")).cpu()


def _time_loop(fn: Any, iters: int, warmup: int) -> float:
    result = None
    for _ in range(warmup):
        result = fn()
    _materialize(result)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn()
    _materialize(result)
    _sync()
    return (time.perf_counter() - t0) / iters


def _materialize(result: Any) -> None:
    if isinstance(result, torch.Tensor):
        # Keep the full result on device; synchronization is handled separately.
        return
    if isinstance(result, tuple):
        for item in result:
            _materialize(item)


def _torch_compile(fn: Any, enabled: bool) -> Any:
    if not enabled:
        return fn
    return torch.compile(fn, fullgraph=True)


def build_feature_specs(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    if args.layout == "fused-table":
        vocab_size = args.num_tables * args.rows
        effective_batch = args.num_tables * args.batch
        table_spec = S.TableSpec(
            vocabulary_size=vocab_size,
            embedding_dim=args.dim,
            initializer=jax.nn.initializers.truncated_normal(
                stddev=1.0 / args.dim**0.5
            ),
            optimizer=S.AdagradOptimizerSpec(
                learning_rate=args.learning_rate,
                initial_accumulator_value=args.initial_accumulator,
            ),
            combiner="sum",
            name="t0",
            max_ids_per_partition=args.sc_limit,
            max_unique_ids_per_partition=args.sc_limit,
        )
        return (
            {
                "f0": S.FeatureSpec(
                    name="f0",
                    table_spec=table_spec,
                    input_shape=(effective_batch, args.pool),
                    output_shape=(effective_batch, args.dim),
                )
            },
            ["t0"],
        )

    specs = {}
    table_names = []
    for table_idx in range(args.num_tables):
        name = f"t{table_idx}"
        table_names.append(name)
        table_spec = S.TableSpec(
            vocabulary_size=args.rows,
            embedding_dim=args.dim,
            initializer=jax.nn.initializers.truncated_normal(
                stddev=1.0 / args.dim**0.5
            ),
            optimizer=S.AdagradOptimizerSpec(
                learning_rate=args.learning_rate,
                initial_accumulator_value=args.initial_accumulator,
            ),
            combiner="sum",
            name=name,
            max_ids_per_partition=args.sc_limit,
            max_unique_ids_per_partition=args.sc_limit,
        )
        specs[name] = S.FeatureSpec(
            name=name,
            table_spec=table_spec,
            input_shape=(args.batch, args.pool),
            output_shape=(args.batch, args.dim),
        )
    return specs, table_names


def override_stacked_spec(
    specs: dict[str, Any], sc_limit: int, coo_buffer_size: int
) -> None:
    seen = set()
    for feature_spec in specs.values():
        stacked_table_spec = feature_spec.table_spec.stacked_table_spec
        stack_name = stacked_table_spec.stack_name
        if stack_name in seen:
            continue
        seen.add(stack_name)
        replacements = {
            "max_ids_per_partition": sc_limit,
            "max_unique_ids_per_partition": sc_limit,
        }
        if coo_buffer_size > 0:
            replacements["suggested_coo_buffer_size_per_device"] = coo_buffer_size
        new_stacked_table_spec = dataclasses.replace(
            stacked_table_spec,
            **replacements,
        )
        for update_feature_spec in specs.values():
            if (
                update_feature_spec.table_spec.stacked_table_spec.stack_name
                == stack_name
            ):
                update_feature_spec.table_spec.stacked_table_spec = (
                    new_stacked_table_spec
                )


def make_ids(args: argparse.Namespace, feature_names: list[str]) -> dict[str, Any]:
    if args.layout == "fused-table":
        return {
            "f0": np.random.randint(
                0,
                args.num_tables * args.rows,
                size=(args.num_tables * args.batch, args.pool),
                dtype=np.int32,
            )
        }
    return {
        name: np.random.randint(
            0,
            args.rows,
            size=(args.batch, args.pool),
            dtype=np.int32,
        )
        for name in feature_names
    }


def get_single_stack(specs: dict[str, Any]) -> str:
    tsp = E.get_table_specs(specs)
    stacks = collections.defaultdict(list)
    for table_spec in jax.tree.leaves(tsp):
        stacks[table_spec.setting_in_stack.stack_name].append(table_spec)
    if len(stacks) != 1:
        raise ValueError(f"expected one stacked SparseCore table, got {list(stacks)}")
    return next(iter(stacks))


def make_torch_inputs(
    args: argparse.Namespace,
    specs: dict[str, Any],
    stack_name: str,
    device: torch.device,
    num_sc: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    int,
    int,
]:
    ids = make_ids(args, list(specs.keys()))
    csr, stats = E.preprocess_sparse_dense_matmul_input(
        ids,
        None,
        specs,
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=num_sc,
        sharding_strategy="MOD",
        enable_minibatching=args.enable_minibatching,
        allow_id_dropping=args.allow_id_dropping,
    )
    max_ids = int(np.max(stats.max_ids_per_partition[stack_name]))
    max_unique_ids = int(np.max(stats.max_unique_ids_per_partition[stack_name]))
    if args.enable_minibatching:
        max_ids = args.sc_limit
        max_unique_ids = args.sc_limit
    if args.op_max_ids > 0:
        max_ids = args.op_max_ids
    if args.op_max_unique_ids > 0:
        max_unique_ids = args.op_max_unique_ids
    row_pointers_np = np.asarray(csr.lhs_row_pointers[stack_name]).copy()
    if args.enable_minibatching:
        row_pointers_np = sanitize_minibatched_row_pointers(
            row_pointers_np,
            stats.max_ids_per_partition[stack_name],
            csr.num_minibatches,
        )
    row_pointers = _to_torch_device(row_pointers_np, device)
    embedding_ids = _to_torch_device(csr.lhs_embedding_ids[stack_name], device)
    sample_ids = _to_torch_device(csr.lhs_sample_ids[stack_name], device)
    gains = _to_torch_device(csr.lhs_gains[stack_name], device)
    num_minibatches = None
    if args.enable_minibatching:
        csr_minibatches = getattr(csr, "num_minibatches", None)
        if csr_minibatches is None:
            raise RuntimeError("JTE preprocessing did not return num_minibatches")
        if isinstance(csr_minibatches, dict):
            csr_minibatches = csr_minibatches[stack_name]
        num_minibatches = torch.tensor(
            int(np.max(np.asarray(csr_minibatches))),
            dtype=torch.int32,
            device=device,
        )
    return (
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        num_minibatches,
        max(max_ids, 1),
        max(max_unique_ids, 1),
    )


def assert_torchtpu_ops(optimizer: str, enable_minibatching: bool) -> None:
    suffix = "_with_minibatching" if enable_minibatching else ""
    missing = [
        name
        for name in [
            f"sparse_dense_matmul{suffix}",
            f"sparse_dense_matmul_grad_with_{optimizer}{suffix}",
        ]
        if not hasattr(torch.ops.tpu, name)
    ]
    if missing:
        raise RuntimeError(
            "TorchTPU SparseCore op(s) are not registered: "
            + ", ".join(missing)
            + ". Install/run a torch_tpu build that includes the experimental "
            "sparse_dense_matmul optimizer-update ops."
        )


def bench(args: argparse.Namespace) -> None:  # noqa: C901
    try:
        import torch_tpu  # noqa: F401
    except Exception as error:
        raise RuntimeError("failed to import torch_tpu") from error

    assert_torchtpu_ops(args.optimizer, args.enable_minibatching)
    device = torch.device("tpu")
    num_sc = _num_sc()
    effective_batch = args.num_tables * args.batch
    total_ids = effective_batch * args.pool
    if args.sc_limit <= 0:
        args.sc_limit = (
            1024 if args.enable_minibatching else max(1, _ceil_div(total_ids, num_sc))
        )
    coo_buffer_size = (
        args.coo_buffer_size
        if args.coo_buffer_size > 0
        else (
            0
            if args.enable_minibatching
            else int(math.ceil(total_ids * args.coo_slack))
        )
    )

    specs, table_names = build_feature_specs(args)
    TS.stack_tables(
        specs,
        table_names=table_names,
        global_device_count=1,
        num_sc_per_device=num_sc,
        stack_name="t0",
    )
    override_stacked_spec(specs, args.sc_limit, coo_buffer_size)
    stack_name = get_single_stack(specs)
    (
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        num_minibatches,
        op_max_ids,
        op_max_unique_ids,
    ) = make_torch_inputs(args, specs, stack_name, device, num_sc)

    vocab_size = args.num_tables * args.rows
    table = torch.empty((vocab_size, args.dim), dtype=torch.float32, device=device)
    if args.optimizer == "adagrad":
        accumulator = torch.full_like(table, args.initial_accumulator)
    else:
        accumulator = torch.full(
            (vocab_size,),
            args.initial_accumulator,
            dtype=torch.float32,
            device=device,
        )
    activations_grad = torch.ones(
        (effective_batch, args.dim), dtype=torch.float32, device=device
    )
    learning_rate = torch.tensor(args.learning_rate, dtype=torch.float32, device=device)
    epsilon = torch.tensor(args.epsilon, dtype=torch.float32, device=device)

    def fwd_raw() -> torch.Tensor:
        if args.enable_minibatching:
            if num_minibatches is None:
                raise RuntimeError("missing num_minibatches")
            return torch.ops.tpu.sparse_dense_matmul_with_minibatching(
                row_pointers,
                embedding_ids,
                sample_ids,
                gains,
                num_minibatches,
                table,
                device_batch_size=effective_batch,
                max_ids_per_partition=op_max_ids,
                max_unique_ids_per_partition=op_max_unique_ids,
            )
        return torch.ops.tpu.sparse_dense_matmul(
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            table,
            device_batch_size=effective_batch,
            max_ids_per_partition=op_max_ids,
            max_unique_ids_per_partition=op_max_unique_ids,
        )

    grad_suffix = "_with_minibatching" if args.enable_minibatching else ""
    grad_op = getattr(
        torch.ops.tpu,
        f"sparse_dense_matmul_grad_with_{args.optimizer}{grad_suffix}",
    )

    def fwd_bwd_raw() -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        forward_output = fwd_raw()
        if args.enable_minibatching:
            if num_minibatches is None:
                raise RuntimeError("missing num_minibatches")
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
                    device_batch_size=effective_batch,
                    max_ids_per_partition=op_max_ids,
                    max_unique_ids_per_partition=op_max_unique_ids,
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
                    device_batch_size=effective_batch,
                    max_ids_per_partition=op_max_ids,
                    max_unique_ids_per_partition=op_max_unique_ids,
                )
            return forward_output, updated
        if args.optimizer == "adagrad":
            updated = grad_op(
                row_pointers,
                embedding_ids,
                sample_ids,
                gains,
                table,
                accumulator,
                activations_grad,
                learning_rate,
                device_batch_size=effective_batch,
                max_ids_per_partition=op_max_ids,
                max_unique_ids_per_partition=op_max_unique_ids,
            )
        else:
            updated = grad_op(
                row_pointers,
                embedding_ids,
                sample_ids,
                gains,
                table,
                accumulator,
                activations_grad,
                learning_rate,
                epsilon,
                device_batch_size=effective_batch,
                max_ids_per_partition=op_max_ids,
                max_unique_ids_per_partition=op_max_unique_ids,
            )
        return forward_output, updated

    fwd = _torch_compile(fwd_raw, args.compile)
    fwd_bwd = _torch_compile(fwd_bwd_raw, args.compile)

    b_fwd, b_fb = bytes_per_sample(args.num_tables, args.dim, args.pool, args.optimizer)
    print(
        "TorchTPU SC direct ops | "
        f"optimizer={args.optimizer} layout={args.layout} "
        f"minibatching={args.enable_minibatching} "
        f"sc/dev={num_sc} effective_batch={effective_batch} pool={args.pool} "
        f"sc_limit={args.sc_limit} coo={coo_buffer_size} "
        f"op_max_ids={op_max_ids} op_max_unique_ids={op_max_unique_ids}",
        flush=True,
    )
    print(
        f"CSR shapes: row_pointers={tuple(row_pointers.shape)} "
        f"embedding_ids={tuple(embedding_ids.shape)} "
        f"sample_ids={tuple(sample_ids.shape)} gains={tuple(gains.shape)}",
        flush=True,
    )
    if num_minibatches is not None:
        print(
            f"num_minibatches_per_sparse_core={int(num_minibatches.cpu())}", flush=True
        )

    t_fwd = _time_loop(fwd, args.iters, args.warmup)
    bw_fwd = b_fwd * args.batch / t_fwd / 1e9
    line = (
        f"RESULT bs={args.batch} pool={args.pool} | "
        f"fwd {t_fwd * 1e3:7.2f}ms {bw_fwd:7.1f} GB/s/chip "
        f"({100 * bw_fwd / args.peak_gbps:4.1f}%)"
    )
    if args.bwd:
        t_fb = _time_loop(fwd_bwd, args.iters, args.warmup)
        bw_fb = b_fb * args.batch / t_fb / 1e9
        line += (
            f" | fwd+bwd {t_fb * 1e3:7.2f}ms {bw_fb:7.1f} GB/s/chip "
            f"({100 * bw_fb / args.peak_gbps:4.1f}%)"
        )
    print(line, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-tables", type=int, default=16)
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--pool", type=int, default=100)
    p.add_argument(
        "--layout",
        choices=["stacked-tables", "fused-table"],
        default="fused-table",
    )
    p.add_argument(
        "--optimizer",
        choices=["adagrad", "rowwise_adagrad"],
        default="adagrad",
        help=(
            "adagrad is the full elementwise slot update matching JTE; "
            "rowwise_adagrad is the TorchTPU PR op with a 1-D accumulator."
        ),
    )
    p.add_argument(
        "--sc-limit",
        type=int,
        default=0,
        help=(
            "JTE preprocessing max_ids/max_unique_ids per partition. "
            "0 uses ceil(num_tables*batch*pool/num_sparsecores), or 1024 with "
            "--enable-minibatching."
        ),
    )
    p.add_argument(
        "--coo-buffer-size",
        type=int,
        default=0,
        help="StackedTableSpec suggested COO buffer size per device; 0 auto-sizes.",
    )
    p.add_argument("--coo-slack", type=float, default=1.25)
    p.add_argument("--op-max-ids", type=int, default=0)
    p.add_argument("--op-max-unique-ids", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--initial-accumulator", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--allow-id-dropping", action="store_true")
    p.add_argument("--enable-minibatching", action="store_true")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bwd", action="store_true")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--peak-gbps", type=float, default=_V7X_HBM_PEAK_GBPS)
    args = p.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    bench(args)


if __name__ == "__main__":
    main()
