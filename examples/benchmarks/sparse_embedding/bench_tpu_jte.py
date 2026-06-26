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

"""TPU SparseCore embedding bench (jax_tpu_embedding) matched to bench_gpu_tbe.py.

Mirrors the FBGEMM TBE GPU bench. The GPU TBE fuses all N tables into ONE table
(one weight buffer + one kernel), so the SparseCore-equivalent is a SINGLE fused
table (vocab = num_tables*rows) with effective batch = num_tables*batch — which
is the single-stack path that SparseCore + minibatching handles cleanly (the
N-separate-stacks layout blows up per-call overhead / max-ids validation).

Forward = tpu_sparse_dense_matmul; fwd+bwd = + tpu_sparse_dense_matmul_grad with
fused Adagrad. Minibatching ON so large pool*batch fits the per-partition
max-ids (TileSpmem) budget. Identical bytes/sample accounting to the GPU bench.

One (batch, pool) config per invocation (core-halt isolation); sweep via shell.
"""

import argparse
import collections
import dataclasses
import functools
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax_tpu_embedding.sparsecore.lib.nn import (
    embedding as E,
    embedding_spec as S,
    table_stacking as TS,
)
from jax_tpu_embedding.sparsecore.utils import utils as U

_V7X_HBM_PEAK_GBPS = 7400.0


def _num_sc() -> int:
    try:
        return U.num_sparsecores_per_device()
    except Exception:
        return 2


def bytes_per_sample(num_tables: int, dim: int, pool: int) -> tuple[int, int]:
    fwd = (pool + 1) * dim * 4
    bwd = dim * 4 + pool * (4 * dim * 4)
    return num_tables * fwd, num_tables * (fwd + bwd)


def make_global_csr(csr_inputs: Any, mesh: jax.sharding.Mesh) -> tuple[Any, Any]:
    """Lift process-local JTE CSR arrays onto the benchmark device mesh."""
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
    repl_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def shard(x: Any) -> jax.Array:
        arr = np.asarray(x)
        sharding = data_sharding if arr.ndim > 0 else repl_sharding
        return jax.make_array_from_process_local_data(sharding, arr)

    def spec(x: Any) -> jax.sharding.PartitionSpec:
        return (
            jax.sharding.PartitionSpec("x")
            if np.asarray(x).ndim > 0
            else jax.sharding.PartitionSpec()
        )

    return jax.tree.map(shard, csr_inputs), jax.tree.map(spec, csr_inputs)


def build_sc_backward(
    csr_in_specs: Any,
    specs: dict[str, Any],
    feature_names: list[str],
    mesh: jax.sharding.Mesh,
    num_sc: int,
) -> Any:
    """Build SparseCore backward so the grad op sees local embedding shards."""

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            {name: jax.sharding.PartitionSpec("x", None) for name in feature_names},
            csr_in_specs,
            jax.sharding.PartitionSpec("x", None),
        ),
        out_specs=jax.sharding.PartitionSpec("x", None),
        check_vma=False,
    )
    def sc_bwd(
        act_grads_local: dict[str, jax.Array],
        csr_local: Any,
        emb_vars_local: Any,
    ) -> Any:
        return E.tpu_sparse_dense_matmul_grad(
            act_grads_local,
            csr_local,
            emb_vars_local,
            specs,
            sharding_strategy="MOD",
            num_sc_per_device=num_sc,
            enable_minibatching=True,
        )

    return sc_bwd


def build_feature_specs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], int]:
    """Build either the old one-feature model or production-style stacked tables."""
    if args.layout == "fused-table":
        vocab_size = args.num_tables * args.rows
        table_spec = S.TableSpec(
            vocabulary_size=vocab_size,
            embedding_dim=args.dim,
            initializer=jax.nn.initializers.truncated_normal(
                stddev=1.0 / args.dim**0.5
            ),
            optimizer=S.AdagradOptimizerSpec(
                learning_rate=0.1, initial_accumulator_value=0.1
            ),
            combiner="sum",
            name="t0",
            max_ids_per_partition=args.max_ids,
            max_unique_ids_per_partition=args.max_ids,
        )
        effective_batch = args.num_tables * args.batch
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
            effective_batch,
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
                learning_rate=0.1, initial_accumulator_value=0.1
            ),
            combiner="sum",
            name=name,
            max_ids_per_partition=args.max_ids,
            max_unique_ids_per_partition=args.max_ids,
        )
        specs[name] = S.FeatureSpec(
            name=name,
            table_spec=table_spec,
            input_shape=(args.batch, args.pool),
            output_shape=(args.batch, args.dim),
        )
    return specs, table_names, args.batch


def make_ids(
    args: argparse.Namespace, feature_names: list[str], vocab_size: int
) -> dict[str, Any]:
    return {
        name: np.random.randint(
            0,
            vocab_size,
            size=(
                args.batch
                if args.layout == "stacked-tables"
                else args.num_tables * args.batch,
                args.pool,
            ),
        ).astype(np.int32)
        for name in feature_names
    }


def override_stacked_limits(specs: dict[str, Any], max_ids: int) -> None:
    """Set stacked-table limits without inflating the COO buffer from autotune."""
    seen = set()
    for feature_spec in specs.values():
        stacked_table_spec = feature_spec.table_spec.stacked_table_spec
        stack_name = stacked_table_spec.stack_name
        if stack_name in seen:
            continue
        seen.add(stack_name)
        new_stacked_table_spec = dataclasses.replace(
            stacked_table_spec,
            max_ids_per_partition=max_ids,
            max_unique_ids_per_partition=max_ids,
        )
        for update_feature_spec in specs.values():
            if (
                update_feature_spec.table_spec.stacked_table_spec.stack_name
                == stack_name
            ):
                update_feature_spec.table_spec.stacked_table_spec = (
                    new_stacked_table_spec
                )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-tables", type=int, default=16)
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--pool", type=int, default=100)
    p.add_argument("--max-ids", type=int, default=1024)
    p.add_argument(
        "--override-stacked-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After table stacking, set the StackedTableSpec max-id limits to "
            "--max-ids without changing the COO buffer size."
        ),
    )
    p.add_argument(
        "--autotune-sc-limits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a sample preprocess pass to update stacked-table COO limits. "
            "Off by default because high-pool microbenchmarks can exceed the "
            "SparseCore scratchpad row limit after autotuning."
        ),
    )
    p.add_argument(
        "--allow-id-dropping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow JTE to drop ids beyond the configured per-partition limit.",
    )
    p.add_argument(
        "--layout",
        choices=["stacked-tables", "fused-table"],
        default="fused-table",
        help=(
            "stacked-tables keeps one feature/table per embedding table and stacks "
            "them into one SC stack; fused-table is the older one-feature 16M-row model."
        ),
    )
    p.add_argument("--bwd", action="store_true")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--peak-gbps", type=float, default=_V7X_HBM_PEAK_GBPS)
    args = p.parse_args()

    gd, ld, nsc = jax.device_count(), jax.local_device_count(), _num_sc()
    chips = max(gd // 2, 1)
    dim, pool, M = args.dim, args.pool, args.max_ids
    specs, table_names, effective_batch = build_feature_specs(args)
    vocab_size = (
        args.num_tables * args.rows if args.layout == "fused-table" else args.rows
    )
    feature_names = list(specs.keys())
    print(
        f"devices={gd} ({chips} chips) sc/dev={nsc} | layout={args.layout}: "
        f"{args.num_tables} tables x {args.rows} rows x {dim}, "
        f"effective_batch={effective_batch}, pool={pool}, max_ids={M}",
        flush=True,
    )

    TS.stack_tables(
        specs,
        table_names=table_names,
        global_device_count=gd,
        num_sc_per_device=nsc,
        stack_name="t0",
    )
    if args.override_stacked_limits:
        override_stacked_limits(specs, args.max_ids)
    devs = np.array(jax.devices())
    mesh = jax.sharding.Mesh(devs, ("x",))
    gsh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", None))
    tsp = E.get_table_specs(specs)
    stacks = collections.defaultdict(list)
    for t in jax.tree.leaves(tsp):
        stacks[t.setting_in_stack.stack_name].append(t)
    for s in stacks.values():
        s.sort(key=lambda t: t.setting_in_stack.row_offset_in_shard)
    ax = next(s for s in gsh.spec if s is not None)
    w = gsh.mesh.size * nsc
    rngs = jax.random.split(jax.random.PRNGKey(0), w * len(stacks))
    emb = {}
    for i, (sn, ss) in enumerate(stacks.items()):
        emb[sn] = E._init_stacked_embedding_table(
            rng=rngs[i * w : (i + 1) * w],
            stack_name=sn,
            table_specs=ss,
            global_sharding=gsh,
            sharding_axis=ax,
            num_sparsecore_per_device=nsc,
        )

    ids = make_ids(args, feature_names, vocab_size)
    if args.autotune_sc_limits:
        _, stats = E.preprocess_sparse_dense_matmul_input(
            ids,
            None,
            specs,
            ld,
            gd,
            num_sc_per_device=nsc,
            sharding_strategy="MOD",
            enable_minibatching=True,
            allow_id_dropping=True,
        )
        E.update_preprocessing_parameters(specs, stats, nsc)

    csr_local, stats = E.preprocess_sparse_dense_matmul_input(
        ids,
        None,
        specs,
        ld,
        gd,
        num_sc_per_device=nsc,
        sharding_strategy="MOD",
        enable_minibatching=True,
        allow_id_dropping=args.allow_id_dropping,
    )
    print(
        "preprocess_stats: "
        f"num_minibatches={np.asarray(csr_local.num_minibatches).tolist()} "
        f"max_ids_per_partition={jax.tree.map(lambda x: np.asarray(x).tolist(), stats.max_ids_per_partition)} "
        f"max_unique_ids_per_partition={jax.tree.map(lambda x: np.asarray(x).tolist(), stats.max_unique_ids_per_partition)}",
        flush=True,
    )
    csr, csr_in_specs = make_global_csr(csr_local, mesh)
    fwd = jax.jit(
        lambda c, e: E.tpu_sparse_dense_matmul(
            c,
            e,
            specs,
            global_device_count=ld,
            sharding_strategy="MOD",
            num_sc_per_device=nsc,
            enable_minibatching=True,
        )
    )

    def time_fn(fn, iters, warmup=5):
        for _ in range(warmup):
            r = fn()
        jax.block_until_ready(r)
        t0 = time.perf_counter()
        for _ in range(iters):
            r = fn()
        jax.block_until_ready(r)
        return (time.perf_counter() - t0) / iters

    b_fwd, b_fb = bytes_per_sample(args.num_tables, dim, pool)
    t_fwd = time_fn(lambda: fwd(csr, emb), args.iters)
    bw_fwd = b_fwd * args.batch / t_fwd / 1e9
    line = (
        f"RESULT bs={args.batch} pool={pool} | fwd {t_fwd * 1e3:7.2f}ms "
        f"agg {bw_fwd:7.1f} /chip {bw_fwd / chips:6.1f} GB/s "
        f"({100 * bw_fwd / chips / args.peak_gbps:4.1f}%)"
    )

    if args.bwd:
        grad_fn = build_sc_backward(csr_in_specs, specs, feature_names, mesh, nsc)

        def fb():
            a = fwd(csr, emb)
            act_grads = {name: jnp.ones_like(a[name]) for name in feature_names}
            return grad_fn(act_grads, csr, emb)

        t_fb = time_fn(fb, args.iters)
        bw_fb = b_fb * args.batch / t_fb / 1e9
        line += (
            f" | fwd+bwd {t_fb * 1e3:7.2f}ms agg {bw_fb:7.1f} "
            f"/chip {bw_fb / chips:6.1f} GB/s "
            f"({100 * bw_fb / chips / args.peak_gbps:4.1f}%)"
        )
    print(line, flush=True)


if __name__ == "__main__":
    main()
