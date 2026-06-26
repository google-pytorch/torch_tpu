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

"""Raw SparseCore Pallas row-gather benchmark.

This is intentionally lower level than FBGEMM TBE/JTE/TorchTPU embedding-bag
benchmarks. It times a SparseCore Pallas gather of D-wide rows:

    output[i, :] = table[indices[i], :]

It does not pool multiple IDs per sample and does not run an optimizer update.
Use it to measure whether a hand-written Pallas SC gather can move table rows at
high bandwidth. Do not compare it directly to pooled TBE fwd/fwd+bwd numbers
unless the byte model is adjusted for the different output shape.
"""

import argparse
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax._src.pallas.mosaic import sc_core
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

_V7X_HBM_PEAK_GBPS = 7400.0


def _time_loop(fn: Any, iters: int, warmup: int) -> float:
    result = None
    for _ in range(warmup):
        result = fn()
    jax.block_until_ready(result)
    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn()
    jax.block_until_ready(result)
    return (time.perf_counter() - t0) / iters


def build_sc_gather(num_indices: int, dim: int, block_rows: int) -> Any:
    sc_info = plsc.get_sparse_core_info()
    mesh = sc_core.VectorSubcoreMesh(
        core_axis_name="core",
        num_cores=sc_info.num_cores,
        num_subcores=2,
        subcore_axis_name="subcore",
    )

    @jax.jit
    def gather(table: jax.Array, indices: jax.Array) -> jax.Array:
        flat_indices = indices.reshape((1, num_indices))

        @pl.kernel(
            out_shape=jax.ShapeDtypeStruct((num_indices, dim), table.dtype),
            mesh=mesh,
        )
        def kernel(table_hbm, indices_hbm, output_hbm):
            def body(indices_vmem, output_vmem):
                # This is the primitive pattern from JAX's
                # tpu_sparsecore_pallas_test.py::test_gather_with_emit.
                pltpu.sync_copy(table_hbm.at[indices_vmem.at[0]], output_vmem)

            pltpu.emit_pipeline(
                body,
                grid=(num_indices // block_rows,),
                in_specs=[
                    pl.BlockSpec((1, block_rows), lambda i: (0, i)),
                ],
                out_specs=[
                    pl.BlockSpec((block_rows, dim), lambda i: (i, 0)),
                ],
                core_axis_name="subcore",
                dimension_semantics=(pltpu.PARALLEL,),
            )(indices_hbm, output_hbm)

        return kernel(table, flat_indices)

    return gather


def build_sc_copy(num_indices: int, dim: int, block_rows: int) -> Any:
    sc_info = plsc.get_sparse_core_info()
    mesh = sc_core.VectorSubcoreMesh(
        core_axis_name="core",
        num_cores=sc_info.num_cores,
        num_subcores=2,
        subcore_axis_name="subcore",
    )

    @jax.jit
    def copy(table: jax.Array) -> jax.Array:
        @pl.kernel(
            out_shape=jax.ShapeDtypeStruct((num_indices, dim), table.dtype),
            mesh=mesh,
        )
        def kernel(table_hbm, output_hbm):
            def body(output_vmem):
                i = pl.program_id(0)
                pltpu.sync_copy(
                    table_hbm.at[pl.ds(i * block_rows, block_rows), pl.ds(0, dim)],
                    output_vmem,
                )

            pltpu.emit_pipeline(
                body,
                grid=(num_indices // block_rows,),
                in_specs=[],
                out_specs=[
                    pl.BlockSpec((block_rows, dim), lambda i: (i, 0)),
                ],
                core_axis_name="subcore",
                dimension_semantics=(pltpu.PARALLEL,),
            )(output_hbm)

        return kernel(table)

    return copy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-indices", type=int, default=65_536)
    parser.add_argument("--block-rows", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--index-mode",
        choices=["random", "sequential", "strided", "hot"],
        default="random",
        help="ID pattern used for the row gather.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=9973,
        help="Stride for --index-mode=strided.",
    )
    parser.add_argument(
        "--all-devices",
        action="store_true",
        help="Run one independent gather per visible JAX device with pmap.",
    )
    parser.add_argument(
        "--kernel-mode",
        choices=["gather", "copy"],
        default="gather",
        help="Use indexed gather or direct contiguous block copy.",
    )
    parser.add_argument("--peak-gbps", type=float, default=_V7X_HBM_PEAK_GBPS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.dim != 128:
        raise ValueError("this Pallas SC gather kernel currently assumes dim=128")
    if args.num_indices % args.block_rows != 0:
        raise ValueError("--num-indices must be divisible by --block-rows")

    sc_info = plsc.get_sparse_core_info()
    table = jnp.arange(args.rows * args.dim, dtype=jnp.float32).reshape(
        args.rows, args.dim
    )
    if args.index_mode == "random":
        indices = jax.random.randint(
            jax.random.key(args.seed),
            (args.num_indices,),
            minval=0,
            maxval=args.rows,
            dtype=jnp.int32,
        )
    elif args.index_mode == "sequential":
        indices = jnp.arange(args.num_indices, dtype=jnp.int32) % args.rows
    elif args.index_mode == "strided":
        indices = (
            jnp.arange(args.num_indices, dtype=jnp.int32) * args.stride
        ) % args.rows
    else:
        indices = jnp.zeros((args.num_indices,), dtype=jnp.int32)
    gather = build_sc_gather(args.num_indices, args.dim, args.block_rows)
    copy = build_sc_copy(args.num_indices, args.dim, args.block_rows)
    num_replicas = jax.device_count() if args.all_devices else 1
    if args.all_devices:
        mesh = Mesh(np.asarray(jax.devices()), ("replica",))
        table_arg = jax.device_put(
            jnp.broadcast_to(table, (num_replicas, args.rows, args.dim)),
            NamedSharding(mesh, P("replica", None, None)),
        )
        indices_arg = jax.device_put(
            jnp.broadcast_to(indices, (num_replicas, args.num_indices)),
            NamedSharding(mesh, P("replica", None)),
        )
        run_gather = jax.pmap(gather)
        run_copy = jax.pmap(copy)
    else:
        table_arg = table
        indices_arg = indices
        run_gather = gather
        run_copy = copy
    if args.kernel_mode == "copy":
        run_kernel = lambda: run_copy(table_arg)
    else:
        run_kernel = lambda: run_gather(table_arg, indices_arg)

    if args.check:
        actual = run_kernel()
        if args.all_devices:
            actual = actual[0]
        expected = table[: args.num_indices] if args.kernel_mode == "copy" else table[indices]
        np.testing.assert_allclose(
            np.asarray(actual[: min(args.num_indices, 1024)]),
            np.asarray(expected[: min(args.num_indices, 1024)]),
        )
        print("check_ok=True", flush=True)

    t = _time_loop(run_kernel, args.iters, args.warmup)
    # Raw gather/copy reads one D-wide row and writes one D-wide output row.
    logical_bytes = num_replicas * args.num_indices * args.dim * 4 * 2
    bw = logical_bytes / t / 1e9
    # TPU7x exposes two JAX devices per physical chip. Keep the single-device
    # case normalized to one full chip to avoid implying half-chip peak specs.
    physical_chips = max(1.0, num_replicas / 2.0)
    bw_per_physical_chip = bw / physical_chips
    print(
        "Pallas SC raw gather | "
        f"visible_devices={jax.device_count()} replicas={num_replicas} "
        f"sc_cores={sc_info.num_cores} "
        f"rows={args.rows} dim={args.dim} num_indices={args.num_indices} "
        f"block_rows={args.block_rows} index_mode={args.index_mode} "
        f"kernel_mode={args.kernel_mode}",
        flush=True,
    )
    print(
        f"RESULT gather {t * 1e3:7.2f}ms agg {bw:7.1f} GB/s "
        f"per_physical_chip {bw_per_physical_chip:7.1f} GB/s "
        f"({100 * bw_per_physical_chip / args.peak_gbps:4.1f}% of --peak-gbps/chip)",
        flush=True,
    )


if __name__ == "__main__":
    main()
