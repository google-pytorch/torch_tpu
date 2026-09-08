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

"""Repro for a logical CPU export mesh in TorchTPU workers.

This script is intentionally safe to run in TorchTPU workers: it configures JAX
to use CPU before importing JAX, creates `WORLD_SIZE` logical CPU devices, and
exports shard_map programs for TPU without initializing JAX's TPU backend.

The CPU devices are placeholders, not a second runtime mesh. TorchTPU selects a
partition using the physical partition ID, which need not equal the Torch
global rank. This script prints both identities so that the otherwise implicit
ordering contract is visible.

Export-only smoke test on one process:

  python3 tests/pallas/jax_cpu_mesh_multihost_repro.py --world-size=16

Single-host torchrun smoke test, export only:

  torchrun --standalone --nproc_per_node=8 \\
    tests/pallas/jax_cpu_mesh_multihost_repro.py

Single-host TorchTPU smoke test, including `tpu_dist` all-reduce:

  python3 tests/pallas/jax_cpu_mesh_multihost_repro.py \\
    --launch-singlehost=8 --check-torchtpu

Multihost shape. Run this on every host after the pod/job has populated
`TPU_WORKER_HOSTNAMES` and `JOB_COMPLETION_INDEX`:

  eval "$(python3 -m torch_tpu._internal.distributed.launchers.multihost_wrapper)"
  torchrun \\
    --nnodes="$NNODES" \\
    --node_rank="$NODE_RANK" \\
    --nproc_per_node="${LOCAL_WORLD_SIZE:-8}" \\
    --master_addr="$MASTER_ADDR" \\
    --master_port="$MASTER_PORT" \\
    tests/pallas/jax_cpu_mesh_multihost_repro.py --check-torchtpu
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from typing import Any

_CPU_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"


def _configure_jax_cpu_export_env(world_size: int) -> None:
  if world_size < 1:
    raise ValueError(f"world_size must be positive, got {world_size}")
  configured = os.environ.get("JAX_PLATFORMS")
  if configured is not None and configured.strip().lower() != "cpu":
    raise RuntimeError(
        "This repro requires JAX_PLATFORMS=cpu before importing JAX; got"
        f" {configured!r}."
    )
  if configured is None and "jax" in sys.modules:
    raise RuntimeError(
        "JAX was imported before JAX_PLATFORMS=cpu was configured."
    )
  os.environ["JAX_PLATFORMS"] = "cpu"

  xla_flags = os.environ.get("XLA_FLAGS", "")
  pattern = re.compile(
      rf"(?:^|\s){re.escape(_CPU_DEVICE_COUNT_FLAG)}(?:=|\s+)(\d+)(?=\s|$)"
  )
  configured_counts = [int(value) for value in pattern.findall(xla_flags)]
  if configured_counts:
    if len(configured_counts) != 1 or configured_counts[0] != world_size:
      raise RuntimeError(
          f"{_CPU_DEVICE_COUNT_FLAG} must equal global WORLD_SIZE={world_size};"
          f" XLA_FLAGS={xla_flags!r}."
      )
    return
  os.environ["XLA_FLAGS"] = " ".join(
      part
      for part in (
          xla_flags.strip(),
          f"{_CPU_DEVICE_COUNT_FLAG}={world_size}",
      )
      if part
  )


def _get_shard_map(jax):
  if hasattr(jax, "shard_map"):
    shard_map = jax.shard_map
  else:
    from jax.experimental.shard_map import shard_map

  check_arg = (
      {"check_vma": False}
      if "check_vma" in inspect.signature(shard_map).parameters
      else {"check_rep": False}
  )
  return shard_map, check_arg


def _backend_state() -> dict[str, Any]:
  import jax
  from jax._src import xla_bridge as xb
  from jax._src.lib import _jax

  def plugin_state(name: str) -> dict[str, Any]:
    out = {}
    for field, fn in (
        ("loaded", _jax.pjrt_plugin_loaded),
        ("initialized", _jax.pjrt_plugin_initialized),
    ):
      try:
        out[field] = bool(fn(name))
      except Exception as exc:  # JAX raises if plugin API is uninitialized.
        out[field] = {"type": type(exc).__name__, "message": str(exc)}
    return out

  default_backend = None
  if getattr(xb, "_default_backend", None) is not None:
    default_backend = xb._default_backend.platform
  return {
      "jax_version": getattr(jax, "__version__", None),
      "jax_platforms": os.environ.get("JAX_PLATFORMS"),
      "xla_flags": os.environ.get("XLA_FLAGS"),
      "initialized_backends": sorted(getattr(xb, "_backends", {}).keys()),
      "default_backend": default_backend,
      "tpu_plugin": plugin_state("tpu"),
  }


@contextlib.contextmanager
def _tpu_info_for_cpu_export(device_kind: str):
  from jax._src.pallas.mosaic import tpu_info
  from jax.experimental.pallas import tpu as pltpu

  original_public = pltpu.get_tpu_info
  original_private = tpu_info.get_tpu_info
  chip_version = tpu_info.chip_version_from_device_kind(device_kind)
  if chip_version is None:
    raise RuntimeError(f"Unsupported Pallas TPU device kind: {device_kind}")
  info = tpu_info.get_tpu_info_for_chip(chip_version, 1)
  pltpu.get_tpu_info = lambda: info
  tpu_info.get_tpu_info = lambda: info
  try:
    yield
  finally:
    pltpu.get_tpu_info = original_public
    tpu_info.get_tpu_info = original_private


def _make_cpu_mesh(world_size: int):
  import jax
  import numpy as np
  from jax.sharding import Mesh

  devices = tuple(jax.devices("cpu"))
  if len(devices) < world_size:
    raise RuntimeError(
        f"expected at least {world_size} logical CPU devices, got"
        f" {len(devices)}. Check {_CPU_DEVICE_COUNT_FLAG}."
    )
  mesh = Mesh(np.asarray(devices[:world_size]).reshape(world_size), ("x",))
  return mesh, devices


def _export_collective(world_size: int, local_elems: int, mesh) -> dict[str, Any]:
  import jax
  import jax.numpy as jnp
  from jax import lax
  from jax.sharding import PartitionSpec as P

  shard_map, check_arg = _get_shard_map(jax)

  def shard_fn(x):
    return lax.psum(x, "x")

  mapped = jax.jit(
      shard_map(
          shard_fn,
          mesh=mesh,
          in_specs=P("x"),
          out_specs=P("x"),
          **check_arg,
      )
  )
  exported = jax.export.export(mapped, platforms=["tpu"])(
      jax.ShapeDtypeStruct((world_size * local_elems,), jnp.float32)
  )
  module = exported.mlir_module().lower()
  return {
      "nr_devices": exported.nr_devices,
      "serialized_size": len(exported.mlir_module_serialized),
      "serialized_sha256": hashlib.sha256(
          exported.mlir_module_serialized
      ).hexdigest(),
      "has_all_reduce": "all_reduce" in module,
  }


def _export_pallas_remote_copy(
    world_size: int, local_elems: int, mesh, device_kind: str
) -> dict[str, Any]:
  import jax
  import jax.numpy as jnp
  from jax import lax
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.sharding import PartitionSpec as P

  shard_map, check_arg = _get_shard_map(jax)

  def remote_copy_kernel(x_ref, y_ref, send_sem, recv_sem):
    rank = lax.axis_index("x")
    destination = (rank + 1) % lax.axis_size("x")
    copy = pltpu.make_async_remote_copy(
        src_ref=x_ref,
        dst_ref=y_ref,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=(destination,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    copy.wait()

  def remote_copy_shard(x_local):
    block_spec = pl.BlockSpec(x_local.shape)
    return pl.pallas_call(
        remote_copy_kernel,
        out_shape=jax.ShapeDtypeStruct(x_local.shape, x_local.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[block_spec],
            out_specs=block_spec,
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=1024 * 1024),
    )(x_local)

  mapped = jax.jit(
      shard_map(
          remote_copy_shard,
          mesh=mesh,
          in_specs=P("x"),
          out_specs=P("x"),
          **check_arg,
      )
  )
  exporter = jax.export.export(
      mapped,
      platforms=["tpu"],
      disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
  )
  with _tpu_info_for_cpu_export(device_kind):
    exported = exporter(
        jax.ShapeDtypeStruct((world_size * local_elems,), jnp.float32)
    )
  module = exported.mlir_module().lower()
  return {
      "nr_devices": exported.nr_devices,
      "serialized_size": len(exported.mlir_module_serialized),
      "serialized_sha256": hashlib.sha256(
          exported.mlir_module_serialized
      ).hexdigest(),
      "has_custom_call": "custom_call" in module,
  }


def _initialize_torchtpu(result: dict[str, Any]) -> None:
  import torch
  import torch.distributed as dist
  import torch_tpu  # pylint: disable=unused-import
  from torch_tpu._internal.distributed import tpu_distributed

  dist.init_process_group(backend="tpu_dist")
  runtime_rank = dist.get_rank()
  physical_partition_id = tpu_distributed.global_device_id()
  result["torchtpu_runtime_rank"] = runtime_rank
  result["torchtpu_physical_partition_id"] = physical_partition_id
  result["runtime_rank_matches_physical_partition"] = (
      runtime_rank == physical_partition_id
  )


def _finish_torchtpu_check(result: dict[str, Any]) -> None:
  import torch
  import torch.distributed as dist

  x = torch.tensor([float(result["rank"] + 1)], device="tpu")
  dist.all_reduce(x, op=dist.ReduceOp.SUM)
  result["torchtpu_all_reduce"] = x.cpu().tolist()
  result["torchtpu_current_device"] = torch.tpu.current_device()
  result["torchtpu_device_count"] = torch.tpu.device_count()
  dist.barrier()
  dist.destroy_process_group()


def _launch_singlehost(args: argparse.Namespace) -> int:
  from torch_tpu._internal.distributed.launchers import singlehost_wrapper

  singlehost_wrapper.prepare_tpu_environment(args.launch_singlehost)
  cmd = [
      sys.executable,
      "-m",
      "torch.distributed.run",
      "--standalone",
      f"--nproc_per_node={args.launch_singlehost}",
      __file__,
      f"--local-elems={args.local_elems}",
      f"--device-kind={args.device_kind}",
  ]
  if args.check_torchtpu:
    cmd.append("--check-torchtpu")
  if args.skip_pallas:
    cmd.append("--skip-pallas")

  completed = subprocess.run(
      cmd,
      env=os.environ.copy(),
      text=True,
      capture_output=True,
      timeout=args.launch_timeout_s,
      check=False,
  )
  print("=== command ===")
  print(" ".join(cmd))
  print("=== returncode ===")
  print(completed.returncode)
  print("=== stdout ===")
  print(completed.stdout)
  print("=== stderr ===")
  print(completed.stderr)
  records = [
      record
      for record in _json_objects(completed.stdout)
      if "rank" in record and "collective_export" in record
  ]
  if records:
    records.sort(key=lambda record: record["rank"])
    summary = {
        "rank_to_physical_partition": {
            str(record["rank"]): record.get("torchtpu_physical_partition_id")
            for record in records
        },
        "jax_initialized_backends": sorted(
            {
                backend
                for record in records
                for backend in record["jax_backend_after_export"][
                    "initialized_backends"
                ]
            }
        ),
        "collective_module_sha256": sorted(
            {
                record["collective_export"]["serialized_sha256"]
                for record in records
            }
        ),
        "pallas_module_sha256": sorted(
            {
                record["pallas_export"]["serialized_sha256"]
                for record in records
                if "pallas_export" in record
            }
        ),
    }
    print("=== summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
  return completed.returncode


def _json_objects(text: str):
  """Extract JSON objects from interleaved multi-process stdout."""
  decoder = json.JSONDecoder()
  offset = 0
  while offset < len(text):
    start = text.find("{", offset)
    if start < 0:
      return
    try:
      value, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError:
      offset = start + 1
      continue
    offset = end
    if isinstance(value, dict):
      yield value


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--world-size",
      type=int,
      default=None,
      help=(
          "Global logical mesh size for export-only runs. Under torchrun,"
          " WORLD_SIZE from the environment is used."
      ),
  )
  parser.add_argument("--local-elems", type=int, default=4)
  parser.add_argument("--device-kind", default="TPU7x")
  parser.add_argument(
      "--launch-singlehost",
      type=int,
      default=None,
      metavar="N",
      help=(
          "Prepare single-host TorchTPU env and launch this repro with"
          " torchrun --nproc_per_node=N."
      ),
  )
  parser.add_argument("--launch-timeout-s", type=int, default=180)
  parser.add_argument(
      "--check-torchtpu",
      action="store_true",
      help="Also initialize tpu_dist and run a TPU all-reduce.",
  )
  parser.add_argument(
      "--skip-pallas",
      action="store_true",
      help="Only export a shard_map/lax.psum program, not the Pallas DMA kernel.",
  )
  args = parser.parse_args()

  if args.launch_singlehost is not None:
    raise SystemExit(_launch_singlehost(args))

  rank = int(os.environ.get("RANK", "0"))
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  world_size = int(os.environ.get("WORLD_SIZE", args.world_size or "1"))
  local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

  _configure_jax_cpu_export_env(world_size)
  result: dict[str, Any] = {
      "rank": rank,
      "local_rank": local_rank,
      "world_size": world_size,
      "local_world_size": local_world_size,
  }
  if args.check_torchtpu:
    _initialize_torchtpu(result)

  mesh, cpu_devices = _make_cpu_mesh(world_size)
  result.update(
      {
          "mesh_shape": dict(mesh.shape),
          "mesh_platforms": sorted({d.platform for d in mesh.devices.flat}),
          "cpu_device_count": len(cpu_devices),
      }
  )
  result["collective_export"] = _export_collective(
      world_size, args.local_elems, mesh
  )
  if not args.skip_pallas:
    result["pallas_export"] = _export_pallas_remote_copy(
        world_size, args.local_elems, mesh, args.device_kind
    )
  result["jax_backend_after_export"] = _backend_state()

  if args.check_torchtpu:
    _finish_torchtpu_check(result)

  print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
