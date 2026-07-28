# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare JAX physical discovery and a CPU export mesh after TorchTPU init.

TorchTPU is always initialized first. Each mode must run in a fresh process
because JAX backend selection is process-global and cached.

  python3 tests/pallas/jax_mesh_runtime_init_repro.py \
    --mode=physical-jax-devices --world-size=8
  python3 tests/pallas/jax_mesh_runtime_init_repro.py \
    --mode=cpu-export-mesh --world-size=8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def _jax_state() -> dict:
  from jax._src import xla_bridge as xb
  from jax._src.lib import _jax

  try:
    tpu_initialized = bool(_jax.pjrt_plugin_initialized("tpu"))
  except Exception as exc:
    tpu_initialized = {"type": type(exc).__name__, "message": str(exc)}
  return {
      "initialized_backends": sorted(getattr(xb, "_backends", {}).keys()),
      "default_backend": getattr(
          getattr(xb, "_default_backend", None), "platform", None
      ),
      "jax_tpu_plugin_initialized": tpu_initialized,
  }


def _worker(mode: str) -> None:
  import torch
  import torch.distributed as dist
  import torch_tpu  # pylint: disable=unused-import
  from torch_tpu._internal.distributed import tpu_distributed

  dist.init_process_group(backend="tpu_dist")
  result = {
      "mode": mode,
      "torch_runtime_initialized_first": True,
      "rank": dist.get_rank(),
      "world_size": dist.get_world_size(),
      "physical_partition_id": tpu_distributed.global_device_id(),
      "jax_already_imported_after_torchtpu_init": "jax" in sys.modules,
  }

  import jax
  import numpy as np
  from jax.sharding import Mesh

  result["jax_state_before_devices"] = _jax_state()
  if mode == "physical-jax-devices":
    devices = tuple(jax.devices("tpu"))
    mesh = Mesh(np.asarray(devices).reshape(-1), ("x",))
  else:
    devices = tuple(jax.devices("cpu"))
    mesh = Mesh(np.asarray(devices).reshape(-1), ("x",))
  result.update(
      {
          "mesh_shape": dict(mesh.shape),
          "mesh_platforms": sorted({device.platform for device in devices}),
          "mesh_device_ids": [device.id for device in devices],
          "jax_state_after_devices": _jax_state(),
      }
  )

  value = torch.tensor([float(result["rank"] + 1)], device="tpu")
  dist.all_reduce(value)
  result["torchtpu_all_reduce_after_jax"] = value.cpu().tolist()
  print(json.dumps(result, sort_keys=True), flush=True)
  dist.destroy_process_group()


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


def _parent(mode: str, world_size: int, timeout_s: int) -> int:
  from torch_tpu._internal.distributed.launchers import singlehost_wrapper

  singlehost_wrapper.prepare_tpu_environment(world_size)
  env = os.environ.copy()
  env["PYTHONUNBUFFERED"] = "1"
  if mode == "cpu-export-mesh":
    env["JAX_PLATFORMS"] = "cpu"
    flag = f"--xla_force_host_platform_device_count={world_size}"
    env["XLA_FLAGS"] = " ".join(
        part for part in (env.get("XLA_FLAGS", "").strip(), flag) if part
    )
  else:
    env.pop("JAX_PLATFORMS", None)

  command = [
      sys.executable,
      "-m",
      "torch.distributed.run",
      "--standalone",
      f"--nproc_per_node={world_size}",
      __file__,
      "--worker",
      f"--mode={mode}",
  ]
  completed = subprocess.run(
      command,
      env=env,
      text=True,
      capture_output=True,
      timeout=timeout_s,
      check=False,
  )
  print(completed.stdout)
  print(completed.stderr, file=sys.stderr)
  records = [
      record
      for record in _json_objects(completed.stdout)
      if record.get("mode") == mode
  ]
  if records:
    records.sort(key=lambda record: record["rank"])
    summary = {
        "mode": mode,
        "rank_to_physical_partition": {
            str(record["rank"]): record["physical_partition_id"]
            for record in records
        },
        "mesh_platforms": sorted(
            {platform for record in records for platform in record["mesh_platforms"]}
        ),
        "jax_tpu_plugin_initialized": [
            record["jax_state_after_devices"]["jax_tpu_plugin_initialized"]
            for record in records
        ],
        "torchtpu_all_reduce_after_jax": sorted(
            {tuple(record["torchtpu_all_reduce_after_jax"]) for record in records}
        ),
    }
    print("=== summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
  return completed.returncode


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--mode",
      choices=("physical-jax-devices", "cpu-export-mesh"),
      required=True,
  )
  parser.add_argument("--world-size", type=int, default=8)
  parser.add_argument("--timeout-s", type=int, default=180)
  parser.add_argument("--worker", action="store_true")
  args = parser.parse_args()

  if args.worker:
    _worker(args.mode)
  else:
    raise SystemExit(_parent(args.mode, args.world_size, args.timeout_s))


if __name__ == "__main__":
  main()
