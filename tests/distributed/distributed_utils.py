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

"""Distributed testing utilities for TorchTPU."""

import os
import re
from typing import Any, Callable

from absl import logging
import portpicker
import torch.multiprocessing as mp
from torch_tpu._internal import env
from torch_tpu._internal import env_utils
from torch_tpu._internal.utils import hardware
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils

_SLICEBUILDER_ADDRESSES_ENV = "TORCH_TPU_SLICEBUILDER_ADDRESSES"

# Matches the error libtpu emits when the local TPU agent socket (GKE
# `vbar-control-agent`) does not accept the connection within its timeout, e.g.
# "Failed to connect to unix:///var/run/tpu-plugin/vbar.sock". Only the
# essential parts (a connect failure to a unix socket under `tpu-plugin`) are
# matched, so that rewording or a different socket path does not silently
# disable the retry. This happens before the SliceBuilder session or the c10d
# rendezvous is set up, so retrying with fresh ports is safe.
_TPU_AGENT_CONNECT_FAILURE_RE = re.compile(
    r"connect.*unix://\S*tpu-plugin", re.IGNORECASE
)

# Every `TORCH_TPU_SLICEBUILDER_ADDRESSES` value picked by `dist_run` itself in
# this process, used to tell our own leftover value apart from one pinned by the
# caller or a launcher (e.g. after a test restores an earlier `os.environ`).
_picked_slicebuilder_addresses: set[str] = set()


def _worker_fn(
    local_rank: int,
    nproc_per_node: int,
    default_master_port: int,
    base_seed: int,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
  """Worker function for mp.spawn."""
  # Parse absl flags inside each mp.spawn child subprocess so tests can pass
  # `fn` directly to `dist_run`.
  multiprocessing.parse_absl_flags()

  node_rank = int(os.environ.get("NODE_RANK", "0"))
  if "WORLD_SIZE" in os.environ:
    world_size = int(os.environ["WORLD_SIZE"])
  elif node_rank > 0:
    raise ValueError(
        f"NODE_RANK: {node_rank} > 0 suggests multihost job. Expected"
        " WORLD_SIZE environment variable to be set"
    )
  else:
    world_size = nproc_per_node
  global_rank = node_rank * nproc_per_node + local_rank
  if global_rank >= world_size:
    raise ValueError(
        f"Calculated global_rank {global_rank} >= world_size {world_size}"
    )

  if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
  if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = str(default_master_port)

  os.environ["RANK"] = str(global_rank)
  os.environ["LOCAL_RANK"] = str(local_rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_WORLD_SIZE"] = str(nproc_per_node)
  os.environ["GROUP_RANK"] = str(node_rank)

  if "XLA_FLAGS" in os.environ:
    os.environ["XLA_FLAGS"] = os.path.expandvars(os.environ["XLA_FLAGS"])

  seed_test_utils.seed_rngs(base_seed + global_rank)

  fn(*args, **kwargs)


def dist_run(
    nproc_per_node: int,
    fn: Callable[..., Any],
    *args: Any,
    base_seed: int | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> None:
  """Runs the given function in a distributed environment using mp.spawn.

  Think of dist_run(n, foo, *args, **kwargs) as running
  n copies of foo(*args, **kwargs) in parallel, each with a different rank.
  Automatically bootstraps the TPU launch environment (`set_tpu_launch_env`)
  when TPU hardware or a multi-host TPU job environment is present, while
  remaining a no-op on CPU-only or GPU-only runners. Preserves multi-host
  environment variables (MASTER_ADDR, MASTER_PORT, NODE_RANK, WORLD_SIZE) if
  already present in os.environ.

  On a single host, a fresh set of SliceBuilder ports
  (`TORCH_TPU_SLICEBUILDER_ADDRESSES`) is picked for every call and every retry
  unless the caller pinned the variable before calling `dist_run`. Reusing the
  ports picked by the first `dist_run` in a process for all later calls would
  break when a later call uses a different `nproc_per_node`, and would retry a
  failed attempt against the very same ports.

  Args:
    nproc_per_node: The number of processes to spawn on the current node.
    fn: The function to be executed by each distributed worker.
    *args: Positional arguments to pass to the function.
    base_seed: The base RNG seed for worker processes. Defaults to the seed
      chosen by RepeatableTest.choose_seed().
    max_retries: Maximum number of attempts if a transient failure occurs: a
      port collision (EADDRINUSE) when auto-selecting a port, or the TPU runtime
      failing to come up before the workers rendezvoused.
    **kwargs: Keyword arguments to pass to the function.
  """
  multiprocessing.init_spawn()

  repick_slicebuilder_ports = False
  if (
      hardware.get_tpu_device_count()
      or env.get_str_env_once("JOB_COMPLETION_INDEX") is not None
  ):
    # Imported lazily so GPU runs (TORCH_DEVICE_BACKEND_AUTOLOAD=0) never depend
    # on `torch.tpu`, which only exists once the TorchTPU backend is loaded.
    from torch.tpu import distributed  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    # Read through `os.environ` rather than `env.get_str_env_once()`, which
    # would memoize the variable as unset before `set_tpu_launch_env` sets it.
    # Addresses picked by an earlier `dist_run` in this process are ours to
    # replace; only a value set by someone else counts as pinned.
    current_addresses = os.environ.get(  # PY_ENVIRON_OK=must not memoize.
        _SLICEBUILDER_ADDRESSES_ENV
    )
    repick_slicebuilder_ports = (
        current_addresses is None
        or current_addresses in _picked_slicebuilder_addresses
    ) and _is_single_host()
    if repick_slicebuilder_ports:
      _set_fresh_slicebuilder_addresses(nproc_per_node)

    distributed.set_tpu_launch_env(nproc_per_node=nproc_per_node)

  if base_seed is None:
    base_seed = seed_test_utils.RepeatableTest.choose_seed()

  for attempt in range(max_retries):
    if attempt > 0 and repick_slicebuilder_ports:
      _set_fresh_slicebuilder_addresses(nproc_per_node)
    default_master_port = (
        int(os.environ["MASTER_PORT"])
        if "MASTER_PORT" in os.environ
        else portpicker.pick_unused_port()
    )
    try:
      mp.spawn(
          _worker_fn,
          args=(
              nproc_per_node,
              default_master_port,
              base_seed,
              fn,
              args,
              kwargs,
          ),
          nprocs=nproc_per_node,
          join=True,
      )
      return
    except mp.ProcessRaisedException as e:
      reason = _retryable_failure_reason(
          e, can_repick_ports=repick_slicebuilder_ports
      )
      if reason is None or attempt >= max_retries - 1:
        raise
      logging.warning(
          "%s on attempt %d/%d (MASTER_PORT=%d); retrying.",
          reason,
          attempt + 1,
          max_retries,
          default_master_port,
      )


def _is_single_host() -> bool:
  """Returns whether this is a single-host (non multi-host) TPU job."""
  hostnames = env.get_str_env_once("TPU_WORKER_HOSTNAMES", default_value="")
  return (
      env.get_str_env_once("JOB_COMPLETION_INDEX") is None
      and len([h for h in hostnames.split(",") if h]) <= 1
  )


def _set_fresh_slicebuilder_addresses(nproc_per_node: int) -> None:
  """Points `TORCH_TPU_SLICEBUILDER_ADDRESSES` at unused local ports."""
  addresses = ",".join(
      f"localhost:{portpicker.pick_unused_port()}"
      for _ in range(nproc_per_node)
  )
  env_utils.set_str_env(_SLICEBUILDER_ADDRESSES_ENV, addresses)
  _picked_slicebuilder_addresses.add(addresses)


def _retryable_failure_reason(
    e: mp.ProcessRaisedException, *, can_repick_ports: bool
) -> str | None:
  """Returns why a worker failure is worth retrying, or None if it is not."""
  message = str(e)
  if "EADDRINUSE" in message:
    # If MASTER_PORT is pinned by the caller or a multi-host orchestrator and
    # the SliceBuilder ports cannot be re-picked either, retrying would reuse
    # the very same ports and fail again.
    if "MASTER_PORT" in os.environ and not can_repick_ports:
      return None
    return "Port collision (EADDRINUSE)"
  if _TPU_AGENT_CONNECT_FAILURE_RE.search(message):
    return "TPU runtime failed to connect to the local TPU agent"
  return None
