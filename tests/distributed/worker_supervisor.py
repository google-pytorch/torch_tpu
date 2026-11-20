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

"""`torch.multiprocessing.spawn` with a deadline.

The launcher enforces the deadline from outside the workers, so it also covers a
worker that hangs before it runs any of its own code. When the deadline passes,
the launcher kills the workers that are still running (and the processes they
started) and raises `WorkerTimeoutError`. A worker that fails before the
deadline is handled by `torch.multiprocessing`, exactly as without a deadline.
"""

from collections.abc import Callable, Sequence
import os
import signal
import time
from typing import Any

from absl import logging
import torch.multiprocessing as mp

# Wait after SIGTERM before SIGKILL; the same as `ProcessContext.join` uses.
_GRACE_SECONDS = 30.0


class WorkerTimeoutError(TimeoutError):
  """Workers were still running at their deadline and were killed."""


def _supervised_worker(
    local_rank: int,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
) -> None:
  """Runs `fn` in its own process group so the launcher can kill its subtree."""
  # Lead a process group, so that the launcher kills the processes that this
  # worker starts together with it.
  os.setsid()
  fn(local_rank, *args)


def _signal_group(process: Any, signum: int) -> None:
  """Sends `signum` to the process group that the worker `process` leads."""
  try:
    os.killpg(process.pid, signum)
  except ProcessLookupError:
    # No such group: the worker has exited, or has not called `os.setsid` yet.
    if process.is_alive():
      os.kill(process.pid, signum)


def _kill_workers(alive: Sequence[Any]) -> None:
  """Kills `alive` workers and their process groups (SIGTERM, then SIGKILL)."""
  for process in alive:
    _signal_group(process, signal.SIGTERM)
  grace_end = time.monotonic() + _GRACE_SECONDS
  for process in alive:
    process.join(max(0.0, grace_end - time.monotonic()))
  for process in alive:
    _signal_group(process, signal.SIGKILL)
    process.join()


def spawn_and_join(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    nprocs: int = 1,
    *,
    deadline: float | None = None,
) -> None:
  """Runs `mp.spawn(fn, args, nprocs)` and waits for the workers.

  Args:
    fn: Picklable worker function, called as `fn(local_rank, *args)`.
    args: Arguments passed to `fn` after the local rank.
    nprocs: Number of workers to spawn.
    deadline: The `time.monotonic()` time by which the workers must have
      finished, or None to wait for them indefinitely.

  Raises:
    WorkerTimeoutError: If workers were still running at `deadline`. They are
      killed first.
  """
  if deadline is None:
    mp.spawn(fn, args=args, nprocs=nprocs, join=True)
    return
  context = mp.spawn(
      _supervised_worker,
      args=(fn, args),
      nprocs=nprocs,
      join=False,
  )
  # `join` returns False if it timed out or only some workers have finished,
  # and raises if a worker failed.
  while not context.join(timeout=max(0.0, deadline - time.monotonic())):
    if time.monotonic() < deadline:
      continue
    alive = [process for process in context.processes if process.is_alive()]
    if alive:
      pids = [process.pid for process in alive]
      logging.error(
          "Distributed workers %s are still running at the deadline; killing"
          " them.",
          pids,
      )
      _kill_workers(alive)
      raise WorkerTimeoutError(
          "Distributed workers did not finish by the deadline; killed PIDs"
          f" {pids}."
      )
