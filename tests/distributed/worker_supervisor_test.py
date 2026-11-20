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

"""Tests for worker_supervisor module."""

from collections.abc import Callable, Sequence
import os
import signal
import subprocess
import time
from typing import Any
from unittest import mock

from absl.testing import absltest
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import worker_supervisor

# Upper bound for spawned workers to start or exit; only reached on failure.
_STARTUP_TIMEOUT_SECONDS = 120.0
# Bounded sleep for "hung" test workers, so that even a leaked one exits.
_HANG_SECONDS = 120


def _mark_ready(local_rank: int, output_dir: str) -> None:
  with open(os.path.join(output_dir, f"{local_rank}.ready"), "w"):
    pass


def _num_ready(output_dir: str) -> int:
  return sum(name.endswith(".ready") for name in os.listdir(output_dir))


def _wait_until_ready(output_dir: str, num_ready: int) -> None:
  """Waits until `num_ready` workers have called `_mark_ready`."""
  deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
  while _num_ready(output_dir) < num_ready:
    if time.monotonic() > deadline:
      raise TimeoutError(
          f"Workers did not start within {_STARTUP_TIMEOUT_SECONDS} s."
      )
    time.sleep(0.05)


def _worker_exit(local_rank: int, output_dir: str) -> None:
  _mark_ready(local_rank, output_dir)


def _worker_hang(local_rank: int, output_dir: str) -> None:
  _mark_ready(local_rank, output_dir)
  time.sleep(_HANG_SECONDS)


def _worker_hang_ignoring_sigterm(local_rank: int, output_dir: str) -> None:
  signal.signal(signal.SIGTERM, signal.SIG_IGN)
  _mark_ready(local_rank, output_dir)
  time.sleep(_HANG_SECONDS)


def _worker_start_child_and_hang(local_rank: int, output_dir: str) -> None:
  child = subprocess.Popen(["sleep", str(_HANG_SECONDS)])
  with open(os.path.join(output_dir, "child.pid"), "w") as f:
    f.write(str(child.pid))
  _worker_hang(local_rank, output_dir)


# Starting Python workers dominates the test runtime, so one spawn covers a
# hanging, a SIGTERM-ignoring and an exiting worker.
_RANK_HANGING = 0
_RANK_IGNORING_SIGTERM = 1
_RANK_EXITING = 2
_WORKER_BY_RANK = (_worker_hang, _worker_hang_ignoring_sigterm, _worker_exit)


def _worker_by_rank(local_rank: int, output_dir: str) -> None:
  _WORKER_BY_RANK[local_rank](local_rank, output_dir)


def _worker_raise_once_peer_ready(local_rank: int, output_dir: str) -> None:
  """Rank 0 hangs; rank 1 waits until rank 0 is ready, then raises."""
  if local_rank == 0:
    _worker_hang(local_rank, output_dir)
    return
  _wait_until_ready(output_dir, 1)
  raise ValueError(f"worker_supervisor_test failure on rank {local_rank}")


def _pid_exists(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  return True


class WorkerSupervisorTest(seed_test_utils.MultiProcessRepeatableTest):

  def _spawn_and_join_past_deadline(
      self,
      fn: Callable[..., Any],
      nprocs: int,
      exiting_ranks: Sequence[int] = (),
  ) -> tuple[Any, list[str], str]:
    """Runs `fn` with a deadline that passes once every worker is ready.

    Args:
      fn: Worker function, called as `fn(local_rank, output_dir)`.
      nprocs: Number of workers.
      exiting_ranks: Workers that exit on their own, which are waited for.

    Returns:
      The process context, the logged errors, and the workers' output directory.
    """
    output_dir = self.create_tempdir().full_path
    real_spawn = mp.spawn
    contexts = []

    def spawn_and_wait_ready(*args, **kwargs):
      context = real_spawn(*args, **kwargs)
      contexts.append(context)
      _wait_until_ready(output_dir, nprocs)
      for rank in exiting_ranks:
        context.processes[rank].join(_STARTUP_TIMEOUT_SECONDS)
      return context

    with (
        mock.patch.object(
            mp, "spawn", autospec=True, side_effect=spawn_and_wait_ready
        ),
        mock.patch.object(worker_supervisor, "_GRACE_SECONDS", 2.0),
        self.assertLogs(level="ERROR") as logs,
        self.assertRaisesRegex(  # ASSERT_RAISES_OK=The deadline has passed.
            worker_supervisor.WorkerTimeoutError,
            "did not finish by the deadline",
        ),
    ):
      worker_supervisor.spawn_and_join(
          fn, args=(output_dir,), nprocs=nprocs, deadline=time.monotonic()
      )
    return contexts[0], logs.output, output_dir

  def test_completes_without_deadline(self) -> None:
    output_dir = self.create_tempdir().full_path

    worker_supervisor.spawn_and_join(_worker_exit, args=(output_dir,), nprocs=2)

    self.assertEqual(_num_ready(output_dir), 2)

  def test_completes_before_deadline(self) -> None:
    output_dir = self.create_tempdir().full_path

    worker_supervisor.spawn_and_join(
        _worker_exit,
        args=(output_dir,),
        nprocs=2,
        deadline=time.monotonic() + _STARTUP_TIMEOUT_SECONDS,
    )

    self.assertEqual(_num_ready(output_dir), 2)

  def test_deadline_kills_running_workers(self) -> None:
    context, errors, _ = self._spawn_and_join_past_deadline(
        _worker_by_rank,
        nprocs=len(_WORKER_BY_RANK),
        exiting_ranks=[_RANK_EXITING],
    )

    self.assertEqual(
        [process.exitcode for process in context.processes],
        [-signal.SIGTERM, -signal.SIGKILL, 0],
    )
    pids = context.pids()
    self.assertIn(
        str([pids[_RANK_HANGING], pids[_RANK_IGNORING_SIGTERM]]), errors[0]
    )
    self.assertNotIn(str(pids[_RANK_EXITING]), errors[0])

  def test_deadline_kills_processes_started_by_worker(self) -> None:
    _, _, output_dir = self._spawn_and_join_past_deadline(
        _worker_start_child_and_hang, nprocs=1
    )

    with open(os.path.join(output_dir, "child.pid")) as f:
      child_pid = int(f.read())
    end = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while _pid_exists(child_pid) and time.monotonic() < end:
      time.sleep(0.05)
    self.assertFalse(_pid_exists(child_pid))

  def test_worker_failure_before_deadline_raises_like_mp_spawn(self) -> None:
    output_dir = self.create_tempdir().full_path

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=A worker raises.
        mp.ProcessRaisedException, "worker_supervisor_test failure on rank 1"
    ):
      worker_supervisor.spawn_and_join(
          _worker_raise_once_peer_ready,
          args=(output_dir,),
          nprocs=2,
          deadline=time.monotonic() + _STARTUP_TIMEOUT_SECONDS,
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
