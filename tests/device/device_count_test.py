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

"""Tests device APIs before and after the device runtime is initialized.

PyTorch CUDA's `torch.cuda.device_count()` returns the number of devices this
process can use both before and after CUDA is initialized: before
initialization it applies `CUDA_VISIBLE_DEVICES` to the NVML device count, and
it doesn't cache the count until CUDA is initialized. Querying memory stats
before CUDA is initialized doesn't initialize it and reports no memory. The
same tests run against `torch.cuda` (`--device_type=cuda`) and `torch.tpu`
(`--device_type=tpu`) to check that TorchTPU matches this behavior.

Each scenario runs in a freshly spawned process, since the device runtime can
be initialized only once per process.
"""

import dataclasses
import os
import traceback
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.utils import annotations
from torch_tpu._internal.distributed import multiprocessing
from tests import accelerator_test_utils

_DEVICE_TYPE = flags.DEFINE_enum(
    "device_type", "tpu", ["cuda", "tpu"], "The device type to test."
)

# The environment variable that restricts which devices a process can use.
_VISIBLE_DEVICES_ENV_VARS = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "tpu": "TPU_VISIBLE_DEVICES",
}

_WORKER_TIMEOUT_SECONDS = 300


@dataclasses.dataclass(frozen=True)
class _DeviceCounts:
  """The device counts that a worker observed.

  Attributes:
    before_init: `device_count()` when the process starts, before the device
      runtime is initialized.
    after_env_change: `device_count()` after the worker changed the visible
      devices environment variable, still before the runtime is initialized, or
      None if the worker didn't change it.
    after_init: `device_count()` after the device runtime is initialized.
  """

  before_init: int
  after_env_change: int | None
  after_init: int


def _count_devices(
    conn: Any,
    device_type: str,
    visible_devices_before_init: str | None,
) -> None:
  """Reports the device counts before and after initializing the runtime.

  Runs in a spawned process. Sends either `("ok", _DeviceCounts)` or
  `("error", traceback)` over `conn`.

  Args:
    conn: The sending end of a pipe to the test process.
    device_type: "cuda" or "tpu".
    visible_devices_before_init: If not None, the value to set the visible
      devices environment variable to after the first `device_count()` call and
      before initializing the runtime.
  """
  # `setUpModule` doesn't run in spawned processes.
  annotations._internal_apis_allowed_for_testing = True  # pylint: disable=protected-access
  try:
    device_module = accelerator_test_utils.get_backend_module(device_type)
    if device_module.is_initialized():
      raise AssertionError(f"{device_type} is initialized at process start.")
    before_init = device_module.device_count()

    after_env_change = None
    if visible_devices_before_init is not None:
      os.environ[_VISIBLE_DEVICES_ENV_VARS[device_type]] = (
          visible_devices_before_init
      )
      after_env_change = device_module.device_count()

    # Running a computation and copying its result to the host initializes the
    # device runtime; allocating a tensor alone doesn't on TPU.
    device = accelerator_test_utils.get_device(device_type)
    (torch.ones(1, device=device) + 1).cpu()
    if not device_module.is_initialized():
      raise AssertionError(
          f"{device_type} isn't initialized after running a computation."
      )
    after_init = device_module.device_count()

    conn.send(("ok", _DeviceCounts(before_init, after_env_change, after_init)))
  except Exception:  # pylint: disable=broad-exception-caught
    conn.send(("error", traceback.format_exc()))
  finally:
    conn.close()


@dataclasses.dataclass(frozen=True)
class _MemorySnapshot:
  """The `torch.accelerator` memory stats and whether the runtime is initialized.

  Attributes:
    memory_stats: `memory_stats()`.
    memory_allocated: `memory_allocated()`.
    max_memory_allocated: `max_memory_allocated()`.
    memory_reserved: `memory_reserved()`.
    max_memory_reserved: `max_memory_reserved()`.
    is_initialized: Whether the device runtime is initialized after the queries
      above, i.e. whether they initialized it if it wasn't.
  """

  memory_stats: dict[str, int]
  memory_allocated: int
  max_memory_allocated: int
  memory_reserved: int
  max_memory_reserved: int
  is_initialized: bool


@dataclasses.dataclass(frozen=True)
class _MemoryStats:
  """The memory stats that a worker observed.

  Attributes:
    before_init: The snapshot when the process starts, before the device runtime
      is initialized.
    after_init: The snapshot after the device runtime is initialized and holds a
      tensor.
  """

  before_init: _MemorySnapshot
  after_init: _MemorySnapshot


def _snapshot_memory_stats(device_module: Any) -> _MemorySnapshot:
  """Returns the `torch.accelerator` memory stats and whether it's initialized.

  Args:
    device_module: `torch.cuda` or `torch.tpu`.
  """
  memory_stats = dict(torch.accelerator.memory_stats())
  memory_allocated = torch.accelerator.memory_allocated()
  max_memory_allocated = torch.accelerator.max_memory_allocated()
  memory_reserved = torch.accelerator.memory_reserved()
  max_memory_reserved = torch.accelerator.max_memory_reserved()
  return _MemorySnapshot(
      memory_stats=memory_stats,
      memory_allocated=memory_allocated,
      max_memory_allocated=max_memory_allocated,
      memory_reserved=memory_reserved,
      max_memory_reserved=max_memory_reserved,
      # Read after the queries above, so it tells whether they initialized the
      # device runtime.
      is_initialized=device_module.is_initialized(),
  )


def _query_memory_stats(conn: Any, device_type: str) -> None:
  """Reports the memory stats before and after initializing the runtime.

  Runs in a spawned process. Sends either `("ok", _MemoryStats)` or
  `("error", traceback)` over `conn`.

  Args:
    conn: The sending end of a pipe to the test process.
    device_type: "cuda" or "tpu".
  """
  # `setUpModule` doesn't run in spawned processes.
  annotations._internal_apis_allowed_for_testing = True  # pylint: disable=protected-access
  try:
    device_module = accelerator_test_utils.get_backend_module(device_type)
    if device_module.is_initialized():
      raise AssertionError(f"{device_type} is initialized at process start.")

    torch.accelerator.reset_peak_memory_stats()
    before_init = _snapshot_memory_stats(device_module)

    # Running a computation and copying its result to the host initializes the
    # device runtime; allocating a tensor alone doesn't on TPU.
    device = accelerator_test_utils.get_device(device_type)
    x = torch.ones(1024, device=device) + 1
    x.cpu()
    # `x` is still alive, so the stats count it.
    after_init = _snapshot_memory_stats(device_module)

    conn.send(("ok", _MemoryStats(before_init, after_init)))
  except Exception:  # pylint: disable=broad-exception-caught
    conn.send(("error", traceback.format_exc()))
  finally:
    conn.close()


def _hold_device(conn: Any, device_type: str) -> None:
  """Initializes the device runtime and holds a device until told to exit.

  Runs in a spawned process. Sends `("ok", None)` over `conn` once the runtime
  holds a device, or `("error", traceback)`, then waits for a message on `conn`
  before exiting.

  Args:
    conn: A duplex connection to the test process.
    device_type: "cuda" or "tpu".
  """
  try:
    device = accelerator_test_utils.get_device(device_type)
    (torch.ones(1, device=device) + 1).cpu()
    conn.send(("ok", None))
  except Exception:  # pylint: disable=broad-exception-caught
    conn.send(("error", traceback.format_exc()))
    conn.close()
    return
  conn.recv()
  conn.close()


# Not a `seed_test_utils` test class: those seed the device RNGs in `setUp()`,
# which initializes the device runtime in the test process.
class DeviceCountTest(
    absltest.TestCase  # ABSLTEST_OK=must not initialize the device runtime
):
  """Tests `device_count()` before and after runtime initialization."""

  def setUp(self):
    super().setUp()
    self.device_type = _DEVICE_TYPE.value
    self.visible_devices_env_var = _VISIBLE_DEVICES_ENV_VARS[self.device_type]

  def _run_worker(
      self,
      target: Any,
      visible_devices_at_start: str | None,
      *args: Any,
  ) -> Any:
    """Runs `target(conn, device_type, *args)` in a new process.

    Args:
      target: A worker function, e.g. `_count_devices`, that sends `("ok",
        result)` or `("error", traceback)` over `conn`.
      visible_devices_at_start: The visible devices environment variable that
        the new process starts with, or None to start it with the variable
        unset.
      *args: More arguments to `target`.

    Returns:
      The result that `target` sent.
    """
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=target,
        args=(sender, self.device_type, *args),
    )
    # The spawned process inherits the environment at `start()`.
    with mock.patch.dict(os.environ):
      if visible_devices_at_start is None:
        os.environ.pop(self.visible_devices_env_var, None)
      else:
        os.environ[self.visible_devices_env_var] = visible_devices_at_start
      process.start()
    sender.close()

    try:
      if not receiver.poll(_WORKER_TIMEOUT_SECONDS):
        self.fail(
            f"The worker sent no result within {_WORKER_TIMEOUT_SECONDS}s"
            f" (exit code: {process.exitcode})."
        )
      status, payload = receiver.recv()
    finally:
      receiver.close()
      process.join(_WORKER_TIMEOUT_SECONDS)
      if process.is_alive():
        process.kill()
        process.join()

    if status != "ok":
      self.fail(f"The worker failed:\n{payload}")
    return payload

  def test_count_is_the_same_before_and_after_init(self):
    """Verifies that initializing the runtime doesn't change the count."""
    counts = self._run_worker(_count_devices, None, None)

    self.assertGreater(counts.before_init, 0)
    self.assertEqual(counts.before_init, counts.after_init)

  def test_count_before_init_respects_visible_devices(self):
    """Verifies the count before init applies the visible devices variable."""
    counts = self._run_worker(_count_devices, "0", None)

    self.assertEqual(counts.before_init, 1)
    self.assertEqual(counts.after_init, 1)

  def test_count_is_not_cached_before_init(self):
    """Verifies the count follows visible devices changes made before init."""
    counts = self._run_worker(_count_devices, None, "0")

    # With one device, the count is always 1 with or without the env change.
    if counts.before_init <= 1:
      self.skipTest("This test needs a host with multiple devices.")

    self.assertEqual(counts.after_env_change, 1)
    # Like PyTorch's `test_device_count_not_cached_pre_init`, this doesn't check
    # the count after init: CUDA can read `CUDA_VISIBLE_DEVICES` before the
    # change above (e.g. when the driver is loaded at process start), in which
    # case the count after init is still the host's device count.

  def test_count_includes_devices_that_other_processes_hold(self):
    """Tests that devices other processes hold are still counted.

    The count must not drop as other processes, e.g. the other workers of the
    same job, initialize the runtime and take devices.
    """
    device_module = accelerator_test_utils.get_backend_module(self.device_type)
    self.assertFalse(device_module.is_initialized())
    count = device_module.device_count()
    self.assertGreater(count, 0)

    ctx = multiprocessing.get_context("spawn")
    conn, child_conn = ctx.Pipe()
    process = ctx.Process(
        target=_hold_device, args=(child_conn, self.device_type)
    )
    # The worker inherits the visible devices of this process, e.g. the device
    # that the test runner assigned to this test, so it doesn't take a device
    # that another test is using.
    process.start()
    child_conn.close()
    try:
      if not conn.poll(_WORKER_TIMEOUT_SECONDS):
        self.fail(
            f"The worker sent no result within {_WORKER_TIMEOUT_SECONDS}s"
            f" (exit code: {process.exitcode})."
        )
      status, payload = conn.recv()
      if status != "ok":
        self.fail(f"The worker failed:\n{payload}")

      self.assertEqual(device_module.device_count(), count)
      self.assertFalse(device_module.is_initialized())
    finally:
      try:
        conn.send("exit")
      except (BrokenPipeError, EOFError, OSError):
        pass
      conn.close()
      process.join(_WORKER_TIMEOUT_SECONDS)
      if process.is_alive():
        process.kill()
        process.join()

  def test_memory_stats_do_not_initialize_the_runtime(self):
    """Verifies memory stats report no memory and don't initialize the runtime."""
    # The worker inherits the visible devices of this process.
    stats = self._run_worker(
        _query_memory_stats, os.environ.get(self.visible_devices_env_var)
    )

    self.assertEqual(
        stats.before_init,
        _MemorySnapshot(
            memory_stats={},
            memory_allocated=0,
            max_memory_allocated=0,
            memory_reserved=0,
            max_memory_reserved=0,
            is_initialized=False,
        ),
    )
    self.assertTrue(stats.after_init.is_initialized)
    self.assertGreater(stats.after_init.memory_allocated, 0)


def setUpModule():
  """Called by absltest after flags are parsed and before tests are run."""
  annotations._internal_apis_allowed_for_testing = True  # pylint: disable=protected-access


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
