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

"""TorchTPU adaptation of PyTorch's `MultiProcContinuousTest`.

Instead of spawning fresh worker processes for every test,
`MultiProcContinuousTest` spawns `world_size` workers once per test class and
keeps them, and their process group, alive across the class's tests:

- The parent process is the only test runner (it decides test order, sharding,
  filtering and skips, and reports results). For each test it sends the test id
  to every worker and waits for all of them to report back.
- Each worker runs the test method body with `self.rank` and `self.world_size`
  set, and reports success or the exception.
- If a test fails on any rank, the remaining tests in the class are skipped,
  because the process group may be left in a bad state.

`TpuMultiProcContinuousTest` plugs TorchTPU's launch environment into that
harness.
"""

import functools
import os
import types
from typing import Any, Callable
import unittest
from unittest import mock

import portpicker
import torch
from torch import distributed as dist
from torch.testing._internal import common_distributed
from torch.testing._internal import common_utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils

# Whether the parent drains every worker's completion queue before it raises a
# worker's result (PyTorch 2.14+). Older versions raise the first rank's skip
# and leave the other ranks' results queued for the next test.
_PARENT_DRAINS_COMPLETION_QUEUES = hasattr(
    common_distributed, "retrieve_result_from_completion_queue"
)

# Per-test-case worker timeout. A large number, so effectively unlimited; the
# BUILD target's `timeout`/`size` bounds the whole test instead.
_TEST_CASE_TIMEOUT_SEC = 100_000


class _ReportSkipsQueue:
  """Wraps a worker's completion queue to report worker skips as skips.

  Before PyTorch 2.14, `MultiProcContinuousTest._worker_loop()` reports a
  `unittest.SkipTest` raised by a test as a `RuntimeError` (caused by the
  `SkipTest`), which the parent treats as a failure that skips the rest of the
  class. This queue sends the `SkipTest` itself instead.
  """

  def __init__(self, queue: Any):
    self._queue = queue

  def put(self, item: Any, *args: Any, **kwargs: Any) -> None:
    if isinstance(item, RuntimeError) and isinstance(
        item.__cause__, unittest.SkipTest
    ):
      item = item.__cause__
    self._queue.put(item, *args, **kwargs)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._queue, name)


class TpuMultiProcContinuousTest(
    common_distributed.MultiProcContinuousTest,
    seed_test_utils.MultiProcessRepeatableTest,
):
  """Runs each test on `world_size` TPU workers that persist across tests.

  Subclasses must set `world_size`.
  """

  @classmethod
  def device_type(cls) -> str:
    return "tpu"

  @classmethod
  def backend_str(cls) -> str:
    return "tpu_dist"

  @classmethod
  def setUpClass(cls) -> None:
    # Upstream waits on workers (per test case and at teardown) for
    # `common_distributed.TIMEOUT_DEFAULT` seconds. Blaze has no per-test-case
    # timeout, so we effectively disable this one and let the BUILD target's
    # `timeout`/`size` bound the whole test. We also disallow
    # `DISTRIBUTED_TESTS_DEFAULT_TIMEOUT`, which would otherwise be ignored.
    #
    # `TIMEOUT_DEFAULT` is a module global, so `enterClassContext()` restores it
    # once this class is done (after `tearDownClass()`, which also uses it).
    if "DISTRIBUTED_TESTS_DEFAULT_TIMEOUT" in os.environ:
      raise ValueError(
          f"{cls.__name__} ignores `DISTRIBUTED_TESTS_DEFAULT_TIMEOUT`; set the"
          " BUILD target's `timeout`/`size` instead."
      )
    cls.enterClassContext(
        mock.patch.object(
            common_distributed, "TIMEOUT_DEFAULT", _TEST_CASE_TIMEOUT_SEC
        )
    )
    super().setUpClass()

  def setUp(self) -> None:
    # `common_utils.TestCase.setUp()` seeds the torch RNGs, which initializes
    # the PJRT client and claims the TPU devices that the workers need. To stop
    # this (so we can initialize the TPU devices ourselves), we mock
    # `set_rng_seed` to be a no-op. Then, workers call test methods directly and
    # are RNG seeded in `_run_test_given_id()`.
    with mock.patch.object(common_utils, "set_rng_seed", autospec=True):
      super().setUp()

  @classmethod
  def _spawn_processes(cls, world_size: int) -> None:
    distributed_utils.set_launch_env(nproc_per_node=world_size)
    # The TorchTPU runtime requires MASTER_ADDR and MASTER_PORT, and all ranks
    # must agree on them, so they are picked here and inherited by the workers.
    launch_env = {
        "MASTER_ADDR": os.environ.get("MASTER_ADDR", "localhost"),
        "MASTER_PORT": (
            os.environ["MASTER_PORT"]
            if "MASTER_PORT" in os.environ
            else str(portpicker.pick_unused_port())
        ),
    }
    with mock.patch.dict(os.environ, launch_env):
      super()._spawn_processes(world_size)

  @classmethod
  def _init_pg(cls, rank: int, world_size: int, rdvz_file: str | None) -> None:
    # Upstream PyTorch passes `rdvz_file` for file-based rendezvous, but we
    # rely on MASTER_ADDR/MASTER_PORT (env://) injected by `_spawn_processes`
    # instead. We `del` it here to suppress unused-argument linter warnings.
    del rdvz_file

    multiprocessing.parse_absl_flags()
    distributed_utils.set_worker_env(
        local_rank=rank,
        nproc_per_node=world_size,
    )
    dist.init_process_group(
        backend=cls.backend_str(),
        pg_options=cls.opts(),
        timeout=cls.timeout,
    )
    cls.pg = dist.distributed_c10d._get_default_group()

  @classmethod
  def _run_test_given_id(cls, test_id: str, **kwargs: Any) -> None:
    # Upstream `MultiProcContinuousTest` seeds every rank identically. Seed per
    # rank instead, like `dist_run()`.
    def seed_rngs_for_rank(seed: int | None = None) -> None:
      base_seed = cls.choose_seed() if seed is None else seed
      seed_test_utils.seed_rngs(base_seed + cls.rank)

    # Workers call test methods directly, so `SeededTest.setUp()`, which resets
    # Dynamo and lifts its recompile limit, only runs in the parent. Do the same
    # here: otherwise every compiled program of earlier tests stays in the
    # worker's Dynamo caches, and code objects shared across tests (such as the
    # wrapper of every `spmd_safe` function) hit the recompile limit, which is
    # a hard failure under `fullgraph=True`.
    torch._dynamo.reset()  # pylint: disable=protected-access
    try:
      with (
          mock.patch.object(common_utils, "set_rng_seed", seed_rngs_for_rank),
          torch._dynamo.utils.disable_cache_limit(),  # pylint: disable=protected-access
      ):
        super()._run_test_given_id(test_id, **kwargs)
    finally:
      # Fence between tests: no rank reports completion (and thus starts the
      # next test, or exits after the last one) while another rank may still
      # have collectives from this test in flight. This also runs when the test
      # fails, so that the other ranks don't block here waiting for this one.
      # Note: On `tpu_dist`, `dist.barrier()` executes an `allreduce` on a 1-element
      # `int64` tensor (`ProcessGroupTpu::barrier`). Running `dist.barrier()` in
      # `finally:` only avoids a hang if a rank fails *after* all collectives in
      # the test (as in `_FailsOnOneRank`). If a test runs multiple collectives and
      # one rank fails an assertion after the first collective while other ranks
      # proceed to the second collective, the failing rank will block in
      # `finally: dist.barrier()` before `_worker_loop` can catch the exception
      # and put it onto `completion_queue`.
      if dist.is_initialized():
        # Synchronize before and after the barrier to ensure any deferred work
        # or collectives don't fuse across test boundaries.
        torch.accelerator.synchronize()
        dist.barrier()
        torch.accelerator.synchronize()
      torch.tpu._clear_cache()

  @classmethod
  def _worker_loop(
      cls,
      rank: int,
      world_size: int,
      rdvz_file: str | None,
      task_queue: Any,
      completion_queue: Any,
  ) -> None:
    super()._worker_loop(
        rank,
        world_size,
        rdvz_file,
        task_queue,
        _ReportSkipsQueue(completion_queue),
    )

  def _worker_run_main_wait(self, fn: Callable[[], None]) -> Callable[[], None]:
    wait = super()._worker_run_main_wait(fn)
    if _PARENT_DRAINS_COMPLETION_QUEUES:
      return wait

    @functools.wraps(wait)
    def wrapper(self) -> None:
      try:
        wait()
      except unittest.SkipTest:
        if self.rank == self.MAIN_PROCESS_RANK:
          # The parent raised rank 0's skip; take the other ranks' results so
          # they don't answer for the next test.
          for completion_queue in self.completion_queues[1:]:
            completion_queue.get()
          # A worker that raised in a test skips `destroy_process_group()` and
          # may not exit cleanly, so `tearDownClass()` kills the workers.
          self.__class__._worker_skipped = True
        raise

    return types.MethodType(wrapper, self)

  @classmethod
  def tearDownClass(cls) -> None:
    if cls.__dict__.get("_processes_spawned", False) and (
        cls.poison_pill or cls.__dict__.get("_worker_skipped", False)
    ):
      # After a failure, workers may not exit cleanly (e.g. a failed rank skips
      # `destroy_process_group()` and blocks at exit waiting for its peers), so
      # `super().tearDownClass()` would wait on them forever.
      for process in cls.processes:
        process.kill()
    super().tearDownClass()
