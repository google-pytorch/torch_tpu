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

"""Tests persistent Tier-2 and Tier-3 cache for torch.compile."""

import enum
import io
import os
import pathlib
import pickle
import queue
import shutil
import sys
import time
from absl import flags
from absl import logging
from absl.testing import absltest
import torch
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


class TestMode(enum.Enum):
  """Enum for the different test modes."""

  TIER2 = "tier2"
  TIER3 = "tier3"
  TIER3_PICKLE_BY_REF = "tier3_pickle_by_ref"


_TEST_MODE = flags.DEFINE_enum_class(
    "test_mode", None, TestMode, "The test mode to run."
)


def get_tier2_cache_dir(tier2_cache_name: str) -> pathlib.Path:
  """Returns the Tier-2 cache directory."""
  return pathlib.Path("/dev/shm/torch_tpu_cache") / tier2_cache_name


def clear_in_memory_caches() -> None:
  """Clears the in-memory caches."""
  torch.compiler.reset()
  torch.tpu._clear_cache()


def empty_dir(dir_path: pathlib.Path) -> None:
  """Empties the given directory."""
  try:
    shutil.rmtree(dir_path)
  except FileNotFoundError:
    pass
  dir_path.mkdir(parents=True, exist_ok=True)


def list_bin_files(dir_path: pathlib.Path) -> list[pathlib.Path]:
  """Lists all .bin files in the given directory and its subdirectories."""
  return list(dir_path.glob("**/*.bin"))


class CacheTest(absltest.TestCase):
  """Tests persistent Tier-2 and Tier-3 cache for torch.compile."""

  def setUp(self) -> None:
    super().setUp()
    tt_testing.reset_eager_state()
    if _TEST_MODE.value is None:
      self.skipTest("test_mode flag not provided.")

  def _run_persistent_cache_logic(self, cache_dir: pathlib.Path | str):

    device = torch.accelerator.current_accelerator()

    def get_args() -> tuple[torch.Tensor, torch.Tensor]:
      return (
          torch.ones((2, 2), device=device),
          torch.ones((2, 2), device=device),
      )

    # Warmup/initialize by calling args once to populate any initial cache
    # caused by tensor creation, generator initialization, or other
    # torch_tpu initializations.
    for arg in get_args():
      _ = arg.cpu()
    _ = torch.tpu.get_rng_state().cpu()

    clear_in_memory_caches()

    # Record files before the first compile.
    initial_caches = set(list_bin_files(cache_dir))
    logging.info("Initial caches: %s", initial_caches)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      return torch.sin(x + y) * torch.cos(x - y)

    with self.subTest("TestCachePopulation"):
      # Assert that new cache files were written after the first compile (
      # building the cache for torch.compile).
      with torch.inference_mode():
        fn(*get_args())

      caches_after_first_compile = set(list_bin_files(cache_dir))
      logging.info("Caches after first compile: %s", caches_after_first_compile)
      self.assertNotEmpty(
          caches_after_first_compile - initial_caches,
          "No new caches written after first compile",
      )

    # Clear in-memory cache to force a persistent cache lookup.
    clear_in_memory_caches()

    with self.subTest("TestCacheHit"):
      # Assert that no new cache files were written (cache hit in second
      # compile).
      with torch.inference_mode():
        fn(*get_args())

      caches_after_second_compile = set(list_bin_files(cache_dir))
      logging.info(
          "Caches after second compile: %s", caches_after_second_compile
      )
      self.assertEmpty(
          caches_after_second_compile - caches_after_first_compile,
          "New caches written after second compile",
      )

  def test_persistent_cache_tier2(self) -> None:
    self.assertEqual(_TEST_MODE.value, TestMode.TIER2)
    cache_dir = get_tier2_cache_dir(
        os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE")
    )
    empty_dir(cache_dir)
    self._run_persistent_cache_logic(cache_dir)

  def test_persistent_cache_tier3(self) -> None:
    self.assertEqual(_TEST_MODE.value, TestMode.TIER3)
    cache_dir = pathlib.Path(
        os.environ.get("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT")
    )
    empty_dir(cache_dir)
    self._run_persistent_cache_logic(cache_dir)

  def test_persistent_cache_tier3_pickle_by_ref(self) -> None:
    """Executables pickle as a cache-key reference when tier-3 is enabled.

    With TORCH_TPU_INTERNAL_PICKLE_EXECUTABLES_BY_CACHE_KEY=1 and a tier-3
    cache entry present, pickling a TorchTpuCompiledExecutable must embed only
    the compact cache key (a "link") instead of a second copy of the
    serialized executable bytes; unpickling must load the executable back from
    the tiered cache, and must fail once the tiered caches are emptied.
    """
    self.assertEqual(_TEST_MODE.value, TestMode.TIER3_PICKLE_BY_REF)
    tier3_dir = pathlib.Path(
        os.environ.get("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT")
    )
    tier2_dir = get_tier2_cache_dir(
        os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE")
    )
    empty_dir(tier3_dir)
    empty_dir(tier2_dir)

    device = torch.accelerator.current_accelerator()
    x = torch.ones((2, 2), device=device)
    y = torch.ones((2, 2), device=device)
    # Warmup/initialize so that device init compilations do not interfere.
    _ = x.cpu()
    _ = torch.tpu.get_rng_state().cpu()
    clear_in_memory_caches()

    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
      return torch.sin(a + b) * torch.cos(a - b)

    gm = dynamo_graph_capture_for_export(fn)(x, y)
    backend = _backend.TpuBackend()
    compiled_fn = backend(gm, [x, y])
    expected = compiled_fn(x, y)

    executable = backend._compiled_executables[-1]
    cache_key = tpu_torch_compile.find_executable_cache_key(
        executable._executable
    )
    self.assertIsNotNone(
        cache_key,
        "compiled executable was not found in the in-memory compilation cache",
    )

    # The tier-2/tier-3 writes are best-effort and may complete slightly
    # after the compile returns; wait for the entry to land so that
    # __reduce__ takes the by-reference path.
    deadline = time.monotonic() + 60
    while (
        not tpu_torch_compile.serialized_executable_exists_in_tiered_cache(
            cache_key
        )
        and time.monotonic() < deadline
    ):
      time.sleep(0.5)
    self.assertTrue(
        tpu_torch_compile.serialized_executable_exists_in_tiered_cache(
            cache_key
        ),
        "tiered cache entry never appeared for the compiled executable",
    )

    with self.subTest("PickleIsAReferenceNotACopy"):
      pickled = pickle.dumps(executable)
      full_bytes = tpu_torch_compile.serialize_executable(
          executable._executable
      )
      self.assertLess(
          len(pickled),
          len(full_bytes) // 2,
          "by-reference pickle is not substantially smaller than the "
          "serialized executable; the serialized bytes were likely embedded",
      )

    with self.subTest("UnpickleLoadsFromTieredCache"):
      restored = pickle.loads(pickled)
      restored_out = restored(x, y)
      torch.testing.assert_close(
          _backend.to_device(restored_out, "cpu"),
          _backend.to_device(expected, "cpu"),
      )

    with self.subTest("UnpickleFailsAfterCacheEviction"):
      empty_dir(tier3_dir)
      empty_dir(tier2_dir)
      with self.assertRaises(Exception):
        pickle.loads(pickled)


def sub_test_worker_entry(
    q: queue.Queue[str],
    mode: TestMode,
    env_updates: dict[str, str],
) -> None:
  """Serves as the entry point for the worker process.

  Args:
    q: A multiprocessing.Queue to send stdout/stderr from the worker.
    mode: The TestMode (e.g., TIER2, TIER3) for the sub-test.
    env_updates: A dictionary of environment variables to update in the
      subprocess.
  """

  # Update env in the child process.
  os.environ.update(env_updates)

  class QueueWriter:
    """A helper to write stdout/stderr to a queue for multiprocessing."""

    def __init__(self, q: queue.Queue[str]):
      self.q = q

    def write(self, s: str | None) -> None:
      if s:
        self.q.put(s)

    def flush(self) -> None:
      del self

    def fileno(self) -> None:
      del self
      raise io.UnsupportedOperation("QueueWriter does not have a fileno")

  qw = QueueWriter(q)
  sys.stdout = qw
  sys.stderr = qw

  # Target the specific test method based on the mode.
  sys.argv = [
      sys.argv[0],
      f"--test_mode={mode.name}",
      f"CacheTest.test_persistent_cache_{mode.value}",
  ]

  absltest.main()


class MasterCacheTest(absltest.TestCase):
  """A master test that spawns subprocesses for isolation."""

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    if _TEST_MODE.value is not None:
      self.skipTest("Skipping master test in sub-test mode.")

  def _run_sub_test(self, mode: TestMode, env_updates: dict[str, str]) -> None:
    """Runs a sub-test in a separate process."""

    logging.info("Running test in a subprocess for isolation.")
    ctx = g3_multiprocessing.get_context(g3_multiprocessing.ABSL_SPAWN)
    q = ctx.Queue()
    p = ctx.Process(target=sub_test_worker_entry, args=(q, mode, env_updates))
    p.start()

    while p.is_alive() or not q.empty():
      try:
        output = q.get(timeout=0.1)
        sys.stderr.write(output)
        sys.stderr.flush()
      except queue.Empty:
        continue

    p.join()
    if p.exitcode != 0:
      self.fail(
          f"test_persistent_cache_{mode} failed with exit code {p.exitcode}"
      )

  def test_cache_tier2(self) -> None:
    self._run_sub_test(
        TestMode.TIER2,
        {
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": (
                "compile_cache_test_tier2"
            ),
        },
    )

  def test_cache_tier3(self) -> None:
    self._run_sub_test(
        TestMode.TIER3,
        {
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "default",
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": str(
                self.create_tempdir().full_path
            ),
        },
    )

  def test_cache_tier3_pickle_by_ref(self) -> None:
    self._run_sub_test(
        TestMode.TIER3_PICKLE_BY_REF,
        {
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": (
                "compile_cache_test_pickle_by_ref"
            ),
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": str(
                self.create_tempdir().full_path
            ),
            "TORCH_TPU_INTERNAL_PICKLE_EXECUTABLES_BY_CACHE_KEY": "1",
        },
    )


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
