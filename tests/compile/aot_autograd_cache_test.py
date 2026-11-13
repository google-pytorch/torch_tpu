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

"""AOTAutogradCache integration tests for the TPU backend.

The parent test (`ParentAotAutogradCacheTest`, run when `--test_mode` is unset)
spawns one isolated worker subprocess per simulated process start with the
cache environment it wants, and the worker (`AotAutogradCacheTest`, selected
via `--test_mode`) compiles with the TPU backend. See
`torch_tpu.tests.subprocess_test_utils` for the shared scaffolding.
"""

from collections.abc import Sequence
import enum
import glob
import os
import shutil
import uuid

from absl import flags
from absl.testing import absltest
import torch
from torch._dynamo.utils import counters
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests import subprocess_test_utils


class TestMode(enum.Enum):
  """Enum for the different test modes."""

  FALLBACK = "fallback"
  MEDIUM = "medium"
  HIGHEST = "highest"
  ASYNC_HIGHEST_CONTEXT = "async_highest_context"
  HIGHEST_CONTEXT = "highest_context"


# The --test_mode flag controls whether the process runs as the parent test
# (when None) or as a worker subprocess running a specific
# AotAutogradCacheTest mode (when set to FALLBACK).
_TEST_MODE = flags.DEFINE_enum_class(
    "test_mode", None, TestMode, "The test mode to run."
)


class AotAutogradCacheTest(seed_test_utils.RepeatableTest):
  """Tests that run only inside worker subprocesses with `--test_mode` set."""

  def setUp(self) -> None:
    # When running in the parent process, --test_mode is None. Skip these tests
    # so that only ParentAotAutogradCacheTest executes in the parent process.
    if _TEST_MODE.value is None:
      self.skipTest("test_mode flag not provided.")
    super().setUp()
    tt_testing.reset_eager_state()

  def test_fallback_key_entry_stays_in_memory(self) -> None:
    """Compiles without a torch AOTAutogradCache key, like vLLM does."""
    self.assertEqual(_TEST_MODE.value, TestMode.FALLBACK)
    # With torch's autograd cache disabled torch provides no cache key, so the
    # backend builds the bundled entry under its own `torchtpu_` fallback key.
    self.assertEqual(os.environ.get("TORCHINDUCTOR_AUTOGRAD_CACHE"), "0")
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(64),
        torch.nn.Linear(64, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 64),
    ).eval()
    x = torch.randn(4, 16, 64)
    expected = model(x)
    # float32 matmuls run as bfloat16 passes on TPU by default.
    torch.set_float32_matmul_precision("highest")
    device = torch.accelerator.current_accelerator()
    x = x.to(device)
    with torch.inference_mode():
      # Async compile offloads the AOTAutogradCache save to a background
      # thread; the warmup resolves the compilation and flushes those saves.
      compiled = tpu_compile.async_compile(model.to(device), [(x,)])
      out = compiled(x)
    utils.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)


# Under DEFAULT (bf16 math), squaring this matrix rounds the diagonal to 1.0.
# Under HIGHEST (f32 math), the diagonal keeps the bits below bf16 precision.
_EXPECTED_HIGHEST = 1.0 + 2**-7 + 2**-16


class AotAutogradCachePrecisionTest(seed_test_utils.RepeatableTest):
  """Compiles one graph at the matmul precision named by `--test_mode`."""

  def setUp(self) -> None:
    if _TEST_MODE.value in (None, TestMode.FALLBACK):
      self.skipTest("test_mode flag not provided.")
    super().setUp()
    tt_testing.reset_eager_state()
    device = torch.accelerator.current_accelerator()
    self.a = torch.eye(256, device=device) * (1 + 2**-8)
    counters.clear()

  def _assert_cache_lookup(self) -> None:
    # "hit" or "miss", set by the parent for this process start.
    lookup = os.environ["AOT_AUTOGRAD_CACHE_TEST_EXPECTED_LOOKUP"]
    self.assertEqual(counters["aot_autograd"][f"autograd_cache_{lookup}"], 1)

  def _compile_at_precision(self) -> None:
    mode = _TEST_MODE.value
    assert mode is not None
    precision = mode.value
    expected = _EXPECTED_HIGHEST if precision == "highest" else 1.0
    saved = torch.get_float32_matmul_precision()
    try:
      torch.set_float32_matmul_precision(precision)
      with torch.inference_mode():
        actual = torch.compile(lambda a: a @ a)(self.a)[0, 0].item()
    finally:
      torch.set_float32_matmul_precision(saved)
    self.assertEqual(actual, expected)
    self._assert_cache_lookup()

  def test_medium_precision_program(self) -> None:
    self._compile_at_precision()

  def test_highest_precision_program(self) -> None:
    self._compile_at_precision()

  def test_async_highest_context_precision_program(self) -> None:
    # The worker thread that lowers this program does not see the HIGHEST
    # context, so it lowers with the global DEFAULT precision.
    with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
      tpu_compile.async_compile(lambda a: a @ a, [(self.a,)])
    self._assert_cache_lookup()

  def test_highest_context_precision_program(self) -> None:
    # The same graph and grad mode as the async start, compiled synchronously.
    with torch.tpu.precision(torch.tpu.Precision.HIGHEST), torch.no_grad():
      compiled = torch.compile(
          lambda a: a @ a, backend="tpu", fullgraph=True, dynamic=False
      )
      actual = compiled(self.a)[0, 0].item()
    self.assertEqual(actual, _EXPECTED_HIGHEST)
    self._assert_cache_lookup()


class ParentAotAutogradCachePrecisionTest(
    subprocess_test_utils.SubprocessTestMixin,
    seed_test_utils.MultiProcessRepeatableTest,
):
  """Process starts at different precisions share one AOTAutogradCache dir."""

  WORKER_TEST_METHOD_TEMPLATE = (
      "AotAutogradCachePrecisionTest.test_{mode.value}_precision_program"
  )

  def setUp(self) -> None:
    super().setUp()
    tt_testing.reset_eager_state()
    if _TEST_MODE.value is not None:
      self.skipTest("Skipping parent test in sub-test mode.")

  def _run_process_starts(
      self, starts: Sequence[tuple[TestMode, str]]
  ) -> None:
    """Runs one process start per (mode, expected lookup) on one cache dir."""
    env_updates = {
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "TORCHINDUCTOR_CACHE_DIR": self.create_tempdir().full_path,
        "TORCH_TPU_TIER2_COMPILATION_CACHE": "disabled",
    }
    for mode, lookup in starts:
      self.run_sub_test(
          mode,
          env_updates={
              **env_updates,
              "AOT_AUTOGRAD_CACHE_TEST_EXPECTED_LOOKUP": lookup,
          },
      )

  def test_precision_is_part_of_the_cache_key(self) -> None:
    """A warm cache entry from one precision must not serve another."""
    # The highest start must miss the medium entry, and the second medium start
    # must still hit the first one.
    self._run_process_starts((
        (TestMode.MEDIUM, "miss"),
        (TestMode.HIGHEST, "miss"),
        (TestMode.MEDIUM, "hit"),
    ))

  def test_async_compile_entry_keys_on_lowering_precision(self) -> None:
    """An async compile under a context must not serve a sync one under it."""
    # The async start lowers with the global DEFAULT precision, so the
    # synchronous start under the same HIGHEST context must miss its entry.
    self._run_process_starts((
        (TestMode.ASYNC_HIGHEST_CONTEXT, "miss"),
        (TestMode.HIGHEST_CONTEXT, "miss"),
    ))


class ParentAotAutogradCacheTest(
    subprocess_test_utils.SubprocessTestMixin,
    seed_test_utils.MultiProcessRepeatableTest,
):
  """Spawns isolated worker subprocesses, one per simulated process start."""

  WORKER_TEST_METHOD_TEMPLATE = (
      "AotAutogradCacheTest.test_{mode.value}_key_entry_stays_in_memory"
  )

  def setUp(self) -> None:
    super().setUp()
    tt_testing.reset_eager_state()
    # In worker subprocesses (--test_mode is set), skip the parent test so only
    # the target AotAutogradCacheTest method is executed.
    if _TEST_MODE.value is not None:
      self.skipTest("Skipping parent test in sub-test mode.")

  def _create_tier2_cache_name(self) -> str:
    """Returns a Tier-2 compilation cache name that no other process uses."""
    name = f"aot_autograd_cache_test_{uuid.uuid4().hex}"
    self.addCleanup(
        shutil.rmtree,
        os.path.join("/dev/shm/torch_tpu_cache", name),
        ignore_errors=True,
    )
    return name

  def test_fallback_entries_not_persisted_across_process_starts(self) -> None:
    """Two process starts sharing a cache dir leave no `torchtpu_` entries."""
    cache_dir = self.create_tempdir().full_path
    env_updates = {
        # No torch AOTAutogradCache key: the backend uses its `torchtpu_` key.
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
        "TORCHINDUCTOR_CACHE_DIR": cache_dir,
        "TORCH_TPU_TIER2_COMPILATION_CACHE": self._create_tier2_cache_name(),
    }
    # google-pytorch/torch_tpu#4892: every process start used to add one unread
    # `aotautograd/torchtpu_*/<hash>` file per compile (torch never looks the key
    # up and the file name hashes a per-start timestamp), so a shared cache dir
    # grew without bound. Asserting after both children exited also rules out a
    # race with their background cache saves.
    for _ in range(2):
      self.run_sub_test(TestMode.FALLBACK, env_updates=env_updates)
    self.assertEmpty(
        glob.glob(os.path.join(cache_dir, "aotautograd", "*", "*"))
    )


if __name__ == "__main__":
  # Use multiprocessing.handle_test_main to ensure compatibility with
  # absltest in multi-process Google3 test environments.
  multiprocessing.handle_test_main(absltest.main)
