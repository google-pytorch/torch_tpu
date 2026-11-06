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

import enum
import glob
import os
import shutil
import uuid

from absl import flags
from absl.testing import absltest
import torch
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests import subprocess_test_utils


class TestMode(enum.Enum):
  """Enum for the different test modes."""

  FALLBACK = "fallback"


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
