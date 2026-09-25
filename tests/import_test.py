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

"""Tests for TorchTPU module import handling and ordering."""

import os
import sys
import traceback
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu._internal.distributed import multiprocessing


def _worker_import_torch_tpu(q):
  try:
    import torch_tpu  # pylint: disable=unused-import  # noqa: F401

    q.put("OK")
  except Exception:
    q.put(traceback.format_exc())


def _worker_import_torch(q):
  try:
    import torch  # pylint: disable=unused-import  # noqa: F401

    q.put("OK")
  except Exception:
    q.put(traceback.format_exc())


def _worker_import_torch_then_torch_tpu(q):
  try:
    import torch  # pylint: disable=unused-import  # noqa: F401
    import torch_tpu  # pylint: disable=unused-import  # noqa: F401

    q.put("OK")
  except Exception:
    q.put(traceback.format_exc())


def _worker_import_torch_tpu_then_torch(q):
  try:
    import torch_tpu  # pylint: disable=unused-import  # noqa: F401
    import torch  # pylint: disable=unused-import  # noqa: F401

    q.put("OK")
  except Exception:
    q.put(traceback.format_exc())


def _worker_import_torch_tpu_skips_loader(q):
  """Checks that `import torch_tpu` doesn't claim PrivateUse1 without autoload."""
  try:
    import torch  # pylint: disable=unused-import,g-import-not-at-top  # noqa: F401
    import torch_tpu  # pylint: disable=unused-import,g-import-not-at-top  # noqa: F401

    loaded = [
        name
        for name in (
            "torch_tpu._loader",
            "torch_tpu._internal.device._device_ops_backend",
        )
        if name in sys.modules
    ]
    q.put(f"Unexpectedly imported: {loaded}" if loaded else "OK")
  except Exception:
    q.put(traceback.format_exc())


def _run_worker(target_fn, q, env_vars):
  if env_vars:
    os.environ.update(env_vars)
  target_fn(q)


class ImportTest(parameterized.TestCase):  # ABSLTEST_OK=Import test

  def _run_in_isolated_process(self, target_fn, env_vars=None):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()

    p = ctx.Process(target=_run_worker, args=(target_fn, q, env_vars))
    p.start()
    p.join()
    res = q.get() if not q.empty() else "NO RESULT"
    self.assertEqual(p.exitcode, 0, f"Process exited with {p.exitcode}:\n{res}")
    self.assertEqual(res, "OK", f"Import failed:\n{res}")

  @parameterized.product(
      worker_fn=(
          _worker_import_torch_tpu,
          _worker_import_torch,
          _worker_import_torch_then_torch_tpu,
          _worker_import_torch_tpu_then_torch,
      ),
      disable_autoload=(False, True),
  )
  def test_imports(
      self,
      worker_fn,
      disable_autoload: bool,
  ) -> None:
    """Verifies module import resilience under various autoload configurations."""
    env_vars = {}
    if disable_autoload:
      env_vars["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    self._run_in_isolated_process(
        worker_fn,
        env_vars=env_vars,
    )

  def test_import_without_autoload_skips_loader(self) -> None:
    """Verifies `import torch_tpu` leaves PrivateUse1 free without autoload."""
    with mock.patch.dict(os.environ, {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}):
      self._run_in_isolated_process(_worker_import_torch_tpu_skips_loader)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
