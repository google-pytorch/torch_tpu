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
import traceback

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


def _worker_import_torch_tpu_extras(q):
  try:
    import torch_tpu.extras  # pylint: disable=unused-import  # noqa: F401
    import torch_tpu.extras.optimizers  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras import optimizers  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import AdamW  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import Muon  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import Optimizer  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import ReferenceAdamw  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import ReferenceMuon  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import ReferenceSgd  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import SGD  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import SingleTraceTrainer  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import TorchAdamw  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import TorchMuon  # pylint: disable=unused-import  # noqa: F401
    from torch_tpu.extras.optimizers import TorchSgd  # pylint: disable=unused-import  # noqa: F401

    q.put("OK")
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

  @parameterized.parameters(False, True)
  def test_extras_optimizers_import(self, disable_autoload: bool) -> None:
    """Verifies that torch_tpu.extras and optimizers are importable."""
    env_vars = {}
    if disable_autoload:
      env_vars["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    self._run_in_isolated_process(
        _worker_import_torch_tpu_extras,
        env_vars=env_vars,
    )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
