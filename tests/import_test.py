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

import collections
from collections.abc import Mapping, Set
import importlib
import importlib.util
import json
import os
import pkgutil
import queue
import subprocess
import sys
import traceback
import types
from typing import Final
import warnings

from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu._internal.distributed import multiprocessing

# Optional third-party packages declared under `[project.optional-dependencies]`
# in `pyproject.toml` that may not be installed in minimal wheel smoke-test
# environments.
_OPTIONAL_EXTERNAL_DEPS: Final[frozenset[str]] = frozenset({
    "jax",
    "tensorboardX",
})

# Sanity check to ensure that some work was done when walking and importing
# TorchTPU modules. Chosen arbitrarily to be lower than the actual number of
# modules under the assumption that modules are not likely to be removed.
_MIN_EXPECTED_MODULES: Final[int] = 20


def _is_missing_optional_dep(exc: BaseException) -> bool:
  """Returns True if `exc` was caused by an uninstalled optional package."""
  curr: BaseException | None = exc
  while curr is not None:
    if isinstance(curr, ModuleNotFoundError) and curr.name:
      root_pkg = curr.name.partition(".")[0]
      if (
          root_pkg in _OPTIONAL_EXTERNAL_DEPS
          and importlib.util.find_spec(root_pkg) is None
      ):
        return True
    curr = curr.__cause__ or curr.__context__
  return False


def _discover_torch_tpu_modules(package: types.ModuleType) -> Set[str]:
  """Discovers all importable modules and subpackages under `package`."""

  def _on_error(unused_name: str) -> None:
    exc = sys.exc_info()[1]
    if exc is not None and not _is_missing_optional_dep(exc):
      raise exc

  module_names: set[str] = {package.__name__}
  for info in pkgutil.walk_packages(
      getattr(package, "__path__", ()),
      prefix=f"{package.__name__}.",
      onerror=_on_error,
  ):
    module_names.add(info.name)
  return module_names


def _worker_import_torch_tpu(q):
  try:
    import torch_tpu  # pylint: disable=unused-import  # ruff: ignore[unused-import]

    q.put("OK")
  except Exception:
    q.put(traceback.format_exc())


def _worker_import_torch(q):
  try:
    import torch  # pylint: disable=unused-import  # ruff: ignore[unused-import]

    q.put("OK")
  except Exception:
    q.put(traceback.format_exc())


def _worker_import_torch_then_torch_tpu(q):
  try:
    import torch  # pylint: disable=unused-import  # ruff: ignore[unused-import]
    import torch_tpu  # pylint: disable=unused-import  # ruff: ignore[unused-import]

    q.put("OK")
  except Exception:
    q.put(traceback.format_exc())


def _worker_import_torch_tpu_then_torch(q):
  try:
    import torch_tpu  # pylint: disable=unused-import  # ruff: ignore[unused-import]
    import torch  # pylint: disable=unused-import  # ruff: ignore[unused-import]

    q.put("OK")
  except Exception:
    q.put(traceback.format_exc())


def _worker_import_all_submodules(q: queue.Queue[str]) -> None:
  try:
    import torch  # pylint: disable=unused-import  # ruff: ignore[unused-import]
    import torch_tpu

    module_names = _discover_torch_tpu_modules(torch_tpu) | {"torch.tpu"}

    pending_modules: collections.deque[types.ModuleType] = collections.deque()
    for mod_name in sorted(module_names):
      try:
        pending_modules.append(importlib.import_module(mod_name))
      except ImportError as exc:
        if _is_missing_optional_dep(exc):
          continue
        raise

    visited: set[int] = set()
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      while pending_modules:
        mod = pending_modules.popleft()
        if id(mod) in visited:
          continue
        visited.add(id(mod))

        stages = getattr(mod, "__tt_api_stages__", None)
        if not isinstance(stages, Mapping):
          stages = {}

        attr_names = sorted(set(dir(mod)) | set(getattr(mod, "__all__", ())))
        for attr_name in attr_names:
          if attr_name.startswith("__") and attr_name.endswith("__"):
            continue
          # `_DeviceModuleMeta.__getattr__` invokes `val.__get__(None, cls)` on
          # property-like descriptors in `__tt_api_stages__` (such as
          # `_DefaultGeneratorsProperty` for `torch.tpu.default_generators`),
          # which queries active TPU device state and fails on CPU-only test
          # runners. Exclude plain Python functions (`types.FunctionType`),
          # which also implement `__get__` as non-data descriptors.
          stage_info = stages.get(attr_name)
          stage_val = getattr(stage_info, "value", None)
          if (
              stage_val is not None
              and not isinstance(stage_val, types.FunctionType)
              and hasattr(stage_val, "__get__")
          ):
            continue
          try:
            val = getattr(mod, attr_name)
          except ImportError as exc:
            if _is_missing_optional_dep(exc):
              continue
            raise
          if isinstance(val, types.ModuleType) and getattr(
              val, "__name__", ""
          ).startswith(("torch_tpu.", "torch.tpu.")):
            if id(val) not in visited:
              pending_modules.append(val)

    if len(visited) < _MIN_EXPECTED_MODULES:
      raise AssertionError(
          f"Expected to visit at least {_MIN_EXPECTED_MODULES} TorchTPU"
          f" modules, got {len(visited)}"
      )

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
          _worker_import_all_submodules,
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

  def test_autoload_does_not_load_heavy_modules(self) -> None:
    """Verifies plain `import torch` does not load Dynamo, Inductor, etc."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys, torch\n"
            "assert 'torch_tpu._loader' in sys.modules, "
            "'torch_tpu._loader not loaded'\n"
            "forbidden = [\n"
            "    'torch._dynamo',\n"
            "    'torch._inductor',\n"
            "    'torch_tpu._internal.compile._backend',\n"
            "    'zmq',\n"
            "]\n"
            "loaded = [m for m in forbidden if m in sys.modules]\n"
            "print(json.dumps({'loaded': loaded}))\n"
            "if loaded:\n"
            "    sys.exit(1)\n"
        ),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        res.returncode,
        0,
        f"Plain import torch loaded forbidden modules:\nstdout:"
        f" {res.stdout}\nstderr: {res.stderr}",
    )
    data = json.loads(res.stdout.strip())
    self.assertEqual(data["loaded"], [])


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
