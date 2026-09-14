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

"""Regression test for clean interpreter teardown after torch.compile (#3218)."""

import os
import resource
import subprocess
import sys
from absl.testing import absltest


def _worker():
  try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
  except Exception:  # pylint: disable=broad-except
    pass

  import torch
  import torch_tpu  # noqa: F401

  @torch.compile(backend="tpu", dynamic=False, fullgraph=True)
  def add_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

  device = torch.device("tpu")
  x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
  y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device=device)

  out = add_fn(x, y)
  torch.accelerator.synchronize()

  expected = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32)
  assert torch.equal(out.cpu(), expected), (
      f"Expected {expected}, got {out.cpu()}"
  )


class CompileTeardownTest(absltest.TestCase):

  def test_clean_interpreter_teardown(self):
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(sys.path)
    env["TPU_PREMAPPED_BUFFER_SIZE"] = "0"
    env["_COMPILE_TEARDOWN_WORKER"] = "1"

    cmd = [sys.argv[0]]
    if sys.argv[0].endswith(".py"):
      cmd = [sys.executable, sys.argv[0]]

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    self.assertEqual(
        proc.returncode,
        0,
        "Worker process crashed or exited abnormally with returncode"
        f" {proc.returncode}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
    )


if __name__ == "__main__":
  if os.environ.get("_COMPILE_TEARDOWN_WORKER") == "1":
    _worker()
  else:
    absltest.main()
