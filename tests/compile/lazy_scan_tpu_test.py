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

"""Tests the scan HOPs on TPU before `compile.scan` has been imported.

Device init registers the scan HOPs through trampolines that import
`torch_tpu._internal.compile.scan` on first use. Each test runs in a fresh
subprocess so that nothing has imported that module yet.
"""

import os
import subprocess
import sys

from absl.testing import absltest
import torch

_PREAMBLE = """\
import sys
import torch
from torch_tpu import _loader

_loader._init_device("tpu")
assert "torch_tpu._internal.compile.scan" not in sys.modules
"""


class LazyScanTpuTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    if torch.accelerator.device_count() == 0:
      self.skipTest("Requires TPU hardware")

  def _run(self, body: str) -> None:
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run(
        [sys.executable, "-c", _PREAMBLE + body],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    self.assertEqual(
        result.returncode,
        0,
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )

  def test_eager_associative_scan_goes_through_trampoline(self) -> None:
    self._run("""\
from torch._higher_order_ops import associative_scan

x = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
actual = associative_scan(lambda a, b: a + b, x.to("tpu"), dim=1)
assert "torch_tpu._internal.compile.scan" in sys.modules
torch.testing.assert_close(actual.cpu(), torch.cumsum(x, dim=1))
""")

  def test_compiled_scan_imports_scan_before_tracing(self) -> None:
    self._run("""\
from torch._higher_order_ops import scan

def fn(carry, x):
  return carry + x, carry + x

x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="tpu")
init = torch.tensor([0.0], device="tpu")
compiled = torch.compile(scan, backend="tpu", dynamic=False)
carry, ys = compiled(fn, init, x, dim=0)
torch.testing.assert_close(carry.cpu(), torch.tensor([10.0]))
torch.testing.assert_close(
    ys.cpu(), torch.tensor([[1.0], [3.0], [6.0], [10.0]])
)
""")


if __name__ == "__main__":
  absltest.main()
