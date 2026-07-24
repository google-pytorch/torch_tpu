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

"""Tests to verify XLA compile options overrides under different eager modes."""

import glob
import os

from absl.testing import absltest
import torch
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from torch_tpu._internal import execution_mode

EagerMode = execution_mode.EagerMode


class XlaCompileOptionsTest(absltest.TestCase):
  """Tests to ensure materialization worker threads use correct XLA compile options for specific eager modes."""

  def setUp(self):
    super().setUp()
    # Pop any env vars that affect global eager mode default.
    # See go/tt-knobs#eager-mode.
    os.environ.pop("TPU_LAUNCH_BLOCKING", None)
    os.environ.pop("TPU_DEFER_AND_FUSE", None)

    if "TEST_UNDECLARED_OUTPUTS_DIR" not in os.environ:
      os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = (
          self.create_tempdir().full_path
      )
    self._dump_dir = os.path.join(
        os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "xla_dump"
    )
    os.environ["XLA_FLAGS"] = (
        f"--xla_dump_to={self._dump_dir} --xla_dump_hlo_as_text"
    )

  def _read_hlo_module_config(self, module_name: str) -> str:
    """Reads the generated hlo_module_config.txt for the given module name."""
    config_paths = glob.glob(
        os.path.join(
            self._dump_dir, f"module_*.{module_name}*.hlo_module_config.txt"
        )
    )
    self.assertLen(
        config_paths,
        1,
        f"Expected exactly one `module_*.{module_name}*.hlo_module_config.txt`"
        f" path but found {config_paths}",
    )

    with open(config_paths[0], "r") as f:
      return f.read()

  def test_defer_never_eager_mode(self):
    """Tests that the DEFER_NEVER eager mode overrides XLA optimization level to O1."""
    device = torch.device("tpu")

    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      self.assertEqual(execution_mode.eager_mode, EagerMode.DEFER_NEVER)

      x = torch.randn(2, 2, device=device)
      w = torch.randn(2, 2, device=device)
      y = x @ w
      _ = y.cpu()

    # Get HLO module config generated for matmul.
    hlo_module_config: str = self._read_hlo_module_config("tt_jit_mm")
    self.assertIn("\noptimization_level: EFFORT_O1\n", hlo_module_config)


if __name__ == "__main__":
  absltest.main()
