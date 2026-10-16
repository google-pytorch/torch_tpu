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

"""Unit tests for noxfile sessions."""

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_CI_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _CI_TOOLS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import noxfile


class NoxfileTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):

  def test_wheel_test_sets_pythonpath_to_repo_root(self):
    with tempfile.TemporaryDirectory() as tmp:
      temp_dir = pathlib.Path(tmp)
      fake_wheel = temp_dir / "torch_tpu-0.2.0-cp311-cp311-linux_x86_64.whl"
      fake_wheel.write_bytes(b"wheel")

      session = mock.MagicMock()
      session.posargs = ["-k", "WheelStructureTest"]

      with (
          mock.patch.object(noxfile, "_dist_wheel", return_value=fake_wheel),
          mock.patch.object(
              noxfile, "_newest_supported_torch", return_value="2.15.0"
          ),
          mock.patch.object(noxfile, "__file__", str(temp_dir / "noxfile.py")),
      ):
        noxfile.wheel_test(session)

      session.run.assert_called_once()
      run_args, run_kwargs = session.run.call_args
      self.assertEqual(run_args[0], "pytest")
      self.assertIn("-k", run_args)
      env = run_kwargs["env"]
      self.assertEqual(env["PYTHONPATH"], str(temp_dir.resolve()))
      self.assertEqual(env["WHEEL_PATH"], str(fake_wheel.resolve()))
      self.assertEqual(
          env["PYPROJECT_PATH"], str((temp_dir / "pyproject.toml").resolve())
      )


if __name__ == "__main__":
  unittest.main()
