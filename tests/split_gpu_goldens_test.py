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

"""Unit tests for split_gpu_goldens."""

import gzip
import pathlib
import plistlib
from absl.testing import absltest
from tests import split_gpu_goldens


class SplitGpuGoldensTest(absltest.TestCase):  # ABSLTEST_OK=tool test

  def test_split_goldens(self):
    temp_dir = pathlib.Path(self.create_tempdir())
    input_file1 = temp_dir / "golden1.gz"
    input_file2 = temp_dir / "golden2.gz"
    hopper_file = temp_dir / "golden_hopper0.gz"
    out_dir = temp_dir / "out"

    data1 = {
        "test_add": {"base": {"float32": [("inp1", "out1")]}},
        "test_sub": {"base": {"float32": [("inp2", "out2")]}},
    }
    data2 = {
        "test_mul": {"base": {"float32": [("inp3", "out3")]}},
        # Tests that multiple files with the same key extend sample lists.
        "test_sub": {"base": {"float32": [("inp4", "out4")]}},
    }
    hopper_data = {
        "test_hopper_op": {"base": {"float32": [("inp_hopper", "out_hopper")]}},
    }

    for path, data in [
        (input_file1, data1),
        (input_file2, data2),
        (hopper_file, hopper_data),
    ]:
      with gzip.open(path, "wb") as f:
        plistlib.dump(data, f, fmt=plistlib.FMT_BINARY)

    count = split_gpu_goldens.split_goldens(
        base_input_files=[input_file1, input_file2],
        output_dir=out_dir,
        hopper_input_files=[hopper_file],
    )
    self.assertEqual(count, 4)

    with self.subTest("test_add"):
      with open(out_dir / "test_add.plist", "rb") as f:
        tree = plistlib.load(f, fmt=plistlib.FMT_BINARY)
      self.assertEqual(tree, {"base": {"float32": [["inp1", "out1"]]}})

    with self.subTest("test_mul"):
      with open(out_dir / "test_mul.plist", "rb") as f:
        tree = plistlib.load(f, fmt=plistlib.FMT_BINARY)
      self.assertEqual(tree, {"base": {"float32": [["inp3", "out3"]]}})

    with self.subTest("test_sub"):
      with open(out_dir / "test_sub.plist", "rb") as f:
        tree = plistlib.load(f, fmt=plistlib.FMT_BINARY)
      self.assertEqual(
          tree,
          {"base": {"float32": [["inp2", "out2"], ["inp4", "out4"]]}},
      )

    with self.subTest("test_hopper_op"):
      with open(out_dir / "test_hopper_op.plist", "rb") as f:
        tree = plistlib.load(f, fmt=plistlib.FMT_BINARY)
      self.assertEqual(
          tree,
          {"base": {"float32": [["inp_hopper", "out_hopper"]]}},
      )

  def test_split_goldens_cross_arch_overlap_raises(self):
    temp_dir = pathlib.Path(self.create_tempdir())
    input_file = temp_dir / "golden1.gz"
    hopper_file = temp_dir / "golden_hopper0.gz"
    out_dir = temp_dir / "out"

    data = {
        "test_ambiguous": {"base": {"float32": [("inp1", "out1")]}},
    }
    hopper_data = {
        "test_ambiguous": {"base": {"float32": [("inp2", "out2")]}},
    }

    for path, d in [(input_file, data), (hopper_file, hopper_data)]:
      with gzip.open(path, "wb") as f:
        plistlib.dump(d, f, fmt=plistlib.FMT_BINARY)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tool test
        ValueError, "both hold goldens for test_ambiguous"
    ):
      split_gpu_goldens.split_goldens(
          base_input_files=[input_file],
          output_dir=out_dir,
          hopper_input_files=[hopper_file],
      )


if __name__ == "__main__":
  absltest.main()
