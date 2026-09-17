#!/usr/bin/env python3
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

"""Unit tests for partition_targets."""

import io
import pathlib
import tempfile
import unittest
from unittest import mock

from ci.tools import partition_targets


class PartitionTargetsTest(unittest.TestCase):

  def test_load_build_metadata_and_estimate(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = pathlib.Path(tmpdir)
      pkg_dir = root / "tests"
      pkg_dir.mkdir()
      # Test rules with internal blank lines, single quotes, and adjacent rules
      # without blank lines between them.
      (pkg_dir / "BUILD.bazel").write_text(
          "py_test(\n"
          "    name = 'mega_pallas_test',\n\n"
          "    shard_count = 45,\n"
          "    timeout = 'long',\n"
          ")\n"
          "py_test(\n"
          '    name = "custom_mesh_test",\n'
          '    tags = ["requires-tpu-v5lite:8"],\n'
          ")\n"
          "py_test(\n"
          '    name = "simple_unit_test",\n'
          '    tags = ["nopresubmit"],\n'
          ")\n",
          encoding="utf-8",
      )
      targets = [
          "//tests:mega_pallas_test_tpu-v5lite",
          "//tests:custom_mesh_test_tpu-v5lite",
          "//tests:simple_unit_test_tpu-v5lite",
          "//csrc:torch_tpu_lib",
          "//tests:ops_test_build_test",
      ]
      weights, rule_tags = partition_targets.load_build_metadata(targets, root)
      self.assertIn("nopresubmit", rule_tags["simple_unit_test"])

      cases = [
          ("//tests:mega_pallas_test_tpu-v5lite", 180),
          ("//tests:custom_mesh_test_tpu-v5lite", 36),
          ("//tests/distributed:collective_test", 36),
          ("//tests:simple_unit_test_tpu-v5lite", 6),
          ("//tests:unlisted_test", 6),
      ]
      for target, expected in cases:
        with self.subTest(target=target):
          self.assertEqual(
              partition_targets.estimate_target_weight(target, weights),
              expected,
          )

  def test_is_target_compatible_filters_by_config(self):
    rule_tags = {
        "skip_me_test": {"nopresubmit"},
        "explicit_v5_test": {"presubmit-v5"},
    }
    cases = [
        ("//csrc:torch_tpu_lib", "ci_tpu_v5_presubmit", False),
        ("//tests:ops_test_build_test", "ci_tpu_v5_presubmit", False),
        ("//tests:ops_test_cpu", "ci_tpu_v5_presubmit", False),
        ("//tests:ops_test_tpu-v6e", "ci_tpu_v5_presubmit", False),
        ("//tests:skip_me_test_tpu-v5lite", "ci_tpu_v5_presubmit", False),
        ("//tests:ops_test_tpu-v5lite", "ci_tpu_v5_presubmit", True),
        ("//tests:explicit_v5_test", "ci_tpu_v5_presubmit", True),
        ("//tests:ops_test_cpu", "ci_cpu_presubmit", True),
        ("//tests:ops_test_tpu-v5lite", "ci_cpu_presubmit", False),
    ]
    for target, config, expected in cases:
      with self.subTest(target=target, config=config):
        self.assertEqual(
            partition_targets.is_target_compatible(target, config, rule_tags),
            expected,
        )

  def test_partition_targets_balances_heavy_and_multichip_suites(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = pathlib.Path(tmpdir)
      pkg = root / "tests"
      pkg.mkdir()
      (pkg / "BUILD").write_text(
          "\n\n".join(
              f'py_test(name = "heavy_{i}_test", shard_count = 50)'
              for i in range(4)
          ),
          encoding="utf-8",
      )
      heavy = [f"//tests:heavy_{i}_test_tpu-v5lite" for i in range(4)]
      dist = [f"//tests/distributed:dist_{i}_test_tpu-v5lite" for i in range(8)]
      unit = [f"//tests:unit_{i}_test_tpu-v5lite" for i in range(8)]
      noise = ["//csrc:some_cc_library", "//tests:other_test_cpu"]
      all_targets = heavy + dist + unit + noise

      shards = [
          partition_targets.partition_targets(
              all_targets, i, 4, root, bazel_config="ci_tpu_v5_presubmit"
          )
          for i in range(4)
      ]

      self.assertCountEqual([t for s in shards for t in s], heavy + dist + unit)
      for shard in shards:
        self.assertEqual(len(set(shard) & set(heavy)), 1)
        self.assertEqual(len(set(shard) & set(dist)), 2)

  def test_edge_cases_and_cli_io(self):
    self.assertEqual(partition_targets.partition_targets([], 0, 4), [])
    self.assertEqual(
        partition_targets.partition_targets(
            ["//t:a_test", "//t:a_test", "  "], 0, 1
        ),
        ["//t:a_test"],
    )
    for idx, count in [(0, 0), (-1, 4), (4, 4)]:
      with self.subTest(idx=idx, count=count):
        with self.assertRaises(ValueError):
          partition_targets.partition_targets(["//t:a_test"], idx, count)

    with tempfile.TemporaryDirectory() as tmpdir:
      in_path = pathlib.Path(tmpdir) / "in.txt"
      out_path = pathlib.Path(tmpdir) / "out.txt"
      in_path.write_text(
          "//tests:a_test_tpu-v5lite\n//tests:b_test_cpu\n", encoding="utf-8"
      )
      rc = partition_targets.main([
          "--shard-index=0",
          "--shard-count=2",
          "--bazel-config=ci_tpu_v5_presubmit",
          f"--input-file={in_path}",
          f"--output-file={out_path}",
      ])
      self.assertEqual(rc, 0)
      self.assertEqual(
          out_path.read_text(encoding="utf-8").splitlines(),
          ["//tests:a_test_tpu-v5lite"],
      )

    mock_out = io.StringIO()
    with mock.patch(
        "sys.stdin", io.StringIO("//tests:a_test_tpu-v5lite\n")
    ), mock.patch("sys.stdout", mock_out):
      self.assertEqual(
          partition_targets.main(["--shard-index=0", "--shard-count=1"]), 0
      )
    self.assertEqual(mock_out.getvalue(), "//tests:a_test_tpu-v5lite\n")


if __name__ == "__main__":
  unittest.main()
