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

"""Unit tests for calculate_oot_impacted_tests."""

import pathlib
import tempfile
import textwrap
import unittest

from ci.tools import calculate_oot_impacted_tests as calc


class CalculateOotImpactedTestsTest(unittest.TestCase):

  def test_parse_wheel_version_py(self) -> None:
    version_py = textwrap.dedent("""\
        __version__ = '2.16.0.dev20261118+cpu'
        git_version = '2f5bd41363b94a3a09c852baa3d1b296c4a71ee2'
    """)
    ver, sha = calc.parse_wheel_version_py(
        version_py, pathlib.Path("torch.whl")
    )
    self.assertEqual(ver, "2.16.0.dev20261118+cpu")
    self.assertEqual(sha, "2f5bd41363b94a3a09c852baa3d1b296c4a71ee2")

  def test_resolve_main_sha_from_commit_subject(self) -> None:
    main_sha = "dc7247294c6d03bc6403d0c9319f0422d48732a0"
    wheel_sha = "2f5bd41363b94a3a09c852baa3d1b296c4a71ee2"
    self.assertEqual(
        calc.resolve_main_sha_from_commit_subject(
            f"2026-11-18 nightly release ({main_sha})", wheel_sha
        ),
        main_sha,
    )
    self.assertEqual(
        calc.resolve_main_sha_from_commit_subject(
            "[dynamo] Normal commit on main (#12345)", main_sha
        ),
        main_sha,
    )

  def test_discover_nightly_wheel_urls(self) -> None:
    html = textwrap.dedent("""\
        <a href="/whl/nightly/cpu/torch-2.16.0.dev20261117%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=abc">17</a>
        <a href="/whl/nightly/cpu/torch-2.16.0.dev20261118%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl">18</a>
        <a href="/whl/nightly/cpu/torch-2.16.0.dev20261118%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl">py313</a>
    """)
    urls = calc.discover_nightly_wheel_urls(html, "3.14")
    self.assertEqual(
        list(urls),
        [
            "https://download.pytorch.org/whl/nightly/cpu/torch-2.16.0.dev20261117%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cpu/torch-2.16.0.dev20261118%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl",
        ],
    )

  def test_parse_torch_test_targets_and_filter(self) -> None:
    build_text = textwrap.dedent("""\
        torch_tpu_torch_test(
            name = "transformers_test",
            shard_count = 4,
            torch_test_module = "test_transformers",
        )

        torch_tpu_torch_test(
            name = "disabled_test",
            notap = "b/12345",
            torch_test_module = "test_disabled",
        )
    """)
    targets = calc.parse_torch_test_targets(build_text)
    self.assertEqual(len(targets), 2)
    self.assertTrue(targets["//tests/torch:transformers_test"].enabled)
    self.assertEqual(targets["//tests/torch:transformers_test"].shard_count, 4)
    self.assertFalse(targets["//tests/torch:disabled_test"].enabled)

    filtered = calc.filter_enabled_torch_tests(
        [
            "@@//tests/torch:transformers_test",
            "@@//tests/torch:disabled_test",
            "@@+pytorch_test_repo+pytorch_test//:test_transformers_module",
        ],
        targets,
    )
    self.assertEqual(filtered, ("//tests/torch:transformers_test",))

  def test_compute_pytorch_test_fingerprint_ignores_untracked_files(
      self,
  ) -> None:
    tracked_modules = {"test/test_transformers.py", "test/dynamo/test_misc.py"}
    tracked_subdirs = {"dynamo"}
    tree_a = textwrap.dedent("""\
        100644 blob 1111\ttest/test_transformers.py
        100644 blob 2222\ttest/dynamo/utils.py
        100644 blob 3333\ttest/expect/foo.expect
        100644 blob 4444\ttest/test_fx.py
    """)
    tree_b = textwrap.dedent("""\
        100644 blob 1111\ttest/test_transformers.py
        100644 blob 2222\ttest/dynamo/utils.py
        100644 blob 9999\ttest/expect/foo.expect
        100644 blob 8888\ttest/test_fx.py
    """)
    tree_c = textwrap.dedent("""\
        100644 blob 5555\ttest/test_transformers.py
        100644 blob 2222\ttest/dynamo/utils.py
    """)
    fp_a = calc.compute_pytorch_test_fingerprint(
        tree_a, tracked_modules, tracked_subdirs
    )
    fp_b = calc.compute_pytorch_test_fingerprint(
        tree_b, tracked_modules, tracked_subdirs
    )
    fp_c = calc.compute_pytorch_test_fingerprint(
        tree_c, tracked_modules, tracked_subdirs
    )
    self.assertEqual(fp_a, fp_b)
    self.assertNotEqual(fp_a, fp_c)

  def test_is_path_tracked_matches_repo_glob_exclusions(self) -> None:
    tracked_modules = {"test/dynamo/test_misc.py"}
    tracked_subdirs = {"dynamo"}
    self.assertTrue(
        calc.is_path_tracked_by_pytorch_test(
            "test/dynamo/test_data.json", tracked_modules, tracked_subdirs
        )
    )
    self.assertFalse(
        calc.is_path_tracked_by_pytorch_test(
            "test/dynamo/test_other.py", tracked_modules, tracked_subdirs
        )
    )
    self.assertFalse(
        calc.is_path_tracked_by_pytorch_test(
            "test/dynamo/.hidden/helper.py", tracked_modules, tracked_subdirs
        )
    )
    self.assertFalse(
        calc.is_path_tracked_by_pytorch_test(
            "test/dynamo/BUILD.bazel", tracked_modules, tracked_subdirs
        )
    )

  def test_filter_enabled_torch_tests_respects_config_tags(self) -> None:
    build_text = textwrap.dedent("""\
        torch_tpu_torch_test(
            name = "both_tpus_test",
            torch_test_module = "test_both",
        )

        torch_tpu_torch_test(
            name = "v5_only_test",
            tags = ["fails-on-tpu-v7"],
            torch_test_module = "test_v5_only",
        )

        torch_tpu_torch_test(
            name = "newest_deps_only_test",
            tags = ["requires-newest-deps"],
            torch_test_module = "test_newest",
        )
    """)
    targets = calc.parse_torch_test_targets(build_text)
    raw = [
        "@@//tests/torch:both_tpus_test",
        "@@//tests/torch:v5_only_test",
        "@@//tests/torch:newest_deps_only_test",
    ]
    self.assertEqual(
        calc.filter_enabled_torch_tests(raw, targets, bazel_config="ci_tpu_v5"),
        ("//tests/torch:both_tpus_test", "//tests/torch:v5_only_test"),
    )
    self.assertEqual(
        calc.filter_enabled_torch_tests(raw, targets, bazel_config="ci_tpu_v7"),
        ("//tests/torch:both_tpus_test",),
    )
    self.assertEqual(
        calc.filter_enabled_torch_tests(
            raw, targets, bazel_config="ci_tpu_v5_newest_deps"
        ),
        ("//tests/torch:newest_deps_only_test",),
    )

  def test_write_ci_outputs_writes_per_config_files_and_github_output(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = pathlib.Path(tmp)
      gh_out = tmp_dir / "github_output.txt"
      single_out = tmp_dir / "impacted.txt"
      per_config = {
          "ci_tpu_v5": (
              "//tests/torch:both_tpus_test",
              "//tests/torch:v5_only_test",
          ),
          "ci_tpu_v7": (),
      }
      calc.write_ci_outputs(
          per_config_targets=per_config,
          primary_config="ci_tpu_v5",
          impacted_targets_output=single_out,
          impacted_targets_dir=tmp_dir,
          github_output=gh_out,
      )
      self.assertEqual(
          single_out.read_text(encoding="utf-8").splitlines(),
          ["//tests/torch:both_tpus_test", "//tests/torch:v5_only_test"],
      )
      self.assertEqual(
          (tmp_dir / "impacted_targets_ci_tpu_v5.txt")
          .read_text(encoding="utf-8")
          .splitlines(),
          ["//tests/torch:both_tpus_test", "//tests/torch:v5_only_test"],
      )
      self.assertEqual(
          (tmp_dir / "impacted_targets_ci_tpu_v7.txt").read_text(
              encoding="utf-8"
          ),
          "",
      )
      gh_lines = dict(
          line.split("=", 1)
          for line in gh_out.read_text(encoding="utf-8").splitlines()
          if "=" in line
      )
      self.assertEqual(gh_lines["has_impacted_targets"], "true")
      self.assertEqual(gh_lines["is_full_run"], "false")
      self.assertEqual(gh_lines["has_impacted_ci_tpu_v5"], "true")
      self.assertEqual(gh_lines["impacted_count_ci_tpu_v5"], "2")
      self.assertEqual(gh_lines["has_impacted_ci_tpu_v7"], "false")
      self.assertEqual(gh_lines["impacted_count_ci_tpu_v7"], "0")

  def test_parse_args_single_diff_flags(self) -> None:
    ns = calc.parse_args([
        "--pytorch-dir=/tmp/pytorch",
        "--single-diff",
        "--base-sha=aaaa",
        "--head-sha=bbbb",
        "--filter-configs=ci_tpu_v5,ci_tpu_v7",
        "--impacted-targets-dir=/tmp/out",
        "--github-output=/tmp/gh_out",
    ])
    self.assertTrue(ns.single_diff)
    self.assertEqual(ns.base_sha, "aaaa")
    self.assertEqual(ns.head_sha, "bbbb")
    self.assertEqual(ns.filter_configs, "ci_tpu_v5,ci_tpu_v7")
    self.assertEqual(ns.impacted_targets_dir, pathlib.Path("/tmp/out"))
    self.assertEqual(ns.github_output, pathlib.Path("/tmp/gh_out"))


if __name__ == "__main__":
  unittest.main()
