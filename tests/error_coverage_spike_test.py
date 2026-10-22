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

"""THROWAWAY spike: can error_coverage_test run in OSS? DO NOT MERGE.

Always fails on purpose so that CI (--test_output=errors) prints the report.
"""

import concurrent.futures
import os
import subprocess
import sys
import time

from absl.testing import absltest
import yaml

from tests import error_coverage_spike_lib as ecov
from tests import repo_paths

_BINARIES = (
    "tests/errors_test_cov",
    "tests/tpu_errors_test_cov",
    "tests/distributed/collective_errors_test_cov",
)


def _canon(path):
  path = path.removeprefix("./")
  for prefix in ("third_party/py/torch_tpu/", "oss/torch_tpu/"):
    if path.startswith(prefix):
      path = path[len(prefix):]
  return path


def _scan(repo_root):
  files = []
  for root, dirs, names in os.walk(repo_root):
    dirs[:] = [
        d
        for d in dirs
        if not repo_paths.is_ignored_dir(d)
        and not d.startswith(("blaze-", "bazel-"))
    ]
    for n in names:
      if n.endswith((".cc", ".h")):
        files.append(os.path.join(root, n))
  out = {}
  with concurrent.futures.ThreadPoolExecutor() as ex:
    for f, locs in ex.map(lambda f: (f, ecov.get_error_macro_locs(f)), files):
      if locs:
        out[_canon(os.path.relpath(f, repo_root))] = locs
  return out


class SpikeTest(absltest.TestCase):

  def test_spike(self):
    report = []
    t0 = time.time()
    repo_root = repo_paths.get_repo_root_in_runfiles()
    report.append(f"repo_root={repo_root}")
    import torch  # pylint: disable=g-import-not-at-top
    report.append(f"torch={torch.__version__}")
    try:
      from torch._functorch._activation_offloading import offload_ops  # pylint: disable=g-import-not-at-top
      report.append(f"offload_ops available: {offload_ops}")
    except ImportError as e:
      report.append(f"offload_ops NOT available: {e}")
    file_to_locs = _scan(repo_root)
    report.append(
        f"scanned {len(file_to_locs)} files with macros in"
        f" {time.time() - t0:.1f}s"
    )
    env = os.environ.copy()
    env["TORCH_DISABLE_ADDR2LINE"] = "1"
    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    env["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "1"
    for k in sorted(env):
      if k.startswith(("TPU", "TORCH_TPU", "LD_LIB", "RUNFILES", "PYTHON")):
        report.append(f"env {k}={env[k][:200]}")
    outputs = []
    for b in _BINARIES:
      path = os.path.join(repo_root, b)
      t = time.time()
      ret = subprocess.run(
          [path, "--test_mode=cov"],
          env=env,
          capture_output=True,
          check=False,
      )
      out = ret.stdout.decode("utf-8", "replace")
      err = ret.stderr.decode("utf-8", "replace")
      n = out.count("Covered ")
      report.append(
          f"BINARY {b}: exists={os.path.exists(path)} rc={ret.returncode}"
          f" time={time.time() - t:.1f}s covered_lines={n}"
      )
      if ret.returncode != 0 or n == 0:
        report.append(f"--- stdout tail:\n{out[-4000:]}")
        report.append(f"--- stderr tail:\n{err[-8000:]}")
      sample = [l for l in out.splitlines() if l.startswith("Covered ")][:5]
      report.append(f"sample: {sample}")
      outputs.append(out)
    all_out = "".join(outputs)
    interesting = sorted({
        l for l in all_out.splitlines()
        if l.startswith("Covered ") and (".h:" in l or "ao_offload" in l
                                         or not l.startswith("Covered csrc/"))
    })
    report.append(f"INTERESTING COVERED ({len(interesting)}):")
    report += ["  " + l for l in interesting[:300]]
    raw = ecov.parse_cov_output(all_out)
    covered = {ecov.SourceLoc(_canon(l.path), l.line_number) for l in raw}
    per_file = ecov.get_per_file_coverage(file_to_locs, covered)
    actual = {c.cpp_file: c.to_dict() for c in per_file}
    with open(
        os.path.join(repo_root, "tests/error_coverage_spike_data.yaml")
    ) as f:
      expected = {_canon(k): v for k, v in yaml.safe_load(f).items()}
    worse, better, total_diff, only_oss, only_g3 = [], [], [], [], []
    for k, a in sorted(actual.items()):
      e = expected.get(k)
      if e is None:
        only_oss.append((k, a))
        continue
      if a["remaining"] > e["remaining"]:
        worse.append((k, e, a))
      elif a["remaining"] < e["remaining"]:
        better.append((k, e, a))
      if a["total"] != e["total"] or a["infeasible"] != e["infeasible"]:
        total_diff.append((k, e, a))
    for k in sorted(set(expected) - set(actual)):
      only_g3.append(k)
    report.append(f"TOTAL elapsed {time.time() - t0:.1f}s")
    report.append(f"files: oss={len(actual)} g3_yaml={len(expected)}")
    report.append(
        "sum remaining oss=%d g3=%d"
        % (
            sum(a["remaining"] for a in actual.values()),
            sum(
                e["remaining"]
                for k, e in expected.items()
                if k in actual
            ),
        )
    )
    report.append(f"WORSE in OSS ({len(worse)}):")
    report += [f"  {k}: g3={e} oss={a}" for k, e, a in worse]
    report.append(f"BETTER in OSS ({len(better)}):")
    report += [f"  {k}: g3={e} oss={a}" for k, e, a in better]
    report.append(f"TOTAL/INFEASIBLE differ ({len(total_diff)}):")
    report += [f"  {k}: g3={e} oss={a}" for k, e, a in total_diff]
    report.append(f"ONLY OSS ({len(only_oss)}): {only_oss}")
    report.append(f"ONLY G3 YAML ({len(only_g3)}): {only_g3}")
    worse_keys = {k for k, _, _ in worse}
    for c in per_file:
      if c.cpp_file in worse_keys:
        report.append(
            ecov.sub_coverage_report(
                c.cpp_file, c.to_cover_locs, "TO BE COVERED"
            )
        )
    msg = "\n".join(report)
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)
    self.fail("SPIKE REPORT (intentional failure):\n" + msg)


if __name__ == "__main__":
  absltest.main()
