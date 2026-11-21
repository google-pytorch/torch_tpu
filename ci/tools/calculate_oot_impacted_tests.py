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

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Calculates impacted TPU PyTorch device-generic tests across upstream commits.

Resolves the latest and previous PyTorch nightly CPU wheels (or uses explicit
wheel/commit overrides), enumerates first-parent commits on `pytorch/pytorch`
`main` leading up to the new nightly wheel, runs `bazel-diff` against
`@pytorch_test` with `TORCH_SOURCE` pinned to the base nightly wheel, and writes
JSON and Markdown reports of the impacted `//tests/torch:*` targets (both vs.
the base nightly wheel SHA and incrementally commit-to-commit).
"""

import argparse
from collections.abc import Iterable, Mapping, Sequence, Set
import concurrent.futures
import dataclasses
import enum
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Final
import urllib.parse
import urllib.request
import zipfile

_BAZEL_DIFF_VERSION: Final[str] = "v46.1.0"
_BAZEL_DIFF_URL: Final[str] = (
    f"https://github.com/Tinder/bazel-diff/releases/download/{_BAZEL_DIFF_VERSION}/"
    "bazel-diff_deploy.jar"
)
_DEFAULT_BAZEL_DIFF_JAR: Final[pathlib.Path] = pathlib.Path(
    "/tmp/bazel-diff.jar"
)
_PYTORCH_NIGHTLY_CPU_TORCH_INDEX: Final[str] = (
    "https://download.pytorch.org/whl/nightly/cpu/torch/"
)
_PYPI_MIRROR_REWRITE: Final[str] = (
    "rewrite (us-python\\.pkg\\.dev)/"
    "(ml-oss-artifacts-published/pypi-mirror/simple/.*) https://$1/$2\n"
    "rewrite us-python\\.pkg\\.dev/ml-oss-artifacts-published/"
    "pypi-mirror/simple/(.*) https://pypi.org/simple/$1\n"
)
_CQUERY_EXPRESSION: Final[str] = (
    "deps(//tests/torch:all, 1) + deps(@pytorch_test//:all)"
)
_PYTORCH_TEST_CANONICAL_REPO: Final[str] = "+pytorch_test_repo+pytorch_test"
_NIGHTLY_SUBJECT_SHA_RE: Final[re.Pattern[str]] = re.compile(
    r"\(([0-9a-f]{40})\)\s*$"
)
_VERSION_FIELD_RE: Final[re.Pattern[str]] = re.compile(
    r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE
)
_GIT_VERSION_FIELD_RE: Final[re.Pattern[str]] = re.compile(
    r"^git_version\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE
)

# Keep in sync with `_EXTRA_MODULE_GLOBS` in `bazel/pytorch_test_repo.bzl`.
_EXTRA_TRACKED_EXACT_PATHS: Final[frozenset[str]] = frozenset({
    "test/dynamo/test_functions.py",
    "test/dynamo/test_tokenization.py",
    "test/inductor/test_torchinductor.py",
    "test/inductor/test_torchinductor_dynamic_shapes.py",
    "test/minioptest_failures_dict.json",
    "test/test_opaque_obj_v2.py",
})
_EXTRA_TRACKED_PREFIXES: Final[tuple[str, ...]] = (
    "test/ao/",
    "test/autograd/",
    "test/optim/",
)


class OutputBaseIsolation(enum.StrEnum):
  """Whether a hash worker uses the default or an isolated Bazel output_base."""

  SHARED = "shared"
  ISOLATED = "isolated"


@dataclasses.dataclass(frozen=True)
class WheelMetadata:
  """Metadata extracted from a PyTorch wheel archive."""

  path: pathlib.Path
  version: str
  wheel_sha: str
  main_sha: str


@dataclasses.dataclass(frozen=True)
class TorchTestTarget:
  """Metadata for a `torch_tpu_torch_test` target in `//tests/torch:BUILD`."""

  label: str
  module: str
  shard_count: int
  enabled: bool
  tags: frozenset[str] = frozenset()


@dataclasses.dataclass(frozen=True)
class CommitInfo:
  """Metadata for a single upstream PyTorch commit."""

  index: int
  sha: str
  committed_at: str
  subject: str
  fingerprint: str


@dataclasses.dataclass(frozen=True)
class CommitImpact:
  """Impacted TPU test targets for a single upstream PyTorch commit."""

  commit: CommitInfo
  vs_wheel_targets: tuple[str, ...]
  vs_wheel_shards: int
  incremental_targets: tuple[str, ...]
  incremental_shards: int


def parse_wheel_version_py(
    version_py_text: str, wheel_path: pathlib.Path
) -> tuple[str, str]:
  """Extracts `(__version__, git_version)` from `torch/version.py` text."""
  ver_match = _VERSION_FIELD_RE.search(version_py_text)
  sha_match = _GIT_VERSION_FIELD_RE.search(version_py_text)
  if not ver_match or not sha_match:
    raise ValueError(
        f"Failed to parse __version__ or git_version from {wheel_path}."
    )
  return ver_match.group(1), sha_match.group(1)


def resolve_main_sha_from_commit_subject(
    subject: str, fallback_sha: str
) -> str:
  """Extracts the `main` SHA from a PyTorch nightly release commit subject.

  PyTorch nightly wheels record a commit on the `nightly` branch whose subject
  has the form `YYYY-MM-DD nightly release (<40-char-main-sha>)`. When `subject`
  matches that format, returns the parenthesized `main` SHA; otherwise returns
  `fallback_sha`.

  Args:
    subject: Commit subject line of `fallback_sha`.
    fallback_sha: Raw `git_version` SHA from `torch/version.py`.

  Returns:
    The corresponding `main`-branch commit SHA.
  """
  match = _NIGHTLY_SUBJECT_SHA_RE.search(subject.strip())
  if match:
    return match.group(1)
  return fallback_sha


def _commit_exists_locally(pytorch_dir: pathlib.Path, sha: str) -> bool:
  """Returns True if `sha` is present locally without lazy-fetching."""
  if not sha:
    return False
  res = subprocess.run(
      [
          "git",
          "--no-lazy-fetch",
          "-C",
          os.fspath(pytorch_dir),
          "cat-file",
          "-e",
          f"{sha}^{{commit}}",
      ],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      check=False,
  )
  return res.returncode == 0


def _ensure_commit_fetched(pytorch_dir: pathlib.Path, sha: str) -> None:
  """Fetches `sha` bloblessly from `origin` when missing in `pytorch_dir`."""
  if _commit_exists_locally(pytorch_dir, sha):
    return
  subprocess.run(
      [
          "git",
          "-C",
          os.fspath(pytorch_dir),
          "fetch",
          "--no-tags",
          "--depth=1",
          "--filter=blob:none",
          "origin",
          sha,
      ],
      check=True,
  )


def inspect_wheel(
    wheel_path: pathlib.Path, pytorch_dir: pathlib.Path
) -> WheelMetadata:
  """Reads version and resolves `main` SHA for a local PyTorch `.whl` file."""
  with zipfile.ZipFile(wheel_path) as zf:
    version_py = zf.read("torch/version.py").decode("utf-8")
  version, wheel_sha = parse_wheel_version_py(version_py, wheel_path)
  _ensure_commit_fetched(pytorch_dir, wheel_sha)
  subject = subprocess.check_output(
      [
          "git",
          "-C",
          os.fspath(pytorch_dir),
          "show",
          "-s",
          "--format=%s",
          wheel_sha,
      ],
      text=True,
  ).strip()
  candidate_main = resolve_main_sha_from_commit_subject(subject, wheel_sha)
  main_sha = wheel_sha
  if candidate_main != wheel_sha:
    if _commit_exists_locally(pytorch_dir, candidate_main):
      main_sha = candidate_main
    else:
      fetch_res = subprocess.run(
          [
              "git",
              "-C",
              os.fspath(pytorch_dir),
              "fetch",
              "--no-tags",
              "--depth=1",
              "--filter=blob:none",
              "origin",
              candidate_main,
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
          check=False,
      )
      if fetch_res.returncode == 0 and _commit_exists_locally(
          pytorch_dir, candidate_main
      ):
        main_sha = candidate_main
  return WheelMetadata(
      path=wheel_path.resolve(),
      version=version,
      wheel_sha=wheel_sha,
      main_sha=main_sha,
  )


def default_python_version(workspace_dir: pathlib.Path) -> str:
  """Reads `DEFAULT_PYTHON_VERSION` from `bazel/supported_python_versions.bzl`."""
  bzl_path = workspace_dir / "bazel" / "supported_python_versions.bzl"
  for line in bzl_path.read_text(encoding="utf-8").splitlines():
    if line.startswith("DEFAULT_PYTHON_VERSION"):
      parts = line.split('"')
      if len(parts) >= 2:
        return parts[1]
  raise ValueError(f"DEFAULT_PYTHON_VERSION not found in {bzl_path}.")


def discover_nightly_wheel_urls(
    index_html: str, python_version: str
) -> Sequence[str]:
  """Extracts sorted CPU nightly x86_64 wheel URLs for `python_version`."""
  cp_tag = "cp" + python_version.replace(".", "")
  pattern = re.compile(
      rf'href=["\']([^"\']*torch-[^"\']*dev\d{{8}}(?:%2B|\+)cpu-{cp_tag}-{cp_tag}-manylinux[^"\']*x86_64\.whl)(?:#[^"\']*)?["\']'
  )
  urls: set[str] = set()
  for match in pattern.finditer(index_html):
    href = match.group(1)
    urls.add(urllib.parse.urljoin(_PYTORCH_NIGHTLY_CPU_TORCH_INDEX, href))
  return sorted(urls)


def download_file(url: str, dest: pathlib.Path) -> pathlib.Path:
  """Downloads `url` to `dest` if not already present."""
  dest.parent.mkdir(parents=True, exist_ok=True)
  if dest.is_file() and dest.stat().st_size > 0:
    return dest
  with urllib.request.urlopen(url, timeout=120) as resp:
    data = resp.read()
  tmp_dest = dest.with_suffix(dest.suffix + ".part")
  tmp_dest.write_bytes(data)
  tmp_dest.replace(dest)
  return dest


def resolve_latest_two_wheels(
    python_version: str, download_dir: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
  """Downloads the previous and latest PyTorch nightly CPU wheels."""
  with urllib.request.urlopen(
      _PYTORCH_NIGHTLY_CPU_TORCH_INDEX, timeout=30
  ) as resp:
    html = resp.read().decode("utf-8", errors="ignore")
  urls = discover_nightly_wheel_urls(html, python_version)
  if len(urls) < 2:
    raise RuntimeError(
        f"Expected at least 2 nightly CPU wheels for Python {python_version}, "
        f"found {len(urls)}."
    )
  prev_url, latest_url = urls[-2], urls[-1]
  prev_name = urllib.parse.unquote(prev_url.rsplit("/", 1)[-1])
  latest_name = urllib.parse.unquote(latest_url.rsplit("/", 1)[-1])
  prev_path = download_file(prev_url, download_dir / prev_name)
  latest_path = download_file(latest_url, download_dir / latest_name)
  return prev_path, latest_path


def parse_torch_test_targets(
    build_text: str,
) -> Mapping[str, TorchTestTarget]:
  """Parses `torch_tpu_torch_test` rules from `//tests/torch:BUILD`."""
  targets: dict[str, TorchTestTarget] = {}
  blocks = build_text.split("torch_tpu_torch_test(")[1:]
  for block in blocks:
    rule_body = block.split("\n)", 1)[0]
    name_match = re.search(r'name\s*=\s*"([^"]+)"', rule_body)
    mod_match = re.search(r'torch_test_module\s*=\s*"([^"]+)"', rule_body)
    if not name_match or not mod_match:
      continue
    shard_match = re.search(r"shard_count\s*=\s*(\d+)", rule_body)
    shard_count = int(shard_match.group(1)) if shard_match else 1
    tags_match = re.search(r"tags\s*=\s*\[([^\]]*)\]", rule_body, re.DOTALL)
    parsed_tags: frozenset[str] = frozenset(
        re.findall(r'"([^"]+)"', tags_match.group(1)) if tags_match else ()
    )
    disabled = (
        ("notap =" in rule_body)
        or ("notest" in parsed_tags)
        or ("nobuild" in parsed_tags)
    )
    label = f"//tests/torch:{name_match.group(1)}"
    targets[label] = TorchTestTarget(
        label=label,
        module=mod_match.group(1),
        shard_count=shard_count,
        enabled=not disabled,
        tags=parsed_tags,
    )
  return targets


def is_path_tracked_by_pytorch_test(
    rel_path: str,
    tracked_module_files: Set[str],
    tracked_subdirs: Set[str],
) -> bool:
  """Returns True if `rel_path` (under `test/`) is an input to `@pytorch_test`."""
  if rel_path in tracked_module_files or rel_path in _EXTRA_TRACKED_EXACT_PATHS:
    return True
  if rel_path.startswith(_EXTRA_TRACKED_PREFIXES):
    return True
  parts = rel_path.split("/")
  if len(parts) >= 3 and parts[0] == "test" and parts[1] in tracked_subdirs:
    if any(part.startswith(".") for part in parts[2:]):
      return False
    basename = parts[-1]
    if basename in ("BUILD", "BUILD.bazel"):
      return False
    if not (basename.startswith("test_") and basename.endswith(".py")):
      return True
  return False


def compute_pytorch_test_fingerprint(
    ls_tree_output: str,
    tracked_module_files: Set[str],
    tracked_subdirs: Set[str],
) -> str:
  """Computes a SHA-256 digest over git `ls-tree` entries exposed by `@pytorch_test`."""
  relevant: list[str] = []
  for line in ls_tree_output.splitlines():
    if "\t" not in line:
      continue
    path = line.split("\t", 1)[1]
    if is_path_tracked_by_pytorch_test(
        path, tracked_module_files, tracked_subdirs
    ):
      relevant.append(line)
  relevant.sort()
  return hashlib.sha256("\n".join(relevant).encode("utf-8")).hexdigest()[:16]


def is_target_selected_by_config(
    target: TorchTestTarget, bazel_config: str = ""
) -> bool:
  """Returns True if `target` is runnable under `bazel_config`'s tag filters."""
  if not target.enabled:
    return False
  is_newest_deps_cfg = "newest_deps" in bazel_config
  has_newest_deps_tag = "requires-newest-deps" in target.tags
  if is_newest_deps_cfg != has_newest_deps_tag:
    return False
  if "tpu_v5" in bazel_config and "fails-on-tpu-v5" in target.tags:
    return False
  if "tpu_v6" in bazel_config and "fails-on-tpu-v6" in target.tags:
    return False
  if "tpu_v7" in bazel_config and "fails-on-tpu-v7" in target.tags:
    return False
  return True


def filter_enabled_torch_tests(
    impacted_labels: Iterable[str],
    targets: Mapping[str, TorchTestTarget],
    *,
    bazel_config: str = "",
) -> tuple[str, ...]:
  """Filters `bazel-diff` output labels to runnable `//tests/torch:*` targets."""
  matched: set[str] = set()
  for raw in impacted_labels:
    normalized = raw.strip().removeprefix("@@")
    target = targets.get(normalized)
    if target is not None and is_target_selected_by_config(
        target, bazel_config
    ):
      matched.add(normalized)
  return tuple(sorted(matched))


def _tracked_sets_for_targets(
    targets: Mapping[str, TorchTestTarget],
) -> tuple[set[str], set[str]]:
  """Returns `(tracked_module_files, tracked_subdirs)` for `@pytorch_test`."""
  tracked_module_files = {f"test/{t.module}.py" for t in targets.values()}
  tracked_subdirs = {
      t.module.split("/", 1)[0] for t in targets.values() if "/" in t.module
  } | {"inductor"}
  return tracked_module_files, tracked_subdirs


def make_commit_info(
    pytorch_dir: pathlib.Path,
    idx: int,
    sha: str,
    targets: Mapping[str, TorchTestTarget],
) -> CommitInfo:
  """Fetches commit `sha` if needed and computes its `@pytorch_test` fingerprint."""
  _ensure_commit_fetched(pytorch_dir, sha)
  tracked_module_files, tracked_subdirs = _tracked_sets_for_targets(targets)
  meta = subprocess.check_output(
      [
          "git",
          "-C",
          os.fspath(pytorch_dir),
          "show",
          "-s",
          "--format=%H\t%ci\t%s",
          sha,
      ],
      text=True,
  ).strip()
  parts = meta.split("\t", 2)
  full_sha = parts[0]
  committed_at = parts[1] if len(parts) > 1 else ""
  subject = parts[2] if len(parts) > 2 else ""
  ls_tree = subprocess.check_output(
      [
          "git",
          "-C",
          os.fspath(pytorch_dir),
          "ls-tree",
          "-r",
          full_sha,
          "--",
          "test/",
      ],
      text=True,
  )
  fp = compute_pytorch_test_fingerprint(
      ls_tree, tracked_module_files, tracked_subdirs
  )
  return CommitInfo(
      index=idx,
      sha=full_sha,
      committed_at=committed_at,
      subject=subject,
      fingerprint=fp,
  )


def list_commit_window(
    pytorch_dir: pathlib.Path,
    base_sha: str,
    head_sha: str,
    targets: Mapping[str, TorchTestTarget],
) -> tuple[CommitInfo, Sequence[CommitInfo]]:
  """Lists `base_sha` and all first-parent commits in `base_sha..head_sha`."""
  base_info = make_commit_info(pytorch_dir, 0, base_sha, targets)
  _ensure_commit_fetched(pytorch_dir, head_sha)
  raw_shas = subprocess.check_output(
      [
          "git",
          "-C",
          os.fspath(pytorch_dir),
          "log",
          "--first-parent",
          "--reverse",
          "--format=%H",
          f"{base_sha}..{head_sha}",
      ],
      text=True,
  ).splitlines()
  commits = [
      make_commit_info(pytorch_dir, idx, sha.strip(), targets)
      for idx, sha in enumerate(raw_shas, 1)
      if sha.strip()
  ]
  return base_info, commits


def _prepare_local_downloader_config(
    workspace_dir: pathlib.Path, work_dir: pathlib.Path
) -> pathlib.Path:
  """Creates a downloader config adding a pypi.org fallback for local runs."""
  base_cfg = workspace_dir / "bazel_downloader.cfg"
  content = base_cfg.read_text(encoding="utf-8") if base_cfg.is_file() else ""
  out_cfg = work_dir / "local_downloader.cfg"
  out_cfg.write_text(content + "\n" + _PYPI_MIRROR_REWRITE, encoding="utf-8")
  return out_cfg


def _ensure_bazel_diff_jar(jar_path: pathlib.Path) -> pathlib.Path:
  """Downloads `bazel-diff` deploy jar if missing."""
  if jar_path.is_file() and jar_path.stat().st_size > 0:
    return jar_path
  return download_file(_BAZEL_DIFF_URL, jar_path)


def _generate_hashes_worker(
    worker_id: int,
    states: Sequence[tuple[str, str]],
    *,
    workspace_dir: pathlib.Path,
    pytorch_dir: pathlib.Path,
    bazel_bin: str,
    bazel_config: str,
    wheel_path: pathlib.Path,
    downloader_cfg: pathlib.Path,
    bazel_diff_jar: pathlib.Path,
    hashes_dir: pathlib.Path,
    work_dir: pathlib.Path,
    isolation: OutputBaseIsolation,
) -> None:
  """Runs `bazel-diff generate-hashes` for a shard of `(fingerprint, sha)` pairs."""
  wt_dir = work_dir / f"pytorch_wt_{worker_id}"
  output_base_dir = work_dir / f"bazel_out_{worker_id}"
  if wt_dir.exists():
    subprocess.run(
        [
            "git",
            "-C",
            os.fspath(pytorch_dir),
            "worktree",
            "remove",
            "--force",
            os.fspath(wt_dir),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(wt_dir, ignore_errors=True)

  first_sha = states[0][1]
  subprocess.run(
      [
          "git",
          "-C",
          os.fspath(pytorch_dir),
          "worktree",
          "add",
          "--detach",
          "--no-checkout",
          os.fspath(wt_dir),
          first_sha,
      ],
      check=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
  )
  try:
    subprocess.run(
        ["git", "-C", os.fspath(wt_dir), "sparse-checkout", "init", "--cone"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", os.fspath(wt_dir), "sparse-checkout", "set", "test"],
        check=True,
    )
    env = dict(os.environ)
    env["NETRC"] = "/dev/null"

    startup_args: list[str] = []
    if isolation == OutputBaseIsolation.ISOLATED:
      output_base_dir.mkdir(parents=True, exist_ok=True)
      startup_args.extend(
          ["-so", f"--output_base={os.fspath(output_base_dir)}"]
      )

    cquery_opts = (
        f"--config={bazel_config} --config=no_rbe --bes_backend= "
        f"--downloader_config={os.fspath(downloader_cfg)} "
        "--config=local_torch "
        f"--repo_env=TORCH_SOURCE={os.fspath(wheel_path)} "
        f"--repo_env=TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR={os.fspath(wt_dir / 'test')}"
    )

    for step_idx, (fp, sha) in enumerate(states, 1):
      out_json = hashes_dir / f"hashes_{fp}.json"
      if out_json.is_file() and out_json.stat().st_size > 0:
        print(
            f"[worker {worker_id}] [{step_idx}/{len(states)}] Reusing cached "
            f"hashes for fp={fp} ({sha[:10]})"
        )
        continue
      print(
          f"[worker {worker_id}] [{step_idx}/{len(states)}] Hashing fp={fp} "
          f"at {sha[:10]}..."
      )
      subprocess.run(
          ["git", "-C", os.fspath(wt_dir), "checkout", "-q", "-f", sha],
          check=True,
      )
      marker = wt_dir / "test" / ".pytorch_test_tree"
      marker.write_text(f"{fp}\n", encoding="utf-8")
      tmp_json = hashes_dir / f"hashes_{fp}.w{worker_id}.part.json"
      cmd = [
          "java",
          "-jar",
          os.fspath(bazel_diff_jar),
          "generate-hashes",
          "-b",
          bazel_bin,
          "-w",
          os.fspath(workspace_dir),
          *startup_args,
          "--excludeExternalTargets",
          "--useCquery",
          f"--cqueryExpression={_CQUERY_EXPRESSION}",
          f"--cqueryCommandOptions={cquery_opts}",
          f"--fineGrainedHashExternalRepos={_PYTORCH_TEST_CANONICAL_REPO}",
          os.fspath(tmp_json),
      ]
      try:
        subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
      except subprocess.CalledProcessError as exc:
        if exc.stderr:
          print(exc.stderr, file=sys.stderr)
        raise
      tmp_json.replace(out_json)
      print(
          f"[worker {worker_id}] [{step_idx}/{len(states)}] Completed fp={fp} "
          f"({sha[:10]})"
      )
  finally:
    if isolation == OutputBaseIsolation.ISOLATED and output_base_dir.is_dir():
      subprocess.run(
          [
              bazel_bin,
              f"--output_base={os.fspath(output_base_dir)}",
              "shutdown",
          ],
          cwd=os.fspath(workspace_dir),
          check=False,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
    subprocess.run(
        [
            "git",
            "-C",
            os.fspath(pytorch_dir),
            "worktree",
            "remove",
            "--force",
            os.fspath(wt_dir),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(wt_dir, ignore_errors=True)


def compute_impacted_targets_raw(
    start_hashes: pathlib.Path,
    final_hashes: pathlib.Path,
    *,
    workspace_dir: pathlib.Path,
    bazel_bin: str,
    bazel_diff_jar: pathlib.Path,
) -> tuple[str, ...]:
  """Runs `bazel-diff get-impacted-targets` and returns raw output lines."""
  with tempfile.NamedTemporaryFile(mode="r+", suffix=".txt") as tmp_out:
    cmd = [
        "java",
        "-jar",
        os.fspath(bazel_diff_jar),
        "get-impacted-targets",
        "--excludeExternalTargets",
        "-b",
        bazel_bin,
        "-w",
        os.fspath(workspace_dir),
        "-sh",
        os.fspath(start_hashes),
        "-fh",
        os.fspath(final_hashes),
        "-o",
        tmp_out.name,
    ]
    env = dict(os.environ)
    env["NETRC"] = "/dev/null"
    try:
      subprocess.run(
          cmd,
          env=env,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.PIPE,
          text=True,
          check=True,
      )
    except subprocess.CalledProcessError as exc:
      if exc.stderr:
        print(exc.stderr, file=sys.stderr)
      raise
    return tuple(
        pathlib.Path(tmp_out.name).read_text(encoding="utf-8").splitlines()
    )


def compute_impacted_targets_pair(
    start_hashes: pathlib.Path,
    final_hashes: pathlib.Path,
    *,
    workspace_dir: pathlib.Path,
    bazel_bin: str,
    bazel_diff_jar: pathlib.Path,
    targets: Mapping[str, TorchTestTarget],
    bazel_config: str = "",
) -> tuple[str, ...]:
  """Runs `bazel-diff get-impacted-targets` and filters for `bazel_config`."""
  lines = compute_impacted_targets_raw(
      start_hashes,
      final_hashes,
      workspace_dir=workspace_dir,
      bazel_bin=bazel_bin,
      bazel_diff_jar=bazel_diff_jar,
  )
  return filter_enabled_torch_tests(lines, targets, bazel_config=bazel_config)


def write_ci_outputs(
    *,
    per_config_targets: Mapping[str, Sequence[str]],
    primary_config: str,
    impacted_targets_output: pathlib.Path | None,
    impacted_targets_dir: pathlib.Path | None,
    github_output: pathlib.Path | None,
) -> None:
  """Writes per-config impacted target files and `$GITHUB_OUTPUT` keys."""
  if impacted_targets_output is not None:
    primary_targets = per_config_targets.get(primary_config, ())
    impacted_targets_output.parent.mkdir(parents=True, exist_ok=True)
    impacted_targets_output.write_text(
        "\n".join(primary_targets) + ("\n" if primary_targets else ""),
        encoding="utf-8",
    )

  if impacted_targets_dir is not None:
    impacted_targets_dir.mkdir(parents=True, exist_ok=True)
    for cfg, cfg_targets in per_config_targets.items():
      cfg_file = impacted_targets_dir / f"impacted_targets_{cfg}.txt"
      cfg_file.write_text(
          "\n".join(cfg_targets) + ("\n" if cfg_targets else ""),
          encoding="utf-8",
      )

  any_impacted = any(bool(t) for t in per_config_targets.values())
  gh_lines = [
      f"has_impacted_targets={str(any_impacted).lower()}",
      "is_full_run=false",
  ]
  for cfg, cfg_targets in per_config_targets.items():
    gh_lines.append(f"has_impacted_{cfg}={str(bool(cfg_targets)).lower()}")
    gh_lines.append(f"impacted_count_{cfg}={len(cfg_targets)}")
  for line in gh_lines:
    print(f"[ci-output] {line}")
  if github_output is not None:
    github_output.parent.mkdir(parents=True, exist_ok=True)
    with github_output.open("a", encoding="utf-8") as f:
      f.write("\n".join(gh_lines) + "\n")


def render_markdown_report(
    *,
    prev_wheel: WheelMetadata | None,
    latest_wheel: WheelMetadata | None,
    base_commit: CommitInfo,
    impacts: Sequence[CommitImpact],
    targets: Mapping[str, TorchTestTarget],
    bazel_config: str,
) -> str:
  """Renders a Markdown report summarizing per-commit TPU test impact."""
  enabled_targets = [t for t in targets.values() if t.enabled]
  total_enabled = len(enabled_targets)
  total_shards = sum(t.shard_count for t in enabled_targets)
  num_commits = len(impacts)

  full_suite_target_runs = num_commits * total_enabled
  full_suite_shard_runs = num_commits * total_shards

  inc_zero_commits = sum(1 for item in impacts if not item.incremental_targets)
  inc_total_targets = sum(len(item.incremental_targets) for item in impacts)
  inc_total_shards = sum(item.incremental_shards for item in impacts)

  wheel_zero_commits = sum(1 for item in impacts if not item.vs_wheel_targets)
  wheel_total_targets = sum(len(item.vs_wheel_targets) for item in impacts)
  wheel_total_shards = sum(item.vs_wheel_shards for item in impacts)

  inc_shard_saving = (
      100.0 * (1.0 - inc_total_shards / full_suite_shard_runs)
      if full_suite_shard_runs
      else 100.0
  )
  wheel_shard_saving = (
      100.0 * (1.0 - wheel_total_shards / full_suite_shard_runs)
      if full_suite_shard_runs
      else 100.0
  )

  lines: list[str] = [
      "# PyTorch Out-of-Tree TPU Impacted Tests Report (`bazel-diff`)",
      "",
      "## Configuration & Commit Window",
      "",
      (
          f"* **Bazel TPU Config**: `{bazel_config}` (`--config=local_torch` +"
          " `@pytorch_test`)"
      ),
      (
          f"* **Enabled `//tests/torch:*` TPU Targets**: **{total_enabled}**"
          f" targets (**{total_shards}** shards)"
      ),
  ]
  if prev_wheel is not None:
    lines.append(
        "* **Base Nightly Wheel (`TORCH_SOURCE`)**:"
        f" `torch=={prev_wheel.version}` (wheel SHA"
        f" `{prev_wheel.wheel_sha[:12]}`, `main` SHA"
        f" `{prev_wheel.main_sha[:12]}`)"
    )
  else:
    lines.append(f"* **Base `main` SHA**: `{base_commit.sha[:12]}`")
  if latest_wheel is not None:
    lines.append(
        f"* **Target Nightly Wheel**: `torch=={latest_wheel.version}` (wheel"
        f" SHA `{latest_wheel.wheel_sha[:12]}`, `main` SHA"
        f" `{latest_wheel.main_sha[:12]}`)"
    )
  lines.extend([
      f"* **Commits Analyzed on `main`**: **{num_commits}** commits"
      f" (`{base_commit.sha[:12]}..{impacts[-1].commit.sha[:12]}`)"
      if impacts
      else "* **Commits Analyzed on `main`**: 0 commits",
      (
          "* **Distinct `@pytorch_test` States Hashed**:"
          f" **{len({base_commit.fingerprint, *(i.commit.fingerprint for i in impacts)})}**"
      ),
      "",
      "## Executive Summary (Workload & Cache Reduction)",
      "",
      (
          "| Strategy | Commits w/ 0 TPU Tests | Total Target Runs | Total"
          " Shard Runs | Avg Shards / Commit | Shard Reduction vs. Full Suite |"
      ),
      "| :--- | ---: | ---: | ---: | ---: | ---: |",
      (
          f"| Full Suite (no cache / no diff) | 0 / {num_commits} |"
          f" {full_suite_target_runs} | {full_suite_shard_runs} |"
          f" {total_shards:.2f} | 0.0% |"
      ),
      (
          "| Vs. Fixed Base Nightly Wheel (`W_prev -> C_i`) |"
          f" {wheel_zero_commits} / {num_commits} | {wheel_total_targets} |"
          f" {wheel_total_shards} |"
          f" {(wheel_total_shards / num_commits if num_commits else 0.0):.2f} |"
          f" **{wheel_shard_saving:.1f}%** |"
      ),
      (
          "| Incremental w/ Shared RBE Cache (`C_{i-1} -> C_i`) |"
          f" {inc_zero_commits} / {num_commits} | {inc_total_targets} |"
          f" {inc_total_shards} |"
          f" {(inc_total_shards / num_commits if num_commits else 0.0):.2f} |"
          f" **{inc_shard_saving:.1f}%** |"
      ),
      "",
      "## Per-Commit `bazel-diff` Impact Breakdown",
      "",
      (
          "| # | Commit (`main`) | Timestamp | Incremental (`C_{i-1} -> C_i`) |"
          " Vs. Base Wheel (`W_prev -> C_i`) | Incremental Impacted Targets |"
          " Commit Subject |"
      ),
      "| ---: | :--- | :--- | ---: | ---: | :--- | :--- |",
  ])

  for item in impacts:
    c = item.commit
    inc_str = f"{len(item.incremental_targets)} ({item.incremental_shards} sh)"
    whl_str = f"{len(item.vs_wheel_targets)} ({item.vs_wheel_shards} sh)"
    inc_targets_fmt = (
        ", ".join(
            f"`{t.removeprefix('//tests/torch:')}`"
            for t in item.incremental_targets
        )
        if item.incremental_targets
        else "—"
    )
    safe_subj = c.subject.replace("|", "\\|")
    if len(safe_subj) > 72:
      safe_subj = safe_subj[:69] + "..."
    lines.append(
        f"| {c.index} | `{c.sha[:10]}` | {c.committed_at[:19]} | {inc_str} | "
        f"{whl_str} | {inc_targets_fmt} | {safe_subj} |"
    )

  if impacts:
    cumulative = impacts[-1].vs_wheel_targets
    lines.extend([
        "",
        (
            "## Cumulative Impacted Targets Across Full Window (`W_prev ->"
            " W_latest`)"
        ),
        "",
        (
            "Total distinct enabled TPU targets touched across all"
            f" {num_commits} commits: **{len(cumulative)} / {total_enabled}**"
            f" (**{impacts[-1].vs_wheel_shards} / {total_shards}** shards):"
        ),
        "",
    ])
    for label in cumulative:
      t = targets[label]
      lines.append(
          f"* `{label}` (`test/{t.module}.py`, {t.shard_count} shard(s))"
      )

  lines.append("")
  return "\n".join(lines)


def build_json_payload(
    *,
    prev_wheel: WheelMetadata | None,
    latest_wheel: WheelMetadata | None,
    base_commit: CommitInfo,
    impacts: Sequence[CommitImpact],
    targets: Mapping[str, TorchTestTarget],
    bazel_config: str,
) -> Mapping[str, Any]:
  """Builds a machine-readable JSON dictionary of the analysis results."""
  enabled_targets = [t for t in targets.values() if t.enabled]
  return {
      "bazel_config": bazel_config,
      "enabled_target_count": len(enabled_targets),
      "enabled_shard_count": sum(t.shard_count for t in enabled_targets),
      "prev_wheel": (
          dataclasses.asdict(prev_wheel) if prev_wheel is not None else None
      ),
      "latest_wheel": (
          dataclasses.asdict(latest_wheel) if latest_wheel is not None else None
      ),
      "base_commit": dataclasses.asdict(base_commit),
      "commits": [
          {
              "index": item.commit.index,
              "sha": item.commit.sha,
              "committed_at": item.commit.committed_at,
              "subject": item.commit.subject,
              "fingerprint": item.commit.fingerprint,
              "incremental_targets": list(item.incremental_targets),
              "incremental_shards": item.incremental_shards,
              "vs_wheel_targets": list(item.vs_wheel_targets),
              "vs_wheel_shards": item.vs_wheel_shards,
          }
          for item in impacts
      ],
  }


def _optional_path(raw: str) -> pathlib.Path | None:
  """Converts a non-empty CLI string to `Path`, or `None` when empty."""
  return pathlib.Path(raw) if raw.strip() else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses CLI arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--workspace-dir",
      type=pathlib.Path,
      default=pathlib.Path.cwd(),
      help="Path to the TorchTPU OSS Bazel workspace.",
  )
  parser.add_argument(
      "--pytorch-dir",
      type=pathlib.Path,
      required=True,
      help="Path to a local pytorch/pytorch git repository.",
  )
  parser.add_argument(
      "--prev-wheel",
      type=_optional_path,
      default=None,
      help="Optional path to yesterday's PyTorch nightly CPU .whl file.",
  )
  parser.add_argument(
      "--latest-wheel",
      type=_optional_path,
      default=None,
      help="Optional path to the latest PyTorch nightly CPU .whl file.",
  )
  parser.add_argument(
      "--base-sha",
      default="",
      help="Optional explicit base main SHA (overrides prev-wheel SHA).",
  )
  parser.add_argument(
      "--head-sha",
      default="",
      help="Optional explicit head main SHA (overrides latest-wheel SHA).",
  )
  parser.add_argument(
      "--bazel-bin",
      default="bazel",
      help="Bazel executable path or name.",
  )
  parser.add_argument(
      "--bazel-config",
      default="ci_tpu_v5",
      help="TPU Bazel --config name (e.g. ci_tpu_v5 or ci_tpu_v7).",
  )
  parser.add_argument(
      "--bazel-diff-jar",
      type=pathlib.Path,
      default=_DEFAULT_BAZEL_DIFF_JAR,
      help="Path to bazel-diff_deploy.jar (downloaded if absent).",
  )
  parser.add_argument(
      "--work-dir",
      type=pathlib.Path,
      default=pathlib.Path(tempfile.gettempdir()) / "oot_bazel_diff_work",
      help="Scratch directory for wheels, hashes, and temporary worktrees.",
  )
  parser.add_argument(
      "--jobs",
      type=int,
      default=1,
      help="Number of parallel Bazel worker instances for hash generation.",
  )
  parser.add_argument(
      "--json-output",
      type=_optional_path,
      default=None,
      help="Optional path to write JSON report.",
  )
  parser.add_argument(
      "--markdown-output",
      type=_optional_path,
      default=None,
      help="Optional path to write Markdown report.",
  )
  parser.add_argument(
      "--single-diff",
      action="store_true",
      help=(
          "Compare --base-sha vs --head-sha directly as a single pair (CI "
          "pre-filter mode) without walking intermediate commits."
      ),
  )
  parser.add_argument(
      "--filter-configs",
      default="",
      help=(
          "Optional comma-separated Bazel configs to emit filtered target "
          "lists for (defaults to --bazel-config)."
      ),
  )
  parser.add_argument(
      "--impacted-targets-output",
      type=_optional_path,
      default=None,
      help="Optional path to write newline-delimited impacted target labels.",
  )
  parser.add_argument(
      "--impacted-targets-dir",
      type=_optional_path,
      default=None,
      help=(
          "Optional directory to write per-config "
          "impacted_targets_<config>.txt files."
      ),
  )
  parser.add_argument(
      "--github-output",
      type=_optional_path,
      default=None,
      help="Optional path to $GITHUB_OUTPUT file for CI job hand-off.",
  )
  return parser.parse_args(argv)


def _run_single_diff_mode(
    args: argparse.Namespace,
    *,
    workspace_dir: pathlib.Path,
    pytorch_dir: pathlib.Path,
    work_dir: pathlib.Path,
    hashes_dir: pathlib.Path,
    targets: Mapping[str, TorchTestTarget],
) -> int:
  """Executes single-pair CI diff mode (`base_sha -> head_sha`)."""
  filter_configs = tuple(
      dict.fromkeys(
          cfg.strip()
          for cfg in (args.filter_configs or args.bazel_config).split(",")
          if cfg.strip()
      )
  ) or (args.bazel_config,)

  prev_meta: WheelMetadata | None = None
  latest_meta: WheelMetadata | None = None
  base_sha: str = args.base_sha
  head_sha: str = args.head_sha

  if not base_sha or not head_sha:
    prev_wheel_path = args.prev_wheel
    latest_wheel_path = args.latest_wheel
    if (not base_sha and prev_wheel_path is None) or (
        not head_sha and latest_wheel_path is None
    ):
      py_ver = default_python_version(workspace_dir)
      dl_prev, dl_latest = resolve_latest_two_wheels(
          py_ver, work_dir / "wheels"
      )
      prev_wheel_path = prev_wheel_path or dl_prev
      latest_wheel_path = latest_wheel_path or dl_latest
    if prev_wheel_path is not None:
      prev_meta = inspect_wheel(prev_wheel_path, pytorch_dir)
    if latest_wheel_path is not None:
      latest_meta = inspect_wheel(latest_wheel_path, pytorch_dir)
    if not base_sha:
      if prev_meta is None:
        raise ValueError("Unable to resolve base SHA without --prev-wheel.")
      base_sha = prev_meta.main_sha
    if not head_sha:
      if latest_meta is None:
        raise ValueError("Unable to resolve head SHA without --latest-wheel.")
      head_sha = latest_meta.main_sha

  base_commit = make_commit_info(pytorch_dir, 0, base_sha, targets)
  head_commit = make_commit_info(pytorch_dir, 1, head_sha, targets)
  print(
      f"Single-diff mode: base={base_commit.sha[:12]}"
      f" (fp={base_commit.fingerprint}) -> head={head_commit.sha[:12]}"
      f" (fp={head_commit.fingerprint})"
  )

  raw_lines: tuple[str, ...]
  if base_commit.fingerprint == head_commit.fingerprint:
    print(
        "Fast path: @pytorch_test tree fingerprint unchanged "
        f"({base_commit.fingerprint}); 0 TPU targets impacted."
    )
    raw_lines = ()
  else:
    wheel_for_hash = args.latest_wheel or args.prev_wheel
    if wheel_for_hash is None:
      py_ver = default_python_version(workspace_dir)
      _, dl_latest = resolve_latest_two_wheels(py_ver, work_dir / "wheels")
      wheel_for_hash = dl_latest
    if latest_meta is None:
      latest_meta = inspect_wheel(wheel_for_hash, pytorch_dir)

    bazel_diff_jar = _ensure_bazel_diff_jar(args.bazel_diff_jar.resolve())
    downloader_cfg = _prepare_local_downloader_config(workspace_dir, work_dir)
    states = [
        (base_commit.fingerprint, base_commit.sha),
        (head_commit.fingerprint, head_commit.sha),
    ]
    resolved_wheel = wheel_for_hash.resolve()
    num_jobs = max(1, min(args.jobs, len(states)))
    if num_jobs == 1:
      _generate_hashes_worker(
          0,
          states,
          workspace_dir=workspace_dir,
          pytorch_dir=pytorch_dir,
          bazel_bin=args.bazel_bin,
          bazel_config=args.bazel_config,
          wheel_path=resolved_wheel,
          downloader_cfg=downloader_cfg,
          bazel_diff_jar=bazel_diff_jar,
          hashes_dir=hashes_dir,
          work_dir=work_dir,
          isolation=OutputBaseIsolation.SHARED,
      )
    else:
      with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as pool:
        futures = [
            pool.submit(
                _generate_hashes_worker,
                worker_id,
                [state],
                workspace_dir=workspace_dir,
                pytorch_dir=pytorch_dir,
                bazel_bin=args.bazel_bin,
                bazel_config=args.bazel_config,
                wheel_path=resolved_wheel,
                downloader_cfg=downloader_cfg,
                bazel_diff_jar=bazel_diff_jar,
                hashes_dir=hashes_dir,
                work_dir=work_dir,
                isolation=OutputBaseIsolation.ISOLATED,
            )
            for worker_id, state in enumerate(states)
        ]
        for fut in concurrent.futures.as_completed(futures):
          fut.result()
    raw_lines = compute_impacted_targets_raw(
        hashes_dir / f"hashes_{base_commit.fingerprint}.json",
        hashes_dir / f"hashes_{head_commit.fingerprint}.json",
        workspace_dir=workspace_dir,
        bazel_bin=args.bazel_bin,
        bazel_diff_jar=bazel_diff_jar,
    )

  per_config_targets = {
      cfg: filter_enabled_torch_tests(raw_lines, targets, bazel_config=cfg)
      for cfg in filter_configs
  }
  primary_targets = per_config_targets.get(
      args.bazel_config,
      filter_enabled_torch_tests(
          raw_lines, targets, bazel_config=args.bazel_config
      ),
  )
  primary_shards = sum(targets[t].shard_count for t in primary_targets)
  impacts = [
      CommitImpact(
          commit=head_commit,
          vs_wheel_targets=primary_targets,
          vs_wheel_shards=primary_shards,
          incremental_targets=primary_targets,
          incremental_shards=primary_shards,
      )
  ]

  write_ci_outputs(
      per_config_targets=per_config_targets,
      primary_config=args.bazel_config,
      impacted_targets_output=args.impacted_targets_output,
      impacted_targets_dir=args.impacted_targets_dir,
      github_output=args.github_output,
  )

  md_report = render_markdown_report(
      prev_wheel=prev_meta,
      latest_wheel=latest_meta,
      base_commit=base_commit,
      impacts=impacts,
      targets=targets,
      bazel_config=args.bazel_config,
  )
  if args.markdown_output is not None:
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(md_report, encoding="utf-8")
    print(f"Wrote Markdown report to {args.markdown_output}")
  else:
    print(md_report)

  if args.json_output is not None:
    payload = dict(
        build_json_payload(
            prev_wheel=prev_meta,
            latest_wheel=latest_meta,
            base_commit=base_commit,
            impacts=impacts,
            targets=targets,
            bazel_config=args.bazel_config,
        )
    )
    payload["per_config_targets"] = {
        cfg: list(cfg_targets)
        for cfg, cfg_targets in per_config_targets.items()
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote JSON report to {args.json_output}")

  return 0


def main(argv: Sequence[str] | None = None) -> int:
  sys.stdout.reconfigure(line_buffering=True)
  args = parse_args(argv)

  workspace_dir = args.workspace_dir.resolve()
  pytorch_dir = args.pytorch_dir.resolve()
  work_dir = args.work_dir.resolve()
  work_dir.mkdir(parents=True, exist_ok=True)
  hashes_dir = work_dir / f"hashes_{args.bazel_config}"
  hashes_dir.mkdir(parents=True, exist_ok=True)

  build_file = workspace_dir / "tests" / "torch" / "BUILD"
  targets = parse_torch_test_targets(build_file.read_text(encoding="utf-8"))

  if args.single_diff:
    return _run_single_diff_mode(
        args,
        workspace_dir=workspace_dir,
        pytorch_dir=pytorch_dir,
        work_dir=work_dir,
        hashes_dir=hashes_dir,
        targets=targets,
    )

  prev_wheel_path = args.prev_wheel
  latest_wheel_path = args.latest_wheel
  if prev_wheel_path is None or latest_wheel_path is None:
    py_ver = default_python_version(workspace_dir)
    dl_prev, dl_latest = resolve_latest_two_wheels(py_ver, work_dir / "wheels")
    prev_wheel_path = prev_wheel_path or dl_prev
    latest_wheel_path = latest_wheel_path or dl_latest

  prev_meta = inspect_wheel(prev_wheel_path, pytorch_dir)
  latest_meta = inspect_wheel(latest_wheel_path, pytorch_dir)
  base_sha = args.base_sha or prev_meta.main_sha
  head_sha = args.head_sha or latest_meta.main_sha

  print(
      f"Resolved wheels: prev={prev_meta.version}"
      f" (main={prev_meta.main_sha[:12]}), latest={latest_meta.version}"
      f" (main={latest_meta.main_sha[:12]})"
  )

  base_commit, commits = list_commit_window(
      pytorch_dir, base_sha, head_sha, targets
  )
  print(
      f"Found {len(commits)} first-parent commits in"
      f" {base_sha[:12]}..{head_sha[:12]}."
  )

  unique_states: dict[str, str] = {base_commit.fingerprint: base_commit.sha}
  for c in commits:
    unique_states.setdefault(c.fingerprint, c.sha)
  state_items = list(unique_states.items())
  print(
      f"Deduplicated {len(commits) + 1} commits to {len(state_items)} distinct "
      "@pytorch_test input tree states."
  )

  bazel_diff_jar = _ensure_bazel_diff_jar(args.bazel_diff_jar.resolve())
  downloader_cfg = _prepare_local_downloader_config(workspace_dir, work_dir)

  num_jobs = max(1, min(args.jobs, len(state_items)))
  shards: list[list[tuple[str, str]]] = [[] for _ in range(num_jobs)]
  for idx, item in enumerate(state_items):
    shards[idx % num_jobs].append(item)

  if num_jobs == 1:
    _generate_hashes_worker(
        0,
        shards[0],
        workspace_dir=workspace_dir,
        pytorch_dir=pytorch_dir,
        bazel_bin=args.bazel_bin,
        bazel_config=args.bazel_config,
        wheel_path=prev_meta.path,
        downloader_cfg=downloader_cfg,
        bazel_diff_jar=bazel_diff_jar,
        hashes_dir=hashes_dir,
        work_dir=work_dir,
        isolation=OutputBaseIsolation.SHARED,
    )
  else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as pool:
      futures = [
          pool.submit(
              _generate_hashes_worker,
              worker_id,
              shard,
              workspace_dir=workspace_dir,
              pytorch_dir=pytorch_dir,
              bazel_bin=args.bazel_bin,
              bazel_config=args.bazel_config,
              wheel_path=prev_meta.path,
              downloader_cfg=downloader_cfg,
              bazel_diff_jar=bazel_diff_jar,
              hashes_dir=hashes_dir,
              work_dir=work_dir,
              isolation=OutputBaseIsolation.ISOLATED,
          )
          for worker_id, shard in enumerate(shards)
          if shard
      ]
      for fut in concurrent.futures.as_completed(futures):
        fut.result()

  pair_cache: dict[tuple[str, str], tuple[str, ...]] = {}

  def _diff_fps(fp_a: str, fp_b: str) -> tuple[str, ...]:
    if fp_a == fp_b:
      return ()
    key = (fp_a, fp_b)
    if key not in pair_cache:
      pair_cache[key] = compute_impacted_targets_pair(
          hashes_dir / f"hashes_{fp_a}.json",
          hashes_dir / f"hashes_{fp_b}.json",
          workspace_dir=workspace_dir,
          bazel_bin=args.bazel_bin,
          bazel_diff_jar=bazel_diff_jar,
          targets=targets,
          bazel_config=args.bazel_config,
      )
    return pair_cache[key]

  impacts: list[CommitImpact] = []
  prev_fp = base_commit.fingerprint
  for c in commits:
    vs_wheel = _diff_fps(base_commit.fingerprint, c.fingerprint)
    incremental = _diff_fps(prev_fp, c.fingerprint)
    impacts.append(
        CommitImpact(
            commit=c,
            vs_wheel_targets=vs_wheel,
            vs_wheel_shards=sum(targets[t].shard_count for t in vs_wheel),
            incremental_targets=incremental,
            incremental_shards=sum(targets[t].shard_count for t in incremental),
        )
    )
    prev_fp = c.fingerprint

  if (
      args.impacted_targets_output is not None
      or args.impacted_targets_dir is not None
      or args.github_output is not None
  ):
    cumulative = impacts[-1].vs_wheel_targets if impacts else ()
    write_ci_outputs(
        per_config_targets={args.bazel_config: cumulative},
        primary_config=args.bazel_config,
        impacted_targets_output=args.impacted_targets_output,
        impacted_targets_dir=args.impacted_targets_dir,
        github_output=args.github_output,
    )

  md_report = render_markdown_report(
      prev_wheel=prev_meta,
      latest_wheel=latest_meta,
      base_commit=base_commit,
      impacts=impacts,
      targets=targets,
      bazel_config=args.bazel_config,
  )
  if args.markdown_output is not None:
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(md_report, encoding="utf-8")
    print(f"Wrote Markdown report to {args.markdown_output}")
  else:
    print(md_report)

  if args.json_output is not None:
    payload = build_json_payload(
        prev_wheel=prev_meta,
        latest_wheel=latest_meta,
        base_commit=base_commit,
        impacts=impacts,
        targets=targets,
        bazel_config=args.bazel_config,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote JSON report to {args.json_output}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
