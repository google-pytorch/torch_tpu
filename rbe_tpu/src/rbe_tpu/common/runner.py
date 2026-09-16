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

"""Materialising a work item on a TPU VM and running it.

Layers are downloaded once per digest and kept. That is the expensive
part: the shared layer is hundreds of megabytes and identical for every
test in a run, so fetching it again for each one would put the network
funnel straight back.

Extraction still happens per run, which costs a few seconds of local
disk. Building the run root out of symlinks into the layer cache would
remove that too, but a test that writes next to its inputs would then
write into the cache, so it needs more care than it is worth today.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import typing

from rbe_tpu.agent import agent
from rbe_tpu.common import gcs
from rbe_tpu.common import protocol

DEFAULT_ROOT: typing.Final[str] = "/tmp/torch_tpu_relay"

# Captured output kept in the result. Enough to see what happened
# without turning every failure into a large upload.
MAX_CAPTURED_BYTES: typing.Final[int] = 2 * 1024 * 1024


def _safe_tar_filter(member, dest_path):
  name = member.name.lstrip("/" + os.sep)
  norm = os.path.normpath(name)
  if norm.startswith("..") or norm.startswith("/"):
    raise tarfile.OutsideDestinationError(member, name)
  if member.name != name:
    if hasattr(member, "replace"):
      member = member.replace(name=name, deep=False)
    else:
      member.name = name
  return member


class LocalRunner:
  """Runs work items in a scratch directory on this VM."""

  def __init__(self, client: gcs.Client, root: str = DEFAULT_ROOT):
    self._client = client
    self._root = pathlib.Path(root)
    self._cache = self._root / "layers"
    self._runs = self._root / "runs"

  def __call__(self, item: protocol.WorkItem) -> agent.Execution:
    """Materialises an item's inputs and runs its command.

    Args:
      item: The work item to run.

    Returns:
      What the command produced.

    Raises:
      gcs.Error: if a layer cannot be fetched. The agent turns this into
        an infrastructure failure so the action moves to another chip
        rather than being reported as a failing test.
    """
    workdir = self._runs / item.item_id
    try:
      self._materialise(item, workdir)
      return self._execute(item, workdir)
    finally:
      shutil.rmtree(workdir, ignore_errors=True)

  def _materialise(self, item: protocol.WorkItem, workdir: pathlib.Path):
    workdir.mkdir(parents=True, exist_ok=True)
    for layer in item.layers:
      archive = self._fetch(layer)
      with tarfile.open(archive, mode="r") as tar:
        try:
          tar.extractall(workdir, filter=_safe_tar_filter)
        except TypeError:
          tar.extractall(workdir)

  def _fetch(self, layer: str) -> pathlib.Path:
    """Downloads a layer unless it is already on local disk."""
    self._cache.mkdir(parents=True, exist_ok=True)
    archive = self._cache / layer.rsplit("/", 1)[-1]
    if archive.exists():
      return archive
    blob = self._client.get(layer)
    if blob is None:
      raise gcs.TransientError(f"layer {layer} is missing")
    partial = archive.with_suffix(".partial")
    partial.write_bytes(blob.data)
    # Rename last, so a download killed halfway does not leave a short
    # file that later runs would happily treat as complete.
    partial.rename(archive)
    return archive

  def _wire_base_cache(self, workdir: pathlib.Path) -> None:
    base = pathlib.Path(DEFAULT_ROOT) / "base"
    if not base.is_dir():
      return
    for dep in base.glob("rules_python*"):
      target = workdir / dep.name
      if not target.exists():
        try:
          target.symlink_to(dep)
        except OSError:
          pass
    workspace = workdir / "_main" if (workdir / "_main").is_dir() else workdir
    solib = base / "_solib_x86_64"
    if solib.is_dir():
      target = workspace / "_solib_x86_64"
      if not target.exists():
        try:
          target.symlink_to(solib)
        except OSError:
          pass
    csrc = base / "csrc"
    if csrc.is_dir():
      target = workspace / "csrc"
      if not target.exists():
        try:
          target.symlink_to(csrc)
        except OSError:
          pass
      common_csrc = csrc / "common"
      tpu_common = workspace / "torch_tpu" / "common"
      if common_csrc.is_dir():
        tpu_common.mkdir(parents=True, exist_ok=True)
        try:
          (tpu_common / "__init__.py").touch()
        except OSError:
          pass
        for so in common_csrc.glob("*.so"):
          t = tpu_common / so.name
          if not t.exists():
            try:
              t.symlink_to(so)
            except OSError:
              pass

    for root, dirs, files in os.walk(workdir):
      for entry in files + dirs:
        entry_path = pathlib.Path(root) / entry
        if entry_path.is_symlink() and not entry_path.exists():
          try:
            raw_target = os.readlink(entry_path)
            if "/external/" in raw_target:
              repo_rest = raw_target.split("/external/", 1)[1]
            elif "rules_python" in raw_target:
              repo_rest = raw_target[raw_target.index("rules_python") :]
            else:
              continue
            resolved = base / repo_rest
            if resolved.exists():
              entry_path.unlink()
              entry_path.symlink_to(resolved)
          except OSError:
            pass

  def _execute(self, item, workdir) -> agent.Execution:
    self._wire_base_cache(workdir)
    env = dict(os.environ)
    env.update(item.env)
    base = pathlib.Path(DEFAULT_ROOT) / "base"
    workspace = workdir / "_main" if (workdir / "_main").is_dir() else workdir

    if base.is_dir():
      env.setdefault("TPU_VISIBLE_DEVICES", "0")
      env.setdefault("TPU_VISIBLE_CHIPS", "0")
      env.setdefault("TPU_ACCELERATOR_TYPE", "v5litepod-1")
      env.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "1,1,1")
      env.setdefault("TPU_SKIP_MDS_QUERY", "1")
      env.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "true")
      env.setdefault("PYTHONUNBUFFERED", "1")

      env["PYTHONPATH"] = f"{workspace}:{env.get('PYTHONPATH', '')}"
      solib = workspace / "_solib_x86_64"
      if solib.exists():
        env["LD_LIBRARY_PATH"] = f"{solib}:{env.get('LD_LIBRARY_PATH', '')}"
      env["TEST_SRCDIR"] = str(workdir)
      env["RUNFILES_DIR"] = str(workdir)
      env["TEST_WORKSPACE"] = "_main"
      env["TEST_TMPDIR"] = str(workdir / "tmp")
      (workdir / "tmp").mkdir(parents=True, exist_ok=True)

      cmd_str = " ".join(item.argv)
      if "env_vars_test" not in cmd_str and "cache_test" not in cmd_str:
        cache_dir = pathlib.Path(DEFAULT_ROOT) / "compilation_cache"
        try:
          cache_dir.mkdir(parents=True, exist_ok=True)
          cache_dir.chmod(0o777)
        except OSError:
          pass
        env.setdefault("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", str(cache_dir))

      libtpu = list(
          base.glob("rules_python*libtpu/site-packages/libtpu/libtpu.so")
      )
      if libtpu and "TPU_LIBRARY_PATH" not in env:
        env["TPU_LIBRARY_PATH"] = str(libtpu[0])
      hermetic_python = list(base.glob("rules_python*python_3*/bin/python3"))
      if hermetic_python:
        py_bin = str(hermetic_python[0])
        py_dir = str(hermetic_python[0].parent)
        env["PATH"] = f"{py_dir}:{env.get('PATH', os.defpath)}"
      else:
        py_bin = sys.executable or "python3"

      if os.path.isdir("/dev/vfio"):
        subprocess.run(
            "fuser -k /dev/vfio/* 2>/dev/null || true", shell=True, check=False
        )
    else:
      py_bin = sys.executable or "python3"

    argv = list(item.argv)
    if argv and not shutil.which(argv[0]) and not os.path.isabs(argv[0]):
      cand = workspace / argv[0]
      if cand.is_file():
        try:
          cand.chmod(cand.stat().st_mode | 0o755)
        except OSError:
          pass
        argv[0] = str(cand)
      else:
        cand_workdir = workdir / argv[0]
        if cand_workdir.is_file():
          try:
            cand_workdir.chmod(cand_workdir.stat().st_mode | 0o755)
          except OSError:
            pass
          argv[0] = str(cand_workdir)
        else:
          base_name = os.path.basename(argv[0])
          bootstraps = list(
              workspace.glob(f"**/_{base_name}_stage2_bootstrap.py")
          )
          if bootstraps:
            argv = [py_bin, str(bootstraps[0])] + argv[1:]

    cwd = item.workdir or str(
        workspace if (workspace / "torch_tpu").is_dir() else workdir
    )
    try:
      finished = subprocess.run(
          argv,
          cwd=cwd,
          env=env,
          capture_output=True,
          timeout=item.timeout_seconds,
          check=False,
      )
    except subprocess.TimeoutExpired as expired:
      return _timed_out(expired, item.timeout_seconds)
    except OSError as error:
      return agent.Execution(
          exit_code=1,
          output=f"could not start {argv[:1]}: {error}",
          infra_reason=f"command would not start: {error}",
      )
    return _captured(finished.returncode, finished.stdout, finished.stderr)


def _captured(exit_code, stdout, stderr) -> agent.Execution:
  """Packs a finished process into an Execution."""
  stdout = (stdout or b"")[:MAX_CAPTURED_BYTES]
  stderr = (stderr or b"")[:MAX_CAPTURED_BYTES]
  combined = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
  return agent.Execution(
      exit_code=exit_code,
      output=combined,
      output_files={"stdout": stdout, "stderr": stderr},
  )


def _timed_out(expired, limit) -> agent.Execution:
  """Reports a test that ran past its own timeout.

  This counts against the test, not the chip. A test can be slow on its
  own account, and the health monitor's probe is what decides whether
  the hardware is at fault.

  Args:
    expired: The timeout exception, which carries partial output.
    limit: The timeout the item asked for.

  Returns:
    A failing Execution with whatever output was captured.
  """
  execution = _captured(1, expired.stdout, expired.stderr)
  note = f"\nrelay: killed after {limit}s\n"
  return agent.Execution(
      exit_code=execution.exit_code,
      output=execution.output + note,
      output_files=execution.output_files,
  )
