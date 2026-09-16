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

"""Hands one command to a TPU and reports what happened.

Meant for bazel's --run_under, so everything after the program name is
the command to run.

Settings come from the environment:

  TORCH_TPU_RELAY_BUCKET      relay bucket name (required)
  TORCH_TPU_RELAY_LAYER_DIRS  colon-separated input layers, in extraction
                              order, biggest and least-changing first
  TORCH_TPU_RELAY_TIMEOUT     per-item timeout in seconds, default 900
  TORCH_TPU_RELAY_WORKER_ID   identity for logs, default host and pid

Exit status is the test's own. An action that could not be placed on any
chip exits 1 with the reason on stderr, which is a genuine failure: we
could not tell whether the test passes.
"""

from __future__ import annotations

import os
import pathlib
import sys

import hashlib
import io
import tarfile

from ci.tools.relay_mailbox import config
from ci.tools.relay_mailbox import protocol
from ci.tools.relay_mailbox import worker as worker_lib

_FORWARDED_ENV_PREFIXES = (
    "TORCH_",
    "TPU_",
    "XLA_",
    "JAX_",
    "PJRT_",
    "LIBTPU_",
    "CUDA_",
    "IS_OSS",
)


def forwarded_env(env) -> dict[str, str]:
  """Extracts test environment variables that should be passed to the TPU."""
  fwd = {}
  for k, v in env.items():
    if any(k.startswith(p) for p in _FORWARDED_ENV_PREFIXES):
      if not k.startswith("TORCH_TPU_RELAY_"):
        fwd[k] = v
    elif k in ("TEST_SHARD_INDEX", "TEST_TOTAL_SHARDS"):
      fwd[k] = v
  return fwd

DEFAULT_TIMEOUT_SECONDS = 900.0


def layer_dirs(env) -> list[str]:
  """Returns the input layers to pack, in extraction order."""
  raw = env.get(config.LAYER_DIRS_ENV, "").strip()
  return [part for part in raw.split(":") if part]


def find_runfiles_dir(argv, env) -> pathlib.Path | None:
  """Locates the Bazel runfiles directory for the target under test."""
  test_srcdir = env.get("TEST_SRCDIR")
  if test_srcdir and pathlib.Path(test_srcdir).is_dir():
    return pathlib.Path(test_srcdir)
  runfiles_dir = env.get("RUNFILES_DIR")
  if runfiles_dir and pathlib.Path(runfiles_dir).is_dir():
    return pathlib.Path(runfiles_dir)
  if argv:
    cand = pathlib.Path(f"{argv[0]}.runfiles")
    if cand.is_dir():
      return cand
  return None


def pack_runfiles_layer(runfiles_dir: pathlib.Path) -> bytes:
  """Packs runfiles into a lightweight test layer blob.

  Excludes base cache components (rules_python repositories, _solib_x86_64,
  csrc) that are pre-staged on the TPU VMs.
  """
  buffer = io.BytesIO()
  with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.GNU_FORMAT) as tar:
    for root, dirs, files in os.walk(runfiles_dir):
      dirs[:] = [
          d for d in dirs
          if d not in ("_solib_x86_64", "__pycache__")
          and not d.startswith("rules_python")
      ]
      for name in list(dirs):
        p = os.path.join(root, name)
        if os.path.islink(p):
          arcname = os.path.relpath(p, runfiles_dir)
          target = os.readlink(p)
          if ".venv" in p and ("/external/" in target or target.startswith("../")):
            info = tar.gettarinfo(p, arcname=arcname)
            tar.addfile(info)
            dirs.remove(name)

      for f in files:
        if f.endswith((".pyc", ".a", ".o", ".params", ".cppmap")) or f in (
            "MANIFEST",
            "_repo_mapping",
        ):
          continue
        p = os.path.join(root, f)
        if p.endswith(".so") and ("torch_tpu/common" in p or "csrc/common" in p):
          continue
        arcname = os.path.relpath(p, runfiles_dir)
        if os.path.islink(p):
          target = os.readlink(p)
          if ".venv" in p and ("/external/" in target or target.startswith("../")):
            info = tar.gettarinfo(p, arcname=arcname)
            tar.addfile(info)
          elif os.path.exists(p):
            info = tar.gettarinfo(os.path.realpath(p), arcname=arcname)
            with open(p, "rb") as handle:
              tar.addfile(info, handle)
        elif os.path.isfile(p):
          info = tar.gettarinfo(p, arcname=arcname)
          with open(p, "rb") as handle:
            tar.addfile(info, handle)
  return buffer.getvalue()


def relative_argv(argv, env) -> list[str]:
  """Translates host execution argv into workspace-relative argv for the TPU."""
  if not argv:
    return []
  first = argv[0]
  if os.path.isabs(first):
    runfiles_dir = find_runfiles_dir(argv, env)
    if runfiles_dir:
      main_dir = runfiles_dir / "_main"
      if main_dir.is_dir() and first.startswith(str(main_dir) + "/"):
        first = os.path.relpath(first, main_dir)
      elif first.startswith(str(runfiles_dir) + "/"):
        first = os.path.relpath(first, runfiles_dir)
    if os.path.isabs(first):
      for marker in ("/bin/", "/execroot/_main/"):
        if marker in first:
          first = first.split(marker, 1)[1]
          break
  return [first] + list(argv[1:])


def timeout_seconds(env) -> float:
  """Returns the per-item timeout, falling back to the default."""
  raw = env.get(config.TIMEOUT_ENV, "").strip()
  if not raw:
    return DEFAULT_TIMEOUT_SECONDS
  try:
    return float(raw)
  except ValueError as error:
    raise config.ConfigError(
        f"{config.TIMEOUT_ENV} is not a number: {raw!r}"
    ) from error


def run(argv, env, client=None) -> int:
  """Dispatches one command and returns the status to exit with.

  Args:
    argv: The command to run on the chip.
    env: Environment mapping.
    client: Storage client, built from the environment when omitted.

  Returns:
    A process exit status.
  """
  client = client or config.bucket_client(env)
  worker = worker_lib.Worker(
      client,
      config.worker_id(env),
      deps=worker_lib.WorkerDeps(log_fn=config.log),
  )
  shard_status = env.get("TEST_SHARD_STATUS_FILE")
  if shard_status:
    try:
      pathlib.Path(shard_status).touch()
    except OSError:
      pass

  dirs = layer_dirs(env)
  if dirs:
    layers = worker.upload_layers(dirs)
  else:
    runfiles_dir = find_runfiles_dir(argv, env)
    if runfiles_dir:
      data = pack_runfiles_layer(runfiles_dir)
      digest = hashlib.sha256(data).hexdigest()
      path = protocol.blob_path(digest)
      if client.get(path) is None:
        client.put(path, data)
      layers = [path]
    else:
      layers = []

  request = worker_lib.Request(
      argv=relative_argv(argv, env),
      timeout_seconds=timeout_seconds(env),
      layers=layers,
      env=forwarded_env(env),
  )
  try:
    outcome = worker.dispatch(request)
    if not outcome.ok:
      config.log(f"could not run this test on any chip: {outcome.failure}")
      return 1
    _replay(client, outcome)
    return outcome.result.exit_code
  finally:
    if env.get("TORCH_TPU_RELAY_HOLD_BINDING", "0") != "1":
      worker.binder.release()


def _replay(client, outcome) -> None:
  """Prints what the test printed, so bazel's log looks normal."""
  for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
    if name not in outcome.result.outputs:
      continue
    blob = client.get(
        protocol.output_path(outcome.tpu, outcome.result.item_id, name)
    )
    if blob is not None:
      stream.write(blob.data.decode("utf-8", errors="replace"))
  sys.stdout.flush()
  sys.stderr.flush()


def main(argv=None, env=None) -> int:
  argv = sys.argv[1:] if argv is None else argv
  env = os.environ if env is None else env
  if not argv:
    config.log("usage: relay_worker <command> [args...]")
    return 2
  try:
    return run(argv, env)
  except config.ConfigError as error:
    config.log(str(error))
    return 2


if __name__ == "__main__":
  sys.exit(main())
