# Copyright 2025 Google LLC
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

"""TorchTPU library."""

import ctypes
import ctypes.util
import multiprocessing
import os
import pathlib
import sys


def _ensure_tcmalloc() -> None:
  """Ensures tcmalloc is preloaded on Linux/GCP regardless of container image.

  On multi-socket Cloud TPU VMs (e.g., 224-vCPU dual-socket NUMA hosts),
  default glibc ptmalloc2 suffers severe arena lock contention across threads.
  If tcmalloc is not already linked (as in G3) or preloaded via LD_PRELOAD,
  this function locates bundled or system libtcmalloc, exports LD_PRELOAD for
  child worker processes, and re-executes the main Python process if safe.
  """
  if sys.platform != "linux":
    return
  if os.environ.get("TORCH_TPU_DISABLE_TCMALLOC", "0") == "1":
    return

  # Check if tcmalloc is already statically linked (e.g., G3 internal builds).
  try:
    if hasattr(ctypes.CDLL(None), "tc_version"):
      return
  except OSError:
    pass

  # Check if tcmalloc or a sanitizer (ASan/TSan) is already mapped in-process.
  try:
    with open("/proc/self/maps", "r", encoding="utf-8") as f:
      maps = f.read()
      if "tcmalloc" in maps or "tsan" in maps or "asan" in maps:
        return
  except OSError:
    pass

  pkg_dir = pathlib.Path(__file__).parent
  candidates = [
      pkg_dir / "lib" / "libtcmalloc_minimal.so.4",
      pkg_dir / "lib" / "libtcmalloc.so.4",
      pathlib.Path("/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"),
      pathlib.Path("/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"),
      pathlib.Path("/usr/lib64/libtcmalloc_minimal.so.4"),
      pathlib.Path("/usr/lib64/libtcmalloc.so.4"),
  ]
  for lib_name in ("tcmalloc_minimal", "tcmalloc"):
    found = ctypes.util.find_library(lib_name)
    if found:
      candidates.append(pathlib.Path(found))

  tcmalloc_path = None
  for candidate in candidates:
    if candidate.is_file():
      try:
        ctypes.CDLL(str(candidate))
        tcmalloc_path = str(candidate.resolve())
        break
      except OSError:
        continue

  if not tcmalloc_path:
    return

  # Always prepend to LD_PRELOAD so spawned child processes (torchrun,
  # multiprocessing workers) inherit tcmalloc from startup.
  current_preload = os.environ.get("LD_PRELOAD", "")
  if tcmalloc_path not in current_preload:
    os.environ["LD_PRELOAD"] = (
        f"{tcmalloc_path}:{current_preload}"
        if current_preload
        else tcmalloc_path
    )

  # Re-execute the main Python script with LD_PRELOAD active if running
  # standalone outside Bazel/internal test runners or interactive REPLs.
  if (
      os.environ.get("_TORCH_TPU_TCMALLOC_REEXEC") != "1"
      and os.environ.get("TORCH_TPU_TCMALLOC_REEXEC", "1") == "1"
      and "TEST_TMPDIR" not in os.environ
      and not hasattr(sys, "ps1")
      and sys.argv
      and sys.argv[0]
      and sys.argv[0] != "-c"
      and multiprocessing.current_process().name == "MainProcess"
  ):
    os.environ["_TORCH_TPU_TCMALLOC_REEXEC"] = "1"
    try:
      os.execv(sys.executable, [sys.executable] + sys.argv)
    except OSError:
      pass


_ensure_tcmalloc()

# Install the version-dispatch import hook before any torch_tpu._internal C++
# extension is imported, so imports resolve to the glue built for the installed
# PyTorch. This is a no-op for an unversioned build, which has no bundled glue.
from torch_tpu import _versioned_so_loader

_versioned_so_loader.install_hook(pathlib.Path(__file__).parent)

try:
  import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  libtpu.configure_library_path()
except ImportError:
  print("libtpu not found.")
  pass


# TODO: b/477401982 - Introduce a new flag module to hold all TorchTPU flags.
