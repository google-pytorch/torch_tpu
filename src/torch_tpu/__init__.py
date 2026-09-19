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
import os
import pathlib
import re
import sys
import warnings

_TCMALLOC_OR_SANITIZER_RE = re.compile(
    r"\blib(?:tcmalloc|clang_rt\.[at]san|[at]san)\b"
)
_tcmalloc_warning_emitted = False


def _check_tcmalloc() -> None:
  """Emits a one-time warning on Linux if tcmalloc is not loaded.

  In TorchTPU eager execution, the Python dispatch thread and background runtime
  worker threads allocate and free C++ runtime objects across threads, causing
  arena lock contention under default glibc ptmalloc2. In Bazel tests/benchmarks
  and the standard TorchTPU Docker image, tcmalloc is linked or preloaded
  automatically. On bare-metal environments where tcmalloc is not active, this
  function emits a one-time warning.
  """
  global _tcmalloc_warning_emitted
  if _tcmalloc_warning_emitted or sys.platform != "linux":
    return
  env = os.environ  # PY_ENVIRON_OK=pre-pywrap
  if (
      env.get("TORCH_TPU_DISABLE_TCMALLOC", "0") == "1"
      or env.get("TORCH_TPU_DISABLE_TCMALLOC_CHECK", "0") == "1"
  ):
    return

  # Check if tcmalloc or sanitizer symbols are already present in the global
  # symbol table (e.g., statically linked in G3 builds or preloaded via
  # LD_PRELOAD). Passing None to ctypes.CDLL queries RTLD_DEFAULT without
  # loading any file.
  try:
    global_syms = ctypes.CDLL(None)
    if (
        hasattr(global_syms, "MallocExtension_GetAllocatedSize")
        or hasattr(global_syms, "tc_version")
        or hasattr(global_syms, "__asan_init")
        or hasattr(global_syms, "__tsan_init")
    ):
      return
  except OSError:
    pass

  # Check if tcmalloc or a sanitizer (ASan/TSan) is mapped in /proc/self/maps.
  try:
    with open("/proc/self/maps", "r", encoding="utf-8") as f:
      if _TCMALLOC_OR_SANITIZER_RE.search(f.read()):
        return
  except OSError:
    return

  _tcmalloc_warning_emitted = True
  warnings.warn(
      "tcmalloc is not loaded; TorchTPU host dispatch performance may be "
      "degraded due to glibc ptmalloc contention. Preload libtcmalloc before "
      "starting Python (e.g. LD_PRELOAD=/usr/lib/x86_64-linux-gnu/"
      "libtcmalloc_minimal.so.4) or set TORCH_TPU_DISABLE_TCMALLOC_CHECK=1 "
      "to suppress this warning.",
      RuntimeWarning,
      stacklevel=2,
  )


_check_tcmalloc()

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
