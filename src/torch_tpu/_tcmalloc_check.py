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

"""Startup check that warns when tcmalloc is not the process allocator.

Importing this module installs TorchTPU's default filter for
`torch_tpu.exceptions.TCMallocWarning` as a side effect; see
`install_default_filters`. The check itself only runs when `check_tcmalloc` is
called, which `torch_tpu/__init__.py` does once at import time.
"""

import ctypes
import re
from typing import Final
import warnings

from torch_tpu import exceptions

# Matches e.g. libtcmalloc.so.4, libtcmalloc_minimal.so.4, libtsan.so.0 and
# libclang_rt.asan-x86_64.so. The trailing separator class is deliberate: a
# trailing `\b` would not match `libtcmalloc_minimal` because `_` is a word
# character.
_TCMALLOC_OR_SANITIZER_RE: Final[re.Pattern[str]] = re.compile(
    r"\blib(?:tcmalloc|clang_rt\.[at]san|[at]san)[._-]"
)


def install_default_filters() -> None:
  """Installs TorchTPU's default warning filter for `TCMallocWarning`.

  The warning is emitted at most once per process. `append=True` keeps this
  filter behind any user-supplied filter, so PYTHONWARNINGS, `python -W`, and
  `warnings.filterwarnings` still take precedence.
  """
  warnings.filterwarnings(
      "once", category=exceptions.TCMallocWarning, append=True
  )


def check_tcmalloc() -> None:
  """Warns if tcmalloc is not the allocator of the current process.

  In TorchTPU eager execution, the Python dispatch thread and background runtime
  worker threads allocate and free C++ runtime objects across threads, causing
  arena lock contention under the default glibc ptmalloc2. In Bazel
  tests/benchmarks and the standard TorchTPU Docker image, tcmalloc is linked or
  preloaded automatically. On bare-metal environments where tcmalloc is not
  active, this emits a `TCMallocWarning`.

  Detection uses the process' global symbol table and, where available,
  `/proc/self/maps`. Neither probe is assumed to succeed: if a platform offers
  no way to tell, the check stays quiet rather than warn spuriously.
  """
  # Check if tcmalloc or sanitizer symbols are already present in the global
  # symbol table (e.g., statically linked in G3 builds or preloaded via
  # LD_PRELOAD). Passing None to ctypes.CDLL queries RTLD_DEFAULT without
  # loading any file.
  try:
    global_syms = ctypes.CDLL(None)
    if (
        hasattr(global_syms, "MallocExtension_Internal_GetAllocatedSize")
        or hasattr(global_syms, "MallocExtension_GetAllocatedSize")
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
    # Without /proc/self/maps there is no way to tell, so stay quiet rather
    # than warn spuriously.
    return

  warnings.warn(
      "tcmalloc is not loaded; TorchTPU host performance may be degraded due"
      " to glibc ptmalloc contention. Preload libtcmalloc before starting"
      " Python (e.g."
      " LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4), or set"
      ' PYTHONWARNINGS="ignore:tcmalloc is not loaded" to suppress this'
      " warning.",
      exceptions.TCMallocWarning,
      stacklevel=2,
  )


install_default_filters()
