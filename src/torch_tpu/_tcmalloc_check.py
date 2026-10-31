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

Importing this module installs TorchTPU's default filter for the tcmalloc
warning as a side effect; see `install_default_filters`. The check itself only
runs when `check_tcmalloc` is called, which `torch_tpu/__init__.py` does once at
import time after `torch_tpu._loader` has been imported.

Module-level code in this module must not import `torch`, not even indirectly,
because `torch_tpu/__init__.py` imports `_tcmalloc_check` before installing the
version-dispatch import hook of `_versioned_so_loader`. `check_tcmalloc` itself
runs after `torch_tpu._loader`, so it can safely import
`torch_tpu._internal.env` to skip the warning in processes with a TorchTPU
ancestor process.
The warning
below uses `warnings.warn` instead of `tt_warnings.warn` because the check only
runs at import time, never under `torch.compile`, and hard-codes the warning
category prefix so `install_default_filters` can run at module import time
before TorchTPU's C++ extensions are loaded.
"""

import ctypes
import re
from typing import Final
import warnings

# Sanitizers (ASan, MSan, TSan) replace the allocator, so tcmalloc cannot be
# active under them.
_TCMALLOC_OR_SANITIZER_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    \blib
    (?:
        # libtcmalloc.so.4, libtcmalloc_minimal.so.4
        tcmalloc
        # libclang_rt.asan-x86_64.so
      | clang_rt\.(?:asan|msan|tsan)
        # libasan.so.8, libmsan.so.0, libtsan.so.0
      | (?:asan|msan|tsan)
    )
    # Intentionally not have a trailing `\b`, otherwise it would not match
    # `libtcmalloc_minimal` because `_` is a word character.
    [._-]
    """,
    re.VERBOSE,
)


def _has_sanitizer_init(global_syms: ctypes.CDLL) -> bool:
  """Returns whether a sanitizer runtime's init symbol is in `global_syms`.

  The symbol is present when ASan, MSan, or TSan is statically linked or
  preloaded.

  Args:
    global_syms: The process' global symbol table.
  """
  return (
      hasattr(global_syms, "__asan_init")
      or hasattr(global_syms, "__msan_init")
      or hasattr(global_syms, "__tsan_init")
  )


# The prefix of `torch_tpu._internal.warnings.TCMALLOC` warnings; see the module
# docstring for why it's hard-coded. Tests check that it matches the C++ value.
TCMALLOC_WARNING_PREFIX: Final[str] = "<TorchTpu/Performance/TcMalloc/> "


def install_default_filters() -> None:
  """Installs TorchTPU's default warning filter for the tcmalloc warning.

  The warning is emitted at most once per process. `append=True` keeps this
  filter behind any user-supplied filter, so PYTHONWARNINGS, `python -W`, and
  `warnings.filterwarnings` still take precedence.
  """
  warnings.filterwarnings(
      "once",
      message=re.escape(TCMALLOC_WARNING_PREFIX),
      category=UserWarning,
      append=True,
  )


def check_tcmalloc() -> None:
  """Warns if tcmalloc is not the allocator of the current process.

  The check is skipped in processes that have a TorchTPU ancestor process (see
  `env.has_torch_tpu_ancestor_process`) to avoid spamming the log in
  multi-process programs.

  In TorchTPU eager execution, the Python dispatch thread and background runtime
  worker threads allocate and free C++ runtime objects across threads, causing
  arena lock contention under the default glibc ptmalloc2. In Bazel
  tests/benchmarks and the standard TorchTPU Docker image, tcmalloc is linked or
  preloaded automatically. On bare-metal environments where tcmalloc is not
  active, this emits a warning in the `TCMALLOC` TorchTPU warning category.

  Detection uses the process' global symbol table and, where available,
  `/proc/self/maps`. Neither probe is assumed to succeed: if a platform offers
  no way to tell, the check stays quiet rather than warn spuriously.
  """
  from torch_tpu._internal import env  # pylint: disable=g-import-not-at-top

  if env.has_torch_tpu_ancestor_process():
    return

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
        or _has_sanitizer_init(global_syms)
    ):
      return
  except OSError:
    pass

  # Check if tcmalloc or a sanitizer is mapped in /proc/self/maps.
  try:
    with open("/proc/self/maps", "r", encoding="utf-8") as f:
      if _TCMALLOC_OR_SANITIZER_RE.search(f.read()):
        return
  except OSError:
    # Without /proc/self/maps there is no way to tell, so stay quiet rather
    # than warn spuriously.
    return

  # See the module docstring for why this doesn't use `tt_warnings.warn`.
  warnings.warn(  # RAW_WARNINGS_OK=import time only, no torch import
      f"{TCMALLOC_WARNING_PREFIX}tcmalloc is not loaded; TorchTPU host"
      " performance may be degraded due to glibc ptmalloc contention. Preload"
      " libtcmalloc before starting Python (e.g."
      " LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4), or set"
      ' PYTHONWARNINGS="ignore:<TorchTpu/Performance/TcMalloc/" to suppress'
      " this warning.",
      UserWarning,
      stacklevel=2,
  )


install_default_filters()
