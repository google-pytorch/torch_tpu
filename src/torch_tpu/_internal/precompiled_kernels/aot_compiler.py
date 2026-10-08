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

r"""Ahead-of-time compiles TorchTPU's precompiled kernels into a persistent cache.

Compiles the precompiled kernels (or loads them if they are already cached) and
stores them in the tier-3 compilation cache rooted at `--cache_dir`. TorchTPU
processes that are later started with the same tier-3 cache root pick the
kernels up from there instead of compiling them.

Usage:
  python -m torch_tpu._internal.precompiled_kernels.aot_compiler
  \ --cache_dir=/path/to/cache
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from torch_tpu._internal import env_utils
from torch_tpu._internal.precompiled_kernels import _tpu_torch_precompiled_kernels

_CACHE_DIR = flags.DEFINE_string(  # PYTHON_FLAG_OK=This is a standalone script.
    "cache_dir",
    None,
    "Directory to use as the tier-3 compilation cache root. The precompiled"
    " kernels are loaded from it if present and written to it otherwise. May be"
    " a local path or a remote filesystem path supported by the tier-3 cache"
    " (e.g. gs://...).",
)

# Must match kTorchTpuTier3CompilationCacheRootEnvVar in csrc/common/env_vars.h.
_TIER3_CACHE_ROOT_ENV_VAR = "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"


def load_kernels(cache_dir: str | os.PathLike[str]) -> None:
  """Loads the precompiled kernels into the compilation cache.

  `cache_dir` is used as the root of the tier-3 (persistent) compilation cache:
  kernels already present there are loaded instead of compiled, and kernels
  that have to be compiled are written there for future runs. Enabling the
  tier-3 cache also enables the tier-2 (shared memory) cache, through which
  tier-3 entries are served.

  The compilation cache reads its configuration once, on first use, so this
  must be called before anything is compiled in the process. Otherwise
  `cache_dir` is ignored and the kernels are only loaded into the in-memory
  cache.

  This call blocks until the kernels have been compiled or loaded from the
  persistent compilation cache. It is idempotent: repeat calls are no-ops.

  Args:
    cache_dir: The directory to use as the tier-3 compilation cache root.
  """
  env_utils.set_str_env(_TIER3_CACHE_ROOT_ENV_VAR, os.fspath(cache_dir))
  _tpu_torch_precompiled_kernels.load_kernels()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")
  if not _CACHE_DIR.value:
    raise app.UsageError("--cache_dir is required.")
  logging.info("Loading precompiled kernels into %s", _CACHE_DIR.value)
  load_kernels(_CACHE_DIR.value)
  logging.info("Done.")


if __name__ == "__main__":
  app.run(main)
