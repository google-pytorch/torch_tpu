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

import pathlib

# Install the version-dispatch import hook before any torch_tpu._internal C++
# extension is imported, so imports resolve to the glue built for the installed
# PyTorch. This is a no-op for an unversioned build (no versioned glues bundled).
from torch_tpu import _versioned_so_loader  # pylint: disable=g-import-not-at-top

_versioned_so_loader.install_hook(pathlib.Path(__file__).parent)

try:
  import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  libtpu.configure_library_path()
except ImportError:
  print("libtpu not found.")
  pass


def _configure_native_scan_gate() -> None:
  """Gates whether cumulative ops may emit the native scan emitter.

  Cumulative ops (cumsum/cumprod/cummax/cummin/logcumsumexp) lower to the native
  scan emitter (chlo.ScanOp), which needs a libtpu new enough to compile it.
  Internal builds always do; OSS pins a wheel that may not, so gate on the
  libtpu
  version and fall back to the while-loop lowering otherwise (b/529376045).

  This MUST run after libtpu.configure_library_path(): importing the
  _internal.env C++ extension runs static initializers that register the TPU
  PJRT plugin and require TPU_LIBRARY_PATH, which configure_library_path() sets.
  Importing env before that aborts with "TPU_LIBRARY_PATH is not set", so the
  import is kept local to this function rather than at module top.
  """
  from torch_tpu._internal import env  # pylint: disable=g-import-not-at-top
  from torch_tpu._internal import native_scan  # pylint: disable=g-import-not-at-top

  env.set_native_scan_emitter_supported(
      bool(env.IS_INTERNAL_TORCH_TPU)
      or native_scan.libtpu_supports_native_scan()
  )


_configure_native_scan_gate()

# TODO: b/477401982 - Introduce a new flag module to hold all TorchTPU flags.
