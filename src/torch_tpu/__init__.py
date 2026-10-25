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

import importlib.metadata
import pathlib

from torch_tpu import _tcmalloc_check
from torch_tpu import _versioned_so_loader


def _get_version(distribution: str = "torch_tpu") -> str:
  """Returns the version of `distribution` from its installed metadata.

  Wheels and Bazel builds both generate this metadata from the package version,
  so the version number isn't duplicated here.

  Args:
    distribution: The distribution to look up.

  Returns:
    The version, or "0.0.0+unknown" if no metadata for `distribution` is on
    `sys.path`. That fallback is a valid PEP 440 version that sorts before every
    release, so callers that parse the version keep working.
  """
  try:
    return importlib.metadata.version(distribution)
  except importlib.metadata.PackageNotFoundError:
    return "0.0.0+unknown"


__version__: str = _get_version()

# Install the version-dispatch import hook before any torch_tpu._internal C++
# extension is imported, so imports resolve to the glue built for the installed
# PyTorch. This is a no-op for an unversioned build, which has no bundled glue.
_versioned_so_loader.install_hook(pathlib.Path(__file__).parent)

try:
  import libtpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  libtpu.configure_library_path()
except ImportError:
  print("libtpu not found.")
  pass

# Warn if tcmalloc is not the allocator of this process. Checked last so the
# probe sees every library `import torch_tpu` pulls in, libtpu included.
_tcmalloc_check.check_tcmalloc()

from torch_tpu import _loader as _  # pylint: disable=g-import-not-at-top  # noqa: F401

# TODO: b/477401982 - Introduce a new flag module to hold all TorchTPU flags.
