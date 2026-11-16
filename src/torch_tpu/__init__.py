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

from torch_tpu import _tcmalloc_check
from torch_tpu import _versioned_so_loader

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

from torch_tpu import _loader as _  # pylint: disable=g-import-not-at-top  # noqa: F401

# Warn if tcmalloc is not the allocator of this process. Checked last so the
# probe sees every library `import torch_tpu` pulls in, libtpu included, and so
# `torch_tpu._internal.env` is ready for TorchTPU ancestor process detection.
_tcmalloc_check.check_tcmalloc()

# TODO: b/477401982 - Introduce a new flag module to hold all TorchTPU flags.
