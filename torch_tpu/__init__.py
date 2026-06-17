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
import re

try:
  _torch_version = importlib.metadata.version("torch")
  _match = re.match(r"^(\d+)\.(\d+)\.(\d+)", _torch_version)
  if _match:
    _version_suffix = f"{_match.group(1)}_{_match.group(2)}_{_match.group(3)}"
  else:
    _version_suffix = _torch_version.replace(".", "_")
except Exception:  # pylint: disable=broad-exception-caught
  _version_suffix = ""

from torch_tpu import _versioned_so_loader  # pylint: disable=g-import-not-at-top

_versioned_so_loader.install_hook(_version_suffix)

try:
  import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  libtpu.configure_library_path()
except ImportError:
  print("libtpu not found.")
  pass

# TODO: b/477401982 - Introduce a new flag module to hold all TorchTPU flags.
