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

"""The TorchTPU version, which `torch.tpu.__version__` returns."""

import functools
import importlib.metadata

from torch import torch_version


@functools.cache
def get_version() -> torch_version.TorchVersion:
  """Returns the version of the installed `torch_tpu` distribution.

  It follows PEP 440 and torch's conventions, e.g. "0.2.1" (release),
  "0.2.1rc1" (release candidate) or "0.2.1.dev20261014" (nightly). Like
  `torch.__version__`, it's a `TorchVersion`, so it also compares with version
  strings, `packaging.version.Version`s and tuples, e.g.
  `get_version() >= (0, 2)`.

  The version is read on the first call rather than on import, so that importing
  TorchTPU doesn't need the distribution metadata.
  """
  return torch_version.TorchVersion(importlib.metadata.version("torch_tpu"))
