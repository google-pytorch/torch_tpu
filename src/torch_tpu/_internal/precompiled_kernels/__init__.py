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

"""Precompiled kernels API, exposed as `torch.tpu.experimental.precompiled_kernels`."""

from torch_tpu._internal.precompiled_kernels import _tpu_torch_precompiled_kernels
from torch_tpu._internal.utils import annotations


@annotations.experimental(
    "torch.tpu.experimental.precompiled_kernels.load_kernels is experimental"
    " and subject to change."
)
def load_kernels() -> None:
  """Loads the precompiled kernels into the compilation cache.

  Once loaded, subsequent `torch.nonzero` calls on inputs with
  supported shapes and dtypes reuse the kernel instead of triggering a
  per-shape compilation.

  This call blocks until the kernels have been compiled or loaded from the
  persistent compilation cache. It is idempotent: repeat calls are no-ops.
  """
  _tpu_torch_precompiled_kernels.load_kernels()


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "load_kernels",
    # go/keep-sorted end
]
