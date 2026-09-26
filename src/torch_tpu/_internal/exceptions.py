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

"""Internal exception and warning types raised or emitted by TorchTPU.

Every warning TorchTPU emits derives from `TorchTPUWarning` (a subclass of
`UserWarning`). Because these warning classes are internal, users should filter
warnings by message or by `UserWarning`:

  # Silence the tcmalloc warning.
  export PYTHONWARNINGS="ignore:tcmalloc is not loaded"

  # Silence every warning TorchTPU emits while importing.
  export PYTHONWARNINGS="ignore::UserWarning"

Warnings emitted while `import torch_tpu` is still running -- `TCMallocWarning`
is one -- fire before user code can install a `warnings.filterwarnings` rule by
class anyway, and resolving a package-qualified category name in
`PYTHONWARNINGS` would import `torch_tpu` (or fail before `site-packages` is on
`sys.path`) before the filter is installed.
"""

class TorchTPUWarning(UserWarning):
  """Base class for all warnings emitted by TorchTPU.

  Derives from `UserWarning` rather than `RuntimeWarning` because `UserWarning`
  is the category `torch` itself surfaces `TORCH_WARN` as, and because numeric
  stacks routinely install a blanket `ignore::RuntimeWarning` filter to hide
  NumPy overflow and invalid-value warnings, which would also swallow these.
  """


class TCMallocWarning(TorchTPUWarning):
  """Warns that tcmalloc is not the allocator of the current process.

  TorchTPU host performance degrades measurably under the default glibc
  ptmalloc2 allocator because eager dispatch allocates and frees C++ runtime
  objects across threads, contending on ptmalloc2 arena locks.

  Emitted at most once per process by default; see
  `torch_tpu._tcmalloc_check.install_default_filters`.
  """
