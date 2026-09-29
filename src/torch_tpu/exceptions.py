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

"""Exception and warning types raised or emitted by TorchTPU.

Every warning TorchTPU emits derives from `TorchTpuWarning`, so warnings can be
silenced, escalated, or filtered individually or as a group once TorchTPU has
been imported:

  import warnings

  import torch_tpu
  from torch_tpu import exceptions

  warnings.filterwarnings("ignore", category=exceptions.TcMallocWarning)

Warnings emitted while `import torch_tpu` is still running -- `TcMallocWarning`
is one -- have already fired by the time such a filter can be installed, so
those must be filtered by message from the environment instead:

  # Silence the tcmalloc warning.
  export PYTHONWARNINGS="ignore:tcmalloc is not loaded"

  # Silence every warning TorchTPU emits while importing.
  export PYTHONWARNINGS="ignore::UserWarning"

Note that `PYTHONWARNINGS="ignore::torch_tpu.exceptions.TcMallocWarning"` does
*not* suppress an import-time warning, even though it looks like it should:
resolving that category name makes the interpreter import `torch_tpu` in order
to look the class up, which emits the warning before the filter is installed.
On a `pip install`ed TorchTPU the name cannot be resolved at that point at all,
because site-packages is not yet on `sys.path`, and the interpreter reports
`Invalid -W option ignored: invalid module name: 'torch_tpu.exceptions'` on
every subsequent Python invocation in that environment.
"""


class TorchTpuWarning(UserWarning):
  """Base class for all warnings emitted by TorchTPU.

  Derives from `UserWarning` rather than `RuntimeWarning` because `UserWarning`
  is the category `torch` itself surfaces `TORCH_WARN` as, and because numeric
  stacks routinely install a blanket `ignore::RuntimeWarning` filter to hide
  NumPy overflow and invalid-value warnings, which would also swallow these.
  """


class PerformanceWarning(TorchTpuWarning):
  """Warns about potential performance issues."""


class CorrectnessWarning(TorchTpuWarning):
  """Warns about potential correctness issues."""


class ApiStageWarning(TorchTpuWarning):
  """Warns about APIs that are not in the STABLE stage."""


class ExperimentalApiWarning(ApiStageWarning):
  """Warns about APIs that are in the EXPERIMENTAL stage."""


class DeprecatedApiWarning(ApiStageWarning, DeprecationWarning):
  """Warns about APIs that are in the DEPRECATED stage."""


class UnoptimizedBuildWarning(PerformanceWarning):
  """Warns that TorchTPU was compiled in an unoptimized mode."""


class TcMallocWarning(PerformanceWarning):
  """Warns that tcmalloc is not the allocator of the current process.

  TorchTPU host performance degrades measurably under the default glibc
  ptmalloc2 allocator because eager dispatch allocates and frees C++ runtime
  objects across threads, contending on ptmalloc2 arena locks.

  Emitted at most once per process by default; see
  `torch_tpu._tcmalloc_check.install_default_filters`.
  """
