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

"""`torch.compile`-friendly replacement for `warnings.warn`.

TorchDynamo cannot trace the builtin `warnings.warn`, so every call to it from
code that runs inside a `torch.compile` region causes a graph break (or a hard
error under `fullgraph=True`). TorchTPU library code should therefore emit
Python warnings through `warn` below instead of calling `warnings.warn`
directly.
"""

from typing import Any
import warnings

import torch
from torch_tpu._internal import exceptions


def warn(
    message: str | Warning,
    category: type[Warning] = exceptions.TorchTpuWarning,
    stacklevel: int = 1,
    **kwargs: Any,
) -> bool:
  """Emits a Python warning, except while TorchDynamo is tracing.

  Behaves like `warnings.warn`, but returns immediately when
  `torch.compiler.is_compiling()` is true. Dynamo constant-folds that check, so
  the call traces away entirely and never causes a graph break. The trade-off is
  that warnings are only emitted during eager execution: a warning raised from
  code that only ever runs inside a compiled region is never shown.

  This matches what PyTorch core does for its own warnings, which it guards with
  `if not torch.compiler.is_compiling():` (e.g. the deprecation warnings in
  `torch.distributed._functional_collectives` and `_warn_once` in
  `torch.nn.attention.flex_attention`).

  Args:
    message: The warning message, or a `Warning` instance.
    category: The warning category. Ignored if `message` is a `Warning`.
    stacklevel: Same meaning as for `warnings.warn`, relative to the caller of
      this function: `1` attributes the warning to the line calling `warn`, `2`
      to that function's caller, and so on.
    **kwargs: Further keyword arguments forwarded to `warnings.warn`, e.g.
      `skip_file_prefixes` (Python 3.12+).

  Returns:
    Whether the warning was issued, i.e. False if it was skipped because
    TorchDynamo is tracing. The active warning filters may still suppress an
    issued warning.
  """
  if torch.compiler.is_compiling():
    return False
  # +1 so that `stacklevel` is relative to our caller rather than to this frame.
  warnings.warn(  # RAW_WARNINGS_OK=This is the wrapper.
      message, category, stacklevel=stacklevel + 1, **kwargs
  )
  return True
