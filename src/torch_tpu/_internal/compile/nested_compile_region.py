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

"""`torch.compiler.nested_compile_region` with TorchTPU-specific options."""

from collections.abc import Callable
import functools
from typing import Any

import torch
from torch.fx import traceback as fx_traceback
from torch_tpu._internal.compile.fx_passes import invoke_subgraph_to_scan


def nested_compile_region(
    fn: Callable[..., Any] | None = None,
    *,
    scan_unroll: int | None = None,
    **kwargs: Any,
) -> Any:
  """Marks `fn` as a nested compile region, like the PyTorch decorator.

  Can be used as `@nested_compile_region` or
  `@nested_compile_region(scan_unroll=k)`.

  Args:
    fn: The function or method to wrap.
    scan_unroll: If set, TorchTPU rolls a chain of consecutive calls of the
      region (e.g. a stack of decoder layers) into a loop running `scan_unroll`
      calls per iteration, so that XLA compiles the region once instead of once
      per call. None leaves the calls unrolled.
    **kwargs: Passed to `torch.compiler.nested_compile_region`.

  Returns:
    The wrapped `fn`, or a decorator if `fn` is None.
  """
  if scan_unroll is not None and scan_unroll < 1:
    raise ValueError(f"scan_unroll must be at least 1, got {scan_unroll}")

  def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
    region = torch.compiler.nested_compile_region(f, **kwargs)
    if scan_unroll is None:
      return region
    annotation = {invoke_subgraph_to_scan.SCAN_UNROLL_ANNOTATION: scan_unroll}

    @functools.wraps(f)
    def call(*args: Any, **call_kwargs: Any) -> Any:
      # Annotates the `invoke_subgraph` call that the region becomes.
      with fx_traceback.annotate(annotation):
        return region(*args, **call_kwargs)

    return call

  return wrap if fn is None else wrap(fn)
