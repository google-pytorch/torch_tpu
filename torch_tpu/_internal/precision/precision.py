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

"""Op precision context manager."""

import contextlib
from typing import Generator

import torch
import torch.fx.traceback as fx_traceback
from torch_tpu._internal.precision import precision_impl

__all__ = ["precision", "Precision"]


# This enum provides a direct binding to the StableHLO precision configuration.
# For explicit definitions of 'DEFAULT', 'HIGH', and 'HIGHEST', please see the
# StableHLO documentation for 'dot_general':
# https://openxla.org/stablehlo/spec#dot_general
Precision = precision_impl.Precision

_VALID_PRECISIONS = (
    Precision.DEFAULT,
    Precision.HIGH,
    Precision.HIGHEST,
)

# Namespaced key under `node.meta["custom"]`. "custom" is a dict shared with
# PyTorch itself (it also stores keys such as "stream" and
# "nested_region_config"),
# so the key is prefixed to avoid collisions. This value is the contract between
# the producer (the compile branch below) and the consumer
# (`precision_from_node_meta`, invoked by the TPU lowering interpreter); keep it
# stable.
TPU_PRECISION_META_KEY = "torch_tpu::precision"


@contextlib.contextmanager
def precision(mode: Precision) -> Generator[None, None, None]:
  """Context manager to set op precisions.

  This provides a direct binding to the StableHLO precision configuration. It is
  primarily used for operations involving matrix multiplication (e.g.,
  matmul and convolution). On TPU, float32 matrix multiplications are natively
  executed in bfloat16. This context manager allows you to increase the
  precision of these operations, mapping directly to StableHLO precision
  settings.

  This setting overrides `torch.set_float32_matmul_precision`, the global knob
  for controlling matmul precision in PyTorch, with a local context. While
  StableHLO ops are generally stable, their lowering to device-specific ISA is
  not fully documented, and as a result, precision may vary across different
  hardwares.

  Args:
      mode: The precision to use. Must be a Precision enum. See the StableHLO
        documentation for details on the numerical behavior of each mode:
        https://openxla.org/stablehlo/spec#dot_general

  Yields:
      None
  """

  if not isinstance(mode, Precision) or mode not in _VALID_PRECISIONS:
    raise ValueError(
        "expected to be one of 'PrecisionConfig.DEFAULT',"
        f" 'PrecisionConfig.HIGH', or 'PrecisionConfig.HIGHEST', got '{mode}'"
    )

  # Under `torch.compile`/`torch.export`, the opaque pybind11 push/pop below is
  # untraceable and would abandon the whole frame to eager (a graph break).
  # Dynamo constant-folds `torch.compiler.is_compiling()` to True while tracing,
  # so the eager branch becomes dead code that is never traced. Instead, stamp
  # the region's precision onto `node.meta["custom"]` of every op created inside
  # the region via the allow-listed `torch.fx.traceback.annotate`; the TPU
  # lowering pass reads it back through `precision_from_node_meta`. In eager,
  # `is_compiling()` is False and the C++ TLS push/pop  runs exactly as before.
  if torch.compiler.is_compiling():
    with fx_traceback.annotate({TPU_PRECISION_META_KEY: mode.value}):
      yield
    return

  # Push the new state.
  precision_impl._push_precision(mode)  # pylint: disable=protected-access
  try:
    yield
  finally:
    # Guarantee the state is restored even if Python code throws an exception.
    precision_impl._pop_precision()  # pylint: disable=protected-access


@contextlib.contextmanager
def precision_from_node_meta(
    node: torch.fx.Node,
) -> Generator[None, None, None]:
  """Re-establishes the precision state annotated on an FX node during lowering.

  Ops created inside a `with precision(...)` region under `torch.compile` or
  `torch.export` carry the region's precision under
  `node.meta["custom"][TPU_PRECISION_META_KEY]` (see `precision`). The TPU
  lowering interpreter wraps each node's dispatch in this context so that the
  precision-sensitive kernels (such as matmul and convolution) observe the same
  thread-local precision they would in eager. Nodes without an annotation fall
  through to the ambient default, preserving today's behavior for code outside
  any region.

  Args:
      node: The FX node about to be lowered.

  Yields:
      None
  """
  custom = node.meta.get("custom")
  mode = custom.get(TPU_PRECISION_META_KEY) if custom else None
  if mode is None:
    yield
    return

  precision_impl._push_precision(Precision(mode))  # pylint: disable=protected-access
  try:
    yield
  finally:
    precision_impl._pop_precision()  # pylint: disable=protected-access
