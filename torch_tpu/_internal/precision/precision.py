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
from torch_tpu._internal.precision import precision_impl

__all__ = ["precision", "Precision", "dot_algorithm"]


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


@contextlib.contextmanager
def precision(mode: Precision) -> Generator[None, None, None]:
  """Context manager to set op precisions.

  This provides a direct binding to the StableHLO precision configuration. It is
  primarily used for operations involving matrix multiplication (e.g.,
  matmul and convolution). On TPU, float32 matrix multiplications are natively
  executed in bfloat16. This context manager allows you to increase the
  precision of these operations, mapping directly to StableHLO precision
  settings.

  WARNING: This has different behavior compared to
  `torch.set_float32_matmul_precision` which has unpredictable precision
  behavior across varying GPUs. While StableHLO ops are generally stable, their
  lowering to device-specific ISA is not fully documented, and as a result
  precision may vary across different hardware.

  Args:
      mode: The precision to use. Must be a Precision enum. See the StableHLO
        documentation for details on the numerical behavior of each mode:
        https://openxla.org/stablehlo/spec#dot_general

  Yields:
      None
  """

  if mode not in _VALID_PRECISIONS:
    raise ValueError(
        "expected to be one of 'PrecisionConfig.DEFAULT',"
        f" 'PrecisionConfig.HIGH', or 'PrecisionConfig.HIGHEST', got '{mode}'"
    )

  # Push the new state.
  precision_impl._push_precision(mode)  # pylint: disable=protected-access
  try:
    yield
  finally:
    # Guarantee the state is restored even if Python code throws an exception.
    precision_impl._pop_precision()  # pylint: disable=protected-access


@contextlib.contextmanager
def dot_algorithm(
    accumulation_type: torch.dtype,
    lhs_component_count: int = 1,
    rhs_component_count: int = 1,
    num_primitive_operations: int = 1,
    allow_imprecise_accumulation: bool = False,
) -> Generator[None, None, None]:
  """Context manager to set dot algorithm configuration.

  This provides a way to configure the DotAlgorithm for dot_general operations.

  Args:
      accumulation_type: The accumulation type.
      lhs_component_count: The LHS component count.
      rhs_component_count: The RHS component count.
      num_primitive_operations: The number of primitive operations.
      allow_imprecise_accumulation: Whether to allow imprecise accumulation.

  Yields:
      None
  """
  config = precision_impl.DotAlgorithmConfig()
  config.accumulation_type = accumulation_type
  config.lhs_component_count = lhs_component_count
  config.rhs_component_count = rhs_component_count
  config.num_primitive_operations = num_primitive_operations
  config.allow_imprecise_accumulation = allow_imprecise_accumulation

  precision_impl._push_dot_algorithm(config)  # pylint: disable=protected-access
  try:
    yield
  finally:
    precision_impl._pop_dot_algorithm()  # pylint: disable=protected-access
