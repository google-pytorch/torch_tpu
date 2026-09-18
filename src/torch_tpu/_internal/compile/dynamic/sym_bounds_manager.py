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

"""Symbolic bounds manager for dynamic shape compilation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, NamedTuple

from absl import logging
import torch
from torch._inductor.utils import InputType
from torch.fx.experimental import symbolic_shapes
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.dynamic import symbol_bounds

ConstraintViolationError = symbolic_shapes.ConstraintViolationError
get_symint_bounds = symbol_bounds.get_symint_bounds


class ShapeBoundInfo(NamedTuple):
  """Shape bound information for dynamic dimensions of a tensor."""

  dynamic_dims: tuple[int, ...]
  upper_bounds: tuple[int, ...]


@dataclasses.dataclass
class InputTensorBoundsMetadata:
  """Bounds metadata for an input tensor."""

  static_shape: list[int]
  dynamic_bounds: list[tuple[int, int]]
  dynamic_dims: list[int]


class SymBoundsManager:
  """Manages lower and upper bounds for dynamic SymInts and generates bounded inputs."""

  input_tensors_metadata: dict[int, InputTensorBoundsMetadata]
  symints_to_bounds: list[tuple[torch.SymInt, tuple[int, int]]]
  _user_defined_bound_for_symbols: set[str]

  def __init__(self, example_inputs: Sequence[InputType]):
    self._example_inputs = example_inputs
    self._user_defined_bound_for_symbols = set()
    self._populate_bounds_metadata()

  @property
  def example_inputs(self) -> Sequence[torch.Tensor | int]:
    return self._example_inputs

  def _populate_bounds_metadata(self) -> None:
    """Populates bounds metadata for input tensors and SymInt scalars."""
    self.input_tensors_metadata = {}
    self.symints_to_bounds = []
    for idx, arg in enumerate(self._example_inputs):
      if isinstance(arg, torch.Tensor):
        static_shape = []
        dynamic_bounds = []
        dynamic_dims = []
        for shape_idx, s in enumerate(arg.shape):
          if isinstance(s, torch.SymInt):
            lower, upper = get_symint_bounds(s)
            if symbol_bounds.is_user_defined_bound(s):
              self._user_defined_bound_for_symbols.add(str(s))
            self.symints_to_bounds.append((s, (lower, upper)))
            if lower == upper:
              static_shape.append(lower)
            else:
              static_shape.append(-1)
              dynamic_dims.append(shape_idx)
              dynamic_bounds.append((lower, upper))
          else:
            static_shape.append(int(s))

        self.input_tensors_metadata[idx] = InputTensorBoundsMetadata(
            static_shape=static_shape,
            dynamic_bounds=dynamic_bounds,
            dynamic_dims=dynamic_dims,
        )
      elif isinstance(arg, torch.SymInt):
        lower, upper = get_symint_bounds(arg)
        if symbol_bounds.is_user_defined_bound(arg):
          self._user_defined_bound_for_symbols.add(str(arg))
        self.symints_to_bounds.append((arg, (lower, upper)))

  def get_num_dynamic_dims(self, tensor_pos: int) -> int:
    """Returns the number of dynamic dimensions for the tensor at tensor_pos."""
    metadata = self.input_tensors_metadata.get(tensor_pos)
    if metadata is None:
      return 0
    return len(metadata.dynamic_dims)

  def get_symint_upper_bound(self, symint: torch.SymInt) -> int:
    """Returns the upper bound of a given SymInt symbol."""
    sym_str = str(symint)
    for s, (_, upper) in self.symints_to_bounds:
      if str(s) == sym_str:
        return upper
    raise ValueError(f"No bounds found for SymInt {sym_str}")

  def _update_symint_upper_bounds(
      self, upper_bounds: Mapping[str, int]
  ) -> None:
    """Updates the upper bounds for symbols across internal metadata."""
    self.symints_to_bounds = [
        (s, (lower, upper_bounds.get(str(s), upper)))
        for s, (lower, upper) in self.symints_to_bounds
    ]

    for idx, arg in enumerate(self._example_inputs):
      if isinstance(arg, torch.Tensor):
        metadata = self.input_tensors_metadata.get(idx)
        if metadata is None:
          continue
        metadata.dynamic_bounds = [
            (lower, upper_bounds.get(str(arg.shape[dim_idx]), upper))
            for dim_idx, (lower, upper) in zip(
                metadata.dynamic_dims, metadata.dynamic_bounds
            )
        ]

  def get_bounded_shape(self, tensor_pos: int) -> list[int]:
    """Returns the shape of the tensor with dynamic dimensions replaced by their bounds."""
    metadata = self.input_tensors_metadata.get(tensor_pos)
    if metadata is None:
      raise ValueError(
          f"Tensor position {tensor_pos} not in input tensors metadata."
      )

    if not metadata.dynamic_dims:
      return list(metadata.static_shape)

    bounded_shape = list(metadata.static_shape)
    for (_, upper_bound), dim in zip(
        metadata.dynamic_bounds, metadata.dynamic_dims
    ):
      if 0 <= dim < len(bounded_shape):
        bounded_shape[dim] = upper_bound
      else:
        raise ValueError(
            f"Dynamic dimension index {dim} is out of bounds for static"
            f" shape {metadata.static_shape} of input index {tensor_pos}."
        )
    return bounded_shape

  def add_bound_checks(self) -> None:
    """Adds bound checks for all SymInts in the graph."""
    unique_symints = {str(s): (s, b) for s, b in self.symints_to_bounds}
    for _, (symint, (lower, upper)) in unique_symints.items():
      torch._check(symint >= lower, "Dynamic shape lower bound check failed")  # pylint: disable=protected-access
      torch._check(symint <= upper, "Dynamic shape upper bound check failed")  # pylint: disable=protected-access

  def validate_bounds(self) -> None:
    """Validates that all SymInt upper bounds satisfy active shape guards."""
    unique_symints = {str(s): (s, b) for s, b in self.symints_to_bounds}
    if not unique_symints:
      return

    shape_env = None
    upper_bounds = {}
    lower_bounds = {}
    for sym_name, (symint, (lower, upper)) in unique_symints.items():
      upper_bounds[sym_name] = upper
      lower_bounds[sym_name] = lower
      if shape_env is None and hasattr(symint, "node"):
        shape_env = getattr(symint.node, "shape_env", None)

    if shape_env is None:
      return

    violated_guard = symbol_bounds.find_violated_guard(shape_env, upper_bounds)
    if violated_guard is None:
      return

    # In case of failure, fall back compiler-selected bounds.
    has_fallen_back = False
    for sym_name, lower in lower_bounds.items():
      fallback_upper = symbol_bounds.get_fallback_upper_bound(lower)
      if (
          sym_name not in self._user_defined_bound_for_symbols
          and upper_bounds[sym_name] != fallback_upper
      ):
        logging.debug(
            "Bound %s violated guard %s; falling back SymInt %s to %s",
            upper_bounds[sym_name],
            violated_guard,
            sym_name,
            fallback_upper,
        )
        upper_bounds[sym_name] = fallback_upper
        has_fallen_back = True

    if not has_fallen_back:
      raise ConstraintViolationError(
          f"Upper bounds {upper_bounds} violate active shape guard: "
          f"{violated_guard}."
      )

    self._update_symint_upper_bounds(upper_bounds)

    if (
        still_violated := symbol_bounds.find_violated_guard(
            shape_env, upper_bounds
        )
    ) is not None:
      raise ConstraintViolationError(
          f"Upper bounds {upper_bounds} violate active shape guard: "
          f"{still_violated}."
      )

  def get_example_inputs(self) -> tuple[list[torch.Tensor | int], list[Any]]:
    """Specializes dynamic inputs to their static upper bounds and constructs TensorBounds.

    Returns:
      A tuple containing:
        - A list of updated example inputs with SymInts replaced by their upper
          bounds, and new runtime size placeholders added.
        - A list of TensorBounds (or None) aligned with the updated example
          inputs.
    """
    updated_example_inputs = []
    aligned_bounds = []
    for index, arg in enumerate(self._example_inputs):
      if isinstance(arg, torch.Tensor):
        num_dynamic_dims = self.get_num_dynamic_dims(index)
        if num_dynamic_dims == 0:
          updated_example_inputs.append(arg)
          aligned_bounds.append(None)
          continue
        bounded_shape = self.get_bounded_shape(index)
        updated_example_inputs.append(
            torch.zeros(
                bounded_shape,
                dtype=arg.dtype,
                requires_grad=arg.requires_grad,
            )
        )
        tensor_metadata = self.input_tensors_metadata[index]
        dynamic_dims = list(tensor_metadata.dynamic_dims)
        upper_bounds = [upper for _, upper in tensor_metadata.dynamic_bounds]
        bounds_obj = tpu_torch_compile.TensorBounds(
            (dynamic_dims, upper_bounds)
        )
        aligned_bounds.append(bounds_obj)

      elif isinstance(arg, torch.SymInt):
        upper = self.get_symint_upper_bound(arg)
        updated_example_inputs.append(upper)
        aligned_bounds.append(None)
        # Add an input for the runtime size placeholder for the symint.
        updated_example_inputs.append(torch.tensor(upper, dtype=torch.int32))
        aligned_bounds.append(None)
      else:
        updated_example_inputs.append(arg)
        aligned_bounds.append(None)

    return updated_example_inputs, aligned_bounds

  def get_precomputed_bounds_list(self) -> list[ShapeBoundInfo]:
    """Precomputes a list of ShapeBoundInfo for dynamic input tensors."""
    bounds_list = []
    for idx, arg in enumerate(self._example_inputs):
      if isinstance(arg, torch.Tensor) and self.get_num_dynamic_dims(idx) > 0:
        tensor_metadata = self.input_tensors_metadata[idx]
        bounds_list.append(
            ShapeBoundInfo(
                dynamic_dims=tuple(tensor_metadata.dynamic_dims),
                upper_bounds=tuple(
                    upper for _, upper in tensor_metadata.dynamic_bounds
                ),
            )
        )
    return bounds_list
