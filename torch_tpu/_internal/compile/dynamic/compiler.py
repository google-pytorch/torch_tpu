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

"""Compiler for handling dynamic shape in torch.compile()."""

from __future__ import annotations
from collections.abc import Callable, Sequence
import numbers
from typing import Any, NamedTuple
from absl import logging
import torch
from torch._inductor.utils import InputType
from torch._logging import LazyString
from torch.utils import _pytree
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.dynamic.graph_transformations import apply_dynamism_transformations
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager
from torch_tpu._internal.compile.dynamic.symbol_bounds import get_symint_bounds


def _get_example_inputs(
    example_inputs: Sequence[Any], sym_shape_manager: SymShapeManager
) -> list[torch.Tensor | int]:
  """Specializes dynamic inputs to their static upper bounds.

  Args:
    example_inputs: The example inputs to the FX graph.
    sym_shape_manager: Symbolic shape manager to get the dynamic shape
      information.

  Returns:
    A list of updated example inputs with SymInts in inputs replaced by
    their upper bounds. The updated example inputs also include
    any new runtime size placeholders introduced for dynamic tensors and
    generative ops.

  Example:
    example_inputs = [SymInt(s0), tensor(s0, 2)]  # s0 <= 10

    Returns:
      updated_example_inputs = [
          10,  # s0 -> upper bound
          tensor(10, 2),  # tensor with upper bounds
          tensor(), # runtime size placeholder for s0
      ]
  """
  updated_example_inputs = []
  for index, arg in enumerate(example_inputs):
    if isinstance(arg, torch.Tensor):
      num_dynamic_dims = sym_shape_manager.get_num_dynamic_dims(index)
      if num_dynamic_dims == 0:
        updated_example_inputs.append(arg)
        continue
      bounded_shape = sym_shape_manager.get_bounded_shape(index)
      updated_example_inputs.append(
          torch.zeros(
              bounded_shape,
              dtype=arg.dtype,
              requires_grad=arg.requires_grad,
          )
      )
      # The dynamic shape info stores upper bounds for the dynamic dimensions.
      # We add a placeholder for each dynamic dimension to be used by the
      # set_dimension_size operation.
      dynamic_bounds = sym_shape_manager.input_tensors_metadata[
          index
      ].dynamic_bounds
      for _, upper_bound in dynamic_bounds:
        updated_example_inputs.append(
            torch.tensor(upper_bound, dtype=torch.int32)
        )
    elif isinstance(arg, torch.SymInt):
      _, upper = get_symint_bounds(arg)
      updated_example_inputs.append(upper)
      # For generative ops, we may create a new placeholder for the symint
      # that is passed to set_dimension_size as an argument.
      if str(arg) in sym_shape_manager.symint_to_placeholder:
        updated_example_inputs.append(torch.tensor(upper, dtype=torch.int32))
    else:
      updated_example_inputs.append(arg)

  return updated_example_inputs


class _TensorInfo(NamedTuple):
  """Tensor information.

  Attributes:
    shape: The static shape of the tensor.
    dtype: The data type of the tensor.
  """

  shape: list[int]
  dtype: torch.dtype


class _ShapeBoundInfo(NamedTuple):
  """Shape bound information for dynamic dimensions of a tensor.

  Attributes:
    dynamic_dims: A list of indices of the dynamic dimensions.
    upper_bounds: A list of upper bounds for each corresponding dynamic
      dimension.
  """

  dynamic_dims: list[int]
  upper_bounds: list[int]


def _get_pad_subgraph_inputs(
    args: Sequence[Any],
    sym_shape_manager: SymShapeManager,
) -> tuple[
    list[torch.Tensor],
    list[_TensorInfo],
    list[_ShapeBoundInfo],
]:
  """Returns inputs for the pad subgraph."""
  tensor_args = []
  tensor_info = []
  bounds_list = []

  backend_device = torch.accelerator.current_accelerator()

  for idx, arg in enumerate(args):
    if arg is None:
      continue

    if isinstance(arg, numbers.Integral):
      example_arg = sym_shape_manager.example_inputs[idx]
      if isinstance(example_arg, torch.SymInt):
        if str(example_arg) in sym_shape_manager.symint_to_placeholder:
          tensor_val = torch.tensor(
              arg, dtype=torch.int32, device=backend_device
          )
          tensor_args.append(tensor_val)
          tensor_info.append(_TensorInfo(shape=[], dtype=torch.int32))
          bounds_list.append(_ShapeBoundInfo(dynamic_dims=[], upper_bounds=[]))
      continue

    # Handle any other missing type above
    assert isinstance(arg, torch.Tensor)

    if arg.device.type == "cpu":
      raise ValueError(
          "CPU tensors are not supported in dynamic shapes compilation, please"
          " move inputs to TPU."
      )

    tensor_args.append(arg)
    static_shape = list(arg.shape)
    tensor_info.append(_TensorInfo(shape=static_shape, dtype=arg.dtype))

    tensor_metadata = sym_shape_manager.input_tensors_metadata[idx]
    if not tensor_metadata.dynamic_dims:
      bounds_list.append(_ShapeBoundInfo(dynamic_dims=[], upper_bounds=[]))
      continue
    bounds_list.append(
        _ShapeBoundInfo(
            dynamic_dims=tensor_metadata.dynamic_dims,
            upper_bounds=[upper for _, upper in tensor_metadata.dynamic_bounds],
        )
    )
  return tensor_args, tensor_info, bounds_list


def _compile_and_execute_pad_subgraph(
    args: Sequence[Any],
    sym_shape_manager: SymShapeManager,
) -> list[torch.Tensor]:
  """Executes the pad subgraph using PJRT host-assisted DMA."""
  tensor_args, tensor_info, bounds_list = _get_pad_subgraph_inputs(
      args, sym_shape_manager
  )

  logging.debug(
      "[DynamicTpuBackend] Execute Pad Subgraph via Host DMA, tensor_info: %s,"
      " bounds_list: %s",
      tensor_info,
      bounds_list,
  )

  pad_outputs = []
  for idx, arg in enumerate(tensor_args):
    bounds = bounds_list[idx]
    if not bounds.dynamic_dims:
      pad_outputs.append(arg)
    else:
      padded_shape = list(arg.shape)
      for dim_idx, upper_bound in zip(bounds.dynamic_dims, bounds.upper_bounds):
        padded_shape[dim_idx] = upper_bound

      padded_tensor = tpu_torch_compile.pad_via_host_dma(arg, padded_shape)
      pad_outputs.append(padded_tensor)

      for dim_idx in bounds.dynamic_dims:
        dim_size = arg.shape[dim_idx]
        size_tensor = torch.tensor(
            dim_size, dtype=torch.int32, device=arg.device
        )
        pad_outputs.append(size_tensor)

  return pad_outputs


def _compile_and_execute_slice_subgraph(
    tensor_outputs: Sequence[torch.Tensor],
    output_shapes: Sequence[list[int]],
) -> list[torch.Tensor]:
  """Executes the slice subgraph using PJRT host-assisted DMA."""
  logging.debug(
      "[DynamicTpuBackend] Execute Slice Subgraph via Host DMA,"
      " output_shapes: %s",
      LazyString(lambda: str(output_shapes)),
  )

  sliced_outputs = []
  for tensor, target_shape in zip(tensor_outputs, output_shapes):
    sliced_tensor = tpu_torch_compile.slice_via_host_dma(tensor, target_shape)
    sliced_outputs.append(sliced_tensor)

  return sliced_outputs


def _compute_output_shapes(
    sym_shape_manager: SymShapeManager,
    args: Sequence[Any],
) -> list[list[int]] | None:
  """Computes the runtime output shapes from symbolic shapes and arguments."""
  if not sym_shape_manager.outputs_sym_shape:
    return None

  output_shapes = [
      output_sym_shape.get_output_runtime_shape(args)
      for output_sym_shape in sym_shape_manager.outputs_sym_shape
  ]
  logging.debug(
      "[_DynamicTpuCompiledExecutable] Output shapes: %s", output_shapes
  )
  return output_shapes


class _DynamicTpuCompiledExecutable(compiler.CompiledArtifact):
  """A callable wrapper for dynamic MLIR programs with TPU executable."""

  def __init__(
      self,
      model_executable: Any,
      sym_shape_manager: SymShapeManager,
  ):
    self.model_executable = model_executable
    self.sym_shape_manager = sym_shape_manager

  def __call__(self, *args: Any) -> Any:
    logging.debug("[DynamicTpuBackend] Execute Model")

    if not args:
      # If no args are passed, we assume there is no dynamic shape and we can
      # directly execute the model executable.
      return self.model_executable(list(args))

    # Compile and run pad executable to get statically padded tensors
    # and runtime size tensors
    pad_outputs = _compile_and_execute_pad_subgraph(
        args,
        self.sym_shape_manager,
    )

    # Run model executable with pads, sizes, and explicit output shapes
    outputs = self.model_executable(list(pad_outputs))

    output_shapes = _compute_output_shapes(self.sym_shape_manager, args)

    # If the output has dynamic shape, we need to slice it back
    if output_shapes is not None:
      flat_outputs, spec = _pytree.tree_flatten(outputs)
      tensor_outputs = []
      tensor_indices = []
      target_shapes = []

      for idx, (output_tensor, target_shape) in enumerate(
          zip(flat_outputs, output_shapes)
      ):
        if isinstance(output_tensor, torch.Tensor):
          tensor_outputs.append(output_tensor)
          tensor_indices.append(idx)
          target_shapes.append(target_shape)

      if tensor_outputs:
        sliced_tensors = _compile_and_execute_slice_subgraph(
            tensor_outputs, target_shapes
        )
        for idx, sliced_tensor in zip(tensor_indices, sliced_tensors):
          flat_outputs[idx] = sliced_tensor

      return _pytree.tree_unflatten(flat_outputs, spec)
    return outputs

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    # TODO(b/903508278): Add support for pickling to DynamicCompiler.
    raise NotImplementedError(
        "Serialization support for DynamicCompiler not yet implemented"
    )


class DynamicCompiler(compiler.Compiler):
  """Compiler for handling dynamic shapes in torch.compile()."""

  def __init__(
      self,
      compilation_context: compiler.CompilationContext | None = None,
      debug: bool = False,
  ):
    """Initializes the DynamicCompiler instance.

    Args:
      compilation_context: A `compiler.CompilationContext` instance used for
        maintaining compilation state.
      debug: A `bool` that, when `True`, enables debug logging and artifact
        generation.
    """
    if compilation_context is None:
      compilation_context = compiler.CompilationContext()
    super().__init__(compilation_context, debug=debug)
    self.static_compiler = compiler.StaticCompiler(
        compilation_context, debug=debug
    )

  def execute_pre_grad_passes(
      self,
      graph_module: torch.fx.GraphModule,
  ) -> None:
    self.static_compiler.execute_pre_grad_passes(graph_module)

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
  ) -> _DynamicTpuCompiledExecutable:
    """Called by AOT Autograd to compile the graph.

    Args:
      graph_module: The FX graph module to be compiled.
      example_inputs: A list of example inputs for the graph module.
      is_fwd: Indicates whether the forward or backward pass is being compiled.

    Returns:
      A callable `_DynamicTpuCompiledExecutable` that wraps the compiled model
      and handles dynamic shapes at runtime.
    """
    # TODO: Prevent truncation of log lines.
    logging.debug(
        "[DynamicTpuBackend] Compile FX Graph: %s",
        LazyString(graph_module.print_readable),
    )

    # Create a SymInt shape manager.
    sym_shape_manager = SymShapeManager(graph_module, example_inputs)

    # Add bound checks on symints. This forces dynamo to trigger a recompilation
    # if the bound constraint isn't satisfied.
    sym_shape_manager.add_bound_checks()

    # Transform the graph module to handle dynamic shapes.
    apply_dynamism_transformations(graph_module, sym_shape_manager)

    # TODO: Prevent truncation of log lines.
    logging.debug(
        "[DynamicTpuBackend] Transformed FX Graph: %s",
        LazyString(graph_module.print_readable),
    )

    # Create example inputs for the model executable.
    model_example_inputs = _get_example_inputs(
        example_inputs, sym_shape_manager
    )

    logging.debug(
        "[DynamicTpuBackend] Example inputs: %s",
        model_example_inputs,
    )

    # Create a static model executable with padded shape tensors as inputs
    static_model_executable = self.static_compiler(
        graph_module,
        model_example_inputs,
        is_fwd=is_fwd,
    )

    logging.debug(
        "[DynamicTpuBackend] Static Model Executable MLIR: %s",
        LazyString(lambda: static_model_executable.mlir_text),
    )

    return _DynamicTpuCompiledExecutable(
        model_executable=static_model_executable,
        sym_shape_manager=sym_shape_manager,
    )
