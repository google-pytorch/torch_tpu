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
from typing import Any, NamedTuple
from absl import logging
import torch
from torch._inductor.utils import InputType
from torch._logging import LazyString
from torch.utils import _pytree
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.dynamic.dynamic_adapters import DynamicAdapterLinearHypothesis
from torch_tpu._internal.compile.dynamic.dynamic_adapters import ShapeBoundInfo
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
          tensor(),  # runtime size placeholder for s0
          tensor(10, 2),  # tensor with upper bounds
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
    elif isinstance(arg, torch.SymInt):
      _, upper = get_symint_bounds(arg)
      updated_example_inputs.append(upper)
      if index in sym_shape_manager.unique_dynamic_scalar_indices:
        # Add an input for the runtime size placeholder for the symint.
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


def _compile_and_execute_slice_subgraph(
    tensor_outputs: Sequence[torch.Tensor],
    output_shapes: Sequence[list[int]],
) -> list[torch.Tensor]:
  """Compiles and executes the slice subgraph."""
  target_shapes = []
  padded_shapes = []
  input_scalar_types = []

  for tensor, target_shape in zip(tensor_outputs, output_shapes):
    target_shapes.append(target_shape)
    padded_shapes.append(list(tensor.shape))
    input_scalar_types.append(tensor.dtype)

  logging.debug(
      "[DynamicTpuBackend] Compile Slice Subgraph, target_shapes: %s,"
      " padded_shapes: %s",
      LazyString(lambda: str(target_shapes)),
      LazyString(lambda: str(padded_shapes)),
  )

  build_mlir_module = logging.vlog_is_on(logging.DEBUG)

  compile_result = tpu_torch_compile.get_or_compile_slice_module(
      target_shapes,
      padded_shapes,
      input_scalar_types,
      build_mlir_module=build_mlir_module,
  )

  if compile_result.module is not None:
    logging.debug(
        "[DynamicTpuBackend] MLIR slice module: %s",
        LazyString(
            lambda: tpu_torch_compile.serialize_mlir_text(compile_result.module)
        ),
    )

  return tpu_torch_compile.execute(compile_result.executable, tensor_outputs)


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
      precompile_steps: int,
  ):
    self.model_executable = model_executable
    self.sym_shape_manager = sym_shape_manager
    self._precompile_steps = precompile_steps
    self._precomputed_bounds_list = self._precompute_bounds_list()
    self.input_linear_hypothesis = DynamicAdapterLinearHypothesis(
        self._precomputed_bounds_list
    )
    self._default_scalar_tensor_cache = {}
    self._prepare_input_packing_plan()
    self._backend_device = None
    self._padded_output_shapes = self._get_padded_output_shapes()

  def _get_backend_device(self) -> torch.device:
    if self._backend_device is None:
      self._backend_device = torch.accelerator.current_accelerator()
    return self._backend_device

  def _get_pad_subgraph_inputs(
      self,
      args: Sequence[Any],
  ) -> tuple[list[torch.Tensor], list[_TensorInfo]]:
    """Returns inputs for the pad subgraph."""
    tensor_args = []
    tensor_info = []

    for idx, arg in enumerate(args):
      # Only pass tensors with dynamic dimensions to the pad subgraph.
      if (
          isinstance(arg, torch.Tensor)
          and self.sym_shape_manager.get_num_dynamic_dims(idx) > 0
      ):
        if arg.device.type == "cpu":
          raise ValueError(
              "CPU tensors are not supported in dynamic shapes compilation,"
              " please move inputs to TPU."
          )
        tensor_args.append(arg)
        static_shape = list(arg.shape)
        tensor_info.append(_TensorInfo(shape=static_shape, dtype=arg.dtype))

    return tensor_args, tensor_info

  def _compile_and_execute_pad_subgraph(
      self,
      tensor_args: Sequence[torch.Tensor],
      tensor_info: Sequence[_TensorInfo],
  ) -> list[torch.Tensor]:
    """Compiles and executes the pad subgraph."""
    if not tensor_args:
      return []

    logging.debug(
        "[DynamicTpuBackend] Compile Pad Subgraph, tensor_info: %s,"
        " bounds_list: %s",
        tensor_info,
        self._precomputed_bounds_list,
    )

    build_mlir_module = logging.vlog_is_on(logging.DEBUG)

    compile_result = tpu_torch_compile.get_or_compile_pad_module(
        tensor_info,
        self._precomputed_bounds_list,
        build_mlir_module=build_mlir_module,
    )

    if compile_result.module is not None:
      logging.debug(
          "[DynamicTpuBackend] MLIR pad module: %s",
          LazyString(
              lambda: tpu_torch_compile.serialize_mlir_text(
                  compile_result.module
              )
          ),
      )

    return tpu_torch_compile.execute(compile_result.executable, tensor_args)

  def _precompute_bounds_list(self) -> list[ShapeBoundInfo]:
    """Precomputes a list of shape bound information for dynamic input tensors.

    This method iterates through the example inputs provided to the
    `SymShapeManager`. For each input tensor that contains one or more dynamic
    dimensions, it extracts the indices of these dynamic dimensions and their
    corresponding upper bounds. This information is
    compiled into a list of `ShapeBoundInfo` objects.

    Returns:
      A list of `ShapeBoundInfo`, where each entry contains the dynamic
      dimension indices and their upper bounds for each dynamic input tensor.
    """
    bounds_list = []
    for idx, arg in enumerate(self.sym_shape_manager.example_inputs):
      if (
          isinstance(arg, torch.Tensor)
          and self.sym_shape_manager.get_num_dynamic_dims(idx) > 0
      ):
        tensor_metadata = self.sym_shape_manager.input_tensors_metadata[idx]
        bounds_list.append(
            ShapeBoundInfo(
                dynamic_dims=tensor_metadata.dynamic_dims,
                upper_bounds=[
                    upper for _, upper in tensor_metadata.dynamic_bounds
                ],
            )
        )
    return bounds_list

  def _prepare_input_packing_plan(self):
    self._static_tensor_map = []
    self._dynamic_tensor_map = []
    self._dynamic_scalar_map = []

    target_idx = 0
    pad_idx = 0

    for idx, arg in enumerate(self.sym_shape_manager.example_inputs):
      if arg is None:
        continue

      if isinstance(arg, torch.SymInt):
        if idx in self.sym_shape_manager.unique_dynamic_scalar_indices:
          self._dynamic_scalar_map.append((idx, target_idx))
          target_idx += 1
        continue

      if isinstance(arg, torch.Tensor):
        if self.sym_shape_manager.get_num_dynamic_dims(idx) > 0:
          self._dynamic_tensor_map.append((pad_idx, target_idx))
          pad_idx += 1
          target_idx += 1
        else:
          self._static_tensor_map.append((idx, target_idx))
          target_idx += 1

    self._model_inputs_template_length = target_idx

  def _get_model_inputs(
      self,
      args: Sequence[Any],
      pad_outputs: Sequence[torch.Tensor],
  ) -> list[Any]:
    """Assembles the actual runtime inputs for the model executable.

    Args:
      args: The original arguments passed to the `__call__` method.
      pad_outputs: The output tensors from the pad subgraph, which are the
        statically padded versions of the dynamic input tensors.

    Returns:
      A list of inputs for the model executable, where dynamic tensors are
      replaced by their padded versions and dynamic scalar SymInts are
      converted to tensors.
    """
    model_inputs = [None] * self._model_inputs_template_length
    backend_device = self._get_backend_device()

    for arg_idx, target_idx in self._static_tensor_map:
      model_inputs[target_idx] = args[arg_idx]

    for pad_idx, target_idx in self._dynamic_tensor_map:
      model_inputs[target_idx] = pad_outputs[pad_idx]

    for arg_idx, target_idx in self._dynamic_scalar_map:
      val = args[arg_idx]
      tensor = self._default_scalar_tensor_cache.get(val)
      if tensor is None:
        tensor = torch.tensor(val, dtype=torch.int32, device=backend_device)
        self._default_scalar_tensor_cache[val] = tensor
      model_inputs[target_idx] = tensor

    return model_inputs

  def _get_padded_output_shapes(self):
    padded_input_shapes = []
    for idx, arg in enumerate(self.sym_shape_manager.example_inputs):
      if isinstance(arg, torch.SymInt):
        upper_bound = self.sym_shape_manager.get_symint_upper_bound(arg)
        padded_input_shapes.append(upper_bound)
      else:
        padded_input_shapes.append(
            self.sym_shape_manager.get_bounded_shape(idx)
        )
    padded_output_shapes = _compute_output_shapes(
        self.sym_shape_manager, padded_input_shapes
    )
    return padded_output_shapes

  def _precompile_dynamic_adapters(
      self,
      args: Sequence[Any],
      tensor_info: Sequence[_TensorInfo],
  ) -> None:
    """Precompiles dynamic adapters in the background with a linear hypothesis.

    This method updates the input linear tracker, and if the input behavior is
    linear, predicts the future input shapes and output slice shapes, and
    enqueues their compilation.
    Args:
      args: The runtime positional arguments passed to the executable call.
      tensor_info: Tensor metadata representing current input shapes and dtypes.
    """
    # Map from original tensor index to its index in tensor_info
    # (which only has dynamic tensors).
    original_to_dynamic_idx = {}
    dyn_idx = 0
    for idx, arg in enumerate(self.sym_shape_manager.example_inputs):
      if not isinstance(arg, torch.Tensor) or (
          self.sym_shape_manager.get_num_dynamic_dims(idx) == 0
      ):
        continue
      original_to_dynamic_idx[idx] = dyn_idx
      dyn_idx += 1

    self.input_linear_hypothesis.update([info.shape for info in tensor_info])
    if self.input_linear_hypothesis.is_linear:
      logging.debug(
          "[DynamicTpuBackend] Linear hypothesis detected, precompiling dynamic"
          " adapters"
      )
      output_types = self.sym_shape_manager.get_output_dtypes()
      updated_tensor_info = [
          _TensorInfo(shape=list(info.shape), dtype=info.dtype)
          for info in tensor_info
      ]
      for shape_update in self.input_linear_hypothesis.get_shape_updates(
          self._precompile_steps
      ):
        for (tensor_idx, dim_idx), value in shape_update.items():
          updated_tensor_info[tensor_idx].shape[dim_idx] = value
        tpu_torch_compile.precompile_pad_module(
            updated_tensor_info, self._precomputed_bounds_list
        )

        sym_values = {}
        for sym_name, (
            t_idx,
            d_idx,
        ) in self.sym_shape_manager.symint_to_tensor_and_dim_idx.items():
          dyn_idx = original_to_dynamic_idx.get(t_idx)
          assert dyn_idx is not None, f"Expected tensor {t_idx} to be dynamic"
          sym_values[sym_name] = updated_tensor_info[dyn_idx].shape[d_idx]

        for (
            sym_name,
            arg_idx,
        ) in self.sym_shape_manager.symint_to_arg_idx.items():
          if sym_name not in sym_values:
            sym_values[sym_name] = args[arg_idx]

        runtime_output_shapes = (
            self.sym_shape_manager.compute_output_shapes_from_sym_values(
                sym_values
            )
        )

        if runtime_output_shapes is not None:
          tpu_torch_compile.precompile_slice_module(
              runtime_output_shapes, self._padded_output_shapes, output_types
          )
    else:
      logging.debug(
          "[DynamicTpuBackend] Non-linear hypothesis detected, not precompiling"
          " slice subgraph"
      )

  def __call__(self, *args: Any) -> Any:
    logging.debug("[DynamicTpuBackend] Execute Model")

    if not args:
      # If no args are passed, we assume there is no dynamic shape and we can
      # directly execute the model executable.
      return self.model_executable(list(args))

    tensor_args, tensor_info = self._get_pad_subgraph_inputs(args)

    # Compile and run pad executable to get statically padded tensors
    pad_outputs = self._compile_and_execute_pad_subgraph(
        tensor_args, tensor_info
    )
    # Create input for the model executable
    model_inputs = self._get_model_inputs(args, pad_outputs)

    # Run model executable with constructed inputs
    outputs = self.model_executable(model_inputs)

    if self._precompile_steps > 0:
      self._precompile_dynamic_adapters(args, tensor_info)

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
      precompile_steps: int = 0,
  ):
    """Initializes the DynamicCompiler instance.

    Args:
      compilation_context: A `compiler.CompilationContext` instance used for
        maintaining compilation state.
      debug: A `bool` that, when `True`, enables debug logging and artifact
        generation.
      precompile_steps: The number of steps to precompile dynamic adapters.
    """
    if compilation_context is None:
      compilation_context = compiler.CompilationContext()
    super().__init__(compilation_context, debug=debug)
    self._precompile_steps = precompile_steps
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

    print("[DEBUG] Index nodes in transformed graph:", flush=True)
    for node in graph_module.graph.nodes:
      if node.op == "call_function" and "index" in str(node.target):
        print(
            f"  [DEBUG INDEX] node: {node}, args: {node.args}, kwargs:"
            f" {node.kwargs}",
            flush=True,
        )
        for arg in node.args:
          if isinstance(arg, torch.fx.Node):
            val = arg.meta.get("val")
            if val is not None:
              print(
                  f"    [DEBUG INDEX] arg {arg.name} val type: {type(val)},"
                  f" val: {val}",
                  flush=True,
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
        LazyString(lambda: getattr(static_model_executable, "mlir_text", None)),
    )

    return _DynamicTpuCompiledExecutable(
        model_executable=static_model_executable,
        sym_shape_manager=sym_shape_manager,
        precompile_steps=self._precompile_steps,
    )
