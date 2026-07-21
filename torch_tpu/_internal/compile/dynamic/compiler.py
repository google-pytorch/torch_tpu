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
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import NamedTuple
from absl import logging
import torch
from torch._inductor.utils import InputType
from torch._logging import LazyString
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.dynamic.dynamic_adapters import DynamicAdapterLinearHypothesis
from torch_tpu._internal.compile.dynamic.dynamic_adapters import ShapeBoundInfo
from torch_tpu._internal.compile.dynamic.graph_transformations import apply_dynamism_transformations
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager
from torch_tpu._internal.compile.dynamic.symbol_bounds import get_symint_bounds


def _get_example_inputs(
    example_inputs: Sequence[Any], sym_shape_manager: SymShapeManager
) -> tuple[list[torch.Tensor | int], list[Any]]:
  """Specializes dynamic inputs to their static upper bounds and constructs bounds.

  Args:
    example_inputs: The example inputs to the FX graph.
    sym_shape_manager: Symbolic shape manager to get the dynamic shape
      information.

  Returns:
    A tuple containing:
      - A list of updated example inputs with SymInts replaced by their upper
        bounds, and new runtime size placeholders added.
      - A list of TensorBounds (or None) aligned with the updated example
        inputs.
  """
  updated_example_inputs = []
  aligned_bounds = []
  for index, arg in enumerate(example_inputs):
    if isinstance(arg, torch.Tensor):
      num_dynamic_dims = sym_shape_manager.get_num_dynamic_dims(index)
      if num_dynamic_dims == 0:
        updated_example_inputs.append(arg)
        aligned_bounds.append(None)
        continue
      bounded_shape = sym_shape_manager.get_bounded_shape(index)
      updated_example_inputs.append(
          torch.zeros(
              bounded_shape,
              dtype=arg.dtype,
              requires_grad=arg.requires_grad,
          )
      )
      # Construct TensorBounds
      tensor_metadata = sym_shape_manager.input_tensors_metadata[index]
      dynamic_dims = list(tensor_metadata.dynamic_dims)
      upper_bounds = [upper for _, upper in tensor_metadata.dynamic_bounds]
      bounds_obj = tpu_torch_compile.TensorBounds((dynamic_dims, upper_bounds))
      aligned_bounds.append(bounds_obj)

    elif isinstance(arg, torch.SymInt):
      _, upper = get_symint_bounds(arg)
      updated_example_inputs.append(upper)
      aligned_bounds.append(None)
      # Add an input for the runtime size placeholder for the symint.
      updated_example_inputs.append(torch.tensor(upper, dtype=torch.int32))
      aligned_bounds.append(None)
    else:
      updated_example_inputs.append(arg)
      aligned_bounds.append(None)

  return updated_example_inputs, aligned_bounds


def _extract_minor_to_major(
    parameter_layouts: Sequence[Any] | None,
) -> tuple[tuple[int, ...], ...]:
  """Extracts minor_to_major layouts from XLA parameter layouts."""
  if parameter_layouts is None:
    return ()
  layouts = []
  for layout in parameter_layouts:
    if layout is not None:
      layouts.append(tuple(layout[0]))
    else:
      layouts.append(())
  return tuple(layouts)


class _TensorInfo(NamedTuple):
  """Tensor information.

  Attributes:
    shape: The static shape of the tensor.
    dtype: The data type of the tensor.
  """

  shape: list[int]
  dtype: torch.dtype


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
      static_compiler: compiler.StaticCompiler,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[Any],
      sym_shape_manager: SymShapeManager,
      is_fwd: bool,
      precompile_steps: int,
      default_executable: Any,
      default_layout_key: tuple[tuple[int, ...], ...],
  ):
    self.static_compiler = static_compiler
    self.graph_module = graph_module
    self.example_inputs = example_inputs
    self.sym_shape_manager = sym_shape_manager
    self.is_fwd = is_fwd
    self._precompile_steps = precompile_steps

    # Cache for compiled executables: layout_key -> executable.
    self.model_executables: dict[tuple[tuple[int, ...], ...], Any] = {
        default_layout_key: default_executable
    }

    # Precompute static inputs and bounds for the transformed graph.
    self.model_example_inputs, self.aligned_bounds = _get_example_inputs(
        example_inputs, sym_shape_manager
    )

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

    outputs = [None] * len(tensor_args)
    pad_module_indices = []
    pad_module_args = []
    pad_module_info = []
    pad_module_bounds = []

    for i, (tensor, info) in enumerate(zip(tensor_args, tensor_info)):
      bounds = self._precomputed_bounds_list[i]
      padded_shape = list(tensor.shape)
      for dim, upper_bound in zip(bounds.dynamic_dims, bounds.upper_bounds):
        padded_shape[dim] = upper_bound

      if tpu_torch_compile.is_device_shape_dynamic(tensor):
        # Already dynamic, reuse it
        outputs[i] = tensor
      else:
        # Non-dynamic tensor, pad and make it dynamic
        pad_module_indices.append(i)
        pad_module_args.append(tensor)
        pad_module_info.append(info)
        pad_module_bounds.append(bounds)

    if pad_module_args:
      logging.debug(
          "[DynamicTpuBackend] Compile Pad Subgraph, tensor_info: %s,"
          " bounds_list: %s",
          pad_module_info,
          pad_module_bounds,
      )

      compile_result = tpu_torch_compile.get_dynamic_pad_module(
          pad_module_info,
          pad_module_bounds,
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

      pad_outputs = tpu_torch_compile.execute(
          compile_result.executable, pad_module_args
      )

      for idx, output in zip(pad_module_indices, pad_outputs):
        outputs[idx] = output

    assert all(x is not None for x in outputs), "Not all outputs were filled"
    return outputs

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
        if idx in self.sym_shape_manager.dynamic_scalar_indices:
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

  def _get_model_inputs_layouts(
      self, model_inputs: Sequence[Any]
  ) -> tuple[tuple[tuple[int, ...], ...], list[list[int]]]:
    argument_layouts = []
    for val in model_inputs:
      if isinstance(val, torch.Tensor):
        layout = tpu_torch_compile.get_device_layout_if_materialized(val)
        if layout is not None:
          minor_to_major = layout[0]
        else:
          default_layout_info = tpu_torch_compile.get_default_layout(
              val.dtype, val.shape
          )
          minor_to_major = default_layout_info[0] if default_layout_info else []
        argument_layouts.append(minor_to_major)

    layout_key = tuple(tuple(l) for l in argument_layouts)
    return layout_key, argument_layouts

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
      # Since we compile it at compile time, it is guaranteed to be in cache.
      return self.model_executables[()](list(args))

    tensor_args, tensor_info = self._get_pad_subgraph_inputs(args)

    # Run pad subgraph first (no model_executable needed)
    pad_outputs = self._compile_and_execute_pad_subgraph(
        tensor_args, tensor_info
    )

    # Construct inputs for the model executable
    model_inputs = self._get_model_inputs(args, pad_outputs)

    # Extract layouts of the constructed model_inputs to construct layout_key
    # and argument_layouts
    layout_key, argument_layouts = self._get_model_inputs_layouts(model_inputs)

    # Check cache, compile if miss
    if layout_key not in self.model_executables:
      logging.debug(
          "[DynamicTpuBackend] Compilation cache miss for layouts: %s."
          " Compiling new executable.",
          layout_key,
      )
      executable = self.static_compiler(
          self.graph_module,
          self.model_example_inputs,
          is_fwd=self.is_fwd,
          bounds=self.aligned_bounds,
          argument_layouts=argument_layouts,
      )
      logging.debug(
          "[DynamicTpuBackend] Compiled model executable layouts %s",
          executable.parameter_layouts,  # type: ignore[attr-defined]
      )
      self.model_executables[layout_key] = executable

    executable = self.model_executables[layout_key]

    output_shapes = _compute_output_shapes(self.sym_shape_manager, args)

    # Run model executable with constructed inputs
    outputs = executable(model_inputs, output_shapes=output_shapes)

    if self._precompile_steps > 0:
      self._precompile_dynamic_adapters(args, tensor_info)
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
        compilation_context, debug=debug, use_stablehlo_bounds=True
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

    # Create example inputs for the model executable and construct aligned
    # bounds.
    model_example_inputs, aligned_bounds = _get_example_inputs(
        example_inputs, sym_shape_manager
    )

    logging.debug(
        "[DynamicTpuBackend] Example inputs: %s, aligned bounds: %s",
        model_example_inputs,
        aligned_bounds,
    )

    # Create a model executable using the provided example inputs and bounds.
    default_executable = self.static_compiler(
        graph_module,
        model_example_inputs,
        is_fwd=is_fwd,
        bounds=aligned_bounds,
    )

    logging.debug(
        "[DynamicTpuBackend] Static Model Executable MLIR: %s",
        LazyString(lambda: str(getattr(default_executable, "mlir_text", None))),  # pyrefly: ignore[bad-argument-type]
    )

    default_layout_key = _extract_minor_to_major(
        default_executable.parameter_layouts,  # type: ignore[attr-defined]
    )
    logging.info(
        "[DynamicTpuBackend] Compiled default executable with layouts: %s",
        default_layout_key,
    )

    return _DynamicTpuCompiledExecutable(
        static_compiler=self.static_compiler,
        graph_module=graph_module,
        example_inputs=example_inputs,
        sym_shape_manager=sym_shape_manager,
        is_fwd=is_fwd,
        precompile_steps=self._precompile_steps,
        default_executable=default_executable,
        default_layout_key=default_layout_key,
    )
