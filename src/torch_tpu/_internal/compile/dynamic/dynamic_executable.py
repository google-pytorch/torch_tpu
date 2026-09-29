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

"""Compiled executable wrapper for dynamic shapes in torch.compile()."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from absl import logging
import torch
from torch._logging import LazyString
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic import view_decomposition as decompose
from torch_tpu._internal.compile.dynamic.output_sym_shape import OutputSymShape


class _TensorInfo(NamedTuple):
  """Tensor information.

  Attributes:
    shape: The static shape of the tensor.
    dtype: The data type of the tensor.
  """

  shape: list[int]
  dtype: torch.dtype


class DynamicTpuCompiledExecutable(compiler.CompiledArtifact):
  """A callable wrapper for dynamic MLIR programs with TPU executable."""

  def __init__(
      self,
      static_compiler: compiler.StaticCompiler,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[Any],
      output_sym_shapes: Sequence[OutputSymShape],
      is_fwd: bool,
      bounded_model_inputs: list[torch.Tensor | int],
      model_input_bounds: list[Any],
      view_arg_indices: set[int],
      use_pad_real_to_static: bool = True,
      module_name: str | None = None,
  ):
    """Initializes DynamicTpuCompiledExecutable.

    Args:
      static_compiler: Compiler used to compile static MLIR executables on
        layout cache misses.
      graph_module: Transformed FX graph module specialized for this
        executable's symbol bounds.
      example_inputs: Original example inputs used to build the runtime input
        packing plan.
      output_sym_shapes: Symbolic shape specs for computing runtime shapes of
        output tensors.
      is_fwd: Whether this executable represents a forward pass (True) or
        backward pass (False).
      bounded_model_inputs: Example inputs specialized to static upper bounds
        for compilation.
      model_input_bounds: Per-input TensorBounds (or None for static inputs) for
        lowering and padding.
      view_arg_indices: Input argument indices whose view tensors were
        decomposed into base tensors.
      use_pad_real_to_static: Whether inputs are padded in-graph via
        pad_real_to_static instead of a pad subgraph.
      module_name: Optional custom name for the compiled MLIR module.
    """
    self._static_compiler = static_compiler
    self._graph_module = graph_module
    self._example_inputs = example_inputs
    self._output_sym_shapes = output_sym_shapes
    self._promoted_scalar_output_indices = [
        idx for idx, s in enumerate(output_sym_shapes) if s.is_promoted_scalar
    ]
    self._is_fwd = is_fwd
    self._view_arg_indices = view_arg_indices
    self._module_name = module_name

    # Cache for compiled executables: layout_key -> executable.
    self._model_executables: dict[tuple[tuple[int, ...], ...], Any] = {}

    self._bounded_model_inputs = bounded_model_inputs
    self._model_input_bounds = model_input_bounds

    self._dynamic_outputs = [
        out_shape.is_dynamic for out_shape in output_sym_shapes
    ]
    self._unique_output_indices: Sequence[int] | None = None

    self._use_pad_subgraph = not use_pad_real_to_static
    self._default_scalar_tensor_cache = {}
    self._prepare_input_packing_plan()
    self._backend_device = None

  def _get_backend_device(self) -> torch.device:
    if self._backend_device is None:
      self._backend_device = torch.accelerator.current_accelerator()
    return self._backend_device

  def _get_dynamic_input_tensors(
      self,
      args: Sequence[Any],
  ) -> list[torch.Tensor]:
    """Returns dynamic input tensors and validates they are on device."""
    tensor_args = []

    for idx in self._dynamic_input_indices:
      arg = args[idx]
      if arg.device.type == "cpu":
        raise ValueError(
            "CPU tensors are not supported in dynamic shapes compilation,"
            " please move inputs to TPU."
        )
      tensor_args.append(arg)

    return tensor_args

  def _compile_and_execute_pad_subgraph(
      self,
      tensor_args: Sequence[torch.Tensor],
  ) -> list[torch.Tensor]:
    """Compiles and executes the pad subgraph."""
    if not tensor_args:
      return []

    outputs = [None] * len(tensor_args)
    pad_module_indices = []
    pad_module_args = []
    pad_module_info = []
    pad_module_bounds = []

    for i, tensor in enumerate(tensor_args):
      bounds = self._dynamic_tensor_bounds[i]

      if tpu_torch_compile.is_device_shape_dynamic(tensor):
        # Already dynamic, reuse it
        outputs[i] = tensor
      else:
        # Non-dynamic tensor, pad and make it dynamic
        pad_module_indices.append(i)
        pad_module_args.append(tensor)
        pad_module_info.append(
            _TensorInfo(shape=list(tensor.shape), dtype=tensor.dtype)
        )
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

  def _prepare_input_packing_plan(self):
    self._static_tensor_map = []
    self._dynamic_tensor_map = []
    self._dynamic_scalar_map = []
    self._static_scalar_map = []
    self._dynamic_input_indices = []
    self._dynamic_tensor_bounds = [
        b for b in self._model_input_bounds if b is not None
    ]

    target_idx = 0
    pad_idx = 0

    for idx, arg in enumerate(self._example_inputs):
      if arg is None:
        continue

      if isinstance(arg, torch.SymInt):
        self._dynamic_scalar_map.append((idx, target_idx))
      elif isinstance(arg, torch.Tensor):
        if sym_utils.has_dynamic_shape(arg):
          self._dynamic_tensor_map.append((pad_idx, target_idx))
          self._dynamic_input_indices.append(idx)
          pad_idx += 1
        else:
          self._static_tensor_map.append((idx, target_idx))
      else:
        self._static_scalar_map.append((idx, target_idx))

      target_idx += 1

    self._model_inputs_template_length = target_idx
    self._dynamic_target_indices = {
        target_idx for _, target_idx in self._dynamic_tensor_map
    }

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

    for arg_idx, target_idx in self._static_scalar_map:
      model_inputs[target_idx] = args[arg_idx]

    return model_inputs

  def _get_model_inputs_layouts(
      self, model_inputs: Sequence[Any]
  ) -> tuple[tuple[tuple[int, ...], ...], list[list[int]]]:
    argument_layouts = []
    for idx, val in enumerate(model_inputs):
      if isinstance(val, torch.Tensor):
        is_dynamic = idx in self._dynamic_target_indices
        if is_dynamic:
          layout = tpu_torch_compile.get_device_layout_if_materialized(val)
          if layout is not None:
            minor_to_major = layout[0]
          else:
            default_layout_info = tpu_torch_compile.get_default_layout(
                val.dtype, val.shape
            )
            minor_to_major = (
                default_layout_info[0] if default_layout_info else []
            )
        else:
          minor_to_major = []
        argument_layouts.append(minor_to_major)

    layout_key = tuple(tuple(l) for l in argument_layouts)
    return layout_key, argument_layouts

  def _compute_output_shapes(
      self,
      args: Sequence[Any],
  ) -> list[tpu_torch_compile.OutputShape] | None:
    """Computes the runtime output shapes from symbolic shapes and arguments."""
    if not self._output_sym_shapes:
      return None

    raw_output_shapes = [
        tpu_torch_compile.OutputShape(
            dimensions=output_sym_shape.get_output_runtime_shape(args),
            is_dynamic=output_sym_shape.is_dynamic,
        )
        if output_sym_shape.is_tensor
        else None
        for output_sym_shape in self._output_sym_shapes
    ]
    if self._unique_output_indices is not None:
      output_shapes = [
          raw_output_shapes[i]
          for i in self._unique_output_indices
          if raw_output_shapes[i] is not None
      ]
    else:
      output_shapes = [s for s in raw_output_shapes if s is not None]

    logging.debug(
        "[DynamicTpuCompiledExecutable] Output shapes: %s",
        LazyString(
            lambda: "["
            + ", ".join(
                f"OutputShape(dimensions={s.dimensions},"
                f" is_dynamic={s.is_dynamic})"
                for s in output_shapes
                if s is not None
            )
            + "]"
        ),
    )
    return output_shapes

  def _convert_promoted_outputs_to_scalars(self, outputs: Any) -> Any:
    """Converts output tensors at promoted scalar positions back to Python scalars."""
    if not self._promoted_scalar_output_indices or not isinstance(
        outputs, (list, tuple)
    ):
      return outputs

    outputs_list = list(outputs)
    for idx in self._promoted_scalar_output_indices:
      if idx < len(outputs_list) and isinstance(
          outputs_list[idx], torch.Tensor
      ):
        outputs_list[idx] = outputs_list[idx].item()

    return tuple(outputs_list) if isinstance(outputs, tuple) else outputs_list

  def get_executable(
      self,
      layout_key: tuple[tuple[int, ...], ...],
      argument_layouts: Sequence[Sequence[int]] | None = None,
  ) -> Any:
    """Returns the resolved compiled executable for `layout_key`, compiling if needed."""
    if layout_key not in self._model_executables:
      logging.debug(
          "[DynamicTpuBackend] Compilation cache miss for layouts: %s."
          " Compiling new executable.",
          layout_key,
      )
      try:
        executable = self._static_compiler(
            self._graph_module,
            self._bounded_model_inputs,
            is_fwd=self._is_fwd,
            bounds=self._model_input_bounds,
            argument_layouts=argument_layouts,
            dynamic_outputs=self._dynamic_outputs,
            module_name=self._module_name,
        )
        if hasattr(executable, "resolve"):
          executable.resolve()
      except (RuntimeError, ValueError) as e:
        raise NotImplementedError(
            "Failed to lower FX graph with dynamic shapes to MLIR with error:"
            f" {e}. Please set torch.compile(..., dynamic=False, ...) and try"
            " again."
        ) from e

      logging.debug(
          "[DynamicTpuBackend] Static Model Executable MLIR: %s",
          LazyString(lambda: str(getattr(executable, "mlir_text", None))),
      )
      logging.debug(
          "[DynamicTpuBackend] Compiled model executable layouts %s",
          getattr(executable, "parameter_layouts", None),
      )
      self._unique_output_indices = getattr(
          executable, "unique_output_indices", None
      )
      self._model_executables[layout_key] = executable

    return self._model_executables[layout_key]

  def __call__(self, *args: Any) -> Any:
    logging.debug("[DynamicTpuBackend] Execute Model")

    if not args:
      # If no args are passed, we assume there is no dynamic shape and we can
      # directly execute the model executable.
      return self.get_executable(())(list(args))

    # Resolve base tensor for arguments that were decomposed views at compile time.
    if self._view_arg_indices:
      args_list = list(args)
      for idx in self._view_arg_indices:
        if idx < len(args_list) and isinstance(args_list[idx], torch.Tensor):
          args_list[idx] = decompose.get_base_tensor(args_list[idx])
      args = tuple(args_list)

    tensor_args = self._get_dynamic_input_tensors(args)

    # Run pad subgraph first if needed (for libtpu < 0.0.48 or 64-bit/128-bit
    # tensors)
    if self._use_pad_subgraph:
      pad_outputs = self._compile_and_execute_pad_subgraph(tensor_args)
    else:
      pad_outputs = tensor_args

    # Construct inputs for the model executable
    model_inputs = self._get_model_inputs(args, pad_outputs)

    # Extract layouts of the constructed model_inputs to construct layout_key
    # and argument_layouts
    layout_key, argument_layouts = self._get_model_inputs_layouts(model_inputs)

    logging.debug(
        "[DynamicTpuBackend] Layout key: %s, argument layouts: %s",
        layout_key,
        argument_layouts,
    )

    executable = self.get_executable(layout_key, argument_layouts)

    output_shapes = self._compute_output_shapes(args)

    # Run model executable with constructed inputs
    outputs = executable(model_inputs, output_shapes=output_shapes)
    return self._convert_promoted_outputs_to_scalars(outputs)

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    # TODO(b/903508278): Add support for pickling to DynamicCompiler.
    raise NotImplementedError(
        "Serialization support for DynamicCompiler not yet implemented"
    )

  def updates_default_generator_state(self) -> bool:
    return any(
        module.updates_default_generator_state()
        for module in self._model_executables.values()
    )
