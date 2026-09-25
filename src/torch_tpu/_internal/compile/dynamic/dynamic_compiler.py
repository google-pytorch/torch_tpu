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

from collections.abc import Sequence
import copy
from typing import Any

from absl import logging
import torch
from torch._inductor.utils import InputType
from torch._logging import LazyString
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic import view_ops_passes
from torch_tpu._internal.compile.dynamic.dynamic_executable import DynamicTpuCompiledExecutable
from torch_tpu._internal.compile.dynamic.graph_transformations import apply_dynamism_transformations
from torch_tpu._internal.compile.dynamic.graph_transformations import apply_pre_dynamism_transformations
from torch_tpu._internal.compile.dynamic.output_sym_shape import create_output_sym_shapes
from torch_tpu._internal.compile.dynamic.output_sym_shape import OutputSymShape
from torch_tpu._internal.compile.dynamic.sym_bounds_manager import SymBoundsManager
from torch_tpu._internal.compile.dynamic.symbol_bounds import SymbolBounds
from torch_tpu._internal.utils import utils


def _create_bounded_model_inputs(
    example_inputs: Sequence[InputType],
    bounds: SymbolBounds,
) -> tuple[list[torch.Tensor | int], list[Any]]:
  """Specializes dynamic inputs to their static upper bounds and constructs TensorBounds.

  Args:
    example_inputs: Sequence of original example inputs (tensors, SymInts,
      scalars).
    bounds: The SymbolBounds object providing upper bounds for symbolic
      expressions.

  Returns:
    A tuple containing:
      - A list of bounded model inputs with SymInts replaced by their upper
        bounds, and new runtime size placeholders added.
      - A list of TensorBounds (or None) aligned with the bounded model
        inputs.
  """
  bounded_model_inputs = []
  model_input_bounds = []
  for arg in example_inputs:
    if isinstance(arg, torch.Tensor):
      dynamic_dims = [
          dim for dim, d in enumerate(arg.shape) if sym_utils.is_symint(d)
      ]
      if not dynamic_dims:
        bounded_model_inputs.append(arg)
        model_input_bounds.append(None)
        continue
      bounded_shape = [bounds.get_upper_bound(d) for d in arg.shape]
      bounded_model_inputs.append(
          torch.zeros(
              bounded_shape,
              dtype=arg.dtype,
              requires_grad=arg.requires_grad,
          )
      )
      upper_bounds = [bounded_shape[dim] for dim in dynamic_dims]
      bounds_obj = tpu_torch_compile.TensorBounds((dynamic_dims, upper_bounds))
      model_input_bounds.append(bounds_obj)

    elif isinstance(arg, torch.SymInt):
      upper = bounds.get_upper_bound(arg)
      bounded_model_inputs.append(upper)
      model_input_bounds.append(None)
      # Add an input for the runtime size placeholder for the symint.
      bounded_model_inputs.append(torch.tensor(upper, dtype=torch.int32))
      model_input_bounds.append(None)
    else:
      bounded_model_inputs.append(arg)
      model_input_bounds.append(None)

  return bounded_model_inputs, model_input_bounds


def _can_use_pad_real_to_static(
    example_inputs: Sequence[Any],
) -> bool:
  """Returns True if dynamic input tensors can be lowered to pad_real_to_static."""
  libtpu = utils.libtpu_version()
  if libtpu is not None and libtpu < (0, 0, 48):
    return False
  for arg in example_inputs:
    # dtype.itemsize is the element size in bytes (8 bytes = 64-bit,
    # 16 bytes = 128-bit).
    if (
        isinstance(arg, torch.Tensor)
        and sym_utils.has_dynamic_shape(arg)
        and arg.dtype.itemsize in (8, 16)
    ):
      return False
  return True


def _get_inputs_dynamic_flag(example_inputs: Sequence[Any]) -> list[bool]:
  """Returns a list of booleans indicating which model example inputs are dynamic."""
  return [
      isinstance(arg, torch.Tensor) and sym_utils.has_dynamic_shape(arg)
      for arg in example_inputs
  ]


def _extract_minor_to_major(
    parameter_layouts: Sequence[Any] | None,
    parameter_dynamic_flags: Sequence[bool],
) -> tuple[tuple[int, ...], ...]:
  """Extracts minor_to_major layouts from XLA parameter layouts for dynamic inputs."""
  if parameter_layouts is None:
    return ()
  layouts = []
  for is_dynamic, layout in zip(parameter_dynamic_flags, parameter_layouts):
    if is_dynamic and layout is not None:
      layouts.append(tuple(layout[0]))
    else:
      layouts.append(())
  return tuple(layouts)


class DynamicCompiler(compiler.Compiler):
  """Compiler for handling dynamic shapes in torch.compile()."""

  def __init__(
      self,
      compilation_context: compiler.CompilationContext | None = None,
      *,
      debug: bool | compiler.TpuCompileDebug | None = None,
  ):
    """Initializes the DynamicCompiler instance.

    Args:
      compilation_context: A `compiler.CompilationContext` instance used for
        maintaining compilation state.
      debug: Optional TpuCompileDebug container or bool for recording
        compilation debug artifacts. Note: Passing a bool is deprecated; pass a
        TpuCompileDebug container instead.
    """
    if compilation_context is None:
      compilation_context = compiler.CompilationContext()
    super().__init__(compilation_context, debug=debug)
    self.static_compiler = compiler.StaticCompiler(
        compilation_context, debug=debug, use_stablehlo_bounds=True
    )

  def execute_pre_grad_passes(
      self,
      graph_module: torch.fx.GraphModule,
  ) -> None:
    self.static_compiler.execute_pre_grad_passes(graph_module)

    GraphTransformObserver(
        graph_module, "replace_dynamic_output_broadcast_ops_pre_grad"
    ).apply_gm_pass(
        view_ops_passes.ReplaceDynamicOutputBroadcastOpsPreGradPass()
    )

  def _compile_for_symbol_bounds(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      symbol_bounds: SymbolBounds,
      output_sym_shapes: Sequence[OutputSymShape],
      is_fwd: bool,
      module_name: str | None,
      use_pad_real_to_static: bool,
      view_arg_indices: set[int],
  ) -> DynamicTpuCompiledExecutable:
    """Transforms the FX graph and compiles an executable for the given SymbolBounds.

    Clones `graph_module.graph` before running `apply_dynamism_transformations`
    so the caller's `graph_module` is not mutated in place and this helper can
    be called repeatedly for different `SymbolBounds`.
    """
    cloned_graph_module = torch.fx.GraphModule(
        graph_module, copy.deepcopy(graph_module.graph)
    )
    cloned_graph_module.meta = copy.copy(graph_module.meta)
    graph_module = cloned_graph_module

    apply_dynamism_transformations(
        graph_module,
        example_inputs,
        symbol_bounds,
        use_pad_real_to_static=use_pad_real_to_static,
    )

    # TODO: Prevent truncation of log lines.
    logging.debug(
        "[DynamicTpuBackend] Transformed FX Graph: %s",
        LazyString(graph_module.print_readable),
    )

    # Create example inputs for the model executable and construct bounds.
    bounded_model_inputs, model_input_bounds = _create_bounded_model_inputs(
        example_inputs, symbol_bounds
    )

    logging.debug(
        "[DynamicTpuBackend] Bounded model inputs: %s, model input bounds: %s",
        bounded_model_inputs,
        model_input_bounds,
    )

    dynamic_outputs = [out_shape.is_dynamic for out_shape in output_sym_shapes]

    # Create a model executable using the provided example inputs and bounds.
    try:
      default_executable = self.static_compiler(
          graph_module,
          bounded_model_inputs,
          is_fwd=is_fwd,
          bounds=model_input_bounds,
          dynamic_outputs=dynamic_outputs,
          module_name=module_name,
      )
    except (RuntimeError, ValueError) as e:
      raise NotImplementedError(
          "Failed to lower FX graph with dynamic shapes to MLIR with error:"
          f" {e}. Please set torch.compile(..., dynamic=False, ...) and try"
          " again."
      ) from e

    logging.debug(
        "[DynamicTpuBackend] Static Model Executable MLIR: %s",
        LazyString(lambda: str(getattr(default_executable, "mlir_text", None))),  # pyrefly: ignore[bad-argument-type]
    )

    parameter_dynamic_flags = _get_inputs_dynamic_flag(example_inputs)
    default_layout_key = _extract_minor_to_major(
        default_executable.parameter_layouts,  # type: ignore[attr-defined]
        parameter_dynamic_flags=parameter_dynamic_flags,
    )
    logging.info(
        "[DynamicTpuBackend] Compiled default executable with layouts: %s",
        default_layout_key,
    )

    return DynamicTpuCompiledExecutable(
        static_compiler=self.static_compiler,
        graph_module=graph_module,
        example_inputs=example_inputs,
        output_sym_shapes=output_sym_shapes,
        is_fwd=is_fwd,
        default_executable=default_executable,
        default_layout_key=default_layout_key,
        bounded_model_inputs=bounded_model_inputs,
        model_input_bounds=model_input_bounds,
        view_arg_indices=view_arg_indices,
        use_pad_real_to_static=use_pad_real_to_static,
    )

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
      module_name: str | None = None,
  ) -> DynamicTpuCompiledExecutable:
    """Called by AOT Autograd to compile the graph.

    Args:
      graph_module: The FX graph module to be compiled.
      example_inputs: A list of example inputs for the graph module.
      is_fwd: Indicates whether the forward or backward pass is being compiled.
      module_name: The name for the compiled MLIR module.

    Returns:
      A callable `DynamicTpuCompiledExecutable` that wraps the compiled model
      and handles dynamic shapes at runtime.
    """
    # TODO: Prevent truncation of log lines.
    logging.debug(
        "[DynamicTpuBackend] Compile FX Graph: %s",
        LazyString(graph_module.print_readable),
    )

    logging.debug(
        "[DynamicTpuBackend] Example inputs: %s",
        example_inputs,
    )

    # Pre-pass on FX graph and example inputs.
    # Handles view decomposition and constant SymInt conversion.
    view_arg_indices, example_inputs = apply_pre_dynamism_transformations(
        graph_module, example_inputs
    )

    logging.debug(
        "[DynamicTpuBackend] After pre-dynamism transformations, FX Graph: %s",
        LazyString(graph_module.print_readable),
    )

    logging.debug(
        "[DynamicTpuBackend] After pre-dynamism transformations, example"
        " inputs: %s",
        example_inputs,
    )

    # Create a SymInt bounds manager (validates bounds against shape guards).
    sym_bounds_manager = SymBoundsManager(example_inputs)

    # Add bound checks on symints. This forces dynamo to trigger a recompilation
    # if the bound constraint isn't satisfied.
    sym_bounds_manager.add_bound_checks()

    output_sym_shapes = create_output_sym_shapes(graph_module, example_inputs)
    use_pad_real_to_static = _can_use_pad_real_to_static(example_inputs)

    return self._compile_for_symbol_bounds(
        graph_module=graph_module,
        example_inputs=example_inputs,
        symbol_bounds=sym_bounds_manager.get_symbol_bounds(),
        output_sym_shapes=output_sym_shapes,
        is_fwd=is_fwd,
        module_name=module_name,
        use_pad_real_to_static=use_pad_real_to_static,
        view_arg_indices=view_arg_indices,
    )
