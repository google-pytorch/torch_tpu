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

import bisect
from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, NamedTuple

from absl import logging
import sympy
import torch
from torch._logging import LazyString
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic import view_decomposition as decompose
from torch_tpu._internal.compile.dynamic.output_sym_shape import OutputSymShape
from torch_tpu._internal.compile.dynamic.symbol_bounds import SymbolBounds
from torch_tpu._internal.benchmarks import xprof_adapter


@dataclasses.dataclass(frozen=True)
class BucketCompilationSpec:
  """Pre-transformed FX graph and bounded inputs for a single SymbolBounds bucket."""

  symbol_bounds: SymbolBounds
  graph_module: torch.fx.GraphModule
  bounded_model_inputs: Sequence[torch.Tensor | int]
  model_input_bounds: Sequence[Any]


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
      example_inputs: Sequence[Any],
      bucket_specs: Sequence[BucketCompilationSpec],
      output_sym_shapes: Sequence[OutputSymShape],
      is_fwd: bool,
      view_arg_indices: set[int],
      use_pad_real_to_static: bool = True,
      module_name: str | None = None,
      num_lookahead_buckets: int | None = None,
  ):
    """Initializes DynamicTpuCompiledExecutable.

    Args:
      static_compiler: Compiler used to compile static MLIR executables on
        bucket or layout cache misses.
      example_inputs: Original example inputs used to build the runtime input
        packing plan.
      bucket_specs: Ordered sequence of pre-transformed BucketCompilationSpec
        objects, one per SymbolBounds bucket.
      output_sym_shapes: Symbolic shape specs for computing runtime shapes of
        output tensors.
      is_fwd: Whether this executable represents a forward pass (True) or
        backward pass (False).
      view_arg_indices: Input argument indices whose view tensors were
        decomposed into base tensors.
      use_pad_real_to_static: Whether inputs are padded in-graph via
        pad_real_to_static instead of a pad subgraph.
      module_name: Optional custom name for the compiled MLIR module.
      num_lookahead_buckets: Number of subsequent buckets to asynchronously
        pre-compile ahead of the current bucket. Defaults to the total number of
        buckets.
    """
    super().__init__()
    self._static_compiler = static_compiler
    self._example_inputs = example_inputs
    self._bucket_specs = list(bucket_specs)
    self._output_sym_shapes = output_sym_shapes
    self._promoted_scalar_output_indices = [
        idx for idx, s in enumerate(output_sym_shapes) if s.is_promoted_scalar
    ]
    self._is_fwd = is_fwd
    self._view_arg_indices = view_arg_indices
    self._module_name = module_name
    self._num_lookahead_buckets = (
        len(self._bucket_specs)
        if num_lookahead_buckets is None
        else num_lookahead_buckets
    )

    # Cache for compiled executables: (bucket_idx, layout_key) -> executable.
    self._model_executables: dict[
        tuple[int, tuple[tuple[int, ...], ...]], Any
    ] = {}

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
      bucket_idx: int,
  ) -> list[torch.Tensor]:
    """Compiles and executes the pad subgraph."""
    if not tensor_args:
      return []

    outputs = [None] * len(tensor_args)
    pad_module_indices = []
    pad_module_args = []
    pad_module_info = []
    pad_module_bounds = []
    dynamic_tensor_bounds = self._bucket_dynamic_tensor_bounds[bucket_idx]

    for i, tensor in enumerate(tensor_args):
      if tpu_torch_compile.is_device_shape_dynamic(tensor):
        # Already dynamic, reuse it
        outputs[i] = tensor
      else:
        # Non-dynamic tensor, pad and make it dynamic
        bounds = dynamic_tensor_bounds[i]
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
    self._bucket_dynamic_tensor_bounds = [
        [b for b in spec.model_input_bounds if b is not None]
        for spec in self._bucket_specs
    ]
    self._symbol_to_input_loc: dict[sympy.Symbol, int] = {}

    target_idx = 0
    pad_idx = 0

    for idx, arg in enumerate(self._example_inputs):
      if arg is None:
        continue

      if isinstance(arg, torch.SymInt):
        self._dynamic_scalar_map.append((idx, target_idx))
        sym_expr = getattr(getattr(arg, "node", None), "expr", None)
        if (
            isinstance(sym_expr, sympy.Symbol)
            and sym_expr not in self._symbol_to_input_loc
        ):
          self._symbol_to_input_loc[sym_expr] = idx
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
    self._symbol_bucket_uppers: list[tuple[int, tuple[int, ...]]] = [
        (
            arg_idx,
            tuple(
                int(spec.symbol_bounds.symbol_ranges[sym].upper)
                for spec in self._bucket_specs
            ),
        )
        for sym, arg_idx in self._symbol_to_input_loc.items()
        if all(
            sym in spec.symbol_bounds.symbol_ranges
            for spec in self._bucket_specs
        )
    ]

  def _get_bucket_index(self, args: Sequence[Any]) -> int:
    """Returns the index of the first SymbolBounds bucket that fits `args`."""
    num_buckets = len(self._bucket_specs)
    if num_buckets <= 1 or not self._symbol_bucket_uppers or not args:
      return 0

    bucket_idx = 0
    for arg_idx, uppers in self._symbol_bucket_uppers:
      idx = bisect.bisect_left(uppers, int(args[arg_idx]))
      if idx > bucket_idx:
        bucket_idx = idx

    if bucket_idx >= num_buckets:
      runtime_symbol_vals: dict[sympy.Symbol, int] = {
          sym: int(args[arg_idx])
          for sym, arg_idx in self._symbol_to_input_loc.items()
      }
      raise ValueError(
          f"Runtime symbol values {runtime_symbol_vals} exceed all compiled"
          f" buckets: {[spec.symbol_bounds for spec in self._bucket_specs]}"
      )

    return bucket_idx

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

  def _compile_bucket(
      self,
      bucket_idx: int,
      argument_layouts: Sequence[Sequence[int]] | None = None,
  ) -> compiler.CompiledArtifact:
    """Compiles the pre-transformed FX graph for `bucket_idx` and `argument_layouts`."""
    with xprof_adapter.TraceMe(
        "DynamicTpuCompiledExecutable._compile_bucket",
        bucket_idx=bucket_idx,
    ):
      spec = self._bucket_specs[bucket_idx]
      return self._static_compiler(
          spec.graph_module,
          spec.bounded_model_inputs,
          is_fwd=self._is_fwd,
          bounds=spec.model_input_bounds,
          argument_layouts=argument_layouts,
          dynamic_outputs=self._dynamic_outputs,
          module_name=self._module_name,
      )

  def get_executable(
      self,
      layout_key: tuple[tuple[int, ...], ...],
      argument_layouts: Sequence[Sequence[int]] | None = None,
      bucket_idx: int = 0,
  ) -> Any:
    """Returns the resolved compiled executable for `(bucket_idx, layout_key)`, compiling if needed."""
    cache_key = (bucket_idx, layout_key)
    is_new_or_unresolved = (
        cache_key not in self._model_executables
        or not getattr(self._model_executables[cache_key], "is_resolved", True)
    )
    submitted_lookahead_keys: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    try:
      if cache_key not in self._model_executables:
        logging.debug(
            "[DynamicTpuBackend] Compilation cache miss for bucket %d,"
            " layouts: %s. Compiling new executable.",
            bucket_idx,
            layout_key,
        )
        self._model_executables[cache_key] = self._compile_bucket(
            bucket_idx,
            argument_layouts=argument_layouts,
        )

      if getattr(self._static_compiler, "_async_compile", False):
        # Enqueue lookahead bucket compilations before blocking on
        # `executable.resolve()` for the current bucket so that subsequent
        # buckets compile in parallel on background threads while the current
        # bucket is resolving or executing.
        max_lookahead_idx = min(
            bucket_idx + 1 + max(0, self._num_lookahead_buckets),
            len(self._bucket_specs),
        )
        for lookahead_idx in range(bucket_idx + 1, max_lookahead_idx):
          lookahead_cache_key = (lookahead_idx, layout_key)
          if lookahead_cache_key not in self._model_executables:
            logging.debug(
                "[DynamicTpuBackend] Issuing async lookahead compilation for"
                " bucket %d, layouts: %s.",
                lookahead_idx,
                layout_key,
            )
            self._model_executables[lookahead_cache_key] = self._compile_bucket(
                lookahead_idx,
                argument_layouts=argument_layouts,
            )
            submitted_lookahead_keys.append(lookahead_cache_key)

      if is_new_or_unresolved:
        executable = self._model_executables[cache_key]
        if hasattr(executable, "resolve"):
          executable.resolve()
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
    except (RuntimeError, ValueError) as e:
      self._model_executables.pop(cache_key, None)
      for lookahead_key in submitted_lookahead_keys:
        self._model_executables.pop(lookahead_key, None)
      raise NotImplementedError(
          "Failed to lower FX graph with dynamic shapes to MLIR with error:"
          f" {e}. Please set torch.compile(..., dynamic=False, ...) and try"
          " again."
      ) from e

    return self._model_executables[cache_key]

  def __call__(self, *args: Any) -> Any:
    logging.debug("[DynamicTpuBackend] Execute Model")
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
      args = tuple(args[0])

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

    bucket_idx = self._get_bucket_index(args)
    logging.debug(
        "[DynamicTpuBackend] Selected bucket %d (%s)",
        bucket_idx,
        self._bucket_specs[bucket_idx].symbol_bounds,
    )

    tensor_args = self._get_dynamic_input_tensors(args)

    # Run pad subgraph first if needed (for libtpu < 0.0.49 or 64-bit/128-bit
    # tensors)
    if self._use_pad_subgraph:
      pad_outputs = self._compile_and_execute_pad_subgraph(
          tensor_args, bucket_idx
      )
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

    executable = self.get_executable(
        layout_key, argument_layouts, bucket_idx=bucket_idx
    )

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
