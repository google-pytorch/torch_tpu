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

"""Core components for the TorchTPU compilation flow.

This module defines the essential classes and interfaces used within the
torch.compile backend for TorchTPU. It includes the building blocks
for defining different compilation strategies and for managing the state and
artifacts produced during compilation.
"""

import abc
import collections
from collections.abc import Sequence
import concurrent.futures
import dataclasses
import os
from typing import Any

import torch
from torch._dynamo.utils import dynamo_timed
from torch._functorch._aot_autograd.schemas import AOTDispatchCompiler
from torch._inductor.fx_passes import post_grad
from torch._inductor.fx_passes import reinplace
from torch._inductor.utils import InputType
from torch._logging import trace_structured
from torch._logging._internal import trace_log
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.passes import graph_transform_observer
from torch.utils import _pytree
from torch_tpu._internal import export as torch_tpu_export
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.fx_passes import mark_activation_checkpoints
from torch_tpu._internal.compile.fx_passes import mark_embedded_constants
from torch_tpu._internal.compile.torch_tpu_compiled_executable import AsyncCompiledArtifact
from torch_tpu._internal.compile.torch_tpu_compiled_executable import CompiledArtifact
from torch_tpu._internal.compile.torch_tpu_compiled_executable import NoOpCompiledArtifact
from torch_tpu._internal.compile.torch_tpu_compiled_executable import TorchTpuCompiledExecutable
from torch_tpu._internal.profiler import xprof_adapter


@dataclasses.dataclass
class CompilationContext:
  """Compilation state for a single graph module.

  This class is used to pass information and state between different
  stages of the compilation process, including across the forward and
  backward pass compilations.
  """


class Compiler(abc.ABC, AOTDispatchCompiler):
  """Abstract base class for a compiler.

  Defines the interface for compiling an FX graph module, including
  hooks for pre-AOT Autograd passes and the main compilation logic.

  Attributes:
    compilation_context: The context object holding compilation state.
  """

  def __init__(
      self,
      compilation_context: CompilationContext | None = None,
      *,
      debug: bool = False,
      use_stablehlo_bounds: bool = False,
  ):
    """Initializes the Compiler.

    Args:
      compilation_context: An optional CompilationContext object to share state
        across compilations. If None, a new CompilationContext will be default
        constructed.
      debug: Enable debug mode, which may store additional artifacts and log
        more information.
      use_stablehlo_bounds: Whether to use StableHLO bounds during compilation
        for dynamic inputs.
    """
    self.compilation_context = (
        compilation_context
        if compilation_context is not None
        else CompilationContext()
    )
    self._debug = debug
    self._use_stablehlo_bounds = use_stablehlo_bounds

  def execute_pre_grad_passes(
      self,
      graph_module: torch.fx.GraphModule,
  ) -> None:
    """Executes graph passes on the FX graph module before AOT Autograd.

    This method is a hook to apply any necessary transformations to the
    graph module after it has been traced and before it is processed by
    AOT Autograd. Subclasses can override this method to add custom
    graph passes.

    Args:
      graph_module: The FX graph module to process.
    """
    pass

  @abc.abstractmethod
  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
  ) -> CompiledArtifact:
    """Compiles the FX graph module into a callable artifact.

    This method takes the FX graph module produced by Dynamo/AOT Autograd
    and compiles it into an executable format. The exact compilation
    process will depend on the specific compiler implementation.

    Args:
      graph_module: The FX graph module to compile.
      example_inputs: A sequence of example input tensors that can be used to
        guide the compilation process (e.g., for shape inference).
      is_fwd: Indicates whether the forward or backward pass is being compiled.

    Returns:
      A CompiledArtifact object, which is a callable representation of the
      compiled graph.
    """
    pass


def has_dynamic_symints(args: Any) -> bool:
  """Checks whether truly dynamic SymInt inputs or shapes exist.

  A SymInt whose expression has been fully concretized (no free symbols)
  is treated as static.

  Args:
    args: example program inputs provided for compilation.

  Returns:
    True if any args or tensor dimensions have unresolved symbolic variables.
  """
  flat_args, _ = _pytree.tree_flatten(args)

  def _is_dynamic_symint(val: Any) -> bool:
    return isinstance(val, torch.SymInt) and bool(val.node.expr.free_symbols)

  for arg in flat_args:
    if _is_dynamic_symint(arg):
      return True

    if isinstance(arg, torch.Tensor):
      if any(_is_dynamic_symint(s) for s in arg.shape):
        return True

  return False


def _is_tracing_enabled() -> bool:
  """Returns whether structured trace logging handlers are enabled.

  When enabled, FX graph and MLIR artifact representations are logged
  for debugging and structured trace visualization.
  """
  return bool(trace_log.handlers)


def _create_storage_aware_placeholders(
    example_inputs: Sequence[Any],
    bounds: Sequence[Any] | None = None,
    enabled: bool = True,
) -> list[Any]:
  """Creates placeholder arguments preserving shared storage and view relationships."""
  # Group tensor inputs by their underlying UntypedStorage identity.
  storage_groups = collections.defaultdict(list)
  for i, arg in enumerate(example_inputs):
    if isinstance(arg, torch.Tensor):
      storage_groups[arg.untyped_storage()].append((i, arg))

  # Initialize with example_inputs to automatically preserve non-tensor arguments.
  placeholders = list(example_inputs)

  for storage, entries in storage_groups.items():
    # Fall back to 1:1 placeholders if:
    # 1. Storage-aware placeholders are disabled.
    # 2. Only a single tensor references this storage.
    # 3. Mixed dtypes exist in the storage group (reinterpreted views).
    # 4. Any tensor has dynamic bounds (dynamic views on shared storage not yet supported).
    has_mixed_dtypes = len({arg.dtype for _, arg in entries}) > 1
    has_dynamic_bounds = bounds is not None and any(
        bounds[i] is not None for i, _ in entries
    )

    if (
        not enabled
        or len(entries) == 1
        or has_mixed_dtypes
        or has_dynamic_bounds
    ):
      for i, arg in entries:
        arg_bounds = bounds[i] if bounds is not None else None
        if arg_bounds is not None:
          placeholders[i] = tpu_torch_compile.dynamic_placeholder(
              arg.shape, arg.dtype, arg_bounds, arg.requires_grad
          )
        else:
          placeholders[i] = tpu_torch_compile.placeholder_like(arg)
    else:
      _, primary_arg = entries[0]
      elem_size = primary_arg.element_size()
      storage_numel = storage.size() // elem_size if elem_size > 0 else 0

      # Check if one of the tensors already spans the full storage contiguously.
      base_tensor = next(
          (
              arg
              for _, arg in entries
              if arg.is_contiguous()
              and arg.storage_offset() == 0
              and arg.numel() == storage_numel
          ),
          None,
      )
      if base_tensor is not None:
        base_ph = tpu_torch_compile.placeholder_like(base_tensor)
      else:
        # Otherwise, synthesize a 1D flat placeholder matching total storage elements.
        flat_meta = torch.empty(
            (storage_numel,),
            dtype=primary_arg.dtype,
            device=torch.device("meta"),
        )
        base_ph = tpu_torch_compile.placeholder_like(flat_meta)

      for i, arg in entries:
        ph = base_ph.as_strided(
            size=arg.shape,
            stride=arg.stride(),
            storage_offset=arg.storage_offset(),
        )
        if arg.requires_grad:
          ph.requires_grad_(True)
        placeholders[i] = ph

  return placeholders


class StaticCompiler(Compiler):
  """Compiler for static shapes.

  This compiler handles the standard compilation flow for FX graphs
  with static shapes, including applying necessary graph passes,
  converting to MLIR, and compiling to a PjRt executable.

  This compiler assumes that the input graph and the provided example_inputs
  will result in static shapes throughout the computation. It refuses to compile
  programs with dynamic shapes by raising an exception.
  """

  _async_compile_executor = concurrent.futures.ThreadPoolExecutor(
      max_workers=8, thread_name_prefix="tpu_async_compile"
  )

  def __init__(self, *args, async_compile: bool = False, **kwargs):
    """Initializes a StaticCompiler instance.

    Args:
      *args: Positional arguments forwarded to the base Compiler class.
      async_compile: Whether to offload MLIR lowering and PjRt compilation to a
        background thread pool, returning an AsyncCompiledArtifact proxy.
      **kwargs: Keyword arguments forwarded to the base Compiler class.
    """
    super().__init__(*args, **kwargs)
    self._async_compile = async_compile

  def _compile(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
      bounds: Sequence[Any] | None = None,
      argument_layouts: Sequence[Sequence[int]] | None = None,
      dynamic_outputs: Sequence[bool] | None = None,
      donated_inputs: Sequence[int] | None = None,
  ) -> CompiledArtifact:
    with xprof_adapter.TraceMe("StaticCompiler._compile"):
      tracing_enabled = _is_tracing_enabled()

      # AOT autograd will trace the model with FakeTensorMode enabled. This
      # converts all tensors to FakeTensors which do not have valid storage to
      # avoid computation.

      # TorchTPU tracing requires all tensors to have valid storage, so is not
      # compatible with FakeTensorMode. This applies to placeholder tensors as
      # well, fake-tensor will attempt to convert our placeholder tensors into
      # FakeTensors before continuing eager tracing.
      with unset_fake_temporarily():
        # Convert example_inputs to placeholders. This is done to:
        # (1) Prevent the unintentional compilation of deferred operations.
        #     placeholders will error.
        # (2) Act as TPU-compatible FakeTensors, so that tracing does not depend
        #     on tensor data.
        placeholder_args = _create_storage_aware_placeholders(
            example_inputs,
            bounds,
            enabled=os.environ.get("TORCHTPU_STORAGE_AWARE_PLACEHOLDERS", "0")
            == "1",
        )

        if argument_layouts is None:
          extracted_layouts = []
          for val in example_inputs:
            if isinstance(val, torch.Tensor):
              if val.device.type == "tpu" and not isinstance(val, FakeTensor):
                layout = tpu_torch_compile.get_device_layout_if_materialized(
                    val
                )
                if layout is not None:
                  extracted_layouts.append(layout[0])
                else:
                  extracted_layouts.append([])
              else:
                extracted_layouts.append([])
          if any(extracted_layouts):
            argument_layouts = extracted_layouts

        with dynamo_timed("torchtpu_fx_to_mlir"):
          exported_mlir = torch_tpu_export.fx_to_mlir(
              graph_module,
              placeholder_args,
              build_mlir_module=(tracing_enabled or self._debug),
              use_stablehlo_bounds=self._use_stablehlo_bounds,
              argument_layouts=argument_layouts,  # pyrefly: ignore[bad-argument-type]
              dynamic_outputs=dynamic_outputs,
              donated_inputs=donated_inputs,
          )

      if exported_mlir.is_noop:
        # The FX graph produced no computed output tensors (e.g. a
        # fullgraph=False seam between two graph breaks). There is nothing to
        # compile; return a callable that reconstructs the graph's (all-None /
        # passthrough) output and runs nothing on device, rather than compiling a
        # trivial executable.
        return NoOpCompiledArtifact(exported_mlir.reconstruct_fx_outputs_fn)

      mlir_module = exported_mlir.module

      # Emit StableHLO artifact for tlparse when TORCH_TRACE is set.
      if tracing_enabled and mlir_module is not None:
        mlir_text = tpu_torch_compile.serialize_mlir_text(
            mlir_module, enable_debug_info=self._debug
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "torchtpu_stablehlo_graph",
                "encoding": "string",
            },
            payload_fn=lambda: mlir_text,
            expect_trace_id=True,
        )

      executable = TorchTpuCompiledExecutable(
          executable=exported_mlir.executable,
          reconstruct_fx_outputs_fn=exported_mlir.reconstruct_fx_outputs_fn,
          updates_default_generator_state=exported_mlir.updates_default_generator_state,
          dynamic_outputs=exported_mlir.dynamic_outputs,
          unique_output_indices=exported_mlir.unique_output_indices,
      )

      if self._debug and mlir_module is not None:
        # Avoid print_readable() as it includes verbose original code lines.
        executable.graph_module_debug_str = str(graph_module.code)
        executable.mlir_text = tpu_torch_compile.serialize_mlir_text(
            mlir_module, enable_debug_info=True
        )

      return executable

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
      bounds: Sequence[Any] | None = None,
      argument_layouts: Sequence[Sequence[int]] | None = None,
      dynamic_outputs: Sequence[bool] | None = None,
      donated_inputs: Sequence[int] | None = None,
  ) -> CompiledArtifact:
    """Compiles the FX graph module for static shapes.

    This method performs the steps below. Steps 4-8 are executed asynchronously
    if `StaticCompiler` was initialized with `async_compile=True`:
    1.  Applies pre-compilation graph transformations:
        -   Prepares auto functionalized ops and auto-detects donated inputs.
        -   Decomposes auto functionalized operations.
        -   Marks embedded constants.
    2.  Lints and recompiles the graph module.
    3.  Emits FX graph artifacts if tracing is enabled.
    4.  Converts the FX graph to MLIR StableHLO, using placeholder tensors
        derived from example_inputs. This step is done under an
        unset_fake_temporarily context to ensure tensor storage.
    5.  Emits MLIR artifacts if tracing is enabled.
    6.  Compiles the MLIR module into a PjRtLoadedExecutable.
    7.  Wraps the executable in a TorchTpuCompiledExecutable.
    8.  Stores debug information (graph code, MLIR text) in the
        executable if debug mode is enabled.

    Args:
      graph_module: The FX graph module to compile.
      example_inputs: A sequence of example input tensors used to guide the
        conversion to MLIR, particularly for determining input specifications.
      is_fwd: Indicates whether the forward or backward pass is being compiled.
      bounds: Optional sequence of TensorBounds for dynamic inputs.
      argument_layouts: Optional sequence of argument layouts for inputs.
      dynamic_outputs: Optional sequence of booleans indicating dynamic outputs.
      donated_inputs: Optional sequence of flat tensor indices to donate. This
        is an "internal" feature only available to StaticCompiler intended to be
        used directly after tracing with make_fx. TODO(b/545738245): Investigate
        doing this automatically.

    Returns:
      A `CompiledArtifact` object (`TorchTpuCompiledExecutable` if
      `async_compiled=True`, otherwise `AsyncCompiledArtifact`), which can be
      called to execute the compiled graph on TPU.
    """  # fmt: skip
    # Run PyTorch Inductor's reinplace pass to determine which mutable ops can
    # safely execute in-place without defensive copies.
    graph_transform_observer.GraphTransformObserver(
        graph_module, "reinplace_inplaceable_ops"
    ).apply_graph_pass(reinplace.reinplace_inplaceable_ops_core)

    # Decompose auto functionalized ops, we need to explicitly do this because
    # the default behaviour inserts flatten and unflatten ops at the boundaries
    # which then blocks buffer donation.
    # TODO(b/491716758): Replace with our own fork.
    graph_transform_observer.GraphTransformObserver(
        graph_module, "decompose_auto_functionalized"
    ).apply_graph_pass(post_grad.decompose_auto_functionalized)
    graph_transform_observer.GraphTransformObserver(
        graph_module, "mark_embedded_constants"
    ).apply_graph_pass(mark_embedded_constants.apply)
    if not is_fwd:
      graph_transform_observer.GraphTransformObserver(
          graph_module, "mark_activation_checkpoints"
      ).apply_graph_pass(mark_activation_checkpoints.apply)

    graph_module.graph.lint()
    graph_module.recompile()

    # Emit FX graph artifact for tlparse when TORCH_TRACE is set.
    if _is_tracing_enabled():
      trace_structured(
          "artifact",
          metadata_fn=lambda: {
              "name": "torchtpu_fx_graph",
              "encoding": "string",
          },
          payload_fn=lambda: graph_module.print_readable(print_output=False),
          expect_trace_id=True,
      )

    if self._async_compile:
      future = StaticCompiler._async_compile_executor.submit(
          self._compile,
          graph_module,
          example_inputs,
          is_fwd,
          bounds,
          argument_layouts,
          dynamic_outputs,
          donated_inputs,
      )
      return AsyncCompiledArtifact(future)
    else:
      return self._compile(
          graph_module,
          example_inputs,
          is_fwd,
          bounds,
          argument_layouts,
          dynamic_outputs,
          donated_inputs,
      )
