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
from collections.abc import Sequence
import dataclasses
from typing import Any

import torch
from torch._dynamo.utils import dynamo_timed
from torch._functorch._aot_autograd.schemas import AOTDispatchCompiler
from torch._inductor.fx_passes import post_grad
from torch._inductor.utils import InputType
from torch._logging import trace_structured
from torch._logging._internal import trace_log
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.passes import graph_transform_observer
from torch.utils import _pytree
from torch_tpu._internal import export as torch_tpu_export
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.fx_passes import mark_activation_checkpoints
from torch_tpu._internal.compile.fx_passes import mark_embedded_constants
from torch_tpu._internal.compile.torch_tpu_compiled_executable import CompiledArtifact
from torch_tpu._internal.compile.torch_tpu_compiled_executable import TorchTpuCompiledExecutable


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
  ):
    """Initializes the Compiler.

    Args:
      compilation_context: An optional CompilationContext object to share state
        across compilations. If None, a new CompilationContext will be default
        constructed.
      debug: Enable debug mode, which may store additional artifacts and log
        more information.
    """
    self.compilation_context = (
        compilation_context
        if compilation_context is not None
        else CompilationContext()
    )
    self._debug = debug

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


def _concretize_tensor_for_placeholder(tensor: torch.Tensor) -> torch.Tensor:
  """Converts a potentially symbolic FakeTensor to a dummy real CPU tensor with concrete shapes and strides."""
  has_symbolic = False
  if hasattr(tensor, "size"):
    has_symbolic = has_symbolic or any(
        isinstance(s, torch.SymInt) for s in tensor.size()
    )
  if hasattr(tensor, "stride"):
    has_symbolic = has_symbolic or any(
        isinstance(s, torch.SymInt) for s in tensor.stride()
    )

  try:
    offset = tensor.storage_offset()
    if isinstance(offset, torch.SymInt):
      has_symbolic = True
  except Exception:
    has_symbolic = True

  if not has_symbolic:
    return tensor

  try:
    concrete_sizes = [int(s) for s in tensor.size()]
    concrete_strides = [int(s) for s in tensor.stride()]
    concrete_offset = int(tensor.storage_offset())
  except Exception as e:
    raise NotImplementedError(
        "TPU backend static compiler cannot compile tensor with unresolved"
        f" dynamic shapes: {tensor}. Please enable dynamism in TpuBackend or"
        " ensure shapes are static."
    ) from e

  if any(s < 0 for s in concrete_strides):
    raise NotImplementedError(
        "Negative strides are not supported in concretization helper yet."
    )

  if len(concrete_sizes) == 0:
    return torch.empty((), dtype=tensor.dtype, device="cpu")

  max_offset = concrete_offset
  for size, stride in zip(concrete_sizes, concrete_strides):
    if size > 0:
      max_offset += (size - 1) * stride

  base_numel = max_offset + 1
  base_tensor = torch.empty((base_numel,), dtype=tensor.dtype, device="cpu")
  view_tensor = base_tensor.as_strided(
      concrete_sizes, concrete_strides, concrete_offset
  )

  if tensor.requires_grad:
    view_tensor.requires_grad_(True)

  return view_tensor


class StaticCompiler(Compiler):
  """Compiler for static shapes.

  This compiler handles the standard compilation flow for FX graphs
  with static shapes, including applying necessary graph passes,
  converting to MLIR, and compiling to a PjRt executable.

  This compiler assumes that the input graph and the provided example_inputs
  will result in static shapes throughout the computation. It refuses to compile
  programs with dynamic shapes by raising an exception.
  """

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
  ) -> CompiledArtifact:
    """Compiles the FX graph module for static shapes.

    This method performs the following steps:
    1.  Applies pre-compilation graph transformations:
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
      example_inputs: A sequence of example input tensors used to guide
        the conversion to MLIR, particularly for determining input
        specifications.
      is_fwd: Indicates whether the forward or backward pass is being compiled.

    Returns:
      A _TorchTpuCompiledExecutable object, which can be called to execute
      the compiled graph on TPU.
    """  # fmt: skip

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
    tracing_enabled = bool(trace_log.handlers)
    if tracing_enabled:
      trace_structured(
          "artifact",
          metadata_fn=lambda: {
              "name": "torchtpu_fx_graph",
              "encoding": "string",
          },
          payload_fn=lambda: graph_module.print_readable(print_output=False),
          expect_trace_id=True,
      )

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
      placeholder_args = []
      for arg in example_inputs:
        if isinstance(arg, torch.Tensor):
          concretized_arg = _concretize_tensor_for_placeholder(arg)
          ph = tpu_torch_compile.placeholder_like(concretized_arg)
          placeholder_args.append(ph)
        else:
          placeholder_args.append(arg)

      with dynamo_timed("torchtpu_fx_to_mlir"):
        exported_mlir = torch_tpu_export.fx_to_mlir(
            graph_module,
            placeholder_args,
            build_mlir_module=(tracing_enabled or self._debug),
        )

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
    )

    if self._debug and mlir_module is not None:
      # Avoid print_readable() as it includes verbose original code lines.
      executable.graph_module_debug_str = str(graph_module.code)
      executable.mlir_text = tpu_torch_compile.serialize_mlir_text(
          mlir_module, enable_debug_info=True
      )

    return executable
