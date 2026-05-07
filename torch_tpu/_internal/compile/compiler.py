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
from collections.abc import Callable, Sequence
import dataclasses
import operator
from typing import Any, TypeAlias

from absl import logging
import torch
from torch._dynamo.utils import dynamo_timed
from torch._functorch._aot_autograd.schemas import AOTDispatchCompiler
from torch._inductor.fx_passes import post_grad
from torch._inductor.output_code import OutputCode
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

_UNSET_GRAPH_HELPER_STR = (
    "FX/MLIR graph is not recorded by default. To record it, use debug mode:"
    " from torch_tpu._internal.compile import TpuBackend; "
    "torch.compile(..., backend=TpuBackend(debug=True))"
)

# Callable type for reconstructing the FX graph outputs from TPU execution
# results. The callable takes:
#   1. original_args: The original positional arguments to the compiled
#   function.
#   2. tpu_outputs: The sequence of torch.Tensor results from the TPU
#   execution.
# It should return a Sequence[Any] representing the final outputs.
_ReconstructFxOutputsFn: TypeAlias = Callable[
    [Sequence[Any], Sequence[torch.Tensor]], Sequence[Any]
]


class CompiledArtifact(abc.ABC, OutputCode):
  """Abstract base class for a compiled executable.

  This class defines the interface for the result of a compilation,
  which can be called like a function and supports pickling/unpickling.
  Pickling support is required for integration into PyTorch-native caching
  mechanisms at the Dynamo/AOT Autograd layers.
  """

  @abc.abstractmethod
  def __call__(self, inputs: Sequence[InputType]) -> Any:
    """Executes the compiled artifact.

    Args:
      inputs: Arguments to be passed to the compiled function.

    Returns:
      The result of executing the compiled code with the provided arguments.
      The type of the result depends on the compiled function.
    """
    pass

  @abc.abstractmethod
  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Enables pickling of the compiled artifact.

    This method is part of Python's pickle protocol. It should return a tuple
    containing two elements:
      - A callable object that will be called to recreate the object.
      - A tuple of arguments to be passed to the callable object.

    Returns:
      A tuple (callable, args_tuple) to be used by the pickle module to
      serialize the object.
    """
    pass


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


def _unpickle_compiled_executable(
    serialized_bytes: bytes,
    reconstruct_fx_outputs_fn: _ReconstructFxOutputsFn | None,
) -> "_TorchTpuCompiledExecutable":
  """Reconstructs a _TorchTpuCompiledExecutable from serialized bytes.

  This function is used as the callable in the tuple returned by
  _TorchTpuCompiledExecutable.__reduce__, enabling the object to be
  unpickled.

  Args:
    serialized_bytes: A byte string containing the serialized
      PjRtLoadedExecutable, originally produced by
      tpu_torch_compile.serialize_executable.
    reconstruct_fx_outputs_fn: An optional callable that transforms the output
      from the TPU executable back to the output structure expected by PyTorch.
      This is the same function that was passed to the
      _TorchTpuCompiledExecutable constructor.

  Returns:
    A deserialized _TorchTpuCompiledExecutable instance.
  """
  return _TorchTpuCompiledExecutable(
      executable=tpu_torch_compile.load_serialized_executable(serialized_bytes),
      reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
  )


class _TorchTpuCompiledExecutable(CompiledArtifact):
  """A callable for a TorchTPU compiled executable.

  This class is returned to dynamo and stored in the dynamo cache. Any time an
  FX graph being traced by dynamo satisfies the guards set up to produce this
  executable, a call to the __call__ method is made.
  """

  def __init__(
      self,
      executable: tpu_torch_compile.PjRtLoadedExecutable,
      reconstruct_fx_outputs_fn: _ReconstructFxOutputsFn | None,
  ):
    """Initializes the compiled executable wrapper.

    Args:
      executable: The result returned by `tpu_torch_compile.compile_mlir`.
      reconstruct_fx_outputs_fn: An optional callable that transforms the output
        from the TPU executable back to the output structure expected by
        PyTorch. The existing use case is to re-insert `None` values into the
        output list if they were filtered out during MLIR conversion. This
        callable should accept two arguments:

          1. original_args: The original positional arguments passed to the
             __call__ method of this executable.
          2. tpu_outputs: The sequence of torch.Tensor results returned by the
             underlying PjRt execution.

        It should return a Sequence[Any] representing
        the final outputs as expected by the FX graph's consumer.
    """  # fmt: skip
    self._executable = executable
    self._reconstruct_fx_outputs_fn = reconstruct_fx_outputs_fn
    self._tensor_arg_indices = None
    self._graph_module_debug_str: str | None = None
    self._mlir_text: str | None = None

  @property
  def graph_module_debug_str(self) -> str | None:
    """The string representation of the FX graph module's code.

    Returns:
        A string containing the Python code of the FX graph module, or None
        if not set. Accessing this property before it's set will log a
        message indicating how to enable debug mode.
    """
    if self._graph_module_debug_str is None:
      logging.warning("%s", _UNSET_GRAPH_HELPER_STR)
    return self._graph_module_debug_str

  @graph_module_debug_str.setter
  def graph_module_debug_str(self, value: str) -> None:
    """Sets the string representation of the FX graph module's code.

    This is typically set only when debug mode is enabled in the Compiler.

    Args:
      value: The string representation of the graph module's code.
    """
    self._graph_module_debug_str = value

  @property
  def mlir_text(self) -> str | None:
    """The MLIR text representation of the compiled module.

    Returns:
        A string containing the MLIR code, or None if not set.
        Accessing this property before it's set will log a message
        indicating how to enable debug mode.
    """
    if self._mlir_text is None:
      logging.warning("%s", _UNSET_GRAPH_HELPER_STR)
    return self._mlir_text

  @mlir_text.setter
  def mlir_text(self, value: str) -> None:
    """Sets the MLIR text representation of the compiled module.

    This is typically set only when debug mode is enabled in the Compiler.

    Args:
      value: The string representation of the MLIR code.
    """
    self._mlir_text = value

  def _take_tensor_args(
      self, args: tuple[Any, ...]
  ) -> tuple[torch.Tensor, ...]:
    """Filters out non-tensor arguments from the top level of the input tuple.

    This method iterates through the provided arguments and returns a new tuple
    containing only the elements that are instances of torch.Tensor. It does
    not recursively traverse nested structures like lists, dicts, or tuples
    within the arguments.

    Args:
      args: A tuple of arguments.

    Returns:
      A tuple containing only the torch.Tensor arguments from the input.
    """
    if self._tensor_arg_indices is None:
      # Pre-compute indices for filtering out non-tensor args once.
      # Assume args will have the same structure for subsequent calls.
      self._tensor_arg_indices = tuple(
          i for i, arg in enumerate(args) if isinstance(arg, torch.Tensor)
      )

    if len(self._tensor_arg_indices) == len(args):
      return args

    if not self._tensor_arg_indices:
      return ()

    filtered = operator.itemgetter(*self._tensor_arg_indices)(args)

    # operator.itemgetter returns a single item if called with a single index,
    # but a tuple if called with multiple indices. We always want a tuple.
    if len(self._tensor_arg_indices) == 1:
      return (filtered,)
    return filtered

  def __call__(
      self, *args: Any, output_shapes: Sequence[list[int]] | None = None
  ) -> Any:
    """Executes the compiled TPU function.

    This method takes the original arguments as inputs, filters out any
    non-Tensor arguments, executes the underlying PjRtLoadedExecutable,
    and then potentially reconstructs the output structure to match what
    the original FX graph's consumer expects.

    Args:
      *args: The positional arguments to the compiled function. These should
        match the structure and types expected by the original FX graph before
        compilation.
      output_shapes: Optional sequence of shapes for the output tensors. This is
        typically used in dynamic shape scenarios.

    Returns:
      The result of the computation. The structure and types of the result
      are made to match the expected output of the original FX graph,
      potentially after being processed by `reconstruct_fx_outputs_fn`.
    """
    actual_output_shapes = output_shapes or []
    # aot_autograd with SerializableAOTDispatchCompiler passes args as a
    # single list: fn([t1, t2, ...]). Unwrap when we detect this pattern.
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
      args = args[0]

    device = torch.accelerator.current_accelerator()
    device_module = getattr(torch, device.type)

    # Update RNG state as the last argument.
    generators = [device_module.default_generators[device.index or 0]] + [
        arg for arg in args if isinstance(arg, torch.Generator)
    ]
    # Map all generators (including user-passed custom generators) to the
    # canonical default generator for their device. This aligns with PyTorch's
    # observed CUDA behavior where graphsafe RNG ops always mutate the global
    # generator's state, making `torch.manual_seed()` work for compiled graphs.
    # TODO(b/501205098): Confirm the intended behavior and right implementation.
    generators = [
        device_module.default_generators[gen.device.index] for gen in generators
    ]

    device_state_tensors = [
        tpu_torch_compile.get_device_state_tensor(gen) for gen in generators
    ]

    executable_args = (
        *self._take_tensor_args(args),
        *device_state_tensors,
    )

    # When output_shapes for user-defined outputs are provided, we need to
    # append the shapes of the device state tensors to match the number of
    # outputs from the executable.
    if actual_output_shapes:
      actual_output_shapes = list(actual_output_shapes) + [
          list(t.shape) for t in device_state_tensors
      ]
    outputs_with_device_state_tensors = tpu_torch_compile.execute(
        self._executable, executable_args, actual_output_shapes
    )
    outputs, device_state_tensors = (
        outputs_with_device_state_tensors[: -len(device_state_tensors)],
        outputs_with_device_state_tensors[-len(device_state_tensors) :],
    )

    # Once the executable has completed, restore the newly updated RNG state
    # back to each generator.
    for gen, device_state_tensor in zip(generators, device_state_tensors):
      tpu_torch_compile.set_device_state_tensor(gen, device_state_tensor)

    if self._reconstruct_fx_outputs_fn is not None:
      outputs = self._reconstruct_fx_outputs_fn(args, outputs)

    return outputs

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Enables pickling of the _TorchTpuCompiledExecutable.

    This method is part of Python's pickle protocol. It returns a tuple
    containing:
      1. A callable (_unpickle_compiled_executable) that can be called to
         recreate the object.
      2. A tuple of arguments to be passed to the callable. These arguments
         include the serialized PjRtLoadedExecutable and the
         reconstruct_fx_outputs_fn.

    Serialization of the executable is necessary for caching compiled
    artifacts within Dynamo.

    Returns:
      A tuple (callable, args_tuple) used by the pickle module to
      serialize the object.
    """  # fmt: skip
    serialized = tpu_torch_compile.serialize_executable(self._executable)
    return (
        _unpickle_compiled_executable,
        (serialized, self._reconstruct_fx_outputs_fn),
    )

  def prepare_for_serialization(self) -> None:
    pass

  def post_compile(
      self,
      example_inputs: Sequence[Any],
      constants: Any,
      graph_kwargs: Any,
  ) -> None:
    pass

  def set_triton_bundle(self, triton_bundle: Any) -> None:
    pass


def _has_activation_checkpoint(graph: torch.fx.Graph) -> bool:
  """Determines whether activation checkpointing was used within the program."""
  ac_ops = graph.find_nodes(
      op="call_function",
      target=torch.ops.higher_order.tag_activation_checkpoint,
  )
  return len(ac_ops) > 0


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
  ) -> _TorchTpuCompiledExecutable:
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
    7.  Wraps the executable in a _TorchTpuCompiledExecutable.
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
      placeholder_args = [
          tpu_torch_compile.placeholder_like(arg)
          if isinstance(arg, torch.Tensor)
          else arg
          for arg in example_inputs
      ]

      with dynamo_timed("torchtpu_fx_to_mlir"):
        exported_mlir = torch_tpu_export.fx_to_mlir(
            graph_module,
            placeholder_args,
        )

    # Emit StableHLO artifact for tlparse when TORCH_TRACE is set.
    if tracing_enabled:
      mlir_text = exported_mlir.serialize_text(enable_debug_info=self._debug)
      trace_structured(
          "artifact",
          metadata_fn=lambda: {
              "name": "torchtpu_stablehlo_graph",
              "encoding": "string",
          },
          payload_fn=lambda: mlir_text,
          expect_trace_id=True,
      )

    with dynamo_timed("torchtpu_pjrt_compile"):
      cached_executable = tpu_torch_compile.compile_mlir(exported_mlir.module)

    executable = _TorchTpuCompiledExecutable(
        executable=cached_executable,
        reconstruct_fx_outputs_fn=exported_mlir.reconstruct_fx_outputs_fn,
    )

    if self._debug:
      # Do not use print_readable() as it includes original lines of code
      # which is too verbose.
      executable.graph_module_debug_str = str(graph_module.code)
      executable.mlir_text = exported_mlir.serialize_text(
          enable_debug_info=True
      )

    return executable
