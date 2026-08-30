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

"""Compiled artifact and executable definitions for TorchTPU."""

import abc
from collections.abc import Callable, Sequence
import concurrent.futures
import operator
import threading
from typing import Any, TypeAlias

from absl import logging
import torch
from torch._inductor.output_code import OutputCode
from torch._inductor.utils import InputType
from torch_tpu._internal.compile import tpu_torch_compile

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

  def resolve(self) -> None:
    """Waits for any pending background compilation to complete."""
    pass

  @property
  def is_resolved(self) -> bool:
    """Returns True if the compilation is complete and resolved."""
    return True

  def updates_default_generator_state(self) -> bool:
    """Returns whether the executable updates the default generator state."""
    return False


def _unpickle_compiled_executable(
    serialized_bytes: bytes,
    reconstruct_fx_outputs_fn: _ReconstructFxOutputsFn | None,
    updates_default_generator_state: bool,
    dynamic_outputs: Sequence[bool] | None = None,
    unique_output_indices: Sequence[int] | None = None,
) -> "TorchTpuCompiledExecutable":
  """Reconstructs a TorchTpuCompiledExecutable from serialized bytes.

  This function is used as the callable in the tuple returned by
  TorchTpuCompiledExecutable.__reduce__, enabling the object to be
  unpickled.

  Args:
    serialized_bytes: A byte string containing the serialized
      PjRtLoadedExecutable, originally produced by
      tpu_torch_compile.serialize_executable.
    reconstruct_fx_outputs_fn: An optional callable that transforms the output
      from the TPU executable back to the output structure expected by PyTorch.
      This is the same function that was passed to the
      TorchTpuCompiledExecutable constructor.
    updates_default_generator_state: Whether the executable updates the default
      generator state.
    dynamic_outputs: Optional sequence of booleans indicating whether each
      output tensor is dynamic.
    unique_output_indices: Optional sequence of original output indices
      corresponding to deduplicated outputs.

  Returns:
    A deserialized TorchTpuCompiledExecutable instance.
  """
  return TorchTpuCompiledExecutable(
      executable=tpu_torch_compile.load_serialized_executable(serialized_bytes),
      reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
      updates_default_generator_state=updates_default_generator_state,
      dynamic_outputs=dynamic_outputs,
      unique_output_indices=unique_output_indices,
  )


class TorchTpuCompiledExecutable(CompiledArtifact):
  """A callable for a TorchTPU compiled executable.

  This class is returned to dynamo and stored in the dynamo cache. Any time an
  FX graph being traced by dynamo satisfies the guards set up to produce this
  executable, a call to the __call__ method is made.
  """

  def __init__(
      self,
      executable: tpu_torch_compile.PjRtLoadedExecutable,
      reconstruct_fx_outputs_fn: _ReconstructFxOutputsFn | None,
      updates_default_generator_state: bool,
      dynamic_outputs: Sequence[bool] | None = None,
      unique_output_indices: Sequence[int] | None = None,
  ):
    """Initializes the compiled executable wrapper.

    Args:
      executable: The result returned by `tpu_torch_compile.compile_mlir`.
      reconstruct_fx_outputs_fn: An optional callable that transforms the output
        from the TPU executable back to the output structure expected by
        PyTorch. The existing use case is to re-insert `None` values into the
        output list if they were filtered out during MLIR conversion. This
        callable should accept two arguments:  1. original_args: The original
        positional arguments passed to the __call__ method of this executable.
        2. tpu_outputs: The sequence of torch.Tensor results returned by the
        underlying PjRt execution.  It should return a Sequence[Any]
        representing the final outputs as expected by the FX graph's consumer.
      updates_default_generator_state: Whether the executable updates the
        default generator state.
      dynamic_outputs: Optional sequence of booleans indicating whether each
        output tensor is dynamic.
      unique_output_indices: Optional sequence of original output indices
        corresponding to deduplicated outputs.
    """  # fmt: skip
    self._executable = executable
    self._reconstruct_fx_outputs_fn = reconstruct_fx_outputs_fn
    self._tensor_arg_indices = None
    self._graph_module_debug_str: str | None = None
    self._mlir_text: str | None = None
    self._updates_default_generator_state: bool = (
        updates_default_generator_state
    )
    self._dynamic_outputs: Sequence[bool] | None = dynamic_outputs
    self._unique_output_indices: Sequence[int] | None = unique_output_indices
    self._has_dynamic_outputs: bool = (
        any(dynamic_outputs) if dynamic_outputs is not None else False
    )
    if self._has_dynamic_outputs and dynamic_outputs is not None:
      self._cached_dynamic_output_shapes: (
          list[tpu_torch_compile.OutputShape] | None
      ) = [
          tpu_torch_compile.OutputShape(is_dynamic=is_dyn)
          for is_dyn in dynamic_outputs
      ]
    else:
      self._cached_dynamic_output_shapes = None

  @property
  def unique_output_indices(self) -> Sequence[int] | None:
    return self._unique_output_indices

  def updates_default_generator_state(self) -> bool:
    return self._updates_default_generator_state

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

  @property
  def parameter_layouts(self) -> list[Any]:
    """Returns the parameter layouts expected by the executable."""
    return self._executable.get_parameter_layouts()

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
      self,
      *args: Any,
      output_shapes: Sequence[tpu_torch_compile.OutputShape] | None = None,
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
      output_shapes: Optional sequence of OutputShape objects for the output
        tensors. This is typically used in dynamic shape scenarios.

    Returns:
      The result of the computation. The structure and types of the result
      are made to match the expected output of the original FX graph,
      potentially after being processed by `reconstruct_fx_outputs_fn`.
    """
    # aot_autograd with SerializableAOTDispatchCompiler passes args as a
    # single list: fn([t1, t2, ...]). Unwrap when we detect this pattern.
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
      args = args[0]  # pyrefly: ignore[bad-assignment]

    device = torch.accelerator.current_accelerator()
    device_module = getattr(torch, device.type)

    # Update RNG state as the last argument.
    generators = [arg for arg in args if type(arg) is torch.Generator]  # pylint: disable=unidiomatic-typecheck
    if self._updates_default_generator_state:
      generators.append(device_module.default_generators[device.index or 0])

    # Map all generators (including user-passed custom generators) to the
    # canonical default generator for their device. This aligns with PyTorch's
    # observed CUDA behavior where graphsafe RNG ops always mutate the global
    # generator's state, making `torch.manual_seed()` work for compiled graphs.
    # TODO(b/501205098): Confirm the intended behavior and right implementation.
    generators = [
        device_module.default_generators[gen.device.index] for gen in generators
    ]

    # Lock all generators to prevent race conditions when updating their states.
    # TODO: b/501205098 - Generators are not locked during export.py. Address
    # this in a follow-up.
    with tpu_torch_compile.MultiGeneratorLocker(generators):
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
      if output_shapes:
        executable_output_shapes = [
            s
            if isinstance(s, tpu_torch_compile.OutputShape)
            else tpu_torch_compile.OutputShape(
                dimensions=s[0] if isinstance(s, tuple) else s,
                is_dynamic=s[1] if isinstance(s, tuple) else False,
            )
            for s in output_shapes
        ] + [
            tpu_torch_compile.OutputShape(
                dimensions=list(t.shape), is_dynamic=False
            )
            for t in device_state_tensors
        ]
      elif self._has_dynamic_outputs and self._cached_dynamic_output_shapes:
        if device_state_tensors:
          executable_output_shapes = self._cached_dynamic_output_shapes + [
              tpu_torch_compile.OutputShape(
                  dimensions=list(t.shape), is_dynamic=False
              )
              for t in device_state_tensors
          ]
        else:
          executable_output_shapes = self._cached_dynamic_output_shapes
      else:
        executable_output_shapes = []

      outputs_with_device_state_tensors = tpu_torch_compile.execute(
          self._executable,
          executable_args,
          executable_output_shapes,
      )

      if not device_state_tensors:
        outputs = outputs_with_device_state_tensors
        updated_device_state_tensors = []
      else:
        outputs, updated_device_state_tensors = (
            outputs_with_device_state_tensors[: -len(device_state_tensors)],
            outputs_with_device_state_tensors[-len(device_state_tensors) :],
        )

      # Once the executable has completed, restore the newly updated RNG state
      # back to each generator.
      for gen, device_state_tensor in zip(
          generators, updated_device_state_tensors
      ):
        tpu_torch_compile.set_device_state_tensor(gen, device_state_tensor)

    if self._reconstruct_fx_outputs_fn is not None:
      outputs = self._reconstruct_fx_outputs_fn(args, outputs)

    return outputs

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Enables pickling of the TorchTpuCompiledExecutable.

    This method is part of Python's pickle protocol. It returns a tuple
    containing:
      1. A callable (_unpickle_compiled_executable) that can be called to
         recreate the object.
      2. A tuple of arguments to be passed to the callable. These arguments
         include the serialized PjRtLoadedExecutable, the
         reconstruct_fx_outputs_fn, the updates_default_generator_state flag,
         and the dynamic_outputs.

    Serialization of the executable is necessary for caching compiled
    artifacts within Dynamo.

    Returns:
      A tuple (callable, args_tuple) used by the pickle module to
      serialize the object.
    """  # fmt: skip
    serialized = tpu_torch_compile.serialize_executable(self._executable)
    return (
        _unpickle_compiled_executable,
        (
            serialized,
            self._reconstruct_fx_outputs_fn,
            self._updates_default_generator_state,
            self._dynamic_outputs,
            self._unique_output_indices,
        ),
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

  def fingerprint(self) -> str:
    return self._executable.fingerprint_executable()


class AsyncCompiledArtifact(CompiledArtifact):
  """A proxy for TorchTpuCompiledExecutable that compiles asynchronously.

  This class allows MLIR compilation to happen concurrently in the background.
  It transparently blocks to resolve the compilation when missing attributes
  or execution methods are accessed.
  """

  # Specifies that the callable receives inputs as a single list and prevents
  # premature resolution.
  _boxed_call: bool = True
  # Disables graph cache lookup and prevents premature resolution.
  _fx_graph_cache_key: Any = None
  # Indicates this is not a fallback callable and prevents premature
  # resolution.
  _is_fallback: bool = False

  def __init__(self, future: concurrent.futures.Future[CompiledArtifact]):
    super().__init__()
    self._future = future
    self._resolved = False
    self._lock = threading.Lock()

  def prepare_for_serialization(self) -> None:
    pass

  def post_compile(self, *args: Any, **kwargs: Any) -> None:
    pass

  def set_triton_bundle(self, triton_bundle: Any) -> None:
    pass

  def _resolve(self) -> None:
    """Blocks on the future and swaps the dict with the compiled executable."""
    if not self._resolved:
      with self._lock:
        if not self._resolved:
          real_artifact = self._future.result()
          self.__dict__.update(real_artifact.__dict__)
          self.__class__ = real_artifact.__class__  # pyrefly: ignore[bad-argument-type]
          self._resolved = True

  def resolve(self) -> None:
    """Waits for any pending background compilation to complete."""
    self._resolve()

  @property
  def is_resolved(self) -> bool:
    """Returns True if the background compilation has finished and resolved."""
    return self._resolved

  def __getattr__(self, name: str) -> Any:
    """Resolves the background compilation when an unknown attr is accessed."""
    # Do not resolve on Python introspection / magic attributes
    if name.startswith("__"):
      raise AttributeError(
          f"{type(self).__name__!s} object has no attribute {name!r}"
      )
    self._resolve()
    return getattr(self, name)

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Resolves the background compilation and executes the real callable."""
    self._resolve()
    return self(*args, **kwargs)

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Resolves the background compilation and serializes the executable."""
    self._resolve()
    return self.__reduce__()


def _unpickle_noop_compiled_artifact(
    reconstruct_fx_outputs_fn: _ReconstructFxOutputsFn | None,
) -> "NoOpCompiledArtifact":
  """Reconstructs a NoOpCompiledArtifact when unpickling (Dynamo/AOT cache)."""
  return NoOpCompiledArtifact(reconstruct_fx_outputs_fn)


class NoOpCompiledArtifact(CompiledArtifact):
  """Callable for an FX graph that produces no computed output tensors.

  A ``torch.compile(fullgraph=False)`` partition can yield a segment that
  threads a live input through but contains no traceable ops -- e.g. a seam
  between two graph breaks, ``def forward(x): return ()``. There is nothing to
  lower to MLIR, so rather than compile a trivial executable the backend returns
  this object. When Dynamo invokes it, it reconstructs the graph's original
  output structure (all ``None`` and/or input passthroughs) and runs nothing on
  device.
  """

  def __init__(
      self,
      reconstruct_fx_outputs_fn: _ReconstructFxOutputsFn | None,
  ) -> None:
    self._reconstruct_fx_outputs_fn = reconstruct_fx_outputs_fn

  def __call__(
      self, *args: Any, output_shapes: Sequence[list[int]] | None = None
  ) -> Any:
    del output_shapes  # Unused
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
      args = args[0]  # pyrefly: ignore[bad-assignment]
    if self._reconstruct_fx_outputs_fn is None:
      return ()
    # No compiled result tensors: reconstruct the FX output structure (Nones
    # and input passthroughs) from an empty output list.
    return self._reconstruct_fx_outputs_fn(args, [])

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    return (
        _unpickle_noop_compiled_artifact,
        (self._reconstruct_fx_outputs_fn,),
    )

  def updates_default_generator_state(self) -> bool:
    return False

  @property
  def graph_module_debug_str(self) -> str | None:
    return None

  @property
  def mlir_text(self) -> str | None:
    return None

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
