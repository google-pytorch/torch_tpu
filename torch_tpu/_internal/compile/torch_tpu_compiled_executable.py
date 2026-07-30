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
import operator
import os
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

# When set to "1", TorchTpuCompiledExecutable pickles as a reference (the
# compact-format compilation cache key) instead of embedding the serialized
# executable bytes, provided the corresponding entry exists in the tier-2 or
# tier-3 compilation cache. This avoids storing a second copy of every
# compiled executable when a consumer (e.g. vLLM's compilation cache) pickles
# executables to disk alongside the tier-3 cache.
#
# Off by default: unpickling a by-reference executable requires the tiered
# cache entry to still exist, so consumers must be prepared to recompile when
# the entry has been evicted. Consumers that handle that failure (and only
# those) should set this variable.
_PICKLE_BY_CACHE_KEY_ENV_VAR = "TORCH_TPU_INTERNAL_PICKLE_EXECUTABLES_BY_CACHE_KEY"


def _pickle_executables_by_cache_key() -> bool:
  """Returns True if executables should be pickled by cache key reference."""
  return os.environ.get(_PICKLE_BY_CACHE_KEY_ENV_VAR, "0") == "1"


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


def _unpickle_compiled_executable(
    serialized_bytes: bytes,
    reconstruct_fx_outputs_fn: _ReconstructFxOutputsFn | None,
    updates_default_generator_state: bool,
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

  Returns:
    A deserialized TorchTpuCompiledExecutable instance.
  """
  return TorchTpuCompiledExecutable(
      executable=tpu_torch_compile.load_serialized_executable(serialized_bytes),
      reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
      updates_default_generator_state=updates_default_generator_state,
  )


def _unpickle_compiled_executable_by_cache_key(
    cache_key: str,
    reconstruct_fx_outputs_fn: _ReconstructFxOutputsFn | None,
    updates_default_generator_state: bool,
) -> "TorchTpuCompiledExecutable":
  """Reconstructs a TorchTpuCompiledExecutable from a tiered cache entry.

  This is the by-reference counterpart of _unpickle_compiled_executable: the
  pickle carries only the compact-format compilation cache key, and the
  serialized executable bytes are read from the tier-2 or tier-3 compilation
  cache at unpickle time.

  The restored executable is not registered in the in-memory compilation
  cache, so re-pickling it falls back to embedding the serialized bytes.

  Args:
    cache_key: The compact-format compilation cache key identifying the
      executable's entry in the tiered compilation caches.
    reconstruct_fx_outputs_fn: Same as in _unpickle_compiled_executable.
    updates_default_generator_state: Whether the executable updates the default
      generator state.

  Returns:
    A TorchTpuCompiledExecutable loaded from the tiered cache.

  Raises:
    RuntimeError: If no tiered cache holds an entry for the key (e.g. the
      tier-3 cache was evicted, or the pickle was produced by a different
      torch_tpu/libtpu build). Callers holding the original graph should
      recompile in that case.
  """
  return TorchTpuCompiledExecutable(
      executable=tpu_torch_compile.load_executable_by_cache_key(cache_key),
      reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
      updates_default_generator_state=updates_default_generator_state,
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
      updates_default_generator_state: Whether the executable updates the
        default generator state.
    """  # fmt: skip
    self._executable = executable
    self._reconstruct_fx_outputs_fn = reconstruct_fx_outputs_fn
    self._tensor_arg_indices = None
    self._graph_module_debug_str: str | None = None
    self._mlir_text: str | None = None
    self._updates_default_generator_state: bool = (
        updates_default_generator_state
    )

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
        executable_output_shapes = list(output_shapes) + [
            list(t.shape) for t in device_state_tensors
        ]
      else:
        executable_output_shapes = []

      outputs_with_device_state_tensors = tpu_torch_compile.execute(
          self._executable, executable_args, executable_output_shapes
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
         include the serialized PjRtLoadedExecutable and the
         reconstruct_fx_outputs_fn.

    Serialization of the executable is necessary for caching compiled
    artifacts within Dynamo.

    When TORCH_TPU_INTERNAL_PICKLE_EXECUTABLES_BY_CACHE_KEY is set to "1",
    the executable is looked up in the in-memory compilation cache at pickle
    time; if it is found there and its entry exists in the tier-2 or tier-3
    cache, the pickle instead carries only the compact-format cache key (a
    by-reference pickle), avoiding a second on-disk copy of the serialized
    executable. Unpickling then reads the bytes back from the tiered cache;
    see _unpickle_compiled_executable_by_cache_key for the failure mode.

    Returns:
      A tuple (callable, args_tuple) used by the pickle module to
      serialize the object.
    """  # fmt: skip
    if _pickle_executables_by_cache_key():
      cache_key = tpu_torch_compile.find_executable_cache_key(self._executable)
      if (
          cache_key is not None
          and tpu_torch_compile.serialized_executable_exists_in_tiered_cache(
              cache_key
          )
      ):
        logging.info(
            "Pickling compiled executable by cache key %s (tiered cache entry"
            " exists; not embedding serialized bytes).",
            cache_key,
        )
        return (
            _unpickle_compiled_executable_by_cache_key,
            (
                cache_key,
                self._reconstruct_fx_outputs_fn,
                self._updates_default_generator_state,
            ),
        )
    serialized = tpu_torch_compile.serialize_executable(self._executable)
    return (
        _unpickle_compiled_executable,
        (
            serialized,
            self._reconstruct_fx_outputs_fn,
            self._updates_default_generator_state,
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
      args = args[0]
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
