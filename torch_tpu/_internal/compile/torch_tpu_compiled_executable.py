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


def _unpickle_compiled_executable(
    serialized_bytes: bytes,
    reconstruct_fx_outputs_fn: _ReconstructFxOutputsFn | None,
    updates_default_generator_state: bool,
    donate_input_output_aliases: tuple[tuple[int, int], ...] = (),
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
    donate_input_output_aliases: ``(arg_index, executable_output_index)`` pairs
      describing executable-level inplace-donate aliases (see the
      ``TorchTpuCompiledExecutable`` constructor).

  Returns:
    A deserialized TorchTpuCompiledExecutable instance.
  """
  return TorchTpuCompiledExecutable(
      executable=tpu_torch_compile.load_serialized_executable(serialized_bytes),
      reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
      updates_default_generator_state=updates_default_generator_state,
      donate_input_output_aliases=donate_input_output_aliases,
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
      donate_input_output_aliases: tuple[tuple[int, int], ...] = (),
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
      donate_input_output_aliases: A sequence of
        ``(arg_index, executable_output_index)`` pairs describing
        executable-level inplace-donate aliases produced by the
        compile-time FX SSA-rename pass. ``arg_index`` is the position of
        a donated tensor within ``self._take_tensor_args(args)`` (i.e. the
        tensor-only filtered runtime args list); ``executable_output_index``
        is the position of the matching extra output produced by the
        executable. After each execution, this wrapper rebinds the donated
        input tensor's storage to that output's device buffer, then drops
        those extras from what is returned to the caller. Aliases must be
        sorted by ``executable_output_index`` in ascending order; this is
        the order in which ``fx_to_mlir`` appends them.
    """  # fmt: skip
    self._executable = executable
    self._reconstruct_fx_outputs_fn = reconstruct_fx_outputs_fn
    self._tensor_arg_indices = None
    self._graph_module_debug_str: str | None = None
    self._mlir_text: str | None = None
    self._updates_default_generator_state: bool = (
        updates_default_generator_state
    )
    # Defensive copy + validation: stored as a tuple of tuples so the field is
    # cheap to pickle and equality-compare, and so it can't be mutated after
    # construction.
    self._donate_input_output_aliases: tuple[tuple[int, int], ...] = tuple(
        (int(arg_idx), int(out_idx))
        for arg_idx, out_idx in donate_input_output_aliases
    )
    if self._donate_input_output_aliases:
      last_idx = -1
      for arg_idx, out_idx in self._donate_input_output_aliases:
        if arg_idx < 0:
          raise ValueError(
              "donate_input_output_aliases arg_index must be non-negative;"
              f" got {arg_idx}."
          )
        if out_idx <= last_idx:
          raise ValueError(
              "donate_input_output_aliases must be sorted by"
              " executable_output_index in ascending order; got"
              f" {self._donate_input_output_aliases}."
          )
        last_idx = out_idx

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
    tensor_args = self._take_tensor_args(args)
    with tpu_torch_compile.MultiGeneratorLocker(generators):
      device_state_tensors = [
          tpu_torch_compile.get_device_state_tensor(gen) for gen in generators
      ]
      executable_args = (
          *tensor_args,
          *device_state_tensors,
      )

      # When output_shapes for user-defined outputs are provided, we need to
      # append the shapes of the donation-alias outputs and the device state
      # tensors to match the number of outputs from the executable.
      if output_shapes:
        donate_output_shapes = [
            list(tensor_args[arg_idx].shape)
            for arg_idx, _ in self._donate_input_output_aliases
        ]
        executable_output_shapes = (
            list(output_shapes)
            + donate_output_shapes
            + [list(t.shape) for t in device_state_tensors]
        )
      else:
        executable_output_shapes = []

      outputs_with_device_state_tensors = tpu_torch_compile.execute(
          self._executable, executable_args, executable_output_shapes
      )

      if not device_state_tensors:
        outputs = list(outputs_with_device_state_tensors)
        updated_device_state_tensors = []
      else:
        outputs = list(
            outputs_with_device_state_tensors[: -len(device_state_tensors)]
        )
        updated_device_state_tensors = list(
            outputs_with_device_state_tensors[-len(device_state_tensors) :]
        )

      # Strip the donation-alias outputs from the tail of the user-visible
      # outputs and use them to rebind the corresponding donated input
      # tensor's storage. Iterating in reverse keeps the slice indices stable
      # even though we're popping from the list. The aliases are guaranteed
      # sorted by ascending executable_output_index (validated in __init__),
      # which after stripping device-state means they sit on the tail of
      # `outputs` in the same order; reverse iteration covers that tail.
      for arg_idx, _ in reversed(self._donate_input_output_aliases):
        donation_tensor = outputs.pop()
        tpu_torch_compile.rebind_tensor_storage(
            donation_tensor, tensor_args[arg_idx]
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
            self._donate_input_output_aliases,
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
