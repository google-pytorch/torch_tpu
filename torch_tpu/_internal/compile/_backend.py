# Copyright 2025 Google LLC
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

"""TPU backend for `torch.compile()` integration.

The `torch.compile()` function has the following relevant arguments:
  fullgraph (bool):
    We let `torch.compile()` handle graph breaks. We would likely not care.
  dynamic (bool or None): Only False is supported. Dynamic shapes are not
    supported. If a torch.SymInt is encountered in the graph, an error
    will be raised, prompting the user to set dynamic=False.
  [P1]mode (str):
    We are currently only testing the 'default' mode.
    Investigate the other modes with P1
  [p2]options (dict):
    The list is huge.
    The effects of many of these options are still under investigation. P1
  disable (bool):
    Disables the compilation. No ops for us.
"""

from collections.abc import Sequence
import functools
import operator
from typing import Any, Callable, List, TypeAlias
from absl import flags
from absl import logging
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import dynamo_timed
from torch._inductor.fx_passes import post_grad
from torch._logging import trace_structured
from torch._logging._internal import trace_log
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.passes import graph_transform_observer
from torch.utils import _pytree
from torch_tpu._internal import export as torch_tpu_export
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.dynamic import compiler as dynamic_compiler
from torch_tpu._internal.compile.fx_passes import mark_embedded_constants
from torch_tpu._internal.utils import utils

_ENABLE_COMPILED_MODE_DYNAMISM = flags.DEFINE_boolean(
    "torch_tpu_internal_compiled_mode_dynamism",
    False,
    "This is a temporary dev flag as the feature is not ready for production "
    "use. Once ready, this flag will be removed along with assertion check on "
    "SymInt.",
)

GraphTransformObserver = graph_transform_observer.GraphTransformObserver

_ExpectedTypes: TypeAlias = torch.Tensor | torch.nn.Module | torch.SymInt


def to_device(
    x: Any,
    backend: str | torch.device,
) -> Any:
  """Moves complex objects containing tensors to the specified backend device.

  This function uses pytree to traverse the input object, and find torch.Tensor
  objects and move them to the target backend.

  Args:
    x: The object to move. Can be a torch.Tensor, torch.nn.Module, list, or
      tuple containing tensors.
    backend: The target device to move the tensors to (e.g., 'cpu', 'cuda',
      'tpu').

  Returns:
    The object with all contained tensors moved to the backend device.
  """

  def _to(t: _ExpectedTypes) -> _ExpectedTypes:
    match t:
      case torch.Tensor():
        return t.to(backend)
      case torch.nn.Module():
        return t.to(backend)
      case _:
        return t

  return _pytree.tree_map_only(_ExpectedTypes, _to, x)


def _raise_on_symint(
    x: Any,
) -> Any:
  """We will defer the support of symint for now.

  Here it scans the input object for SymInt. And raises error if found.
  """

  def _raise(t: torch.SymInt):
    raise NotImplementedError(
        "TPU backend: does not support dynamic shape. Please set"
        " torch.compile(..., dynamic=False, ...) and try again."
    )

  return _pytree.tree_map_only(torch.SymInt, _raise, x)


def _has_symints(inputs: Sequence[Any]) -> bool:
  """Checks if any of the inputs are or contain torch.SymInt."""
  flattened_inputs, _ = _pytree.tree_flatten(inputs)
  for arg in flattened_inputs:
    if isinstance(arg, torch.SymInt):
      return True
    if isinstance(arg, torch.Tensor):
      for dim in arg.shape:
        if isinstance(dim, torch.SymInt):
          return True
  return False


UNSET_GRAPH_HELPER_STR = (
    "FX/MLIR graph is not recorded by default. Turn on TPU backend debug mode"
    " to record it."
)


class _TorchTpuCompiledExecutable:
  """A callable for a TorchTPU compiled executable.

  This class is returned to dynamo and stored in the dynamo cache. Any time an
  FX graph being traced by dynamo satisfies the guards setup to produce this
  executable, a call to the __call__ method is made.
  """

  def __init__(
      self,
      executable: tpu_torch_compile.PjRtLoadedExecutable,
      reconstruct_fx_outputs_fn: (
          Callable[[Sequence[Any], Sequence[torch.Tensor]], Sequence[Any]]
          | None
      ),
  ):
    """Initializes the compiled executable wrapper.

    Args:
      executable: The result returned by `tpu_torch_compile.compile_mlir`.
      reconstruct_fx_outputs_fn: An optional callable that transforms the output
        from the TPU executable back to the output structure expected by
        PyTorch. The existing use case is to re-insert `None` values into the
        output list if they were filtered out during MLIR conversion.
    """
    self._executable = executable
    self._reconstruct_fx_outputs_fn = reconstruct_fx_outputs_fn
    self._tensor_arg_indices = None
    self._graph_module_debug_str: str | None = None
    self._mlir_text: str | None = None

  @property
  def graph_module_debug_str(self) -> str | None:
    if self._graph_module_debug_str is None:
      logging.info(UNSET_GRAPH_HELPER_STR)
    return self._graph_module_debug_str

  @graph_module_debug_str.setter
  def graph_module_debug_str(self, value) -> None:
    self._graph_module_debug_str = value

  @property
  def mlir_text(self) -> str | None:
    if self._mlir_text is None:
      logging.info(UNSET_GRAPH_HELPER_STR)
    return self._mlir_text

  @mlir_text.setter
  def mlir_text(self, value: str) -> None:
    self._mlir_text = value

  def __call__(
      self, *args: Any, output_shapes: Sequence[list[int]] | None = None
  ) -> Any:
    if output_shapes is None:
      output_shapes = []
    executable_args = self._filter_tensor_args(args)
    outputs = tpu_torch_compile.execute(
        self._executable, executable_args, output_shapes
    )

    if self._reconstruct_fx_outputs_fn is not None:
      outputs = self._reconstruct_fx_outputs_fn(args, outputs)

    return outputs

  def _filter_tensor_args(
      self, args: tuple[Any, ...]
  ) -> tuple[torch.Tensor, ...]:
    """Filters out non-tensor arguments (e.g., concrete integers)."""
    if self._tensor_arg_indices is None:
      # Pre-compute indices for filtering out non-tensor args once.
      # Assume args will have the same structure for subsequent calls.
      self._tensor_arg_indices = tuple(
          i for i, arg in enumerate(args) if isinstance(arg, torch.Tensor)
      )

    if len(self._tensor_arg_indices) == len(args):
      return args

    filtered = operator.itemgetter(*self._tensor_arg_indices)(args)
    if len(self._tensor_arg_indices) == 1:
      return (filtered,)
    return filtered

  def __reduce__(self):
    """Enable pickling by serializing the PjRt executable to bytes."""
    serialized = tpu_torch_compile.serialize_executable(self._executable)
    return (
        _unpickle_compiled_executable,
        (serialized, self._reconstruct_fx_outputs_fn),
    )


def _unpickle_compiled_executable(
    serialized_bytes: bytes,
    reconstruct_fx_outputs_fn,
) -> _TorchTpuCompiledExecutable:
  """Reconstruct a _TorchTpuCompiledExecutable from serialized bytes."""
  executable = tpu_torch_compile.load_serialized_executable(serialized_bytes)
  return _TorchTpuCompiledExecutable(
      executable=executable,
      reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
  )


def _split_str_for_logging(text, limit=13000) -> Sequence[str]:
  """Splits a string longer than the log line char limit into multiple chunks."""
  chunks = []
  while len(text) > limit:
    split_index = text.rfind("\n", 0, limit)
    if split_index == -1:
      split_index = limit
    chunks.append(text[:split_index])
    text = text[split_index + 1 :]
  if text:
    chunks.append(text)
  return chunks


def _log_gm_and_inputs(
    loc: str,
    pre_or_post: str,
    gm: torch.fx.GraphModule,
    inputs: Sequence[Any],
) -> None:
  """Logs the FX graph and example inputs."""
  if not logging.vlog_is_on(logging.DEBUG):
    return

  inputs_chunks = _split_str_for_logging(str(utils.InputMetadata(inputs)))
  gm_chunks = _split_str_for_logging(gm.print_readable(print_output=False))
  preamble = f"[TpuBackend.{loc}] {pre_or_post}-AOT Autograd"
  logging.debug("%s Graph:", preamble)
  for c in gm_chunks:
    logging.debug(c)
  logging.debug("%s Inputs Length: %d", preamble, len(inputs))
  logging.debug("%s Sample Inputs:", preamble)
  for c in inputs_chunks:
    logging.debug(c)


class TpuBackend:
  """TPU backend for torch.compile() integratation."""

  def __init__(
      self,
      debug: bool = False,
  ):
    """Initializes the TPU backend.

    Args:
      debug (bool): If True, enable debug logging and save a dump of the fx
        graph.
    """
    self._debug = debug
    # Stores information about each compiled executable.
    # Organized by order of compilation (index 0 is the first compilation, etc.)
    self._compiled_executables: list[_TorchTpuCompiledExecutable] = []

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: List[torch.Tensor],
      **kwargs,
  ) -> Callable[[torch.fx.GraphModule, List[torch.Tensor]], Callable[..., Any]]:
    # The TorchTPU team decided to defer symint support for now. Here it scans
    # the input object for SymInt. And raises error if found.
    #
    # We should consider ways around this, perhaps we lazily compile in the
    # function invocation when we have arg shapes. Or perhaps we can add a
    # mechanism for backends to notify dynamo that dynamic shapes aren't
    # supported so the user isn't expected to use `dynamic=False` on all compile
    # calls.
    #
    # Per @wan, we might want to remove SymInt from the graph and static
    # recompile to avoid user changes -- this is potentially unsafe since dynamo
    # will store the returned executable in the cache and might not re-dispatch
    # to our backend compiler if future SymInts fit the allowed range. We could
    # consider compiling lazily on each function invocation, but this will have
    # performance impact.
    #
    # TODO: Figure out how to remove symint from the graph and recompile
    # without dynamism.
    if not flags.FLAGS["torch_tpu_internal_compiled_mode_dynamism"].value:
      _raise_on_symint(example_inputs)

    _log_gm_and_inputs("__call__", "Pre", graph_module, example_inputs)

    has_symints = _has_symints(example_inputs)
    if has_symints:
      compiler = dynamic_compiler.DynamicCompiler(self)
    else:
      compiler = functools.partial(self._compile_graph_module)

    return aot_autograd(
        fw_compiler=compiler,
        # This is to avoid inplace generating graph modules that contains
        # inplace update.
        keep_inference_input_mutations=False,
    )(graph_module, example_inputs)

  def _compile_graph_module(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: List[torch.Tensor],
  ) -> Callable[[torch.fx.GraphModule, List[torch.Tensor]], Callable[..., Any]]:
    """Compiles the graph_module with the given inputs for TPU.

    torch.compile() will generate a graph_module and call this function to
    finish
    the compilation.

    Args:
      graph_module: The FX graph module to compile.
      example_inputs: Example inputs to the FX graph for tracing (not the actual
        inputs).

    Returns:
      A function that executes the compiled graph on the TPU.
    """

    _log_gm_and_inputs(
        "_compile_graph_module", "Post", graph_module, example_inputs
    )

    # Decompose auto functionalized ops, we need to explicitly do this because
    # the default behaviour inserts flatten and unflatten ops at the boundaries
    # which then blocks buffer donation.
    # This should be replaced with our own fork, see b/491716758.
    GraphTransformObserver(
        graph_module, "decompose_auto_functionalized"
    ).apply_graph_pass(post_grad.decompose_auto_functionalized)
    GraphTransformObserver(
        graph_module, "mark_embedded_constants"
    ).apply_graph_pass(mark_embedded_constants.apply)

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
    self._compiled_executables.append(executable)

    if self._debug:
      # Do not use print_readable() as it include original line of code which is
      # too verbose.
      executable.graph_module_debug_str = str(graph_module.code)
      executable.mlir_text = exported_mlir.serialize_text(
          enable_debug_info=True
      )

    return executable
