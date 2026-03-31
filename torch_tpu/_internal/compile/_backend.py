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

import functools
import operator
from typing import Any, Callable, List, Sequence, TypeAlias
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
from torch_tpu._internal.utils import utils

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
      map_output_fn: (
          Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] | None
      ),
  ):
    """Initializes the compiled executable wrapper.

    Args:
      executable: The result returned by `tpu_torch_compile.compile_mlir`.
      executable_output_shapes: The output shapes of the executable.
      executable_output_dtypes: The output dtypes of the executable.
      map_output_fn: An optional callable that transforms the output from the
        TPU executable back to the output structure expected by PyTorch. The
        existing use case is to re-insert `None` values into the output list if
        they were filtered out during MLIR conversion.
    """
    self._executable = executable
    self._map_output_fn = map_output_fn
    self._graph_module_debug_str: str | None = None
    self._mlir_graph: str | None = None

  @property
  def graph_module_debug_str(self) -> str | None:
    if self._graph_module_debug_str is None:
      logging.info(UNSET_GRAPH_HELPER_STR)
    return self._graph_module_debug_str

  @graph_module_debug_str.setter
  def graph_module_debug_str(self, value) -> None:
    self._graph_module_debug_str = value

  @property
  def mlir_graph(self) -> str | None:
    if self._mlir_graph is None:
      logging.info(UNSET_GRAPH_HELPER_STR)
    return self._mlir_graph

  @mlir_graph.setter
  def mlir_graph(self, value: str) -> None:
    self._mlir_graph = value

  def __call__(self, *args):
    outputs = tpu_torch_compile.execute(self._executable, args)
    if self._map_output_fn is not None:
      outputs = self._map_output_fn(outputs)

    return outputs

  def __reduce__(self):
    """Enable pickling by serializing the PjRt executable to bytes."""
    serialized = tpu_torch_compile.serialize_executable(self._executable)
    return (
        _unpickle_compiled_executable,
        (serialized, self._map_output_fn),
    )


def _unpickle_compiled_executable(
    serialized_bytes: bytes,
    map_output_fn,
) -> _TorchTpuCompiledExecutable:
  """Reconstruct a _TorchTpuCompiledExecutable from serialized bytes."""
  executable = tpu_torch_compile.load_serialized_executable(serialized_bytes)
  return _TorchTpuCompiledExecutable(
      executable=executable,
      map_output_fn=map_output_fn,
  )


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
    _raise_on_symint(example_inputs)

    logging.info("[TpuBackend] Compiling FX Graph")

    donate_args = list()
    # "options" is part of the torch.compile API but it is not tied to any
    # specific backend and therefore we can extend it with our own fields.
    if "options" in kwargs:
      options = kwargs["options"]
      if "donate_args" in options:
        donate_args = options["donate_args"]

    # Wrap _compile_graph_module to capture the layouts of the example inputs.
    # This information won't be present in the FakeTensors so we have to capture
    # it in advance.
    fw_compiler = functools.partial(
        self._compile_graph_module,
        donate_args=donate_args,
        layouts=[
            tpu_torch_compile.get_device_layout_if_materialized(example_input)
            if isinstance(example_input, torch.Tensor)
            else None
            for example_input in example_inputs
        ],
    )
    # We don't have the layouts for the backwards pass so we don't capture them
    # and we let the compiler handle them.
    bw_compiler = functools.partial(
        self._compile_graph_module,
        donate_args=donate_args,
    )
    # This is to avoid inplace generating graph modules that contains
    # inplace update.

    return aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        keep_inference_input_mutations=False,
    )(graph_module, example_inputs)

  def _compile_graph_module(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: List[torch.Tensor],
      donate_args: Sequence[int],
      layouts: List[str] | None = None,
  ) -> Callable[[torch.fx.GraphModule, List[torch.Tensor]], Callable[..., Any]]:
    """Compiles the graph_module with the given inputs for TPU.

    torch.compile() will generate a graph_module and call this function to
    finish
    the compilation.

    Args:
      graph_module: The FX graph module to compile.
      example_inputs: Example inputs to the FX graph for tracing (not the actual
        inputs).
      donate_args: The list of argument indices that are allowed to be donated.
      layouts: The original layouts of the inputs seen by the backend before
        aot_autograd tracing.

    Returns:
      A function that executes the compiled graph on the TPU.
    """
    if logging.vlog_is_on(logging.DEBUG):
      logging.debug(
          "[TpuBackend.compile_graph_module] Graph:\n%s",
          graph_module.print_readable(print_output=False),
      )
      logging.debug(
          "[TpuBackend.compile_graph_module] Sample Inputs (len = %d): \n%s",
          len(example_inputs),
          utils.InputMetadata(example_inputs),
      )

    # Decompose auto functionalized ops, we need to explicitly do this because
    # the default behaviour inserts flatten and unflatten ops at the boundaries
    # which then blocks buffer donation.
    # This should be replaced with our own fork, see b/491716758.
    GraphTransformObserver(
        graph_module, "decompose_auto_functionalized"
    ).apply_graph_pass(post_grad.decompose_auto_functionalized)
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

      if layouts is None:
        layouts = [None] * len(placeholder_args)
      for arg, layout in zip(placeholder_args, layouts, strict=True):
        if isinstance(arg, torch.Tensor):
          tpu_torch_compile.set_layout_hint(arg, layout)

      # Use debug info format when TORCH_TRACE is set so the artifact has
      # human-readable MLIR with source location annotations.
      print_config = torch_tpu_export.MlirPrintConfig.MLIR_SERIALIZED
      if self._debug or tracing_enabled:
        print_config = torch_tpu_export.MlirPrintConfig.MLIR_DEBUG_INFO

      with dynamo_timed("torchtpu_fx_to_mlir"):
        mlir_graph, _, map_output_fn = torch_tpu_export.fx_to_mlir(
            graph_module,
            placeholder_args,
            print_config=print_config,
            donate_args=donate_args,
        )

    # Emit StableHLO artifact for tlparse when TORCH_TRACE is set.
    if tracing_enabled:
      mlir_str = (
          mlir_graph.decode("utf-8")
          if isinstance(mlir_graph, bytes)
          else mlir_graph
      )
      trace_structured(
          "artifact",
          metadata_fn=lambda: {
              "name": "torchtpu_stablehlo_graph",
              "encoding": "string",
          },
          payload_fn=lambda: mlir_str,
          expect_trace_id=True,
      )

    with dynamo_timed("torchtpu_pjrt_compile"):
      cached_executable = tpu_torch_compile.compile_mlir(mlir_graph)

    executable = _TorchTpuCompiledExecutable(
        executable=cached_executable,
        map_output_fn=map_output_fn,
    )

    self._compiled_executables.append(executable)

    if self._debug:
      # Do not use print_readable() as it include original line of code which is
      # too verbose.
      executable.graph_module_debug_str = str(graph_module.code)
      executable.mlir_graph = mlir_graph

    # If there are non-tensor inputs (e.g. concrete ints from dynamic shapes),
    # wrap the executable to filter them out at call time since
    # tpu_torch_compile.execute() only accepts tensors.
    # Pre-compute indices for _TensorFilterExecutable for performance
    tensor_indices = tuple(
        i
        for i, arg in enumerate(example_inputs)
        if isinstance(arg, torch.Tensor)
    )
    if len(tensor_indices) < len(example_inputs):
      return _TensorFilterExecutable(executable, tensor_indices)

    return executable


class _TensorFilterExecutable:
  """Wraps a compiled executable to filter out non-tensor args at call time.

  Used when aot_autograd passes concrete integers alongside tensors.
  Implemented as a class (not a closure) to support pickling.
  """

  def __init__(
      self,
      executable: _TorchTpuCompiledExecutable,
      tensor_indices: tuple[int, ...] = (),
  ):
    self._executable = executable
    self._tensor_indices = tensor_indices
    # Precompute an itemgetter that does all index lookups in C.
    # itemgetter returns a scalar for 1 key, so wrap it to always return a
    # tuple.
    _getter = operator.itemgetter(*tensor_indices)
    self._itemgetter = (
        (lambda args: (_getter(args),)) if len(tensor_indices) == 1 else _getter
    )

  def __call__(self, *args):
    return self._executable(*self._itemgetter(args))

  def __reduce__(self):
    return (type(self), (self._executable, self._tensor_indices))
