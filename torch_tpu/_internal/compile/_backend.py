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

from collections.abc import Callable, Iterator, Sequence
import contextlib
import functools
import hashlib
from typing import Any, TypeAlias

from absl import logging
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd import autograd_cache as _autograd_cache
from torch._functorch._aot_autograd import graph_compile as _graph_compile
from torch._functorch._aot_autograd.schemas import AOTAutogradCacheInfo
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler
import torch._functorch.config as functorch_config
from torch.utils import _pytree
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import split_compiler
from torch_tpu._internal.compile.dynamic import compiler as dynamic_compiler
from torch_tpu._internal.utils import utils

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

  Args:
    x: The input object to scan for SymInt.

  Returns:
    The input object if no SymInt is found.
  """

  def _raise(t: torch.SymInt):
    raise NotImplementedError(
        "TPU backend: does not support dynamic shape. Please set"
        " torch.compile(..., dynamic=False, ...) and try again."
    )

  return _pytree.tree_map_only(torch.SymInt, _raise, x)


@contextlib.contextmanager
def _serialization_context(enable: bool = False) -> Iterator[list[Any] | None]:
  """Optionally configures aot_autograd to produce a BundledAOTAutogradResult.

  When enabled, patches ``_cache_inference_info`` to inject a per-graph
  ``cache_info`` (so the bundled entry is created) and suppresses
  ``generate_guards_expression`` (which requires a TracingContext with a
  ShapeEnv that we cannot safely provide without leaking FakeTensorMode
  into subsequent compilations).

  Args:
    enable: If True, activate the serialization patches and yield a one-element
      list whose ``[0]`` slot will hold the captured
      ``BundledAOTAutogradResult`` after compilation.  If False, yield None and
      do nothing.

  Yields:
    A one-element ``list[Any]`` containing the captured entry (when
    *enable* is True), or None (when *enable* is False).

  TODO: Once we bump torch past
  https://github.com/pytorch/pytorch/pull/170443, replace this entire
  context manager with:
    functorch_config.patch({
        "bundled_autograd_cache": True,
        "bypass_autograd_cache_key": True,
    })
  and remove the _cache_inference_info and generate_guards_expression
  patches. The bypass_autograd_cache_key flag makes try_load generate a
  nonce cache_key on failure, so cache_info is set naturally.
  """
  if not enable:
    yield None
    return

  captured_entry: list[Any] = [None]

  with contextlib.ExitStack() as stack:
    stack.enter_context(
        functorch_config.patch("bundled_autograd_cache", True),
    )

    orig_cache_inference_info = _graph_compile._cache_inference_info  # pylint: disable=protected-access

    def _patched_cache_inference_info(
        aot_config,
        fw_metadata,
        maybe_subclass_meta,
        compiled_fw,
        aot_forward_graph_str,
        wrappers,
    ):
      has_cache_info = aot_config.cache_info is not None
      if not has_cache_info:
        import torch_tpu  # pylint: disable=g-import-not-at-top; buildcleaner: ignore

        version_prefix = (
            f"torch={torch.__version__}"
            f"_torchtpu={getattr(torch_tpu, '__version__', 'dev')}"
        )
        graph_hash = hashlib.sha256(
            f"{version_prefix}:{aot_forward_graph_str or ''}".encode()
        ).hexdigest()
        object.__setattr__(
            aot_config,
            "cache_info",
            AOTAutogradCacheInfo(
                cache_key=f"torchtpu_{graph_hash}",
                start_time_ns=0,
                forward_symints=[],
            ),
        )

      # TODO: Remove once we bump torch past
      # https://github.com/pytorch/pytorch/pull/171600 which makes
      # generate_guards_expression handle a missing ShapeEnv gracefully.
      orig_guards = _autograd_cache.AOTAutogradCache.generate_guards_expression
      _autograd_cache.AOTAutogradCache.generate_guards_expression = (
          staticmethod(lambda *a, **kw: None)
      )
      try:
        entry = orig_cache_inference_info(
            aot_config,
            fw_metadata,
            maybe_subclass_meta,
            compiled_fw,
            aot_forward_graph_str,
            wrappers,
        )
      finally:
        _autograd_cache.AOTAutogradCache.generate_guards_expression = (
            orig_guards
        )

      if not has_cache_info:
        object.__setattr__(aot_config, "cache_info", None)
      if entry is not None:
        captured_entry[0] = entry
      return entry

    _graph_compile._cache_inference_info = _patched_cache_inference_info  # pylint: disable=protected-access
    stack.callback(
        setattr,
        _graph_compile,
        "_cache_inference_info",
        orig_cache_inference_info,
    )

    yield captured_entry


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
  """TPU backend for torch.compile() integration."""

  def __init__(
      self,
      debug: bool = False,
      dynamism: bool = False,
      enable_serialization: bool = True,
      precompile_steps: int = 0,
  ):
    """Initializes the TPU backend.

    Args:
      debug (bool): If True, enable debug logging and save a dump of the fx
        graph.
      dynamism (bool): If True, enable dynamism.
      enable_serialization: If True, enable the aot_autograd bundled cache so
        that .serialize() is attached to compiled functions. This defaults to
        True so PyTorch's AOTAutogradCache can save/load compilation artifacts.
        Debug mode disables serialization so debug callers inspect freshly
        compiled artifacts instead of cache hits.
      precompile_steps: The number of steps to precompile dynamic adapters for.
    """
    self._debug = debug
    self._dynamism = dynamism
    self._enable_serialization = False if debug else enable_serialization
    self._precompile_steps = precompile_steps
    # Stores information about each compiled executable.
    # Organized by order of compilation (index 0 is the first compilation, etc.)
    self._compiled_executables: list[compiler.CompiledArtifact] = []

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[torch.Tensor],
      **kwargs,
  ) -> Callable[
      [torch.fx.GraphModule, Sequence[torch.Tensor]], Callable[..., Any]
  ]:
    # Dynamism support is currently experimental.
    if not self._dynamism:
      _raise_on_symint(example_inputs)

    _log_gm_and_inputs("__call__", "Pre", graph_module, example_inputs)

    has_dynamic_symints = compiler.has_dynamic_symints(example_inputs)
    if has_dynamic_symints:
      base_compiler = dynamic_compiler.DynamicCompiler(
          debug=self._debug,
          precompile_steps=self._precompile_steps,
      )
    else:
      base_compiler = compiler.StaticCompiler(debug=self._debug)

    compiler_instance = split_compiler.SplitCompiler(base_compiler)
    compiler_instance.execute_pre_grad_passes(graph_module)

    # DynamicCompiler artifacts are not pickleable yet, so only static
    # compilations can participate in AOTAutogradCache.
    enable_serialization = (
        self._enable_serialization and not has_dynamic_symints
    )

    if enable_serialization:
      fw_compiler = SerializableAOTDispatchCompiler(
          output_code_ty=compiler.CompiledArtifact,
          compiler_fn=functools.partial(  # pyrefly: ignore[bad-specialization]
              self._compile_graph_module, compiler_instance, True
          ),
      )
    else:
      fw_compiler = functools.partial(
          self._compile_graph_module, compiler_instance, True
      )

    bw_compiler = functools.partial(
        self._compile_graph_module, compiler_instance, False
    )

    with _serialization_context(enable_serialization) as captured_entry:
      result = aot_autograd(
          fw_compiler=fw_compiler,
          bw_compiler=bw_compiler,
          keep_inference_input_mutations=False,
      )(
          graph_module, example_inputs
      )  # pytype: disable=wrong-arg-types

      if captured_entry is not None and captured_entry[0] is not None:
        entry = captured_entry[0]
        result.serialize = lambda: entry  # pyrefly: ignore[missing-attribute]

      return result

  def _compile_graph_module(
      self,
      compiler_instance: compiler.Compiler,
      is_fwd: bool,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[torch.Tensor],
  ) -> Callable[..., Any]:
    """Compiles the graph_module with the given inputs for TPU.

    torch.compile() will generate a graph_module and call this function to
    finish the compilation.

    Args:
      compiler_instance: Responsible for compiling the graph module.
      is_fwd: Indicates whether the forward or backward pass is being compiled.
      graph_module: The FX graph module to compile.
      example_inputs: Example inputs to the FX graph for tracing (not the actual
        inputs).

    Returns:
      A function that executes the compiled graph on the TPU.
    """

    fwd_or_bwd_str = "FORWARD" if is_fwd else "BACKWARD"

    _log_gm_and_inputs(
        "_compile_graph_module",
        f"***{fwd_or_bwd_str}*** Post",
        graph_module,
        example_inputs,
    )

    executable = compiler_instance(graph_module, example_inputs, is_fwd)

    self._compiled_executables.append(executable)

    return executable
