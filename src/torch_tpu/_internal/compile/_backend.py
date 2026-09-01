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
import concurrent.futures
import contextlib
import functools
import hashlib
import threading
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
from torch_tpu._internal.compile.dynamic import dynamic_compiler
from torch_tpu._internal.utils import utils
from torch_tpu._internal.profiler import xprof_adapter

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


def make_backend_compiler(
    example_inputs: Sequence[Any],
    async_compile: bool = False,
    debug: bool = False,
) -> split_compiler.SplitCompiler:
  """Creates a SplitCompiler configured for static or dynamic compilation.

  Args:
    example_inputs: Example inputs to inspect for dynamic SymInts.
    async_compile: If True, executes XLA compilation asynchronously.
    debug: If True, enable debug mode on the base compiler.

  Returns:
    A SplitCompiler instance wrapping either DynamicCompiler or StaticCompiler.
  """
  has_dynamic_symints = compiler.has_dynamic_symints(example_inputs)
  if has_dynamic_symints:
    base_compiler = dynamic_compiler.DynamicCompiler(debug=debug)
  else:
    base_compiler = compiler.StaticCompiler(
        async_compile=async_compile, debug=debug
    )

  return split_compiler.SplitCompiler(base_compiler)


class _CacheSavesManager:
  """Manages asynchronously saving to the AOTAutograd cache in a thread-safe manner."""

  def __init__(self):
    self._cache_save_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="aot_cache_saver"
    )
    self._cache_save_lock = threading.RLock()
    self._pending_cache_saves: set[concurrent.futures.Future[Any]] = set()

  def _get_futures(self) -> Sequence[concurrent.futures.Future]:
    with self._cache_save_lock:
      return list(self._pending_cache_saves)

  def _add(self, f: concurrent.futures.Future) -> None:
    with self._cache_save_lock:
      self._pending_cache_saves.add(f)

  def _discard(self, f: concurrent.futures.Future) -> None:
    with self._cache_save_lock:
      self._pending_cache_saves.discard(f)

  def _difference_update(
      self, futures: Sequence[concurrent.futures.Future]
  ) -> None:
    with self._cache_save_lock:
      self._pending_cache_saves.difference_update(futures)

  def flush(self) -> None:
    """Waits for all pending asynchronous AOTAutograd cache saves to complete."""
    futures = self._get_futures()
    if futures:
      concurrent.futures.wait(futures)
      self._difference_update(futures)

  @contextlib.contextmanager
  def async_aot_cache_saves(self) -> Iterator[None]:
    """Offloads AOTAutogradCache.save to a background thread during async compile."""
    cache_type = _autograd_cache.AOTAutogradCache
    with self._cache_save_lock:
      original_save = cache_type.save

      def async_save(*args: Any, **kwargs: Any) -> None:
        def _safe_save() -> None:
          try:
            original_save(*args, **kwargs)
          except Exception as e:  # pylint: disable=broad-exception-caught
            logging.exception(
                "Error saving AOTAutograd cache asynchronously: %s", e
            )

        fut = self._cache_save_executor.submit(_safe_save)
        self._add(fut)
        fut.add_done_callback(self._discard)

      cache_type.save = staticmethod(async_save)
      try:
        yield
      finally:
        cache_type.save = staticmethod(original_save)


_cache_saves_manager: _CacheSavesManager = _CacheSavesManager()


# pylint: disable=g-bad-exception-name
class AsyncCompilationSubmitted(Exception):
  """Control-flow signal to force AOTAutograd to skip execution after compile.

  This prevents blocking immediately following submission, which would negate
  any benefits of asynchronous compilation.

  Attributes:
    artifact: The CompiledArtifact (or AsyncCompiledArtifact) produced by the
      compilation that was just submitted.
  """

  def __init__(self, artifact: Any = None):
    super().__init__()
    self.artifact = artifact

  def resolve(self) -> None:
    """Waits for the pending background compilation to complete."""
    if self.artifact is not None:
      resolve_fn = getattr(self.artifact, "resolve", None) or getattr(
          self.artifact, "_resolve", None
      )
      if callable(resolve_fn):
        resolve_fn()


class TpuBackend:
  """TPU backend for torch.compile() integration."""

  def __init__(
      self,
      debug: bool = False,
      dynamism: bool = False,
      enable_serialization: bool = True,
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
    """
    self._debug = debug
    self._dynamism = dynamism
    self._enable_serialization = False if debug else enable_serialization
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
    options = kwargs.get("options") or {}
    async_compile = options.get("async_compile", False)
    bounded_dynamism = options.get("bounded_dynamism", self._dynamism)

    # Dynamism support is currently experimental.
    if not bounded_dynamism:
      _raise_on_symint(example_inputs)
    elif not torch._dynamo.config.assume_static_by_default:  # pylint: disable=protected-access
      raise NotImplementedError(
          "TPU backend does not support torch.compile(..., dynamic=True, ...)."
          " Please run torch.compile() with default dynamism i.e."
          " torch.compile(model) and try again."
      )

    _log_gm_and_inputs("__call__", "Pre", graph_module, example_inputs)

    compiler_instance = make_backend_compiler(
        example_inputs, async_compile=async_compile, debug=self._debug
    )
    compiler_instance.execute_pre_grad_passes(graph_module)

    # DynamicCompiler artifacts are not pickleable yet, so only static
    # compilations can participate in AOTAutogradCache.
    enable_serialization = (
        self._enable_serialization
        and not compiler.has_dynamic_symints(example_inputs)
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

    save_context = (
        _cache_saves_manager.async_aot_cache_saves()
        if async_compile
        else contextlib.nullcontext()
    )
    with _serialization_context(enable_serialization) as captured_entry:
      with save_context:
        result = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            keep_inference_input_mutations=False,
        )(
            graph_module, example_inputs
        )  # pytype: disable=wrong-arg-types

      # The following section allows us to avoid blocking within Dynamo on
      # first execution following compilation when `async_compile=True`. This
      # effectively makes the first execution turn into a compile warmup.
      is_warmup_execution = True

      def skip_execution_on_warmup_if_async(*args: Any, **kwargs: Any) -> Any:
        nonlocal is_warmup_execution
        if async_compile and is_warmup_execution:
          is_warmup_execution = False
          artifact = (
              self._compiled_executables[-1]
              if self._compiled_executables
              else None
          )
          raise AsyncCompilationSubmitted(artifact)
        return result(*args, **kwargs)

      if captured_entry is not None and captured_entry[0] is not None:
        entry = captured_entry[0]
        result.serialize = lambda: entry  # pyrefly: ignore[missing-attribute]
        skip_execution_on_warmup_if_async.serialize = lambda: entry  # pyrefly: ignore[missing-attribute]

      return skip_execution_on_warmup_if_async

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


def resolve_compilations(artifacts: Sequence[Any]) -> int:
  """Waits for a sequence of pending compilation artifacts to finish compiling.

  Args:
    artifacts: A sequence of compilation artifacts or AsyncCompilationSubmitted
      exceptions.

  Returns:
    The number of compilation artifacts that were resolved.
  """
  resolved_count = 0
  for item in artifacts:
    if isinstance(item, AsyncCompilationSubmitted):
      item.resolve()
      resolved_count += 1
    else:
      resolve_fn = getattr(item, "resolve", None) or getattr(
          item, "_resolve", None
      )
      if callable(resolve_fn):
        resolve_fn()
        resolved_count += 1
  _cache_saves_manager.flush()
  return resolved_count


def _unpack_warmup_args(
    args: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Unpacks warmup argument item into positional tuple and keyword dictionary."""
  if isinstance(args, tuple):
    if (
        len(args) == 2
        and isinstance(args[0], tuple)
        and isinstance(args[1], dict)
    ):
      return args[0], args[1]
    return args, {}
  elif isinstance(args, dict):
    return (), args
  elif isinstance(args, list):
    return (args,), {}
  else:
    return (args,), {}


def async_compile(
    fn: Callable[..., Any],
    warmup_inputs: Sequence[Any],
    *,
    no_grad: bool = True,
    **backend_kwargs: Any,
) -> Callable[..., Any]:
  """Compiles `fn` across multiple static shapes asynchronously, overlapping

  Dynamo tracing with XLA compilation.

  Args:
    fn: The PyTorch callable or nn.Module to compile.
    warmup_inputs: A sequence of argument tuples or keyword dictionaries
      representing the different static shapes/buckets to compile. Note:
      multiple positional arguments must be passed as tuples (e.g., `[(x, y),
      (z, w)]`); passing a list (e.g., `[[t1, t2]]`) is treated as a single
      positional list argument `fn([t1, t2])`.
    no_grad: Whether to wrap execution with `torch.no_grad()`. Defaults to True
      for inference workloads. Set to False when gradients are required.
    **backend_kwargs: Additional keyword arguments passed to `TpuBackend`.

  Returns:
    The compiled callable ready for execution.
  """
  compiled = torch.compile(
      fn,
      backend="tpu",
      fullgraph=True,
      dynamic=False,
      options={
          "async_compile": True,
          **backend_kwargs,
      },
  )
  if no_grad:
    compiled = torch.no_grad()(compiled)

  artifacts = []
  for args in warmup_inputs:
    with xprof_adapter.TraceMe("_backend.async_compile_warmup"):
      pos_args, kw_args = _unpack_warmup_args(args)
      try:
        compiled(*pos_args, **kw_args)
      except AsyncCompilationSubmitted as e:
        if e.artifact is not None:
          artifacts.append(e.artifact)

  # blocking
  resolve_compilations(artifacts)
  return compiled
