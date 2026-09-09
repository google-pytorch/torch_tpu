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
import dataclasses
import functools
import hashlib
import re
import threading
from typing import Any, TypeAlias, cast

from absl import logging
import torch
from torch._decomp import get_decompositions
from torch._dynamo import guards as dynamo_guards
from torch._dynamo import source as dynamo_source
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd import autograd_cache as _autograd_cache
from torch._functorch._aot_autograd import graph_compile as _graph_compile
from torch._functorch._aot_autograd.schemas import AOTAutogradCacheInfo
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler
import torch._functorch.config as functorch_config
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch.utils import _pytree
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import split_compiler
from torch_tpu._internal.compile.debug import TpuCompileDebug
from torch_tpu._internal.compile.dynamic import dynamic_compiler
from torch_tpu._internal.compile.torch_tpu_compiled_executable import AsyncCompiledArtifact
from torch_tpu._internal.utils import utils
from torch_tpu._internal.benchmarks import xprof_adapter

_TPU_DECOMPOSITIONS = get_decompositions([
    # We decompose masked_fill to align behavior with GPU (Inductor), where
    # masked_fill_ calls are decomposed into convert_element_type + where and
    # succeed even on overflow conversion via modular truncation, rather than
    # failing due to placeholder tensor materialization during compilation.
    torch.ops.aten.masked_fill.Tensor,
    torch.ops.aten.masked_fill_.Tensor,
])

_ExpectedTypes: TypeAlias = torch.Tensor | torch.nn.Module | torch.SymInt


def get_default_cpu_backend() -> str:
  """Returns the default Dynamo backend to use for CPU-only graphs.

  In open-source environments this is inductor which is the PyTorch default.

  Returns:
    The default backend to use for CPU-only graphs.
  """
  # pylint: disable=protected-access
  if torch._inductor.config.cpp.cxx[0] is None:
    return "aot_eager"
  return "inductor"


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
def _serialization_context(
    example_inputs: Sequence[Any],
    enable: bool = False,
) -> Iterator[list[Any] | None]:
  """Optionally configures aot_autograd to produce a BundledAOTAutogradResult.

  When enabled, patches ``_cache_inference_info`` to inject a per-graph
  ``cache_info`` (so the bundled entry is created) and suppresses
  ``generate_guards_expression`` (which requires a TracingContext with a
  ShapeEnv that we cannot safely provide without leaking FakeTensorMode
  into subsequent compilations).

  Args:
    example_inputs: Example inputs to differentiate cache keys based on input
      properties such as storage offsets.
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

    # The default tensor metadata extraction methods below are used during cache
    # key generation and unfortunately they zero-out `storage_offset` in the
    # returned metadata object which is harmful for TPU. Override them so that
    # we can prevent this behavior and restore `storage_offset` as a part of
    # any cache key that is generated.
    orig_extract_meta = _autograd_cache.extract_tensor_metadata_for_cache_key
    orig_codecache_extract_meta = (
        torch._inductor.codecache.extract_tensor_metadata_for_cache_key
    )

    # Restores `storage_offset`, but does set `storage_bytes=None` just like
    # the patched method.
    def _patched_extract_meta(t: torch.Tensor):
      meta = torch._subclasses.fake_tensor.extract_tensor_metadata(t)
      if not hasattr(t, "_is_inductor_static"):
        meta = dataclasses.replace(meta, storage_bytes=None)
      return meta

    _autograd_cache.extract_tensor_metadata_for_cache_key = (
        _patched_extract_meta
    )
    torch._inductor.codecache.extract_tensor_metadata_for_cache_key = (
        _patched_extract_meta
    )
    stack.callback(
        setattr,
        _autograd_cache,
        "extract_tensor_metadata_for_cache_key",
        orig_extract_meta,
    )
    stack.callback(
        setattr,
        torch._inductor.codecache,
        "extract_tensor_metadata_for_cache_key",
        orig_codecache_extract_meta,
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
        # Without inputs_meta, the custom key would collide for graphs
        # differing only in `storage_offset`, causing runtime input size
        # mismatches between the compiled artifact and the provided input.
        inputs_meta = []
        for t in example_inputs:
          if isinstance(t, torch.Tensor):
            inputs_meta.append(_patched_extract_meta(t))
          else:
            inputs_meta.append(type(t))

        graph_hash = hashlib.sha256(
            f"{version_prefix}:{aot_forward_graph_str or ''}:{inputs_meta}"
            .encode()
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
    debug: TpuCompileDebug | None = None,
) -> split_compiler.SplitCompiler:
  """Creates a SplitCompiler configured for static or dynamic compilation.

  Args:
    example_inputs: Example inputs to inspect for dynamic SymInts.
    async_compile: If True, executes XLA compilation asynchronously.
    debug: Optional TpuCompileDebug container for recording compilation debug
      artifacts.

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


def _guard_input_storage_offsets(graph_module: torch.fx.GraphModule) -> None:
  """Installs Dynamo guards on storage_offset for all tensor placeholder inputs.

  Dynamo's default TENSOR_MATCH guard does not guard against tensor
  storage_offset. For static shape compilations, tensor views with different
  storage offsets bake static slice indices into the compiled executable.
  Installing an EQUALS_MATCH guard on TensorProperty.STORAGE_OFFSET ensures
  Dynamo recompiles when an input tensor is passed with a different storage
  offset.

  Args:
    graph_module: The FX graph module whose placeholder inputs to guard.
  """
  # Ensure we're within a tracing context to install guards
  tracing_context = torch._guards.TracingContext.try_get()
  if tracing_context is None:
    return

  for node in graph_module.graph.find_nodes(op="placeholder"):
    if not isinstance(node.meta.get("example_value"), torch.Tensor):
      continue

    offset_source = dynamo_source.TensorPropertySource(
        node._dynamo_source, dynamo_source.TensorProperty.STORAGE_OFFSET
    )
    guard = offset_source.make_guard(dynamo_guards.GuardBuilder.EQUALS_MATCH)
    dynamo_guards.install_guard(guard)


_VALID_SYMBOL_REGEX = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_module_name(name: str) -> str:
  """Ensures name is a valid MLIR symbol (alphanumeric + underscores)."""
  return _VALID_SYMBOL_REGEX.sub("_", name)


def _get_module_name(
    custom_module_name: str | None,
    is_fwd: bool,
    split_idx: int,
) -> str | None:
  """Extracts a human-readable module name for the compiled MLIR graph."""
  if not custom_module_name:
    return None
  clean_name = _sanitize_module_name(custom_module_name)
  phase = "fwd" if is_fwd else "bwd"
  return f"{clean_name}_split{split_idx}_{phase}"


class TpuBackend:
  """TPU backend for torch.compile() integration."""

  # TODO(yho): Delete this ctor once debug, dynamism, and enable_serialization
  # are removed.
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

    Note: The `debug`, `dynamism`, and `enable_serialization` arguments are
    internal flags slated for deprecation. If set on the instance, they will
    override the corresponding keys passed in the `options` dict to `__call__`.
    New callers should pass options to `torch.compile(..., backend="tpu",
      options={...})`.
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
    """Compiles the provided `torch.fx.GraphModule` with the XLA compiler.

    Args:
      graph_module: The graph module to compile.
      example_inputs: Example inputs.
      options: Options dict to pass to the backend. Available keys:
        async_compile: Default False. If True, executes XLA compilation
          asynchronously.
        bounded_dynamism: Default False. If True, allows bounded dynamic shapes.
        serializable: Default True. If True, uses
          SerializableAOTDispatchCompiler so that .serialize() is attached to
          compiled functions, which enables PyTorch's AOTAutogradCache to
          save/load compilation artifacts. If False, serialization disabled so
          that callers can inspect freshly compiled artifacts instead of cache
          hits. Ignored and treated as False if symints and bounded dynamism are
          both present.
        debug_callback: Callback function to receive debug information. Must
          accept one arg of type `TpuCompileDebug`. Note: For training graphs
          with backward passes, the callback is invoked upon forward compilation
          with a `TpuCompileDebug` object; backward compilation artifacts could
          be populated into the same object in-place during subsequent backward
          pass execution (e.g. `loss.backward()`). It is recommended to inspect
          backward artifacts after backward execution.

    Notes:
      The options dict to this method replaces removed args to the TpuBackend
        constructor.
      The `debug` arg is replaced by the `serializable` and `debug_callback`
        options.
    """
    # Process keys in options dict.
    options = kwargs.pop("options", {}) or {}

    async_compile = options.get("async_compile", False)
    bounded_dynamism = options.get("bounded_dynamism", False)
    # DynamicCompiler artifacts are not pickleable yet, so only static
    # compilations can participate in AOTAutogradCache.
    serializable = cast(
        bool, options.get("serializable", True)
    ) and not compiler.has_dynamic_symints(example_inputs)

    debug_callback = options.get("debug_callback", None)
    if debug_callback is not None and not callable(debug_callback):
      raise TypeError("'debug_callback' must be callable")
    debug = TpuCompileDebug() if debug_callback else None
    if kwargs:
      raise TypeError("Unexpected keyword arguments: {}".format(kwargs))

    # TODO(yho): The following self._* attributes are slated for deprecation.
    # Overwrite options if legacy constructor arguments were specified.
    if self._dynamism:
      bounded_dynamism = self._dynamism
    if not self._enable_serialization or self._debug:
      serializable = self._enable_serialization
    if self._debug and debug is None:
      debug = TpuCompileDebug()
    # self._debug, self._dynamism, and self._enable_serialization should
    # NOT be accessed after this point.

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
    _guard_input_storage_offsets(graph_module)
    custom_module_name = (
        options.get("custom_module_name") or options.get("name")
        if options
        else None
    )

    if debug is not None:
      debug.pre_autograd_fx_code.append(graph_module.code)
      debug.pre_autograd_fx_readable.append(
          graph_module.print_readable(print_output=False)
      )

    compiler_instance = make_backend_compiler(
        example_inputs, async_compile=async_compile, debug=debug
    )
    compiler_instance.execute_pre_grad_passes(graph_module)

    compiled_artifacts: list[compiler.CompiledArtifact] = []

    if serializable:
      fw_compiler = SerializableAOTDispatchCompiler(
          output_code_ty=compiler.CompiledArtifact,
          compiler_fn=functools.partial(  # pyrefly: ignore[bad-specialization]
              self._compile_graph_module,
              compiler_instance,
              True,
              custom_module_name,
              debug,
              compiled_artifacts,
          ),
      )
    else:
      fw_compiler = functools.partial(
          self._compile_graph_module,
          compiler_instance,
          True,
          custom_module_name,
          debug,
          compiled_artifacts,
      )

    bw_compiler = functools.partial(
        self._compile_graph_module,
        compiler_instance,
        False,
        custom_module_name,
        debug,
        compiled_artifacts,
    )

    save_context = (
        _cache_saves_manager.async_aot_cache_saves()
        if async_compile
        else contextlib.nullcontext()
    )
    with _serialization_context(example_inputs, serializable) as captured_entry:
      with save_context:
        result = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=_TPU_DECOMPOSITIONS,  # pyrefly: ignore[bad-argument-type]
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
          artifact = compiled_artifacts[-1] if compiled_artifacts else None
          raise AsyncCompilationSubmitted(artifact)
        return result(*args, **kwargs)

      if captured_entry is not None and captured_entry[0] is not None:
        entry = captured_entry[0]
        result.serialize = lambda: entry  # pyrefly: ignore[missing-attribute]
        skip_execution_on_warmup_if_async.serialize = lambda: entry  # pyrefly: ignore[missing-attribute]

      if debug is not None:
        debug.compiled_executables = list(compiled_artifacts)

      # Call the debug callback, or if async, submit a job that waits on the compile futures.
      if debug is not None and debug_callback is not None:
        if async_compile:
          futures = [
              cast(AsyncCompiledArtifact, e)._future
              for e in compiled_artifacts
              if isinstance(e, AsyncCompiledArtifact)
          ]

          # Ensure callbacks that fail are visible
          def wait_and_call():
            try:
              for f in futures:
                f.result()
              debug_callback(debug)
            except Exception:
              logging.exception("debug_callback crashed.")
              raise

          compiler.StaticCompiler._async_compile_executor.submit(wait_and_call)
        else:
          debug_callback(debug)

      return skip_execution_on_warmup_if_async

  def _compile_graph_module(
      self,
      compiler_instance: compiler.Compiler,
      is_fwd: bool,
      custom_module_name: str | None,
      debug: TpuCompileDebug | None,
      compiled_artifacts: list[compiler.CompiledArtifact],
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[torch.Tensor],
  ) -> Callable[..., Any]:
    """Compiles the graph_module with the given inputs for TPU.

    torch.compile() will generate a graph_module and call this function to
    finish the compilation.

    Args:
      compiler_instance: Responsible for compiling the graph module.
      is_fwd: Indicates whether the forward or backward pass is being compiled.
      custom_module_name: The base custom module name provided in backend
        options, if any.
      debug: Container for recording compilation debug artifacts.
      compiled_artifacts: Per-call container tracking compiled artifacts.
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

    if debug is not None:
      if is_fwd:
        debug.post_autograd_fx_forward_code.append(graph_module.code)
        debug.post_autograd_fx_forward_readable.append(
            graph_module.print_readable(print_output=False)
        )
      else:
        debug.post_autograd_fx_backward_code.append(graph_module.code)
        debug.post_autograd_fx_backward_readable.append(
            graph_module.print_readable(print_output=False)
        )

    split_idx = len(self._compiled_executables)
    module_name = _get_module_name(custom_module_name, is_fwd, split_idx)

    executable = compiler_instance(
        graph_module, example_inputs, is_fwd, module_name=module_name
    )
    compiled_artifacts.append(executable)
    self._compiled_executables.append(executable)
    if debug is not None:
      debug.compiled_executables = list(compiled_artifacts)

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
