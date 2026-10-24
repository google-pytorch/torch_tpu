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

from collections.abc import Callable, Iterator, Mapping, Sequence
import concurrent.futures
import contextlib
import dataclasses
import functools
import hashlib
import operator
import pathlib
import re
import threading
from typing import Any, TypeAlias, cast

from absl import logging
import torch
from torch._decomp import get_decompositions
from torch._dynamo import guards as dynamo_guards
from torch._dynamo import source as dynamo_source
from torch._dynamo.backends import registry
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd import autograd_cache as _autograd_cache
from torch._functorch._aot_autograd import descriptors as _aot_descriptors
from torch._functorch._aot_autograd import graph_compile as _graph_compile
from torch._functorch._aot_autograd.schemas import AOTAutogradCacheInfo
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler
import torch._functorch.config as functorch_config
from torch._functorch.partitioners import min_cut_rematerialization_partition
import torch.compiler.config as compiler_config
from torch.utils import _python_dispatch
from torch.utils import _pytree
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import split_compiler
from torch_tpu._internal.compile.debug import TpuCompileDebug
from torch_tpu._internal.compile.dynamic import dynamic_compiler
from torch_tpu._internal.distributed import device_mesh
from torch_tpu._internal.distributed import process_group_utils
from torch_tpu._internal.precision import precision_impl
from torch_tpu._internal.sync import sync
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
# Maximum number of dynamic shape buckets allowed per compilation, matching the
# worker limit of StaticCompiler._async_compile_executor.
_MAX_DYNAMIC_BUCKETS = 8


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


def _is_ao_op(node: torch.fx.Node, op_name: str) -> bool:
  """Returns whether `node` represents a torch.ops.ao.{op_name} operation."""
  if node.op != "call_function":
    return False

  ao_op = getattr(getattr(torch.ops, "ao", None), op_name, None)
  return (
      ao_op is not None
      and getattr(node.target, "_overloadpacket", node.target) == ao_op
  )


def _is_view_node(node: torch.fx.Node) -> bool:
  """Returns whether `node` is a view, container unpack, or `index_select`."""
  if node.op != "call_function":
    return False

  is_container_unpack = node.target is operator.getitem
  if is_container_unpack:
    return True

  is_view = getattr(node.target, "is_view", False)
  if is_view:
    return True

  target_overload = getattr(node.target, "_overloadpacket", node.target)
  is_unsafe_view = target_overload == getattr(
      torch.ops.aten, "_unsafe_view", None
  )
  is_index_select = target_overload == getattr(
      torch.ops.aten, "index_select", None
  )
  return is_unsafe_view or is_index_select


def _extract_tensor_node(node: torch.fx.Node) -> torch.fx.Node | None:
  """Extracts the input tensor node from positional args or kwargs."""
  if node.args:
    arg = node.args[0]
    return arg if isinstance(arg, torch.fx.Node) else None

  for key in ("tensor", "self", "input"):
    val = node.kwargs.get(key)
    if isinstance(val, torch.fx.Node):
      return val

  return None


def _get_reload_source_placeholder(
    node: torch.fx.Node | None,
) -> torch.fx.Node | None:
  """Unwraps views from a reload argument to find its source placeholder, if any."""
  if node is None:
    return None

  visited = set()
  current = node
  while True:
    if current in visited:
      return None
    visited.add(current)

    if current.op == "placeholder":
      return current

    if _is_ao_op(current, "offload"):
      return None

    if _is_view_node(current):
      next_node = _extract_tensor_node(current)
      if next_node is None:
        return None

      current = next_node
      continue

    return None


def _mark_pinned_host_inputs(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
) -> None:
  """Marks example inputs corresponding to offloaded activations."""
  placeholders = []
  reload_placeholders = set()
  for node in graph_module.graph.nodes:
    if node.op == "placeholder":
      placeholders.append(node)
    elif _is_ao_op(node, "reload"):
      tensor_arg = _extract_tensor_node(node)
      source_placeholder = _get_reload_source_placeholder(tensor_arg)
      if source_placeholder is not None:
        reload_placeholders.add(source_placeholder)

  for node, example_input in zip(placeholders, example_inputs):
    if isinstance(example_input, torch.Tensor) and (
        compiler.is_pinned_host_tensor(example_input)
        or node in reload_placeholders
    ):
      example_input._is_pinned_host = True


def _get_device_layouts(
    x: Any, desc: _aot_descriptors.AOTInput
) -> dict[_aot_descriptors.AOTInput, Any]:
  """Returns the non-default on-device layouts of `x`.

  AOTAutograd passes the inner tensors of a tensor subclass, e.g. the local
  tensor of a DTensor, to the program instead of the subclass itself.

  Args:
    x: An input from Dynamo, e.g. a tensor or a DTensor.
    desc: The AOTAutograd input descriptor of `x`.

  Returns:
    The non-default layouts of `x`, or of its inner tensors if it is a tensor
    subclass, keyed by AOTAutograd input descriptor.
  """
  if not _python_dispatch.is_traceable_wrapper_subclass(x):
    layout = compiler.get_non_default_device_layout(x)
    return {} if layout is None else {desc: layout}
  layouts = {}
  inner_tensor_names, _ = x.__tensor_flatten__()
  for name in inner_tensor_names:
    inner_desc = _aot_descriptors.SubclassGetAttrAOTInput(desc, name)
    layouts.update(_get_device_layouts(getattr(x, name), inner_desc))
  return layouts


def _mark_device_layouts(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    input_layouts: Mapping[_aot_descriptors.AOTInput, Any],
) -> None:
  """Marks example inputs with the layouts of the real inputs they stand for."""
  if not input_layouts:
    return
  placeholders = graph_module.graph.find_nodes(op="placeholder")
  for node, example_input in zip(placeholders, example_inputs, strict=True):
    # AOTAutograd describes which input of Dynamo's graph module, if any, each
    # placeholder is.
    layout = input_layouts.get(node.meta.get("desc"))
    if layout is not None and isinstance(example_input, torch.Tensor):
      example_input._tpu_device_layout = layout


def _process_group_cache_key_tag(graph_module: torch.fx.GraphModule) -> str:
  """Returns the devices of the process groups that `graph_module` refers to.

  Cached executables embed the devices of the groups their collectives use, but
  graphs name groups by creation order, which does not determine their members.
  """
  groups = process_group_utils.get_process_groups(graph_module)
  return ";".join(
      f"{name}:{','.join(map(str, device_mesh.get_device_assignment(pg)))}"
      for name, pg in sorted(groups.items())
  )


def _layouts_key(
    layouts: Mapping[_aot_descriptors.AOTInput, Any],
) -> frozenset[tuple[_aot_descriptors.AOTInput, Any]]:
  """Returns a hashable key for layouts, with their minor_to_major and tiles."""
  return frozenset(
      (desc, (tuple(layout[0]), tuple(map(tuple, layout[1]))))
      for desc, layout in layouts.items()
  )


@contextlib.contextmanager
def _serialization_context(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    enable: bool = False,
) -> Iterator[list[Any] | None]:
  """Optionally configures aot_autograd to produce a BundledAOTAutogradResult.

  When enabled, patches ``_cache_inference_info`` to inject a per-graph
  ``cache_info`` (so the bundled entry is created) and suppresses
  ``generate_guards_expression`` (which requires a TracingContext with a
  ShapeEnv that we cannot safely provide without leaking FakeTensorMode
  into subsequent compilations). It also tags cache keys with the devices of
  the process groups the graph uses.

  Args:
    graph_module: The Dynamo graph being compiled; its process groups tag the
      cache key.
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
    pg_tag = _process_group_cache_key_tag(graph_module)
    precision_tag = f"prec={precision_impl._get_precision().name}"  # pylint: disable=protected-access
    extra_tag = f"{pg_tag}:{precision_tag}" if pg_tag else precision_tag
    tag = compiler_config.cache_key_tag
    stack.enter_context(
        compiler_config.patch(  # pyrefly: ignore[missing-attribute]
            cache_key_tag=f"{tag}:{extra_tag}" if tag else extra_tag
        )
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
            f"{version_prefix}:{compiler_config.cache_key_tag}:"
            f"{aot_forward_graph_str or ''}:{inputs_meta}".encode()
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
    num_dynamic_buckets: int = 1,
    bucket_size: int = 128,
) -> compiler.Compiler:
  """Creates a compiler configured for static or dynamic compilation.

  Args:
    example_inputs: Example inputs to inspect for dynamic SymInts.
    async_compile: If True, executes XLA compilation asynchronously.
    debug: Optional TpuCompileDebug container for recording compilation debug
      artifacts.
    num_dynamic_buckets: Number of `bucket_size`-aligned dynamic shape buckets
      to generate when compiling with dynamic shapes.
    bucket_size: Step size for dynamic shape buckets.

  Returns:
    A SplitCompiler instance wrapping either DynamicCompiler or StaticCompiler.
  """
  has_dynamic_symints = compiler.has_dynamic_symints(example_inputs)
  if has_dynamic_symints:
    base_compiler = dynamic_compiler.DynamicCompiler(
        debug=debug,
        num_buckets=num_dynamic_buckets,
        bucket_size=bucket_size,
    )
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

    source = getattr(node, "_dynamo_source", None)
    if source is None:
      continue

    offset_source = dynamo_source.TensorPropertySource(
        source, dynamo_source.TensorProperty.STORAGE_OFFSET
    )
    guard = offset_source.make_guard(dynamo_guards.GuardBuilder.EQUALS_MATCH)
    dynamo_guards.install_guard(guard)


def _guard_matmul_precision() -> None:
  """Installs a Dynamo guard on the active matmul precision state."""
  tracing_context = torch._guards.TracingContext.try_get()
  if tracing_context is None:
    return

  get_precision_source = dynamo_source.ImportSource("torch_tpu")
  for attr in ["_internal", "precision", "precision_impl", "_get_precision"]:
    get_precision_source = dynamo_source.AttrSource(get_precision_source, attr)
  precision_source = dynamo_source.AttrSource(
      dynamo_source.CallFunctionNoArgsSource(get_precision_source),
      "value",
  )
  guard = precision_source.make_guard(dynamo_guards.GuardBuilder.EQUALS_MATCH)
  dynamo_guards.install_guard(guard)


_VALID_SYMBOL_REGEX = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_module_name(name: str) -> str:
  """Ensures name is a valid MLIR symbol (alphanumeric + underscores)."""
  return _VALID_SYMBOL_REGEX.sub("_", name)


def _get_first_node_trace(
    graph_module: torch.fx.GraphModule,
) -> torch.fx.graph._ParsedStackTrace | None:
  """Returns the parsed stack trace of the first node that has one."""
  for node in graph_module.graph.nodes:
    if node.stack_trace:
      trace = torch.fx.graph._parse_stack_trace(node.stack_trace)
      if trace and trace.file:
        return trace
  return None


def _infer_default_base_name(graph_module: torch.fx.GraphModule) -> str:
  """Infers default module name from the FX graph stack trace: <filename>_L<line>."""
  try:
    trace = _get_first_node_trace(graph_module)
    if trace and trace.file:
      return f"{pathlib.Path(trace.file).stem}_L{trace.lineno}"
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return "module"


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
      dynamism (bool): Deprecated/unused. Retained for signature compatibility.
      enable_serialization: If True, enable the aot_autograd bundled cache so
        that .serialize() is attached to compiled functions. This defaults to
        True so PyTorch's AOTAutogradCache can save/load compilation artifacts.
        Debug mode disables serialization so debug callers inspect freshly
        compiled artifacts instead of cache hits.

    Note: The `debug` and `enable_serialization` arguments are internal flags
    slated for deprecation. If set on the instance, they will override the
    corresponding keys passed in the `options` dict to `__call__`. New callers
    should pass options to `torch.compile(..., backend="tpu", options={...})`.
    """
    self._debug = debug
    self._dynamism = dynamism
    self._enable_serialization = False if debug else enable_serialization
    # Stores information about each compiled executable when self._debug is enabled.
    # Organized by order of compilation (index 0 is the first compilation, etc.)
    self._compiled_executables: list[compiler.CompiledArtifact] = []
    self._num_compiled_executables: int = 0

  def clear_cache(self) -> None:
    """Clears any cached/retained compiled executables and resets counters."""
    self._compiled_executables.clear()
    self._num_compiled_executables = 0

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
      name: Forwarded from `torch.compile(..., name=...)` (requires PyTorch >=
        2.15; older releases do not pass `name` through to custom backends).
        Custom base name for compiled modules. If unset, a name is inferred from
        the FX graph's source location (`<file>_L<line>`).
      options: Options dict to pass to the backend. Available keys:
        async_compile: Default False. If True, executes XLA compilation
          asynchronously.
        num_dynamic_buckets: Default 1. Number of `bucket_size`-aligned dynamic
          shape buckets to generate when compiling with dynamic shapes. Must be
          between 1 and 8.
        bucket_size: Default 128. Step size for dynamic shape buckets when
          compiling with dynamic shapes.
        serializable: Default True. If True, uses
          SerializableAOTDispatchCompiler so that .serialize() is attached to
          compiled functions, which enables PyTorch's AOTAutogradCache to
          save/load compilation artifacts. If False, serialization disabled so
          that callers can inspect freshly compiled artifacts instead of cache
          hits. Ignored and treated as False if dynamic symints are present.
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
    num_dynamic_buckets = options.get("num_dynamic_buckets", 1)
    num_dynamic_buckets = (
        1 if num_dynamic_buckets is None else num_dynamic_buckets
    )
    if (
        not isinstance(num_dynamic_buckets, int)
        or not 1 <= num_dynamic_buckets <= _MAX_DYNAMIC_BUCKETS
    ):
      raise ValueError(
          "`num_dynamic_buckets` must be an integer between 1 and"
          f" {_MAX_DYNAMIC_BUCKETS}, got {num_dynamic_buckets}"
      )
    bucket_size = options.get("bucket_size", 128)
    bucket_size = 128 if bucket_size is None else bucket_size
    if not isinstance(bucket_size, int) or bucket_size <= 0:
      raise ValueError(
          f"`bucket_size` must be a positive integer, got {bucket_size}"
      )
    # DynamicCompiler artifacts are not pickleable yet, so only static
    # compilations can participate in AOTAutogradCache.
    serializable = cast(
        bool, options.get("serializable", True)
    ) and not compiler.has_dynamic_symints(example_inputs)

    debug_callback = options.get("debug_callback", None)
    if debug_callback is not None and not callable(debug_callback):
      raise TypeError("'debug_callback' must be callable")
    debug = TpuCompileDebug() if debug_callback else None

    # Swallow the "mode" kwarg. This is a common kwarg that is generally ignored
    # and warned about instead of raising if it is not meaningful to the
    # backend. While other backends warn about all unknown kwargs, we check
    # "mode" specifically so we can continue to catch other unexpected kwargs.
    # See google-pytorch/torch_tpu#4298.
    mode = kwargs.pop("mode", None)
    if mode is not None and mode != "default":
      logging.warning('TPU backend ignores `mode` kwarg, got mode="%s".', mode)

    # `name` is forwarded by torch.compile(..., name=...).
    name = kwargs.pop("name", None)

    if kwargs:
      raise TypeError("Unexpected keyword arguments: {}".format(kwargs))

    # TODO(yho): The following self._* attributes are slated for deprecation.
    # Overwrite options if legacy constructor arguments were specified.
    if not self._enable_serialization or self._debug:
      serializable = self._enable_serialization
    if self._debug and debug is None:
      debug = TpuCompileDebug()
    # self._debug, self._dynamism, and self._enable_serialization should
    # NOT be accessed after this point.

    if not torch._dynamo.config.assume_static_by_default:  # pylint: disable=protected-access
      raise NotImplementedError(
          "TPU backend does not support torch.compile(..., dynamic=True, ...)."
          " Please run torch.compile() with dynamic=None or default dynamism"
          " i.e. torch.compile(model) and try again."
      )

    _log_gm_and_inputs("__call__", "Pre", graph_module, example_inputs)
    _guard_input_storage_offsets(graph_module)
    _guard_matmul_precision()
    input_layouts = {}
    # The inputs with non-default layouts, which are checked on each call.
    input_indices = []
    for i, x in enumerate(example_inputs):
      layouts = _get_device_layouts(x, _aot_descriptors.PlainAOTInput(i))
      input_layouts.update(layouts)
      if layouts:
        input_indices.append(i)
    if input_layouts:
      # AOTAutogradCache keys don't include device layouts.
      serializable = False

    custom_module_name = name or _infer_default_base_name(graph_module)

    if debug is not None:
      debug.pre_autograd_fx_code.append(graph_module.code)
      debug.pre_autograd_fx_readable.append(
          graph_module.print_readable(print_output=False)
      )

    compiler_instance = make_backend_compiler(
        example_inputs,
        async_compile=async_compile,
        debug=debug,
        num_dynamic_buckets=num_dynamic_buckets,
        bucket_size=bucket_size,
    )
    compiler_instance.execute_pre_grad_passes(graph_module)

    compiled_artifacts: list[compiler.CompiledArtifact] = []

    if serializable:
      fw_compiler = SerializableAOTDispatchCompiler(  # pyrefly: ignore[bad-specialization]
          output_code_ty=compiler.CompiledArtifact,
          compiler_fn=functools.partial(  # pyrefly: ignore[bad-specialization]
              self._compile_graph_module,
              compiler_instance,
              True,
              custom_module_name,
              debug,
              compiled_artifacts,
              input_layouts,
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
          input_layouts,
      )

    bw_compiler = functools.partial(
        self._compile_graph_module,
        compiler_instance,
        False,
        custom_module_name,
        debug,
        compiled_artifacts,
        input_layouts,
    )

    save_context = (
        _cache_saves_manager.async_aot_cache_saves()
        if async_compile
        else contextlib.nullcontext()
    )
    with _serialization_context(
        graph_module, example_inputs, serializable
    ) as captured_entry:
      with save_context:
        with execution_mode.set_eager_mode(
            execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
        ):
          result = aot_autograd(
              fw_compiler=fw_compiler,
              bw_compiler=bw_compiler,
              partition_fn=min_cut_rematerialization_partition,
              decompositions=_TPU_DECOMPOSITIONS,  # pyrefly: ignore[bad-argument-type]
              keep_inference_input_mutations=False,
          )(graph_module, example_inputs)

      if input_indices:
        # An input can lose its layout after compilation, e.g. copy_() and
        # load_state_dict() give a tensor the default layout. Like Dynamo does
        # for new shapes, compile again for the new layouts of these inputs.
        programs = {_layouts_key(input_layouts): result}

        def call_program_for_layouts(*args: Any, **kwargs: Any) -> Any:
          # A tensor's layout is known once it starts materializing.
          sync.synchronize([args[i] for i in input_indices])
          layouts = {}
          for i in input_indices:
            layouts.update(
                _get_device_layouts(args[i], _aot_descriptors.PlainAOTInput(i))
            )
          key = _layouts_key(layouts)
          if key not in programs:
            logging.warning(
                "Compiling %s again for new on-device layouts of its inputs.",
                custom_module_name,
            )
            programs[key] = self(
                graph_module,
                list(args),
                options={
                    **options,
                    "async_compile": False,
                    "serializable": False,
                },
                name=name,
            )
          return programs[key](*args, **kwargs)

        result = call_program_for_layouts

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
          artifacts_to_wait = list(compiled_artifacts)

          # Ensure callbacks that fail are visible
          def wait_and_call():
            try:
              for artifact in artifacts_to_wait:
                artifact.resolve()
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
      input_layouts: Mapping[_aot_descriptors.AOTInput, Any],
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[torch.Tensor],
  ) -> Callable[..., Any]:
    """Compiles the graph_module with the given inputs for TPU.

    torch.compile() will generate a graph_module and call this function to
    finish the compilation.

    Args:
      compiler_instance: Responsible for compiling the graph module.
      is_fwd: Indicates whether the forward or backward pass is being compiled.
      custom_module_name: The base module name from `torch.compile(name=...)` or
        inferred from the FX graph.
      debug: Container for recording compilation debug artifacts.
      compiled_artifacts: Per-call container tracking compiled artifacts.
      input_layouts: The non-default on-device layouts of the inputs from
        Dynamo, keyed by AOTAutograd input descriptors.
      graph_module: The FX graph module to compile.
      example_inputs: Example inputs to the FX graph for tracing (not the actual
        inputs).

    Returns:
      A function that executes the compiled graph on the TPU.
    """
    _mark_pinned_host_inputs(graph_module, example_inputs)
    _mark_device_layouts(graph_module, example_inputs, input_layouts)

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

    split_idx = self._num_compiled_executables
    self._num_compiled_executables += 1
    module_name = _get_module_name(custom_module_name, is_fwd, split_idx)

    executable = compiler_instance(
        graph_module, example_inputs, is_fwd, module_name=module_name
    )
    compiled_artifacts.append(executable)
    if self._debug:
      self._compiled_executables.append(executable)
    if debug is not None:
      debug.compiled_executables = list(compiled_artifacts)

    return executable


def clear_cache() -> None:
  """Flushes pending AOTAutograd cache saves and clears TpuBackend state."""
  _cache_saves_manager.flush()
  if "tpu" in registry.list_backends():
    backend_fn = registry.lookup_backend("tpu")
    if isinstance(backend_fn, TpuBackend):
      backend_fn.clear_cache()


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
