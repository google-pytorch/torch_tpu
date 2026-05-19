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

# pylint: disable=protected-access

from collections.abc import Callable, Iterator, Sequence
import contextlib
import functools
import hashlib
import operator
from typing import Any, TypeAlias

from absl import logging
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd import autograd_cache as _autograd_cache
from torch._functorch._aot_autograd import graph_compile as _graph_compile
from torch._functorch._aot_autograd.schemas import AOTAutogradCacheInfo
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler
import torch._functorch.config as functorch_config
import torch._functorch.partitioners
from torch._guards import detect_fake_mode
from torch._prims.rng_prims import graphsafe_run_with_rng_state
from torch._prims.rng_prims import run_and_save_rng_state
from torch._prims.rng_prims import run_with_rng_state
from torch._prims.rng_prims import track_tensor_tree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.utils import _pytree
from torch_tpu._internal.compile.compiler import CompiledArtifact
from torch_tpu._internal.compile.compiler import Compiler
from torch_tpu._internal.compile.compiler import has_dynamic_symints
from torch_tpu._internal.compile.compiler import StaticCompiler
from torch_tpu._internal.compile.dynamic import compiler as dynamic_compiler
from torch_tpu._internal.utils import utils


def _custom_functionalize_rng_ops(
    joint_module, fw_module, bw_module, num_fwd_outputs
):
  """Custom functionalize RNG ops wrapper."""
  orig_fn = getattr(
      torch._functorch.partitioners, "_orig_functionalize_rng_ops"
  )

  fw_module, bw_module = orig_fn(
      joint_module, fw_module, bw_module, num_fwd_outputs
  )

  fake_mode = detect_fake_mode()
  if fake_mode is not None:
    with fake_mode:
      tpu_val = fake_mode.from_tensor(
          torch.zeros(632, dtype=torch.uint8, device="tpu")
      )
  else:
    tpu_val = torch.zeros(632, dtype=torch.uint8, device="tpu")

  for node in fw_module.graph.nodes:
    if node.op == "call_function" and node.target == operator.getitem:
      if (
          node.args[0].op == "call_function"
          and node.args[0].target == run_and_save_rng_state
      ):
        if node.args[1] == 0:
          node.meta["val"] = tpu_val

  bw_rng_placeholders = set()
  for node in bw_module.graph.nodes:
    if node.op == "call_function" and node.target == run_with_rng_state:
      bw_rng_placeholders.add(node.args[0])

  for ph in bw_rng_placeholders:
    ph.meta["val"] = tpu_val

  return fw_module, bw_module


if not hasattr(torch._functorch.partitioners, "_orig_functionalize_rng_ops"):
  torch._functorch.partitioners._orig_functionalize_rng_ops = (
      torch._functorch.partitioners.functionalize_rng_ops
  )
  torch._functorch.partitioners.functionalize_rng_ops = (
      _custom_functionalize_rng_ops
  )


def _run_and_save_impl_tpu(op, *args, **kwargs):
  """TPU implementation for run_and_save_rng_state."""
  from torch_tpu._internal.compile import tpu_torch_compile  # pylint: disable=g-import-not-at-top

  device = torch.accelerator.current_accelerator()
  device_module = getattr(torch, device.type)
  gen = device_module.default_generators[device.index or 0]
  tpu_state = tpu_torch_compile.get_device_state_tensor(gen)
  tpu_state_bytes = tpu_state.view(torch.uint8)
  padding = torch.zeros(616, dtype=torch.uint8, device=tpu_state.device)
  padded_state = torch.cat([tpu_state_bytes, padding], dim=0)
  return padded_state, op(*args, **kwargs)


run_and_save_rng_state.py_impl(torch._C.DispatchKey.PrivateUse1)(
    _run_and_save_impl_tpu
)

_orig_run_and_save_backend_select = run_and_save_rng_state.py_kernels[
    torch._C.DispatchKey.BackendSelect
]


def _custom_run_and_save_backend_select(op, *args, **kwargs):
  """Custom BackendSelect for run_and_save_rng_state."""
  devices = {arg.device.type for arg in args if isinstance(arg, torch.Tensor)}
  if any(dev == "tpu" for dev in devices):
    return _run_and_save_impl_tpu(op, *args, **kwargs)
  return _orig_run_and_save_backend_select(op, *args, **kwargs)


run_and_save_rng_state.py_kernels[torch._C.DispatchKey.BackendSelect] = (
    _custom_run_and_save_backend_select
)


def _custom_run_and_save_fake_mode(mode, op, *args, **kwargs):
  """Custom FakeTensorMode for run_and_save_rng_state."""
  with mode:
    return _custom_run_and_save_backend_select(op, *args, **kwargs)


def _custom_run_and_save_proxy_mode(mode, op, *args, **kwargs):
  """Custom ProxyTorchDispatchMode for run_and_save_rng_state."""
  out = _custom_run_and_save_backend_select(op, *args, **kwargs)
  proxy_args = _pytree.tree_map(mode.tracer.unwrap_proxy, (op, *args))
  proxy_kwargs = _pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
  out_proxy = mode.tracer.create_proxy(
      "call_function", run_and_save_rng_state, proxy_args, proxy_kwargs
  )
  return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


run_and_save_rng_state.python_key_table[FakeTensorMode] = (
    _custom_run_and_save_fake_mode
)
run_and_save_rng_state.python_key_table[ProxyTorchDispatchMode] = (
    _custom_run_and_save_proxy_mode
)


def _run_with_impl_tpu(rng_state, op, *args, **kwargs):
  """TPU implementation for run_with_rng_state."""
  from torch_tpu._internal.compile import tpu_torch_compile  # pylint: disable=g-import-not-at-top

  device = torch.accelerator.current_accelerator()
  device_module = getattr(torch, device.type)
  gen = device_module.default_generators[device.index or 0]
  current_state = tpu_torch_compile.get_device_state_tensor(gen)
  tpu_state_bytes = rng_state[:16]
  tpu_state = tpu_state_bytes.view(torch.uint64)
  tpu_torch_compile.set_device_state_tensor(gen, tpu_state)
  out = op(*args, **kwargs)
  tpu_torch_compile.set_device_state_tensor(gen, current_state)
  return out


run_with_rng_state.py_impl(torch._C.DispatchKey.PrivateUse1)(_run_with_impl_tpu)

_orig_run_with_backend_select = run_with_rng_state.py_kernels[
    torch._C.DispatchKey.BackendSelect
]


def _custom_run_with_backend_select(rng_state, op, *args, **kwargs):
  """Custom BackendSelect for run_with_rng_state."""
  devices = {arg.device.type for arg in args if isinstance(arg, torch.Tensor)}
  if any(dev == "tpu" for dev in devices) or rng_state.device.type == "tpu":
    return _run_with_impl_tpu(rng_state, op, *args, **kwargs)
  return _orig_run_with_backend_select(rng_state, op, *args, **kwargs)


run_with_rng_state.py_kernels[torch._C.DispatchKey.BackendSelect] = (
    _custom_run_with_backend_select
)


def _graphsafe_run_with_impl_tpu(op, *args, rng_state=None, **kwargs):
  """TPU implementation for graphsafe_run_with_rng_state."""
  device_idx = rng_state.device.index
  generator = torch.tpu.default_generators[device_idx]
  current_state = generator.graphsafe_get_state()

  generator.graphsafe_set_state(rng_state)
  out = op(*args, **kwargs)
  generator.graphsafe_set_state(current_state)
  return out


graphsafe_run_with_rng_state.py_impl(torch._C.DispatchKey.PrivateUse1)(
    _graphsafe_run_with_impl_tpu
)

_orig_graphsafe_backend_select = graphsafe_run_with_rng_state.py_kernels[
    torch._C.DispatchKey.BackendSelect
]


def _custom_graphsafe_backend_select(op, *args, rng_state=None, **kwargs):
  """Custom BackendSelect for graphsafe_run_with_rng_state."""
  devices = {arg.device.type for arg in args if isinstance(arg, torch.Tensor)}
  if any(dev == "tpu" for dev in devices) or (
      rng_state is not None and rng_state.device.type == "tpu"
  ):
    return _graphsafe_run_with_impl_tpu(
        op, *args, rng_state=rng_state, **kwargs
    )
  return _orig_graphsafe_backend_select(
      op, *args, rng_state=rng_state, **kwargs
  )


graphsafe_run_with_rng_state.py_kernels[torch._C.DispatchKey.BackendSelect] = (
    _custom_graphsafe_backend_select
)

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
        aot_config.cache_info = AOTAutogradCacheInfo(
            cache_key=f"torchtpu_{graph_hash}",
            start_time_ns=0,
            forward_symints=[],
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
        aot_config.cache_info = None
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
      enable_serialization: bool = False,
  ):
    """Initializes the TPU backend.

    Args:
      debug (bool): If True, enable debug logging and save a dump of the fx
        graph.
      dynamism (bool): If True, enable dynamism.
      enable_serialization: If True, enable the aot_autograd bundled cache so
        that .serialize() is attached to compiled functions. This allows
        compilation artifacts to be saved/loaded without re-running
        aot_autograd.
    """
    self._debug = debug
    self._dynamism = dynamism
    self._enable_serialization = enable_serialization
    # Stores information about each compiled executable.
    # Organized by order of compilation (index 0 is the first compilation, etc.)
    self._compiled_executables: list[CompiledArtifact] = []

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

    if has_dynamic_symints(example_inputs):
      compiler_instance = dynamic_compiler.DynamicCompiler(debug=self._debug)
    else:
      compiler_instance = StaticCompiler(debug=self._debug)

    compiler_instance.execute_pre_grad_passes(graph_module)

    if self._enable_serialization:
      fw_compiler = SerializableAOTDispatchCompiler(
          output_code_ty=CompiledArtifact,
          compiler_fn=functools.partial(
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

    with _serialization_context(self._enable_serialization) as captured_entry:
      result = aot_autograd(
          fw_compiler=fw_compiler,
          bw_compiler=bw_compiler,
          keep_inference_input_mutations=False,
      )(graph_module, example_inputs)

      if captured_entry is not None and captured_entry[0] is not None:
        entry = captured_entry[0]
        result.serialize = lambda: entry

      return result

  def _compile_graph_module(
      self,
      compiler_instance: Compiler,
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

    executable = compiler_instance(graph_module, example_inputs, is_fwd=is_fwd)

    self._compiled_executables.append(executable)

    return executable
