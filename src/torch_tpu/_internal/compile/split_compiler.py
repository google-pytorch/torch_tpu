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

"""Compiler split functionality for TorchTPU.

This module contains the logic for splitting a GraphModule on collective
operations and device boundaries, compiling its TPU submodules, and running its
CPU submodules eagerly.
"""

from collections.abc import Callable, Collection, Iterator, Sequence
import contextlib
import copy
import functools
import operator
import threading
from typing import Any

from absl import logging
import torch
from torch._dynamo.utils import detect_fake_mode
from torch._inductor.utils import InputType
from torch._logging import LazyString
from torch._subclasses.fake_tensor import unset_fake_temporarily
import torch.distributed as dist
from torch.fx.experimental import symbolic_shapes
from torch.fx.passes import graph_transform_observer
from torch.fx.passes.split_module import split_module
from torch.utils import _pytree as pytree
from torch_tpu._internal import env
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import torch_tpu_compiled_executable
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.fx_passes import clone_mutated_returned_placeholders
from torch_tpu._internal.compile.fx_passes import force_collectives_output
from torch_tpu._internal.compile.fx_passes import mark_embedded_constants
from torch_tpu._internal.compile.fx_passes import propagate_symints
from torch_tpu._internal.compile.fx_passes import reorder_symints
from torch_tpu._internal.compile.fx_passes import sink_get_attr_constants
from torch_tpu._internal.distributed import collective_ops
from torch_tpu._internal.distributed import handshake
from torch_tpu._internal.distributed import process_group_utils
from torch_tpu._internal.distributed import spmd_util

ProcessGroupId = handshake.ProcessGroupId
RankCollectiveCounts = handshake.RankCollectiveCounts
AsyncCompiledArtifact = torch_tpu_compiled_executable.AsyncCompiledArtifact
CollectiveHandshakeRequest = handshake.CollectiveHandshakeRequest
CompiledArtifact = torch_tpu_compiled_executable.CompiledArtifact
TorchTpuCompiledExecutable = (
    torch_tpu_compiled_executable.TorchTpuCompiledExecutable
)
Handshake = handshake.Handshake

# We define a per process global collective count to keep track of the
# collectives that have been submitted to the split compiler. This is used to
# determine if a handshake is needed and to submit the handshake message.
_RANK_COLLECTIVE_COUNTS: RankCollectiveCounts = RankCollectiveCounts()
_RANK_COLLECTIVE_COUNTS_LOCK = threading.Lock()


def _submit_handshake(
    executable_fingerprint: str,
    pg_to_num_collectives: dict[ProcessGroupId, int],
    graph_module_str: str | None = None,
) -> bool:
  """Submits a handshake message across participating ranks.

  Args:
    executable_fingerprint: The fingerprint of the executable.
    pg_to_num_collectives: The mapping of ProcessGroupId to collective counts.
    graph_module_str: The graph module string representation, for logging
      purposes.

  Returns True if all ranks match.
  """
  assert (
      dist.is_initialized()
  ), "Distributed backend must be initialized when submitting a handshake."

  with torch.profiler.record_function("split_compiler._submit_handshake"):
    with _RANK_COLLECTIVE_COUNTS_LOCK:
      _RANK_COLLECTIVE_COUNTS.increment_pg_collective_counts(
          pg_to_num_collectives
      )
      pg_collective_counts = copy.deepcopy(_RANK_COLLECTIVE_COUNTS)

    rank = dist.get_rank()

    msg = CollectiveHandshakeRequest(
        pg_collective_counts=pg_collective_counts,
        executable_fingerprint=executable_fingerprint,
        rank=rank,
    )
    result = Handshake().submit(msg)

    if result:
      logging.info(
          "Handshake succeeded for rank %d with executable fingerprint %s and"
          " pg_to_num_collectives %s",
          rank,
          executable_fingerprint,
          pg_to_num_collectives,
      )
    else:
      logging.warning(
          "Handshake failed for rank %d with executable fingerprint %s and"
          " pg_to_num_collectives %s",
          rank,
          executable_fingerprint,
          pg_to_num_collectives,
      )
      if graph_module_str:
        logging.warning(
            "Handshake failed on rank %d for graph module:\n"
            "--- [BEGIN GRAPH MODULE: %s] ---\n"
            "%s\n"
            "--- [END GRAPH MODULE: %s] ---",
            rank,
            executable_fingerprint,
            graph_module_str,
            executable_fingerprint,
        )

    return result


def _get_unique_wait_tensor_producer(
    node: torch.fx.Node,
) -> torch.fx.Node:
  """Returns the unique wait tensor producer node if it exists.

  If the wait_tensor consumer is separated from its collective producer by
  coalesced operations (operator.getitem) this function traverses up to return
  the collective node.

  Args:
    node: The wait_tensor node to find the producer for.

  Returns:
    The unique wait tensor producer node which HAS TO BE a collective node.
  Raises:
    ValueError: When:
    - the node is not a wait_tensor node
    - the node does not have exactly one argument
    - the argument is not a call_function node
    - the argument is not a collective node
  """

  if (
      getattr(node.target, "overloadpacket", node.target)
      != torch.ops._c10d_functional.wait_tensor  # pylint: disable=protected-access
  ):
    raise ValueError(f"Expected wait_tensor for node, got {node}")

  if len(node.args) != 1:
    raise ValueError(
        f"Expected exactly one argument for wait_tensor, got {len(node.args)}"
    )
  producer = node.args[0]
  # Move up the graph through getitem nodes.
  while (
      isinstance(producer, torch.fx.Node)
      and producer.op == "call_function"
      and (
          producer.target == operator.getitem
          or getattr(producer.target, "__name__", "") == "getitem"
      )
  ):
    if len(producer.args) == 0:
      break
    producer = producer.args[0]

  if not isinstance(producer, torch.fx.Node):
    raise ValueError(
        f"Expected node for wait_tensor producer, got {type(producer)}"
    )

  if producer.op != "call_function":
    raise ValueError(
        f"Expected call_function for wait_tensor producer, got {producer.op}"
    )
  if not collective_ops.is_collective_op(producer.target):
    raise ValueError(
        f"Expected collective for wait_tensor producer, got {producer.target}"
    )
  return producer


def _is_getitem_of_collective(node: torch.fx.Node) -> bool:
  """Checks whether node is a getitem extracting from a collective."""
  if node.op != "call_function" or not (
      node.target == operator.getitem
      or getattr(node.target, "__name__", "") == "getitem"
  ):
    return False
  if len(node.args) == 0 or not isinstance(node.args[0], torch.fx.Node):
    return False
  curr = node.args[0]
  while (
      isinstance(curr, torch.fx.Node)
      and curr.op == "call_function"
      and (
          curr.target == operator.getitem
          or getattr(curr.target, "__name__", "") == "getitem"
      )
  ):
    if len(curr.args) == 0:
      break
    curr = curr.args[0]
  if isinstance(curr, torch.fx.Node) and collective_ops.is_collective_op(
      curr.target
  ):
    return True
  return False


def _collect_namedtuple_types(arg: Any, seen: dict[type[Any], None]) -> None:
  """Recursively collects namedtuple types from an FX node argument."""
  if isinstance(arg, tuple) and hasattr(arg, "_fields"):
    seen.setdefault(type(arg), None)
    for elem in arg:
      _collect_namedtuple_types(elem, seen)
  elif isinstance(arg, (tuple, list)):
    for elem in arg:
      _collect_namedtuple_types(elem, seen)
  elif isinstance(arg, dict):
    for elem in arg.values():
      _collect_namedtuple_types(elem, seen)


# TODO: b/576283069 - Remove once minimum OSS PyTorch is >= 2.12.0.
def _register_output_namedtuple_globals(gm: torch.fx.GraphModule) -> None:
  """Registers namedtuple output types in the FX graph's codegen globals.

  In PyTorch < 2.12, `torch.fx.graph.CodeGen.generate_output` formats
  namedtuple outputs using `repr(output_args)` (e.g.
  `return Single(sum=submod_0)`) without registering the namedtuple class in
  `PythonCode.globals`, which raises a `NameError` at runtime when the
  namedtuple class is not in global scope.
  """
  namedtuple_types: dict[type[Any], None] = {}
  for node in gm.graph.find_nodes(op="output"):
    _collect_namedtuple_types(node.args, namedtuple_types)
  if not namedtuple_types:
    return

  codegen = gm.graph._codegen  # pylint: disable=protected-access
  orig_additional_globals = codegen.additional_globals
  extra_globals = [(cls.__name__, cls) for cls in namedtuple_types]
  codegen.additional_globals = (
      lambda: list(orig_additional_globals()) + extra_globals
  )


_CPU = "cpu"
_TPU = "tpu"

# Ops with these tags are never constant expressions, even if all their inputs
# are constants: tracing them into a TPU program would change their random
# stream or require concrete input values.
_NON_CONSTANT_OP_TAGS = frozenset({
    torch.Tag.nondeterministic_seeded,
    torch.Tag.data_dependent_output,
    torch.Tag.dynamic_output_shape,
})


def _tensor_device_types(node: torch.fx.Node) -> set[str]:
  """Returns the device types of the tensors in the value of `node`."""
  return {
      leaf.device.type
      for leaf in pytree.tree_leaves(node.meta.get("val"))
      if isinstance(leaf, torch.Tensor)
  }


def _assign_devices(graph: torch.fx.Graph) -> dict[torch.fx.Node, str | None]:
  """Returns the device type each node must run on, or None if any works.

  Nodes that produce CPU tensors run eagerly on CPU, as do nodes that produce no
  tensors from CPU tensors or values computed on CPU, e.g. `.item()` and
  arithmetic on its result. Other nodes that produce tensors, or that have no
  value metadata, run on TPU. Expressions computed only from `get_attr`
  constants are unconstrained since TPU programs embed constants, as are other
  nodes that touch no tensors, such as SymInt arithmetic. An unconstrained node
  runs on CPU if any of its users does.

  Tensors offloaded by `torch.ops.ao.offload`, and tensors computed from them,
  also run on TPU: fake tensors report them as CPU tensors, but TPU programs
  keep them in pinned host memory.

  Args:
    graph: The graph to assign devices to.

  Returns:
    A mapping from each node other than placeholders and outputs to "cpu",
    "tpu", or None.
  """
  # Registered when torch._functorch._activation_offloading is imported.
  offload = getattr(torch.ops.ao, "offload", None)
  constants = set()
  pinned_host = set()
  devices = {}
  for node in graph.nodes:
    if node.op in ("placeholder", "output"):
      continue
    outputs = _tensor_device_types(node)
    op = getattr(node.target, "overloadpacket", node.target)
    if (offload is not None and op is offload) or (
        _CPU in outputs and not pinned_host.isdisjoint(node.all_input_nodes)
    ):
      pinned_host.add(node)
      devices[node] = _TPU
      continue
    inputs = set().union(*(
        _tensor_device_types(n)
        for n in node.all_input_nodes
        if n not in pinned_host
    ))
    if node.op == "get_attr" or (
        node.all_input_nodes
        and constants.issuperset(node.all_input_nodes)
        and len(inputs | outputs) <= 1
        and not _NON_CONSTANT_OP_TAGS.intersection(
            getattr(node.target, "tags", ())
        )
    ):
      constants.add(node)
      devices[node] = None
    elif _CPU in outputs or (
        not outputs
        and (_CPU in inputs or _CPU in map(devices.get, node.all_input_nodes))
    ):
      devices[node] = _CPU
    else:
      devices[node] = _TPU if outputs or "val" not in node.meta else None

  for node in reversed(devices):
    if devices[node] is None and any(
        devices.get(user) == _CPU
        or (user.op == "output" and _CPU in _tensor_device_types(node))
        for user in node.users
    ):
      devices[node] = _CPU
  return devices


def _call_partitions_in_order(split_gm: torch.fx.GraphModule) -> None:
  """Reorders the submodule calls of a split graph by partition id.

  split_module only orders partitions by their data dependencies, so e.g. CPU
  partitions could otherwise call random ops in a different order than eager.

  Args:
    split_gm: A graph split by `split_module` with partition ids that respect
      the dependencies between partitions.
  """
  calls = split_gm.graph.find_nodes(op="call_module")
  if not calls:
    return
  anchor = calls[0].prev
  for call in sorted(calls, key=lambda n: int(str(n.target).split("_")[-1])):
    unpacked = [u for u in call.users if u.target is operator.getitem]
    for node in (call, *unpacked):
      anchor.append(node)
      anchor = node


class _WrapperModule(torch.nn.Module):
  """Wrapper module for compiled submodules."""

  # Positions of arguments that are CPU tensors, e.g. outputs of CPU partitions.
  # TPU executables only accept TPU tensors, so these are transferred first.
  cpu_arg_indices: tuple[int, ...] = ()

  def __init__(
      self,
      submod: Callable[..., Any],
      unwrap_singleton_tuple: bool,
      cpu_arg_indices: Sequence[int] = (),
  ):
    super().__init__()
    self.submod = submod
    self.unwrap_singleton_tuple = unwrap_singleton_tuple
    self.cpu_arg_indices = tuple(cpu_arg_indices)

  def forward(self, *args: Any) -> Any:
    if self.cpu_arg_indices:
      device = torch.accelerator.current_accelerator()
      args = tuple(
          arg.to(device) if i in self.cpu_arg_indices else arg
          for i, arg in enumerate(args)
      )
    x = self.submod(*args)
    # AOT Autograd compiled submodules always wrap their output in a tuple.
    # If the original subgraph did not return a tuple, we unwrap the singleton
    # tuple here to match the parent graph's schema.
    if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
      return x[0]
    return x


@contextlib.contextmanager
def _allow_non_fake_inputs(
    fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,  # pylint: disable=protected-access
) -> Iterator[None]:
  """Temporarily lets `fake_mode` accept real tensors alongside fake ones.

  Submodules produced by `split_module` keep their constant weights as real
  `get_attr` tensors, so tracing them under a fake mode mixes real and fake
  operands. This relaxes the check for the duration of the block only,
  following the example of `ExternKernel._allow_non_fake_constant_args` in
  `torch._inductor.ir`.

  Args:
    fake_mode: The fake mode to relax.

  Yields:
    None.
  """
  previous = fake_mode.allow_non_fake_inputs
  fake_mode.allow_non_fake_inputs = True
  try:
    yield
  finally:
    fake_mode.allow_non_fake_inputs = previous


class _SubmodCompiler(torch.fx.interpreter.Interpreter):
  """Interpreter to propagate shape information and compile submodules.

  This class traverses the topologically-sorted split GraphModule sequentially,
  compiles all TPU submodules and propagates input/output shapes from one
  submodule to the next. CPU submodules are left uncompiled to run eagerly.
  """

  def __init__(
      self,
      module: torch.fx.GraphModule,
      compiler_fn: Callable[..., CompiledArtifact],
      fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,
      base_module_name: str | None = None,
      cpu_submods: Collection[str] = frozenset(),
  ):
    super().__init__(module)
    self.compiler_fn = compiler_fn
    self.fake_mode = fake_mode
    self.base_module_name = base_module_name
    self.cpu_submods = cpu_submods
    self.has_multiple_submods = (
        sum(1 for n in module.graph.nodes if n.op == "call_module") > 1
    )

  # This logic is copied from torch/_functorch/_aot_autograd/frontend_utils.py
  def _convert_to_fake_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
    symbolic_context = None
    source = None
    if tracing_context := torch._guards.TracingContext.try_get():
      if tensor in tracing_context.tensor_to_context:
        symbolic_context = tracing_context.tensor_to_context[tensor]
        source = symbolic_context.tensor_source

    # If there's no symbolic context from Dynamo, treat as static
    # weight/buffer/constant
    if not symbolic_context:
      return self.fake_mode.from_tensor(tensor, static_shapes=True)

    return self.fake_mode.from_tensor(
        tensor,
        static_shapes=False,
        symbolic_context=symbolic_context,
        source=source,
    )

  def compile_submod(
      self,
      input_mod: torch.fx.GraphModule,
      args: list[torch.Tensor],
      kwargs: Any,
      submod_name: str | None = None,
  ) -> Any:
    """Compiles a single submodule into a PJRT executable wrapper."""
    if len(kwargs) != 0:
      raise AssertionError("We assume only args for these modules")

    # AOT Autograd requires all outputs to be wrapped in a tuple to preserve
    # target specifications. If a submodule's output is a single tensor, we
    # temporarily wrap it into a singleton tuple before compiling.
    unwrap_singleton_tuple = False
    for sn in input_mod.graph.nodes:
      if sn.op == "output":
        if not isinstance(sn.args[0], tuple):
          unwrap_singleton_tuple = True
          sn.args = (sn.args,)

    input_mod.recompile()
    full_submod_name = self.base_module_name
    if self.base_module_name and submod_name and self.has_multiple_submods:
      full_submod_name = f"{self.base_module_name}_{submod_name}"

    cpu_arg_indices = [
        i
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor) and arg.device.type == _CPU
    ]
    if symbolic_shapes.has_free_unbacked_symbols(
        [args[i] for i in cpu_arg_indices]
    ):
      raise NotImplementedError(
          f"{submod_name} runs on TPU but takes CPU tensors with data-dependent"
          " shapes, e.g. from `torch.nonzero`, which TPU programs cannot be"
          " compiled for. Let TorchDynamo graph break on such ops instead by"
          " disabling `torch._dynamo.config.capture_dynamic_output_shape_ops`"
          " and `capture_scalar_outputs`."
      )

    compiled_submod_real = self.compiler_fn(
        input_mod, args, module_name=full_submod_name
    )
    wrapper = _WrapperModule(
        compiled_submod_real, unwrap_singleton_tuple, cpu_arg_indices
    )
    return wrapper

  def run_node(self, n: torch.fx.Node) -> Any:
    """Intercepts submodule calls to compile them and propagate shapes."""
    args, kwargs = self.fetch_args_kwargs_from_env(n)
    new_args = []
    if not self.fake_mode:
      raise AssertionError("fake_mode must be set")

    # Use FakeTensors for tracing.
    for arg in args:
      if isinstance(arg, torch.Tensor) and not isinstance(
          arg, torch._subclasses.FakeTensor  # pylint: disable=protected-access
      ):
        new_args.append(self._convert_to_fake_tensor(arg))
      else:
        new_args.append(arg)

    if n.op == "call_module" and n.target in self.cpu_submods:
      # CPU submodules run eagerly, so only their output shapes are propagated.
      with self.fake_mode, _allow_non_fake_inputs(self.fake_mode):
        return self.fetch_attr(str(n.target))(*new_args, **kwargs)

    if n.op == "call_module":
      real_mod = self.fetch_attr(str(n.target))

      # Copy the submodule's graph before compiling to prevent compiler-time
      # metadata mutation and structural corruption of nodes/inputs. Only the
      # graph structure is copied; tensor attributes are shared by reference so
      # that large weights are not duplicated and constant tensors keep their
      # underlying device buffers.
      with unset_fake_temporarily():
        comp_mod = torch.fx.GraphModule(real_mod, copy.deepcopy(real_mod.graph))
        # Preserve compiler annotations without deep-copying tensor attributes.
        comp_mod.meta = copy.copy(real_mod.meta)
      compiled_submod_real = self.compile_submod(
          comp_mod, new_args, kwargs, submod_name=str(n.target)
      )

      # Propagate fake output shapes to downstream submodules. The submodule's
      # sunk-in constants are real (not fake) tensors, so the fake mode is
      # relaxed to accept them.
      with self.fake_mode, _allow_non_fake_inputs(self.fake_mode):
        fake_out = real_mod(*new_args, **kwargs)

      # Stitch the compiled submodule back into the parent GraphModule
      self.module.delete_submodule(n.target)
      n.target = "compiled_" + n.target  # pyrefly:ignore [unsupported-operation]
      self.module.add_submodule(n.target, compiled_submod_real)
      return fake_out
    else:
      return getattr(self, n.op)(n.target, new_args, kwargs)


class _SplitCompiledExecutable(CompiledArtifact):
  """A CompiledArtifact supporting split submodules."""

  def __init__(
      self,
      target: torch.fx.GraphModule,
      recompile_fn: Callable[[], "_SplitCompiledExecutable"] | None = None,
      pg_to_num_collectives: dict[ProcessGroupId, int] | None = None,
      graph_module_str: str | None = None,
  ):
    """Initializes a _SplitCompiledExecutable.

    Args:
      target: The split GraphModule.
      recompile_fn: Optional callable for recompiling the graph.
      pg_to_num_collectives: Mapping of ProcessGroupId to collective counts for
        this executable.
      graph_module_str: The optional unpartitioned graph module string for
        logging.
    """
    super().__init__()
    self.recompile_fn = recompile_fn
    self.pg_to_num_collectives = (
        pg_to_num_collectives if pg_to_num_collectives else {}
    )
    self._graph_module_str = graph_module_str

    self._split_gm = target

  def _maybe_handshake_and_recompile(self) -> None:
    """Verifies rank consensus on dispatch when running in DISPATCH_STAGE."""
    if self.recompile_fn is None:
      return

    self.resolve()
    executable_fingerprint = self.fingerprint()
    assert executable_fingerprint, "Executable fingerprint is empty."

    fingerprints_match = _submit_handshake(
        executable_fingerprint,
        self.pg_to_num_collectives,
        self._graph_module_str,
    )

    if not fingerprints_match:
      new_exec = self.recompile_fn()
      new_exec.resolve()
      self.__dict__.clear()
      self.__dict__.update(new_exec.__dict__)

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    self._maybe_handshake_and_recompile()

    if len(args) == 1 and isinstance(args[0], (list, tuple)):
      args = args[0]  # pyrefly: ignore[bad-assignment]
    return self._split_gm(*args)

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Enables pickling of the _SplitCompiledExecutable.

    This method is part of Python's pickle protocol. It returns a tuple
    containing:
      1. A callable (_unpickle_split_compiled_executable) that can be called to
         recreate the object.
      2. A tuple of arguments containing target, recompile_fn,
         pg_to_num_collectives, and graph_module_str.

    Returns:
      A tuple (callable, args_tuple) used by the pickle module to
      serialize the object.
    """
    self.resolve()
    return (
        _unpickle_split_compiled_executable,
        (
            self._split_gm,
            self.recompile_fn,
            self.pg_to_num_collectives,
            self._graph_module_str,
        ),
    )

  @property
  def compiled_executables(self) -> Sequence[CompiledArtifact]:
    """Returns the list of underlying compiled executables."""
    return [
        module.submod
        for module in self._split_gm.children()
        if isinstance(module, _WrapperModule)
        and isinstance(module.submod, CompiledArtifact)
    ]

  @property
  def graph_module_debug_strs(self) -> Sequence[str]:
    """List of string representations of the FX graph module's code for each submodule."""
    return [
        s
        for exe in self.compiled_executables
        for s in exe.graph_module_debug_strs
    ]

  @property
  def mlir_texts(self) -> Sequence[str]:
    """List of MLIR text representations of the compiled submodule's code."""
    return [
        text for exe in self.compiled_executables for text in exe.mlir_texts
    ]

  def updates_default_generator_state(self) -> bool:
    return any(
        exe.updates_default_generator_state()
        for exe in self.compiled_executables
    )

  def resolve(self) -> None:
    """Waits for any pending background compilation in submodules to complete."""
    for exe in self.compiled_executables:
      exe.resolve()

  def _resolve(self) -> None:
    self.resolve()

  @property
  def is_resolved(self) -> bool:
    """Returns True if all submodule compilations are resolved."""
    return all(exe.is_resolved for exe in self.compiled_executables)

  def fingerprint(self) -> str:
    fps = [
        e.fingerprint() for e in self.compiled_executables if e.fingerprint()
    ]
    return ":".join(fps)


def _unpickle_split_compiled_executable(
    target: torch.fx.GraphModule,
    recompile_fn: Callable[[], "_SplitCompiledExecutable"] | None = None,
    pg_to_num_collectives: dict[ProcessGroupId, int] | None = None,
    graph_module_str: str | None = None,
) -> _SplitCompiledExecutable:
  """Reconstructs a _SplitCompiledExecutable from target.

  This function is used as the callable in the tuple returned by
  _SplitCompiledExecutable.__reduce__, enabling the object to be unpickled.

  Args:
    target: The split GraphModule.
    recompile_fn: Callable for recompiling the graph.
    pg_to_num_collectives: The mapping of process groups to collective counts.
    graph_module_str: The optional unpartitioned graph module string.

  Returns:
    A deserialized _SplitCompiledExecutable instance.
  """
  return _SplitCompiledExecutable(
      target=target,
      recompile_fn=recompile_fn,
      pg_to_num_collectives=pg_to_num_collectives,
      graph_module_str=graph_module_str,
  )


class SplitCompiler(compiler.Compiler):
  """Compiler that splits the graph on collectives and compiles submodules."""

  def __init__(self, base_compiler: compiler.Compiler):
    super().__init__(
        base_compiler.compilation_context, debug=base_compiler._debug
    )
    self.base_compiler = base_compiler

  def execute_pre_grad_passes(
      self,
      graph_module: torch.fx.GraphModule,
  ) -> None:
    self.base_compiler.execute_pre_grad_passes(graph_module)

  def _compile_graph(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
      materialize_collectives: bool = True,
      module_name: str | None = None,
      **kwargs,
  ) -> _SplitCompiledExecutable:
    """Splits the graph on collectives and devices and compiles the submodules."""
    graph_transform_observer.GraphTransformObserver(
        graph_module, "clone_mutated_returned_placeholders"
    ).apply_graph_pass(clone_mutated_returned_placeholders.apply)
    devices = _assign_devices(graph_module.graph)
    # Constants used on CPU must stay CPU tensors. TPU partitions using them
    # still embed them when the base compiler marks its own copy of the graph.
    graph_transform_observer.GraphTransformObserver(
        graph_module, "mark_embedded_constants"
    ).apply_graph_pass(
        functools.partial(
            mark_embedded_constants.apply,
            skip_targets={
                str(node.target)
                for node, device in devices.items()
                if node.op == "get_attr" and device == _CPU
            },
        )
    )
    graph_module.graph.lint()
    _register_output_namedtuple_globals(graph_module)
    graph_module.recompile()

    partition_id = 0
    partition_map = {}
    # Even stages run on CPU and odd stages on TPU. Each node runs in the
    # earliest stage after its inputs, so that CPU work that does not depend on
    # TPU results is batched instead of splitting TPU programs.
    stages = {}
    last_random_ops = {}
    # Maps device-agnostic nodes to the partition id at their position.
    deferred = {}
    for node in graph_module.graph.nodes:
      # split_module does not partition these.
      if node.op in ("placeholder", "get_attr", "output"):
        continue

      device = devices[node]
      inputs = list(node.all_input_nodes)
      if torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ()):
        # Random ops on each device keep their order to draw the same values.
        inputs.append(last_random_ops.get(device))
        last_random_ops[device] = node
      stage = max((stages[n] for n in inputs if n in stages), default=0)
      if device is not None and (stage % 2 == 1) != (device == _TPU):
        stage += 1
      stages[node] = stage

      in_spmd_safe_region = bool(
          isinstance(getattr(node, "meta", None), dict)
          and node.meta.get("custom", {}).get(
              spmd_util.SPMD_SAFE_METADATA_KEY, False
          )
      )

      maybe_collective = getattr(node.target, "overloadpacket", node.target)
      is_collective = (
          node.op == "call_function"
          and collective_ops.is_collective_op(maybe_collective)
      )

      # `wait_tensor`` is a special case. It is a collective-related op, but it
      # should not be used to split the graph.
      # While tracing the graph in `fx_to_mlir` tracing a non-wait_tensor
      # collective will populate a global registry with TpuWork objects with the
      # key being the tensor Storage ptr.
      #
      # If we were to separate the corresponding wait_tensor as its own
      # submodule we would leak the inserted TpuWork objects.
      #
      # This is because the role of wait_tensor in PyTorch is to "pop" TpuWork
      # from the global registry.
      # Tracing wait_tensor separately means that it would receive a different
      # Storage ptr than the one used by the original collective so we would
      # never pop the TpuWork object from the global registry.
      if maybe_collective == torch.ops._c10d_functional.wait_tensor:  # pylint: disable=protected-access
        producer = _get_unique_wait_tensor_producer(node)
        if producer not in partition_map:
          raise ValueError("wait_tensor producer not found in partition map.")
        partition_map[node] = partition_map[producer]
      elif _is_getitem_of_collective(node):
        if node.args[0] not in partition_map:
          raise ValueError(
              "getitem of a collective producer not found in partition map."
          )
        partition_map[node] = partition_map[node.args[0]]
      elif (
          is_collective and not in_spmd_safe_region and materialize_collectives
      ):
        # Make sure the collective doesn't end up in the previous partition.
        partition_id += 1
        partition_map[node] = partition_id
        # Make sure the next node doesn't end up in a partition with the
        # collective.
        partition_id += 1
      elif device is None:
        deferred[node] = partition_id
      else:
        partition_map[node] = partition_id

    # Device-agnostic nodes, e.g. constant expressions that TPU programs can
    # embed, run in the partition of their first user. Those without users in
    # partitions, e.g. returned SymInts, default to TPU. TPU programs cannot
    # return values other than tensors, such as SymInts, so those are recomputed
    # in every partition that uses them.
    keys = {node: (partition_map[node], stages[node]) for node in partition_map}
    for node in reversed(deferred):
      user_keys = sorted({keys[user] for user in node.users if user in keys})
      keys[node] = (
          user_keys[0] if user_keys else (deferred[node], max(stages[node], 1))
      )
      if _tensor_device_types(node):
        continue
      for key in user_keys[1:]:
        with graph_module.graph.inserting_after(node):
          duplicate = graph_module.graph.node_copy(node)
        node.replace_all_uses_with(
            duplicate,
            delete_user_cb=lambda user, key=key: keys.get(user) == key,
        )
        keys[duplicate] = key
    # Sorting the keys numbers partitions in an order that respects their
    # dependencies.
    partition_ids = {key: i for i, key in enumerate(sorted(set(keys.values())))}
    cpu_submods = {
        f"submod_{i}"
        for (_, stage), i in partition_ids.items()
        if stage % 2 == 0
    }
    logging.info(
        "Split graph into %d partitions, %d of which run on CPU",
        len(partition_ids),
        len(cpu_submods),
    )

    # WARNING: agents tend to fix regressions by falling back to the base_compiler
    # if the number of partitions is 1.
    # This is because the split_module pass will sometimes do things that are
    # incompatible with the downstream torch_tpu stack so short circuiting masks
    # these issues.
    # The proper fix isn't to avoid splitting onto a single partition but
    # rather investigating what the incompatibility with the torch_tpu stack
    # is, and addressing that issue by altering or adding a consolidating fx pass.
    # The reason we ALWAYS want to go down the split_module path is that if we
    # skip it we are potentially going to hit the exact same issue once we have
    # to do actual splits. This will result in us solving the exact same
    # problem, just at a later, more convoluted stage.

    split_gm = split_module(
        graph_module,
        None,  # type: ignore[arg-type]
        lambda node: partition_ids[keys[node]],
    )
    _call_partitions_in_order(split_gm)
    _register_output_namedtuple_globals(split_gm)
    graph_transform_observer.GraphTransformObserver(
        split_gm, "sink_get_attr_constants"
    ).apply_gm_pass(sink_get_attr_constants.apply)
    graph_transform_observer.GraphTransformObserver(
        split_gm, "force_collectives_output"
    ).apply_gm_pass(force_collectives_output.apply)
    graph_transform_observer.GraphTransformObserver(
        split_gm, "propagate_symints"
    ).apply_gm_pass(propagate_symints.apply)
    graph_transform_observer.GraphTransformObserver(
        split_gm, "reorder_symints"
    ).apply_gm_pass(reorder_symints.apply)

    logging.debug("Split graph\n%s", LazyString(split_gm.print_readable))

    fake_mode = detect_fake_mode(example_inputs)
    if fake_mode is None:
      fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()  # pylint: disable=protected-access

    compiler_fn = functools.partial(
        self.base_compiler.__call__, is_fwd=is_fwd, **kwargs
    )
    submod_compiler = _SubmodCompiler(
        split_gm,
        compiler_fn,
        fake_mode,
        base_module_name=module_name,
        cpu_submods=cpu_submods,
    )

    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    with torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():  # pylint: disable=protected-access
      submod_compiler.run(*example_inputs)
    split_gm.recompile()
    return _SplitCompiledExecutable(
        target=split_gm,
        graph_module_str=graph_module.print_readable(print_output=False),
    )

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
      module_name: str | None = None,
  ) -> _SplitCompiledExecutable:
    """Splits the graph on collectives and compiles the submodules."""

    handshake_stage = tpu_torch_compile.get_handshake_stage_env_var_once()
    materialize_collectives = env.get_bool_env_once(
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", default_value=True
    )

    should_handshake = False
    pg_to_num_collectives: dict[ProcessGroupId, int] = {}
    if handshake_stage != tpu_torch_compile.HandshakeStage.OFF:
      pg_to_num_collectives = process_group_utils.get_num_collectives_per_pg(
          graph_module
      )
      should_handshake = handshake.should_handshake(pg_to_num_collectives)
      if not materialize_collectives:
        should_handshake = False
        logging.warning(
            "Materialize collectives is set to False. Handshake will be "
            "skipped."
        )

    recompile_fn = None

    if (
        handshake_stage == tpu_torch_compile.HandshakeStage.COMPILE_STAGE
        and should_handshake
    ):
      fingerprint = tpu_torch_compile.fingerprint64(graph_module.code)
      fingerprints_match = _submit_handshake(
          str(fingerprint),
          pg_to_num_collectives,
          graph_module.print_readable(print_output=False),
      )
      materialize_collectives = not fingerprints_match
    elif (
        handshake_stage == tpu_torch_compile.HandshakeStage.DISPATCH_STAGE
        and should_handshake
    ):
      materialize_collectives = (
          False  # We set this so the first compilation doesn't split the graph
      )
      recompile_fn = functools.partial(
          self._compile_graph,
          graph_module,
          example_inputs,
          is_fwd,
          # Split the graph if a recompilation occurs.
          materialize_collectives=True,
          module_name=module_name,
      )

    executable = self._compile_graph(
        graph_module,
        example_inputs,
        is_fwd,
        materialize_collectives=materialize_collectives,
        module_name=module_name,
    )

    executable.recompile_fn = recompile_fn
    executable.pg_to_num_collectives = pg_to_num_collectives

    return executable
