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
operations and compiling it's submodules.
"""

from collections.abc import Callable, Sequence
import copy
import functools
import operator
import threading
from typing import Any, List

from absl import logging
import torch
from torch._dynamo.utils import detect_fake_mode
from torch._inductor.utils import InputType
from torch._subclasses.fake_tensor import unset_fake_temporarily
import torch.distributed as dist
from torch.fx.passes import graph_transform_observer
from torch.fx.passes.split_module import split_module
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

_COLLECTIVE_OPS = collective_ops.COLLECTIVE_OPS


# We define a per process global collective count to keep track of the
# collectives that have been submitted to the split compiler. This is used to
# determine if a handshake is needed and to submit the handshake message.
_RANK_COLLECTIVE_COUNTS: RankCollectiveCounts = RankCollectiveCounts()
_RANK_COLLECTIVE_COUNTS_LOCK = threading.Lock()


def _submit_handshake(
    executable_fingerprint: str,
    pg_to_num_collectives: dict[ProcessGroupId, int],
    graph_module: torch.fx.GraphModule | None = None,
) -> bool:
  """Submits a handshake message across participating ranks.

  Args:
    executable_fingerprint: The fingerprint of the executable.
    pg_to_num_collectives: The mapping of ProcessGroupId to collective counts.
    graph_module: The graph module of the executable, for logging purposes.

  Returns True if all ranks match.
  """
  assert (
      dist.is_initialized()
  ), "Distributed backend must be initialized when submitting a handshake."

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
    if graph_module:
      logging.debug(
          "Handshake failed on rank %d for graph module: %s", rank, graph_module
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
  if (
      getattr(producer.target, "overloadpacket", producer.target)
      not in _COLLECTIVE_OPS
  ):
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
  if isinstance(curr, torch.fx.Node) and (
      getattr(curr.target, "overloadpacket", curr.target) in _COLLECTIVE_OPS
  ):
    return True
  return False


class _WrapperModule(torch.nn.Module):
  """Wrapper module for compiled submodules."""

  def __init__(self, submod: Callable[..., Any], unwrap_singleton_tuple: bool):
    super().__init__()
    self.submod = submod
    self.unwrap_singleton_tuple = unwrap_singleton_tuple

  def forward(self, *args: Any) -> Any:
    x = self.submod(*args)
    # AOT Autograd compiled submodules always wrap their output in a tuple.
    # If the original subgraph did not return a tuple, we unwrap the singleton
    # tuple here to match the parent graph's schema.
    if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
      return x[0]
    return x


class _SubmodCompiler(torch.fx.interpreter.Interpreter):
  """Interpreter to propagate shape information and compile submodules.

  This class traverses the topologically-sorted split GraphModule sequentially,
  compiles all submodules and propagates input/output shapes from one submodule
  to the next.
  """

  def __init__(
      self,
      module: torch.fx.GraphModule,
      compiler_fn: Callable[..., CompiledArtifact],
      fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,
  ):
    super().__init__(module)
    self.compiler_fn = compiler_fn
    self.fake_mode = fake_mode

  # This logic is copied from torch/_functorch/_aot_autograd/frontend_utils.py
  def _convert_to_fake_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
    symbolic_context = None
    source = None
    if tracing_context := torch._guards.TracingContext.try_get():  # pytype: disable=protected-access
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
    compiled_submod_real = self.compiler_fn(input_mod, args)
    wrapper = _WrapperModule(
        compiled_submod_real,
        unwrap_singleton_tuple,
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

    if n.op == "call_module":
      real_mod = self.fetch_attr(str(n.target))

      # Deepcopy the submodule before compiling to prevent compiler-time
      # metadata mutation and structural corruption of nodes/inputs.
      with unset_fake_temporarily():
        comp_mod = copy.deepcopy(real_mod)
      compiled_submod_real = self.compile_submod(comp_mod, new_args, kwargs)

      # Propagate fake output shapes to downstream submodules
      with self.fake_mode:
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
      split_gm: torch.fx.GraphModule,
      recompile_fn: Callable[[], "_SplitCompiledExecutable"] | None = None,
      pg_to_num_collectives: dict[ProcessGroupId, int] | None = None,
  ):
    """Initializes a _SplitCompiledExecutable.

    Args:
      split_gm: The split GraphModule containing compiled and eager submodules.
      recompile_fn: Optional callable for recompiling the graph.
      pg_to_num_collectives: Mapping of ProcessGroupId to collective counts for
        this executable.
    """
    self._split_gm = split_gm
    self.recompile_fn = recompile_fn
    self.pg_to_num_collectives = (
        pg_to_num_collectives if pg_to_num_collectives else {}
    )

  def _maybe_handshake_and_recompile(self):
    if self.recompile_fn is None:
      return

    compiled_execs = [
        module.submod
        for module in self._split_gm.modules()
        if isinstance(module, _WrapperModule)
        and isinstance(
            module.submod, (TorchTpuCompiledExecutable, AsyncCompiledArtifact)
        )
    ]
    assert len(compiled_execs) == 1, (
        "Expected split_gm to contain exactly one TorchTpuCompiledExecutable"
        f" or AsyncCompiledArtifact, found {len(compiled_execs)}."
    )
    executable_fingerprint = compiled_execs[0].fingerprint()
    fingerprints_match = _submit_handshake(
        executable_fingerprint,
        self.pg_to_num_collectives,
    )
    if fingerprints_match:
      return

    new_exec = self.recompile_fn()
    self.__dict__.update(new_exec.__dict__)

  def __call__(self, *args: Any) -> Any:
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
      args = args[0]  # pyrefly: ignore[bad-assignment]

    self._maybe_handshake_and_recompile()

    return self._split_gm(*args)

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Enables pickling of the _SplitCompiledExecutable.

    This method is part of Python's pickle protocol. It returns a tuple
    containing:
      1. A callable (_unpickle_split_compiled_executable) that can be called to
         recreate the object.
      2. A tuple of arguments containing the parent GraphModule split_gm.

    Returns:
      A tuple (callable, args_tuple) used by the pickle module to
      serialize the object.
    """
    return (
        _unpickle_split_compiled_executable,
        (
            self._split_gm,
            self.recompile_fn,
            self.pg_to_num_collectives,
        ),
    )

  @property
  def graph_module_debug_strs(self) -> List[str]:
    """List of string representations of the FX graph module's code for each submodule."""
    graph_module_debug_strs = []
    for module in self._split_gm.modules():
      if isinstance(module, _WrapperModule):
        graph_module_debug_strs.append(
            module.submod.graph_module_debug_str  # pyrefly: ignore[missing-attribute]
        )
    return graph_module_debug_strs

  @property
  def mlir_texts(self) -> List[str]:
    """List of MLIR text representations of the compiled submodule's code."""

    mlir_texts = []
    for module in self._split_gm.modules():
      if isinstance(module, _WrapperModule):
        mlir_texts.append(
            module.submod.mlir_text  # pyrefly: ignore[missing-attribute]
        )
    return mlir_texts

  def post_compile(
      self,
      example_inputs: Sequence[Any],
      constants: Any,
      graph_kwargs: Any,
  ) -> None:
    pass

  def prepare_for_serialization(self) -> None:
    pass

  def updates_default_generator_state(self) -> bool:
    return any(
        module.submod.updates_default_generator_state()  # pyrefly: ignore[missing-attribute]
        for module in self._split_gm.modules()
        if isinstance(module, _WrapperModule)
    )

  def resolve(self) -> None:
    """Waits for any pending background compilation in submodules to complete."""
    for module in self._split_gm.modules():
      if isinstance(module, _WrapperModule):
        resolve_fn = getattr(module.submod, "resolve", None) or getattr(
            module.submod, "_resolve", None
        )
        if callable(resolve_fn):
          resolve_fn()

  def _resolve(self) -> None:
    self.resolve()

  @property
  def is_resolved(self) -> bool:
    """Returns True if all submodule compilations are resolved."""
    return all(
        getattr(module.submod, "is_resolved", True)
        for module in self._split_gm.modules()
        if isinstance(module, _WrapperModule)
    )


def _unpickle_split_compiled_executable(
    split_gm: torch.fx.GraphModule,
    recompile_fn: Callable[[], "_SplitCompiledExecutable"] | None = None,
    pg_to_num_collectives: dict[ProcessGroupId, int] | None = None,
) -> "_SplitCompiledExecutable":
  """Reconstructs a _SplitCompiledExecutable from a split GraphModule.

  This function is used as the callable in the tuple returned by
  _SplitCompiledExecutable.__reduce__, enabling the object to be unpickled.

  Args:
    split_gm: The split GraphModule containing compiled and eager submodules.
    recompile_fn: Callable for recompiling the graph.
    pg_to_num_collectives: The mapping of process groups to collective counts.

  Returns:
    A deserialized _SplitCompiledExecutable instance.
  """
  return _SplitCompiledExecutable(
      split_gm,
      recompile_fn=recompile_fn,
      pg_to_num_collectives=pg_to_num_collectives,
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
      **kwargs,
  ) -> _SplitCompiledExecutable:
    """Splits the graph on collectives and compiles the submodules."""
    graph_transform_observer.GraphTransformObserver(
        graph_module, "mark_embedded_constants"
    ).apply_graph_pass(mark_embedded_constants.apply)

    graph_transform_observer.GraphTransformObserver(
        graph_module, "clone_mutated_returned_placeholders"
    ).apply_graph_pass(clone_mutated_returned_placeholders.apply)
    graph_module.graph.lint()
    graph_module.recompile()

    partition_id = 0
    partition_map = {}
    for node in graph_module.graph.nodes:
      if node.op in ("placeholder", "output"):
        continue

      in_spmd_safe_region = bool(
          isinstance(getattr(node, "meta", None), dict)
          and node.meta.get("custom", {}).get(
              spmd_util.SPMD_SAFE_METADATA_KEY, False
          )
      )

      maybe_collective = getattr(node.target, "overloadpacket", node.target)
      is_collective = (
          node.op == "call_function" and maybe_collective in _COLLECTIVE_OPS
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
      else:
        partition_map[node] = partition_id

    num_partitions = len(set(partition_map.values()))

    logging.info(
        "Split graph into %d partitions",
        num_partitions,
    )

    split_gm = split_module(
        graph_module,
        None,  # type: ignore[arg-type]
        lambda node: partition_map[node],
    )
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

    logging.debug("Split graph\n%s", split_gm.print_readable())

    fake_mode = detect_fake_mode(example_inputs)
    if fake_mode is None:
      fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()  # pylint: disable=protected-access

    compiler_fn = functools.partial(
        self.base_compiler.__call__, is_fwd=is_fwd, **kwargs
    )
    submod_compiler = _SubmodCompiler(split_gm, compiler_fn, fake_mode)

    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    with torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():  # pylint: disable=protected-access
      submod_compiler.run(*example_inputs)
    split_gm.recompile()
    return _SplitCompiledExecutable(split_gm)

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
  ) -> _SplitCompiledExecutable:
    """Splits the graph on collectives and compiles the submodules."""

    handshake_stage = tpu_torch_compile.get_handshake_stage_env_var_once()
    materialize_collectives = (
        tpu_torch_compile.get_materialize_collective_tensors_env_value()
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
          str(fingerprint), pg_to_num_collectives, graph_module
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
      )

    executable = self._compile_graph(
        graph_module,
        example_inputs,
        is_fwd,
        materialize_collectives=materialize_collectives,
    )

    executable.recompile_fn = recompile_fn
    executable.pg_to_num_collectives = pg_to_num_collectives

    return executable
