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
from typing import Any

from absl import logging
import torch
from torch._dynamo.utils import detect_fake_mode
from torch._inductor.utils import InputType
from torch.fx.passes.split_module import split_module
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile.torch_tpu_compiled_executable import CompiledArtifact

# Required to register the SPMD safe region ops.
from torch_tpu._internal.distributed import spmd_util as _spmd_util  # pylint: disable=unused-import

# pylint: disable=protected-access
_COLLECTIVE_OPS = (
    torch.ops._c10d_functional.all_reduce,
    torch.ops._c10d_functional.all_reduce_,
    torch.ops._c10d_functional.all_reduce_coalesced,
    torch.ops._c10d_functional.all_reduce_coalesced_,
    torch.ops._c10d_functional.all_gather_into_tensor,
    torch.ops._c10d_functional.all_gather_into_tensor_out,
    torch.ops._c10d_functional.all_gather_into_tensor_coalesced,
    torch.ops._c10d_functional.reduce_scatter_tensor,
    torch.ops._c10d_functional.reduce_scatter_tensor_out,
    torch.ops._c10d_functional.reduce_scatter_tensor_coalesced,
    torch.ops._c10d_functional.all_to_all_single,
    torch.ops._c10d_functional.broadcast,
    torch.ops._c10d_functional.broadcast_,
)
# pylint: enable=protected-access


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
        new_args.append(
            torch._dynamo.utils.to_fake_tensor(arg, self.fake_mode)  # pylint: disable=protected-access
        )
      else:
        new_args.append(arg)

    if n.op == "call_module":
      real_mod = self.fetch_attr(str(n.target))

      # Deepcopy the submodule before compiling to prevent compiler-time
      # metadata mutation and structural corruption of nodes/inputs.
      comp_mod = copy.deepcopy(real_mod)
      compiled_submod_real = self.compile_submod(comp_mod, new_args, kwargs)

      # Propagate fake output shapes to downstream submodules
      with self.fake_mode:
        fake_out = real_mod(*new_args, **kwargs)

      # Stitch the compiled submodule back into the parent GraphModule
      self.module.delete_submodule(n.target)
      n.target = "compiled_" + n.target
      self.module.add_submodule(n.target, compiled_submod_real)
      return fake_out
    else:
      return getattr(self, n.op)(n.target, new_args, kwargs)


class _SplitCompiledExecutable(CompiledArtifact):
  """A CompiledArtifact supporting split submodules."""

  def __init__(self, split_gm: torch.fx.GraphModule):
    self._split_gm = split_gm

  def __call__(self, *args: Any) -> Any:
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
      args = args[0]
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
        (self._split_gm,),
    )


def _unpickle_split_compiled_executable(
    split_gm: torch.fx.GraphModule,
) -> "_SplitCompiledExecutable":
  """Reconstructs a _SplitCompiledExecutable from a split GraphModule.

  This function is used as the callable in the tuple returned by
  _SplitCompiledExecutable.__reduce__, enabling the object to be unpickled.

  Args:
    split_gm: The split GraphModule containing compiled and eager submodules.

  Returns:
    A deserialized _SplitCompiledExecutable instance.
  """
  return _SplitCompiledExecutable(split_gm)


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

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
  ) -> CompiledArtifact:
    """Splits the graph on collectives and compiles the submodules."""

    enter_target = torch.ops.torch_tpu.enter_spmd_safe_region.default
    exit_target = torch.ops.torch_tpu.exit_spmd_safe_region.default

    spmd_safe_depth = 0
    partition_id = 0
    partition_map = {}
    for node in graph_module.graph.nodes:
      if node.op in ("placeholder", "output"):
        continue

      if node.target == enter_target:
        spmd_safe_depth += 1
      elif node.target == exit_target:
        spmd_safe_depth -= 1

      in_spmd_safe_region = spmd_safe_depth > 0

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
      elif is_collective and not in_spmd_safe_region:
        # Make sure the collective doesn't end up in the previous partition.
        partition_id += 1
        partition_map[node] = partition_id
        # Make sure the next node doesn't end up in a partition with the
        # collective.
        partition_id += 1
      else:
        partition_map[node] = partition_id

    num_partitions = len(set(v for v in partition_map.values()))
    if num_partitions <= 1:
      logging.info(
          "Skipping split because there is only %d partition", num_partitions
      )
      return self.base_compiler(graph_module, example_inputs, is_fwd)

    logging.info(
        "Split graph into %d partitions",
        num_partitions,
    )

    split_gm = split_module(
        graph_module,
        None,  # type: ignore[arg-type]
        lambda node: partition_map[node],
    )

    fake_mode = detect_fake_mode(example_inputs)
    if fake_mode is None:
      fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()  # pylint: disable=protected-access

    compiler_fn = functools.partial(self.base_compiler.__call__, is_fwd=is_fwd)
    submod_compiler = _SubmodCompiler(split_gm, compiler_fn, fake_mode)

    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    with torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():  # pylint: disable=protected-access
      submod_compiler.run(*example_inputs)
    split_gm.recompile()
    return _SplitCompiledExecutable(split_gm)
