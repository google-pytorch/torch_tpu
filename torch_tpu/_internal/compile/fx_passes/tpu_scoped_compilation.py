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

"""TPU scoped compilation pass."""

from collections.abc import Callable, Sequence
import functools
import types
from typing import Any, TypeVar

from absl import logging
import torch
from torch._dynamo.backends import common as backend_common
from torch._subclasses import fake_tensor
from torch.fx.passes import fake_tensor_prop
from torch.fx.passes import split_module
from torch_tpu._internal.compile.fx_passes import pass_utils

_CompileCallback = TypeVar(
    "_CompileCallback",
    bound=Callable[..., Any],
)

_TPU_SUBMODULE_NAME = "submod_tpu"
_TPU_PARTITION_NAME = "tpu"
_CPU_PRE_PARTITION_NAME = "cpu_pre"
_CPU_POST_PARTITION_NAME = "cpu_post"


class CompiledWrapper(torch.nn.Module):
  """Wraps a compiled forward function so it can be assigned as a submodule."""

  def __init__(self, compiled_forward_fn):
    super().__init__()
    self.compiled_forward_fn = compiled_forward_fn

  def forward(self, *args, **kwargs):
    return self.compiled_forward_fn(*args, **kwargs)


def _compute_partition_map(
    gm: torch.fx.GraphModule,
) -> dict[torch.fx.Node, str]:
  """Computes a mapping from each node to its respective partition.

  Nodes are partitioned into 'cpu_pre', 'tpu', or 'cpu_post'. All TPU nodes
  are assigned to 'tpu'. CPU nodes that have a dependency on a TPU node
  are assigned to 'cpu_post', and CPU nodes without such dependencies are
  assigned to 'cpu_pre'.

  Args:
    gm: The input graph module to partition.

  Returns:
    A dictionary mapping each node in the graph to its partition name.
  """
  partition_map = {}
  tpu_dependent = set()

  for node in gm.graph.nodes:
    if node.op == "output":
      is_tpu_dep = False
      for arg in node.all_input_nodes:
        if arg in tpu_dependent:
          is_tpu_dep = True
          break
      partition_map[node] = (
          _CPU_POST_PARTITION_NAME if is_tpu_dep else _CPU_PRE_PARTITION_NAME
      )
      continue

    device = pass_utils.get_node_device(node)
    if device == pass_utils.DeviceId.TRANSFER:
      device = pass_utils.DeviceId.CPU
    if device == pass_utils.DeviceId.UNKNOWN:
      raise ValueError("Unknown device for node {}".format(node.format_node()))

    is_tpu_dep = False
    if device == pass_utils.DeviceId.TPU:
      is_tpu_dep = True
    else:
      for arg in node.all_input_nodes:
        if arg in tpu_dependent:
          is_tpu_dep = True
          break

    if is_tpu_dep:
      tpu_dependent.add(node)

    if device == pass_utils.DeviceId.TPU:
      partition_map[node] = _TPU_PARTITION_NAME
    else:
      partition_map[node] = (
          _CPU_POST_PARTITION_NAME if is_tpu_dep else _CPU_PRE_PARTITION_NAME
      )

  return partition_map


def _get_tpu_partition_call(
    split_gm: torch.fx.GraphModule,
) -> torch.fx.Node | None:
  """Finds and returns the call_module node for the TPU partition."""
  tpu_part = None
  for n in split_gm.graph.nodes:
    if n.op != "call_module":
      continue
    target_str = str(n.target)
    if target_str == _TPU_SUBMODULE_NAME:
      tpu_part = n
    elif target_str.startswith(_TPU_SUBMODULE_NAME):
      raise ValueError(
          f"Multiple TPU partitions found (e.g. {target_str})! Only one is"
          " supported."
      )
  return tpu_part


def _get_tpu_submod(
    split_gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule | None:
  """Finds and returns the TPU submodule from the partitioned GraphModule."""
  tpu_submod = None
  for name, sm in split_gm.named_children():
    assert isinstance(sm, torch.fx.GraphModule)
    if name == _TPU_SUBMODULE_NAME:
      tpu_submod = sm
    elif name.startswith(_TPU_SUBMODULE_NAME):
      raise ValueError(
          f"Multiple TPU submodules found (e.g. {name})! Only one is supported."
      )
  return tpu_submod


def _compute_tpu_example_inputs(
    example_inputs: Sequence[torch.Tensor],
    tpu_part: torch.fx.Node | None,
    graph: torch.fx.Graph,
) -> list[torch.Tensor]:
  """Computes the expected example inputs for the TPU partition."""
  new_example_inputs = []
  if tpu_part is not None:
    graph_ins = {
        node: example_inputs[i]
        for i, node in enumerate(
            [n for n in graph.nodes if n.op == "placeholder"]
        )
    }
    for arg in tpu_part.args:
      if arg in graph_ins:
        new_example_inputs.append(graph_ins[arg])
      else:
        new_example_inputs.append(arg.meta["val"])
  return new_example_inputs


def _propagate_fake_tensors(
    split_gm: torch.fx.GraphModule,
    example_inputs: Sequence[torch.Tensor],
) -> None:
  """Propagates fake tensors through the partitioned GraphModule."""
  fake_mode = fake_tensor.FakeTensorMode()
  with fake_mode:
    try:
      fake_tensor_prop.FakeTensorProp(split_gm).propagate(*example_inputs)
    except Exception as e:
      raise RuntimeError(
          "Fake tensor propagation failed through CPU partition, please raise "
          "issue with torch_tpu team and request eager cpu execution support"
      ) from e

  tpu_part = _get_tpu_partition_call(split_gm)
  if tpu_part is None:
    return
  if any(
      pass_utils.has_dynamic_shape(arg.meta["val"])
      for arg in tpu_part.args
      if "val" in arg.meta
  ):
    logging.warning(
        "Warning: tpu backend traced dynamic shapes through leading CPU ops. If"
        " eager execution was intended before compilation, please raise issue"
        " with torch_tpu team and request support."
    )


def _tpu_scoped_compilation(
    compile_callback: _CompileCallback,
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[torch.Tensor],
) -> Callable[..., Any]:
  """Partitions the graph into CPU and TPU submodules and compiles the TPU part."""

  partition_map = _compute_partition_map(gm)

  split_gm = split_module.split_module(gm, None, lambda node: partition_map[node], tuple_return=True)  # pytype: disable=wrong-arg-types

  _propagate_fake_tensors(split_gm, example_inputs)

  tpu_part = _get_tpu_partition_call(split_gm)
  new_example_inputs = _compute_tpu_example_inputs(
      example_inputs, tpu_part, split_gm.graph
  )

  tpu_submod = _get_tpu_submod(split_gm)
  if tpu_submod is not None:
    setattr(
        split_gm,
        _TPU_SUBMODULE_NAME,
        CompiledWrapper(compile_callback(tpu_submod, new_example_inputs)),
    )

  split_gm.graph.eliminate_dead_code()
  split_gm.graph.lint()
  split_gm.recompile()

  return split_gm


class TpuScopedCompilation:
  """Applies the given backend to only ops on the TPU.

  This enables tpu/cpu heterogeneity support in the torch tpu backend. The
  approach here is rather naive currently, only graphs with a single tpu offload
  are supported for the common case.

  Prepended CPU partitions are just fake tensor traced. This means the tpu
  compilation will not wait for any cpu ops, and proceed right away with
  only the shapes of the cpu partition outputs.

  This approach may be extended to support interpreter like behavior, eagerly
  executing cpu ops and pushing data backed tensors to compilation if it
  is benificial (decided dynamically) to do so. It also may be desirable to
  offload to another optimized cpu compilation backend (like inductor) at some
  point.
  """

  def __init__(self, compile_callback: _CompileCallback = lambda x, y: x):
    self.compile_callback = compile_callback

  def __get__(self, instance, owner):
    if instance is None:
      return self
    bound_callback = types.MethodType(self.compile_callback, instance)
    return TpuScopedCompilation(compile_callback=bound_callback)

  def __call__(
      self, gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
  ) -> Callable[..., Any]:
    fw_compiler = torch.functorch.compile.make_boxed_compiler(
        functools.partial(_tpu_scoped_compilation, self.compile_callback)
    )
    autograd_compiler = backend_common.aot_autograd(fw_compiler=fw_compiler)  # pytype: disable=wrong-arg-types
    split_gm = autograd_compiler(gm, example_inputs)  # pytype: disable=wrong-arg-types

    return split_gm
