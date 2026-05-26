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
from absl import logging
import torch
from torch._dynamo import decorators
from torch._dynamo.backends import registry
from torch._functorch._aot_autograd import utils as aot_utils
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile._backend import TpuBackend

# Register "tpu" backend
registry.register_backend(compiler_fn=TpuBackend(), name="tpu")


def _initialize_graphsafe_rng():
  """Initializes graphsafe RNG operations for TPU.

  Registers 'privateuseone' (TPU) for graphsafe RNG and enables
  necessary config flags.
  """
  aot_utils.register_graphsafe_rng_device_type("privateuseone")
  # Add "tpu" to the supported device types for graphsafe RNG, as torch_tpu
  # renames "privateuseone" to "tpu".
  aot_utils._GRAPHSAFE_RNG_DEVICE_TYPES.add("tpu")  # pylint: disable=protected-access

  # pylint: disable=protected-access
  torch._functorch.config.graphsafe_rng_functionalization = True
  torch._functorch.config.functionalize_rng_ops = False
  # pylint: enable=protected-access


# TODO: b/496168350 - Remove if-condition once generic graphsafe RNG is
# available in OSS PyTorch release.
if hasattr(aot_utils, "register_graphsafe_rng_device_type"):
  _initialize_graphsafe_rng()

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
    torch.ops._c10d_functional.wait_tensor,
)
# pylint: enable=protected-access


def _disallow_collective_ops_in_graph() -> None:
  """Disallows collective ops in the Dynamo graph.

  Compiling a graph with collective ops can cause deadlocks on TPU if there are
  slight graph differences between ranks (e.g. from "if rank == 0: ..."). We
  avoid this by triggering a graph break for collective ops. This behavior can
  be disabled by setting the environment variable
  `TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS` to `"false"` or `"0"`.
  """

  if not tpu_torch_compile.get_materialize_collective_tensors_env_value():
    return

  logging.info("[TpuBackend] Force graph break for collective ops enabled.")
  for op in _COLLECTIVE_OPS:
    decorators.disallow_in_graph(op)


_disallow_collective_ops_in_graph()


# New Dynamo flag that enables tracing through the `backward` function call.
# This enables the possibility of generating of a single fx graph containing
# both forward and backward subgraphs, rather than a discrete fx graph for each.
# See b/480979694 for details.
# pylint: disable=protected-access
if hasattr(torch._dynamo.config, "trace_autograd_ops"):
  torch._dynamo.config.trace_autograd_ops = True
# pylint: enable=protected-access

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "TpuBackend",
    # go/keep-sorted end
]
