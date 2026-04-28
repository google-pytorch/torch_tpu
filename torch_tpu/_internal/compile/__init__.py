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
from absl import app
from absl import flags
from absl import logging
import torch
from torch._dynamo import decorators
from torch._dynamo.backends import registry
from torch_tpu._internal.compile._backend import TpuBackend

# Register "tpu" backend
registry.register_backend(compiler_fn=TpuBackend(), name="tpu")


def _disallow_collective_ops_in_graph():
  # Compiling a graph with collective ops can cause deadlocks on TPU if there are
  # slight graph differences between ranks (e.g. from "if rank == 0: ..."). We
  # avoid this by triggering a graph break for collective ops.
  # This behavior can be disabled by setting
  # --torch_tpu_internal_materialize_collective_tensors=False.
  # flag is defined in /third_party/py/torch_tpu/distributed/process_group_tpu.cc
  # with default=True.
  # Note that as the flag is defined in distributed/process_group_tpu.cc, it is
  # possible that it is not linked by the Blaze build. So we have to check
  # if the flag exists, before using it.
  # TODO: Add necessary distributed ops with tests.
  if (
      "torch_tpu_internal_materialize_collective_tensors" in flags.FLAGS
      and not flags.FLAGS.torch_tpu_internal_materialize_collective_tensors
  ):
    return

  logging.info("[TpuBackend] Force graph break for collective ops enabled.")
  decorators.disallow_in_graph(torch.ops._c10d_functional.all_reduce)
  decorators.disallow_in_graph(torch.ops._c10d_functional.wait_tensor)


# Note: absl flags are not parsed until init_google is called, so we need to use
# call_after_init to register the callback.
app.call_after_init(_disallow_collective_ops_in_graph)

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
