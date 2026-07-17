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

"""Public API for torch_tpu._internal.compile."""

from absl import logging
import torch
from torch._dynamo import decorators
from torch._dynamo.backends import registry
from torch._functorch._aot_autograd import utils as aot_utils
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile._backend import TpuBackend

# Register "tpu" backend
registry.register_backend(compiler_fn=TpuBackend(), name="tpu")  # pyrefly: ignore[bad-argument-type]


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


# New Dynamo flag that enables tracing through the `backward` function call.
# This enables the possibility of generating of a single fx graph containing
# both forward and backward subgraphs, rather than a discrete fx graph for each.
# See b/480979694 for details.
# pylint: disable=protected-access
if hasattr(torch._dynamo.config, "trace_autograd_ops"):
  torch._dynamo.config.trace_autograd_ops = True
# pylint: enable=protected-access


def _register_scan_operator() -> None:
  """Registers the custom scan operator implementation for TPU."""
  from torch_tpu._internal.compile import scan as _  # pylint: disable=g-import-not-at-top


_register_scan_operator()

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "TpuBackend",
    # go/keep-sorted end
]
