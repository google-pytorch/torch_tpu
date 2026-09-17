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

import torch
from torch._dynamo.backends import registry
from torch._functorch._aot_autograd import utils as aot_utils
from torch_tpu._internal.compile._backend import async_compile
from torch_tpu._internal.compile._backend import AsyncCompilationSubmitted
from torch_tpu._internal.compile._backend import clear_cache
from torch_tpu._internal.compile._backend import resolve_compilations
from torch_tpu._internal.compile._backend import TpuBackend
from torch_tpu._internal.compile.debug import TpuCompileDebug
from torch_tpu._internal.compile.trace_compiler import CompiledCallable
from torch_tpu._internal.compile.trace_compiler import trace_and_compile

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
# Disables PyTorch Dynamo's default graph break on RNN, LSTM, and GRU modules,
# allowing Dynamo to trace recurrent layers into aten::lstm and aten::gru nodes
# for lowering to TorchTPU's StableHLO kernels.
if hasattr(torch._dynamo.config, "allow_rnn"):
  torch._dynamo.config.allow_rnn = True
# pylint: enable=protected-access


def _register_scan_operator() -> None:
  """Registers the custom scan operator implementation for TPU."""
  from torch_tpu._internal.compile import scan as _  # pylint: disable=g-import-not-at-top  # noqa: F401


_register_scan_operator()


def _patch_flex_attention() -> None:
  """Patches PyTorch flex_attention to support TPU tensors."""
  # pylint: disable=g-import-not-at-top,protected-access
  from torch._higher_order_ops.flex_attention import flex_attention_backward
  from torch.nn.attention import flex_attention

  original_validate = flex_attention._validate_device

  def _validate_device(query, key, value):
    if query.device.type != "tpu":
      original_validate(query, key, value)

  flex_attention._validate_device = _validate_device

  dispatch_key = torch._C.DispatchKey.CompositeExplicitAutograd
  original_bwd = flex_attention_backward.py_kernels.pop(dispatch_key)

  @flex_attention_backward.py_impl(dispatch_key)
  def _sdpa_dense_backward(query, key, value, *args, **kwargs):
    with query.device:
      return original_bwd(query, key, value, *args, **kwargs)

  # pylint: enable=g-import-not-at-top,protected-access


_patch_flex_attention()

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "AsyncCompilationSubmitted",
    "CompiledCallable",
    "TpuBackend",
    "TpuCompileDebug",
    "async_compile",
    "clear_cache",
    "resolve_compilations",
    "trace_and_compile",
    # go/keep-sorted end
]
