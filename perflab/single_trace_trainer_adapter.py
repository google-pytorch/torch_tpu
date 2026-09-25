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

"""Adapter for SingleTraceTrainer binding StaticCompiler for benchmarks."""

from typing import Any, Callable

import torch
from torch.utils import _pytree
from torch_tpu._internal.compile import compiler
from torch_tpu.extras.optimizers import adamw
from torch_tpu.extras.optimizers import muon
from torch_tpu.extras.optimizers import optim
from torch_tpu.extras.optimizers import sgd
from torch_tpu.extras.optimizers import single_trace_trainer

# Re-export optimizer classes
Optimizer = optim.Optimizer
AdamW = adamw.AdamW
SGD = sgd.SGD
ReferenceAdamw = adamw.ReferenceAdamw
TorchAdamw = adamw.TorchAdamw
ReferenceSgd = sgd.ReferenceSgd
TorchSgd = sgd.TorchSgd
Muon = muon.Muon
ReferenceMuon = muon.ReferenceMuon
TorchMuon = muon.TorchMuon


class SingleTraceTrainerAdapter(single_trace_trainer.SingleTraceTrainer):
  """Adapter for SingleTraceTrainer that re-internalizes StaticCompiler."""

  def __init__(
      self,
      model: torch.nn.Module,
      optimizer: Any = None,
      backend_compiler: compiler.StaticCompiler | None = None,
  ):
    super().__init__(model=model, optimizer=optimizer)
    self.backend_compiler = backend_compiler or compiler.StaticCompiler()

  def make_compiled_train_step(
      self,
      example_inputs: Any,
      example_targets: Any = None,
      compile_fn: Callable[..., Any] | None = None,
  ) -> Callable[..., torch.Tensor]:
    """Compiles train step using StaticCompiler by default."""
    if compile_fn is None:

      def default_compile_fn(graph_module, flat_inputs):
        donated_inputs = None
        # Only donate parameter buffers when an optimizer updates parameters in-place.
        # Without an optimizer (or when input donation is explicitly disabled),
        # donating parameters would invalidate them across repeated training iterations.
        if self.optimizer is not None and getattr(
            self.optimizer, "donate_inputs", True
        ):
          flat_param_group, _ = _pytree.tree_flatten(self.param_group)
          donated_inputs = [
              i
              for i, x in enumerate(flat_param_group)
              if isinstance(x, torch.Tensor)
          ]
        return self.backend_compiler(
            graph_module,
            flat_inputs,
            donated_inputs=donated_inputs,
        )

      compile_fn = default_compile_fn

    return super().make_compiled_train_step(
        example_inputs=example_inputs,
        example_targets=example_targets,
        compile_fn=compile_fn,
    )
