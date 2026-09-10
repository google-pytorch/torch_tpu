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
from examples.tracing import optimizers
from examples.tracing import single_trace_trainer

# Re-export optimizer classes
Optimizer = optimizers.Optimizer
AdamW = optimizers.AdamW
SGD = optimizers.SGD
ReferenceAdamw = optimizers.ReferenceAdamw
TorchAdamw = optimizers.TorchAdamw
ReferenceSgd = optimizers.ReferenceSgd
TorchSgd = optimizers.TorchSgd
Muon = optimizers.Muon
ReferenceMuon = optimizers.ReferenceMuon
TorchMuon = optimizers.TorchMuon


class SingleTraceTrainerAdapter(single_trace_trainer.SingleTraceTrainer):
  """Adapter for SingleTraceTrainer that re-internalizes StaticCompiler."""

  def __init__(
      self,
      model: torch.nn.Module,
      optimizer: Any,
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
