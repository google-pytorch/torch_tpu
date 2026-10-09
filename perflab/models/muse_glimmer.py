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

"""Muse Glimmer single-device benchmark definitions.

The model and its training setup are those of the team's Muse Glimmer TPU
benchmark (`torch_tpu/google/prototypes/glimmer/glimmer_tpu_test.py`) on one
device; see `model_utils/muse_glimmer`. The 30B flavor is truncated to its last
4 layers (one full sliding-window / global attention cycle) and fed 4 packed
documents of 2048 tokens, as in the prototype's 1-device run.
"""

from typing import Any

import torch
from torch import nn
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab.model_utils import muse_glimmer


def _build(
    ctx: context_lib.Context,
) -> tuple[muse_glimmer.ModelBundle, torch.Tensor, dict[str, Any]]:
  """Builds the model and its packed batch for `ctx`."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers, num_docs, doc_len = 2, 2, 128
  else:
    num_layers, num_docs, doc_len = 4, 4, 2048
  device_str = ctx.device_kind.value
  bundle = muse_glimmer.build_model(
      "30B",
      num_layers,
      doc_len,
      device_str,
      torch_device_ops.get_torch_dtype(ctx.dtype),
  )
  tokens, kwargs = muse_glimmer.make_batch(
      bundle, num_docs, doc_len, rank=0, world_size=1, device=device_str
  )
  return bundle, tokens, kwargs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def muse_glimmer_30b_4_layers_forward(
    ctx: context_lib.Context,
) -> tuple[nn.Module, tuple[torch.Tensor, ...], dict[str, Any]]:
  """Benchmark factory for Muse Glimmer (30B arch, 4 layers) forward.

  Like the prototype's `--forward_only`, the forward pass stops before the
  output head and returns the final hidden states.

  Args:
    ctx: Benchmark context.
  """
  bundle, tokens, kwargs = _build(ctx)
  del kwargs["labels"]
  model = bundle.model
  model._skip_lm_head = True  # pylint: disable=protected-access
  model.eval()
  return model, (tokens,), kwargs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    stepper_kwargs={
        "compute_loss": muse_glimmer.cross_entropy_loss,
        "use_single_trace": False,
    },
    compile_config=compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=False
    ),
)
def muse_glimmer_30b_4_layers_training_standard(
    ctx: context_lib.Context,
) -> tuple[
    nn.Module, tuple[torch.Tensor, ...], dict[str, Any], torch.optim.Optimizer
]:
  """Benchmark factory for Muse Glimmer (30B arch, 4 layers) standard training.

  Args:
    ctx: Benchmark context.
  """
  bundle, tokens, kwargs = _build(ctx)
  return (
      bundle.model,
      (tokens,),
      kwargs,
      muse_glimmer.make_optimizer(bundle.model),
  )
