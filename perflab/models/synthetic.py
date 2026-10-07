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

"""Synthetic benchmarks that isolate per-output host dispatch overhead.

Each benchmark runs a program whose device work is negligible but which has
many small, independent outputs. Post-warmup step time is then dominated by
host-side work that scales with the number of program outputs: buffer shape
checks, ready-future chaining, completion callbacks and output
materialization. Sweeping the output count exposes that per-output cost
directly, which real models hide behind long device steps.

* `synthetic_<N>_outputs_forward`: a forward program returning N tensors.
* `synthetic_<N>_params_training`: a training step over N small parameters.
  With AdamW the compiled step has roughly 3N outputs (parameters plus both
  moment buffers).
"""

import torch
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops

# Small enough that device time is negligible, large enough to be a realistic
# (non-scalar) buffer.
_TENSOR_SHAPE = (8, 128)
_NUM_OUTPUTS_SWEEP = (16, 256, 1024, 4096)
# Presubmit only smoke-tests that the benchmark compiles and runs.
_PRESUBMIT_MAX_NUM_OUTPUTS = 16


class _ManyOutputs(torch.nn.Module):
  """Returns one small, independent tensor per parameter."""

  def __init__(self, num_outputs: int, dtype: torch.dtype, device: str):
    super().__init__()
    self.weights = torch.nn.ParameterList(
        torch.nn.Parameter(
            torch.randn(_TENSOR_SHAPE, dtype=dtype, device=device)
        )
        for _ in range(num_outputs)
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(x * w for w in self.weights)


class _ManyParamsScalarLoss(_ManyOutputs):
  """Reduces every parameter into one scalar loss so all of them get grads."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:  # pytype: disable=signature-mismatch
    return torch.stack([(x * w).sum() for w in self.weights]).sum()


def _check_scope(ctx: context_lib.Context, num_outputs: int) -> None:
  if (
      ctx.run_scope == context_lib.RunScope.PRESUBMIT
      and num_outputs > _PRESUBMIT_MAX_NUM_OUTPUTS
  ):
    raise target_lib.UnsupportedBenchmark(
        f"{num_outputs} outputs only run in the full scope."
    )


def _make_input(ctx: context_lib.Context) -> torch.Tensor:
  return torch.randn(
      _TENSOR_SHAPE,
      dtype=torch_device_ops.get_torch_dtype(ctx.dtype),
      device=ctx.device_kind.value,
  )


def _make_forward_factory(num_outputs: int) -> registry_lib.Factory:
  """Creates a forward benchmark factory with `num_outputs` outputs."""

  def factory(ctx: context_lib.Context):
    _check_scope(ctx, num_outputs)
    model = _ManyOutputs(
        num_outputs,
        dtype=torch_device_ops.get_torch_dtype(ctx.dtype),
        device=ctx.device_kind.value,
    ).eval()
    return model, (_make_input(ctx),), {}

  return factory


def _make_training_factory(num_params: int) -> registry_lib.Factory:
  """Creates a training benchmark factory with `num_params` parameters."""

  def factory(ctx: context_lib.Context):
    _check_scope(ctx, num_params)
    model = _ManyParamsScalarLoss(
        num_params,
        dtype=torch_device_ops.get_torch_dtype(ctx.dtype),
        device=ctx.device_kind.value,
    ).train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        capturable=True,
        fused=True,
    )
    return model, (_make_input(ctx),), {}, opt

  return factory


def _register_synthetic_benchmarks() -> None:
  for n in _NUM_OUTPUTS_SWEEP:
    registry_lib.add_benchmark(
        factory=_make_forward_factory(n),
        stepper=step_lib.StepperType.FORWARD,
        name=f"synthetic_{n}_outputs_forward",
    )
    registry_lib.add_benchmark(
        factory=_make_training_factory(n),
        stepper=step_lib.StepperType.TRAINING,
        name=f"synthetic_{n}_params_training",
    )


_register_synthetic_benchmarks()
