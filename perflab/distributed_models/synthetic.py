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

"""Synthetic distributed benchmarks that isolate per-output host dispatch cost.

Distributed counterparts of `models/synthetic.py`. Each benchmark runs a
compiled program with negligible device work but many small, independent
outputs on every rank, so post-warmup step time is dominated by host-side work
that scales with the number of program outputs, with `world_size` processes
contending for the same host.

* `synthetic_replicated_<N>_outputs_forward`: every rank runs a forward
  program returning N tensors, plus one scalar all-reduce that keeps ranks in
  lockstep. Isolates per-rank dispatch overhead under multi-process host
  contention.
* `synthetic_fsdp_<N>_params_training`: a SimpleFSDP (fully sharded) training
  step over N small parameters, each in its own module, with AdamW. Every
  parameter adds an all-gather and a reduce-scatter, and the compiled step has
  roughly 3N outputs per rank (parameter shards plus both moment buffers).

This module depends on torchtitan's SimpleFSDP, so it is only built internally.
"""

import torch
from torch import nn
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops
from torchtitan.experiments.graph_trainer import simple_fsdp

_SUPPORTED_PLATFORMS: frozenset[target_lib.Platform] = frozenset({
    # world_size=2: one chip, one process per TensorCore.
    target_lib.Platform.V7_1X1X1_2TC,
    # world_size=4: two chips x 2 TensorCores.
    target_lib.Platform.V7_1X2X1,
    # world_size=8: one host, 4 chips x 2 TensorCores.
    target_lib.Platform.V7_2X2X1,
})

# Dim 0 is the FSDP shard dim and must divide every supported world size.
_TENSOR_SHAPE = (8, 128)
_NUM_OUTPUTS_SWEEP = (16, 256, 1024)
# Presubmit only smoke-tests that the benchmark compiles and runs.
_PRESUBMIT_MAX_NUM_OUTPUTS = 16

_EAGER_RUN_MODES = frozenset({
    "eager_default",
    "eager_optimized",
    "eager_defer_never_and_launch_blocking",
})


class _Scale(nn.Module):
  """Holds one small parameter, so SimpleFSDP gathers it individually."""

  def __init__(self, dtype: torch.dtype, device: str):
    super().__init__()
    self.weight = nn.Parameter(
        torch.randn(_TENSOR_SHAPE, dtype=dtype, device=device)
    )


class _ManyOutputs(nn.Module):
  """Returns one small, independent tensor per parameter."""

  def __init__(self, num_outputs: int, dtype: torch.dtype, device: str):
    super().__init__()
    self.scales = nn.ModuleList(
        _Scale(dtype, device) for _ in range(num_outputs)
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(x * scale.weight for scale in self.scales)


class _ManyOutputsSynced(_ManyOutputs):
  """`_ManyOutputs` plus one scalar all-reduce that keeps ranks in lockstep.

  Without any collective each rank runs its own loop, so ranks that finish
  compiling first time their steps while peers on the same host are still
  compiling, and measure host contention rather than dispatch cost.
  """

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    sync = x.sum()
    torch.distributed.all_reduce(sync)
    return (*super().forward(x), sync)


class _ManyParamsScalarLoss(_ManyOutputs):
  """Reduces every parameter into one scalar loss so all of them get grads."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:  # pytype: disable=signature-mismatch
    return torch.stack(
        [(x * scale.weight).sum() for scale in self.scales]
    ).sum()


def _check_supported(ctx: context_lib.Context) -> None:
  """Raises UnsupportedBenchmark if this configuration should not run."""
  if ctx.target.platform not in _SUPPORTED_PLATFORMS:
    raise target_lib.UnsupportedBenchmark(
        f"Synthetic distributed benchmarks do not support {ctx.target.platform}"
    )
  if not torch.distributed.is_initialized():
    raise target_lib.UnsupportedBenchmark(
        "Synthetic distributed benchmarks require an initialized"
        " torch.distributed process group (run them through"
        " distributed_runner)."
    )


def make_input(rank: int, dtype: torch.dtype, device: str) -> torch.Tensor:
  """Returns a per-rank input, as in data-parallel training."""
  generator = torch.Generator().manual_seed(1234 + rank)
  return torch.randn(_TENSOR_SHAPE, generator=generator, dtype=dtype).to(device)


def build_fsdp_model(
    num_params: int,
    device: str,
    world_size: int,
    dtype: torch.dtype,
) -> nn.Module:
  """Builds a scalar-loss model over `num_params` params and SimpleFSDP-shards it.

  Args:
    num_params: Number of small parameters, each in its own module.
    device: Device string (e.g. "tpu" or "cpu").
    world_size: Number of ranks to fully shard the parameters across.
    dtype: Parameter / compute / gradient-reduction dtype.

  Returns:
    The SimpleFSDP-wrapped model in train mode.
  """
  torch.manual_seed(12345)
  model = _ManyParamsScalarLoss(num_params, dtype=dtype, device=device)
  model.train()
  mesh = torch.distributed.device_mesh.init_device_mesh(
      device, (world_size,), mesh_dim_names=("fsdp",)
  )
  model = simple_fsdp.data_parallel(
      model,
      device_mesh=mesh,
      mode="fully_shard",
      mp_policy=simple_fsdp.MixedPrecisionPolicy(
          param_dtype=dtype, reduce_dtype=dtype
      ),
  )
  if device == target_lib.DeviceKind.TPU.value:
    # Lets torch_tpu fuse the SimpleFSDP collectives with the surrounding ops,
    # as in `qwen3_8.py`. `torch.tpu` only exists on TPU.
    model.forward = torch.tpu.distributed.spmd_safe(model.forward)
  return model


def build_replicated_model(
    num_outputs: int, device: str, dtype: torch.dtype
) -> nn.Module:
  """Builds the replicated forward model with `num_outputs` outputs.

  The model returns `num_outputs` tensors followed by the all-reduced sum of its
  input, so it needs an initialized process group.

  Args:
    num_outputs: Number of independent tensor outputs.
    device: Device string (e.g. "tpu" or "cpu").
    dtype: Parameter / compute dtype.

  Returns:
    The model in eval mode.
  """
  model = _ManyOutputsSynced(num_outputs, dtype=dtype, device=device).eval()
  if device == target_lib.DeviceKind.TPU.value:
    # Keeps the all-reduce in the same program as the outputs instead of
    # splitting the step into separate compute and collective executables.
    # `torch.tpu` only exists on TPU.
    model.forward = torch.tpu.distributed.spmd_safe(model.forward)
  return model


def _make_replicated_forward_factory(num_outputs: int) -> registry_lib.Factory:
  """Creates a replicated forward benchmark factory with `num_outputs` outputs."""

  def factory(ctx: context_lib.Context):
    _check_supported(ctx)
    device = ctx.device_kind.value
    dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
    model = build_replicated_model(num_outputs, device, dtype)
    inputs = make_input(torch.distributed.get_rank(), dtype, device)
    return model, (inputs,), {}

  return factory


def _make_fsdp_training_factory(num_params: int) -> registry_lib.Factory:
  """Creates a SimpleFSDP training benchmark factory with `num_params` params."""

  def factory(ctx: context_lib.Context):
    _check_supported(ctx)
    device = ctx.device_kind.value
    dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
    model = build_fsdp_model(
        num_params, device, torch.distributed.get_world_size(), dtype
    )
    # capturable=True keeps the step counter on device, so the compiled step has
    # no host sync.
    opt = torch.optim.AdamW(
        model.parameters(), lr=1e-4, fused=True, capturable=True
    )
    inputs = make_input(torch.distributed.get_rank(), dtype, device)
    return model, (inputs,), {}, opt

  return factory


def _presubmit_policy(
    num_outputs: int,
) -> tuple[registry_lib.PresubmitPolicy, str | None]:
  """Returns the presubmit policy and reason for a sweep point."""
  if num_outputs <= _PRESUBMIT_MAX_NUM_OUTPUTS:
    return registry_lib.PresubmitPolicy.RUN, None
  return (
      registry_lib.PresubmitPolicy.SKIP,
      f"{num_outputs} outputs only run in the full scope.",
  )


# Compile settings differ from `models/synthetic.py` on purpose, to match the
# other distributed benchmarks:
# * dynamic=False, as in `qwen.py` and `qwen3_8.py`: all shapes are static, and
#   this keeps the TPU backend config (`TpuBackend(dynamism=False)`) identical
#   to the real-model distributed benchmarks these are compared against.
# * Eager modes are skipped: these benchmarks measure compiled-program dispatch.
#   Per-op eager overhead is already covered single-device by
#   `models/synthetic.py`.
def _register_synthetic_distributed_benchmarks() -> None:
  for n in _NUM_OUTPUTS_SWEEP:
    presubmit, presubmit_reason = _presubmit_policy(n)
    registry_lib.add_benchmark(
        factory=_make_replicated_forward_factory(n),
        stepper=step_lib.StepperType.FORWARD,
        # Scope.MODEL compiles the `spmd_safe` forward itself, as in the
        # distributed `qwen.py` forward benchmark.
        compile_config=compile_lib.CompileConfig(
            scope=compile_lib.Scope.MODEL, dynamic=False
        ),
        name=f"synthetic_replicated_{n}_outputs_forward",
        skipped_run_modes=_EAGER_RUN_MODES,
        presubmit=presubmit,
        presubmit_reason=presubmit_reason,
        # Measures host overhead; has no batch.
        workload=None,
    )
    registry_lib.add_benchmark(
        factory=_make_fsdp_training_factory(n),
        stepper=step_lib.StepperType.TRAINING,
        # The whole zero_grad + forward + backward + optimizer.step() in one
        # torch.compile (Scope.STEP), not SingleTraceTrainer, as in
        # `qwen3_8.py`.
        stepper_kwargs={"use_single_trace": False},
        compile_config=compile_lib.CompileConfig(
            scope=compile_lib.Scope.STEP, dynamic=False
        ),
        name=f"synthetic_fsdp_{n}_params_training",
        skipped_run_modes=_EAGER_RUN_MODES,
        presubmit=presubmit,
        presubmit_reason=presubmit_reason,
        # Measures host overhead; has no batch.
        workload=None,
    )


_register_synthetic_distributed_benchmarks()
