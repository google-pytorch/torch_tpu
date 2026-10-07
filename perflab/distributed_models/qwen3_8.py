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

"""Qwen3.8 (4 layers) SimpleFSDP training benchmark for the distributed harness.

Qwen3.8 is the torchtitan `qwen3_5` architecture (`torchtitan.models.qwen3_8` is
only a registry over it; Qwen3.8 differs from Qwen3.5 only in checkpoint
weights, which a synthetic benchmark does not load): a hybrid decoder with three
GatedDeltaNet (linear attention) layers per gated softmax attention layer. The
model is the dense 27B flavor truncated to 4 layers (one full 3:1
GatedDeltaNet:Attention cycle), trained with torchtitan's fully-sharded
SimpleFSDP, per-layer activation checkpointing and AdamW, in pure bf16 (params,
grads, optimizer state and gradient reduction; no fp32 master weights), like the
other perflab training benchmarks.

torchtitan's GatedDeltaNet kernel calls the CUDA/Triton-only `attn_gym` package,
which is unavailable in google3; `model_utils/torchtitan_qwen3_5` swaps every
`GatedDeltaKernel` for a pure-torch chunked gated delta rule, so the rest of the
model is unmodified torchtitan.

This module is kept separate from `qwen.py` and only built internally: it is the
only distributed benchmark that depends on torchtitan.
"""

import functools
from typing import Any

from absl import logging
import torch
from torch import nn
from torch.utils import checkpoint as checkpoint_utils
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops
from perflab.model_utils import torchtitan_qwen3_5
from torchtitan.experiments.graph_trainer import simple_fsdp

_SUPPORTED_PLATFORMS: frozenset[target_lib.Platform] = frozenset({
    # world_size=2: one chip, one process per TensorCore.
    target_lib.Platform.V7_1X1X1_2TC,
    # world_size=4: two chips x 2 TensorCores.
    target_lib.Platform.V7_1X2X1,
    # world_size=8: one host, 4 chips x 2 TensorCores.
    target_lib.Platform.V7_2X2X1,
    # GPU counterparts with the same world sizes (one process per B200).
    target_lib.Platform.B200_2,
    target_lib.Platform.B200_4,
    target_lib.Platform.B200_8,
})

# Per-rank sequence length in RunScope.FULL (local batch 1, bf16) on every
# supported platform. GPUs use the TPU value for a like-for-like comparison, not
# the largest that fits in B200 HBM.
_FULL_SEQ_LEN = 2048


def build_model(
    config: torchtitan_qwen3_5.qwen3_5.Qwen35Model.Config,
    device: str,
    world_size: int,
    dtype: torch.dtype,
) -> nn.Module:
  """Builds the torchtitan model in `dtype`, checkpoints and SimpleFSDP-shards it.

  Training is pure `dtype` (bf16 by default): parameters, gradients, AdamW
  state and the gradient reduce-scatter are all in `dtype`.

  Args:
    config: torchtitan Qwen3.5 model config.
    device: Device string (e.g. "tpu", "cuda" or "cpu").
    world_size: Number of ranks to fully shard the parameters across.
    dtype: Parameter / compute / gradient-reduction dtype.

  Returns:
    The SimpleFSDP-wrapped model in train mode.
  """
  # Built on meta and materialized directly on the device: every rank holding a
  # full CPU copy would exhaust host RAM at ws8.
  torch.manual_seed(12345)
  model = torchtitan_qwen3_5.build_model(config, device, dtype)
  model.train()

  # Per-layer activation checkpointing, as SimpleFSDP expects: each layer's
  # all-gathered weights are freed after its forward and re-gathered for the
  # backward recompute, so memory and step time reflect sharded weights rather
  # than all layers' unsharded weights staying resident until backward.
  for layer in model.layers.values():
    layer.forward = functools.partial(
        checkpoint_utils.checkpoint, layer.forward, use_reentrant=False
    )

  mesh = torch.distributed.device_mesh.init_device_mesh(
      device, (world_size,), mesh_dim_names=("fsdp",)
  )
  # SimpleFSDP sums gradients across ranks (no 1 / world_size scaling). That
  # does not change the work done per step, but a real training loop would
  # scale the loss or learning rate.
  model = simple_fsdp.data_parallel(
      model,
      device_mesh=mesh,
      mode="fully_shard",
      mp_policy=simple_fsdp.MixedPrecisionPolicy(
          param_dtype=dtype,
          reduce_dtype=dtype,
      ),
  )
  if device == target_lib.DeviceKind.TPU.value:
    # Lets torch_tpu fuse the SimpleFSDP collectives with the surrounding ops
    # (what torchtitan's TPU `spmd_utils.apply_spmd_safe` does). Under
    # torch.compile this annotates the traced forward nodes, and AOTAutograd
    # copies the annotation to their backward nodes (including the gradient
    # reduce-scatters). `torch.tpu` only exists on TPU, hence the lookup here.
    model.forward = torch.tpu.distributed.spmd_safe(model.forward)
  return model


def cross_entropy_loss(
    model: nn.Module,
    input_args: tuple[Any, ...],
    input_kwargs: dict[str, Any],
) -> torch.Tensor:
  """Next-token cross entropy over fp32 logits (torchtitan returns logits)."""
  input_kwargs = dict(input_kwargs)
  labels = input_kwargs.pop("labels")
  logits = model(*input_args, **input_kwargs)
  return nn.functional.cross_entropy(logits.float(), labels)


def make_inputs(
    vocab_size: int, seq_len: int, rank: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns one packed `(seq_len,)` token / label sequence for `rank`.

  torchtitan takes packed `(T,)` tokens. The data differs per rank, as in real
  data-parallel training.

  Args:
    vocab_size: Token ids are drawn from `[0, vocab_size)`.
    seq_len: Tokens per rank.
    rank: Seeds the per-rank data.
    device: Device to place the tensors on.
  """
  generator = torch.Generator().manual_seed(1234 + rank)
  tokens, labels = [
      torch.randint(
          0, vocab_size, (seq_len,), dtype=torch.long, generator=generator
      ).to(device)
      for _ in range(2)
  ]
  return tokens, labels


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    # Regular training infra: zero_grad + forward + loss + backward +
    # optimizer.step() wrapped in one torch.compile (Scope.STEP). Explicitly NOT
    # SingleTraceTrainer.
    stepper_kwargs={
        "use_single_trace": False,
        "compute_loss": cross_entropy_loss,
    },
    compile_config=compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=False
    ),
    # Training is only benchmarked compiled.
    skipped_run_modes={
        "eager_default",
        "eager_optimized",
        "eager_defer_never_and_launch_blocking",
    },
)
def qwen3_8_27b_4_layers_simple_fsdp_training(
    ctx: context_lib.Context,
) -> tuple[
    torch.nn.Module,
    tuple[torch.Tensor, ...],
    dict[str, Any],
    torch.optim.Optimizer,
]:
  """Benchmark factory for Qwen3.8 (27B arch, 4 layers) SimpleFSDP training."""
  if ctx.target.platform not in _SUPPORTED_PLATFORMS:
    raise target_lib.UnsupportedBenchmark(
        "Qwen3.8 SimpleFSDP training is not supported on platform"
        f" {ctx.target.platform}"
    )
  if not torch.distributed.is_initialized():
    raise target_lib.UnsupportedBenchmark(
        "Qwen3.8 SimpleFSDP training requires an initialized torch.distributed"
        " process group (run it through distributed_runner)."
    )
  world_size = torch.distributed.get_world_size()
  rank = torch.distributed.get_rank()

  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    seq_len = 128
    num_layers = 2
  else:
    seq_len = _FULL_SEQ_LEN
    num_layers = 4

  device_str = ctx.device_kind.value
  config = torchtitan_qwen3_5.text_config("27B", num_layers, seq_len)
  logging.info(
      "Rank %d/%d: building torchtitan Qwen3.8 (27B arch) num_layers=%d"
      " seq_len=%d with SimpleFSDP fully_shard",
      rank,
      world_size,
      num_layers,
      seq_len,
  )
  model = build_model(
      config,
      device_str,
      world_size,
      dtype=torch_device_ops.get_torch_dtype(ctx.dtype),
  )
  tokens, labels = make_inputs(config.vocab_size, seq_len, rank, device_str)
  # capturable=True keeps the step counter on device, so the compiled step has
  # no host sync.
  opt = torch.optim.AdamW(
      model.parameters(), lr=1e-4, fused=True, capturable=True
  )
  return model, (tokens,), {"labels": labels}, opt
