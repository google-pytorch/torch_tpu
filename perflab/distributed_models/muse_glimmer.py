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

"""Distributed Muse Glimmer benchmark definitions.

The model and its training setup are those of the team's Muse Glimmer TPU
benchmark (`torch_tpu/google/prototypes/glimmer/glimmer_tpu_test.py`); see
`model_utils/muse_glimmer`. The model is the text-only 30B flavor truncated to
its last 4 layers (of 52), one full cycle of its attention pattern: 3 sliding
window (2048 tokens) RoPE layers and a global NoPE layer. It is fully sharded
over all ranks with SimpleFSDP (`spmd_types` backend, bf16 compute, fp32
gradient reduction), with full activation checkpointing and AdamW after
gradient clipping. On TPU every block's forward is `spmd_safe`, so its
collectives fuse with the surrounding ops.

Each rank trains on `batch_size` packed documents of 2048 tokens (`_bs<N>`
variants for `batch_size > 1`; the prototype's 30B runs use 4): RoPE positions
restart per document and attention is masked per document.

Like the prototype, the benchmark runs eagerly with deferred, fused execution
(`eager_optimized`) as well as with the whole step compiled. The prototype also
makes the eager backward pass `spmd_safe`; the perflab training stepper calls
`backward()` itself, so here only the compiled step's backward is (via the
forward annotations, which AOTAutograd copies to the backward).
"""

import functools
from typing import Any

from absl import logging
import torch
from torch import nn
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops
from perflab.model_utils import muse_glimmer

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

# Document length in RunScope.FULL, as in the prototype's 30B runs. Equal to
# the sliding window.
_FULL_DOC_LEN = 2048
# Per-rank batch sizes (documents per rank), each registered as its own
# benchmark.
_BATCH_SIZES = (1, 2, 4)

# eager_optimized (deferred, fused eager execution) is the prototype's default
# mode; the other eager modes are not benchmarked.
_SKIPPED_RUN_MODES = frozenset({
    "eager_default",
    "eager_defer_never_and_launch_blocking",
})


def muse_glimmer_30b_4_layers_simple_fsdp_training(
    ctx: context_lib.Context,
    batch_size: int = 1,
) -> tuple[
    nn.Module,
    tuple[torch.Tensor, ...],
    dict[str, Any],
    torch.optim.Optimizer,
]:
  """Benchmark factory for Muse Glimmer (30B, 4 layers) SimpleFSDP training.

  Args:
    ctx: Benchmark context.
    batch_size: Documents per rank, packed into one token stream.
  """
  if ctx.target.platform not in _SUPPORTED_PLATFORMS:
    raise target_lib.UnsupportedBenchmark(
        "Muse Glimmer SimpleFSDP training is not supported on platform"
        f" {ctx.target.platform}"
    )
  if not torch.distributed.is_initialized():
    raise target_lib.UnsupportedBenchmark(
        "Muse Glimmer SimpleFSDP training requires an initialized "
        "torch.distributed process group."
    )
  world_size = torch.distributed.get_world_size()
  rank = torch.distributed.get_rank()

  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    flavor = "debugmodel"
    doc_len = 128
  else:
    flavor = "30B"
    doc_len = _FULL_DOC_LEN
  num_layers = 4

  device_str = ctx.device_kind.value
  logging.info(
      "Rank %d/%d: building Muse Glimmer (%s arch) num_layers=%d doc_len=%d"
      " batch_size=%d with SimpleFSDP fully_shard",
      rank,
      world_size,
      flavor,
      num_layers,
      doc_len,
      batch_size,
  )
  bundle = muse_glimmer.build_model(
      flavor,
      num_layers,
      doc_len,
      device_str,
      torch_device_ops.get_torch_dtype(ctx.dtype),
      world_size=world_size,
      spmd_safe=ctx.device_kind == target_lib.DeviceKind.TPU,
  )
  muse_glimmer.allow_block_mask_cache_under_checkpoint()
  tokens, kwargs = muse_glimmer.make_batch(
      bundle, batch_size, doc_len, rank, world_size, device_str
  )
  opt = muse_glimmer.make_optimizer(bundle.model)
  return bundle.model, (tokens,), kwargs, opt


def _register_benchmarks() -> None:
  for batch_size in _BATCH_SIZES:
    suffix = "" if batch_size == 1 else f"_bs{batch_size}"
    registry_lib.add_benchmark(
        factory=functools.partial(
            muse_glimmer_30b_4_layers_simple_fsdp_training,
            batch_size=batch_size,
        ),
        stepper=step_lib.StepperType.TRAINING,
        # zero_grad + forward + loss + backward + optimizer.step(); compiled
        # as one torch.compile (Scope.STEP).
        stepper_kwargs={
            "use_single_trace": False,
            "compute_loss": muse_glimmer.cross_entropy_loss,
        },
        compile_config=compile_lib.CompileConfig(
            scope=compile_lib.Scope.STEP, dynamic=False
        ),
        name=f"muse_glimmer_30b_4_layers_simple_fsdp_training{suffix}",
        skipped_run_modes=_SKIPPED_RUN_MODES,
    )


_register_benchmarks()
