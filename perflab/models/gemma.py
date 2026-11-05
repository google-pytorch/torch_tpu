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

"""Gemma model benchmark definitions."""

from typing import Any
import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib
from perflab.model_utils import ragged_moe


def _load_hf_gemma(
    ctx: context_lib.Context,
    model_name: str,
    is_training: bool,
    seq_len: int = 2048,
    batch_size: int = 2,
    num_layers: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads the HuggingFace Gemma model and generates sample inputs for benchmarking."""

  def _truncate_layers(cfg: Any, n: int) -> None:
    orig_layers = getattr(cfg, "num_hidden_layers", None)
    orig_shared = getattr(cfg, "num_kv_shared_layers", None)
    if (
        isinstance(orig_layers, int)
        and isinstance(orig_shared, int)
        and orig_shared > 0
    ):
      first_shared = orig_layers - orig_shared
      cfg.num_kv_shared_layers = max(0, n - first_shared)
    if hasattr(cfg, "num_hidden_layers"):
      cfg.num_hidden_layers = n
    if getattr(cfg, "layer_types", None) is not None:
      cfg.layer_types = cfg.layer_types[:n]

  def _modify_config_hook(config: Any) -> Any:
    setattr(config, "use_cache", False)
    if num_layers:
      if hasattr(config, "text_config") and config.text_config is not None:
        _truncate_layers(config.text_config, num_layers)
      _truncate_layers(config, num_layers)
    return config

  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device_str = ctx.device_kind.value

  return model_helpers.load_hf_llm(
      model_name,
      dtype=dtype,
      device_str=device_str,
      is_training=is_training,
      seq_len=seq_len,
      batch_size=batch_size,
      modify_config_hook=_modify_config_hook,
  )


# --- Gemma 3 270M ---


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(16, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_3_270m_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-3 270M forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers = 6
  else:
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-3-270m",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(4, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_3_270m_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-3 270M training."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers = 6
  else:
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-3-270m",
      is_training=True,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(16, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_3_270m_decode(ctx: context_lib.Context):
  """Benchmark factory for Gemma-3 270M decode."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers = 6
  else:
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-3-270m",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


# --- Gemma 2 2B ---


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_2_2b_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-2 2B forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers = 2
  else:
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-2-2b",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_2_2b_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-2 2B training."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers = 2
  else:
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-2-2b",
      is_training=True,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_2_2b_decode(ctx: context_lib.Context):
  """Benchmark factory for Gemma-2 2B decode."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers = 2
  else:
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-2-2b",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=num_layers,
  )
  return model, (), inputs


# --- Gemma 4 31B (6 Layers) ---


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 1024),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_4_31b_6_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 31B 6 layers forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-31b",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=6,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY,
    presubmit_reason=(
        "Gemma 4 31B training uses too much memory to run on TPUs with"
        " smaller memory."
    ),
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 1024),
        presubmit=workload_lib.Workload(
            model_helpers.SMOKE_BATCH_SIZE, model_helpers.SMOKE_SEQ_LEN
        ),
    ),
)
def gemma_4_31b_6_layers_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 31B 6 layers training."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-31b",
      is_training=True,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=6,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 1024),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_4_31b_6_layers_decode(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 31B 6 layers decode."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-31b",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=6,
  )
  return model, (), inputs


# --- Gemma 4 E2B ---


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 1024),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_4_e2b_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 E2B forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    # Covers the all the sliding window attention + full attention layers.
    num_layers = 5
  else:
    num_layers = None
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-e2b",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY,
    presubmit_reason=(
        "Gemma 4 E2B training uses too much memory to run on TPUs with"
        " smaller memory."
    ),
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 1024),
        presubmit=workload_lib.Workload(
            model_helpers.SMOKE_BATCH_SIZE, model_helpers.SMOKE_SEQ_LEN
        ),
    ),
)
def gemma_4_e2b_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 E2B training."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if model_helpers.is_presubmit_smoke(ctx):
    # Layers 0-4 cover both the sliding-window and full-attention layer types;
    # the KV-shared layers are not built.
    num_layers = 5
  else:
    num_layers = None
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-e2b",
      is_training=True,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 1024),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_4_e2b_decode(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 E2B decode."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    # Covers the all the sliding window attention + full attention layers.
    num_layers = 5
  else:
    num_layers = None
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-e2b",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


# --- Gemma 4 26B A4B Ragged MoE ---


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_4_26b_a4b_6_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 26B-A4B Ragged MoE forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-26B-A4B",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=6,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY,
    presubmit_reason=(
        "Gemma 4 26B-A4B training uses too much memory to run on TPUs with"
        " smaller memory."
    ),
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 2048),
        presubmit=workload_lib.Workload(
            model_helpers.SMOKE_BATCH_SIZE, model_helpers.SMOKE_SEQ_LEN
        ),
    ),
)
def gemma_4_26b_a4b_6_layers_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 26B-A4B Ragged MoE training."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-26B-A4B",
      is_training=True,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=6,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_4_26b_a4b_6_layers_decode(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 26B-A4B Ragged MoE decode."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-26B-A4B",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=6,
  )
  return model, (), inputs


# --- Gemma 4 26B A4B with ragged-dot experts (12 Layers) ---


def _load_hf_gemma_ragged_moe(
    ctx: context_lib.Context,
    model_name: str,
    seq_len: int,
    batch_size: int,
    num_layers: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a Gemma 4 MoE model whose experts use `torch.tpu.ops.ragged_dot`.

  Replaces the HuggingFace experts (`torch._grouped_mm`) of every MoE layer
  with `ragged_moe.RaggedExpertsGemma4`.

  Args:
    ctx: The benchmark context.
    model_name: HuggingFace model identifier.
    seq_len: Tokens per sequence.
    batch_size: Number of sequences per step.
    num_layers: Number of decoder layers.

  Returns:
    A tuple of (model, inputs dictionary).
  """
  model, inputs = _load_hf_gemma(
      ctx,
      model_name,
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  text_config = model.config.get_text_config()
  with torch.device(ctx.device_kind.value):
    for layer in model.model.language_model.layers:
      if getattr(layer, "enable_moe_block", False):
        # Free the HuggingFace experts before allocating their replacement.
        layer.experts = None
        layer.experts = ragged_moe.RaggedExpertsGemma4(text_config, dtype=dtype)
  # Randomly initialized weights overflow in bfloat16 without rescaling.
  model.apply(model_helpers.init_small_random_weights)
  # Gemma 4 takes `mm_token_type_ids` only with image inputs.
  inputs.pop("mm_token_type_ids", None)
  return model, inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_4_26b_a4b_ragged_moe_12_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 26B-A4B (12 layers) ragged-dot MoE forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers = 2
  else:
    num_layers = 12
  model, inputs = _load_hf_gemma_ragged_moe(
      ctx,
      "google/gemma-4-26B-A4B",
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(2, 2048),
        presubmit=workload_lib.Workload(1, 128),
    ),
)
def gemma_4_26b_a4b_ragged_moe_12_layers_decode(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 26B-A4B (12 layers) ragged-dot MoE decode."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers = 2
  else:
    num_layers = 12
  model, inputs = _load_hf_gemma_ragged_moe(
      ctx,
      "google/gemma-4-26B-A4B",
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs
