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

"""Qwen model benchmark definitions for distributed benchmark harness."""

import math
import re
from typing import Any

from absl import logging
from fairscale.nn.model_parallel import initialize
from fairscale.nn.model_parallel import layers
import torch
from torch import nn
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness import registry as registry_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness import torch_device_ops
from examples.benchmarks.e2e.harness.model_utils import ragged_moe
from tests import module_registry
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

_SUPPORTED_PLATFORMS: frozenset[target_lib.Platform] = frozenset({
    target_lib.Platform.V7_2X2X2,
    target_lib.Platform.V7_2X2X4,
    target_lib.Platform.V7_2X4X4,
})


def _warn_if_attn_dimensions_not_divisible(
    default_config: modeling_qwen3_5_moe.Qwen3_5MoeConfig,
    world_size: int,
    rank: int,
) -> None:
  """Checks if tensor parallelism can be applied."""
  if default_config.num_attention_heads % world_size != 0:
    logging.info(
        "Rank %d: Tensor Parallelism Info: num_attention_heads"
        " (%d) must be divisible by world_size (%d). Use a world_size that is a"
        " divisor of %d and change gather_output to False for self_attn linear"
        " layer.",
        rank,
        default_config.num_attention_heads,
        world_size,
        default_config.num_attention_heads,
    )

  if default_config.num_key_value_heads % world_size != 0:
    logging.info(
        "Rank %d: Tensor Parallelism Info: num_key_value_heads"
        " (%d) must be divisible by world_size (%d). Use a world_size that is a"
        " divisor of %d and change gather_output to False for self_attn linear"
        " layer.",
        rank,
        default_config.num_key_value_heads,
        world_size,
        default_config.num_key_value_heads,
    )


def _apply_tensor_parallel_plan(
    module: nn.Module,
    name_prefix: str = "",
    tp_plan: dict[str, str] | None = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
  """Applies a tensor parallel plan to a model."""
  if not tp_plan:
    return

  for name, child in module.named_children():
    full_name = f"{name_prefix}.{name}" if name_prefix else name
    full_name = full_name.removeprefix("model.")

    if isinstance(child, nn.Linear):
      for pattern, tp_type in tp_plan.items():
        if re.fullmatch(pattern, full_name):
          logging.info(
              "Rank %d: Replacing %s with %s parallel layer.",
              rank,
              full_name,
              tp_type,
          )
          original_linear = child
          new_linear = None

          # Setting gather_output=False for ColumnParallelLinear layer does not
          # work for self_attn linear layer. It requires world_size to be the
          # divisor of num_attention_heads (32) and num_key_value_heads (2),
          # which is not supported yet.
          if "self_attn" in full_name or "linear_attn" in full_name:
            gather_output = True
          else:
            gather_output = False

          if tp_type in ("colwise", "colwise_gather_output"):
            new_linear = layers.ColumnParallelLinear(
                original_linear.in_features,
                original_linear.out_features,
                bias=original_linear.bias is not None,
                gather_output=gather_output,
                init_method=lambda w: w,
            )
          elif tp_type == "rowwise":
            new_linear = layers.RowParallelLinear(
                original_linear.in_features,
                original_linear.out_features,
                bias=original_linear.bias is not None,
                input_is_parallel=not gather_output,
                init_method=lambda w: w,
            )

          if new_linear:
            setattr(module, name, new_linear)
            logging.info(
                "Rank %d: Replaced %s (tp_type=%s, %s -> %s)."
                " in=%s, out=%s. weight: %s -> %s;",
                rank,
                full_name,
                tp_type,
                type(original_linear).__name__,
                type(new_linear).__name__,
                original_linear.in_features,
                original_linear.out_features,
                original_linear.weight.shape,
                new_linear.weight.shape,
            )
          break  # Found a match, move to the next child

    # Recurse for submodules
    _apply_tensor_parallel_plan(child, full_name, tp_plan, world_size, rank)


def _initialize_weights_and_biases(module: nn.Module) -> None:
  """Initializes weights and biases for a module."""
  if isinstance(module, modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm):
    torch.nn.init.zeros_(module.weight)
    return

  if hasattr(module, "weight") and module.weight is not None:
    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    logging.debug(
        "Initializing weight: %s %s",
        module.weight.shape,
        module.weight.dtype,
    )

  if hasattr(module, "bias") and module.bias is not None:
    torch.nn.init.zeros_(module.bias)
    logging.debug(
        "Initializing bias: %s %s",
        module.bias.shape,
        module.bias.dtype,
    )

  for attr in ("gate_up_proj", "down_proj"):
    if hasattr(module, attr):
      param = getattr(module, attr)
      if isinstance(param, torch.Tensor):
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
        logging.debug(
            "Initializing %s: %s %s",
            attr,
            param.shape,
            param.dtype,
        )

  # TODO: Remove custom ragged moe implementation. It's not needed now that
  # torch.grouped_mm lowers to RaggedMoe implementation.
  if isinstance(module, ragged_moe.RaggedMoeQwen35):
    with torch.no_grad():
      hidden_size = module.up.shape[1]
      torch.nn.init.normal_(module.up, std=1.0 / math.sqrt(hidden_size))
      torch.nn.init.normal_(module.gate, std=1.0 / math.sqrt(hidden_size))
      world_size = (
          torch.distributed.get_world_size()
          if module.is_tensor_parallel and torch.distributed.is_initialized()
          else 1
      )
      intermediate_dim = module.down.shape[1] * world_size
      torch.nn.init.normal_(module.down, std=1.0 / math.sqrt(intermediate_dim))

  elif isinstance(module, modeling_qwen3_5_moe.Qwen3_5MoeTextRotaryEmbedding):
    if module.rope_type is not None and module.rope_type != "default":
      rope_init_fn = modeling_qwen3_5_moe.ROPE_INIT_FUNCTIONS[module.rope_type]
    else:
      rope_init_fn = module.compute_default_rope_parameters
    inv_freq, attention_scaling = rope_init_fn(
        module.config, module.inv_freq.device
    )
    with torch.no_grad():
      module.inv_freq.copy_(inv_freq)
      module.original_inv_freq.copy_(inv_freq)
      module.attention_scaling = attention_scaling


# TODO: Remove custom ragged moe implementation. It's not needed now that
# torch.grouped_mm lowers to RaggedMoe implementation.
def _replace_moe_layers_with_ragged_dot_moe(
    model: modeling_qwen3_5_moe.Qwen3_5MoeForCausalLM,
    config: Any,
    world_size: int,
) -> None:
  for layer in model.model.layers:
    if not isinstance(layer.mlp, modeling_qwen3_5_moe.Qwen3_5MoeSparseMoeBlock):
      raise TypeError(
          "Expected layer.mlp to be Qwen3_5MoeSparseMoeBlock, got"
          f" {type(layer.mlp)}"
      )
    layer.mlp = ragged_moe.RaggedMoeQwen35(
        config, is_tensor_parallel=world_size > 1
    )


def get_qwen3_5_397b_a17b_model(
    rank: int,
    world_size: int,
    device: str,
    torch_dtype: torch.dtype,
    use_ragged_dot_moe: bool,
    num_hidden_layers: int = -1,
) -> modeling_qwen3_5_moe.Qwen3_5MoeForCausalLM:
  """Returns the Qwen3.5-397B-A17B HuggingFace model."""
  logging.info("Rank %d: Getting Qwen3.5-397B-A17B model.", rank)

  def modify_config_hook(base_config):
    # Qwen3_5MoeForCausalLM expects Qwen3_5MoeTextConfig, which is nested within
    # the multimodal Qwen3_5MoeConfig returned by AutoConfig.
    config = (
        base_config.get_text_config()
        if hasattr(base_config, "get_text_config")
        else base_config
    )
    if num_hidden_layers >= 0:
      config.num_hidden_layers = num_hidden_layers
      config.layer_types = config.layer_types[: config.num_hidden_layers]
    return config

  registry = module_registry.ModuleRegistry()
  module_spec = registry.get_module_spec(
      "transformers",
      "Qwen/Qwen3.5-397B-A17B",
      load_weights=False,
      modify_config_hook=modify_config_hook,
  )
  config = module_spec.config
  assert config is not None

  # 1. Instantiate on meta device without allocating storage
  with torch.device("meta"):
    torch.set_default_dtype(torch_dtype)
    model = modeling_qwen3_5_moe.Qwen3_5MoeForCausalLM(config)

  with torch.device(device):
    torch.manual_seed(12345)
    torch.set_default_dtype(torch_dtype)

    # 2. Shard model / replace MoE layers
    if use_ragged_dot_moe:
      _replace_moe_layers_with_ragged_dot_moe(model, config, world_size)

    if world_size > 1:
      # FairScale library setup for distributed model parallelism.
      initialize.initialize_model_parallel(world_size)

      _warn_if_attn_dimensions_not_divisible(config, world_size, rank)
      _apply_tensor_parallel_plan(
          model,
          name_prefix="",
          tp_plan=config.base_model_tp_plan,
          world_size=world_size,
          rank=rank,
      )

    # 3. Materialize on the local device
    model.to_empty(device=device)

    # 4. Custom initialization handling non-standard layers
    model.apply(_initialize_weights_and_biases)

  logging.info(
      "Rank %d: Successfully initialized Qwen3.5-397B-A17B model.", rank
  )
  logging.debug("Model on Rank %d: %s", rank, model)
  return model


def _load_qwen3_5_397b(
    ctx: context_lib.Context,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
  """Loads the Qwen3.5-397B-A17B model with Tensor Parallelism."""
  if not torch.distributed.is_initialized():
    raise target_lib.UnsupportedBenchmark(
        "Qwen 397B benchmark requires an initialized torch.distributed"
        " environment for FairScale model parallelism."
    )

  world_size = torch.distributed.get_world_size()
  rank = torch.distributed.get_rank()

  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_hidden_layers = 2
  else:
    batch_size = 1
    seq_len = 2048
    num_hidden_layers = 60

  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device_str = ctx.device_kind.value

  prev_dtype = torch.get_default_dtype()
  try:
    model = get_qwen3_5_397b_a17b_model(
        rank=rank,
        world_size=world_size,
        device=device_str,
        torch_dtype=dtype,
        use_ragged_dot_moe=True,
        num_hidden_layers=num_hidden_layers,
    )
    model.eval()

    input_ids = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), device=device_str
    )
    attention_mask = torch.ones(
        (batch_size, seq_len), dtype=torch.int64, device=device_str
    )
    return model, (input_ids, attention_mask)
  finally:
    torch.set_default_dtype(prev_dtype)


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    compile_config=compile_lib.CompileConfig(
        scope=compile_lib.Scope.MODEL, dynamic=False
    ),
    skipped_run_modes={"eager_default", "eager_optimized"},
)
def qwen3_5_397b_a17b_forward(
    ctx: context_lib.Context,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...], dict[str, Any]]:
  """Benchmark factory for Qwen 3.5 397B forward across multi-host slices."""
  if ctx.target.platform not in _SUPPORTED_PLATFORMS:
    raise target_lib.UnsupportedBenchmark(
        f"Qwen 397B is not supported on platform {ctx.target.platform}"
    )

  model, example_inputs = _load_qwen3_5_397b(ctx)
  return model, example_inputs, {}
