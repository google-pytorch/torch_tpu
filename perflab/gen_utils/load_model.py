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

from collections.abc import MutableMapping
import functools
from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import torch_device_ops


def _set_config_dtype(config: Any, dtype: torch.dtype) -> None:
  """Sets both `dtype` and `torch_dtype` on `config` and nested sub-configs."""
  if config is None:
    return
  if isinstance(config, MutableMapping):
    config["dtype"] = dtype
    config["torch_dtype"] = dtype
    sub_configs = (
        config.get("text_config"),
        config.get("vision_config"),
        config.get("thinker_config"),
    )
  else:
    config.dtype = dtype
    config.torch_dtype = dtype
    sub_configs = (
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
        getattr(config, "thinker_config", None),
    )
  for sub_cfg in sub_configs:
    if sub_cfg is not None:
      _set_config_dtype(sub_cfg, dtype)


def _run_blip_forward_on_model_device(model: torch.nn.Module) -> None:
  """Wraps BLIP `model.forward` in `torch.device` of the model parameters.

  `BlipForQuestionAnswering.forward` and `BlipForImageTextRetrieval.forward` in
  `modeling_blip.py` create `image_attention_mask` / `image_atts` with
  `torch.ones(image_embeds.size()[:-1], dtype=torch.long)` without `device=`.
  On CUDA `compiled` benchmarks, that places the mask on CPU and causes
  TorchInductor to emit a CPU C++ kernel that fails on GPU benchmark hosts
  without a C++ compiler (`InvalidCxxCompiler`).
  TODO(b/327271919): Remove once `modeling_blip.py` passes `device=` upstream or
  Inductor CPU codegen works on GPU hosts.
  """
  if getattr(getattr(model, "config", None), "model_type", None) == "blip":
    orig_forward = model.forward

    @functools.wraps(orig_forward)
    def forward_on_model_device(*args: Any, **kwargs: Any) -> Any:
      with torch.device(next(model.parameters()).device):
        return orig_forward(*args, **kwargs)

    model.forward = forward_on_model_device


def default_load_model(
    ctx: context_lib.Context,
    model_id: str,
    is_training: bool,
):
  """Loads the model and generates sample inputs for benchmarking.

  Args:
    ctx: Benchmark execution context containing device and run scope options,
      and the workload whose batch size to generate the inputs at.
    model_id: Name of the transformer model to retrieve from the registry.
    is_training: Whether to generate training labels and configurations.

  Returns:
    A tuple of (model, inputs dictionary) ready for execution.
  """
  batch_size = ctx.require_workload().batch_size
  # The requested sequence length of token based models. The input generator
  # caps it, e.g. at the positions of the model, or ignores it, e.g. for vision
  # models, so the workloads of generated benchmarks declare no `seq_len`.
  seq_len = 128 if ctx.run_scope == context_lib.RunScope.PRESUBMIT else 2048

  registry = model_helpers.get_module_registry()
  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)

  def modify_config_hook(config: Any) -> Any:
    _set_config_dtype(config, dtype)
    # Time series models such as TimesFM pad or truncate every series to
    # `context_length` rather than taking the length of the generated inputs.
    # Cap it so that they run on seq_len steps like the other benchmarks.
    context_length = getattr(config, "context_length", None)
    if isinstance(context_length, int) and context_length > seq_len:
      config.context_length = seq_len
      # Patch-based models such as PatchTSMixer size their patch-mixing layers
      # by `num_patches`, which the config derived from the uncapped
      # `context_length` when it was created.
      if all(
          isinstance(getattr(config, name, None), int)
          for name in ("num_patches", "patch_length", "patch_stride")
      ):
        config.num_patches = (
            max(seq_len, config.patch_length) - config.patch_length
        ) // config.patch_stride + 1
    if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
      if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = 2
        # DPT feeds its neck the hidden states of the layers in
        # `backbone_out_indices`, so keep at least one layer per index and read
        # the first ones.
        out_indices = getattr(config, "backbone_out_indices", None)
        if isinstance(out_indices, list):
          config.num_hidden_layers = max(2, len(out_indices))
          config.backbone_out_indices = list(range(len(out_indices)))
      if hasattr(config, "text_config") and config.text_config is not None:
        if hasattr(config.text_config, "num_hidden_layers"):
          config.text_config.num_hidden_layers = 2
      if hasattr(config, "vision_config") and config.vision_config is not None:
        if hasattr(config.vision_config, "num_hidden_layers"):
          config.vision_config.num_hidden_layers = 2
        if hasattr(config.vision_config, "depth"):
          config.vision_config.depth = 2
    return config

  module_spec = registry.get_module_spec(
      "transformers",
      model_id,
      load_weights=False,
      modify_config_hook=modify_config_hook,
  )

  device_str = ctx.device_kind.value
  with torch.device(device_str):
    model = module_spec.module_factory(dtype=dtype)
  _run_blip_forward_on_model_device(model)

  if is_training:
    model.train()
  else:
    model.eval()

  _, inputs = module_spec.sample_inputs_factory(
      (batch_size, seq_len), device_str
  )
  # Pop attention_mask to trigger transformers fully static causal attention
  # mask fallback, avoiding control-flow tracing errors.
  inputs.pop("attention_mask", None)

  inputs = torch.utils._pytree.tree_map(
      lambda v: v.to(dtype=dtype)
      if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
      else v,
      inputs,
  )

  if is_training and "labels" not in inputs:
    num_labels = getattr(model.config, "num_labels", None)
    labels_shape = (
        inputs["input_ids"].shape
        if "input_ids" in inputs
        else (batch_size, seq_len)
    )
    if num_labels is not None and type(model).__name__.endswith(
        "ForSequenceClassification"
    ):
      # The labels below are class indices of shape `(batch_size,)`. Pin the
      # matching `problem_type` so that heads do not take a data-dependent path:
      # with `problem_type` unset, `DebertaForSequenceClassification` and
      # `DebertaV2ForSequenceClassification` select labels with
      # `(labels >= 0).nonzero()`, whose output shape is unknown under
      # `jax.jit`. Regression heads (`num_labels == 1`) are left unchanged.
      config: Any = model.config
      if num_labels > 1 and config.problem_type is None:
        config.problem_type = "single_label_classification"
      inputs["labels"] = torch.randint(
          0,
          num_labels,
          (batch_size,),
          device=device_str,
          dtype=torch.long,
          requires_grad=False,
      )
    elif num_labels is not None and type(model).__name__.endswith(
        "ForTokenClassification"
    ):
      # Token classification models (such as `LayoutLMForTokenClassification` /
      # `microsoft/layoutlm-base-uncased`) compute token-level CrossEntropyLoss
      # over `num_labels` classes (typically 2) with per-token targets of shape
      # `(batch_size, seq_len)`. Without this branch, labels would be sampled
      # in `[0, vocab_size)` (30522) and trigger a CUDA device-side assertion
      # (`t >= 0 && t < n_classes`) in `nll_loss_forward_reduce_cuda_kernel_2d`.
      inputs["labels"] = torch.randint(
          0,
          num_labels,
          labels_shape,
          device=device_str,
          dtype=torch.long,
          requires_grad=False,
      )
    else:
      text_cfg = getattr(model.config, "text_config", None)
      # Check `src_vocab_size` before `vocab_size` for asymmetric seq2seq
      # models such as `FSMTForConditionalGeneration` (`facebook/wmt19-ru-en`),
      # where `config.vocab_size` aliases `tgt_vocab_size` (31640) while the
      # source embedding table only has `src_vocab_size` (31232) entries.
      vocab_size = (
          getattr(model.config, "src_vocab_size", None)
          or getattr(model.config, "vocab_size", None)
          or (getattr(text_cfg, "vocab_size", None) if text_cfg else None)
          or 32000
      )
      inputs["labels"] = torch.randint(
          0,
          vocab_size,
          labels_shape,
          device=device_str,
          dtype=torch.long,
          requires_grad=False,
      )

  return model, inputs


def tapas_load_model(
    ctx: context_lib.Context,
    model_id: str,
    is_training: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a TAPAS QA model with binary `{0, 1}` cell-selection `labels`.

  Unlike standard HuggingFace `*ForQuestionAnswering` heads (which take
  `start_positions`/`end_positions`), `TapasForQuestionAnswering`
  (`google/tapas-*-finetuned-sqa`) computes binary cell-selection loss via
  `torch.distributions.Bernoulli(logits=...).log_prob(labels.float())`, which
  requires `{0, 1}` targets of shape `(batch_size, seq_len)` rather than random
  vocabulary token IDs in `[0, vocab_size)`.

  Args:
    ctx: Benchmark execution context containing device and run scope options.
    model_id: Name of the transformer model to retrieve from the registry.
    is_training: Whether to generate training labels and configurations.

  Returns:
    A tuple of `(model, inputs)` ready for execution.
  """
  model, inputs = default_load_model(ctx, model_id, is_training)
  if is_training and "labels" in inputs:
    inputs["labels"] = torch.randint(
        0,
        2,
        inputs["labels"].shape,
        device=inputs["labels"].device,
        dtype=torch.long,
        requires_grad=False,
    )
  return model, inputs
