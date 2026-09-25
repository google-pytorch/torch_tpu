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

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import torch_device_ops


def default_load_model(
    ctx: context_lib.Context, model_id: str, is_training: bool
):
  """Loads the model and generates sample inputs for benchmarking.

  Args:
    ctx: Benchmark execution context containing device and run scope options.
    model_id: Name of the transformer model to retrieve from the registry.
    is_training: Whether to generate training labels and configurations.

  Returns:
    A tuple of (model, inputs dictionary) ready for execution.
  """
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
  else:
    batch_size = 1
    seq_len = 2048

  registry = model_helpers.get_module_registry()

  def modify_config_hook(config):
    if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
      if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = 2
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

  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device_str = ctx.device_kind.value
  with torch.device(device_str):
    model = module_spec.module_factory().to(dtype=dtype)

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
