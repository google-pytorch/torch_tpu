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
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness import torch_device_ops
from tests import module_registry


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

  registry = module_registry.ModuleRegistry()

  def modify_config_hook(config):
    if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
      config.num_hidden_layers = 2
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

  if is_training:
    vocab_size = getattr(model.config, "vocab_size", 128256)
    inputs["labels"] = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=device_str,
        dtype=torch.long,
        requires_grad=False,
    )

  return model, inputs
