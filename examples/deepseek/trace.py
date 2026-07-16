# Copyright 2025 Google LLC
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

"""Trace aten ops in the DeepSeek model.

This script traces the aten ops in the DeepSeek model. It's useful for
identifying the aten ops that need to be implemented in torch_tpu.

The core algorithm is adapted from work by @gunhyun.
"""

from absl import app
from absl import logging
import torch
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import tracer_utils
from torch_tpu._internal.utils import utils
from examples.deepseek import model

log_utils.log_to_stderr()


def main(argv):
  """Trace the ops in the forward pass of the DeepSeek model."""

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Create a mini config for tracing.
  config = model.ModelArgs(
      max_batch_size=2,
      max_seq_len=128,
      dtype="bf16",
      vocab_size=128,
      dim=32,
      inter_dim=64,
      moe_inter_dim=32,
      # These two lines create a model with one dense layer, one MoE layer.
      n_layers=2,
      n_dense_layers=1,
      n_heads=8,
      # moe
      n_routed_experts=4,
      n_shared_experts=2,
      n_activated_experts=2,
      n_expert_groups=1,
      n_limited_groups=1,
      score_func="softmax",
      route_scale=1.0,
      # mla
      q_lora_rank=0,
      kv_lora_rank=4,
      qk_nope_head_dim=4,  # dim (32) / heads (8)
      qk_rope_head_dim=2,  # Half of nope head
      v_head_dim=2,
  )

  # This setup code based on model.py::main
  torch.set_default_dtype(torch.bfloat16)
  torch.set_default_device("cpu")
  torch.manual_seed(0)
  x = torch.randint(0, config.vocab_size, (2, 128))

  # Trace the ops in the model constructor.
  with utils.OpTracer() as tracer:
    transformer = model.Transformer(config)
    transformer(x)

  logging.info(tracer_utils.pformat_op_tracer(tracer))


if __name__ == "__main__":
  app.run(main)
