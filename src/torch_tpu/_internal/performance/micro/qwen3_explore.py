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

"""Construct qwen3 layer by layer and compare TPU/GPU performance.

///////////////////////////////////////////////////////////////
NOTE: This file/model is meant to DEBUG and TRIAGE remaining
      issues in the torch_tpu implementation. It will contain
      some experimental things and lots of TODOs to bypass
      existing problems.
//////////////////////////////////////////////////////////////
"""

import sys

from absl import app
from absl import flags
import torch
from torch import nn
from torch_tpu._internal.performance.micro import utils

configs = {
    "0.6B": {
        "vocab_size": 151_936,  # Vocabulary size
        "context_length": (
            40_960
        ),  # Context length that was used to train the model
        "emb_dim": 1024,  # Embedding dimension
        "n_heads": 16,  # Number of attention heads
        "n_layers": 28,  # Number of layers
        "hidden_dim": 3072,  # Size of the intermediate dimension in FeedForward
        "head_dim": 128,  # Size of the heads in GQA
        "qk_norm": True,  # Whether to normalize queries and values in GQA
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
        # TODO: There are still issues with type casts.
        #       See TODO's below. Setting the whole model to float32
        #       works but is only a short term workaround.
        # "dtype": torch.bfloat16,  # Lower-precision dtype,reduce memory usage
        "dtype": torch.float32,  # Lower-precision dtype to reduce memory usage
        "use_cache": False,
    },
    "32B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,
        "n_heads": 64,  # 60% larger than above
        "n_layers": 64,  # 60% larger than above
        "hidden_dim": 25600,  # 47% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,  # TODO: bfloat16,
        "use_cache": False,
    },
}


_MODEL = flags.DEFINE_string(
    "torch_tpu_internal_model",
    "32B",
    "Model flavor (one of: 0.6B, 32B)",
)
_ITERS = flags.DEFINE_integer(
    "torch_tpu_internal_iterations", 100, "# of iterations to run."
)


class FeedForward(nn.Module):
  """Traditional feed forward layer with 2 FC's, a silu, and a final FC."""

  def __init__(self, cfg):
    super().__init__()
    self.fc1 = nn.Linear(
        cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
    )
    self.fc2 = nn.Linear(
        cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
    )
    self.fc3 = nn.Linear(
        cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
    )

  def forward(self, x):
    x_fc1 = self.fc1(x)
    x_fc2 = self.fc2(x)
    x = nn.functional.silu(x_fc1) * x_fc2
    return self.fc3(x)


def main(argv):
  del argv  # Unused
  torch.manual_seed(123)

  model_flavor = _MODEL.value
  iterations = _ITERS.value

  print(f"Qwen3 Explorer, flavor: '{model_flavor}'", file=sys.stderr)
  if model_flavor not in configs.keys():
    raise ValueError("Unknown model flavor: " + model_flavor)
  cfg = configs[model_flavor]

  tensor_in = torch.randn(cfg["emb_dim"])
  model = FeedForward(cfg)
  utils.run_model_cpu_acc(
      cfg,
      model,
      tensor_in,
      iterations=iterations,
      warmup=1,
      desc="Simple FeedForward Layer",
  )


if __name__ == "__main__":
  app.run(main)
