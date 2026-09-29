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

"""Utility for SDPA tests."""

import dataclasses
import sys
from absl import flags
import torch

asdict = dataclasses.asdict

_RUN_CONFIGS = flags.DEFINE_enum(
    "run_configs",
    "all",
    ["first", "all"],
    "Which test configurations to run. 'first' runs only the first config,"
    " while 'all' runs all configurations.",
)


def generate_configs_for_parameterized(configs):
  """Generates test parameters from a list of configuration dataclasses."""
  # Parse flags if they haven't been parsed yet. This is necessary because
  # parameterized decorators are executed at module import time, before
  # absltest.main() has a chance to parse them.
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)

  for config in configs:
    config_dict = asdict(config)
    name_parts = (
        f"{k}_{str(v).replace('torch.', '')}" if k == "dtype" else f"{k}_{v}"
        for k, v in config_dict.items()
    )
    testcase_name = "_".join(name_parts)
    yield dict(testcase_name=testcase_name, config=config)
    if _RUN_CONFIGS.value == "first":
      # We consider only the first configuration.
      break


@dataclasses.dataclass(frozen=True)
class SdpaConfig:
  """Configuration for SDPA tests."""

  q_seq_len: int
  kv_seq_len: int
  q_num_heads: int
  kv_num_heads: int
  qk_head_dim: int
  v_head_dim: int
  is_causal: bool
  batch_size: int = 1
  enable_gqa: bool = True
  dtype: torch.dtype = torch.bfloat16
  scale: float | None = None
  attn_bias_type: torch.dtype | None = None
  backend: torch.nn.attention.SDPBackend = (
      torch.nn.attention.SDPBackend.FLASH_ATTENTION
  )
  requires_grad: bool = True


SDPA_CONFIGS = (
    # Default config for smoke test.
    SdpaConfig(
        q_seq_len=512,
        kv_seq_len=512,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=True,
        scale=1.1 / 64**0.5,  # Slightly different from default 1.0 / sqrt(64)
    ),
    SdpaConfig(
        q_seq_len=128,
        kv_seq_len=128,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=True,
    ),
    # Cross-Attention.
    SdpaConfig(
        q_seq_len=512,
        kv_seq_len=1024,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=False,
    ),
    # Configs for Llama3 70B attention layers
    SdpaConfig(
        q_seq_len=2048,
        kv_seq_len=2048,
        q_num_heads=64,
        kv_num_heads=64,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
    SdpaConfig(
        batch_size=4,
        q_seq_len=2048,
        kv_seq_len=2048,
        q_num_heads=64,
        kv_num_heads=64,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
    # Configs for Qwen3 480B attention layers
    SdpaConfig(
        q_seq_len=2048,
        kv_seq_len=2048,
        q_num_heads=96,
        kv_num_heads=96,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
    SdpaConfig(
        batch_size=4,
        q_seq_len=2048,
        kv_seq_len=2048,
        q_num_heads=96,
        kv_num_heads=96,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
    # Configs for Gemma3 27B attention layers
    SdpaConfig(
        q_seq_len=2048,
        kv_seq_len=2048,
        q_num_heads=32,
        kv_num_heads=32,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
    SdpaConfig(
        batch_size=4,
        q_seq_len=2048,
        kv_seq_len=2048,
        q_num_heads=32,
        kv_num_heads=32,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
    # Explicit MQA config (kv_num_heads=1)
    SdpaConfig(
        batch_size=2,
        q_seq_len=512,
        kv_seq_len=512,
        q_num_heads=8,
        kv_num_heads=1,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=True,
    ),
    # AFM v7 configs
    SdpaConfig(
        batch_size=2,
        q_seq_len=2048,
        kv_seq_len=2048,
        q_num_heads=16,
        kv_num_heads=2,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
    # Configs for testing attn_bias.
    SdpaConfig(
        q_seq_len=512,
        kv_seq_len=512,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=False,
        attn_bias_type=torch.bool,
    ),
    SdpaConfig(
        q_seq_len=512,
        kv_seq_len=512,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=False,
        attn_bias_type=torch.bfloat16,
    ),
    # Configs with requires_grad=False
    SdpaConfig(
        q_seq_len=512,
        kv_seq_len=512,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=True,
        requires_grad=False,
    ),
    SdpaConfig(
        q_seq_len=128,
        kv_seq_len=128,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=False,
        requires_grad=False,
    ),
    SdpaConfig(
        q_seq_len=123,
        kv_seq_len=123,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=True,
        requires_grad=True,
    ),
    SdpaConfig(
        q_seq_len=1000,
        kv_seq_len=1000,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=False,
        attn_bias_type=torch.bfloat16,
        requires_grad=True,
    ),
    # Long sequence length configs
    SdpaConfig(
        q_seq_len=8192,
        kv_seq_len=8192,
        q_num_heads=8,
        kv_num_heads=2,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
    SdpaConfig(
        q_seq_len=10000,
        kv_seq_len=10000,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=256,
        v_head_dim=256,
        is_causal=False,
        attn_bias_type=torch.bfloat16,
    ),
    # Non-square configs
    SdpaConfig(
        q_seq_len=128,
        kv_seq_len=256,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
    SdpaConfig(
        q_seq_len=256,
        kv_seq_len=128,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=128,
        v_head_dim=128,
        is_causal=True,
    ),
)


# Also test with OVERRIDEABLE backend.
SDPA_CONFIGS = SDPA_CONFIGS + tuple(
    dataclasses.replace(
        config, backend=torch.nn.attention.SDPBackend.OVERRIDEABLE
    )
    for config in SDPA_CONFIGS
    if config.backend is not torch.nn.attention.SDPBackend.OVERRIDEABLE
)

# Also test with float32 dtype.
SDPA_CONFIGS += tuple(
    dataclasses.replace(config, dtype=torch.float32)
    for config in SDPA_CONFIGS
    if config.dtype is not torch.float32
)
