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

"""Attention layer benchmarks.

The sdpa benchmarks run an attention layer around
`nn.functional.scaled_dot_product_attention` over `SdpaConfig`, with each
config's SDPA backend: flash attention on TPUs, and cuDNN and memory-efficient
attention on GPUs. The backend is part of the microbenchmark name, so the TPU
and GPU benchmark names are distinct. The GPU benchmarks do not run in
presubmit scope, as the FakeTensorMode CPU smoke test cannot build them.
"""

from torch.nn import attention
from perflab import target as target_lib
from perflab.layer_utils import builders
from perflab.layer_utils import configs
from perflab.layer_utils import registration


def _sdpa_args(config: configs.SdpaConfig) -> builders.LayerArgs:
  return builders.LayerArgs(
      model_name="nn.f.scaled_dot_product_attention",
      batch_size=config.batch_size,
      custom_kwargs={
          "embed_dim": config.embed_dim,
          "q_seq_len": config.q_seq_len,
          "q_num_heads": config.q_num_heads,
          "kv_num_heads": config.kv_num_heads,
          "qk_head_dim": config.qk_head_dim,
          "v_head_dim": config.v_head_dim,
          "is_causal": config.is_causal,
          "enable_gqa": config.enable_gqa,
          "backend": config.backend,
      },
  )


def _backend_supports_config(config: configs.SdpaConfig) -> bool:
  """Returns whether the SDPA backend of `config` supports its head counts."""
  # Memory-efficient attention needs as many key/value heads as query heads.
  return (
      config.backend != attention.SDPBackend.EFFICIENT_ATTENTION
      or config.q_num_heads == config.kv_num_heads
  )


registration.add_layer_sweep(
    "sdpa",
    tuple(
        configs.SdpaConfig.configs_with_backends(
            attention.SDPBackend.FLASH_ATTENTION
        )
    ),
    _sdpa_args,
    device_kinds=frozenset({target_lib.DeviceKind.TPU}),
)

registration.add_layer_sweep(
    "sdpa",
    tuple(
        config
        for config in configs.SdpaConfig.configs_with_backends(
            attention.SDPBackend.CUDNN_ATTENTION,
            attention.SDPBackend.EFFICIENT_ATTENTION,
        )
        if _backend_supports_config(config)
    ),
    _sdpa_args,
    device_kinds=frozenset({target_lib.DeviceKind.CUDA}),
    presubmit_skip_reason=(
        "The CPU has no cuDNN or memory-efficient attention kernel, so the"
        " FakeTensorMode CPU smoke test cannot build these layers."
    ),
)
