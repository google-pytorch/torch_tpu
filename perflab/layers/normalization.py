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

"""Normalization layer benchmarks: `nn.LayerNorm` over `LAYER_NORM_CONFIGS`."""

from perflab.layer_utils import builders
from perflab.layer_utils import configs
from perflab.layer_utils import registration


def _layer_norm_args(config: configs.LayerNormConfig) -> builders.LayerArgs:
  # The input shape comes from `custom_kwargs`.
  return builders.LayerArgs(
      model_name="nn.LayerNorm",
      batch_size=None,
      custom_kwargs={
          "normalized_shape": config.normalized_shape,
          "shape": config.shape,
      },
  )


registration.add_layer_sweep(
    "layernorm",
    configs.LAYER_NORM_CONFIGS,
    _layer_norm_args,
)
