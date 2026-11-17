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

"""Linear layer benchmarks: `nn.Linear` over `configs.LINEAR_CONFIGS`."""

from perflab.layer_utils import builders
from perflab.layer_utils import configs
from perflab.layer_utils import registration


def _to_layer_args(config: configs.LinearConfig) -> builders.LayerArgs:
  return builders.LayerArgs(
      model_name="nn.Linear",
      batch_size=config.batch_size,
      sequence_length=config.seq_len,
      custom_kwargs={
          "in_features": config.in_features,
          "out_features": config.out_features,
      },
  )


registration.add_layer_sweep(
    "linear",
    configs.LINEAR_CONFIGS,
    _to_layer_args,
)
