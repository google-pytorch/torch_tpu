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

"""Configs for meta-llama3 example."""

import json
import llama_models.llama3.model as m
from examples import paths

llama3_8b_args = json.loads(
    '{"dim": 4096, "ffn_dim_multiplier": 1.3, "multiple_of": 1024, "n_heads":'
    ' 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-05, "rope_theta":'
    ' 500000.0, "use_scaled_rope": true, "vocab_size": 128256}'
)

llama3_70b_args = json.loads(
    '{"dim": 8192, "ffn_dim_multiplier": 1.3, "multiple_of": 4096, "n_heads":'
    ' 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "rope_theta":'
    ' 500000.0, "use_scaled_rope": true, "vocab_size": 128256}'
)

transformer_config = {
    # 8B model may crash due to OOM.
    '8B': m.ModelArgs(max_seq_len=2048, max_batch_size=1, **llama3_8b_args),
    # 70B model may crash due to OOM.
    '70B': m.ModelArgs(max_seq_len=2048, max_batch_size=1, **llama3_70b_args),
}

checkpoint_dir = {
    '8B': (
        f'{paths.XM_HOME}weights/meta-llama/llama-models/checkpoints/Llama-3.1-8b-resharded-tp8/Llama-3.1-8b-resharded-tp8/'
    ),
    '70B': (
        f'{paths.XM_HOME}weights/meta-llama/llama-models/checkpoints/Llama3.1-70B/'
    ),
}
