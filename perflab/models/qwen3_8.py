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

"""Qwen3.8 (torchtitan) single-device benchmark definitions.

Kept separate from `qwen.py` and only built internally, since it is the only
single-process benchmark that depends on torchtitan. The multi-rank Qwen3.8
SimpleFSDP training benchmark lives in `perflab/distributed_models/qwen3_8.py`.
"""

import torch
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab.model_utils import torchtitan_qwen3_5


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def qwen3_8_27b_4_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for Qwen3.8 (27B arch, 4 layers) forward.

  Single-device reference for the distributed SimpleFSDP training benchmark
  (`perflab/distributed_models/qwen3_8.py`): same torchtitan model (3
  GatedDeltaNet + 1 attention layer, full 248K vocab) and per-rank shape (one
  packed 2048-token sequence), forward only.
  """
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    seq_len = 128
    num_layers = 2
  else:
    seq_len = 2048
    num_layers = 4

  device_str = ctx.device_kind.value
  config = torchtitan_qwen3_5.text_config("27B", num_layers, seq_len)
  torch.manual_seed(12345)
  model = torchtitan_qwen3_5.build_model(
      config, device_str, torch_device_ops.get_torch_dtype(ctx.dtype)
  )
  model.eval()
  tokens = torch.randint(
      0,
      config.vocab_size,
      (seq_len,),
      dtype=torch.long,
      generator=torch.Generator().manual_seed(1234),
  ).to(device_str)
  return model, (tokens,), {}
