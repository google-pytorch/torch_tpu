#!/bin/bash
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

# A script to run the hf_llama3 worker using standard torchrun.

# Ensure TPU isolation on GKE.
if [ -z "${TPU_VISIBLE_DEVICES}" ] && [ ! -z "${TPU_VISIBLE_CHIPS}" ]; then
  export TPU_VISIBLE_DEVICES="${TPU_VISIBLE_CHIPS}"
fi

# Detect and export TPU configuration dynamically.
eval $(python3 -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE

export MASTER_PORT=$(python3 -c 'import portpicker; print(portpicker.pick_unused_port())')
python3 -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=$WORLD_SIZE ./hf_llama3_worker.py
