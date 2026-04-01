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

# A script to run the tensor parallel worker on a single host.

# Detect and export TPU configuration dynamically for a single host.
echo "Running singlehost_wrapper to detect topology..."
eval "$(python3 -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)"
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "Configuration detected:"
echo "  TOPOLOGY: ${TORCH_TPU_TOPOLOGY}"

echo "Launching torch.distributed.run..."
python3 -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=8 \
    ./tp_worker.py
echo "torch.distributed.run finished with exit code $?"
