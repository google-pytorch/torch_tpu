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

# A script to run the tensor parallel worker on multiple hosts using torchrun.

echo "TPU_VISIBLE_CHIPS: $TPU_VISIBLE_CHIPS"
echo "HOSTNAME: $HOSTNAME"

export TORCH_TPU_TOPOLOGY=4,4,8,2
export WORLD_SIZE=256

# Detect and export TPU configuration dynamically.
echo "Running multihost_wrapper to detect topology..."
eval $(python3 -m torch_tpu._internal.distributed.launchers.multihost_wrapper)
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES
export TORCH_TPU_XPROF_SESSION_ID
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT

echo "Configuration detected:"
echo "  NNODES: $NNODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  TOPOLOGY: $TORCH_TPU_TOPOLOGY"

echo "Launching torch.distributed.run..."
python3 -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=8 \
    ./tp_worker.py
echo "torch.distributed.run finished with exit code $?"
