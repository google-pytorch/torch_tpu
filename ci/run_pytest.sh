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
# ==============================================================================
# Runs Pytest tests against pre-installed TorchTPU wheels in an isolated venv.
set -exu -o history

ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-cpu}

echo "=== Verifying Python and Wheel Environment ==="
python3 -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python3 -c "import torch_tpu; print('TorchTPU Version:', torch_tpu.__version__)"
python3 -m uv pip freeze

export PY_COLORS=1
export TF_CPP_MIN_LOG_LEVEL=0
export TORCH_TPU_SKIP_SLOW_TESTS=true

mkdir -p test-artifacts

if [[ "${ACCELERATOR_TYPE}" == "tpu" ]]; then
    echo "=== Verifying TPU Environment ==="

    export ALLOW_MULTIPLE_LIBTPU_LOAD=true
    TPU_CORES=${TPU_CORES:-8}

    echo "=== Running TorchTPU TPU Pytest Suite ==="
    python3 -m pytest -n="${TPU_CORES}" --tb=short \
        --dist=loadfile \
        --junitxml=test-artifacts/junit_tpu.xml \
        --maxfail=20 \
        tests third_party/py/torch_tpu/examples
else
    echo "=== Running TorchTPU CPU Pytest Suite ==="
    MAX_PROCESSES=${MAX_PROCESSES:-$(nproc)}

    python3 -m pytest -n "${MAX_PROCESSES}" --tb=short \
        --junitxml=test-artifacts/junit_cpu.xml \
        --maxfail=20 \
        -m "not require_tpu" \
        tests third_party/py/torch_tpu/examples
fi
