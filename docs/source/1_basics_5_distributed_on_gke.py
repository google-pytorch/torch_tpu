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

# pylint: skip-file
import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # GKE TPUv7 example
    
    Recent TPUs from v7 onwards are only supported on GKE.
    This tutorial shows how to run a simple tensor parallel example on TPUv7 through GKE.
    It assumes that you have a TPU cluster and node pool already created.

  

    ## Single-Host Workload Execution (4-chip / 8-device slice)

    Submit a single-host Kubernetes Job by first updating the manifest with your
    docker image and node pool name and then launching the job using kubectl.
    
    This manifest is interesting in a couple of ways. It applies to a single host of 4 v7 TPUs (or 8 cores) so the topology is described as 2x2x1:
    ```yaml
        cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ```
    
    and where TorchTPU interprets this as a 2x2x1x2 topology because it will launch with a process per TPU core:
    ```yaml
        - name: TORCH_TPU_TOPOLOGY
          value: "2,2,1,2"
    ```
    

    and it runs the tensor_parallel manual code from the docker image:
    
    ```yaml
      command:
        - /bin/bash
        - -c
        - |
          set -ex
          export PYTHONUNBUFFERED=1
          export TPU_LAUNCH_BLOCKING=1
          export ABSL_LOGGING_MIN_LOG_LEVEL=0
          eval $(python3 -m torch.tpu.distributed)
          echo "=== Starting Single-Host 4-Chip (8-Device) TPU v7x Workload ==="
          cd /workspace/examples/tensor_parallel/manual/
          bash ./launch_singlehost.sh
          echo "=== Single-Host 4-Chip Workload Completed Successfully ==="
    ```
    
    The worker code is dispatched using torchrun, and depends on the same environment variables that the VM example used:
    
    ```bash
    # Detect and export TPU configuration dynamically.
    eval $(python3 -m torch.tpu.distributed)
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE

    python3 -m torch.distributed.run --nproc_per_node=$WORLD_SIZE ./dtp_worker.py
    ```
    
    
    This code is in the examples folder.
    
    As kubectl consumes a manifest file off disk, we patch in the environment
    variables here for convenience, and then apply the manifest.

    ```bash
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
    kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
    ```

    Check workload and pod status:

    ```bash
    kubectl get job -n default
    kubectl get pods -n default
    ```

    #### Multi-Host Workload Execution (64-chip / 128-device slice)

    Submit a multi-host Kubernetes JobSet by first updating the manifest with your
    docker image and node pool name and then launching the job using kubectl:

    ```bash
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
    kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
    ```

    Check JobSet status:

    ```bash
    kubectl get jobset -n default
    kubectl get pods -n default
    ```

    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    """)
  return


if __name__ == "__main__":
  app.run()
