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
    
    We can run the same earlier very basic example through GKE.
    This example is stored in github at examples/simple_pytorch/simple_pytorch.py
    
    Recent TPUs from v7 onwards are only supported on GKE.
    Here we assume that you have a TPU cluster and node pool already created.

   ## One-time setup that we can reuse for other GKE examples later.
    
    ### Check out the TorchTPU repo and set environment variables

    Check out the TorchTPU repo to run this tutorial. You need access to the
    kubernetes manifests and optionally the source code from which to build the docker image.
    

    All of the steps in this tutorial rely on the following variables, set them
    accordingly.

    ```bash
    export CLUSTER=<YOUR_CLUSTER_NAME>
    export ZONE_NAME=<YOUR_ZONE_NAME>
    export NODEPOOL_NAME=<YOUR_NODEPOOL_NAME>
    export PROJECT=<YOUR_PROJECT_ID>
    export TORCH_TPU_SRC=/path/to/torch_tpu
    export TPU_TYPE=tpu7x-8
    # Artifact registry to store the docker image.
    # If you have your own, use it instead, but these default to the nightly build.
    export ARTIFACT_REGISTRY_URL=us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container
    export DOCKER_IMAGE=torch-tpu:nightly-latest
    ```

    If using a fuse bucket, also setup:

    ```bash
    export BUCKET=<YOUR_BUCKET_NAME>
    ```
    
    You may need to install the gcloud CLI following instructions at:
    https://docs.cloud.google.com/sdk/docs/install-sdk
    as well as setting up a cluster and node pool. 
    

    ### GCloud setup

    Configure docker credentials so we can push to artifact registry for GKE to
    consume.

    ```bash
    gcloud auth login
    gcloud auth application-default login
    ```
    
    ### Cluster Credentials Setup
    At this point you may need to install kubectl and the gke plugin:
    ```bash
    gcloud components install kubectl
    gcloud components install gke-gloud-auth-plugin
    ```

    Follow instructions at
    https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin
    for more information.

    Get cluster credentials to interact with the GKE TPU cluster using `kubectl`:

    ```bash
    gcloud container clusters get-credentials ${CLUSTER} --zone=${ZONE_NAME} --project=${PROJECT}
    ```

    ### GCloud and docker setup

    Configure docker credentials so we can push to artifact registry for GKE to
    consume.

    ```bash
    gcloud auth configure-docker ${ZONE_NAME}-docker.pkg.dev
    sudo usermod -aG docker $USER # relaunch the terminal and activate venv
    newgrp docker
    docker run hello-world # Test Docker
    ```

    Follow
    [this tutorial](https://docs.cloud.google.com/artifact-registry/docs/docker/store-docker-container-images)
    to initialize your artifact registry to store this new container image for use
    in GKE (skip if one already exists).

    ### [Optional] Build TorchTPU docker
    
    If you want to build the docker image yourself because you have changed torchtpu or its examples, follow these instructions.
    Otherwise skip this section and use the pre-built docker image that we defaulted to in the environment variables above.
    
    Check out the repo and build it and upload it as follows:

    ```bash
    export DOCKER_IMAGE=torch-tpu-local-$USER
    cd $TORCH_TPU_SRC/docker
    ./build_image_multistage.sh
    docker tag torch-tpu:latest ${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}
    docker push ${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}
    ```

    ### Submitting Workloads via Kubectl

    Workloads are submitted to GKE and managed via Kueue queue admission using
    standard Kubernetes `Job` or `JobSet` manifests. We generally recommend Kueue for queueing. TPU jobs because it maps well to the multi-slice structure of TPU deployments.

    Reference manifests are located in the TorchTPU examples `third_party/py/torch_tpu/examples/`.
    
    ### Kueue Cohort Setup

    Before submitting workloads to Kueue queues, ensure Kueue ClusterQueues and
    LocalQueues are applied. First we patch the node pool name into the manifest and then apply the kueue manifest:

    ```bash
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_kueue_cohort_setup.yaml
    kubectl apply -f examples/tensor_parallel/torchtpu_kueue_cohort_setup.yaml
    ```
    
    #### Universal Queue `lq` and Automatic Topology Routing

    For seamless compatibility with XPK tools and standard Kubernetes jobs, our
    setup provides a universal default queue named `LocalQueue: lq`.

    When you specify `kueue.x-k8s.io/queue-name: lq` in your workload labels:

    1.  **Universal Queue**: Workloads are admitted through `LocalQueue lq` into
        `ClusterQueue tpu-cluster-queue` within `cohort: tpu-shared-reservation`.
    2.  **Automatic Flavor Routing**: Kueue automatically matches your workload's
        `nodeSelector` topology labels (`cloud.google.com/gke-tpu-topology`) and
        assigns the correct `ResourceFlavor` (`rf-4chip`, `rf-8chip`, `rf-16chip`,
        `rf-32chip`, or `rf-64chip`).
    3.  **Automatic Suspension**: Kueue injects `spec.suspend: true` upon submission
        and removes it (`spec.suspend: false`) once shared cohort quota is reserved.


    ## Single-Host Workload Execution (1-chip / 1-device slice)

    Submit a single-host Kubernetes Job by first updating the manifest with your
    docker image and node pool name and then launching the job using kubectl.
    
    This manifest is interesting in a couple of ways. It applies to a single host of 4 v7 TPUs (or 8 cores) so the topology is described as 2x2x1:
    ```yaml
        cloud.google.com/gke-tpu-slice-topology: 2x2x1
        ...
        cloud.google.com/gke-tpu-topology: 2x2x1
    ```
    
    and where TorchTPU interprets this as a 1x1x1x1 topology because we will launch with a single process for a single TPU core. We will ignore the other TPU devices on the host:
    ```yaml
        - name: TORCH_TPU_TOPOLOGY
          value: "1,1,1,1"
    ```
  

    and it runs the simple_pytorch manual code from the docker image:
    
    ```yaml
      command:
        - /bin/bash
        - -c
        - |
          export PYTHONUNBUFFERED=1
          export TPU_LAUNCH_BLOCKING=1
          export ABSL_LOGGING_MIN_LOG_LEVEL=0
          echo "=== Starting Single-Device 1-Chip, 1-Device TPU v7x Workload ==="
          cd /workspace/examples/simple_pytorch/
          python3 ./simple_pytorch.py
          echo "=== Single-Host single-device Workload Completed Successfully ==="
    ```
    
    This is a very simple single-device pytorch program, with nothing surprising:

    """)
  return


@app.cell(hide_code=True)
def _(mo):
  import torch

  device = torch.device("tpu")  # Change from cuda to tpu
  print(f"TPU Initialization complete. Primary device: {device}")

  # 1. Create a tensor on the CPU
  cpu_tensor = torch.ones((5, 5))

  # 2. Move it to the TPU
  tpu_tensor = cpu_tensor.to(device)  # Using the device object
  # OR
  tpu_tensor_alt = cpu_tensor.to("tpu")  # Using the "tpu" string

  # 3. Create a tensor directly on the TPU
  direct_tpu_tensor = torch.randn((5, 5), device="tpu")

  print(f"cpu_tensor is on: {cpu_tensor.device}")
  print(f"tpu_tensor is on: {tpu_tensor.device}")
  print(f"direct_tpu_tensor is on: {direct_tpu_tensor.device}")


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    and is in `examples/simple_pytorch/simple_pytorch.py`
    
    As kubectl consumes a manifest file off disk, we patch in the environment
    variables here for convenience, and then apply the manifest.

    ```bash
    sed -i "s|<DOCKER_IMAGE>|${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
    kubectl apply -f examples/tensor_parallel/torchtpu_singledevice_slice_job.yaml
    ```

    Check workload and pod status:

    ```bash
    kubectl get job -n default
    kubectl get pods -n default
    ```
    
    and you can query the logs for the pod based on the job name:
    ```bash
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job
    ```
    
    We should be able to see the job and pod running and eventually complete with the expected output logged.

    """)
  return

@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    """)
  return

if __name__ == "__main__":
  app.run()
