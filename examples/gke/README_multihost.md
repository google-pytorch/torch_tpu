# Running Multi-Host Tensor Parallel on GKE

This guide walks you through authenticating with GCloud/Docker, creating a
Dockerfile, building and pushing the image, and launching the multi-host
**Tensor Parallel** example on a Google Kubernetes Engine (GKE) TPU cluster.

We will focus on the **manual tensor parallel** example
(`examples/distributed/tensor_parallel/manual`), which implements a
Megatron-LM-style column-parallel and row-parallel feed-forward network.

--------------------------------------------------------------------------------

## 1. GCloud & Docker Setup

To interact with the GKE cluster and push container images to Google Artifact
Registry, you need to authenticate your local environment.

### Authenticate GCloud CLI

Run the following commands to log in and set up your active Google Cloud
project:

```bash
# Log in to your Google Account
gcloud auth login

# Log in to set up Application Default Credentials (ADC) for local client APIs
gcloud auth application-default login

# Configure Docker to authenticate with Google Artifact Registry (e.g. us-central1)
gcloud auth configure-docker us-central1-docker.pkg.dev

# (Optional) Ensure the correct project is billed for your operations
export PROJECT="tpu-pytorch"
gcloud auth application-default set-quota-project $PROJECT
```

### Environment Variables

Define the following environment variables for use in subsequent steps:

```bash
export PROJECT="tpu-pytorch"
export ZONE_NAME="us-central1"
export CLUSTER="torchtpu-v7x-8-spot"  # Or your specific GKE cluster name
export ARTIFACT_REGISTRY_URL="us-central1-docker.pkg.dev/tpu-pytorch/torchtpu-development-images"
export DOCKER_IMAGE="torch-tpu-tp-$USER:latest"
export TPU_TYPE="tpu7x-16"  # Or tpu7x-32, etc. (Must be multi-host, e.g. > 8 chips)
```

--------------------------------------------------------------------------------

## 2. Build and Push the Image

You must build the Docker image using the optimized multi-stage Dockerfile
provided in the repository. Because of our BuildKit configurations, the build
cache is preserved when you only make changes to example scripts, resulting in
near-instant rebuilds.

Make sure you run the build command from the **repository root** (e.g.
`third_party/py/torch_tpu`) so the build context has access to all required
files:

```bash
# Run the build script from the repository root
# This will build and tag a local image named `torch-tpu:latest`
cd third_party/py/torch_tpu
./docker/build_image_multistage.sh

# Tag the built image with your remote registry URL
docker tag torch-tpu:latest ${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}

# Push the image to the remote registry
docker push ${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}
```

--------------------------------------------------------------------------------

## 3. Launching the Workload

You can launch the multi-host workload using either **XPK** (highly recommended)
or **Kubectl** (direct Kubernetes manifests).

### Option A: Launching with XPK

XPK is the easiest way to launch TPU workloads as it manages scheduling,
queueing, and environment variables automatically.

```bash
xpk workload create \
  --workload=tp-multihost-job-$USER \
  --cluster=${CLUSTER} \
  --zone=${ZONE_NAME} \
  --project=${PROJECT} \
  --tpu-type=${TPU_TYPE} \
  --num-slices=1 \
  --docker-image=${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE} \
  --command="cd /workspace/examples/distributed/tensor_parallel/manual && ./launch_multihost.sh"
```

--------------------------------------------------------------------------------

### Option B: Launching with Kubectl

If you want to bypass XPK and interact directly with the Kubernetes API, you
must create a headless Service (for pod-to-pod DNS resolution) and an Indexed
Job.

Create a manifest file named `tp-job.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tp-headless-svc
spec:
  clusterIP: None
  publishNotReadyAddresses: true # REQUIRED for TPU mesh initialization
  selector:
    job-name: tp-multihost-job
---
apiVersion: batch/v1
kind: Job
metadata:
  name: tp-multihost-job
spec:
  backoffLimit: 0
  completions: 2       # Number of TPU VMs (e.g., 2 hosts for v7x-16)
  parallelism: 2       # Must run all hosts concurrently for ICI mesh initialization
  completionMode: Indexed
  template:
    spec:
      subdomain: tp-headless-svc
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v7x-slice
        cloud.google.com/gke-tpu-topology: 2x2x4  # Matches a 16-chip v7x slice (2 hosts)
      containers:
      - name: tpu-job
        image: us-central1-docker.pkg.dev/tpu-pytorch/torchtpu-development-images/torch-tpu-tp-amainz:latest # Replace amainz with your LDAP
        ports:
        - containerPort: 8471
        - containerPort: 29400 # Rendezvous Port
        - containerPort: 29500 # Master Port
        env:
        - name: TOTAL_COMPLETIONS
          value: "2"  # Must match completions count above
        securityContext:
          privileged: true  # Required for direct TPU hardware access
        command:
        - /bin/bash
        - -c
        - |
          # 1. Dynamically construct TPU_WORKER_HOSTNAMES based on indexed job dns
          # Under completionMode: Indexed, GKE maps pod index i to <job-name>-<i>.<subdomain>
          HOSTNAMES=""
          for i in $(seq 0 $((TOTAL_COMPLETIONS - 1))); do
            HOSTNAMES="${HOSTNAMES}tp-multihost-job-${i}.tp-headless-svc,"
          done
          export TPU_WORKER_HOSTNAMES="${HOSTNAMES%,}"

          # 2. RUN the launcher
          cd /workspace/examples/distributed/tensor_parallel/manual
          ./launch_multihost.sh
        resources:
          requests:
            google.com/tpu: 8   # Requests all 8 TPU chips on each VM
          limits:
            google.com/tpu: 8
```

Apply the manifest:

```bash
kubectl apply -f tp-job.yaml
```

To monitor logs:

```bash
# Follow logs for Worker Node 0 (the master rendezvous node)
kubectl logs -f tp-multihost-job-0-pod -c tpu-job
```

--------------------------------------------------------------------------------

## 5. Under the Hood: The Launcher script

The `launch_multihost.sh` executes the following sequence:

1.  Runs the `multihost_wrapper` to parse `TPU_WORKER_HOSTNAMES` and
    `JOB_COMPLETION_INDEX`.
2.  Computes the distributed environment variables (`MASTER_ADDR`,
    `MASTER_PORT`, `NNODES`, `NODE_RANK`, `TORCH_TPU_SLICEBUILDER_ADDRESSES`).
3.  Executes `torch.distributed.run` (torchrun) targeting `tp_worker.py` with 8
    processes (one per local TPU chip):

```bash
python3 -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=8 \
    ./tp_worker.py
```
