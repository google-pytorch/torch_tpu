<!-- LINT.IfChange -->

# GKE TPUv7 example

### Set environment variables for later

All of the steps in this tutorial rely on the following variables, set them
accordingly.

```bash
export CLUSTER=<YOUR_CLUSTER_NAME>
export ZONE_NAME=<YOUR_ZONE_NAME>
export PROJECT=<YOUR_PROJECT_ID>
export ARTIFACT_REGISTRY_URL=<YOUR_ARTIFACT_REGISTRY_URL>
export DOCKER_IMAGE=torch-tpu-local-$USER
export TPU_TYPE=tpu7x-8
```

If using a fuse bucket, also setup:

```bash
export BUCKET=<YOUR_BUCKET_NAME>
```

### Cluster Credentials Setup

Get cluster credentials to interact with the GKE TPU cluster using `kubectl`:

```bash
gcloud container clusters get-credentials ${CLUSTER} --zone=${ZONE_NAME} --project=${PROJECT}
```

### GCloud and docker setup

Configure docker credentials so we can push to artifact registry for GKE to
consume.

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker ${ZONE_NAME}-docker.pkg.dev
sudo usermod -aG docker $USER # relaunch the terminal and activate venv
docker run hello-world # Test Docker
```

Follow
[this tutorial](https://docs.cloud.google.com/artifact-registry/docs/docker/store-docker-container-images)
to initialize your artifact registry to store this new container image for use
in GKE (skip if one already exists).

### Build TorchTPU docker with Tensor Parallel Example

```bash
cd $TORCH_TPU_SRC/docker
./build_image_multistage.sh
docker tag torch-tpu:latest ${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}
docker push ${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}
```

### Submitting Workloads via Kubectl

Workloads are submitted to GKE and managed via Kueue queue admission using
standard Kubernetes `Job` or `JobSet` manifests.

Reference manifests are located at `third_party/py/torch_tpu/examples/gke/`.

#### Kueue Cohort Setup

Before submitting workloads to Kueue queues, ensure Kueue ClusterQueues and
LocalQueues are applied:

```bash
kubectl apply -f third_party/py/torch_tpu/examples/gke/torchtpu_kueue_cohort_setup.yaml
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

#### Single-Host Workload Execution (4-chip / 8-device slice)

Submit a single-host Kubernetes Job:

```bash
kubectl apply -f third_party/py/torch_tpu/examples/gke/torchtpu_singlehost_slice_job.yaml
```

Check workload and pod status:

```bash
kubectl get job -n default
kubectl get pods -n default
```

#### Multi-Host Workload Execution (64-chip / 128-device slice)

Submit a multi-host JobSet:

```bash
kubectl apply -f third_party/py/torch_tpu/examples/gke/torchtpu_multihost_slice_jobset.yaml
```

Check JobSet status:

```bash
kubectl get jobset -n default
kubectl get pods -n default
```

<!-- LINT.ThenChange($INTERNAL_DOCS/gke.md) -->
