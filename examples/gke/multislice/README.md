# GKE Multi-Slice TorchTitan HSDP Example

This example runs multi-slice **LLaMA 3 70B** (`llama3_70b`) training with
**TorchTitan** and **Hybrid Sharded Data Parallel (HSDP)** on GKE TPU using a
single self-contained Kubernetes manifest (`multislice_jobset.yaml`).

## Step-by-Step Guide

### 1. Configure `multislice_jobset.yaml`

Set `<YOUR_IMAGE_URI>` in `multislice_jobset.yaml` to a container image with
`torch`, `torch_tpu`, and `torchtitan` installed.

### 2. Connect to Your GKE Cluster and Launch the JobSet

```bash
gcloud container clusters get-credentials <YOUR_CLUSTER_NAME> \
  --region <YOUR_REGION> \
  --project <YOUR_PROJECT>

kubectl apply -f examples/gke/multislice/multislice_jobset.yaml
```

### 3. Monitor Status and Logs

```bash
# Check JobSet status
kubectl get jobset torchtitan-multislice

# Watch pods across all slices
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=torchtitan-multislice -w

# Stream training logs from the coordinator pod (slice 0, VM 0)
kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=torchtitan-multislice,jobset.sigs.k8s.io/job-index=0,batch.kubernetes.io/job-completion-index=0
```

### 4. Clean Up

```bash
kubectl delete -f examples/gke/multislice/multislice_jobset.yaml
```
