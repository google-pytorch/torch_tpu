# 在 GKE 上开始使用 TorchTPU { #get-started-with-torchtpu-on-gke }

本教程分为三个部分，介绍如何在 Google Kubernetes Engine（GKE）TPU 集群上运行 PyTorch 工作负载：

1.  **运行简单的单设备示例**（`examples/simple_pytorch/simple_pytorch.py`），作为 Kubernetes `Job` 运行。
2.  **运行多设备分布式示例**（`examples/tensor_parallel/manual/tp_worker.py`），分别在单台 4 芯片主机上（`Job`）和跨多主机的 64 芯片切片上（`JobSet`）运行。
3.  **运行来自 [`pytorch/examples`](https://github.com/pytorch/examples) 的上游 PyTorch 示例**（例如 MNIST）。

本教程假定您已经有一个带 TPU 节点池的 GKE Standard 集群；如果您需要创建集群，请参阅[在 GKE Standard 中部署 TPU 工作负载](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)。

## 准备工作 { #before-you-begin }

### 前提条件 { #prerequisites }

*   一个带有 `tpu7x` 节点池的 GKE 集群。本教程使用 4 芯片（`2x2x1`）节点池。
*   [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk)。

### 设置环境变量 { #set-environment-variables }

本教程中的所有步骤都依赖以下变量，请相应地设置这些变量。`LOCATION` 是集群所在的可用区或区域。

```bash
export PROJECT=<YOUR_PROJECT_ID>
export CLUSTER=<YOUR_CLUSTER_NAME>
export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>
export NODEPOOL_NAME=<YOUR_4CHIP_NODEPOOL_NAME>
export DOCKER_IMAGE=us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
```

### 进行身份验证并获取集群凭据 { #authenticate-and-get-cluster-credentials }

```bash
gcloud auth login
gcloud auth application-default login
gcloud components install kubectl gke-gcloud-auth-plugin
gcloud container clusters get-credentials ${CLUSTER} --location=${LOCATION} --project=${PROJECT}
```

如需了解详情，请参阅[安装 kubectl 并配置集群访问权限](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)。

## 使用 kubectl 提交工作负载 { #submit-workloads-with-kubectl }

工作负载以标准 Kubernetes `Job` 或 `JobSet` 清单的形式提交。Python 示例脚本已预装在公共容器映像的 `/workspace/examples/` 中，而参考 Kubernetes `.yaml` 清单（`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`、`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` 和 `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml`）位于 TorchTPU 代码库的 `examples/` 目录中。请在本地 `torch_tpu` 代码库检出的根目录中运行下面的 `sed` 和 `kubectl
apply` 命令。

## 运行单设备工作负载（4 芯片主机的一个 TensorCore） { #run-a-single-device-workload-one-tensorcore-of-a-4-chip-host }

清单 `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` 会预留一台配备 4 个 TPU v7x 芯片的主机。每个 v7x 芯片有两个 TensorCore，而 TorchTPU 将每个 TensorCore 作为一个设备公开，因此该主机有 8 个 TPU 设备，其切片拓扑为 `2x2x1`：

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ...
    cloud.google.com/gke-tpu-topology: 2x2x1
```

该示例在单个 TPU 设备（一个 TensorCore）上运行单个进程，主机上的其他 7 个设备保持空闲。这表示为 `1x1x1x1` TorchTPU 拓扑，其中第四个数字是作业使用的每个芯片的 TensorCore 数量：

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "1,1,1,1"
```

容器运行映像中 `/workspace/examples/` 下附带的 `simple_pytorch.py`：

```yaml
  command:
    - /bin/bash
    - -c
    - |
      set -ex
      export PYTHONUNBUFFERED=1
      export TPU_LAUNCH_BLOCKING=1
      export ABSL_LOGGING_MIN_LOG_LEVEL=0
      echo "=== Starting Single-Device 1-Chip, 1-Device TPU v7x Workload ==="
      cd /workspace/examples/simple_pytorch/
      python3 ./simple_pytorch.py
      echo "=== Single-Host single-device Workload Completed Successfully ==="
```

以下是完整的程序：

```python
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
```

将映像和节点池填入清单，然后应用该清单：

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
```

检查作业和 Pod 状态，然后读取日志：

```bash
kubectl get job,pods -n default
kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
```

作业会运行至完成，并输出上面显示的四行内容。

## 在 GKE 上运行分布式工作负载 { #run-distributed-workloads-on-gke }

### 单台主机，多个芯片（一台主机上 4 个芯片/8 个设备） { #single-host-multiple-chips-4-chips-8-devices-on-one-host }

在单设备示例的基础上，下一步是使用一台主机上的所有 TPU 设备。要提交单主机 Kubernetes `Job`，请先使用您的 Docker 映像和节点池名称更新清单，然后使用 `kubectl` 启动作业。

清单 `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` 适用于配备 4 个 v7x TPU（8 个 TensorCore）的单台主机，因此切片拓扑为 `2x2x1`：

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
```

TorchTPU 会将其解读为 `2x2x1x2` 拓扑，因为它会为每个 TPU TensorCore 启动一个进程：

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "2,2,1,2"
```

它会运行 Docker 映像中的手动张量并行示例：

```yaml
  command:
    - /bin/bash
    - -c
    - |
      set -ex
      export PYTHONUNBUFFERED=1
      export TPU_LAUNCH_BLOCKING=1
      export ABSL_LOGGING_MIN_LOG_LEVEL=0
      echo "=== Starting Single-Host 4-Chip (8-Device) TPU v7x Workload ==="
      cd /workspace/examples/tensor_parallel/manual/
      bash ./launch_singlehost.sh
      echo "=== Single-Host 4-Chip Workload Completed Successfully ==="
```

`examples/tensor_parallel/manual/launch_singlehost.sh` 会检测 TPU 拓扑并将其导出，然后使用 `torch.distributed.run`（`torchrun`）为每个 TPU 设备启动一个进程，运行 `examples/tensor_parallel/manual/tp_worker.py`：

```bash
# Detect and export TPU configuration dynamically.
eval "$(python3 -m torch.tpu.distributed)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

python3 -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    --nproc_per_node=8 ./tp_worker.py
```

worker 脚本和启动脚本位于 `examples/tensor_parallel/manual/` 中。

将映像和节点池填入清单，然后应用该清单：

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
```

检查作业和 Pod 状态：

```bash
kubectl get job,pods -n default
```

### 多台主机（16 台主机上的 64 个芯片/128 个设备） { #multiple-hosts-64-chips-128-devices-across-16-hosts }

要扩展到多台主机，同一示例会以 Kubernetes `JobSet` 的形式运行：每台主机一个 Pod，`4x4x4` 切片共 16 台主机，`examples/tensor_parallel/manual/launch_multihost.sh` 会根据每台主机在切片中的位置推导出其 rank。使用您的 Docker 映像和 64 芯片节点池名称更新清单，然后使用 `kubectl` 应用该清单：

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
```

检查 `JobSet` 和 Pod 状态：

```bash
kubectl get jobset,pods -n default
```

## 运行上游 PyTorch 示例 { #run-an-upstream-pytorch-example }

由于 TorchTPU 是原生 PyTorch 后端（`device="tpu"`、`torch.accelerator`、分布式 `backend="tpu_dist"`），因此您可以从上游 [`pytorch/examples`](https://github.com/pytorch/examples) 代码库拉取标准脚本，并直接在 GKE 上运行。

### 示例：在 GKE 上运行上游 MNIST { #example-running-upstream-mnist-on-gke }

上游 [`pytorch/examples/mnist`](https://github.com/pytorch/examples/tree/main/mnist) 训练脚本（`mnist/main.py`）使用 PyTorch 与设备无关的 `torch.accelerator` API：

```python
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
```

如果容器中安装了 `torch_tpu`，`torch.accelerator.current_accelerator()` 会自动选择 `"tpu"` 设备，因此 `mnist/main.py` **无需更改任何代码**。

要从 `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` 开始，在 GKE 上的单个 TPU 设备上运行 `mnist/main.py`，请执行以下操作：

1.  **更新 `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` 中的容器 `command:` 块**，以安装 `git`、克隆 [`https://github.com/pytorch/examples.git`](https://github.com/pytorch/examples)、将 `DataLoader` 的 `num_workers` 保持为默认值（`0`），使其不会派生尝试重新获取 `libtpu` 设备锁文件的 worker 子进程，然后运行 `mnist/main.py`：

    ```yaml
      command:
        - /bin/bash
        - -c
        - |
          set -ex
          export PYTHONUNBUFFERED=1
          apt-get update && apt-get install -y git
          git clone --depth 1 https://github.com/pytorch/examples.git /tmp/pytorch_examples
          cd /tmp/pytorch_examples/mnist
          sed -i "/'num_workers': 1,/d; /'persistent_workers': True,/d" main.py
          python3 main.py --epochs=1 --dry-run
    ```

2.  **删除该作业之前的所有运行**（因为 Kubernetes `Job` 规范一经创建便不可更改），根据需要填充占位符，应用更新后的清单，然后流式传输 MNIST 训练日志：

    ```bash
    kubectl delete -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml --ignore-not-found
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
    ```

### 为其他上游示例选择清单模板 { #choosing-a-manifest-template-for-other-upstream-examples }

要运行其他单设备或分布式上游示例，请选择与目标拓扑匹配的参考清单模板：

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 工作负载类型 | 清单路径 | TPU 拓扑（`gke-tpu-topology`） | `TORCH_TPU_TOPOLOGY` | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **单设备**（4 芯片主机上的 1 个 TensorCore） | `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` | `"1,1,1,1"` | 在 1 个 TPU 设备上运行单个 Python 进程。 |
| **单主机分布式**（4 个芯片/8 个核心） | `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` | `"2,2,1,2"` | 通过 `torchrun` 在单台 4 芯片主机上运行 8 个分布式 rank（`WORLD_SIZE=8`）。 |
| **多主机分布式**（64 个芯片/128 个核心） | `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` | `"4,4,4,2"` | Kubernetes `JobSet`，在 64 芯片切片上协调 16 个 worker Pod。 |
<!-- mdformat on -->

要以其他 TPU 切片拓扑为目标，请更新清单中的 `cloud.google.com/gke-tpu-slice-topology`、`cloud.google.com/gke-tpu-topology` 和 `TORCH_TPU_TOPOLOGY`（请参阅 [TPU 虚拟机系统架构](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)和 [TPU 机器类型和拓扑](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)）。

### 移植上游 PyTorch 示例以在 GKE 上运行 { #porting-an-upstream-pytorch-example-to-run-on-gke }

由于 TorchTPU 是原生 PyTorch 后端，因此让上游 PyTorch 示例或测试在 GKE 上运行只需极少的工作：

*   **非分布式示例无需更改代码**：为标准 PyTorch 设备编写的非分布式示例（通过 `torch.accelerator` 或 `torch.device`）无需更改代码即可与 TorchTPU 搭配使用。
*   **未弃用的示例应该可以正常运行**：仍在积极维护、未弃用的上游 PyTorch 示例和现代 API 可以开箱即用（如需了解详情，请参阅[支持的功能集](./feature_set.md)）。
*   **分布式示例可能需要少量修改**：一些上游分布式示例针对其他设备进行了过度调优（例如特定于 GPU 或 NCCL 的后端、标志或分片假设），可能需要少量修改才能在 TPU 上运行（请参阅上文的分布式部分和 [FSDP 最佳实践](./case_studies_fsdp.md)）。
