# GKE에서 TorchTPU 시작하기 { #get-started-with-torchtpu-on-gke }

이 튜토리얼에서는 Google Kubernetes Engine(GKE) TPU 클러스터에서 PyTorch
워크로드를 실행하는 방법을 세 부분으로 나누어 안내합니다.

1.  **간단한 단일 디바이스 예제**
    (`examples/simple_pytorch/simple_pytorch.py`)를 Kubernetes `Job`으로
    실행합니다.
2.  **멀티 디바이스 분산 예제**
    (`examples/tensor_parallel/manual/tp_worker.py`)를 단일 4칩 호스트(`Job`)와
    멀티 호스트 64칩 슬라이스(`JobSet`)에서 실행합니다.
3.  [`pytorch/examples`](https://github.com/pytorch/examples)의 **업스트림
    PyTorch 예제**(예: MNIST)를 실행합니다.

이 튜토리얼에서는 TPU 노드 풀이 있는 GKE Standard 클러스터가 이미 있다고
가정합니다. 클러스터를 만들어야 하면
[GKE Standard에 TPU 워크로드 배포](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)를
참고하세요.

## 시작하기 전에 { #before-you-begin }

### 기본 요건 { #prerequisites }

*   `tpu7x` 노드 풀이 있는 GKE 클러스터. 이 튜토리얼에서는 4칩(`2x2x1`) 노드
    풀을 사용합니다.
*   [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk)

### 환경 변수 설정 { #set-environment-variables }

이 튜토리얼의 모든 단계는 다음 변수를 사용합니다. 환경에 맞게 설정하세요.
`LOCATION`은 클러스터의 영역 또는 리전입니다.

```bash
export PROJECT=<YOUR_PROJECT_ID>
export CLUSTER=<YOUR_CLUSTER_NAME>
export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>
export NODEPOOL_NAME=<YOUR_4CHIP_NODEPOOL_NAME>
export DOCKER_IMAGE=us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
```

### 인증 및 클러스터 사용자 인증 정보 가져오기 { #authenticate-and-get-cluster-credentials }

```bash
gcloud auth login
gcloud auth application-default login
gcloud components install kubectl gke-gcloud-auth-plugin
gcloud container clusters get-credentials ${CLUSTER} --location=${LOCATION} --project=${PROJECT}
```

자세한 내용은
[kubectl 설치 및 클러스터 액세스 구성](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)을
참고하세요.

## kubectl로 워크로드 제출 { #submit-workloads-with-kubectl }

워크로드는 표준 Kubernetes `Job` 또는 `JobSet` 매니페스트로 제출합니다. Python
예제 스크립트는 공개 컨테이너 이미지의 `/workspace/examples/`에 미리 설치되어
있습니다. 참조용 Kubernetes `.yaml` 매니페스트
(`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`,
`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml`,
`examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml`)는 TorchTPU
저장소의 `examples/` 디렉터리에 있습니다. 아래의 `sed` 및 `kubectl
apply` 명령어는 로컬 `torch_tpu` 저장소 체크아웃의 루트에서 실행하세요.

## 단일 디바이스 워크로드 실행(4칩 호스트의 TensorCore 1개) { #run-a-single-device-workload-one-tensorcore-of-a-4-chip-host }

매니페스트 `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`은
TPU v7x 칩 4개가 있는 호스트 1개를 예약합니다. 각 v7x 칩에는 TensorCore가 2개
있고 TorchTPU는 각 TensorCore를 디바이스 1개로 노출하므로, 이 호스트에는 TPU 디바이스가
8개 있으며 슬라이스 토폴로지는 `2x2x1`입니다.

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ...
    cloud.google.com/gke-tpu-topology: 2x2x1
```

이 예제는 단일 TPU 디바이스(TensorCore 1개)에서 단일 프로세스를 실행하고 호스트의
나머지 디바이스 7개는 유휴 상태로 둡니다. 이는 `1x1x1x1` TorchTPU 토폴로지로
표현하며, 네 번째 숫자는 작업이 사용하는 칩당 TensorCore 수입니다.

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "1,1,1,1"
```

컨테이너는 이미지의 `/workspace/examples/`에 포함된 `simple_pytorch.py`를
실행합니다.

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

전체 프로그램은 다음과 같습니다.

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

매니페스트에 이미지와 노드 풀 이름을 채운 후 적용하세요.

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
```

작업과 포드 상태를 확인한 후 로그를 읽으세요.

```bash
kubectl get job,pods -n default
kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
```

작업이 완료될 때까지 실행되고 위에 표시된 네 줄을 출력합니다.

## GKE에서 분산 워크로드 실행 { #run-distributed-workloads-on-gke }

### 단일 호스트, 여러 칩(호스트 1개에 칩 4개 / 디바이스 8개) { #single-host-multiple-chips-4-chips-8-devices-on-one-host }

단일 디바이스 예제에서 한 단계 나아가 호스트 1개의 모든 TPU 디바이스를 사용합니다. 먼저
매니페스트에 Docker 이미지와 노드 풀 이름을 반영한 후 `kubectl`로 작업을 시작하여
단일 호스트 Kubernetes `Job`을 제출하세요.

매니페스트 `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml`은 v7x
TPU 4개(TensorCore 8개)로 구성된 단일 호스트에 적용되므로 슬라이스 토폴로지는
`2x2x1`입니다.

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
```

TorchTPU는 TPU TensorCore마다 프로세스를 1개씩 시작하므로 이를 `2x2x1x2`
토폴로지로 해석합니다.

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "2,2,1,2"
```

이 매니페스트는 Docker 이미지에 포함된 수동 텐서 병렬 예제를 실행합니다.

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

`examples/tensor_parallel/manual/launch_singlehost.sh`는 TPU 토폴로지를 감지하여
내보내고, `torch.distributed.run`(`torchrun`)으로 TPU 디바이스마다 프로세스를 1개씩
시작하여 `examples/tensor_parallel/manual/tp_worker.py`를 실행합니다.

```bash
# Detect and export TPU configuration dynamically.
eval "$(python3 -m torch.tpu.distributed)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

python3 -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    --nproc_per_node=8 ./tp_worker.py
```

워커 스크립트와 실행 스크립트는 `examples/tensor_parallel/manual/`에 있습니다.

매니페스트에 이미지와 노드 풀 이름을 채운 후 적용하세요.

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
```

작업과 포드 상태를 확인하세요.

```bash
kubectl get job,pods -n default
```

### 여러 호스트(호스트 16개에 칩 64개 / 디바이스 128개) { #multiple-hosts-64-chips-128-devices-across-16-hosts }

호스트 1개를 넘어 확장하려면 같은 예제를 Kubernetes `JobSet`으로 실행합니다.
호스트마다 포드가 1개씩 있고 `4x4x4` 슬라이스에는 호스트가 16개 있으며,
`examples/tensor_parallel/manual/launch_multihost.sh`가 슬라이스 내 위치를
기준으로 각 호스트의 랭크를 도출합니다. 매니페스트에 Docker 이미지와 64칩
노드 풀 이름을 반영한 후 `kubectl`로 적용하세요.

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
```

`JobSet`과 포드 상태를 확인하세요.

```bash
kubectl get jobset,pods -n default
```

## 업스트림 PyTorch 예제 실행 { #run-an-upstream-pytorch-example }

TorchTPU는 네이티브 PyTorch 백엔드(`device="tpu"`, `torch.accelerator`, 분산
`backend="tpu_dist"`)이므로 업스트림
[`pytorch/examples`](https://github.com/pytorch/examples) 저장소의 표준
스크립트를 가져와 GKE에서 바로 실행할 수 있습니다.

### 예: GKE에서 업스트림 MNIST 실행 { #example-running-upstream-mnist-on-gke }

업스트림
[`pytorch/examples/mnist`](https://github.com/pytorch/examples/tree/main/mnist)
학습 스크립트(`mnist/main.py`)는 PyTorch의 디바이스 독립적인 `torch.accelerator`
API를 사용합니다.

```python
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
```

컨테이너에 `torch_tpu`가 설치되어 있으면
`torch.accelerator.current_accelerator()`가 자동으로 `"tpu"` 디바이스를 선택하므로
`mnist/main.py`의 **코드를 전혀 변경할 필요가 없습니다**.

`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`을 기반으로 GKE의
단일 TPU 디바이스에서 `mnist/main.py`를 실행하려면 다음 단계를 따르세요.

1.  `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`의 **컨테이너
    `command:` 블록을 수정**하여 `git`을 설치하고,
    [`https://github.com/pytorch/examples.git`](https://github.com/pytorch/examples)를
    클론하고, `DataLoader` `num_workers`를 기본값(`0`)으로 유지하여 `libtpu` 디바이스
    잠금 파일을 다시 획득하려는 워커 하위 프로세스가 포크되지 않게 한 후,
    `mnist/main.py`를 실행합니다.

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

2.  Kubernetes `Job` 사양은 생성 후 변경할 수 없으므로 **이전에 실행한 작업을
    삭제**하고, 필요하면 자리표시자를 채우고, 수정된 매니페스트를 적용한 후, MNIST
    학습 로그를 스트리밍합니다.

    ```bash
    kubectl delete -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml --ignore-not-found
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
    ```

### 다른 업스트림 예제용 매니페스트 템플릿 선택 { #choosing-a-manifest-template-for-other-upstream-examples }

다른 단일 디바이스 또는 분산 업스트림 예제를 실행하려면 대상 토폴로지에 맞는 참조
매니페스트 템플릿을 선택하세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 워크로드 유형 | 매니페스트 경로 | TPU 토폴로지(`gke-tpu-topology`) | `TORCH_TPU_TOPOLOGY` | 설명 |
| :--- | :--- | :--- | :--- | :--- |
| **단일 디바이스**(4칩 호스트의 TensorCore 1개) | `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` | `"1,1,1,1"` | TPU 디바이스 1개에서 단일 Python 프로세스를 실행합니다. |
| **단일 호스트 분산**(칩 4개 / 코어 8개) | `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` | `"2,2,1,2"` | `torchrun`을 통해 단일 4칩 호스트에서 분산 랭크 8개(`WORLD_SIZE=8`)를 실행합니다. |
| **멀티 호스트 분산**(칩 64개 / 코어 128개) | `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` | `"4,4,4,2"` | 64칩 슬라이스 전체에서 워커 포드 16개를 조정하는 Kubernetes `JobSet`입니다. |
<!-- mdformat on -->

다른 TPU 슬라이스 토폴로지를 대상으로 하려면 매니페스트의
`cloud.google.com/gke-tpu-slice-topology`, `cloud.google.com/gke-tpu-topology`,
`TORCH_TPU_TOPOLOGY`를 수정하세요
([TPU VM 시스템 아키텍처](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
및
[TPU 머신 유형 및 토폴로지](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)
참고).

### 업스트림 PyTorch 예제를 GKE에서 실행하도록 포팅 { #porting-an-upstream-pytorch-example-to-run-on-gke }

TorchTPU는 네이티브 PyTorch 백엔드이므로 업스트림 PyTorch 예제나 테스트를 GKE에서
실행하는 데 필요한 작업이 매우 적습니다.

*   **분산이 아닌 예제는 코드 변경 불필요**: 표준 PyTorch 디바이스용으로
    (`torch.accelerator` 또는 `torch.device`를 통해) 작성된 분산이 아닌 예제는
    코드를 변경하지 않고 TorchTPU에서 작동합니다.
*   **지원 중단되지 않은 예제는 작동 예상**: 활발히 유지 관리되고 지원 중단되지
    않은 업스트림 PyTorch 예제와 최신 API는 바로 작동합니다(자세한 내용은
    [지원 기능 세트](./feature_set.md) 참고).
*   **분산 예제는 약간의 수정이 필요할 수 있음**: 일부 업스트림 분산 예제는 다른
    디바이스(예: GPU 또는 NCCL 전용 백엔드, 플래그, 샤딩 가정)에 과도하게 맞춰져 있어
    TPU에서 실행하려면 약간의 수정이 필요할 수 있습니다(위의 분산 섹션과
    [FSDP 권장사항](./case_studies_fsdp.md) 참고).
