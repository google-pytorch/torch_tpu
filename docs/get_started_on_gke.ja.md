# GKE で TorchTPU を使ってみる { #get-started-with-torchtpu-on-gke }

このチュートリアルでは、Google Kubernetes Engine（GKE）の TPU クラスタで PyTorch ワークロードを実行する手順を、3 つのパートに分けて説明します。

1.  **シンプルな単一デバイスのサンプル**（`examples/simple_pytorch/simple_pytorch.py`）を Kubernetes の `Job` として実行します。
2.  **マルチデバイスの分散サンプル**（`examples/tensor_parallel/manual/tp_worker.py`）を、単一の 4 チップホスト（`Job`）と、マルチホストの 64 チップ スライス（`JobSet`）で実行します。
3.  [`pytorch/examples`](https://github.com/pytorch/examples) の**アップストリームの PyTorch サンプル**（MNIST など）を実行します。

TPU ノードプールを含む GKE Standard クラスタがすでにあることを前提としています。作成する必要がある場合は、[GKE Standard に TPU ワークロードをデプロイする](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)をご覧ください。

## 始める前に { #before-you-begin }

### 前提条件 { #prerequisites }

*   `tpu7x` ノードプールを含む GKE クラスタ。このチュートリアルでは 4 チップ（`2x2x1`）のノードプールを使います。
*   [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk)。

### 環境変数を設定する { #set-environment-variables }

このチュートリアルのすべての手順で、次の変数を使います。環境に合わせて設定してください。`LOCATION` はクラスタのゾーンまたはリージョンです。

```bash
export PROJECT=<YOUR_PROJECT_ID>
export CLUSTER=<YOUR_CLUSTER_NAME>
export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>
export NODEPOOL_NAME=<YOUR_4CHIP_NODEPOOL_NAME>
export DOCKER_IMAGE=us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
```

### 認証してクラスタの認証情報を取得する { #authenticate-and-get-cluster-credentials }

```bash
gcloud auth login
gcloud auth application-default login
gcloud components install kubectl gke-gcloud-auth-plugin
gcloud container clusters get-credentials ${CLUSTER} --location=${LOCATION} --project=${PROJECT}
```

詳しくは、[kubectl をインストールしてクラスタへのアクセスを構成する](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)をご覧ください。

## kubectl でワークロードを送信する { #submit-workloads-with-kubectl }

ワークロードは、標準の Kubernetes `Job` または `JobSet` マニフェストとして送信します。Python のサンプル スクリプトは、公開コンテナ イメージ内の `/workspace/examples/` にあらかじめインストールされています。一方、リファレンスの Kubernetes `.yaml` マニフェスト（`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`、`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml`、`examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml`）は、TorchTPU リポジトリの `examples/` ディレクトリにあります。以下の `sed` と `kubectl
apply`
コマンドは、ローカルにチェックアウトした `torch_tpu` リポジトリのルートから実行してください。

## 単一デバイスのワークロードを実行する（4 チップホストの 1 つの TensorCore） { #run-a-single-device-workload-one-tensorcore-of-a-4-chip-host }

マニフェスト `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` は、4 つの TPU v7x チップを持つホストを 1 台確保します。各 v7x チップには 2 つの TensorCore があり、TorchTPU は各 TensorCore を 1 つのデバイスとして公開するため、このホストには 8 つの TPU デバイスがあり、スライス トポロジは `2x2x1` です。

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ...
    cloud.google.com/gke-tpu-topology: 2x2x1
```

このサンプルは、単一の TPU デバイス（1 つの TensorCore）で単一のプロセスを実行し、ホスト上の残り 7 つのデバイスは使いません。これは `1x1x1x1` の TorchTPU トポロジとして表現されます。4 番目の数字は、ジョブが使うチップあたりの TensorCore 数です。

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "1,1,1,1"
```

コンテナは、イメージ内の `/workspace/examples/` に含まれている `simple_pytorch.py` を実行します。

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

プログラム全体は次のとおりです。

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

マニフェストにイメージとノードプールを書き込み、適用します。

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
```

ジョブと Pod のステータスを確認し、ログを読みます。

```bash
kubectl get job,pods -n default
kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
```

ジョブは完了まで実行され、上に示した 4 行を出力します。

## GKE で分散ワークロードを実行する { #run-distributed-workloads-on-gke }

### 単一ホスト、複数チップ（1 ホストで 4 チップ / 8 デバイス） { #single-host-multiple-chips-4-chips-8-devices-on-one-host }

単一デバイスのサンプルから一歩進んで、1 台のホスト上のすべての TPU デバイスを使います。まずマニフェストを Docker イメージとノードプール名で更新し、`kubectl` でジョブを起動して、単一ホストの Kubernetes `Job` を送信します。

マニフェスト `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` は 4 つの v7x TPU（8 つの TensorCore）を持つ単一ホストを対象とするため、スライス トポロジは `2x2x1` です。

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
```

TorchTPU は TPU の TensorCore ごとに 1 つのプロセスを起動するため、これを `2x2x1x2` トポロジとして解釈します。

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "2,2,1,2"
```

このマニフェストは、Docker イメージに含まれる手動テンソル並列のサンプルを実行します。

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

`examples/tensor_parallel/manual/launch_singlehost.sh` は TPU トポロジを検出してエクスポートし、`torch.distributed.run`（`torchrun`）で TPU デバイスごとに 1 つのプロセスを起動して `examples/tensor_parallel/manual/tp_worker.py` を実行します。

```bash
# Detect and export TPU configuration dynamically.
eval "$(python3 -m torch.tpu.distributed)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

python3 -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    --nproc_per_node=8 ./tp_worker.py
```

ワーカー スクリプトと起動スクリプトは `examples/tensor_parallel/manual/` にあります。

マニフェストにイメージとノードプールを書き込み、適用します。

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
```

ジョブと Pod のステータスを確認します。

```bash
kubectl get job,pods -n default
```

### 複数ホスト（16 ホストで 64 チップ / 128 デバイス） { #multiple-hosts-64-chips-128-devices-across-16-hosts }

1 台のホストを超えてスケールするには、同じサンプルを Kubernetes の `JobSet` として実行します。ホストごとに 1 つの Pod を使い、`4x4x4` スライスでは 16 台のホストになります。`examples/tensor_parallel/manual/launch_multihost.sh` が、スライス内での位置から各ホストのランクを導出します。マニフェストを Docker イメージと 64 チップのノードプール名で更新し、`kubectl` で適用します。

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
```

`JobSet` と Pod のステータスを確認します。

```bash
kubectl get jobset,pods -n default
```

## アップストリームの PyTorch サンプルを実行する { #run-an-upstream-pytorch-example }

TorchTPU はネイティブな PyTorch バックエンド（`device="tpu"`、`torch.accelerator`、分散の `backend="tpu_dist"`）であるため、アップストリームの [`pytorch/examples`](https://github.com/pytorch/examples) リポジトリにある標準のスクリプトを取得して、GKE 上で直接実行できます。

### 例: アップストリームの MNIST を GKE で実行する { #example-running-upstream-mnist-on-gke }

アップストリームの [`pytorch/examples/mnist`](https://github.com/pytorch/examples/tree/main/mnist) のトレーニング スクリプト（`mnist/main.py`）は、PyTorch のデバイス非依存の `torch.accelerator` API を使っています。

```python
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
```

コンテナに `torch_tpu` がインストールされている場合、`torch.accelerator.current_accelerator()` が自動的に `"tpu"` デバイスを選択するため、`mnist/main.py` に**コードの変更はまったく必要ありません**。

`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` をもとに、GKE 上の単一の TPU デバイスで `mnist/main.py` を実行するには、次の手順に従います。

1.  `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` の**コンテナの `command:` ブロックを更新**して、`git` をインストールし、[`https://github.com/pytorch/examples.git`](https://github.com/pytorch/examples) をクローンし、`DataLoader` の `num_workers` をデフォルト（`0`）のままにして、`libtpu` デバイスのロックファイルを再取得しようとするワーカーのサブプロセスを fork しないようにしてから、`mnist/main.py` を実行します。

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

2.  **以前に実行したジョブを削除し**（Kubernetes の `Job` の仕様は作成後に変更できないため）、必要に応じてプレースホルダを書き換え、更新したマニフェストを適用して、MNIST のトレーニング ログをストリーミングします。

    ```bash
    kubectl delete -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml --ignore-not-found
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
    ```

### 他のアップストリームのサンプル用にマニフェスト テンプレートを選ぶ { #choosing-a-manifest-template-for-other-upstream-examples }

他の単一デバイスまたは分散のアップストリーム サンプルを実行するには、対象のトポロジに合ったリファレンス マニフェスト テンプレートを選びます。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| ワークロードの種類 | マニフェストのパス | TPU トポロジ（`gke-tpu-topology`） | `TORCH_TPU_TOPOLOGY` | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| **単一デバイス**（4 チップホストの 1 TensorCore） | `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` | `"1,1,1,1"` | 1 つの TPU デバイスで単一の Python プロセスを実行します。 |
| **単一ホストの分散**（4 チップ / 8 コア） | `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` | `"2,2,1,2"` | `torchrun` を使って、単一の 4 チップホストで 8 つの分散ランク（`WORLD_SIZE=8`）を実行します。 |
| **マルチホストの分散**（64 チップ / 128 コア） | `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` | `"4,4,4,2"` | 64 チップのスライス全体で 16 個のワーカー Pod を連携させる Kubernetes の `JobSet` です。 |
<!-- mdformat on -->

他の TPU スライス トポロジを対象にするには、マニフェストの `cloud.google.com/gke-tpu-slice-topology`、`cloud.google.com/gke-tpu-topology`、`TORCH_TPU_TOPOLOGY` を更新します（[TPU VM のシステム アーキテクチャ](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)と [TPU のマシンタイプとトポロジ](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)をご覧ください）。

### アップストリームの PyTorch サンプルを GKE で実行できるように移植する { #porting-an-upstream-pytorch-example-to-run-on-gke }

TorchTPU はネイティブな PyTorch バックエンドであるため、アップストリームの PyTorch のサンプルやテストを GKE で実行するのに必要な作業はごくわずかです。

*   **分散を使わないサンプルではコードの変更は不要**: 標準の PyTorch デバイス向けに（`torch.accelerator` または `torch.device` を使って）書かれた分散を使わないサンプルは、コードを変更しなくても TorchTPU で動作します。
*   **非推奨でないサンプルは動作する想定**: 活発にメンテナンスされている非推奨でないアップストリームの PyTorch サンプルと最新の API は、そのまま動作します（詳しくは[サポートされている機能セット](./feature_set.md)をご覧ください）。
*   **分散のサンプルでは軽微な変更が必要な場合がある**: アップストリームの分散サンプルの中には、他のデバイスに過度に特化したもの（GPU や NCCL 固有のバックエンド、フラグ、シャーディングの前提など）があり、TPU で実行するには軽微な変更が必要になる場合があります（上記の分散のセクションと [FSDP のベスト プラクティス](./case_studies_fsdp.md)をご覧ください）。
