# Primeiros passos com o TorchTPU no GKE { #get-started-with-torchtpu-on-gke }

Este tutorial mostra como executar cargas de trabalho PyTorch em um cluster de
TPU do Google Kubernetes Engine (GKE) em três partes:

1.  **Execute um exemplo simples em um único dispositivo**
    (`examples/simple_pytorch/simple_pytorch.py`) como um `Job` do Kubernetes.
2.  **Execute um exemplo distribuído em vários dispositivos**
    (`examples/tensor_parallel/manual/tp_worker.py`) em um único host com 4
    chips (`Job`) e em uma fração (slice) multi-host com 64 chips (`JobSet`).
3.  **Execute um exemplo upstream do PyTorch** de
    [`pytorch/examples`](https://github.com/pytorch/examples) (como o MNIST).

Partimos do princípio de que você já tem um cluster GKE Standard com um pool de
nós de TPU. Se precisar criar um, consulte
[Implantar cargas de trabalho de TPU no GKE Standard](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus).

## Antes de começar { #before-you-begin }

### Pré-requisitos { #prerequisites }

*   Um cluster GKE com um pool de nós `tpu7x`. Este tutorial usa um pool de
    nós com 4 chips (`2x2x1`).
*   A [CLI gcloud](https://docs.cloud.google.com/sdk/docs/install-sdk).

### Definir variáveis de ambiente { #set-environment-variables }

Todas as etapas deste tutorial dependem das variáveis a seguir. Defina-as de
acordo com o seu ambiente. `LOCATION` é a zona ou região do seu cluster.

```bash
export PROJECT=<YOUR_PROJECT_ID>
export CLUSTER=<YOUR_CLUSTER_NAME>
export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>
export NODEPOOL_NAME=<YOUR_4CHIP_NODEPOOL_NAME>
export DOCKER_IMAGE=us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
```

### Autenticar e obter as credenciais do cluster { #authenticate-and-get-cluster-credentials }

```bash
gcloud auth login
gcloud auth application-default login
gcloud components install kubectl gke-gcloud-auth-plugin
gcloud container clusters get-credentials ${CLUSTER} --location=${LOCATION} --project=${PROJECT}
```

Consulte
[Instalar o kubectl e configurar o acesso ao cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)
para mais detalhes.

## Enviar cargas de trabalho com o kubectl { #submit-workloads-with-kubectl }

As cargas de trabalho são enviadas como manifestos padrão `Job` ou `JobSet` do
Kubernetes. Os scripts Python de exemplo já vêm pré-instalados na imagem de
contêiner pública em `/workspace/examples/`, enquanto os manifestos `.yaml` de
referência do Kubernetes
(`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`,
`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` e
`examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml`) ficam no
diretório `examples/` do repositório do TorchTPU. Execute os comandos `sed` e
`kubectl
apply` abaixo a partir da raiz de um checkout local do repositório `torch_tpu`.

## Executar uma carga de trabalho em um único dispositivo (um TensorCore de um host com 4 chips) { #run-a-single-device-workload-one-tensorcore-of-a-4-chip-host }

O manifesto `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`
reserva um host com 4 chips TPU v7x. Cada chip v7x tem dois TensorCores, e o
TorchTPU expõe cada TensorCore como um dispositivo, então o host tem 8
dispositivos TPU e a topologia da fração é `2x2x1`:

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ...
    cloud.google.com/gke-tpu-topology: 2x2x1
```

O exemplo executa um único processo em um único dispositivo TPU (um
TensorCore) e deixa ociosos os outros 7 dispositivos do host. Isso é expresso
como uma topologia `1x1x1x1` do TorchTPU, em que o quarto número é a
quantidade de TensorCores por chip que o job usa:

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "1,1,1,1"
```

O contêiner executa `simple_pytorch.py`, que vem na imagem em
`/workspace/examples/`:

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

Este é o programa completo:

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

Insira a imagem e o pool de nós no manifesto e aplique-o:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
```

Verifique o status do job e dos pods e leia os logs:

```bash
kubectl get job,pods -n default
kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
```

O job é executado até o fim e imprime as quatro linhas mostradas acima.

## Executar cargas de trabalho distribuídas no GKE { #run-distributed-workloads-on-gke }

### Um único host, vários chips (4 chips / 8 dispositivos em um host) { #single-host-multiple-chips-4-chips-8-devices-on-one-host }

O primeiro passo além do exemplo com um único dispositivo é usar todos os
dispositivos TPU de um host. Envie um `Job` do Kubernetes de host único:
primeiro atualize o manifesto com sua imagem Docker e o nome do pool de nós e,
em seguida, inicie o job usando o `kubectl`.

O manifesto `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` se
aplica a um único host com 4 TPUs v7x (8 TensorCores), então a topologia da
fração é `2x2x1`:

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
```

O TorchTPU interpreta isso como uma topologia `2x2x1x2` porque inicia um
processo por TensorCore da TPU:

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "2,2,1,2"
```

Ele executa o exemplo de paralelismo de tensor manual a partir da imagem
Docker:

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

`examples/tensor_parallel/manual/launch_singlehost.sh` detecta a topologia da
TPU, exporta essa informação e inicia um processo por dispositivo TPU com
`torch.distributed.run` (`torchrun`), executando
`examples/tensor_parallel/manual/tp_worker.py`:

```bash
# Detect and export TPU configuration dynamically.
eval "$(python3 -m torch.tpu.distributed)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

python3 -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    --nproc_per_node=8 ./tp_worker.py
```

Os scripts de worker e de inicialização ficam em
`examples/tensor_parallel/manual/`.

Insira a imagem e o pool de nós no manifesto e aplique-o:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
```

Verifique o status do job e dos pods:

```bash
kubectl get job,pods -n default
```

### Vários hosts (64 chips / 128 dispositivos em 16 hosts) { #multiple-hosts-64-chips-128-devices-across-16-hosts }

Para escalar além de um host, o mesmo exemplo é executado como um `JobSet` do
Kubernetes: um pod por host, 16 hosts para uma fração `4x4x4`, com
`examples/tensor_parallel/manual/launch_multihost.sh` derivando o rank de cada
host a partir da posição dele na fração. Atualize o manifesto com sua imagem
Docker e o nome do pool de nós de 64 chips e aplique-o com o `kubectl`:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
```

Verifique o status do `JobSet` e dos pods:

```bash
kubectl get jobset,pods -n default
```

## Executar um exemplo upstream do PyTorch { #run-an-upstream-pytorch-example }

Como o TorchTPU é um backend nativo do PyTorch (`device="tpu"`,
`torch.accelerator`, `backend="tpu_dist"` distribuído), você pode baixar e
executar scripts padrão do repositório upstream
[`pytorch/examples`](https://github.com/pytorch/examples) diretamente no GKE.

### Exemplo: como executar o MNIST upstream no GKE { #example-running-upstream-mnist-on-gke }

O script de treinamento upstream
[`pytorch/examples/mnist`](https://github.com/pytorch/examples/tree/main/mnist)
(`mnist/main.py`) usa a API `torch.accelerator` do PyTorch, que é independente
de dispositivo:

```python
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
```

Quando `torch_tpu` está instalado no contêiner,
`torch.accelerator.current_accelerator()` seleciona automaticamente o
dispositivo `"tpu"`, sem exigir **nenhuma mudança no código** de
`mnist/main.py`.

Para executar `mnist/main.py` em um único dispositivo TPU no GKE a partir de
`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`:

1.  **Atualize o bloco `command:` do contêiner** em
    `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` para
    instalar o `git`, clonar
    [`https://github.com/pytorch/examples.git`](https://github.com/pytorch/examples),
    manter o `num_workers` do `DataLoader` no valor padrão (`0`), para que ele
    não crie um subprocesso worker que tente adquirir novamente o arquivo de
    bloqueio do dispositivo da `libtpu`, e executar `mnist/main.py`:

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

2.  **Exclua qualquer execução anterior do job** (porque as especificações de
    `Job` do Kubernetes são imutáveis depois de criadas), substitua os
    placeholders se necessário, aplique o manifesto atualizado e acompanhe os
    logs do treinamento do MNIST:

    ```bash
    kubectl delete -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml --ignore-not-found
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
    ```

### Como escolher um modelo de manifesto para outros exemplos upstream { #choosing-a-manifest-template-for-other-upstream-examples }

Para executar outros exemplos upstream em um único dispositivo ou
distribuídos, escolha o modelo de manifesto de referência que corresponde à
topologia de destino:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Tipo de carga de trabalho | Caminho do manifesto | Topologia da TPU (`gke-tpu-topology`) | `TORCH_TPU_TOPOLOGY` | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| **Dispositivo único** (1 TensorCore em um host com 4 chips) | `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` | `"1,1,1,1"` | Executa um único processo Python em 1 dispositivo TPU. |
| **Distribuído em um único host** (4 chips / 8 núcleos) | `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` | `"2,2,1,2"` | Executa 8 ranks distribuídos (`WORLD_SIZE=8`) em um único host com 4 chips via `torchrun`. |
| **Distribuído em vários hosts** (64 chips / 128 núcleos) | `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` | `"4,4,4,2"` | `JobSet` do Kubernetes que coordena 16 pods worker em uma fração de 64 chips. |
<!-- mdformat on -->

Para usar outras topologias de fração de TPU, atualize
`cloud.google.com/gke-tpu-slice-topology`, `cloud.google.com/gke-tpu-topology`
e `TORCH_TPU_TOPOLOGY` no manifesto (consulte
[Arquitetura do sistema de VMs de TPU](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
e
[Tipos de máquina e topologias de TPU](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)).

### Como portar um exemplo upstream do PyTorch para execução no GKE { #porting-an-upstream-pytorch-example-to-run-on-gke }

Como o TorchTPU é um backend nativo do PyTorch, colocar um exemplo ou teste
upstream do PyTorch para funcionar no GKE exige um processo mínimo:

*   **Nenhuma mudança de código é necessária em exemplos não distribuídos**:
    exemplos não distribuídos escritos para dispositivos padrão do PyTorch
    (via `torch.accelerator` ou `torch.device`) funcionam com o TorchTPU sem
    mudanças no código.
*   **Exemplos não descontinuados devem funcionar**: exemplos upstream do
    PyTorch mantidos ativamente e não descontinuados, assim como APIs
    modernas, funcionam de imediato (consulte
    [Recursos compatíveis](./feature_set.md) para mais detalhes).
*   **Exemplos distribuídos podem precisar de pequenas modificações**: alguns
    exemplos distribuídos upstream são ajustados demais para outros dispositivos
    (como backends, flags ou premissas de fragmentação (sharding) específicos de
    GPU ou NCCL) e podem exigir pequenas modificações para funcionar na TPU
    (consulte as seções sobre execução distribuída acima e
    [Práticas recomendadas de FSDP](./case_studies_fsdp.md)).
