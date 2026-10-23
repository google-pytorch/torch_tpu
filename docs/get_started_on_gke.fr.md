# Premiers pas avec TorchTPU sur GKE { #get-started-with-torchtpu-on-gke }

Ce tutoriel montre comment exécuter des charges de travail PyTorch sur un
cluster TPU Google Kubernetes Engine (GKE), en trois parties :

1.  **Exécuter un exemple simple sur un seul appareil**
    (`examples/simple_pytorch/simple_pytorch.py`) en tant que `Job` Kubernetes.
2.  **Exécuter un exemple distribué sur plusieurs appareils**
    (`examples/tensor_parallel/manual/tp_worker.py`) sur un seul hôte à 4 puces
    (`Job`) et sur une tranche (slice) multi-hôte de 64 puces (`JobSet`).
3.  **Exécuter un exemple PyTorch upstream** issu de
    [`pytorch/examples`](https://github.com/pytorch/examples) (comme MNIST).

Nous supposons que vous disposez déjà d'un cluster GKE Standard avec un pool de
nœuds TPU. Si vous devez en créer un, consultez
[Déployer des charges de travail TPU dans GKE Standard](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus).

## Avant de commencer { #before-you-begin }

### Prérequis { #prerequisites }

*   Un cluster GKE avec un pool de nœuds `tpu7x`. Ce tutoriel utilise un pool
    de nœuds à 4 puces (`2x2x1`).
*   La [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk).

### Définir les variables d'environnement { #set-environment-variables }

Toutes les étapes de ce tutoriel s'appuient sur les variables suivantes.
Définissez-les en fonction de votre environnement. `LOCATION` correspond à la
zone ou à la région de votre cluster.

```bash
export PROJECT=<YOUR_PROJECT_ID>
export CLUSTER=<YOUR_CLUSTER_NAME>
export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>
export NODEPOOL_NAME=<YOUR_4CHIP_NODEPOOL_NAME>
export DOCKER_IMAGE=us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
```

### S'authentifier et obtenir les identifiants du cluster { #authenticate-and-get-cluster-credentials }

```bash
gcloud auth login
gcloud auth application-default login
gcloud components install kubectl gke-gcloud-auth-plugin
gcloud container clusters get-credentials ${CLUSTER} --location=${LOCATION} --project=${PROJECT}
```

Pour en savoir plus, consultez
[Installer kubectl et configurer l'accès au cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl).

## Envoyer des charges de travail avec kubectl { #submit-workloads-with-kubectl }

Les charges de travail sont envoyées sous forme de manifestes Kubernetes `Job`
ou `JobSet` standards. Les scripts Python d'exemple sont déjà préinstallés dans
l'image de conteneur publique, sous `/workspace/examples/`. Les manifestes
Kubernetes `.yaml` de référence
(`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`,
`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` et
`examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml`) se trouvent
dans le répertoire `examples/` du dépôt TorchTPU. Exécutez les commandes `sed`
et `kubectl
apply` ci-dessous depuis la racine d'une copie locale du dépôt `torch_tpu`.

## Exécuter une charge de travail sur un seul appareil (un TensorCore d'un hôte à 4 puces) { #run-a-single-device-workload-one-tensorcore-of-a-4-chip-host }

Le manifeste `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`
réserve un hôte doté de 4 puces TPU v7x. Chaque puce v7x comporte deux
TensorCores, et TorchTPU expose chaque TensorCore comme un appareil : l'hôte
dispose donc de 8 appareils TPU et la topologie de sa tranche est `2x2x1` :

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ...
    cloud.google.com/gke-tpu-topology: 2x2x1
```

L'exemple exécute un seul processus sur un seul appareil TPU (un TensorCore) et
laisse les 7 autres appareils de l'hôte inactifs. Cela s'exprime par une
topologie TorchTPU `1x1x1x1`, où le quatrième nombre est le nombre de
TensorCores par puce utilisés par le job :

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "1,1,1,1"
```

Le conteneur exécute `simple_pytorch.py`, fourni dans l'image sous
`/workspace/examples/` :

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

Voici le programme complet :

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

Insérez l'image et le pool de nœuds dans le manifeste, puis appliquez-le :

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
```

Vérifiez l'état du job et des pods, puis lisez les journaux :

```bash
kubectl get job,pods -n default
kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
```

Le job s'exécute jusqu'au bout et affiche les quatre lignes ci-dessus.

## Exécuter des charges de travail distribuées sur GKE { #run-distributed-workloads-on-gke }

### Un seul hôte, plusieurs puces (4 puces / 8 appareils sur un hôte) { #single-host-multiple-chips-4-chips-8-devices-on-one-host }

L'étape suivante après l'exemple à un seul appareil consiste à utiliser tous
les appareils TPU d'un hôte. Envoyez un `Job` Kubernetes à un seul hôte :
mettez d'abord à jour le manifeste avec votre image Docker et le nom de votre
pool de nœuds, puis lancez le job avec `kubectl`.

Le manifeste `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml`
s'applique à un seul hôte de 4 TPU v7x (8 TensorCores) : la topologie de la
tranche est donc `2x2x1` :

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
```

TorchTPU l'interprète comme une topologie `2x2x1x2`, car il lance un processus
par TensorCore TPU :

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "2,2,1,2"
```

Il exécute l'exemple de parallélisme de tensors manuel à partir de l'image
Docker :

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

`examples/tensor_parallel/manual/launch_singlehost.sh` détecte la topologie
TPU, l'exporte et démarre un processus par appareil TPU avec
`torch.distributed.run` (`torchrun`), qui exécute
`examples/tensor_parallel/manual/tp_worker.py` :

```bash
# Detect and export TPU configuration dynamically.
eval "$(python3 -m torch.tpu.distributed)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

python3 -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    --nproc_per_node=8 ./tp_worker.py
```

Les scripts du worker et de lancement se trouvent dans
`examples/tensor_parallel/manual/`.

Insérez l'image et le pool de nœuds dans le manifeste, puis appliquez-le :

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
```

Vérifiez l'état du job et des pods :

```bash
kubectl get job,pods -n default
```

### Plusieurs hôtes (64 puces / 128 appareils répartis sur 16 hôtes) { #multiple-hosts-64-chips-128-devices-across-16-hosts }

Pour aller au-delà d'un hôte, le même exemple s'exécute en tant que `JobSet`
Kubernetes : un pod par hôte, 16 hôtes pour une tranche `4x4x4`, et
`examples/tensor_parallel/manual/launch_multihost.sh` déduit le rang de chaque
hôte à partir de sa position dans la tranche. Mettez à jour le manifeste avec
votre image Docker et le nom de votre pool de nœuds à 64 puces, puis
appliquez-le avec `kubectl` :

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
```

Vérifiez l'état du `JobSet` et des pods :

```bash
kubectl get jobset,pods -n default
```

## Exécuter un exemple PyTorch upstream { #run-an-upstream-pytorch-example }

Comme TorchTPU est un backend PyTorch natif (`device="tpu"`,
`torch.accelerator`, `backend="tpu_dist"` distribué), vous pouvez récupérer et
exécuter directement sur GKE les scripts standards du dépôt upstream
[`pytorch/examples`](https://github.com/pytorch/examples).

### Exemple : exécuter l'exemple MNIST upstream sur GKE { #example-running-upstream-mnist-on-gke }

Le script d'entraînement upstream
[`pytorch/examples/mnist`](https://github.com/pytorch/examples/tree/main/mnist)
(`mnist/main.py`) utilise l'API `torch.accelerator` de PyTorch, indépendante de
l'appareil :

```python
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
```

Lorsque `torch_tpu` est installé dans le conteneur,
`torch.accelerator.current_accelerator()` sélectionne automatiquement
l'appareil `"tpu"` : **aucune modification de code** n'est nécessaire dans
`mnist/main.py`.

Pour exécuter `mnist/main.py` sur un seul appareil TPU sur GKE à partir de
`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` :

1.  **Mettez à jour le bloc `command:` du conteneur** dans
    `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` pour
    installer `git`, cloner
    [`https://github.com/pytorch/examples.git`](https://github.com/pytorch/examples),
    conserver la valeur par défaut (`0`) de `num_workers` du `DataLoader`, afin
    qu'il ne crée pas de sous-processus worker qui tenterait de réacquérir le
    fichier de verrouillage de l'appareil `libtpu`, et exécuter
    `mnist/main.py` :

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

2.  **Supprimez toute exécution précédente du job** (les spécifications de
    `Job` Kubernetes sont immuables une fois créées), insérez les valeurs des
    espaces réservés si nécessaire, appliquez le manifeste mis à jour et
    affichez en continu les journaux d'entraînement MNIST :

    ```bash
    kubectl delete -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml --ignore-not-found
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
    ```

### Choisir un modèle de manifeste pour d'autres exemples upstream { #choosing-a-manifest-template-for-other-upstream-examples }

Pour exécuter d'autres exemples upstream, sur un seul appareil ou distribués,
choisissez le modèle de manifeste de référence qui correspond à votre topologie
cible :

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Type de charge de travail | Chemin du manifeste | Topologie TPU (`gke-tpu-topology`) | `TORCH_TPU_TOPOLOGY` | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Un seul appareil** (1 TensorCore sur un hôte à 4 puces) | `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` | `"1,1,1,1"` | Exécute un seul processus Python sur 1 appareil TPU. |
| **Distribué sur un seul hôte** (4 puces / 8 cœurs) | `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` | `"2,2,1,2"` | Exécute 8 rangs distribués (`WORLD_SIZE=8`) sur un seul hôte à 4 puces via `torchrun`. |
| **Distribué sur plusieurs hôtes** (64 puces / 128 cœurs) | `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` | `"4,4,4,2"` | `JobSet` Kubernetes qui coordonne 16 pods workers sur une tranche de 64 puces. |
<!-- mdformat on -->

Pour cibler d'autres topologies de tranche TPU, mettez à jour
`cloud.google.com/gke-tpu-slice-topology`, `cloud.google.com/gke-tpu-topology`
et `TORCH_TPU_TOPOLOGY` dans le manifeste (consultez
[Architecture système des VM TPU](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
et
[Types de machines et topologies TPU](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)).

### Porter un exemple PyTorch upstream pour l'exécuter sur GKE { #porting-an-upstream-pytorch-example-to-run-on-gke }

Comme TorchTPU est un backend PyTorch natif, faire fonctionner sur GKE un
exemple ou un test PyTorch upstream demande très peu d'efforts :

*   **Aucune modification de code pour les exemples non distribués** : les
    exemples non distribués écrits pour les appareils PyTorch standards (via
    `torch.accelerator` ou `torch.device`) fonctionnent avec TorchTPU sans
    modification de code.
*   **Les exemples non obsolètes devraient fonctionner** : les exemples PyTorch
    upstream activement maintenus et non obsolètes, ainsi que les API
    modernes, fonctionnent immédiatement (pour en savoir plus, consultez
    [Fonctionnalités prises en charge](./feature_set.md)).
*   **Les exemples distribués peuvent nécessiter de légères modifications** :
    certains exemples distribués upstream sont trop optimisés pour d'autres
    appareils (par exemple, des backends, flags ou hypothèses de sharding
    propres aux GPU ou à NCCL) et peuvent nécessiter de légères modifications
    pour s'exécuter sur TPU (consultez les sections sur le calcul distribué
    ci-dessus et [Bonnes pratiques FSDP](./case_studies_fsdp.md)).
