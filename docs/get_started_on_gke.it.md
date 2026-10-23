# Inizia a utilizzare TorchTPU su GKE { #get-started-with-torchtpu-on-gke }

Questo tutorial mostra come eseguire carichi di lavoro PyTorch su un cluster
TPU di Google Kubernetes Engine (GKE) in tre parti:

1.  **Esegui un semplice esempio su un solo dispositivo**
    (`examples/simple_pytorch/simple_pytorch.py`) come `Job` Kubernetes.
2.  **Esegui un esempio distribuito su più dispositivi**
    (`examples/tensor_parallel/manual/tp_worker.py`) su un singolo host con 4
    chip (`Job`) e su una slice multi-host con 64 chip (`JobSet`).
3.  **Esegui un esempio PyTorch upstream** da
    [`pytorch/examples`](https://github.com/pytorch/examples) (ad esempio
    MNIST).

Presupponiamo che tu abbia già un cluster GKE Standard con un node pool TPU;
consulta
[Eseguire il deployment dei carichi di lavoro TPU in GKE Standard](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)
se devi crearne uno.

## Prima di iniziare { #before-you-begin }

### Prerequisiti { #prerequisites }

*   Un cluster GKE con un node pool `tpu7x`. Questo tutorial usa un node pool
    con 4 chip (`2x2x1`).
*   La [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk).

### Imposta le variabili di ambiente { #set-environment-variables }

Tutti i passaggi di questo tutorial usano le variabili seguenti; impostale di
conseguenza. `LOCATION` è la zona o la regione del cluster.

```bash
export PROJECT=<YOUR_PROJECT_ID>
export CLUSTER=<YOUR_CLUSTER_NAME>
export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>
export NODEPOOL_NAME=<YOUR_4CHIP_NODEPOOL_NAME>
export DOCKER_IMAGE=us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
```

### Esegui l'autenticazione e ottieni le credenziali del cluster { #authenticate-and-get-cluster-credentials }

```bash
gcloud auth login
gcloud auth application-default login
gcloud components install kubectl gke-gcloud-auth-plugin
gcloud container clusters get-credentials ${CLUSTER} --location=${LOCATION} --project=${PROJECT}
```

Per i dettagli, consulta
[Installare kubectl e configurare l'accesso al cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl).

## Invia i carichi di lavoro con kubectl { #submit-workloads-with-kubectl }

I carichi di lavoro vengono inviati come manifest Kubernetes standard `Job` o
`JobSet`.
Gli script Python di esempio sono già preinstallati nell'immagine container
pubblica in `/workspace/examples/`, mentre i manifest Kubernetes `.yaml` di
riferimento (`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`,
`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` e
`examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml`) si trovano
nella directory `examples/` del repository TorchTPU. Esegui i comandi `sed` e `kubectl
apply` riportati di seguito dalla root di un checkout locale del repository
`torch_tpu`.

## Esegui un carico di lavoro su un solo dispositivo (un TensorCore di un host con 4 chip) { #run-a-single-device-workload-one-tensorcore-of-a-4-chip-host }

Il manifest `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`
riserva un host con 4 chip TPU v7x. Ogni chip v7x ha due TensorCore e TorchTPU
espone ogni TensorCore come un dispositivo, quindi l'host ha 8 dispositivi TPU
e la topologia della slice è `2x2x1`:

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ...
    cloud.google.com/gke-tpu-topology: 2x2x1
```

L'esempio esegue un solo processo su un solo dispositivo TPU (un TensorCore) e
lascia inattivi gli altri 7 dispositivi dell'host. Questo si esprime come una
topologia TorchTPU `1x1x1x1`, dove il quarto numero è il numero di TensorCore
per chip che il job usa:

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "1,1,1,1"
```

Il container esegue `simple_pytorch.py`, incluso nell'immagine in
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

Questo è l'intero programma:

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

Inserisci l'immagine e il node pool nel manifest e applicalo:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
```

Controlla lo stato del job e dei pod, poi leggi i log:

```bash
kubectl get job,pods -n default
kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
```

Il job viene eseguito fino al completamento e stampa le quattro righe mostrate
sopra.

## Esegui carichi di lavoro distribuiti su GKE { #run-distributed-workloads-on-gke }

### Singolo host, più chip (4 chip / 8 dispositivi su un host) { #single-host-multiple-chips-4-chips-8-devices-on-one-host }

Il primo passo oltre l'esempio su un solo dispositivo è usare tutti i
dispositivi TPU di un host. Invia un `Job` Kubernetes a singolo host: prima
aggiorna il manifest con l'immagine Docker e il nome del node pool, poi avvia il
job con `kubectl`.

Il manifest `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` si
applica a un singolo host con 4 TPU v7x (8 TensorCore), quindi la topologia
della slice è `2x2x1`:

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
```

TorchTPU la interpreta come una topologia `2x2x1x2` perché avvia un processo per
ogni TensorCore TPU:

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "2,2,1,2"
```

Esegue l'esempio di tensor parallelism manuale dall'immagine Docker:

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

`examples/tensor_parallel/manual/launch_singlehost.sh` rileva la topologia TPU,
la esporta e avvia un processo per ogni dispositivo TPU con
`torch.distributed.run` (`torchrun`), che esegue
`examples/tensor_parallel/manual/tp_worker.py`:

```bash
# Detect and export TPU configuration dynamically.
eval "$(python3 -m torch.tpu.distributed)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

python3 -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    --nproc_per_node=8 ./tp_worker.py
```

Gli script del worker e di avvio si trovano in
`examples/tensor_parallel/manual/`.

Inserisci l'immagine e il node pool nel manifest e applicalo:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
```

Controlla lo stato del job e dei pod:

```bash
kubectl get job,pods -n default
```

### Più host (64 chip / 128 dispositivi su 16 host) { #multiple-hosts-64-chips-128-devices-across-16-hosts }

Per andare oltre un host, lo stesso esempio viene eseguito come `JobSet`
Kubernetes: un pod per host, 16 host per una slice `4x4x4`, con
`examples/tensor_parallel/manual/launch_multihost.sh` che ricava il rank di ogni
host dalla sua posizione nella slice. Aggiorna il manifest con l'immagine Docker
e il nome del node pool con 64 chip e applicalo con `kubectl`:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
```

Controlla lo stato del `JobSet` e dei pod:

```bash
kubectl get jobset,pods -n default
```

## Esegui un esempio PyTorch upstream { #run-an-upstream-pytorch-example }

Poiché TorchTPU è un backend PyTorch nativo (`device="tpu"`,
`torch.accelerator`, `backend="tpu_dist"` per il calcolo distribuito), puoi
scaricare ed eseguire direttamente su GKE gli script standard del repository
upstream [`pytorch/examples`](https://github.com/pytorch/examples).

### Esempio: eseguire MNIST upstream su GKE { #example-running-upstream-mnist-on-gke }

Lo script di addestramento upstream
[`pytorch/examples/mnist`](https://github.com/pytorch/examples/tree/main/mnist)
(`mnist/main.py`) usa l'API `torch.accelerator` di PyTorch, indipendente dal
dispositivo:

```python
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
```

Quando `torch_tpu` è installato nel container,
`torch.accelerator.current_accelerator()` seleziona automaticamente il
dispositivo `"tpu"`, senza richiedere **alcuna modifica al codice** di
`mnist/main.py`.

Per eseguire `mnist/main.py` su un solo dispositivo TPU su GKE partendo da
`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`:

1.  **Aggiorna il blocco `command:` del container** in
    `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` per
    installare `git`, clonare
    [`https://github.com/pytorch/examples.git`](https://github.com/pytorch/examples),
    lasciare `num_workers` di `DataLoader` al valore predefinito (`0`), in modo
    che non esegua il fork di un sottoprocesso worker che tenta di riacquisire
    il file di lock del dispositivo `libtpu`, ed eseguire
    `mnist/main.py`:

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

2.  **Elimina eventuali esecuzioni precedenti del job** (le specifiche dei `Job`
    Kubernetes sono immutabili una volta create), sostituisci i segnaposto se
    necessario, applica il manifest aggiornato e visualizza in streaming i log
    dell'addestramento MNIST:

    ```bash
    kubectl delete -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml --ignore-not-found
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
    ```

### Scegliere un modello di manifest per altri esempi upstream { #choosing-a-manifest-template-for-other-upstream-examples }

Per eseguire altri esempi upstream su un solo dispositivo o distribuiti, scegli
il modello di manifest di riferimento che corrisponde alla topologia di
destinazione:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Tipo di carico di lavoro | Percorso del manifest | Topologia TPU (`gke-tpu-topology`) | `TORCH_TPU_TOPOLOGY` | Descrizione |
| :--- | :--- | :--- | :--- | :--- |
| **Singolo dispositivo** (1 TensorCore su un host con 4 chip) | `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` | `"1,1,1,1"` | Esegue un solo processo Python su 1 dispositivo TPU. |
| **Distribuito a singolo host** (4 chip / 8 core) | `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` | `"2,2,1,2"` | Esegue 8 rank distribuiti (`WORLD_SIZE=8`) su un singolo host con 4 chip tramite `torchrun`. |
| **Distribuito multi-host** (64 chip / 128 core) | `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` | `"4,4,4,2"` | `JobSet` Kubernetes che coordina 16 pod worker su una slice con 64 chip. |
<!-- mdformat on -->

Per usare altre topologie di slice TPU, aggiorna
`cloud.google.com/gke-tpu-slice-topology`, `cloud.google.com/gke-tpu-topology`
e `TORCH_TPU_TOPOLOGY` nel manifest (consulta
[Architettura di sistema delle VM TPU](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
e
[Tipi di macchine e topologie TPU](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)).

### Portare un esempio PyTorch upstream su GKE { #porting-an-upstream-pytorch-example-to-run-on-gke }

Poiché TorchTPU è un backend PyTorch nativo, eseguire su GKE un esempio o un
test PyTorch upstream richiede pochissimo lavoro:

*   **Nessuna modifica al codice per gli esempi non distribuiti**: gli esempi
    non distribuiti scritti per i dispositivi PyTorch standard (tramite
    `torch.accelerator` o `torch.device`) funzionano con TorchTPU senza
    modifiche al codice.
*   **Gli esempi non deprecati dovrebbero funzionare**: gli esempi PyTorch
    upstream non deprecati e mantenuti attivamente, e le API moderne,
    funzionano subito (per i dettagli consulta
    [Funzionalità supportate](./feature_set.md)).
*   **Gli esempi distribuiti possono richiedere piccole modifiche**: alcuni
    esempi distribuiti upstream sono ottimizzati in modo eccessivo per altri
    dispositivi (ad esempio backend, flag o ipotesi di sharding specifici per GPU
    o NCCL) e possono richiedere piccole modifiche per funzionare su TPU
    (consulta le sezioni sul calcolo distribuito qui sopra e
    [Best practice per FSDP](./case_studies_fsdp.md)).
