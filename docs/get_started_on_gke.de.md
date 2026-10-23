# Erste Schritte mit TorchTPU auf GKE { #get-started-with-torchtpu-on-gke }

In dieser Anleitung führen Sie PyTorch-Arbeitslasten auf einem TPU-Cluster in
Google Kubernetes Engine (GKE) aus. Sie besteht aus drei Teilen:

1.  **Ein einfaches Beispiel auf einem einzelnen Gerät ausführen**
    (`examples/simple_pytorch/simple_pytorch.py`) als Kubernetes-`Job`.
2.  **Ein verteiltes Beispiel auf mehreren Geräten ausführen**
    (`examples/tensor_parallel/manual/tp_worker.py`) auf einem einzelnen Host
    mit 4 Chips (`Job`) und über einen Multi-Host-Slice mit 64 Chips (`JobSet`).
3.  **Ein Upstream-Beispiel für PyTorch ausführen** aus
    [`pytorch/examples`](https://github.com/pytorch/examples) (z. B. MNIST).

Wir gehen davon aus, dass Sie bereits einen GKE-Standard-Cluster mit einem
TPU-Knotenpool haben. Falls Sie einen erstellen müssen, lesen Sie
[TPU-Arbeitslasten in GKE Standard bereitstellen](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus).

## Vorbereitung { #before-you-begin }

### Voraussetzungen { #prerequisites }

*   Ein GKE-Cluster mit einem `tpu7x`-Knotenpool. Diese Anleitung verwendet
    einen Knotenpool mit 4 Chips (`2x2x1`).
*   Die [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk).

### Umgebungsvariablen festlegen { #set-environment-variables }

Alle Schritte in dieser Anleitung verwenden die folgenden Variablen. Legen Sie
sie entsprechend fest. `LOCATION` ist die Zone oder Region Ihres Clusters.

```bash
export PROJECT=<YOUR_PROJECT_ID>
export CLUSTER=<YOUR_CLUSTER_NAME>
export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>
export NODEPOOL_NAME=<YOUR_4CHIP_NODEPOOL_NAME>
export DOCKER_IMAGE=us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
```

### Authentifizieren und Cluster-Anmeldedaten abrufen { #authenticate-and-get-cluster-credentials }

```bash
gcloud auth login
gcloud auth application-default login
gcloud components install kubectl gke-gcloud-auth-plugin
gcloud container clusters get-credentials ${CLUSTER} --location=${LOCATION} --project=${PROJECT}
```

Details finden Sie unter
[kubectl installieren und Clusterzugriff konfigurieren](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl).

## Arbeitslasten mit kubectl senden { #submit-workloads-with-kubectl }

Arbeitslasten werden als Standard-Kubernetes-Manifeste vom Typ `Job` oder
`JobSet` gesendet. Die Python-Beispielskripts sind im öffentlichen Container-Image unter
`/workspace/examples/` bereits vorinstalliert. Die Kubernetes-Referenzmanifeste
im Format `.yaml`
(`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`,
`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` und
`examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml`) liegen im
Verzeichnis `examples/` des TorchTPU-Repositorys. Führen Sie die unten
gezeigten Befehle `sed` und `kubectl apply` im Stammverzeichnis eines lokalen
Checkouts des Repositorys `torch_tpu` aus.

## Arbeitslast auf einem einzelnen Gerät ausführen (ein TensorCore eines Hosts mit 4 Chips) { #run-a-single-device-workload-one-tensorcore-of-a-4-chip-host }

Das Manifest `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`
reserviert einen Host mit 4 TPU-v7x-Chips. Jeder v7x-Chip hat zwei TensorCores,
und TorchTPU stellt jeden TensorCore als ein Gerät bereit. Der Host hat also 8
TPU-Geräte, und seine Slice-Topologie ist `2x2x1`:

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ...
    cloud.google.com/gke-tpu-topology: 2x2x1
```

Das Beispiel führt einen einzelnen Prozess auf einem einzelnen TPU-Gerät (einem
TensorCore) aus. Die anderen 7 Geräte auf dem Host bleiben ungenutzt. Das wird
als TorchTPU-Topologie `1x1x1x1` ausgedrückt. Die vierte Zahl ist die Anzahl
der TensorCores pro Chip, die der Job verwendet:

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "1,1,1,1"
```

Der Container führt `simple_pytorch.py` aus, das im Image unter
`/workspace/examples/` enthalten ist:

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

Das ist das vollständige Programm:

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

Tragen Sie das Image und den Knotenpool in das Manifest ein und wenden Sie es
an:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
```

Prüfen Sie den Status von Job und Pod und lesen Sie dann die Logs:

```bash
kubectl get job,pods -n default
kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
```

Der Job läuft bis zum Ende und gibt die vier oben gezeigten Zeilen aus.

## Verteilte Arbeitslasten auf GKE ausführen { #run-distributed-workloads-on-gke }

### Ein Host, mehrere Chips (4 Chips / 8 Geräte auf einem Host) { #single-host-multiple-chips-4-chips-8-devices-on-one-host }

Der nächste Schritt nach dem Beispiel mit einem Gerät ist, alle TPU-Geräte auf
einem Host zu verwenden. Senden Sie einen Kubernetes-`Job` für einen einzelnen
Host. Aktualisieren Sie dazu zuerst das Manifest mit Ihrem Docker-Image und dem
Namen Ihres Knotenpools und starten Sie den Job dann mit `kubectl`.

Das Manifest `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml`
gilt für einen einzelnen Host mit 4 v7x-TPUs (8 TensorCores). Die
Slice-Topologie ist daher `2x2x1`:

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
```

TorchTPU interpretiert das als Topologie `2x2x1x2`, weil es einen Prozess pro
TPU-TensorCore startet:

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "2,2,1,2"
```

Es führt das manuelle Tensor-Parallel-Beispiel aus dem Docker-Image aus:

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

`examples/tensor_parallel/manual/launch_singlehost.sh` erkennt die
TPU-Topologie, exportiert sie und startet mit `torch.distributed.run`
(`torchrun`) einen Prozess pro TPU-Gerät, der
`examples/tensor_parallel/manual/tp_worker.py` ausführt:

```bash
# Detect and export TPU configuration dynamically.
eval "$(python3 -m torch.tpu.distributed)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

python3 -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    --nproc_per_node=8 ./tp_worker.py
```

Die Worker- und Startskripts befinden sich in `examples/tensor_parallel/manual/`.

Tragen Sie das Image und den Knotenpool in das Manifest ein und wenden Sie es
an:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
```

Prüfen Sie den Status von Job und Pod:

```bash
kubectl get job,pods -n default
```

### Mehrere Hosts (64 Chips / 128 Geräte auf 16 Hosts) { #multiple-hosts-64-chips-128-devices-across-16-hosts }

Um über einen Host hinaus zu skalieren, läuft dasselbe Beispiel als
Kubernetes-`JobSet`: ein Pod pro Host, 16 Hosts für einen `4x4x4`-Slice.
Dabei leitet `examples/tensor_parallel/manual/launch_multihost.sh` den Rank
jedes Hosts aus seiner Position im Slice ab. Aktualisieren Sie das Manifest mit
Ihrem Docker-Image und dem Namen Ihres Knotenpools mit 64 Chips und wenden Sie
es mit `kubectl` an:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
```

Prüfen Sie den Status von `JobSet` und Pod:

```bash
kubectl get jobset,pods -n default
```

## Upstream-Beispiel für PyTorch ausführen { #run-an-upstream-pytorch-example }

TorchTPU ist ein natives PyTorch-Backend (`device="tpu"`,
`torch.accelerator`, verteiltes `backend="tpu_dist"`). Deshalb können Sie
Standardskripts aus dem Upstream-Repository
[`pytorch/examples`](https://github.com/pytorch/examples) abrufen und direkt
auf GKE ausführen.

### Beispiel: Upstream-MNIST auf GKE ausführen { #example-running-upstream-mnist-on-gke }

Das Upstream-Trainingsskript
[`pytorch/examples/mnist`](https://github.com/pytorch/examples/tree/main/mnist)
(`mnist/main.py`) verwendet die geräteunabhängige API
`torch.accelerator` von PyTorch:

```python
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
```

Wenn `torch_tpu` im Container installiert ist, wählt
`torch.accelerator.current_accelerator()` automatisch das Gerät `"tpu"` aus.
An `mnist/main.py` sind also **keine Codeänderungen** nötig.

So führen Sie `mnist/main.py` auf einem einzelnen TPU-Gerät auf GKE aus,
ausgehend von `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`:

1.  **Aktualisieren Sie den Block `command:` des Containers** in
    `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`. Der Block
    soll `git` installieren,
    [`https://github.com/pytorch/examples.git`](https://github.com/pytorch/examples)
    klonen, `num_workers` von `DataLoader` auf dem Standardwert (`0`) belassen,
    damit kein Worker-Unterprozess geforkt wird, der versucht, die
    Gerätesperrdatei von `libtpu` erneut zu belegen, und `mnist/main.py`
    ausführen:

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

2.  **Löschen Sie alle vorherigen Ausführungen des Jobs** (Spezifikationen von
    Kubernetes-`Job` sind nach dem Erstellen unveränderlich), ersetzen Sie bei
    Bedarf die Platzhalter, wenden Sie das aktualisierte Manifest an und
    streamen Sie die MNIST-Trainingslogs:

    ```bash
    kubectl delete -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml --ignore-not-found
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
    ```

### Manifestvorlage für andere Upstream-Beispiele auswählen { #choosing-a-manifest-template-for-other-upstream-examples }

Um andere Upstream-Beispiele auf einem Gerät oder verteilt auszuführen, wählen
Sie die Referenzvorlage für das Manifest, die zu Ihrer Zieltopologie passt:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Arbeitslasttyp | Manifestpfad | TPU-Topologie (`gke-tpu-topology`) | `TORCH_TPU_TOPOLOGY` | Beschreibung |
| :--- | :--- | :--- | :--- | :--- |
| **Einzelnes Gerät** (1 TensorCore auf einem Host mit 4 Chips) | `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` | `"1,1,1,1"` | Führt einen einzelnen Python-Prozess auf 1 TPU-Gerät aus. |
| **Verteilt auf einem Host** (4 Chips / 8 Kerne) | `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` | `"2,2,1,2"` | Führt über `torchrun` 8 verteilte Ranks (`WORLD_SIZE=8`) auf einem einzelnen Host mit 4 Chips aus. |
| **Verteilt auf mehreren Hosts** (64 Chips / 128 Kerne) | `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` | `"4,4,4,2"` | Kubernetes-`JobSet`, das 16 Worker-Pods über einen Slice mit 64 Chips koordiniert. |
<!-- mdformat on -->

Um andere TPU-Slice-Topologien zu verwenden, ändern Sie
`cloud.google.com/gke-tpu-slice-topology`, `cloud.google.com/gke-tpu-topology`
und `TORCH_TPU_TOPOLOGY` im Manifest (siehe
[Systemarchitektur von TPU-VMs](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
und
[TPU-Maschinentypen und -Topologien](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)).

### Upstream-Beispiel für PyTorch auf GKE portieren { #porting-an-upstream-pytorch-example-to-run-on-gke }

Da TorchTPU ein natives PyTorch-Backend ist, brauchen Sie nur wenig Aufwand, um
ein Upstream-Beispiel oder einen Upstream-Test für PyTorch auf GKE auszuführen:

*   **Keine Codeänderungen für nicht verteilte Beispiele**: Nicht verteilte
    Beispiele, die für Standard-PyTorch-Geräte geschrieben sind (über
    `torch.accelerator` oder `torch.device`), funktionieren ohne Codeänderungen
    mit TorchTPU.
*   **Nicht veraltete Beispiele sollten funktionieren**: Aktiv gepflegte, nicht
    veraltete Upstream-Beispiele für PyTorch und moderne APIs funktionieren
    sofort (Details unter [Unterstützte Features](./feature_set.md)).
*   **Verteilte Beispiele brauchen eventuell kleine Anpassungen**: Einige
    verteilte Upstream-Beispiele sind stark auf andere Geräte abgestimmt (z. B.
    GPU- oder NCCL-spezifische Backends, Flags oder Sharding-Annahmen). Sie
    brauchen eventuell kleine Anpassungen, um auf TPU zu laufen (siehe die
    Abschnitte zu verteilten Arbeitslasten oben und
    [Best Practices für FSDP](./case_studies_fsdp.md)).
