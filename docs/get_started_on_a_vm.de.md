# Erste Schritte mit TorchTPU auf einer TPU-VM { #get-started-with-torchtpu-on-a-tpu-vm }

<!-- Link to colab and to generated notebook -->

[![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/get_started_on_a_vm.ipynb)

Am einfachsten verwenden Sie TorchTPU auf einer TPU-VM. In diesem Beispiel
führen wir ein einfaches Beispiel mit Tensor-Parallelität aus. Dabei geht es
nicht um das Beispiel selbst, sondern nur um den Einrichtungscode, mit dem Sie
auf einer VM starten können. In der nächsten Anleitung sehen wir, wie Sie
etwas Ähnliches mit GKE erreichen. Das ist nötig, um auf aktuelle TPU-Hardware
zuzugreifen.

## 1. Installieren und einrichten { #1-install-and-set-up }

Bevor Sie dieses Notebook ausführen, prüfen Sie, ob Sie auf einer
[Google Cloud TPU-VM](https://docs.cloud.google.com/tpu/docs/intro-to-tpu?hl=en)
arbeiten (z. B. v6e).

Installieren Sie TorchTPU und die Abhängigkeiten auf Ihrer TPU-VM. Die
Komponenten müssen in dieser Reihenfolge installiert werden, damit der
PyTorch-Dispatcher und die TPU-Treiber korrekt zusammenpassen.

### Schritt 1: Python 3.12 initialisieren { #step-1-initialize-python-312 }

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### Schritt 2: TorchTPU und Abhängigkeiten installieren { #step-2-install-torchtpu-and-dependencies }

```bash
pip install torch_tpu
```

## 2. Hardware initialisieren { #2-initialize-the-hardware }

Wie bei einer CPU oder GPU muss das TPU-Backend explizit initialisiert werden.
Dabei wird der Gerätestring „tpu“ bei PyTorch registriert.

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu

print(f"TPU Initialization complete. Primary device: {device}")
```

## 3. Mit Tensoren arbeiten { #3-work-with-tensors }

Nach der Initialisierung können Sie mit der TPU ähnlich arbeiten wie mit einer
GPU. Sie können vorhandene CPU-Tensoren auf die TPU verschieben oder neue
Tensoren direkt auf der Hardware erstellen.

Es gibt zwei übliche Möglichkeiten, das TPU-Gerät anzugeben:

1.  Mit dem `device`-Objekt, das `torch.device()` zurückgibt.
2.  Mit der String-Kennung `"tpu"`.

```python
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

## 4. Berechnen und Ergebnisse materialisieren { #4-compute-and-materialize-results }

Berechnungen auf der TPU funktionieren nahtlos. Wenn Sie Operationen auf
TPU-Tensoren ausführen, findet die Berechnung auf der Hardware statt. Um die
Daten zurück auf die CPU zu holen, etwa zum Ausgeben oder für die Arbeit mit
Standard-Python-Bibliotheken, verwenden Sie die Methode `.cpu()`.

```python
# Perform a simple operation on the TPU
result_on_tpu = tpu_tensor + direct_tpu_tensor

# Bring the result back to the CPU to print or use in other libraries
result_on_cpu = result_on_tpu.cpu()

print("Computation complete.")
print(f"Result Checksum (Sum): {result_on_cpu.sum():.4f}")
```

### 🎉 Geschafft! { #success }

Sie haben die TPU initialisiert, Tensoren auf das Gerät übertragen, eine
Berechnung durchgeführt und die Ergebnisse abgerufen.
