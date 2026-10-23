# Inizia a utilizzare TorchTPU su una VM TPU { #get-started-with-torchtpu-on-a-tpu-vm }

<!-- Link to colab and to generated notebook -->

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/get_started_on_a_vm.ipynb)

Il modo più semplice per usare TorchTPU è su una VM TPU. In questo esempio
eseguiamo un semplice caso di tensor parallelism. Non ci concentriamo
sull'esempio, ma solo sul codice di configurazione necessario per eseguirlo su
una VM. Nel tutorial successivo vedremo come
ottenere qualcosa di simile tramite GKE, che è necessario per accedere
all'hardware TPU più recente.

## 1. Installa e configura { #1-install-and-set-up }

Prima di eseguire questo notebook, assicurati di lavorare su una
[VM Google Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu?hl=en)
(ad esempio v6e).

Installa TorchTPU e le sue dipendenze sulla tua VM TPU. Devi installare i
componenti in questo ordine per garantire che il dispatcher di PyTorch e i
driver TPU siano allineati correttamente.

### Passaggio 1: inizializza Python 3.12 { #step-1-initialize-python-312 }

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### Passaggio 2: installa TorchTPU e le dipendenze { #step-2-install-torchtpu-and-dependencies }

```bash
pip install torch_tpu
```

## 2. Inizializza l'hardware { #2-initialize-the-hardware }

Come per una CPU o una GPU, devi inizializzare esplicitamente il backend TPU.
In questo modo la stringa di dispositivo "tpu" viene registrata in PyTorch.

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu

print(f"TPU Initialization complete. Primary device: {device}")
```

## 3. Lavora con i tensori { #3-work-with-tensors }

Dopo l'inizializzazione, puoi interagire con la TPU in modo molto simile a
come faresti con una GPU. Puoi spostare tensori CPU esistenti sulla TPU oppure
creare nuovi tensori direttamente sull'hardware.

Hai due modi standard per specificare il dispositivo TPU:

1.  Usare l'oggetto `device` restituito da `torch.device()`.
2.  Usare l'identificatore stringa `"tpu"`.

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

## 4. Calcola e materializza i risultati { #4-compute-and-materialize-results }

Eseguire calcoli sulla TPU è semplice. Quando esegui operazioni su tensori TPU,
il calcolo avviene sull'hardware. Per riportare i dati sulla CPU, per
stamparli o usarli con le librerie Python standard, usa il metodo `.cpu()`.

```python
# Perform a simple operation on the TPU
result_on_tpu = tpu_tensor + direct_tpu_tensor

# Bring the result back to the CPU to print or use in other libraries
result_on_cpu = result_on_tpu.cpu()

print("Computation complete.")
print(f"Result Checksum (Sum): {result_on_cpu.sum():.4f}")
```

### 🎉 Operazione riuscita! { #success }

Hai inizializzato la TPU, hai mappato i tensori sul dispositivo, hai eseguito
un calcolo e hai recuperato i risultati.
