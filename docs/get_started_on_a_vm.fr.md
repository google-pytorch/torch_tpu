# Premiers pas avec TorchTPU sur une VM TPU { #get-started-with-torchtpu-on-a-tpu-vm }

<!-- Link to colab and to generated notebook -->

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/get_started_on_a_vm.ipynb)

Le moyen le plus simple d'utiliser TorchTPU est de passer par une VM TPU. Dans
cet exemple, nous exécutons un exemple simple de parallélisme de tensors. Nous
ne nous concentrons pas sur l'exemple lui-même, mais sur le code de
configuration nécessaire pour l'exécuter sur une VM. Dans le tutoriel suivant,
nous verrons comment obtenir un résultat similaire avec GKE, qui est
nécessaire pour accéder au matériel TPU récent.

## 1. Installer et configurer { #1-install-and-set-up }

Avant d'exécuter ce notebook, assurez-vous de travailler sur une
[VM Google Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu?hl=en)
(par exemple, v6e).

Installez TorchTPU et ses dépendances sur votre VM TPU. Vous devez installer
les composants dans cet ordre pour que le dispatcher PyTorch et les pilotes TPU
soient correctement alignés.

### Étape 1 : initialiser Python 3.12 { #step-1-initialize-python-312 }

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### Étape 2 : installer TorchTPU et ses dépendances { #step-2-install-torchtpu-and-dependencies }

```bash
pip install torch_tpu
```

## 2. Initialiser le matériel { #2-initialize-the-hardware }

Comme pour un CPU ou un GPU, le backend TPU doit être initialisé
explicitement. Cette opération enregistre la chaîne d'appareil « tpu »
auprès de PyTorch.

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu

print(f"TPU Initialization complete. Primary device: {device}")
```

## 3. Travailler avec des tensors { #3-work-with-tensors }

Une fois le TPU initialisé, vous pouvez interagir avec lui à peu près comme
avec un GPU. Vous pouvez déplacer des tensors CPU existants vers le TPU ou
créer de nouveaux tensors directement sur le matériel.

Vous disposez de deux méthodes standards pour indiquer l'appareil TPU :

1.  Utiliser l'objet `device` renvoyé par `torch.device()`.
2.  Utiliser l'identifiant de chaîne `"tpu"`.

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

## 4. Calculer et matérialiser les résultats { #4-compute-and-materialize-results }

Le calcul sur le TPU est transparent. Lorsque vous effectuez des opérations sur
des tensors TPU, le calcul a lieu sur le matériel. Pour rapatrier les données vers le CPU afin de les afficher ou de les
utiliser avec des bibliothèques Python standards, utilisez la méthode `.cpu()`.

```python
# Perform a simple operation on the TPU
result_on_tpu = tpu_tensor + direct_tpu_tensor

# Bring the result back to the CPU to print or use in other libraries
result_on_cpu = result_on_tpu.cpu()

print("Computation complete.")
print(f"Result Checksum (Sum): {result_on_cpu.sum():.4f}")
```

### 🎉 Réussite ! { #success }

Vous avez initialisé le TPU, placé des tensors sur l'appareil, effectué un
calcul et récupéré les résultats.
