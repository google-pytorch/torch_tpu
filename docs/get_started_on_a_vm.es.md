# Primeros pasos con TorchTPU en una VM de TPU { #get-started-with-torchtpu-on-a-tpu-vm }

<!-- Link to colab and to generated notebook -->

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/get_started_on_a_vm.ipynb)

La forma más sencilla de usar TorchTPU es en una VM de TPU. En este ejemplo,
ejecutaremos un ejemplo simple de paralelismo de tensores. No nos enfocamos en
el ejemplo, solo en el código de configuración para ejecutarlo en una VM. En el
siguiente instructivo, veremos cómo lograr algo similar a través de GKE, que es
necesario para acceder al hardware de TPU más reciente.

## 1. Instala y configura { #1-install-and-set-up }

Antes de ejecutar este notebook, asegúrate de estar en una
[VM de Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu?hl=en)
(por ejemplo, v6e).

Instala TorchTPU y sus dependencias en tu VM de TPU. Los componentes deben
instalarse en este orden para garantizar que el despachador (dispatcher) de
PyTorch y los controladores de TPU queden alineados correctamente.

### Paso 1: Inicializa Python 3.12 { #step-1-initialize-python-312 }

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### Paso 2: Instala TorchTPU y sus dependencias { #step-2-install-torchtpu-and-dependencies }

```bash
pip install torch_tpu
```

## 2. Inicializa el hardware { #2-initialize-the-hardware }

Al igual que con una CPU o una GPU, el backend de TPU debe inicializarse de
forma explícita. Esto registra la cadena de dispositivo "tpu" en PyTorch.

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu

print(f"TPU Initialization complete. Primary device: {device}")
```

## 3. Trabaja con tensores { #3-work-with-tensors }

Una vez inicializada, puedes interactuar con la TPU de forma muy similar a como
lo harías con una GPU. Puedes mover tensores existentes de la CPU a la TPU o
crear tensores nuevos directamente en el hardware.

Tienes dos formas estándar de especificar el dispositivo TPU:

1.  Usar el objeto `device` que devuelve `torch.device()`.
2.  Usar el identificador de cadena `"tpu"`.

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

## 4. Calcula y materializa los resultados { #4-compute-and-materialize-results }

Realizar operaciones matemáticas en la TPU es muy sencillo. Cuando realizas
operaciones con tensores de TPU, el cálculo se ejecuta en el hardware. Para
traer los datos de vuelta a la CPU e imprimirlos o usarlos con bibliotecas
estándar de Python, usa el método `.cpu()`.

```python
# Perform a simple operation on the TPU
result_on_tpu = tpu_tensor + direct_tpu_tensor

# Bring the result back to the CPU to print or use in other libraries
result_on_cpu = result_on_tpu.cpu()

print("Computation complete.")
print(f"Result Checksum (Sum): {result_on_cpu.sum():.4f}")
```

### 🎉 ¡Listo! { #success }

Inicializaste la TPU, asignaste tensores al dispositivo, realizaste un cálculo y
recuperaste los resultados correctamente.
