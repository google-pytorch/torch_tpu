# Primeiros passos com o TorchTPU em uma VM de TPU { #get-started-with-torchtpu-on-a-tpu-vm }

<!-- Link to colab and to generated notebook -->

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/get_started_on_a_vm.ipynb)

A forma mais simples de usar o TorchTPU é em uma VM de TPU. Neste exemplo, você
vai executar um exemplo simples de paralelismo de tensor. O foco não é o
exemplo em si, mas o código de configuração necessário para começar a usar uma
VM. No próximo tutorial, você vai ver como fazer algo parecido por meio do GKE,
que é necessário para acessar o hardware de TPU mais recente.

## 1. Instale e configure { #1-install-and-set-up }

Antes de executar este notebook, verifique se você está em uma
[VM do Google Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu?hl=en)
(por exemplo, v6e).

Instale o TorchTPU e as dependências dele na sua VM de TPU. Os componentes
precisam ser instalados nesta ordem para garantir que o dispatcher do PyTorch e
os drivers da TPU fiquem alinhados corretamente.

### Etapa 1: inicializar o Python 3.12 { #step-1-initialize-python-312 }

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### Etapa 2: instalar o TorchTPU e as dependências { #step-2-install-torchtpu-and-dependencies }

```bash
pip install torch_tpu
```

## 2. Inicialize o hardware { #2-initialize-the-hardware }

Assim como em uma CPU ou GPU, o backend da TPU precisa ser inicializado de forma
explícita. Isso registra a string de dispositivo "tpu" no PyTorch.

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu

print(f"TPU Initialization complete. Primary device: {device}")
```

## 3. Trabalhe com tensores { #3-work-with-tensors }

Depois da inicialização, você pode interagir com a TPU de forma muito parecida
com uma GPU. Você pode mover tensores de CPU existentes para a TPU ou criar
tensores novos diretamente no hardware.

Há duas formas padrão de especificar o dispositivo TPU:

1.  Usar o objeto `device` retornado por `torch.device()`.
2.  Usar o identificador de string `"tpu"`.

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

## 4. Calcule e materialize os resultados { #4-compute-and-materialize-results }

Fazer cálculos na TPU é simples. Quando você executa operações em tensores de
TPU, a computação acontece no hardware. Para trazer os dados de volta para a CPU
e imprimi-los ou usá-los com bibliotecas padrão do Python, use o método
`.cpu()`.

```python
# Perform a simple operation on the TPU
result_on_tpu = tpu_tensor + direct_tpu_tensor

# Bring the result back to the CPU to print or use in other libraries
result_on_cpu = result_on_tpu.cpu()

print("Computation complete.")
print(f"Result Checksum (Sum): {result_on_cpu.sum():.4f}")
```

### 🎉 Pronto! { #success }

Você inicializou a TPU, mapeou tensores para o dispositivo, fez um cálculo e
recuperou os resultados.
