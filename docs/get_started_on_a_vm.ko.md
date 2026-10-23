# TPU VM에서 TorchTPU 시작하기 { #get-started-with-torchtpu-on-a-tpu-vm }

<!-- Link to colab and to generated notebook -->

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/get_started_on_a_vm.ipynb)

TorchTPU를 사용하는 가장 간단한 방법은 TPU VM에서 사용하는 것입니다. 이 예에서는
간단한 텐서 병렬 예제를 실행합니다. 예제 자체보다는 VM에서 실행하기 위한 설정
코드에 초점을 맞춥니다. 다음 튜토리얼에서는 GKE를 통해 비슷한 작업을 하는 방법을
알아봅니다. 최신 TPU 하드웨어에 액세스하려면 GKE가 필요합니다.

## 1. 설치 및 설정 { #1-install-and-set-up }

이 노트북을 실행하기 전에
[Google Cloud TPU VM](https://docs.cloud.google.com/tpu/docs/intro-to-tpu?hl=en)
(예: v6e)에서 실행 중인지 확인하세요.

TPU VM에 TorchTPU와 종속 항목을 설치하세요. PyTorch 디스패처와 TPU 드라이버가
올바르게 맞물리도록 하려면 구성요소를 이 순서대로 설치해야 합니다.

### 1단계: Python 3.12 초기화 { #step-1-initialize-python-312 }

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### 2단계: TorchTPU 및 종속 항목 설치 { #step-2-install-torchtpu-and-dependencies }

```bash
pip install torch_tpu
```

## 2. 하드웨어 초기화 { #2-initialize-the-hardware }

CPU나 GPU와 마찬가지로 TPU 백엔드도 명시적으로 초기화해야 합니다. 초기화하면
"tpu" 디바이스 문자열이 PyTorch에 등록됩니다.

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu

print(f"TPU Initialization complete. Primary device: {device}")
```

## 3. 텐서 다루기 { #3-work-with-tensors }

초기화가 끝나면 GPU를 사용할 때와 거의 같은 방식으로 TPU를 사용할 수 있습니다.
기존 CPU 텐서를 TPU로 옮기거나 하드웨어에서 직접 새 텐서를 만들 수 있습니다.

TPU 디바이스를 지정하는 표준 방법은 두 가지입니다.

1.  `torch.device()`가 반환하는 `device` 객체를 사용합니다.
2.  `"tpu"` 문자열 식별자를 사용합니다.

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

## 4. 계산 및 결과 구체화 { #4-compute-and-materialize-results }

TPU에서는 수학 연산을 간편하게 수행할 수 있습니다. TPU 텐서에 연산을 수행하면
계산은 하드웨어에서 이루어집니다. 출력하거나 표준 Python 라이브러리와 함께
사용하기 위해 데이터를 CPU로 다시 가져오려면 `.cpu()` 메서드를 사용하세요.

```python
# Perform a simple operation on the TPU
result_on_tpu = tpu_tensor + direct_tpu_tensor

# Bring the result back to the CPU to print or use in other libraries
result_on_cpu = result_on_tpu.cpu()

print("Computation complete.")
print(f"Result Checksum (Sum): {result_on_cpu.sum():.4f}")
```

### 🎉 성공! { #success }

TPU를 초기화하고, 텐서를 디바이스에 매핑하고, 계산을 수행하고, 결과를 가져오는
데 성공했습니다.
