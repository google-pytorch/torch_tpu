# 在 TPU 虚拟机上开始使用 TorchTPU { #get-started-with-torchtpu-on-a-tpu-vm }

<!-- Link to colab and to generated notebook -->

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/get_started_on_a_vm.ipynb)

使用 TorchTPU 最简单的方式是在 TPU 虚拟机上运行。在本例中，我们将运行一个简单的张量并行示例。我们不关注示例本身，只关注在虚拟机上运行所需的设置代码。在接下来的教程中，我们将介绍如何通过 GKE 实现类似的效果，若要使用较新的 TPU 硬件，必须通过 GKE。

## 1. 安装和设置 { #1-install-and-set-up }

在运行此笔记本之前，请确保您在 [Google Cloud TPU 虚拟机](https://docs.cloud.google.com/tpu/docs/intro-to-tpu?hl=en)（例如 v6e）上运行。

在您的 TPU 虚拟机上安装 TorchTPU 及其依赖项。必须按此顺序安装各组件，以确保 PyTorch 调度器（dispatcher）与 TPU 驱动程序正确匹配。

### 第 1 步：初始化 Python 3.12 { #step-1-initialize-python-312 }

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### 第 2 步：安装 TorchTPU 及其依赖项 { #step-2-install-torchtpu-and-dependencies }

```bash
pip install torch_tpu
```

## 2. 初始化硬件 { #2-initialize-the-hardware }

与 CPU 或 GPU 类似，TPU 后端必须显式初始化。这会向 PyTorch 注册 "tpu" 设备字符串。

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu

print(f"TPU Initialization complete. Primary device: {device}")
```

## 3. 使用张量 { #3-work-with-tensors }

初始化完成后，您可以像使用 GPU 一样与 TPU 交互。您可以将现有的 CPU 张量移到 TPU 上，也可以直接在硬件上创建新张量。

您可以通过两种标准方式指定 TPU 设备：

1.  使用 `torch.device()` 返回的 `device` 对象。
2.  使用 `"tpu"` 字符串标识符。

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

## 4. 计算并具体化结果 { #4-compute-and-materialize-results }

在 TPU 上进行数学运算非常顺畅。当您对 TPU 张量执行操作时，计算会在硬件上进行。如需将数据取回 CPU 以便打印或与标准 Python 库交互，请使用 `.cpu()` 方法。

```python
# Perform a simple operation on the TPU
result_on_tpu = tpu_tensor + direct_tpu_tensor

# Bring the result back to the CPU to print or use in other libraries
result_on_cpu = result_on_tpu.cpu()

print("Computation complete.")
print(f"Result Checksum (Sum): {result_on_cpu.sum():.4f}")
```

### 🎉 成功！ { #success }

您已成功初始化 TPU，将张量映射到设备，执行了计算，并取回了结果。
