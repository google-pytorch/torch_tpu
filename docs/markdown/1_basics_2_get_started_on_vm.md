---
torch_tpu_version: 0.1.1.dev20260922
---

# Get started with TorchTPU on a TPU VM

The simplest way to use TorchTPU is on a TPU VM. In this example we will run a
simple tensor parallel example. We do not focus on the example, just the setup
code to get running on a VM. In the following tutorial we will see how to
achieve something similar through GKE, which is necessary to access recent TPU
hardware.

## 1. Install and set up

Before running this notebook, ensure you are running on a
[Google Cloud TPU VM](https://docs.cloud.google.com/tpu/docs/intro-to-tpu?hl=en)
(e.g., v6e).

Install TorchTPU and its dependencies on your TPU VM. Components must be
installed in this order to ensure the PyTorch dispatcher and TPU drivers align
correctly.

### Step 1: Initialize Python 3.12

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### Step 2: Install TorchTPU and dependencies

```bash
# 1. Authenticate to get access to wheel files
pip install keyrings.google-artifactregistry-auth
gcloud auth login
gcloud auth application-default login

# 2. Install torch_tpu, PyTorch 2.11 CPU will install automatically, do not install manually.
pip install --pre --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" torch_tpu
```

## 2. Initialize the hardware

Similar to a CPU or GPU, the TPU backend must be explicitly initialized. This
registers the "tpu" device string with PyTorch.

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu

print(f"TPU Initialization complete. Primary device: {device}")
```

## 3. Work with tensors

Once initialized, you can interact with the TPU much like you would with a GPU.
You can move existing CPU tensors to the TPU or create new tensors directly on
the hardware.

You have two standard ways to specify the TPU device:

1.  Using the `device` object returned by `torch.device()`.
2.  Using the `"tpu"` string identifier.

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

## 4. Compute and materialize results

Performing math on the TPU is seamless. When you perform operations on TPU
tensors, the computation happens on the hardware. To retrieve the data back to
the CPU for printing or interaction with standard Python libraries, use the
`.cpu()` method.

```python
# Perform a simple operation on the TPU
result_on_tpu = tpu_tensor + direct_tpu_tensor

# Bring the result back to the CPU to print or use in other libraries
result_on_cpu = result_on_tpu.cpu()

print("Computation complete.")
print(f"Result Checksum (Sum): {result_on_cpu.sum():.4f}")
```

### 🎉 Success!

You have successfully initialized the TPU, mapped tensors to the device,
performed a calculation, and retrieved the results.
