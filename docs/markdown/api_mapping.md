---
torch_tpu_version: 0.1.1.dev20260922
---

# **API Mapping Table: `torch.cuda` → `torch.tpu`**

Once the backend is initialized via `device = torch.device("tpu")`, TorchTPU
registers a dedicated Python module: **`torch.tpu`**. This module is designed as
a drop-in replacement for the standard `torch.cuda` API.

Use this reference to quickly translate CUDA-based code to TorchTPU.

## **Core Device Queries**

CUDA API                             | TPU API                                        | Description
:----------------------------------- | :--------------------------------------------- | :----------
`torch.cuda.is_available()`          | `torch.tpu.is_available()`                     | Checks if the TPU backend is ready.
`torch.cuda.device_count()`          | `torch.tpu.device_count()`                     | Number of devices assigned to the process.
`torch.cuda.current_device()`        | `torch.tpu.current_device()`                   | Returns the active core index.
`torch.cuda.get_device_properties()` | *(Implicit in `torch.device("tpu")`)*          | Hardware metadata via PjRt logs.
`torch.cuda.set_device(rank)`        | Handled automatically by `torch.device("tpu")` | Backend init is automatic.

```python
import torch

# Initialize hardware
device = torch.device("tpu")

print(f"torch.tpu.is_available():    {torch.tpu.is_available()}")
print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
print(f"torch.tpu.device_count():    {torch.tpu.device_count()}")
print(f"torch.tpu.current_device():  {torch.tpu.current_device()}")
```

## **Random Number Generation**

CUDA API                           | TPU API                           | Description
:--------------------------------- | :-------------------------------- | :----------
`torch.cuda.manual_seed(seed)`     | `torch.tpu.manual_seed(seed)`     | Seeds the current core's RNG.
`torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | Seeds all TPU cores (for distributed).

```python
# Set the seed across all cores
torch.tpu.manual_seed_all(42)
print("Seeded all TPU cores with value 42.")
```

## **Automatic Mixed Precision (AMP)**

TorchTPU integrates with the standard `torch.amp.autocast` framework. The device
string is `"tpu"`.

CUDA API                               | TPU API                               | Description
:------------------------------------- | :------------------------------------ | :----------
`torch.amp.autocast("cuda", ...)`      | `torch.amp.autocast("tpu", ...)`      | Enables automatic mixed precision.
`torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | Lists hardware-accelerated dtypes.

```python
supported_dtypes = torch.tpu.get_amp_supported_dtype()
print(f"AMP supported dtypes: {supported_dtypes}")
```

## **Streams and Events**

The `torch.tpu` module provides `Stream` and `Event` objects to maintain API
parity with CUDA.

| CUDA API                      | TPU API                      | Status        |
| :---------------------------- | :--------------------------- | :------------ |
| `torch.cuda.current_stream()` | `torch.tpu.current_stream()` | **Placeholder |
:                               :                              : (dummy)**     :
| `torch.cuda.default_stream()` | `torch.tpu.default_stream()` | **Placeholder |
:                               :                              : (dummy)**     :
| `torch.cuda.set_stream()`     | `torch.tpu.set_stream()`     | **Placeholder |
:                               :                              : (dummy)**     :

> [!NOTE] In TorchTPU's deferred execution model, the XLA compiler and PjRt
> runtime manage execution ordering and hardware concurrency automatically.
> Streams are not needed.
