---
torch_tpu_version: 0.1.1.dev20260922
---

# **API Mapping Table: `torch.cuda` → `torch.tpu`**

Once the backend is initialized via `device = torch.device("tpu")`, TorchTPU
registers a dedicated Python module: **`torch.tpu`**. This module is designed as
a drop-in replacement for the standard `torch.cuda` API.

Where PyTorch provides a device-agnostic `torch.accelerator` API, use it instead
of a `torch.tpu` API. Use this reference to quickly translate CUDA-based code to
TorchTPU.

## **Core Device Queries**

CUDA API                             | TPU / Accelerator API                      | Description
:----------------------------------- | :----------------------------------------- | :----------
`torch.cuda.is_available()`          | `torch.accelerator.is_available()`         | Checks if the TPU backend is ready.
`torch.cuda.is_initialized()`        | `torch.tpu.is_initialized()`               | Checks if PyTorch's TPU state has been initialized.
`torch.cuda.device_count()`          | `torch.accelerator.device_count()`         | Number of devices assigned to the process.
`torch.cuda.current_device()`        | `torch.accelerator.current_device_index()` | Returns the active core index.
`torch.cuda.get_device_name()`       | `torch.tpu.get_device_name()`              | Returns human-readable model name of the attached TPU.
`torch.cuda.get_device_properties()` | `torch.tpu.get_device_properties()`        | Returns static device properties (name, memory, cores).
`torch.cuda.set_device(rank)`        | `torch.accelerator.set_device_index(rank)` | Validates the active device index for the current process.
`torch.cuda.device(rank)`            | `torch.accelerator.device_index(rank)`     | Context manager that sets the active device index.

The `torch.accelerator` APIs in this table require PyTorch 2.11 or later. In the
rare case that you use an older PyTorch version, use the corresponding
`torch.tpu` API instead, for example `torch.tpu.device_count()` or
`torch.tpu.current_device()`. In PyTorch 2.11 and later, these `torch.tpu` APIs
are internal to TorchTPU, and calling them from your code raises an error.

```python
import torch

# Initialize hardware
device = torch.device("tpu")

print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
print(f"torch.accelerator.device_count(): {torch.accelerator.device_count()}")
print(
    "torch.accelerator.current_device_index():"
    f" {torch.accelerator.current_device_index()}"
)
```

## **Random Number Generation**

CUDA API                           | TPU / Accelerator API                      | Description
:--------------------------------- | :----------------------------------------- | :----------
`torch.cuda.manual_seed(seed)`     | `torch.tpu.manual_seed(seed)`              | Seeds the current core's RNG.
`torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)`          | Seeds all TPU cores (for distributed).
`torch.cuda.initial_seed()`        | `torch.accelerator.random.initial_seed()`  | Returns initial random seed of current TPU device.
`torch.cuda.get_rng_state()`       | `torch.accelerator.random.get_rng_state()` | Returns RNG state tensor for the current device.
`torch.cuda.set_rng_state(state)`  | `torch.tpu.set_rng_state(state)`           | Sets RNG state tensor for the current device.

The `torch.accelerator.random` APIs in this table require PyTorch 2.13 or later.
In the rare case that you use an older PyTorch version, use
`torch.tpu.initial_seed()` or `torch.tpu.get_rng_state()` instead. In PyTorch
2.13 and later, these `torch.tpu` APIs are internal to TorchTPU, and calling
them from your code raises an error.

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

For streams and events, prefer standard `torch.accelerator` APIs. Use
`torch.tpu` only for TPU-specific features.

CUDA API                      | TPU / Accelerator API                | Description
:---------------------------- | :----------------------------------- | :----------
`torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | Returns the active stream.
`torch.cuda.set_stream()`     | `torch.accelerator.set_stream()`     | Sets the active stream.
`torch.cuda.default_stream()` | `torch.tpu.default_stream()`         | Returns the default stream.
`torch.cuda.synchronize()`    | `torch.accelerator.synchronize()`    | Waits for all pending work on the device to finish.

The `torch.accelerator` APIs in this table require PyTorch 2.11 or later. In the
rare case that you use an older PyTorch version, use
`torch.tpu.current_stream()`, `torch.tpu.set_stream()`, or
`torch.tpu.synchronize()` instead.

> [!NOTE] In TorchTPU's deferred execution model, the XLA compiler and PjRt
> runtime manage execution ordering and hardware concurrency automatically.
> Streams are not needed.
