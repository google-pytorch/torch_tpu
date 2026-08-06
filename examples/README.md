# TorchTPU Examples

`torch_tpu/examples` contains high-quality examples for using
[TorchTPU](https://github.com/google-pytorch/torch_tpu). Our goal is to provide
examples of varying complexity that incrementally demonstrate how to use
`torch_tpu`—from a "Hello World" introduction to scaling PyTorch workloads on
TPUs with eager and compile modes. The provided examples are intended to
demonstrate functionality and have not been optimized for TPU performance.

To initialize the TPU device and run a simple operation:

```python
import torch

device = torch.accelerator.current_accelerator()

# Create a tensor on TPU
x = torch.ones(2, 2, device=device)
print(f"Tensor on TPU: {x}")

# Perform an operation
y = x + x
print(f"Result on TPU: {y}")
```

For `torch.compile()` usage, refer to `compile/example/simple.py`.

For distributed training examples, refer to `examples/distributed/`.

For model inference examples, refer to `examples/`.

## Available Models

-   simple_nn

## Benchmarking

-   TODO

## Contributing

If you'd like to contribute your own example or fix a bug please make sure to
take a look at [CONTRIBUTING.md](../CONTRIBUTING.md).
