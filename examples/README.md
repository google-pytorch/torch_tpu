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

For `torch.compile()` usage, refer to [fsdp/manual](fsdp/manual/README.md).

For distributed training examples, refer to `examples/tensor_parallel/` and
`examples/fsdp/`.

## Available Examples

-   [simple_pytorch](simple_pytorch/README.md): Minimal single-device PyTorch
    example on TPU, plus a reference single-device GKE manifest.
-   [tensor_parallel/manual](tensor_parallel/manual/README.md): Manual
    Megatron-LM style Tensor Parallelism using explicit collective operations,
    with single-host and multi-host GKE manifests.
-   [tensor_parallel/dtensor_wrapped](tensor_parallel/dtensor_wrapped/README.md):
    Tensor Parallelism using PyTorch `DTensor` and `parallelize_module`
    wrappers.
-   [fsdp/manual](fsdp/manual/README.md): Manual Fully Sharded Data Parallel
    ("DIY FSDP") using explicit functional collectives, activation
    checkpointing, and `torch.compile`.
-   [fsdp/fsdp_to_simple_fsdp](fsdp/fsdp_to_simple_fsdp/README.md): Side-by-side
    comparison and migration guide from PyTorch FSDPv2 (`fully_shard`) to
    `SimpleFSDP` (`data_parallel`).
-   [FLUX_optimization](FLUX_optimization/README.md): Step-by-step tutorial on
    optimizing the FLUX diffusion transformer for TPU, including Pallas kernels.
-   [muse_glimmer_30b/llm_inference](muse_glimmer_30b/llm_inference/README.md):
    Single-device LLM prefill and decode with `torch.compile()` and a dynamic KV
    cache.
-   [muse_glimmer_30b/pure_pytorch_fsdp_trainloop](muse_glimmer_30b/pure_pytorch_fsdp_trainloop/README.md):
    Pure PyTorch FSDP training loop for a 30B LLM with `torch.compile` and
    activation checkpointing.

## Benchmarking

-   TODO

## Contributing

If you'd like to contribute your own example or fix a bug please make sure to
take a look at [CONTRIBUTING.md](../CONTRIBUTING.md).
