# TorchTPU

## Description

This project provides a PyTorch backend for Google's Tensor Processing Units
(TPUs), enabling users to run PyTorch models and operations on TPU devices. It
includes custom ATen kernels for various PyTorch operations, a compilation cache
for optimizing execution, and examples for distributed training and model
inference.

## Features

*   PyTorch ATen kernel implementation for TPU.
*   Compilation cache for optimized execution of XLA computations.
*   Support for distributed training (Data Parallelism, Tensor Parallelism).
*   Examples for various models (DeepSeek, Llama, Qwen, ResNet, DLRM, minGPT,
    UNet).
*   Integration with `torch.compile()` for graph mode execution.
*   Utilities for debugging and benchmarking.

## Installation

First use python 3.12 virtual environment

```bash
mkdir wheel; cd wheel
python3.12 -m venv venv; source venv/bin/activate
```

### Install via pip

Authenticate first:

```bash
pip install keyrings.google-artifactregistry-auth
gcloud auth login
gcloud auth application-default login
```

Install from latest nightly:

```bash
pip install --pre --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" torch_tpu
```

### Build from source

Wheels can be built via:

```bash
bazel build //ci/wheel:torch_tpu_wheel --config=wheel
```

(Alternative) For local build, use bazel command below instead:

```bash
bazel build //ci/wheel:torch_tpu_wheel --config=no_rbe --config=wheel
```

Install wheel via:

```bash
cd <path_to_repo>
mkdir wheel; cd wheel
python3.12 -m venv venv; source venv/bin/activate
# The index-url includes the CPU version of torch at higher priority than the CUDA version.
pip install ../bazel-bin/ci/wheel/*.whl --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"
```

Note that this command will install the CUDA version of `torch`; to get the
CPU-only build, use the `--index-url` parameter from the section on installing
nightlies, or install the CPU build separately.

## Getting Started

Some pointers to get you started:

*   [PyTorch: Official PyTorch documentation](https://github.com/pytorch/pytorch)
*   [Tutorials: easy to understand PyTorch code using TorchTPU](./examples/tutorials)
*   [Examples: get you started with sample PyTorch models on TPU](./examples/README.md)

## Dependencies

The project uses Bazel for dependency management. Key Python dependencies are:

TBD

## Bazel setup

This project uses Bazel for building and dependency management.

To install Bazel, we actually install Bazelisk, which is a transparent wrapper
that handles downloading and installing Bazel itself.

There are several ways to install Bazelisk. Here are some common ways:

*   If you have Go installed:

    ```
    go install github.com/bazelbuild/bazelisk@latest
    ```

*   Download the binary directly:

    ```
    wget -O ~/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
    ```

See the
[Bazelisk README](https://github.com/bazelbuild/bazelisk/blob/master/README.md)
for more install instructions

However you install it, add it's location to `PATH`.

To verify the install, run `bazel info` in the repo's root directory:

```
bazel info
```

It should print information about the repo without errors.

## Contributing

We welcome contributions to the PyTorch on TPU project! Please see our
[CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute,
including signing our Contributor License Agreement (CLA) and code review
processes.

## License

This project is licensed under the Apache 2.0 License. See the
[LICENSE](LICENSE) file for details.
