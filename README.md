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
*   Examples for various models (Llama, Qwen, ResNet, DLRM, minGPT, UNet).
*   Integration with `torch.compile()` for graph mode execution.
*   Utilities for debugging and benchmarking.

## User Guide and Interactive Notebooks

A user guide is available at http://google-pytorch.github.io/torch_tpu/ . This
is a static version of the interactive notebooks. They are available in the docs
folder.

The notebooks are Marimo (similar to Jupyter). They can be run with `marimo edit
file.py` and then you can access them in your browser. You will need to be
running this from a TPU VM, a v6e GCE instance is easiest.

## Installation

The same instructions are available in the user guide linked above.

First use python 3.12 virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

or, with `uv`

```bash
uv venv .venv
source .venv/bin/activate
```

### Install via pip

Authenticate first (for `uv` prepend `uv` to all `pip` commands):

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
bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel
```

All OSS Bazel builds default to optimized mode (`-c opt`) automatically via
`.bazelrc`. Unoptimized builds (`-c fastbuild`, `-c dbg`) can degrade
performance and are not used for wheel builds. If you are developing or
debugging C++ code and explicitly want an unoptimized wheel, pass
`--//:allow_unoptimized_wheel=True`:

```bash
bazel build -c dbg --config=wheel_common //ci/wheel:torch_tpu_wheel \
    --//:allow_unoptimized_wheel=True
```

`--config=wheel_common` is required, and the build fails at build time without
it. The config sets `--//:wheel_build=True`, which routes the shared XLA/MLIR
backend through the `torch_version`-reset transition so that pywrap factors it
into a single `libxla_base.so`; built without the flag, every per-version common
ships its own copy of the backend and the resulting wheel aborts on `import
torch` with duplicate static registrations.

`wheel_common` also turns on the Remote Build Execution (RBE) remote cache and
remote executor. If you do not have RBE credentials, append `--config=no_rbe` to
strip those flags back out:

```bash
bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe
```

The order matters here: `--config=no_rbe` has to come after
`--config=wheel_common`, because the later config is the one that wins for the
remote-cache and remote-executor flags they both set.

Install wheel via:

```bash
cd <path_to_repo>
python3.12 -m venv .venv; source .venv/bin/activate
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

## VSCode `clangd` Setup

For C++ code navigation, it is recommended to use VSCode with
[`clangd` extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd).
Relying on a
[compilation database](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
generated from Bazel action graph, [clangd](https://clangd.llvm.org/) is able to
enrich the editor with various smart features, including code completions, go-to
definitions, *etc.*

1.  Install VSCode extensions via UI or the following command:

    ```sh
    code --install-extension llvm-vs-code-extensions.vscode-clangd
    # Uninstall Microsoft C++ extensions to avoid interference.
    code --uninstall-extension ms-vscode.cpptools
    ```

1.  Generate the initial `compile_commands.json`:

    ```sh
    ./setup_clangd.py
    ```

It may take a while (> 30 minutes) for the first run to make a full build, but
reruns will be incremental and cheap. If you have run `bazel build` recently,
generated files already exist so you can pass `--no-build` to skip waiting for
heavy builds (note that on a fresh workspace, skipping the build may prevent
`clangd` from resolving generated files). See more options in the
[script](./setup_clangd.py) file-level comments.

Alternatively, you can invoke the script via `nox` (arguments after `--` are
forwarded):

```sh
nox -s refresh_compile_commands -- --no-build
nox -s refresh_compile_commands -- //torch_tpu/ops/...
```

## Contributing

We welcome contributions to the PyTorch on TPU project! Please see our
[CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute,
including signing our Contributor License Agreement (CLA) and code review
processes.

## License

This project is licensed under the Apache 2.0 License. See the
[LICENSE](LICENSE) file for details.
