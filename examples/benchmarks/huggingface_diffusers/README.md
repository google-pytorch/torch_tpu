# HuggingFace Diffusers Examples and Tests

This directory contains examples and smoke tests for running HuggingFace
Diffusers models.

## Running Smoke Tests

Check the BUILD file for targets. Targets are configured for CUDA, TPU, and CPU

**GPU:**

```sh
blaze test -c opt --config cuda //examples/benchmarks/huggingface_diffusers:sdxl_test_cuda
```

**TPU:**

```sh
blaze test -c opt //examples/benchmarks/huggingface_diffusers:sdxl_test_tpu
```

**CPU:**

```sh
blaze test -c opt //examples/benchmarks/huggingface_diffusers:sdxl_test_cpu
```

### Inference Smoke Tests

Inference tests that the entire HF DiffusionPipeline completes without crash.
This is run with real weights.

### Training Smoke Tests

Training tests 3 backprop steps of a heavily miniaturized version of the main
denoiser model with dummy input data.

## Model Configs

Model configurations from HuggingFace are stored in the `model_configs/`
subdirectory. Model directories contain only configs and no weights.

### Downloading New Model Configs

To download model configs from HuggingFace Hub to the `model_configs/`
directory, use the `import_hf_configs` script.

1.  Build the script:

    ```sh
    blaze build //examples/benchmarks/huggingface_diffusers:import_hf_configs
    ```

2.  Run the binary from within your workspace root, specifying the model ID:

    ```sh
    ./blaze-bin/third_party/py/torch_tpu/examples/benchmarks/huggingface_diffusers/import_hf_configs --model_id="black-forest-labs/FLUX.1-schnell"
    ```

    This will download the configuration files for the specified model and place
    them into `model_configs/black-forest-labs/FLUX.1-schnell/`.
