# Qwen-20B Image Fine-Tuning and Inference on TPU VM

This directory contains the **Qwen-20B Image Prototype** for end-to-end LoRA
fine-tuning and multi-chip inference of the **Qwen-20B Image** diffusion model
using `torch_tpu`.

--------------------------------------------------------------------------------

## Purpose & Need for this Prototype

**Qwen-20B Image** is a 20-billion parameter multimodal text-to-image diffusion
transformer. Due to its large parameter count and architectural complexity
(incorporating 3D Rotary Position Embeddings and dual image/text attention
blocks), standard single-card setups quickly run out of memory or suffer from
severe compute bottlenecks.

This prototype demonstrates how to leverage **Google Cloud TPU VMs** and
`torch_tpu` to achieve high-performance fine-tuning and real-time image
generation. It serves as a reference implementation for scaling large diffusion
transformers on Google Cloud TPU hardware.

--------------------------------------------------------------------------------

## Target Execution Environment

*   **Hardware**: **Google Cloud TPU VM** (specifically **TPU v6e-4** or **TPU
    v6e-8** instances).
*   **Runtime**: `torch_tpu` (`PJRT_DEVICE="TPU"`,
    `TPU_ACCELERATOR_TYPE="v6e-4"`).

--------------------------------------------------------------------------------

## Architectural & Execution Strategies

### 1. Fine-Tuning Strategy: Pre-encoded Dataset Caching + LoRA ($r=128$)

*   **Pre-encoded Dataset Caching**:
    *   *Why*: Encoding images via VAE and generating text embeddings via text
        encoders on every training epoch creates massive CPU-TPU data transfer
        bottlenecks.
    *   *How*: Pre-computes VAE latents and prompt embeddings once and caches
        them to disk/RAM. This removes I/O overhead and allows the TorchTPU
        training loop to run at peak throughput (~1.8 steps/sec).
*   **Low-Rank Adaptation (LoRA)**:
    *   *Why*: Full parameter fine-tuning of a 20B model requires over 80GB of
        memory just for AdamW optimizer states, exceeding single-chip memory
        limits.
    *   *How*: Injects low-rank adapters ($r=128$, $\alpha=16.0$) into image
        attention projections (`to_q`, `to_k`, `to_v`, `to_out`),
        cross-attention layers (`add_q_proj`, `add_k_proj`, `add_v_proj`,
        `to_add_out`), and MLP blocks (`img_mlp_net_2`). This reduces trainable
        parameters from 20B to ~288 tensors (~115MB), drastically lowering
        memory footprint while retaining high image quality.
*   **3D Spatial Rotary Position Embeddings (3D RoPE)**:
    *   Supports Qwen's multi-dimensional spatial patch positioning directly
        within TorchTPU execution.

--------------------------------------------------------------------------------

### 2. Inference Strategy: SPMD Mesh Sharding across TPU v6e Chips

*   **Model Size vs. TPU Memory**:
    *   *Why*: Qwen 20B weights in `bfloat16` consume ~40GB of memory, exceeding
        the High Bandwidth Memory (HBM) of a single TPU v6e chip (32GB HBM).
*   **SPMD (Single Program, Multiple Data) Mesh Sharding**:
    *   *How*: Shards the transformer weight matrices across all 4 TPU chips in
        a `v6e-4` instance using a 1x4 mesh (`mesh_shape=(1, 4)` across
        `tpu:0`..`tpu:3`).
    *   *Why*: Distributes memory consumption evenly (~10GB per chip) and
        parallelizes matrix multiplications across Tensor Processing Cores via
        high-speed Inter-Chip Interconnects (ICI), completely eliminating
        Out-Of-Memory errors without relying on slow disk paging.
*   **Host CPU Offloading for Auxiliary Components**:
    *   Text encoders and VAE decoder are kept on host CPU RAM, reserving 100%
        of TPU HBM for the 20B Diffusion Transformer.

--------------------------------------------------------------------------------

## Directory Structure

```text
torch_tpu/prototypes/qwen_image/
├── README.md               # Quickstart guide & architectural strategy docs
├── train_torch_tpu.py      # TPU LoRA fine-tuning trainer
├── infer_qwen20b_tpu.py    # TPU inference execution engine
├── qwen_image_tpu.py       # Pure-TPU Qwen 20B pipeline wrapper
├── config.py               # Dataclass configurations (TPU, Model, Generation)
├── tpu_sharding.py         # Multi-device TPU sharding & SPMD manager
└── utils.py                # Dataset loaders & PEFT adapter converter
```

--------------------------------------------------------------------------------

## Environment Setup

Run the following inside your TPU VM:

```bash
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE="v6e-4"
export PJRT_DEVICE="TPU"
```

--------------------------------------------------------------------------------

## 1. Training (`train_torch_tpu.py`)

### Basic Command

```bash
python3 train_torch_tpu.py
```

### Full Command with All Options

```bash
python3 train_torch_tpu.py \
    --model_id "Qwen/Qwen-Image" \
    --dataset_name "HK83/Anime_Faces" \
    --dataset_cache "./dataset_cache.pt" \
    --max_samples 50000 \
    --encoding_batch_size 512 \
    --encoding_num_workers 512 \
    --epochs 1.0 \
    --steps 2000 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --output_dir "./output_qwen_image_tpu"
```

### CLI Arguments Reference

Argument                  | Type    | Default                     | Description
:------------------------ | :------ | :-------------------------- | :----------
`--model_id`              | `str`   | `"Qwen/Qwen-Image"`         | Base model HuggingFace ID or path
`--dataset_name`          | `str`   | `"HK83/Anime_Faces"`        | HuggingFace dataset name or local directory
`--dataset_cache`         | `str`   | `"./dataset_cache.pt"`      | Pre-encoded dataset cache tensor file path
`--max_samples`           | `int`   | `1000`                      | Maximum samples to encode and train on
`--encoding_batch_size`   | `int`   | `16`                        | Batch size for dataset pre-encoding phase
`--encoding_num_workers`  | `int`   | `8`                         | DataLoader worker threads for pre-encoding
`--epochs`                | `float` | `1.0`                       | Total training epochs
`--steps`                 | `int`   | `None`                      | Total training steps (overrides `--epochs`)
`--lr`, `--learning_rate` | `float` | `1e-4`                      | Learning rate
`--weight_decay`          | `float` | `1e-2`                      | AdamW weight decay
`--output_dir`            | `str`   | `"./output_qwen_image_tpu"` | Output directory for adapter weights

--------------------------------------------------------------------------------

## 2. Inference (`infer_qwen20b_tpu.py`)

### Basic Command

```bash
python3 infer_qwen20b_tpu.py
```

### Full Command with All Options

```bash
python3 infer_qwen20b_tpu.py \
    --model_id "Qwen/Qwen-Image" \
    --lora_weights_path "./output_qwen_image_tpu" \
    --output_dir "./qwen_image_tpu_results" \
    --custom_prompts "1girl, solo, anime girl face, blue eyes, long silver hair, detailed portrait;1boy, solo, anime boy face, red hair, heroic smile, portrait" \
    --resolution 512 \
    --num_inference_steps 20 \
    --guidance_scale 7.5 \
    --seed 42 \
    --sharding_strategy "spmd" \
    --num_devices 4 \
    --tpu_accelerator_type "v6e-4"
```

### CLI Arguments Reference

Argument                 | Type    | Default                      | Description
:----------------------- | :------ | :--------------------------- | :----------
`--model_id`             | `str`   | `"Qwen/Qwen-Image"`          | Base model HuggingFace ID or path
`--lora_weights_path`    | `str`   | `"./output_qwen_image_tpu"`  | Path to trained PEFT LoRA adapter directory
`--output_dir`           | `str`   | `"./qwen_image_tpu_results"` | Directory to save generated output images
`--custom_prompts`       | `str`   | `None`                       | Semicolon-separated list of custom prompts
`--resolution`           | `int`   | `512`                        | Image height and width in pixels
`--num_inference_steps`  | `int`   | `20`                         | Number of diffusion denoising steps
`--guidance_scale`       | `float` | `7.5`                        | Classifier-free guidance scale
`--seed`                 | `int`   | `42`                         | Random seed for generation
`--sharding_strategy`    | `str`   | `"spmd"`                     | TPU sharding strategy (`spmd`, `layer`, `offload`, `auto`, `single`)
`--num_devices`          | `int`   | `4`                          | Number of TPU chips/devices to utilize
`--tpu_accelerator_type` | `str`   | `"v6e-4"`                    | GCP TPU accelerator type (e.g. `v6e-1`, `v6e-4`, `v6e-8`)
`--worker_hostnames`     | `str`   | `"127.0.0.1"`                | TPU worker hostnames/IPs
`--use_float32`          | `flag`  | `False`                      | Use float32 precision instead of default bfloat16
