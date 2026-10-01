# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Introduction to TorchTPU

    TorchTPU is an open-source, native PyTorch backend for [Google Cloud TPUs](https://cloud.google.com/tpu), developed in collaboration between Google and Meta. With TorchTPU, you can run existing PyTorch models on TPU hardware with minimal code changes and scale training and inference workloads from a single chip to multi-slice TPU pods.

    ## Why use TorchTPU

    Google TPUs are application-specific integrated circuits (ASICs) purpose-built for the matrix and vector operations that power deep learning, offering high throughput, energy efficiency, and performance per dollar at scale. While TPUs have historically been paired with the TensorFlow and JAX ecosystems, TorchTPU brings native TPU support directly to the [PyTorch ecosystem](https://github.com/google-pytorch/torch_tpu) around three design principles:

    * **Native PyTorch citizenship:** TorchTPU integrates directly with PyTorch with no JAX runtime dependencies. It preserves standard PyTorch tensor view and in-place mutation semantics, adopts PyTorch's one-process-per-device execution model, and supports `torch.compile` through TorchDynamo.
    * **Out-of-the-box usability:** Porting a workload from a GPU to a TPU starts with changing `device="cuda"` to `device="tpu"` and `backend="nccl"` to `backend="tpu_dist"`. Your unmodified PyTorch model runs immediately in eager mode, with clear error messages when diagnosing runtime or compiler issues.
    * **Production-scale performance:** Beyond op-by-op eager execution, TorchTPU introduces a deferred-fusion eager mode, a TorchDynamo-to-XLA compiler pipeline, bounded dynamism to avoid shape-driven recompilations, and custom kernel APIs to help you reach peak hardware utilization.

    ## Key features

    * **Comprehensive operator and model coverage:** Supports approximately 450 core ATen operators (both forward and backward passes) and is continuously validated across nearly 5,000 popular Hugging Face models spanning large language models (Llama, Qwen, Gemma, DeepSeek, Mistral), vision (ViT, ResNet, ConvNeXt, DINOv2), audio (Whisper, Wav2Vec2), and diffusion (FLUX, Stable Diffusion).
    * **Three eager execution modes:**
      * **Strict (default):** Dispatches one ATen operator at a time asynchronously to the TPU, matching standard PyTorch eager behavior on GPUs.
      * **Debug (`TORCH_TPU_LAUNCH_BLOCKING=1`):** Runs each operator synchronously with the CPU and enables bounds checking to simplify debugging.
      * **DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`):** Defers and fuses sequences of ATen operators up to a materialization point (such as `.cpu()` or `.item()`), reducing High Bandwidth Memory (HBM) traffic and accelerating eager execution.
    * **Graph compilation with `torch.compile`:** Registers the `"tpu"` TorchDynamo backend (`torch.compile(model, backend="tpu", dynamic=False)`) to lower entire PyTorch FX graphs into [StableHLO](https://openxla.org/stablehlo/spec) and compile them with the [OpenXLA compiler](https://openxla.org/xla). Active support for bounded dynamism (`torch._dynamo.mark_dynamic`) lets compiled graphs handle varying batch sizes or sequence lengths up to a specified bound without recompiling. (Note that as a temporary limitation, we must pass `dynamic=False` to torch.compile() calls because bounded dynamism (discussed below) is required to support SymInts, but it is not enabled by default yet.)
    * **Multi-tier compilation cache:** Eliminates redundant XLA compilations across three cache tiers:
      * **Tier 1:** Process-local in-memory cache (always enabled).
      * **Tier 2:** Host-local shared memory for cross-process reuse in multi-device jobs.
      * **Tier 3:** Persistent remote cache (such as Cloud Storage) for reuse across hosts and job restarts.
    * **Native PyTorch distributed support:** Implements TPU collectives at the `torch.distributed` Process Group (`c10d::Backend`) level via `backend="tpu_dist"`, supporting `DistributedDataParallel` (DDP), `FullyShardedDataParallel` (FSDP2 and SimpleFSDP), and `DTensor`. You can also use `torch.tpu.topology_aware_mesh` to align parallelism dimensions with physical TPU interconnects.
    * **Custom TPU kernels:** Author custom kernels in [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) (exposed to PyTorch via `torch.tpu.pallas.jax_op`, including direct inter-chip RDMA collectives), write tile-based kernels using the experimental [Helion](https://pytorch.org/blog/helion/) DSL, or import optimized TPU kernels from Tokamax.
    * **Ecosystem integrations:** Works with standard PyTorch tooling and serving/training frameworks, including `torch.profiler` (with TensorBoard and XProf), vLLM, SGLang, TorchTitan, Ray, and TorchRec.

    ## Understand the TorchTPU architecture

    When you run PyTorch on a TPU VM with TorchTPU installed, your code interacts with specialized hardware units through the OpenXLA software stack:

    * **TPU hardware components:**
      * **TensorCores:** Each TensorCore contains scalar, vector (VPU), and Matrix Multiply Units (MXUs) built from systolic arrays that natively multiply `bfloat16` inputs and accumulate in `float32`.
      * **SparseCores:** Dedicated processors for sparse operations (such as embedding lookups, gather, and scatter) and collective communication. You can pin a `torch.tpu.Stream` to a SparseCore (`torch.tpu.Stream(pinned="sparse")`) to overlap collective communication with dense TensorCore math.
      * **High Bandwidth Memory (HBM):** High-capacity on-package memory shared across operations on the chip.
      * **Hierarchical interconnects:** High-speed optical/copper Inter-Chip Interconnect (ICI) links connect chips within a TPU slice in a 3D torus or twisted torus topology, while Data Center Network (DCN) links connect multiple slices across pods.
    * **Software lowering pipeline:**
      1. PyTorch's dispatcher intercepts tensor and collective operations on the `"tpu"` device.
      2. TorchTPU translates PyTorch stride views and in-place writes into functional operations via automatic *view decomposition* and *view inversion*.
      3. TorchTPU lowers ATen and `c10d` operations into **StableHLO** (or **Mosaic** for custom Pallas kernels).
      4. A C++ [PjRT client](https://openxla.org/xla/pjrt/cpp_api_overview) compiles the module with XLA into a TPU executable, caches it, and dispatches asynchronous execution on the TPU.

    ## Run a computation on a TPU

    Because TorchTPU uses PyTorch's autoload mechanism, importing `torch` in an environment where `torch_tpu` is installed automatically registers the `"tpu"` device. Run the following cell to initialize the TPU device, allocate tensors on the TPU, perform a matrix multiplication, and copy the result back to the CPU:
    """)
  return


@app.cell
def _():
  import torch

  device = torch.device("tpu")
  x = torch.randn((4, 4), device=device)
  y = torch.ones((4, 4), device=device)

  result = torch.matmul(x, y).cpu()
  print(f"Executed on device: {x.device}")
  print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
  return


if __name__ == "__main__":
  app.run()
