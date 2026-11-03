<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>Discussion Forum</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>PyTorch on TPU</b></a>
</p>

## About

TorchTPU is an open-source, native PyTorch backend for Google Cloud TPUs,
developed in collaboration between Google and Meta. With TorchTPU, you can run
existing PyTorch models on TPU hardware with minimal code changes and scale
training and inference workloads from a single chip to multi-slice TPU pods.

## What are you trying to do today?

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> Get started**

    Provision environments, understand fundamental architectures, and run your first workloads.

    - [<!--mkdocs::octicons-arrow-right-24:--> Introduction to TorchTPU](./intro_to_torchtpu.md): Why TorchTPU was created, what it provides, and how it can help you.
    - [<!--mkdocs::octicons-arrow-right-24:--> Get started with TorchTPU on a VM](./get_started_on_a_vm.md): Setup TorchTPU on a TPU VM.
    - [<!--mkdocs::octicons-arrow-right-24:--> Get started with TorchTPU on GKE](./get_started_on_gke.md): Use TorchTPU through GKE.
    - [<!--mkdocs::octicons-arrow-right-24:--> Supported feature set and roadmap](./feature_set.md): Currently supported features and plans for future releases.
    - [<!--mkdocs::octicons-arrow-right-24:--> API reference](./api_reference.md): The TorchTPU API and environment variables for the current release.
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU FAQ](./faq.md): Frequently asked questions about TorchTPU.

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> In-depth architecture & core concepts**

    Deep-dive into the mechanics of memory layouts, execution engines, and distributed execution.

    - [<!--mkdocs::octicons-arrow-right-24:--> How TorchTPU works](./how_torchtpu_works.md): How TorchTPU integrates with PyTorch and the TPU hardware from top to bottom.
    - [<!--mkdocs::octicons-arrow-right-24:--> The TorchTPU execution model](./torchtpu_execution_model.md): In depth detail of how TorchTPU executes PyTorch code and the different execution modes it supports.
    - [<!--mkdocs::octicons-arrow-right-24:--> The TorchTPU memory model](./torchtpu_memory_model.md): How TorchTPU handles the way the TPU and XLA compiler deal with memory.
    - [<!--mkdocs::octicons-arrow-right-24:--> Distributed support](./distributed_support.md): How TorchTPU supports distributed workloads.
    - [<!--mkdocs::octicons-arrow-right-24:--> Custom kernel support](./custom_kernels.md): How to write custom kernels for TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Dynamic shapes](./dynamic_shapes.md): Support for dynamic tensor shapes.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Case studies**

    Explore practical examples of how to use TorchTPU to solve real-world problems.

    - [<!--mkdocs::octicons-arrow-right-24:--> FLUX Optimization](./case_studies_flux.md): Example demonstrating how to identify and apply multiple optimizations on FLUX.
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B inference](./case_studies_muse_glimmer_30b_inference.md): Example of iterative decoding using real weights with static/dynamic KV cache.
    - [<!--mkdocs::octicons-arrow-right-24:--> DeepSeek-V4-Flash inference](./case_studies_deepseek_v4_flash_base_inference.md): End-to-end zero-graph-break inference with MLA, MoE Sinkhorn routing, and functional fallbacks.
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B training](./case_studies_muse_glimmer_30b_training.md): A training loop on a standalone implementation of Muse Glimmer 30B.
    - [<!--mkdocs::octicons-arrow-right-24:--> Using FSDP](./case_studies_fsdp.md): The recommended approach to using FSDP for TorchTPU.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Analyze your workload**

    - [<!--mkdocs::octicons-arrow-right-24:--> Numerical precision and parity](./numerical_precision.md): Controlling precision in TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Profile performance with XProf](./performance_profiling_with_xprof.md): How to use XProf as a TorchTPU profiler.

<!--mkdocs:</div>-->
<!-- mdformat on -->
