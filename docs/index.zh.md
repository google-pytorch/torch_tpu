<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>讨论论坛</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>在 TPU 上使用 PyTorch</b></a>
</p>

## 简介 { #about }

TorchTPU 是一个面向 Google Cloud TPU 的开源原生 PyTorch 后端，由 Google 和 Meta 合作开发。借助 TorchTPU，您只需对代码做少量修改，就能在 TPU 硬件上运行现有的 PyTorch 模型，并将训练和推理工作负载从单个芯片扩展到多切片（multi-slice）TPU Pod。

## 您今天想做什么？ { #what-are-you-trying-to-do-today }

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> 入门**

    预配环境，了解基础架构，并运行您的第一个工作负载。

    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU 简介](./intro_to_torchtpu.md)：TorchTPU 的创建初衷、它提供的功能，以及它能如何帮助您。
    - [<!--mkdocs::octicons-arrow-right-24:--> 在虚拟机上开始使用 TorchTPU](./get_started_on_a_vm.md)：在 TPU 虚拟机上设置 TorchTPU。
    - [<!--mkdocs::octicons-arrow-right-24:--> 在 GKE 上开始使用 TorchTPU](./get_started_on_gke.md)：通过 GKE 使用 TorchTPU。
    - [<!--mkdocs::octicons-arrow-right-24:--> 支持的功能集和路线图](./feature_set.md)：当前支持的功能以及未来版本的计划。
    - [<!--mkdocs::octicons-arrow-right-24:--> API 参考](./api_reference.md)：当前版本的 TorchTPU API 和环境变量。
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU 常见问题解答](./faq.md)：关于 TorchTPU 的常见问题。

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> 深入了解架构和核心概念**

    深入了解内存布局、执行引擎和分布式执行的工作机制。

    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU 的工作原理](./how_torchtpu_works.md)：TorchTPU 如何自上而下地与 PyTorch 和 TPU 硬件集成。
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU 执行模型](./torchtpu_execution_model.md)：详细介绍 TorchTPU 如何执行 PyTorch 代码，以及它支持的不同执行模式。
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU 内存模型](./torchtpu_memory_model.md)：TorchTPU 如何应对 TPU 和 XLA 编译器处理内存的方式。
    - [<!--mkdocs::octicons-arrow-right-24:--> 分布式支持](./distributed_support.md)：TorchTPU 如何支持分布式工作负载。
    - [<!--mkdocs::octicons-arrow-right-24:--> 自定义 kernel 支持](./custom_kernels.md)：如何为 TorchTPU 编写自定义 kernel。
    - [<!--mkdocs::octicons-arrow-right-24:--> 动态形状](./dynamic_shapes.md)：对动态张量形状的支持。

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> 案例研究**

    通过实际示例了解如何使用 TorchTPU 解决现实问题。

    - [<!--mkdocs::octicons-arrow-right-24:--> FLUX 优化](./case_studies_flux.md)：示例演示如何在 FLUX 上识别并应用多项优化。
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B 推理](./case_studies_muse_glimmer_30b_inference.md)：使用真实权重以及静态/动态 KV 缓存进行迭代解码的示例。
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B 训练](./case_studies_muse_glimmer_30b_training.md)：在 Muse Glimmer 30B 的独立实现上运行的训练循环。
    - [<!--mkdocs::octicons-arrow-right-24:--> 使用 FSDP](./case_studies_fsdp.md)：在 TorchTPU 中使用 FSDP 的推荐方法。

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> 分析工作负载**

    - [<!--mkdocs::octicons-arrow-right-24:--> 数值精度和一致性](./numerical_precision.md)：在 TorchTPU 中控制精度。
    - [<!--mkdocs::octicons-arrow-right-24:--> 使用 XProf 分析性能](./performance_profiling_with_xprof.md)：如何将 XProf 用作 TorchTPU 的性能分析器。

<!--mkdocs:</div>-->
<!-- mdformat on -->
