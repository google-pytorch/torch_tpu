# TorchTPU 常见问题解答 { #torchtpu-faq }

## PyTorch/XLA 与 TorchTPU 有什么区别？ { #what-is-the-difference-between-pytorchxla-and-torchtpu }

Pytorch/XLA 过去一直充当桥接框架，高度依赖 JAX 运行时组件和专用 API。TorchTPU 为 [Google Cloud TPU](https://cloud.google.com/tpu) 提供原生 Pytorch 后端，不依赖任何 JAX 运行时组件。现有 PyTorch 工作负载只需极少的代码更改（例如 `.to("cuda")` → `.to("tpu")`）即可直接运行。

## TorchAX 与 TorchTPU 有什么区别？ { #what-is-the-difference-between-torchax-and-torchtpu }

TorchAX（也称为 torch_xla2）是一种在 Python 层拦截 PyTorch 操作并将其降级（lower）为 XLA（通常经由 JAX）的实现。虽然 TorchAX 已大量用于推理基准测试和原型开发，但 TorchTPU 是新的官方标准，可在 TPU 上提供即插即用的 PyTorch 原生体验。

## PyTorch/XLA 的现状如何？ { #what-is-happening-to-pytorchxla }

PyTorch/XLA 已弃用，由现已公开的 [TorchTPU](https://github.com/google-pytorch/torch_tpu) 取代。

## TorchTPU 会取代 JAX 吗？ { #is-torchtpu-replacing-jax }

不会。TorchTPU 旨在补充 TPU 生态系统，为 Cloud TPU 提供原生 PyTorch 入口。借助 TorchTPU，拥有现有 PyTorch 代码库的开发者无需重写或只需极少改写即可直接在 TPU 上运行，无需用 JAX 重写模型，也无需为 GPU 和 TPU 基础设施分别维护代码库。

*   **何时考虑使用 TorchTPU？** 如果您已有 PyTorch 代码库，希望在不重构的情况下实现多硬件（GPU/TPU）可移植性，正在使用 PyTorch 原生框架（例如 TorchTitan、vLLM、SGLang、Hugging Face），或者希望快速从 GPU 迁移到 TPU，就可以考虑使用 TorchTPU。

## 如何与 TorchTPU 开发者联系？ { #what-is-the-best-way-for-me-to-connect-with-torchtpu-developers }

您可以发送电子邮件至 [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com) 分享反馈，我们也欢迎您直接向代码库贡献代码：[https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu)。您还可以在 [discuss.pytorch.org](https://discuss.pytorch.org/) 上使用“TPU”标签。

## 会有社区交流论坛吗？ { #will-there-be-any-community-engagement-forums }

会。社区交流对我们非常重要，我们会公布此类论坛的详细信息。此外，我们计划每月举办答疑时间（office hours）。

## TorchTPU 为 kernel 开发提供哪些支持？ { #what-support-does-torchtpu-provide-for-kernel-development }

请参阅 [TorchTPU 自定义 kernel 支持](./custom_kernels.md)。

## 目前支持哪些 TPU 世代？未来硬件支持有何计划？ { #which-tpu-generations-are-actively-supported-whats-the-plan-for-future-hardware-support }

TorchTPU 目前支持 v5e、v5p、Trillium（TPU v6e）和 Ironwood（TPU v7x）。作为路线图的一部分，我们正在对 TPU v8 及未来的芯片进行验证和优化。

## 在哪里可以找到 TorchTPU 支持的功能和未来路线图？ { #where-can-we-find-torchtpus-supported-features-future-roadmap }

请参阅[支持的功能](./feature_set.md)，了解当前的功能集和计划推出的功能集。

## 是否有可以简化上手过程的 agent？ { #is-there-an-agent-to-make-my-onboarding-easier }

请查看 TPU agent 套件（实验性）：[https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

TPU Agent Suite 是一个开源插件，提供专用子 agent，可针对 Cloud TPU v6e 和 v7x 上的 PyTorch 模型自动完成上手配置、诊断和推理性能调优。

## 我能否将 TorchTPU 与我们的自定义推理服务方案搭配使用？ { #can-i-use-torchtpu-with-our-custom-serving-solution }

可以。TorchTPU 可与 vLLM 和 SGLang 等业界标准推理服务引擎开箱即用地集成，同时，自定义或内部推理服务方案也可以通过原生 PyTorch 执行 API 和更底层的编译器后端与 TorchTPU 集成。如需高级编排，客户还可以借助 llm-d 在 GKE 上部署自定义推理服务环境。
