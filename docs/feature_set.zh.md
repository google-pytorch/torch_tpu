# TorchTPU 支持的功能集和未来工作 { #torchtpu-supported-feature-set-and-future-work }

## 当前支持的功能集 { #current-supported-feature-set }

下表反映了截至首次开源发布时我们当前的功能覆盖范围。

为了帮助您评估 TorchTPU 是否适合您的工作负载，我们将功能分为两个层级：

*   **已验证**：具有端到端持续测试、自动化 CI/CD 覆盖，并在标准模型架构上经过基准测试验证的功能。
*   **实验性**：早期阶段的实现、新兴 API 或原型功能。这些功能可以正常使用，可供您尽早提供反馈，但可能会根据社区意见调整行为或更改 API。

> **Note:** TorchTPU 正在以开放的方式快速发展。如需了解最新状态或请求特定功能，请访问我们的 [GitHub](https://github.com/google-pytorch/torch_tpu)。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **已验证** | **实验性** |
| ----- | ----- | ----- |
| **PyTorch 社区参与** | <ul><li>TorchTPU、vLLM、SGLang、TorchTitan、TorchRec 的开源发布</li><li>接受外部 PR</li><li>文档/教程</li><li>CI/CD</li><li>Colab</li></ul> | <ul><li>用于 ATEN 迁移的 TPU Agent</li></ul> |
| **TPU 支持** | <ul><li>v5e、v5p、v6e、v7</li></ul> |  |
| **基本功能** | <ul><li>易于迁移</li><li>eager 与编译混合执行</li><li>前向 \+ 反向传播</li><li>Torch.profile</li><li>动态算子（重新编译）</li></ul> | <ul><li>SymInt</li><li>Jagged 张量</li></ul> |
| **模型架构** | <ul><li>Transformer（Dense \+ MoE）：Qwen、Llama、Muse Glimmer、Gemma</li><li>扩散模型：WAN、FLUX</li><li>CNN</li></ul> | <ul><li>GLM、Kimi、DeepSeek、GPT OSS 等众多模型（持续优化中）</li><li>Mamba</li><li>LSTM/GRU</li><li>DLRM</li></ul> |
| **Kernel 支持** | <ul><li>Pallas</li><li>Tokamax</li></ul> | <ul><li>Helion</li></ul> |
| **量化** | <ul><li>全张量 FP8/FP4</li><li>逐通道 FP8</li></ul> |  |
| **分布式** | <ul><li>最优网格（mesh）</li><li>SimpleFSDP</li><li>检查点</li><li>激活值卸载</li><li>多切片，最多 8K 芯片</li></ul> | <ul><li>FSDPv2（性能较低）</li><li>MPMD（性能较低）</li><li>TorchComm</li></ul> |
| **生态系统集成** | <ul><li>Hugging Face</li><li>Ray</li><li>vLLM / SGLang</li><li>TorchTitan</li><li>TorchRec</li></ul> |  |
| **生产就绪的解决方案** | <ul><li>推理（多主机、P/D 分离）</li><li>微调（最多约 8K 芯片）</li></ul> |  |
<!-- mdformat on -->

## 未来工作 { #future-work }

本部分概述我们未来的开发方向和探索计划，并非承诺的发布时间表。

作为一个社区优先的开源项目，我们有意让开发优先级响应社区反馈和开发者的采用情况。随着我们了解哪些工作负载对您最重要，功能、时间表和架构方案都可能会有所变化。

我们将接下来的工作分为两大方向：

*   **近期重点**：核心团队当前优先处理的领域，包括正在进行的设计、原型开发和性能强化。
*   **未来探索**：我们正在积极研究的长期架构探索、潜在的生态系统集成，以及社区请求的待办事项。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **近期重点** | **未来探索** |
| ----- | ----- | ----- |
| **PyTorch 社区参与** | <ul><li>更多文档/教程/博客</li></ul> | <ul><li>用于 kernel 迁移的 TPU Agent</li></ul> |
| **TPU 支持** | <ul><li>v8i、v8t</li></ul> |  |
| **基本功能** | <ul><li>Stream</li><li>图捕获</li><li>启动速度快的 eager 模式</li><li>动态算子（无需重新编译）</li></ul> |  |
| **模型架构** | <ul><li>新模型首日支持</li></ul> |  |
| **量化** | <ul><li>借助 torchao 实现分块 FP8/FP4（在 TPUv8t 上达到峰值性能）</li></ul> |  |
| **分布式** | <ul><li>FSDPv2（具备高性能）</li><li>MPMD（具备高性能）</li><li>超过 8K 芯片的训练</li><li>多层级检查点</li><li>Goodput</li></ul> | <ul><li>远超 8K 芯片的训练</li></ul>  |
| **生态系统集成** | <ul><li>VeRL</li></ul> |  |
| **生产就绪的解决方案** | <ul><li>RL</li></ul> | <ul><li>预训练</li></ul> |
<!-- mdformat on -->
