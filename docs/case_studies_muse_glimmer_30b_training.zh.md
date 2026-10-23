# Muse Glimmer 30B：训练 { #muse-glimmer-30b-training }

在本案例研究中，我们将介绍独立的 Muse Glimmer 30B 示例。该示例演示了如何在 4 个 Ironwood TPU 上使用 FSDP，为完整模型运行一个简单的训练循环。

如需了解如何在您的 GKE（Google Kubernetes Engine）集群中的 TPU 上运行此示例，请参阅示例 [README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop)。本文则介绍其中的一些实现选择。

此示例的重点是简单易读。它只在一台配备 4 个 Ironwood TPU 的主机上运行。显然，这不是真实的预训练规模，但我们的目标是提供一个可运行的示例，让您能够以较低的成本运行、迭代和研究。

这段纯 PyTorch（没有自定义 kernel）代码在序列长度为 5120 时，每个 rank 的训练速度约为 2400 到 2800 TPS（总计 19k 到 22k TPS）。在今后的更新中，我们计划介绍更多在 TPU 上运行时的优化方法和性能最佳实践。

## 模型定义 { #model-definition }
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) 采用相当标准的 Transformer 架构。它由 52 层因果自注意力和前馈（FFN）块组成。模型每隔 4 层使用一次全注意力，其他所有层都使用滑动窗口注意力（SWA）。

模型代码完全独立地包含在 [model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py) 中（旋转位置编码位于 [rope.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/rope.py) 中）。

配置在 [model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py) 中指定：

```python
class Config:
  # General model config
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  # Details
  swa_size: int = 2048
  # ... omitted ...

  # Perf evaluation config
  batch_size: int = 8  # Local (per rank) batch size
  seq_len: int = 5120
```

模型主干的定义可以归结为：

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
self.embed_norm = nn.RMSNorm(
    (config.dmodel,), eps=config.rmsnorm_eps, elementwise_affine=False
)
self.layers = nn.ModuleList(
    DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
)
self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)
```

请注意，这里没有模型头（也称为“LM 头”）。在训练时，通常最好将其与损失函数放在一起，请参阅下文关于分块融合损失的部分。

## 逐层 torch.compile { #per-layer-torchcompile }

TorchTPU 在使用 `torch.compile` 时性能最佳，这得益于 XLA 编译器能够执行各种优化，例如算子融合、内存优化，以及自动将计算与通信重叠。

此示例展示了如何对每个模型层分别应用 `torch.compile`（而不是对整个模型应用）。两种方式都可以正常工作，但整个模型的编译可能需要相当长的时间，而且至少对于此模型来说，在性能方面，编译范围超出单个层并不会带来任何好处。

```python
class MuseGlimmer30B(nn.Module):

  def __init__(self, config: Config):
    # ... omitted ...
    self.layers = nn.ModuleList(
        DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
    )

  def apply_torch_compile(self, backend):
    self.embed_norm = torch.compile(self.embed_norm, dynamic=False, backend=backend)
    self.layers = nn.ModuleList(
        torch.compile(layer, dynamic=False, backend=backend)
        for layer in self.layers
    )
    self.norm = torch.compile(self.norm, dynamic=False, backend=backend)
```

这里的 `dynamic=False` 表示没有动态形状（在我们的训练过程中，所有形状都是静态的），而 `backend` 变量只是为了更方便地在 `"tpu"` 和 `"inductor"`（用于 CUDA）后端之间切换。

在内部，TorchTPU 会缓存已编译的层；由于此模型中只有两种层（全注意力层和 SWA 层），这会大大加快编译过程。

## 激活检查点 { #activation-checkpointing }

模型权重会占用约 60 GiB（bf16）的 HBM，可以轻松放入单个 Ironwood TPU TensorCore（具有 96 GiB）中。

但是，在训练期间，我们还需要为梯度和优化器状态（例如 AdamW 的动量和方差）分配内存。更重要的是，autograd 功能所需的中间激活值很容易成为内存用量的主要部分。激活检查点（也称为重计算，即 rematerialization）是解决后一个问题的一种非常常见的技术。

由于我们是逐层编译模型的，因此以相同的粒度应用激活检查点是很自然的。我们使用标准的 [`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint) 功能来实现这一点：

```python
class DecoderBlock(nn.Module):
  # ... omitted ...

  def forward(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    return torch.utils.checkpoint.checkpoint(
        self.forward_impl,
        h,
        rope_sin_cos,
        use_reentrant=False,
    )

  def forward_impl(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(self.input_layernorm(h), rope_sin_cos)
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h
```

> **Note:** 激活检查点（AC）是在模型层（`DecoderBlock` 模块）的 `forward` 方法内部应用的，而 `torch.compile` 是在外部、对层本身应用的，这个顺序很重要。您可以将 AC 保留在层内部，并对整个模型应用 `torch.compile`，这样也能正常工作。但不能将二者调换，例如，如果您以相同的方式应用 AC，但在内部分别对注意力和 FFN 子层应用 `torch.compile`，就无法按预期工作。

在 TorchTPU 中，`torch.compile` 与 AC 配合得特别好：在（全局）反向传播期间，每一层都会执行一次小规模的联合前向+反向传播（由于重计算），而且由于 TPU 后端编译器（XLA）能够看到完整的联合前向+反向计算图，它可以进行额外的优化，甚至减少中间张量的内存用量。

## “朴素”FSDP { #naive-fsdp }

在此示例中，我们希望演示进行 FSDP（完全分片数据并行）训练的最简单方式，即直接使用原生 PyTorch 集合通信算子。

> **Tip:** 如需查看对此方法更简单的介绍，请参阅我们的[手动 FSDP 示例](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual)。

我们通过手动对所有注意力和 FFN 投影权重进行分片来实现 FSDP，例如：

```python
# QKV projections are fully sharded:
self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
```

> **Note:** PyTorch `nn.Linear` 层以转置的形式存储其权重矩阵，因此我们在这里对第 2 个维度进行分片。权重的实际形状将是 `[num_q_heads * dh / world_size, dm]`。这一点很重要，因为 PyTorch 的 all-gather / reduce-scatter 集合通信是在第 1 个维度上进行的。

然后，在前向传播期间，我们使用 PyTorch 的函数式集合通信在使用权重之前对其进行 gather：

```python
# Functional collectives
import torch.distributed._functional_collectives as func_col

# ... all-gather:
q_proj_w = func_col.all_gather_single(self.q_proj.weight, gather_dim=0, group=self.pg)

# ... use full weight directly:
q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
```

同样的思路也应用于所有其他投影（注意力的 Q/K/V/out/gate，以及 FFN 的 up/gate/down）。

关键在于，`func_col.all_gather_single` 等函数式集合通信已在 PyTorch autograd 引擎中注册，因此在 `.backward()` 传播期间，我们会自动获得相应的 reduce-scatter 集合通信（也就是说，我们不需要自己编写它们）。

您会注意到，此实现相对朴素，因为它不会在运行当前层时尝试预取下一层的权重。这难道不会导致 TPU TensorCore 停顿/空闲，等待集合通信完成吗？

由于我们对整个层应用了 `torch.compile`，TorchTPU 的编译器（XLA）能够对集合通信进行重新排序，并将其卸载到 SparseCore 上异步运行，无论是在前向传播中还是在联合前向+反向传播中都是如此。在本例中，这已足以实现近乎完美的通信/计算重叠。

## 分块融合 LM 头 + 交叉熵损失 { #chunkwise-fused-lm-head-cross-entropy-loss }

此示例默认使用的本地批次大小为 8，序列长度为 5120。如果以朴素的方式计算交叉熵训练损失，我们就需要具体化一个形状为 `[8, 5120, 202048]` 的中间 logits 张量，其中最后一个数字是模型的词汇表大小。如果使用 bf16，仅这一项就需要约 16 GiB 的 HBM 内存，过于浪费。

这个问题的一个常见解决方案是分块计算损失，将词汇表矩阵权重的矩阵乘法与交叉熵损失融合在一起。一种方式是沿词汇表维度分块（这需要 online-softmax，难以用纯 PyTorch 表达），另一种方式是沿序列维度分块。我们在这里采用的是后者：

```python
class MuseGlimmerChunkwiseFusedLMHeadLoss(nn.Module):
  # ... omitted ...

  for i in range(0, h.shape[0], chunk_size):
    h_chunk.data.copy_(h[i : (i + chunk_size)])
    tgt_chunk.data.copy_(tgt[i : (i + chunk_size)])

    self.fused_loss_fn_fwd_bwd(
        h_chunk, tgt_chunk, self.lm_head.weight, alpha, cap, loss_scaler
    )

    h_chunk_grads.append(h_chunk.grad)
    h_chunk.grad = None

  return torch.cat(h_chunk_grads, dim=0).view(bsz, seq_len, dm)
```

`fused_loss_fn_fwd_bwd` 指的是经过 `torch.compile` 编译的静态函数版本，该函数会在单个 token 块上计算融合的词汇表矩阵乘法 + 交叉熵损失 + 反向传播：

```python
@staticmethod
def _chunk_fused_lmhead_loss_fwd_bwd(
    h, tgt, lm_head_w, alpha, cap, loss_scaler
):
  logits = F.tanh(F.linear(h, lm_head_w) * (alpha / cap)) * cap
  loss = F.cross_entropy(input=logits, target=tgt, reduction="sum")
  loss *= loss_scaler
  # Note: in recent versions of PyTorch, .backward is no longer a graph break!
  loss.backward()
  return
```

为了从这种方式中获得最大收益，我们甚至会在块上执行反向传播。在较新版本的 PyTorch 中，调用 `loss.backward()` 不再会导致图中断，这使 TorchTPU 的编译器（XLA）能够执行更多优化和融合。

## 此示例中缺少的内容 { #whats-missing-in-this-example }

此示例仅展示了基本的训练循环，并且为了清晰起见，省略了任何实际训练设置中都很重要的各种细节：

 - 数据加载、分词
 - 模型检查点的加载/保存
 - 作业健康状况监控、收敛诊断
 - 容错

对于使用 TorchTPU 的真实生产级训练设置，我们建议您参阅 [TorchTitan](https://github.com/google-pytorch/torchtitan/)。
