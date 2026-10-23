# Muse Glimmer 30B: treinamento { #muse-glimmer-30b-training }

Neste estudo de caso, vamos analisar nosso exemplo independente do Muse Glimmer
30B, que mostra um loop de treinamento simples para o modelo completo, usando
FSDP em 4 TPUs Ironwood.

Para saber como executar este exemplo em TPUs no seu cluster do GKE (Google
Kubernetes Engine), consulte o
[README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop)
do exemplo. Aqui, explicamos algumas das escolhas de implementação.

O foco deste exemplo é a simplicidade e a legibilidade. Ele é executado em um
único host com 4 TPUs Ironwood. Obviamente, essa não é uma escala realista de
pré-treinamento, mas nosso objetivo é oferecer um exemplo funcional que você
possa executar, modificar e estudar com baixo custo.

Este código em PyTorch puro (sem kernels personalizados) alcança uma velocidade
de treinamento de cerca de 2.400 a 2.800 TPS por rank (ou seja, de 19 mil a 22
mil TPS no total), com comprimento de sequência de 5120. Em atualizações
futuras, planejamos descrever mais otimizações e práticas recomendadas de
desempenho para execução em TPUs.

## Definição do modelo { #model-definition }
O [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) segue
uma arquitetura de transformer bastante padrão.
Ele tem 52 camadas de blocos de autoatenção causal e feed-forward (FFN).
A cada 4 camadas, uma usa atenção completa, e todas as outras usam atenção com
janela deslizante (SWA).

O código do modelo é totalmente independente e está em
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)
(e em
[rope.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/rope.py)
para a codificação posicional rotativa).

A configuração é especificada em
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py):

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

A definição do backbone do modelo se resume a:

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

Observe que a cabeça do modelo (também chamada de "LM head") não está aqui. No
treinamento, costuma ser melhor colocá-la junto com a função de loss. Consulte a
seção sobre loss fundida por partes mais abaixo.

## torch.compile por camada { #per-layer-torchcompile }

O TorchTPU tem o melhor desempenho com `torch.compile`, graças à capacidade do
compilador XLA de fazer várias otimizações, como fusão de operações, otimização
de memória e sobreposição automática de computação com comunicação.

Este exemplo mostra como aplicar `torch.compile` em cada camada do modelo
separadamente (em vez de no modelo inteiro). As duas abordagens funcionam bem,
mas a compilação do modelo inteiro pode demorar bem mais e, em termos de
desempenho, pelo menos neste modelo, não há benefício em compilar além do
escopo de uma única camada.

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

Aqui, `dynamic=False` significa que não há formatos dinâmicos (todos os formatos
são estáticos durante o treinamento), e a variável `backend` serve apenas para
facilitar a troca entre os backends `"tpu"` e `"inductor"` (para CUDA).

Internamente, o TorchTPU armazena em cache as camadas compiladas e, como há
apenas dois tipos de camada (completa e SWA) neste modelo, isso acelera muito o
processo de compilação.

## Activation checkpointing { #activation-checkpointing }

Os pesos do modelo ocupam cerca de 60 GiB (em bf16) de HBM e caberiam com
folga em um único TensorCore de uma TPU Ironwood (que tem 96 GiB).

No entanto, durante o treinamento, também precisamos de memória para os
gradientes e os estados do otimizador (por exemplo, o momentum e a variância do
AdamW). Mais importante ainda: o uso de memória pode ser facilmente dominado
pelas ativações intermediárias necessárias para o autograd. O activation
checkpointing (também chamado de rematerialização) é uma técnica muito comum
para resolver esse último problema.

Como compilamos o modelo por camada, é muito natural aplicar o activation
checkpointing com a mesma granularidade. Fazemos isso com a funcionalidade
padrão
[`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint):

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

> **Note:** o activation checkpointing (AC) é aplicado dentro do método
> `forward` da camada do modelo (módulo `DecoderBlock`), e o `torch.compile` é
> aplicado por fora, na própria camada. Essa ordem é importante.
> Você pode manter o AC dentro da camada e aplicar o `torch.compile` no modelo
> inteiro, e isso vai funcionar bem.
> Mas você não pode inverter os dois. Por exemplo, se você aplicar o AC da mesma
> forma, mas aplicar o `torch.compile` por dentro, separadamente nas subcamadas
> de atenção e FFN, isso não vai funcionar como esperado.

O `torch.compile` funciona especialmente bem com o AC no TorchTPU: durante o
backward pass (global), cada camada executa um pequeno forward+backward pass
conjunto (por causa da rematerialização) e, como o compilador do backend da TPU
(XLA) vê todo o grafo de computação conjunto de forward+backward, ele pode
fazer otimizações adicionais e até reduzir o uso de memória dos tensores
intermediários.

## FSDP "ingênuo" { #naive-fsdp }

Neste exemplo, queríamos mostrar a forma mais simples possível de fazer
treinamento com FSDP (fully-sharded data-parallel), usando diretamente as
operações coletivas nativas do PyTorch.

> **Tip:** para uma descrição ainda mais simples dessa abordagem, confira nosso
> [exemplo de FSDP manual](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual).

Fazemos o FSDP fragmentando manualmente todos os pesos de projeção da atenção e
da FFN, por exemplo:

```python
# QKV projections are fully sharded:
self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
```

> **Note:** a camada `nn.Linear` do PyTorch armazena a matriz de pesos
> transposta, e é por isso que fragmentamos na 2ª dimensão aqui. O formato real
> do peso será `[num_q_heads * dh / world_size, dm]`. Isso é importante porque
> as coletivas de all-gather / reduce-scatter do PyTorch funcionam na 1ª
> dimensão.

Depois, durante o forward pass, usamos as coletivas funcionais do PyTorch para
reunir os pesos antes de usá-los:

```python
# Functional collectives
import torch.distributed._functional_collectives as func_col

# ... all-gather:
q_proj_w = func_col.all_gather_single(self.q_proj.weight, gather_dim=0, group=self.pg)

# ... use full weight directly:
q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
```

A mesma ideia é aplicada a todas as outras projeções (Q/K/V/out/gate da
atenção e up/gate/down da FFN).

O ponto principal é que as coletivas funcionais, como
`func_col.all_gather_single`, são registradas no mecanismo de autograd do
PyTorch. Assim, durante o backward pass (`.backward()`), obtemos automaticamente as
coletivas de reduce-scatter correspondentes (ou seja, não precisamos
escrevê-las).

Você vai notar que essa implementação é relativamente ingênua, porque não tenta
fazer o prefetch dos pesos da próxima camada enquanto executa a camada atual.
Isso não deixaria os TensorCores da TPU parados/ociosos, esperando as coletivas
terminarem?

Como aplicamos o `torch.compile` na camada inteira, o compilador do TorchTPU
(XLA) consegue reordenar as coletivas e transferi-las para execução assíncrona
no SparseCore, tanto no forward quanto nos forward+backward passes conjuntos.
Neste caso, isso é suficiente para alcançar uma sobreposição quase perfeita
entre comunicação e computação.

## LM head e loss de entropia cruzada fundidas por partes { #chunkwise-fused-lm-head-cross-entropy-loss }

Por padrão, o exemplo usa um tamanho de batch local de 8 e um comprimento de
sequência de 5120. Se calculássemos a loss de entropia cruzada do treinamento
de forma ingênua, precisaríamos materializar um tensor intermediário de logits
com o formato `[8, 5120, 202048]`, em que o último número é o tamanho do
vocabulário do modelo. Em bf16, só isso já exigiria cerca de 16 GiB de memória
HBM, o que é um desperdício grande demais.

Uma solução comum para esse problema é calcular a loss por partes (chunks),
fundindo a matmul com a matriz de pesos do vocabulário e a loss de entropia
cruzada. Uma abordagem é dividir ao longo da dimensão do vocabulário (o que
exige softmax online e é difícil de expressar em PyTorch puro), e outra é
dividir ao longo da dimensão de sequência. Aqui, fazemos a segunda:

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

`fused_loss_fn_fwd_bwd` se refere à versão compilada com `torch.compile` da
função estática que calcula, de forma fundida, a matmul do vocabulário + a loss
de entropia cruzada + o backward pass em uma única parte dos tokens:

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

Para aproveitar ao máximo essa abordagem, executamos até o backward pass na
parte. Nas versões mais recentes do PyTorch, invocar `loss.backward()` não causa
mais uma quebra de grafo, o que permite que o compilador do TorchTPU (XLA) faça
mais otimizações e fusões.

## O que falta neste exemplo { #whats-missing-in-this-example }

Este exemplo mostra apenas o loop de treinamento básico e, para ficar mais
claro, omite vários detalhes importantes para qualquer configuração prática de
treinamento:

 - carregamento de dados e tokenização,
 - carregamento/salvamento de checkpoints do modelo,
 - monitoramento da integridade do job e diagnóstico de convergência,
 - tolerância a falhas.

Para uma configuração de treinamento realista e pronta para produção com o
TorchTPU, recomendamos conferir o
[TorchTitan](https://github.com/google-pytorch/torchtitan/).
