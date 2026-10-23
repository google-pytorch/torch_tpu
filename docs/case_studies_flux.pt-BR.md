# Otimização do FLUX { #flux-optimization }

<!-- Link to colab and to generated notebook -->

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_flux.ipynb)

Este estudo de caso mostra como pegamos uma implementação de referência em
PyTorch do `DoubleStreamBlock` do transformer de difusão
[FLUX](https://github.com/black-forest-labs/flux) e deixamos uma etapa de
treinamento **1,48x mais rápida** na Cloud TPU com o TorchTPU, com comprimento
de sequência de 32.768.

Esta página é uma versão resumida que acompanha o tutorial completo. Ela
destaca o conceito principal por trás de cada otimização com trechos curtos e
independentes. Para ver o código completo do modelo, os benchmarks, os traces
do XProf e os testes, consulte:

*   [Tutorial de otimização do FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md):
    explicação completa de cada marco, incluindo traces do XProf e instruções
    para reproduzir todos os benchmarks.
*   [Como criar kernels Pallas personalizados para PyTorch](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md):
    explicação passo a passo do kernel Pallas SplashAttention usado no Marco 4.

A implementação se baseia na
[implementação do FLUX no TorchTitan](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)
e no repositório
[black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/).

Se você ainda não conhece os exemplos, talvez queira se familiarizar com a
execução de exemplos no GKE
[por meio dos nossos outros documentos](./get_started_on_gke.md).

## Visão geral dos resultados { #results-at-a-glance }

Tempo médio da etapa de treinamento (forward + loss + backward + etapa do
otimizador) para um único `DoubleStreamBlock`:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Configuração | Tempo médio da etapa | Aceleração em relação à linha de base |
| :--- | ---: | ---: |
| Referência de linha de base (eager) | 178,61 ms | 1,00x |
| Somente Pallas SplashAttention (eager) | 149,81 ms | 1,19x |
| Somente views sem cópia (eager) | 178,30 ms | 1,00x |
| Somente `torch.compile` | 159,33 ms | 1,12x |
| Views sem cópia + RoPE fundido + Pallas SplashAttention (eager) | 156,73 ms | 1,14x |
| Views sem cópia + `torch.compile` | 159,08 ms | 1,12x |
| Views sem cópia + RoPE fundido + `torch.compile` | 153,60 ms | 1,16x |
| **Views sem cópia + RoPE fundido + Pallas SplashAttention + `torch.compile`** | **122,51 ms** | **1,46x** |
<!-- mdformat on -->

**Configuração da medição:**

-   **Hardware:** um único chip TPU v6e (32 GiB de HBM).
-   **Software:** Python 3.14, PyTorch 2.15 (pré-lançamento), JAX 0.12, einops
    0.8.2 e TorchTPU compilado a partir do código-fonte.
-   **Formato do modelo:** `batch_size=2`, `hidden_size=1024`, `num_heads=8`
    (`head_dim=128`), `img_seq_len=32256`, `txt_seq_len=512` (comprimento total
    da sequência de 32.768).
-   **Precisão:** parâmetros e entradas do bloco em bfloat16.
-   **Tamanhos de bloco do SplashAttention:** 2048 para `block_q`, `block_kv` e
    os tiles do backward de dK/dV.
-   **Medição de tempo:** 3 etapas de aquecimento seguidas de 10 etapas
    cronometradas. As etapas cronometradas são despachadas em sequência, e
    `torch.accelerator.synchronize()` é executado uma vez após a última, de
    modo que o tempo médio da etapa (tempo total dividido pelo número de
    etapas) inclui a execução no dispositivo. O profiler fica desativado
    durante as etapas cronometradas.

Com esse comprimento de sequência, a atenção domina a etapa, então o
SplashAttention e a compilação trazem os maiores ganhos. Com sequências mais
curtas, a compilação tem mais impacto (cerca de 3x com comprimento de sequência
de 640), e o kernel de atenção, menos.

As lições se aplicam à maioria dos modelos transformer em TPU:

1.  **Evite cópias de layout ocultas**: prefira views sem cópia (`.view()`,
    `torch.unbind()`) a reorganizações que materializam novos buffers na HBM.
2.  **Armazene menos, calcule mais**: expresse os embeddings posicionais como
    cálculos compactos elemento a elemento em vez de matrizes materializadas.
3.  **Compile o bloco inteiro**: `torch.compile()` permite que o XLA funda
    cadeias de operações elemento a elemento e elimine idas e voltas
    intermediárias à HBM.
4.  **Use kernels personalizados quando necessário**: kernels Pallas dão
    controle explícito sobre o tiling no chip para operações limitadas por
    memória, como a atenção com sequências longas.

### Como ler esta página { #how-to-read-this-page }

Cada marco é apresentado como um par de seções:

-   **Antes (referência)**: a implementação original e o gargalo dela.
-   **Depois (otimizado)**: a implementação otimizada, seguida de uma
    verificação de que ela produz o mesmo resultado que a referência.

As células de código são identificadas por um comentário `# BEFORE (reference)`
ou `# AFTER
(optimized)` correspondente na primeira linha.

**Pré-requisitos:** a célula de referência do Marco 1 usa
[einops](https://github.com/arogozhnikov/einops) (`pip install einops`).

```python
import torch

device = torch.device("tpu")
print(f"Running on {device}.")
```

## Marco 1: views de tensor sem cópia { #milestone-1-zero-copy-tensor-views }

No `DoubleStreamBlock`, os fluxos de imagem e de texto projetam estados ocultos
de formato `(B, L, 3 * H * D)` em queries, keys e values. A forma como essa
projeção é dividida em Q, K e V determina se a TPU precisa copiar dados na HBM
durante o treinamento.

### Antes (referência): `einops.rearrange` { #before-reference-einopsrearrange }

A implementação de referência move o eixo `3` (Q/K/V) para a frente:

```python
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)
```

No forward pass, `einops.rearrange` retorna views com strides da projeção,
então nenhum dado é movido ainda. O custo aparece no **backward pass**: para
encaminhar os gradientes de volta pela reordenação dos eixos, o autograd
materializa o gradiente em um novo buffer, o que é executado na TPU como uma
cópia extra de HBM para HBM em cada bloco de atenção (visível como uma
operação `copy` no XProf).

```python
# BEFORE (reference): einops.rearrange moves the Q/K/V axis to the front.
import einops

num_heads, head_dim = 4, 32
qkv = torch.randn(2, 16, 3 * num_heads * head_dim, device=device)

q_before, _k, _v = einops.rearrange(
    qkv, "B L (K H D) -> K B L H D", K=3, H=num_heads
)
print(f"q shape: {tuple(q_before.shape)}")
```

### Depois (otimizado): `.view()` + `torch.unbind()` { #after-optimized-view-torchunbind }

A implementação otimizada mantém o eixo `3` adjacente a `(H, D)` e o divide
com `torch.unbind`. Assim, o backward pass empilha os gradientes de Q, K e V
diretamente no layout original, e nenhuma cópia extra é necessária. Na tabela
de resultados, o efeito no tempo da etapa é pequeno (**de 178,61 ms para
178,30 ms** no modo eager, **de 159,33 ms para 159,08 ms** compilado) porque a
cópia é uma pequena fração da etapa, mas ela é eliminada sem nenhum custo.

> **Note:** as células deste marco mostram apenas que as duas abordagens
> produzem os mesmos Q, K e V. No forward pass, as duas retornam as mesmas
> views com strides, então as saídas parecem idênticas. A cópia extra na HBM
> causada por `einops.rearrange` só ocorre no backward pass e aparece em um
> trace do XProf de uma etapa de treinamento completa, e não na saída da
> célula. Consulte o
> [Marco 1 do Tutorial de otimização do FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-1)
> para ver os traces.

```python
# AFTER (optimized): keep Q/K/V adjacent to (H, D) and unbind.
_b, _l, _ = qkv.shape
q_after, _k, _v = torch.unbind(
    qkv.view(_b, _l, 3, num_heads, head_dim), dim=2
)
print(f"q shape: {tuple(q_after.shape)}")
print(f"Matches reference: {torch.equal(q_after.cpu(), q_before.cpu())}")
```

## Marco 2: RoPE algébrico fundido { #milestone-2-fused-algebraic-rope }

O FLUX aplica Rotary Position Embeddings (RoPE) multieixo a queries e keys. As
células abaixo implementam as duas versões para um único eixo de RoPE.

### Antes (referência): matrizes de rotação 2x2 { #before-reference-2x2-rotation-matrices }

A implementação de referência materializa uma matriz de rotação `2x2` completa
`[[cos,
-sin], [sin, cos]]` para cada par de frequências e a aplica como um produto
matriz-vetor sobre tensores 6D de query/key remodelados. Isso armazena `cos` e
`sin` duas vezes, e o empilhamento e a remodelagem extras aparecem como grandes
blocos de cópia no XProf.

```python
# BEFORE (reference): materialize 2x2 rotation matrices, then contract.
def rope_ref(pos, dim, theta=10000):
  """Returns rotation matrices of shape (B, S, D/2, 2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  out = torch.stack(
      [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
      dim=-1,
  )
  return out.reshape(*out.shape[:-1], 2, 2).float()

def apply_rope_ref(xq, xk, freqs_cis):
  """6D matrix-vector contraction against the rotation matrices."""
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return (
      xq_out.reshape(*xq.shape).type_as(xq),
      xk_out.reshape(*xk.shape).type_as(xk),
  )

_b, _s, _h, rope_head_dim = 2, 64, 4, 32
positions = (
    torch.arange(_s, dtype=torch.float32, device=device)
    .unsqueeze(0)
    .expand(_b, _s)
)
rope_xq = torch.randn(_b, _s, _h, rope_head_dim, device=device)
rope_xk = torch.randn(_b, _s, _h, rope_head_dim, device=device)

# Add a broadcast head dimension: (B, S, 1, D/2, 2, 2).
_freqs_ref = rope_ref(positions, rope_head_dim).unsqueeze(2)
q_rope_ref, k_rope_ref = apply_rope_ref(rope_xq, rope_xk, _freqs_ref)

print(f"Rotation matrix freqs shape: {tuple(_freqs_ref.shape)}")
```

### Depois (otimizado): pares `[cos, sin]` compactos + rotação elemento a elemento { #after-optimized-compact-cos-sin-pairs-elementwise-rotation }

A implementação otimizada armazena apenas pares `[cos, sin]` compactos e aplica
a rotação diretamente com cálculos elemento a elemento:

```python
# (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
xq_out_0 = xq_0 * cos - xq_1 * sin
xq_out_1 = xq_0 * sin + xq_1 * cos
```

Isso reduz pela metade o espaço ocupado pelo embedding posicional, é mapeado
para instruções de multiplicação e adição fundidas (fused multiply-add) e se
funde com as operações de projeção e atenção ao redor durante a compilação.
Somado às views sem cópia e ao `torch.compile`, ele reduz o tempo da etapa
**de 159,08 ms para 153,60 ms (3,4%)**.

Como as duas versões realizam as mesmas multiplicações e adições sobre os
mesmos operandos, elas produzem resultados idênticos. A célula abaixo executa a
versão otimizada com as mesmas entradas e a compara com a referência:

> **Note:** estas células mostram o tensor de frequências menor e que as duas
> versões produzem resultados idênticos. A aceleração só aparece quando o RoPE
> é executado dentro do `DoubleStreamBlock` compilado, onde a rotação elemento
> a elemento se funde com as operações ao redor. Consulte o
> [Marco 2 do Tutorial de otimização do FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-2)
> para ver os traces do XProf.

```python
# AFTER (optimized): store compact [cos, sin] pairs, rotate elementwise.
def rope(pos, dim, theta=10000):
  """Returns compact [cos, sin] pairs of shape (B, S, D/2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  return torch.stack([torch.cos(out).float(), torch.sin(out).float()], dim=-1)

def apply_rope(xq, xk, freqs_cis):
  """Direct 2D algebraic rotation with elementwise math."""
  cos = freqs_cis[..., 0]
  sin = freqs_cis[..., 1]
  b, s, h, d = xq.shape
  xq_ = xq.float().view(b, s, h, d // 2, 2)
  xk_ = xk.float().view(b, s, h, d // 2, 2)
  xq_0, xq_1 = xq_[..., 0], xq_[..., 1]
  xk_0, xk_1 = xk_[..., 0], xk_[..., 1]
  xq_out = torch.stack(
      [xq_0 * cos - xq_1 * sin, xq_0 * sin + xq_1 * cos], dim=-1
  ).view(b, s, h, d)
  xk_out = torch.stack(
      [xk_0 * cos - xk_1 * sin, xk_0 * sin + xk_1 * cos], dim=-1
  ).view(b, s, h, d)
  return xq_out.type_as(xq), xk_out.type_as(xk)

# Add a broadcast head dimension: (B, S, 1, D/2, 2).
freqs = rope(positions, rope_head_dim).unsqueeze(2)
_q_new, _k_new = apply_rope(rope_xq, rope_xk, freqs)

print(f"Compact freqs shape:         {tuple(freqs.shape)}")
print(
    "Max |q_ref - q_new|:        "
    f" {(q_rope_ref.cpu() - _q_new.cpu()).abs().max().item():.3e}"
)
print(
    "Max |k_ref - k_new|:        "
    f" {(k_rope_ref.cpu() - _k_new.cpu()).abs().max().item():.3e}"
)
torch.testing.assert_close(
    _q_new.cpu(), q_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_new.cpu(), k_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
```

## Marco 3: compilação XLA do bloco inteiro { #milestone-3-whole-block-xla-compilation }

### Antes (referência): execução eager { #before-reference-eager-execution }

No modo eager, cada operação do PyTorch inicia um programa de TPU
independente, então as ativações intermediárias fazem idas e voltas pela HBM e
o host costuma ficar parado no despacho. Na tabela de resultados, o
`DoubleStreamBlock` de linha de base leva **178,61 ms** por etapa no modo
eager.

A célula abaixo executa o RoPE fundido do Marco 2 no modo eager, operação por
operação:

```python
# BEFORE (reference): eager execution, one TPU program per op.
q_eager, k_eager = apply_rope(rope_xq, rope_xk, freqs)
print("Ran fused RoPE eagerly.")
```

### Depois (otimizado): `torch.compile()` { #after-optimized-torchcompile }

Compilar o `DoubleStreamBlock` inteiro com `torch.compile()` o converte
(lowering) em um único grafo XLA, o que permite que o compilador funda operações
elemento a elemento, normalizações e adições residuais:

```python
block = DoubleStreamBlock(
    hidden_size=1024,
    num_heads=8,
    mlp_ratio=4.0,
    config=OptimizationConfig(
        use_zero_copy_view=True,
        use_fused_rope=True,
        compile=True,
    ),
)
compiled_block = block.maybe_compile()  # Wraps the block in torch.compile().
```

A compilação leva o tempo da etapa **de 178,61 ms para 159,33 ms (1,12x)**. O
ganho é modesto com comprimento de sequência de 32.768 porque a atenção domina
a etapa. Com comprimento de sequência de 640, a compilação é cerca de 3x mais
rápida.

O mesmo padrão se aplica a qualquer função. A célula abaixo compila o RoPE
fundido e verifica se ele corresponde à execução eager:

> **Note:** estas células apenas verificam que a compilação não altera o
> resultado. A fusão e a aceleração aparecem no XProf e nas medições de tempo
> da etapa do `DoubleStreamBlock` completo, e não na saída da célula. Consulte
> o
> [Marco 3 do Tutorial de otimização do FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-3)
> para comparar os traces eager e compilados.

```python
# AFTER (optimized): compile into a single fused XLA graph.
compiled_apply_rope = torch.compile(apply_rope, backend="tpu")
_q_compiled, _k_compiled = compiled_apply_rope(rope_xq, rope_xk, freqs)

torch.testing.assert_close(
    _q_compiled.cpu(), q_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_compiled.cpu(), k_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
print("Compiled fused RoPE matches eager execution.")
```

## Marco 4: kernel Pallas SplashAttention personalizado { #milestone-4-custom-pallas-splashattention-kernel }

Com sequências longas (por exemplo, `S = 2048` ou `S = 4096`), a atenção
conjunta completa passa a ser limitada pela memória. Este marco é mostrado
apenas como trechos de código. Consulte a
[explicação do kernel Pallas](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
para ver a implementação completa e executável.

### Antes (referência): SDPA padrão do PyTorch { #before-reference-standard-pytorch-sdpa }

```python
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

A atenção sem tiling materializa uma matriz de pontuações `O(S^2)` na HBM.

### Depois (otimizado): Pallas SplashAttention { #after-optimized-pallas-splashattention }

```python
attn = splash_sdpa(q, k, v, is_causal=False)  # Drop-in replacement.
```

`splash_sdpa` é um kernel SplashAttention personalizado em
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) que calcula a
atenção em tiles no chip, com base na
[implementação do SplashAttention no TorchTitan](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py).

O kernel é exposto ao PyTorch com `torch.tpu.pallas.jax_op`. As funções de
forward e backward são anotadas com tipos `jax.Array`, a partir dos quais
`jax_op` infere o schema do operador personalizado do PyTorch. O forward pass
salva o resíduo `logsumexp` do softmax para que o kernel de backward não
precise executar novamente a computação do forward:

```python
torch_fwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_fwd_name}", splash_fn)
torch_bwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_bwd_name}", splash_bwd_fn)


def setup_context(ctx, inputs, output):
  q, k, v = inputs
  out, lse = output
  ctx.save_for_backward(q, k, v, out, lse)


def backward(ctx, grad_out, grad_lse):
  del grad_lse
  q, k, v, out, lse = ctx.saved_tensors
  return torch_bwd_fn(q, k, v, out, lse, grad_out)


torch_fwd_fn.register_autograd(backward, setup_context=setup_context)
```

> **Important:** **O JAX não é executado em tempo de execução.** O Pallas é
> distribuído dentro do pacote `jax`, então o kernel é escrito com a sintaxe
> do `jax`. No entanto, `torch.tpu.pallas.jax_op` usa o JAX apenas uma vez, no
> momento do trace, com placeholders abstratos de formato/dtype para exportar o
> kernel como um módulo MLIR. Durante o treinamento e a inferência, tanto no
> modo eager quanto com `torch.compile`, o kernel é executado diretamente nos
> buffers de TPU de `torch.Tensor`.

**Resultados**: na tabela de resultados, o SplashAttention sozinho é **1,19x**
mais rápido que a linha de base eager (de 178,61 ms para 149,81 ms). Com as
views sem cópia e o RoPE fundido ativados, ele é mais rápido que o SDPA padrão
tanto no modo eager (**de 182,61 ms para 156,73 ms, 1,17x**) quanto no modo
compilado (**de 153,60 ms para 122,51 ms, 1,25x**).

Esses resultados dependem de configurar o kernel para a carga de trabalho. Com
os blocos padrão de 512 tokens, o SplashAttention foi cerca de 10% mais lento
que o SDPA padrão com compilação no mesmo comprimento de sequência, e as
entradas em float32 foram ainda mais lentas. Faça benchmarks de alguns
tamanhos de bloco para o seu comprimento de sequência, execute o kernel em
bfloat16 e crie um perfil antes de partir para um kernel personalizado.

**Saiba mais:**

*   **Exemplo de Pallas SplashAttention (usado neste estudo de caso)**:
    [`splash_attention.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention.py)
    contém a integração completa do kernel, e
    [`splash_attention_test.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention_test.py)
    verifica a paridade do forward e dos gradientes em relação ao SDPA do
    PyTorch. A
    [explicação do kernel](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
    explica passo a passo os tamanhos de bloco, as máscaras, o registro no
    autograd e os testes.
*   **Noções básicas de Pallas**:
    [Kernels personalizados com Pallas](./custom_kernels.md) apresenta como
    escrever kernels Pallas com o TorchTPU.
