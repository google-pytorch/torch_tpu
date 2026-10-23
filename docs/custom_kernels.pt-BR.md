# Kernels personalizados no TorchTPU { #torchtpu-custom-kernels }

O TorchTPU acelera nativamente os operadores padrão do framework PyTorch (como
[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)),
mas talvez você precise integrar kernels personalizados para acelerar operações
específicas do modelo, fundir operações para reduzir a sobrecarga de largura de
banda de memória ou implementar arquiteturas novas que ainda não são cobertas
pelos operadores padrão do PyTorch.

## Como escolher uma abordagem de integração { #choosing-an-integration-approach }

O TorchTPU oferece três formas de usar kernels personalizados na TPU, de acordo
com seu caso de uso:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Abordagem | Status | Ideal para |
| :---- | :---- | :---- |
| **Tokamax** | Pronto para produção | Aceleração imediata de blocos básicos comuns de LLM/MoE, sem escrever código de kernel. |
| **Pallas (`torch.tpu.pallas.jax_op`)** | Pronto para produção | Escrever kernels de TPU sob medida e de desempenho máximo em JAX/Pallas e chamá-los a partir do PyTorch. |
| **Helion** | Experimental | Escrever kernels portáveis entre aceleradores (por exemplo, GPU e TPU) com uma DSL nativa do PyTorch. |
<!-- mdformat on -->

## 1. Como usar kernels prontos do Tokamax { #1-using-pre-built-kernels-from-tokamax }

Antes de escrever um kernel personalizado do zero, verifique se o operador está
disponível no [Tokamax](http://openxla.org/tokamax), uma biblioteca de kernels
de TPU de alto desempenho com interoperabilidade nativa com o PyTorch.

### Quando usar { #when-to-use }

Use o Tokamax quando seu modelo precisar de blocos básicos padrão de alto
desempenho, como:

*   **Mixture-of-Experts (MoE):** multiplicação de matrizes agrupada, gathers e
    scatters de tokens.
*   **Variantes de atenção:** paged attention, Gated DeltaNet, Kimi Delta
    Attention e Multi-head Latent Attention (MLA).
*   **Seleção:** cálculo otimizado de `TopK`.

### Como funciona { #how-it-works }

Os kernels do Tokamax são escritos em JAX/Pallas e contam com ajuste automático
integrado (como a busca automática de tamanho de tile), benchmarks de baixa
sobrecarga com o XProf e testes de regressão em várias gerações de TPU.
Internamente, o Tokamax usa `torch.tpu.pallas.jax_op` para cuidar da
interoperabilidade entre frameworks de forma transparente e expõe APIs nativas
do PyTorch diretamente para seu modelo.

## 2. Como escrever kernels personalizados em Pallas { #2-writing-custom-pallas-kernels }

Se você precisa de uma operação sob medida que não é coberta pelo PyTorch nem
pelo Tokamax, pode escrever um kernel de TPU em
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) e registrá-lo como um
operador do PyTorch com `torch.tpu.pallas.jax_op`.

### Como integrar um kernel Pallas { #how-to-integrate-a-pallas-kernel }

1.  **Defina o kernel Pallas** usando referências de memória de entrada e saída
    (`Ref`s).
2.  **Encapsule o kernel em uma função JAX** usando `pl.pallas_call` para
    especificar os formatos e os dtypes da saída.
3.  **Registre a função JAX no TorchTPU** com `torch.tpu.pallas.jax_op`.

```python
import jax
from jax.experimental import pallas as pl
import torch

# 1. Define the Pallas kernel
def add_vectors_pallas_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y

# 2. Wrap the Pallas kernel into a JAX function
def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x, y)

# 3. Wrap it for PyTorch using the functional interface
add_vectors_fn = torch.tpu.pallas.jax_op(
    "pallas::add_vectors",
    add_vectors_jax,
)

# 4. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors_fn(x, y)
```

## 3. Como escrever kernels portáveis com Helion (experimental) { #3-writing-portable-kernels-with-helion-experimental }

Se você quer que seus kernels sejam portáveis entre aceleradores (por exemplo,
TPU e GPU), pode usar o [Helion](https://helionlang.com/), uma DSL nativa do
PyTorch para escrever kernels.

> **Note:** no momento, o suporte ao Helion no TorchTPU é **experimental**. Para
> cargas de trabalho de produção, use o Tokamax ou `torch.tpu.pallas.jax_op`.

### Como escrever um kernel Helion { #how-to-write-a-helion-kernel }

Decore sua função com `@helion.kernel`. Use operações padrão do PyTorch para a
configuração no host e `hl.tile` para definir o loop de execução no dispositivo
sobre tiles particionados:

```python
import helion
import helion.language as hl
import torch

# 1. Define the kernel in Helion
@helion.kernel
def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  # Standard PyTorch host-side setup
  x, y = torch.broadcast_tensors(x, y)
  out = torch.empty_like(x)

  # Device execution loop over automatically partitioned tiles
  for tile in hl.tile(out.size()):
    out[tile] = x[tile] + y[tile]

  return out

# 2. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors(x, y)
```

Para mais informações sobre o Helion, consulte a
[documentação do Helion](https://helionlang.com/), faça perguntas no
[fórum de discussão para desenvolvedores do PyTorch](https://dev-discuss.pytorch.org/c/helion/31)
ou [informe problemas no GitHub](https://github.com/pytorch/helion/issues).
