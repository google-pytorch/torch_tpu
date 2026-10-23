# Kernels personalizados de TorchTPU { #torchtpu-custom-kernels }

Aunque TorchTPU acelera de forma nativa los operadores estándar del framework
PyTorch (como
[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)),
es posible que necesites integrar kernels personalizados para acelerar
operaciones específicas del modelo, fusionar operaciones para reducir la
sobrecarga de ancho de banda de memoria o implementar arquitecturas nuevas que
los operadores estándar de PyTorch aún no cubren.

## Elige un enfoque de integración { #choosing-an-integration-approach }

TorchTPU ofrece tres formas de usar kernels personalizados en TPU, según tu caso
de uso:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Enfoque | Estado | Ideal para |
| :---- | :---- | :---- |
| **Tokamax** | Listo para producción | Aceleración directa de componentes comunes de LLM/MoE sin escribir código de kernels. |
| **Pallas (`torch.tpu.pallas.jax_op`)** | Listo para producción | Escribir kernels de TPU a medida y de máximo rendimiento en JAX/Pallas y llamarlos desde PyTorch. |
| **Helion** | Experimental | Escribir kernels portables entre aceleradores (por ejemplo, GPU y TPU) con un DSL nativo de PyTorch. |
<!-- mdformat on -->

## 1. Usa kernels precompilados de Tokamax { #1-using-pre-built-kernels-from-tokamax }

Antes de escribir un kernel personalizado desde cero, verifica si el operador
está disponible en [Tokamax](http://openxla.org/tokamax), una biblioteca de
kernels de TPU de alto rendimiento con interoperabilidad nativa con PyTorch.

### Cuándo usarlo { #when-to-use }

Usa Tokamax cuando tu modelo requiera componentes estándar de alto rendimiento,
como los siguientes:

*   **Mixture-of-Experts (MoE):** Multiplicación de matrices agrupada, gathers y
    scatters de tokens.
*   **Variantes de atención:** Paged attention, Gated DeltaNet, Kimi Delta
    Attention y Multi-head Latent Attention (MLA).
*   **Selección:** Cálculo optimizado de `TopK`.

### Cómo funciona { #how-it-works }

Los kernels de Tokamax se escriben en JAX/Pallas y cuentan con ajuste automático
integrado (como la búsqueda automatizada del tamaño de tile), comparativas de
rendimiento con XProf de baja sobrecarga y pruebas de regresión en todas las
generaciones de TPU. Internamente, Tokamax usa `torch.tpu.pallas.jax_op` para manejar la
interoperabilidad entre frameworks de forma transparente y expone APIs nativas
de PyTorch directamente a tu modelo.

## 2. Escribe kernels personalizados de Pallas { #2-writing-custom-pallas-kernels }

Si necesitas una operación a medida que PyTorch o Tokamax no cubren, puedes
escribir un kernel de TPU en
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) y registrarlo como
operador de PyTorch con `torch.tpu.pallas.jax_op`.

### Cómo integrar un kernel de Pallas { #how-to-integrate-a-pallas-kernel }

1.  **Define el kernel de Pallas** con referencias de memoria de entrada y
    salida (`Ref`s).
2.  **Encapsula el kernel en una función de JAX** con `pl.pallas_call` para
    especificar las formas y los dtypes de salida.
3.  **Registra la función de JAX en TorchTPU** mediante
    `torch.tpu.pallas.jax_op`.

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

## 3. Escribe kernels portables con Helion (experimental) { #3-writing-portable-kernels-with-helion-experimental }

Si quieres que tus kernels sean portables entre aceleradores (por ejemplo, TPU y
GPU), puedes usar [Helion](https://helionlang.com/), un DSL nativo de PyTorch
para escribir kernels.

> **Note:** Actualmente, la compatibilidad con Helion en TorchTPU es
> **experimental**. Para cargas de trabajo de producción, usa Tokamax o
> `torch.tpu.pallas.jax_op`.

### Cómo escribir un kernel de Helion { #how-to-write-a-helion-kernel }

Decora tu función con `@helion.kernel`. Usa operaciones estándar de PyTorch para
la configuración del lado del host y `hl.tile` para definir el bucle de
ejecución en el dispositivo sobre tiles particionados:

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

Para obtener más información sobre Helion, consulta la
[documentación de Helion](https://helionlang.com/), haz preguntas en el
[foro de discusión para desarrolladores de PyTorch](https://dev-discuss.pytorch.org/c/helion/31)
o [informa problemas en GitHub](https://github.com/pytorch/helion/issues).
