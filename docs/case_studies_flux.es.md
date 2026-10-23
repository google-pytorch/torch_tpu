# Optimización de FLUX { #flux-optimization }

<!-- Link to colab and to generated notebook -->

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_flux.ipynb)

En este caso práctico, se explica cómo tomamos una implementación de
referencia en PyTorch del `DoubleStreamBlock` del transformer de difusión
[FLUX](https://github.com/black-forest-labs/flux) e hicimos que un paso de
entrenamiento fuera **1.48x más rápido** en Cloud TPU con TorchTPU, con una
longitud de secuencia de 32,768.

Esta página es un complemento condensado del instructivo completo. Destaca el
concepto clave detrás de cada optimización con fragmentos de código breves e
independientes. Para ver el código completo del modelo, las comparativas de
rendimiento (benchmarks), las trazas de XProf y las pruebas, consulta lo
siguiente:

*   [Instructivo de optimización de FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md):
    recorrido completo por cada hito, incluidas las trazas de XProf y las
    instrucciones para reproducir cada comparativa de rendimiento.
*   [Crea kernels personalizados de Pallas para PyTorch](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md):
    recorrido paso a paso por el kernel de Pallas SplashAttention que se usa en
    el hito 4.

La implementación se basa en la
[implementación de FLUX de TorchTitan](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)
y en el repositorio
[black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/).

Si los ejemplos son nuevos para ti, te recomendamos que te familiarices con la
ejecución de ejemplos en GKE
[a través de nuestros otros documentos](./get_started_on_gke.md).

## Resultados de un vistazo { #results-at-a-glance }

Tiempo promedio por paso de entrenamiento (pase hacia adelante + pérdida + pase
hacia atrás + paso del optimizador) para un solo `DoubleStreamBlock`:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Configuración | Tiempo promedio por paso | Aceleración respecto del valor de referencia |
| :--- | ---: | ---: |
| Referencia de base (eager) | 178.61 ms | 1.00x |
| Solo Pallas SplashAttention (eager) | 149.81 ms | 1.19x |
| Solo vistas sin copia (eager) | 178.30 ms | 1.00x |
| Solo `torch.compile` | 159.33 ms | 1.12x |
| Vistas sin copia + RoPE fusionado + Pallas SplashAttention (eager) | 156.73 ms | 1.14x |
| Vistas sin copia + `torch.compile` | 159.08 ms | 1.12x |
| Vistas sin copia + RoPE fusionado + `torch.compile` | 153.60 ms | 1.16x |
| **Vistas sin copia + RoPE fusionado + Pallas SplashAttention + `torch.compile`** | **122.51 ms** | **1.46x** |
<!-- mdformat on -->

**Configuración de la medición:**

-   **Hardware:** un solo chip TPU v6e (32 GiB de HBM).
-   **Software:** Python 3.14, PyTorch 2.15 (versión preliminar), JAX 0.12,
    einops 0.8.2 y TorchTPU compilado desde el código fuente.
-   **Forma del modelo:** `batch_size=2`, `hidden_size=1024`, `num_heads=8`
    (`head_dim=128`), `img_seq_len=32256`, `txt_seq_len=512` (longitud total de
    la secuencia: 32,768).
-   **Precisión:** parámetros y entradas del bloque en bfloat16.
-   **Tamaños de bloque de SplashAttention:** 2048 para `block_q`, `block_kv` y
    los tiles de dK/dV del pase hacia atrás.
-   **Medición del tiempo:** 3 pasos de calentamiento seguidos de 10 pasos
    cronometrados. Los pasos cronometrados se despachan uno tras otro y
    `torch.accelerator.synchronize()` se ejecuta una vez después del último,
    por lo que el tiempo promedio por paso (el tiempo total dividido por la
    cantidad de pasos) incluye la ejecución en el dispositivo. El generador de
    perfiles (profiler) está desactivado durante los pasos cronometrados.

Con esta longitud de secuencia, la atención domina el paso, por lo que
SplashAttention y la compilación producen las mayores mejoras. Con longitudes
de secuencia más cortas, la compilación importa más (alrededor de 3x con una
longitud de secuencia de 640) y el kernel de atención importa menos.

Las lecciones se aplican a la mayoría de los modelos transformer en TPU:

1.  **Evita las copias de diseño ocultas**: Prefiere las vistas sin copia
    (`.view()`, `torch.unbind()`) en lugar de reorganizaciones que materializan
    nuevos búferes en HBM.
2.  **Almacena menos, calcula más**: Expresa las incorporaciones posicionales
    (positional embeddings) como operaciones matemáticas compactas elemento por
    elemento en lugar de matrices materializadas.
3.  **Compila el bloque completo**: `torch.compile()` permite que XLA fusione
    cadenas de operaciones elemento por elemento y elimine los viajes de ida y
    vuelta intermedios a HBM.
4.  **Recurre a kernels personalizados cuando sea necesario**: Los kernels de
    Pallas dan control explícito sobre la división en tiles (tiling) en el chip
    para operaciones limitadas por la memoria, como la atención con longitudes
    de secuencia largas.

### Cómo leer esta página { #how-to-read-this-page }

Cada hito se presenta como un par de secciones:

-   **Antes (referencia)**: la implementación original y su cuello de botella.
-   **Después (optimizada)**: la implementación optimizada, seguida de una
    verificación de que produce el mismo resultado que la referencia.

Las celdas de código tienen un comentario `# BEFORE (reference)` o `# AFTER
(optimized)` correspondiente en su primera línea.

**Requisitos previos:** La celda de referencia del hito 1 usa
[einops](https://github.com/arogozhnikov/einops) (`pip install einops`).

```python
import torch

device = torch.device("tpu")
print(f"Running on {device}.")
```

## Hito 1: Vistas de tensores sin copia { #milestone-1-zero-copy-tensor-views }

En `DoubleStreamBlock`, los flujos de imagen y de texto proyectan estados
ocultos de forma `(B, L, 3 * H * D)` en consultas, claves y valores. La manera
en que esa proyección se divide en Q, K y V determina si la TPU tiene que copiar
datos en HBM durante el entrenamiento.

### Antes (referencia): `einops.rearrange` { #before-reference-einopsrearrange }

La implementación de referencia mueve el eje `3` (Q/K/V) al principio:

```python
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)
```

En el pase hacia adelante, `einops.rearrange` devuelve vistas con stride de la
proyección, por lo que todavía no se mueven datos. El costo aparece en el
**pase hacia atrás**: para dirigir los gradientes a través del reordenamiento de
ejes, autograd materializa el gradiente en un búfer nuevo, lo que se ejecuta en
la TPU como una copia adicional de HBM a HBM en cada bloque de atención (visible
como una operación `copy` en XProf).

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

### Después (optimizada): `.view()` + `torch.unbind()` { #after-optimized-view-torchunbind }

La implementación optimizada mantiene el eje `3` junto a `(H, D)` y lo divide
con `torch.unbind`. Luego, el pase hacia atrás apila los gradientes de Q, K y V
directamente en el diseño original, por lo que no se necesita una copia
adicional. En la tabla de resultados, el efecto en el tiempo por paso es pequeño
(**de 178.61 ms a 178.30 ms** en modo eager y **de 159.33 ms a 159.08 ms** con
compilación) porque la copia es una fracción pequeña del paso, pero se elimina
sin ningún costo.

> **Note:** Las celdas de este hito solo muestran que los dos enfoques producen
> los mismos Q, K y V. En el pase hacia adelante, ambos devuelven las mismas
> vistas con stride, por lo que sus resultados se ven idénticos. La copia
> adicional en HBM de `einops.rearrange` solo ocurre en el pase hacia atrás, y
> se puede ver en una traza de XProf de un paso de entrenamiento completo, no
> en el resultado de la celda. Consulta el
> [hito 1 del instructivo de optimización de FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-1)
> para ver las trazas.

```python
# AFTER (optimized): keep Q/K/V adjacent to (H, D) and unbind.
_b, _l, _ = qkv.shape
q_after, _k, _v = torch.unbind(
    qkv.view(_b, _l, 3, num_heads, head_dim), dim=2
)
print(f"q shape: {tuple(q_after.shape)}")
print(f"Matches reference: {torch.equal(q_after.cpu(), q_before.cpu())}")
```

## Hito 2: RoPE algebraico fusionado { #milestone-2-fused-algebraic-rope }

FLUX aplica incorporaciones de posición rotatorias (Rotary Position Embeddings,
RoPE) multieje a las consultas y las claves. Las celdas de abajo implementan
ambas versiones para un solo eje de RoPE.

### Antes (referencia): matrices de rotación de 2x2 { #before-reference-2x2-rotation-matrices }

La implementación de referencia materializa una matriz de rotación `2x2`
completa `[[cos,
-sin], [sin, cos]]` para cada par de frecuencias y la aplica como un producto
de matriz por vector sobre tensores de consulta/clave 6D a los que se les
cambió la forma. Esto almacena `cos` y `sin` dos veces, y el apilamiento y los
cambios de forma adicionales aparecen como grandes bloques de copia en XProf.

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

### Después (optimizada): pares compactos `[cos, sin]` + rotación elemento por elemento { #after-optimized-compact-cos-sin-pairs-elementwise-rotation }

La implementación optimizada almacena solo pares compactos `[cos, sin]` y aplica
la rotación directamente con operaciones matemáticas elemento por elemento:

```python
# (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
xq_out_0 = xq_0 * cos - xq_1 * sin
xq_out_1 = xq_0 * sin + xq_1 * cos
```

Esto reduce a la mitad el espacio que ocupa la incorporación posicional, se
asigna a instrucciones de multiplicación y suma fusionadas (fused multiply-add)
y, durante la compilación, se fusiona con las operaciones de proyección y
atención que la rodean. Sumado a las vistas sin copia y a `torch.compile`,
reduce el tiempo por paso **de 159.08 ms a 153.60 ms (3.4%)**.

Como ambas versiones realizan las mismas multiplicaciones y sumas sobre los
mismos operandos, producen resultados idénticos. La celda de abajo ejecuta la
versión optimizada con las mismas entradas y la compara con la referencia:

> **Note:** Estas celdas muestran el tensor de frecuencias más pequeño y que
> ambas versiones producen resultados idénticos. La aceleración solo aparece
> cuando RoPE se ejecuta dentro del `DoubleStreamBlock` compilado, donde la
> rotación elemento por elemento se fusiona con las operaciones que la rodean.
> Consulta el
> [hito 2 del instructivo de optimización de FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-2)
> para ver las trazas de XProf.

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

## Hito 3: Compilación de XLA del bloque completo { #milestone-3-whole-block-xla-compilation }

### Antes (referencia): ejecución eager { #before-reference-eager-execution }

En modo eager, cada operación de PyTorch inicia un programa de TPU
independiente, por lo que las activaciones intermedias hacen viajes de ida y
vuelta a HBM y el host se detiene con frecuencia en el despacho. En la tabla de
resultados, el `DoubleStreamBlock` de referencia tarda **178.61 ms** por paso en
modo eager.

La celda de abajo ejecuta el RoPE fusionado del hito 2 en modo eager, operación
por operación:

```python
# BEFORE (reference): eager execution, one TPU program per op.
q_eager, k_eager = apply_rope(rope_xq, rope_xk, freqs)
print("Ran fused RoPE eagerly.")
```

### Después (optimizada): `torch.compile()` { #after-optimized-torchcompile }

Si compilas todo el `DoubleStreamBlock` con `torch.compile()`, se reduce a un
solo grafo de XLA, lo que permite que el compilador fusione las operaciones
elemento por elemento, las normalizaciones y las sumas residuales:

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

La compilación reduce el tiempo por paso **de 178.61 ms a 159.33 ms (1.12x)**.
La mejora es moderada con una longitud de secuencia de 32,768 porque la atención
domina el paso. Con una longitud de secuencia de 640, la compilación es
alrededor de 3x más rápida.

El mismo patrón se aplica a cualquier función. La celda de abajo compila el RoPE
fusionado y verifica que coincida con la ejecución eager:

> **Note:** Estas celdas solo verifican que la compilación no cambie el
> resultado. La fusión y la aceleración aparecen en XProf y en las mediciones
> del tiempo por paso del `DoubleStreamBlock` completo, no en el resultado de la
> celda. Consulta el
> [hito 3 del instructivo de optimización de FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-3)
> para ver las trazas de la ejecución eager frente a la compilada.

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

## Hito 4: Kernel personalizado de Pallas SplashAttention { #milestone-4-custom-pallas-splashattention-kernel }

Con longitudes de secuencia largas (por ejemplo, `S = 2048` o `S = 4096`), la
atención conjunta completa queda limitada por la memoria. Este hito se muestra
solo con fragmentos de código; consulta el
[recorrido por el kernel de Pallas](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
para ver la implementación completa y ejecutable.

### Antes (referencia): SDPA estándar de PyTorch { #before-reference-standard-pytorch-sdpa }

```python
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

La atención sin división en tiles materializa una matriz de puntuaciones
`O(S^2)` en HBM.

### Después (optimizada): Pallas SplashAttention { #after-optimized-pallas-splashattention }

```python
attn = splash_sdpa(q, k, v, is_causal=False)  # Drop-in replacement.
```

`splash_sdpa` es un kernel personalizado de SplashAttention escrito en
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) que calcula la
atención en tiles dentro del chip, basado en la
[implementación de SplashAttention de TorchTitan](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py).

El kernel se expone a PyTorch con `torch.tpu.pallas.jax_op`. Sus funciones de
pase hacia adelante y hacia atrás tienen anotaciones de tipo `jax.Array`, a
partir de las cuales `jax_op` infiere el esquema del operador personalizado de
PyTorch. El pase hacia adelante guarda el residual `logsumexp` del softmax para
que el kernel del pase hacia atrás no necesite volver a ejecutar el cómputo del
pase hacia adelante:

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

> **Important:** **JAX no se usa en el entorno de ejecución.** Pallas se
> distribuye dentro del paquete `jax`, por lo que el kernel se escribe con
> sintaxis de `jax`. Sin embargo, `torch.tpu.pallas.jax_op` solo usa JAX una
> vez, en el momento del trazado, con marcadores de posición abstractos de
> forma/dtype para exportar el kernel como un módulo MLIR. Durante el
> entrenamiento y la inferencia, tanto en modo eager como con `torch.compile`,
> el kernel se ejecuta directamente sobre búferes de TPU de `torch.Tensor`.

**Resultados**: En la tabla de resultados, SplashAttention por sí solo es
**1.19x** más rápido que el valor de referencia en modo eager (de 178.61 ms a
149.81 ms). Con las vistas sin copia y el RoPE fusionado habilitados, es más
rápido que el SDPA estándar tanto en modo eager (**de 182.61 ms a 156.73 ms,
1.17x**) como en modo compilado (**de 153.60 ms a 122.51 ms, 1.25x**).

Estos resultados dependen de configurar el kernel según la carga de trabajo. Con
los bloques predeterminados de 512 tokens, SplashAttention era alrededor de un
10% más lento que el SDPA estándar con compilación y con la misma longitud de
secuencia, y las entradas en float32 eran aún más lentas. Antes de recurrir a un
kernel personalizado, compara el rendimiento de algunos tamaños de bloque para
tu longitud de secuencia, ejecuta el kernel en bfloat16 y genera un perfil.

**Más información:**

*   **Ejemplo de Pallas SplashAttention (usado en este caso práctico)**:
    [`splash_attention.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention.py)
    contiene la integración completa del kernel, y
    [`splash_attention_test.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention_test.py)
    verifica la paridad del pase hacia adelante y de los gradientes con el SDPA
    de PyTorch. El
    [recorrido por el kernel](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
    explica paso a paso los tamaños de bloque, las máscaras, el registro de
    autograd y las pruebas.
*   **Conceptos básicos de Pallas**:
    [Kernels personalizados con Pallas](./custom_kernels.md) presenta cómo
    escribir kernels de Pallas con TorchTPU.
