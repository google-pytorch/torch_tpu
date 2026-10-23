# Formas dinámicas { #dynamic-shapes }

<!-- Link to colab and to generated notebook -->

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/dynamic_shapes.ipynb)

Cuando compilas un modelo con `torch.compile(model)`, TorchTPU reduce el cálculo
a StableHLO y lo compila con el compilador XLA (consulta
[Cómo funciona TorchTPU](./how_torchtpu_works.md)). Tradicionalmente, el
compilador XLA se diseña en torno a **formas de tensores conocidas de forma
estática**, de modo que los búferes de la memoria de alto ancho de banda (HBM)
se puedan distribuir con anticipación, las canalizaciones de operadores se
puedan fusionar y las multiplicaciones de matrices se puedan dividir en tiles
para los arreglos sistólicos de la TPU (MXU). Cuando las dimensiones de los
tensores de entrada varían entre llamadas en una carga de trabajo de
`torch.compile`, compilar para cada forma exacta invalidaría las guardas de
forma de Dynamo y activaría una recompilación completa y costosa de XLA con cada
forma nueva.

Para admitir **formas dinámicas** sin recompilaciones constantes, TorchTPU usa
**dinamismo acotado** (Bounded Dynamism): los límites superiores de las
dimensiones dinámicas se fijan en el momento de la compilación, mientras que XLA
hace un seguimiento dinámico de los tamaños reales de las dimensiones en el
entorno de ejecución.

## Cómo maneja TorchTPU las formas dinámicas { #how-dynamic-shapes-are-handled-by-torchtpu }

Cuando compilas un modelo o una función con el dinamismo predeterminado, es
decir, `torch.compile(model, dynamic=None)` (a medida que las formas de entrada
varían entre llamadas), o marcas de forma explícita dimensiones dinámicas con
`torch._dynamo.mark_dynamic`, Dynamo traza el cálculo y produce un grafo FX en
el que las dimensiones que cambian se representan como variables enteras
simbólicas (`torch.SymInt`) en lugar de constantes fijas. TorchTPU maneja estas
dimensiones simbólicas de la siguiente manera:

### Compilación { #compilation }

*   TorchTPU intercepta el grafo FX trazado, determina un rango general
    `[lower_bound, upper_bound]` para cada dimensión simbólica (a partir de
    `mark_dynamic(..., min=..., max=...)` o inferido por Dynamo a partir del
    tamaño inicial `S` en el entorno de ejecución) y divide ese rango en buckets
    contiguos alineados con múltiplos de `bucket_size` (el valor predeterminado
    es `128`, que coincide con la dimensión de tile de 128 carriles de la TPU),
    hasta `num_dynamic_buckets` (el valor predeterminado es `1` y el máximo es
    `8`):
    *   El primer bucket abarca desde `lower_bound` hasta el siguiente múltiplo
        de `128` igual o posterior a `S` (`U_1`), el segundo bucket abarca desde
        `U_1 + 1` hasta el siguiente múltiplo de `128` (`U_2`), y así
        sucesivamente (por ejemplo, `[lower_bound, 128]`, `[129, 256]`, ...).
    *   TorchTPU registra en Dynamo una sola guarda de rango que abarca todos
        los buckets, de modo que Dynamo solo activa una recompilación si la
        dimensión supera el `upper_bound` general.
*   Para cada bucket, TorchTPU reemplaza la dimensión simbólica por el límite
    superior estático `U` del bucket, pasa el tamaño real de la dimensión en el
    entorno de ejecución como entrada y reduce el grafo a StableHLO con tipos de
    tensores acotados que registran tanto el límite superior estático `U` como
    el tamaño activo en el entorno de ejecución en todas las operaciones.
*   XLA compila el módulo resultante para cada límite superior `U` de bucket
    configurado.

### Ejecución { #execution }

*   En el entorno de ejecución, según el tamaño real de la dimensión dinámica,
    TorchTPU selecciona el ejecutable compilado del bucket correspondiente.

*   Cuando ejecuta el módulo compilado, XLA rellena (padding) el tensor de
    entrada dinámico hasta el límite superior estático `U` del bucket y
    restablece los valores de la región rellenada que supera el tamaño activo a
    valores de identidad algebraica. Así, los elementos de relleno no afectan el
    resultado.

*   En el caso de los tensores de salida dinámicos, el relleno permanece al
    final del búfer, lo que permite que TorchTPU devuelva el búfer a PyTorch con
    una actualización ligera de los metadatos de forma, en lugar de cortar el
    tensor físicamente.

## Caso de uso que lo motiva: Decodificación autorregresiva { #motivating-use-case-autoregressive-decoding }

Una de las principales cargas de trabajo que motivan el dinamismo acotado es la
**decodificación autorregresiva de LLM** con una caché de KV que crece de forma
dinámica. En un bucle de decodificación estándar, cada paso agrega los estados
de clave y valor del token nuevo al historial en caché a lo largo de la
dimensión de secuencia mediante `torch.cat`:

*   **Prefill:** El prompt de entrada tiene la forma `[B, S]` (tamaño de batch
    `B`, longitud de secuencia del prompt `S`) y produce una caché de KV con la
    forma `[B, H_kv, S, D]` (`H_kv` cabezas de clave/valor, dimensión de cabeza
    `D`).
*   **Paso de decodificación 1:** El token de entrada tiene la forma `[B, 1]` y
    se concatena con la caché para producir la forma `[B, H_kv, S + 1, D]`.
*   **Paso de decodificación `t`:** La longitud de secuencia de la caché de KV
    crece a `S + t`.

Con la compilación de formas estáticas (`dynamic=False`), Dynamo establece
guardas sobre la longitud de secuencia exacta de la caché de KV. Cada paso de
decodificación cambia la forma del tensor, invalida la guarda y activa una
**recompilación completa de XLA por cada token generado**. Esto convierte pasos
de decodificación de milisegundos en compilaciones de varios segundos, a menos
que reescribas el modelo para preasignar búferes estáticos y realizar
mutaciones in situ.

El **dinamismo acotado** resuelve esto sin necesidad de reescribir el modelo: a
medida que la caché de KV crece en los pasos de decodificación (`S + 1`, `S + 2`,
...), todas las formas dentro del bucket compilado (`<= 128`, `<= 256`, ...)
reutilizan el mismo ejecutable de XLA.

Para ver una explicación práctica que aplica el dinamismo acotado a un bucle de
decodificación completo de un LLM (y lo compara con la preasignación de una
caché de KV estática), consulta
[Inferencia de LLM de extremo a extremo](./case_studies_muse_glimmer_30b_inference.md).

## Ejemplo: Bucle autorregresivo con un tensor que crece { #example-autoregressive-loop-with-a-growing-tensor }

Para ver cómo se comportan en la práctica la compilación estática
(`dynamic=False`) y el dinamismo acotado con el dinamismo predeterminado
(`dynamic=None`), considera un bucle autorregresivo mínimo en el que cada paso
agrega un token nuevo a lo largo de la dimensión de secuencia (`dim=2`) con
`torch.cat`:

```python
import time
import torch
from torch._functorch._aot_autograd import autograd_cache

torch._logging.set_logs()  # pylint: disable=protected-access
device = torch.device("tpu")

def decode_step(
    kv_cache: torch.Tensor, new_token: torch.Tensor
) -> torch.Tensor:
  return torch.cat([kv_cache, new_token], dim=2)

def run_decode_loop(
    compiled_step, num_steps: int = 10, prefix: str = ""
) -> None:
  kv_cache = torch.ones((1, 8, 16, 128), dtype=torch.bfloat16, device=device)
  new_token = torch.ones((1, 8, 1, 128), dtype=torch.bfloat16, device=device)
  torch.accelerator.synchronize()

  for step in range(num_steps):
    start = time.perf_counter()
    kv_cache = compiled_step(kv_cache, new_token)
    torch.accelerator.synchronize()
    step_ms = (time.perf_counter() - start) * 1000
    print(
        f"{prefix}Step {step + 1}/{num_steps} (seq_len={kv_cache.shape[2]}):"
        f" {step_ms:.2f} ms",
        flush=True,
    )
```

### 1. Ejecución con `dynamic=False` { #1-running-with-dynamicfalse }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()
torch._dynamo.config.recompile_limit = 64

step_static = torch.compile(decode_step, backend="tpu", dynamic=False)
run_decode_loop(step_static, prefix="[dynamic=False] ")
```

Resultado de muestra:

```text
[dynamic=False] Step 1/10 (seq_len=17): 49.54 ms
[dynamic=False] Step 2/10 (seq_len=18): 43.51 ms
[dynamic=False] Step 3/10 (seq_len=19): 39.52 ms
[dynamic=False] Step 4/10 (seq_len=20): 38.63 ms
[dynamic=False] Step 5/10 (seq_len=21): 38.40 ms
[dynamic=False] Step 6/10 (seq_len=22): 38.23 ms
[dynamic=False] Step 7/10 (seq_len=23): 38.11 ms
[dynamic=False] Step 8/10 (seq_len=24): 38.55 ms
[dynamic=False] Step 9/10 (seq_len=25): 37.89 ms
[dynamic=False] Step 10/10 (seq_len=26): 37.75 ms
```

**Qué sucedió:** Con `dynamic=False`, Dynamo establece guardas sobre la longitud
de secuencia exacta de `kv_cache` (`16`, `17`, `18`, ...). Como `torch.cat`
aumenta `dim=2` en `1` en cada iteración, la guarda de forma falla en cada
llamada y activa una recompilación completa de XLA en cada paso.

### 2. Ejecución con `dynamic=None` (dinamismo acotado) { #2-running-with-dynamicnone-bounded-dynamism }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()

step_dynamic = torch.compile(decode_step, backend="tpu", dynamic=None)
run_decode_loop(step_dynamic, prefix="[dynamic=None] ")
```

Resultado de muestra:

```text
[dynamic=None] Step 1/10 (seq_len=17): 49.59 ms
[dynamic=None] Step 2/10 (seq_len=18): 206.68 ms
[dynamic=None] Step 3/10 (seq_len=19): 64.29 ms
[dynamic=None] Step 4/10 (seq_len=20): 0.51 ms
[dynamic=None] Step 5/10 (seq_len=21): 0.43 ms
[dynamic=None] Step 6/10 (seq_len=22): 0.38 ms
[dynamic=None] Step 7/10 (seq_len=23): 0.40 ms
[dynamic=None] Step 8/10 (seq_len=24): 0.44 ms
[dynamic=None] Step 9/10 (seq_len=25): 0.45 ms
[dynamic=None] Step 10/10 (seq_len=26): 0.37 ms
```

**Qué sucedió:** Con `dynamic=None`, solo las primeras 3 iteraciones tienen
sobrecarga de compilación y, a partir del paso 4, la ejecución baja a unos
0.4 ms por paso (~100 veces más rápido) sin recompilaciones:

*   **Paso 1 (`seq_len=16 -> 17`):** Dynamo ve las entradas por primera vez y
    compila un ejecutable estático para `seq_len=16`.
*   **Paso 2 (`seq_len=17 -> 18`):** Como `dim=2` cambió de `16` a `17`, Dynamo
    marca automáticamente `dim=2` como dinámica (`SymInt`) y TorchTPU compila un
    ejecutable con dinamismo acotado para el bucket `[17, 128]` con el diseño
    del tensor de entrada inicial.
*   **Paso 3 (`seq_len=18 -> 19`):** El búfer de salida con dinamismo acotado
    que produce el paso 2 tiene el diseño de salida rellenado de XLA, por lo que
    TorchTPU vuelve a compilar el bucket `[17,
    128]` para ese diseño de XLA en estado estable.
*   **Pasos 4–10 (`seq_len=19 -> 26`):** Tanto la longitud de secuencia (dentro
    del bucket `<= 128`) como el diseño del búfer de XLA coinciden con el
    ejecutable en caché, por lo que cada paso posterior despacha directamente
    el ejecutable de XLA precompilado sin recompilar.

> **Tip:** Si ya sabes de antemano qué dimensión variará, puedes llamar a
> `torch._dynamo.mark_dynamic(kv_cache, 2, min=16, max=128)` antes del primer
> paso. Esto le indica a Dynamo que trace `dim=2` como dinámica de inmediato en
> el paso 1, lo que omite la compilación estática inicial del paso 1 y te
> permite especificar límites `[min, max]` explícitos.

## Limitaciones actuales { #current-limitations }

El dinamismo acotado en TorchTPU evoluciona activamente y, por ahora, está
optimizado para el caso de uso de la **caché de KV autorregresiva**. Ten en cuenta
las siguientes limitaciones cuando lo apliques a tus modelos:

*   **Límite de buckets y dinamismo no acotado:** Se compilan como máximo
    **8 buckets** (`num_dynamic_buckets <= 8`) por traza. El dinamismo no
    acotado aún no es compatible: si una dimensión dinámica supera el límite
    superior del bucket compilado más alto, se activa una recompilación.
*   **Restricciones de forma de Dynamo:** Los límites superiores de los buckets
    deben cumplir las restricciones de forma activas de Dynamo (como los
    requisitos de divisibilidad de `reshape` o `view`). De lo contrario,
    TorchTPU recurre a un solo bucket con límite superior `2 * S` (el doble del
    tamaño inicial `S` en el entorno de ejecución).
*   **Sobrecarga de cálculo y HBM hasta el límite superior del bucket:** Como la
    ejecución se realiza en búferes rellenados hasta el límite superior estático
    `U` del bucket, el uso de HBM y los FLOP escalan con `U` y no con el tamaño
    real en el entorno de ejecución. Si estableces un `bucket_size` muy grande o
    `mark_dynamic(..., max=U)` cuando el tamaño real es mucho menor que `U`,
    pagas el costo de cálculo y memoria de `U`.
*   **Cambios en el diseño de XLA:** Incluso cuando la forma de un tensor está
    dentro de un bucket ya compilado, un cambio en el diseño físico de XLA
    subyacente de un búfer de entrada activa una recompilación del bucket.
*   **Sin caché de compilación persistente (`AOTAutogradCache`):** Los
    ejecutables con dinamismo acotado aún no se serializan en
    `AOTAutogradCache`, por lo que se recompilan cada vez que se inicia un
    proceso nuevo.
*   **Cobertura de operadores:** No todos los operadores de PyTorch admiten aún
    formas dinámicas acotadas.
*   **Vistas como entradas del modelo:** La compatibilidad con pasar vistas de
    tensores directamente como entradas al modelo compilado es limitada.
*   **Dinamismo dependiente de los datos:** No se admiten operaciones cuyas
    formas de salida dependen de los valores de los tensores en el entorno de
    ejecución (como `torch.nonzero`).
*   **Kernels personalizados:** Las formas dinámicas acotadas no son compatibles
    con kernels de TPU personalizados (como los kernels de Pallas registrados
    mediante `torch.tpu.pallas.jax_op`).
*   **dynamic=True:** El modo *dynamic=True* de torch.compile() no es
    compatible.
