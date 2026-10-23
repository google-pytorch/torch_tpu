# Administra la precisión numérica en TorchTPU { #manage-numerical-precision-in-torchtpu }

Las TPU de Google dividen las operaciones de punto flotante entre dos unidades de
hardware para equilibrar la capacidad de procesamiento y la exactitud numérica:

*   **Unidad de multiplicación de matrices (MXU):** Ejecuta operaciones de
    contracción (como `torch.matmul`, `torch.mm`, `torch.bmm` y las
    convoluciones) con entradas nativas `bfloat16` (significando de 8 bits) y
    acumulación en `float32` de 32 bits.
*   **Unidad de ejecución vectorial (VPU):** Ejecuta operaciones que no son de
    contracción (como la aritmética elemento por elemento, las activaciones, la
    normalización de capas y softmax) en `float32` IEEE-754 completo
    (significando de 24 bits).

En esta guía, se explica cómo configurar los modos de precisión de la
multiplicación de matrices en `float32`, verificar la paridad numérica entre la
CPU y la TPU, controlar los tipos del acumulador de salida y la precisión
excedente, y validar en código el comportamiento bit a bit de la MXU.

### Compara la precisión de PyTorch en CUDA y en TPU { #compare-pytorch-precision-on-cuda-and-tpu }

TorchTPU admite las APIs estándar de precisión de PyTorch
(`torch.set_float32_matmul_precision` y `torch.get_float32_matmul_precision`)
junto con controles específicos de TPU (`torch.tpu.Precision` y
`torch.tpu.precision`):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Dimensión          | PyTorch en CUDA                      | PyTorch en TPU (TorchTPU)              |
| :----------------- | :----------------------------------- | :------------------------------------- |
| **Modo de precisión predeterminado** | `"highest"` (`float32` puro) | `"medium"` (`Precision.DEFAULT`)       |
| **Primitiva del acelerador para cálculo rápido** | Tensor Cores (`TF32`) | MXU sistólicas (`bfloat16`)             |
| **Bits en cálculo rápido** | 1 bit de signo + 8 de exponente + 10 de mantisa (`TF32` de 19 bits) | 1 bit de signo + 8 de exponente + 7 de mantisa (`bfloat16` de 16 bits) |
| **Precisión de acumulación** | `float32` (`"highest"`, `"high"`), `TF32` (`"medium"`) | `float32` en todos los modos |
| **Mecanismo de mayor precisión** | Unidades de hardware dedicadas `float32`/`TF32` | Descomposición en varias pasadas (3 o 6 pasadas por la MXU) |
| **Impacto en el rendimiento de `high`/`highest`** | Velocidad de referencia o ralentización moderada (~1–2x) | ~3x de costo de cálculo de la MXU para `"high"`; ~6x para `"highest"` |
| **API de alcance** | `torch.set_float32_matmul_precision` global | `torch.set_float32_matmul_precision` <br> global + `torch.tpu.precision` con alcance |
<!-- mdformat on -->

## Especifica los modos de precisión { #specify-precision-modes }

Cuando multiplicas tensores `float32` en la MXU, TorchTPU asigna los niveles de
precisión de PyTorch a la enumeración `torch.tpu.Precision`:

*   `"medium"` / `Precision.DEFAULT` (1 pasada): **Predeterminado.** Redondea
    las entradas `float32` a `bfloat16` (multiplica los 8 bits superiores del
    significando) y acumula en `float32`. Se recomienda para casi todas las
    cargas de trabajo de entrenamiento e inferencia.
*   `"high"` / `Precision.HIGH` (3 pasadas): Divide las entradas `float32` en
    componentes `bfloat16` alto y bajo, y calcula tres términos cruzados
    (multiplica los primeros 16 bits del significando) con aproximadamente 3
    veces el costo de cálculo de la MXU.
*   `"highest"` / `Precision.HIGHEST` (6 pasadas): Divide las entradas `float32`
    en tres componentes `bfloat16` y calcula seis términos cruzados (multiplica
    los 24 bits del significando) para aproximarse a la exactitud completa de
    `float32` IEEE-754 con aproximadamente 6 veces el costo de cálculo de la
    MXU.

Puedes configurar la precisión de la multiplicación de matrices de dos formas:

*   **Configuración global:** Llama a `torch.set_float32_matmul_precision("medium"
    | "high" | "highest")` para establecer el modo predeterminado del proceso.
*   **Administrador de contexto con alcance:** Usa `with torch.tpu.precision(mode):`
    para aplicar una precisión mayor solo a capas o submódulos sensibles
    numéricamente.
*   **Advertencia de rendimiento:** Si seleccionas `"high"` o `"highest"`, se
    emite una advertencia única (`TORCH_WARN_ONCE`). Mantén el modo
    predeterminado `"medium"`, a menos que observes problemas de convergencia
    relacionados con la precisión.

> **Note:** Entrar en `torch.tpu.precision` o salir de él introduce una
> interrupción de grafo de `torch.compile`. Cuando uses
> `torch.compile(..., fullgraph=True)`, establece la precisión fuera de la
> función compilada con `torch.set_float32_matmul_precision(...)` o encapsula
> la llamada externa en `with
> torch.tpu.precision(...):`.

Usa `torch.tpu.precision` para comparar el tiempo de ejecución en estado estable
de los tres modos:

```python
import time
import torch

device = torch.device("tpu")
precision = torch.tpu.precision  # Context manager
Precision = torch.tpu.Precision  # Enum class

print(f"Default matmul precision: {torch.get_float32_matmul_precision()}")

size = (8, 2048, 2048)
a = torch.randn(size, dtype=torch.float32, device=device)
b = torch.randn(size, dtype=torch.float32, device=device)

def run_matmul():
  res = torch.matmul(a, b)
  torch.accelerator.synchronize()
  return res

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    # Warm up to compile the graph before timing steady-state execution
    run_matmul()
    start = time.perf_counter()
    run_matmul()
    print(f"{mode} time: {time.perf_counter() - start:.4f}s")
```

### Compara la exactitud de los modos de precisión { #compare-accuracy-across-precision-modes }

Compara los resultados de la TPU con una referencia `float32` de la CPU sobre las
mismas entradas para ver cómo cada modo de precisión reduce el error numérico:

```python
# CPU float32 reference result
c_ref = torch.matmul(a.cpu(), b.cpu())

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    res = torch.matmul(a, b)
    diff = res.cpu() - c_ref
    print(f"Mode: {mode}")
    print(f"  Max Absolute Error: {diff.abs().max():.6e}")
    print(
        "  Mean Relative Error: "
        f" {(diff.abs() / (c_ref.abs() + 1e-8)).mean():.6e}\n"
    )
```

## Verifica la paridad numérica { #verify-numerical-parity }

Cuando portes un modelo a TPU o depures diferencias de convergencia, verifica la
paridad numérica: ejecuta la misma capa o submódulo en la CPU y en la TPU con
pesos sincronizados y entradas idénticas, y luego compara los resultados con
`torch.testing.assert_close`:

```python
import torch.nn as nn

# 1. Set up identical models in bfloat16 for a fair comparison
model_cpu = nn.Linear(10, 10).cpu().to(torch.bfloat16)
model_tpu = nn.Linear(10, 10).to(device).to(torch.bfloat16)

# 2. Synchronize weights (mandatory for parity checks)
model_tpu.load_state_dict(model_cpu.state_dict())

# 3. Create identical input data for both devices
input_data = torch.randn(5, 10, dtype=torch.bfloat16)
input_tpu = input_data.to(device)

# 4. Run the forward pass
out_cpu = model_cpu(input_data)
out_tpu = model_tpu(input_tpu)

# 5. Compare results
# Note: TorchTPU's bfloat16 accumulation and fusion differences can exceed
# default tolerances, generating the `assert_close` diagnostic report below.
try:
  torch.testing.assert_close(out_tpu.cpu(), out_cpu)
  print("Parity check passed!")
except AssertionError as e:
  print("Parity check failed (expected for hardware precision drift):\n")
  print(e)
```

### Interpreta las discrepancias detalladas { #interpret-detailed-mismatches }

Cuando los tensores superan las tolerancias, `torch.testing.assert_close`
imprime un resumen que ayuda a distinguir la deriva normal de punto flotante de
un error de lógica:

```text
Tensor-likes are not close!

Mismatched elements: 1 / 50 (2.0%)
Greatest absolute difference: 0.006596 at index (3, 9) (up to 1e-05 allowed)
Greatest relative difference: 0.165375 at index (3, 9) (up to 0.016 allowed)
```

*   **Mismatched elements:** Un porcentaje alto de elementos con diferencias
    absolutas pequeñas suele indicar diferencias estándar de redondeo o de orden
    de acumulación debidas a las fusiones del compilador, y no un error.
*   **Greatest absolute / relative difference:** Informa el error más grande y
    su índice en el tensor para que puedas inspeccionar los valores atípicos o
    elegir umbrales adecuados de `atol` y `rtol`.

### Prácticas recomendadas para la paridad { #best-practices-for-parity }

*   **Usa los mismos tipos de datos en ambos dispositivos:** Compara `bfloat16`
    en la TPU con `bfloat16` en la CPU, o `float32` en la TPU (con
    `Precision.HIGHEST`) con `float32` en la CPU. Si comparas `bfloat16` con
    `float32`, mides el error de cuantización y no la deriva del hardware.
*   **Verifica los gradientes además de las activaciones:** Después de llamar a
    `loss.backward()`, compara `model_tpu.weight.grad.cpu()` con
    `model_cpu.weight.grad` con el mismo patrón.
*   **Comienza con las tolerancias estándar:** Usa como referencia las
    tolerancias predeterminadas de la
    [documentación de pruebas de PyTorch](https://docs.pytorch.org/docs/stable/testing.html)
    y flexibiliza `atol` o `rtol` solo donde se espera por la acumulación con
    precisión reducida.

## Especifica la precisión de salida { #specify-output-precision }

Cuando multiplicas tensores `bfloat16` con `torch.mm` o `torch.bmm`, la MXU
acumula los productos en `float32` antes de redondear el resultado final de
vuelta a `bfloat16`. Pasa `out_dtype=torch.float32` para devolver directamente
el acumulador `float32` sin reducir la precisión:

```python
a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device=device)
# Using out_dtype to force float32 output accumulator precision
res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
print(f"Input dtype: {a_bmm.dtype}")
print(f"Output dtype: {res_bmm.dtype}")
```

## Controla la precisión excedente { #control-excess-precision }

Como la MXU de la TPU acumula las multiplicaciones de matrices `bfloat16` en
`float32`, el compilador XLA puede optimizar las operaciones fusionadas si
mantiene los valores intermedios en `float32`, en lugar de redondearlos a
`bfloat16` entre pasos. Conservar estos bits intermedios adicionales se
denomina *precisión excedente* (excess precision).

Puedes configurar este comportamiento de forma global con
`torch.tpu.allow_excess_precision` (que establece la marca subyacente del
compilador `--xla_allow_excess_precision` y cuyo valor predeterminado es
`False`):

*   **Inhabilita la precisión excedente (`False`, predeterminado):** Establece
    `torch.tpu.allow_excess_precision = False` para obligar al compilador a
    conservar las conversiones intermedias a `bfloat16`, de modo que las
    operaciones fusionadas coincidan con el comportamiento de redondeo
    `bfloat16` del modo eager.
*   **Permite la precisión excedente (`True`):** Establece `torch.tpu.allow_excess_precision =
    True` para permitir que el compilador omita las conversiones intermedias y
    mantenga los resultados intermedios fusionados en `float32`.

> **Important:** Establece `torch.tpu.allow_excess_precision` al principio de tu
> secuencia de comandos, antes de ejecutar o compilar cualquier operación de
> TPU. El entorno de ejecución de XLA bloquea esta marca cuando se inicia la
> primera compilación; si intentas cambiarla a un valor diferente después, se
> genera un `RuntimeError`.

## Comprende la precisión de la MXU de la TPU { #understand-tpu-mxu-precision }

Comprender cómo la MXU de la TPU maneja los números de punto flotante explica por
qué la precisión excedente y los modos `float32` de varias pasadas funcionan
como lo hacen:

*   **Rango frente a precisión (`bfloat16`):** La
    [investigación de Google Cloud sobre `bfloat16`](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
    demostró que las redes neuronales pueden entrenarse con una precisión de
    mantisa reducida sin perder exactitud del modelo, siempre que conserven el
    amplio rango dinámico de `float32` (fundamental para operaciones como las
    normas y los gradientes). Mientras que `float32` tiene 23 bits de mantisa
    (24 bits de precisión del significando, incluido el bit inicial implícito),
    `bfloat16` conserva el mismo rango de exponente de 8 bits con solo 7 bits de
    mantisa (8 bits de precisión del significando, con un épsilon de máquina de
    `2**-7` o ~0.78%).
*   **Formatos de intercambio frente a formatos aritméticos:** Los sistemas de
    punto flotante distinguen entre **formatos de intercambio** (cómo se
    almacenan los números en la memoria) y **formatos aritméticos** (cómo el
    hardware calcula y redondea las operaciones intermedias, como los 3 bits
    adicionales de guarda, redondeo y sticky de una ALU estándar). `bfloat16`
    puede referirse a cualquiera de los dos.
*   **Acumulación a lo largo de la dimensión de contracción:** Multiplicar dos
    matrices con la forma `[I, J] @ [J, K]` calcula una cuadrícula de productos
    internos con la forma `[1,
    J] @ [J, 1]`, y cada uno suma `J` productos elemento por elemento a lo largo
    de la **dimensión de contracción** `J`. Como `bfloat16` tiene un épsilon de
    máquina de `2**-7` (~`0.008`), acumular `J` productos en `bfloat16`
    causaría grandes errores de redondeo. En cambio, la MXU acepta entradas
    `bfloat16` (y `fp8` en generaciones más recientes, como TPU v7x) y acumula
    las sumas intermedias en `float32`, que cumple la misma función que los bits
    de guarda, redondeo y sticky de una ALU antes de redondear al formato de
    salida.
*   **Discrepancia entre los formatos de entrada y salida:** Tener entradas
    `bfloat16` y un acumulador `float32` es la causa raíz tanto de la
    *precisión excedente* (conservar valores intermedios `float32` entre
    operaciones fusionadas) como de una *precisión menor a la esperada* (la
    matmul `float32` de 1 pasada trunca las entradas a significandos `bfloat16`
    de 8 bits). En cambio, la unidad de ejecución vectorial (VPU) de la TPU
    opera con los 23 bits de mantisa de `float32` (excepto en casos límite
    menores de IEEE-754, como los números subnormales).

### Expansión polinómica de varias pasadas para `float32` { #multi-pass-polynomial-expansion-for-float32 }

Como una MXU solo acepta entradas `bfloat16` (significando de 8 bits), no puede
calcular una multiplicación de matrices `float32` completa de 24 bits en una
sola pasada. En la práctica, la mayoría de los modelos modernos almacenan los
pesos en `bfloat16` y rara vez ejecutan matmuls en `float32`. Cuando se necesita
la precisión completa de matmul en `float32`, XLA usa el algoritmo de
[Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376)
(Henry, Tang y Heinecke) para representar cada significando `float32` de 24 bits
como una suma de tres valores `bfloat16` de 8 bits (`High + Med + Low`) y
expandir su producto:

```text
(a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
```

Multiplicar dos sumas de 3 términos produce nueve productos `bfloat16`, de los
cuales los tres términos de menor orden siempre producen subdesbordamiento en
`float32`:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|          | High        | Med                      | Low                      |
| :------- | :---------- | :----------------------- | :----------------------- |
| **High** | High x High *(1 pasada)* | High x Med *(3 pasadas)*    | High x Low *(6 pasadas)*    |
| **Med**  | Med x High *(3 pasadas)* | Med x Med *(6 pasadas)*     | Med x Low *(subdesbordamiento)* |
| **Low**  | Low x High *(6 pasadas)* | Low x Med *(subdesbordamiento)* | Low x Low *(subdesbordamiento)* |
<!-- mdformat on -->

### Evalúa el rendimiento de 3 y 6 pasadas { #evaluate-3-pass-and-6-pass-performance }

Aunque `Precision.HIGH` (3 pasadas) y `Precision.HIGHEST` (6 pasadas) requieren
3 y 6 veces más pasadas por la MXU, el impacto de extremo a extremo depende de
tu carga de trabajo:

*   **Dimensiones de contracción grandes:** En los modelos modernos entrenados
    con acumulación en `float32` (como DeepSeek-V3) que tienen dimensiones de
    contracción grandes, el ruido de acumulación y el no determinismo en miles
    de elementos dominan la contribución de los términos cruzados de mantisa de
    menor orden, por lo que 1 pasada (`Precision.DEFAULT`) es suficiente.
*   **Modelos limitados por el ancho de banda de memoria:** Los modelos
    `float32` más antiguos a menudo no realizan suficientes operaciones
    aritméticas por byte transferido desde la memoria de alto ancho de banda
    (HBM) para saturar la MXU (o los Tensor Cores de la GPU). En estas cargas de
    trabajo limitadas por la memoria, las pasadas adicionales por la MXU de la
    ejecución de 3 y 6 pasadas usan ciclos de la MXU que, de otro modo,
    estarían inactivos, lo que genera una ralentización de extremo a extremo
    inferior a 3 o 6 veces.

## Valida el comportamiento de la precisión bit a bit { #validate-bitwise-precision-behavior }

Puedes verificar cómo la MXU maneja los bits de mantisa y la precisión
excedente si construyes entradas con dos reglas binarias de punto flotante:

*   `1 + 2**-n` crea un número con un significando binario de
    `1.000...1000...`, que coloca un `1` fraccionario en el bit de mantisa
    número `n`.
*   `(1 + m) * (1 + n) = 1 + m + n + mn`: Multiplicar dos números con bits
    activados en las posiciones de mantisa 6 y 7 produce un resultado con bits
    activados en las posiciones 6, 7 y `6 + 7 = 13`. Como `bfloat16` solo
    contiene 7 bits de mantisa, la posición 13 desborda `bfloat16` y solo se
    conserva cuando se mantiene en el acumulador `float32`:

```py
value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
```

Si encapsulas estos valores en matrices de `256 x 256`, se activa la ejecución
en la MXU y puedes inspeccionar si el bit de mantisa 13 se pierde por redondeo
o se conserva:

```python
# Python uses float64 by default.
bit_6 = 1 + 2**-6
bit_7 = 1 + 2**-7
bits_6_7 = 1.0 + 2**-6 + 2**-7
bits_6_7_13 = 1.0 + 2**-6 + 2**-7 + 2**-13

print("=== Python CPU values ===")
print(f"{bit_6=:.20f}")
print(f"{bit_7=:.20f}")
print(f"{bits_6_7=:.20f}")
print(f"{bits_6_7_13=:.20f}")
print(f"{(bit_6 * bit_7)=:.20f}")
assert bit_6 * bit_7 == bits_6_7_13

print("=== PyTorch TPU values ===")

eye = torch.eye(256, dtype=torch.bfloat16, device=device)
bit_6_tensor = eye.clone() * bit_6
bit_7_tensor = eye.clone() * bit_7
bits_6_7_tensor = eye.float().clone() * bits_6_7
bits_6_7_13_tensor = eye.float().clone() * bits_6_7_13

# Use a 1-element vector rather than a 0D scalar because PyTorch's type
# promotion rules ignore 0D scalar tensor dtypes when paired with 2D tensors.
one_float32 = torch.tensor([1.0], dtype=torch.float32, device=device)

print(f"{bit_6_tensor[0,0]=:.20f}")
print(f"{bit_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_13_tensor[0,0]=:.20f}")

def mul_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
  return a @ b * c

# In eager execution, `a @ b` rounds to bfloat16 before `* c`, dropping bit 13.
print("=== TPU values: Eager ===")
eager_res = mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
print(f"{eager_res=:.20f}")
assert eager_res == bits_6_7_tensor[0, 0]

# With allow_excess_precision = False, the compiled graph also rounds `a @ b`
# to bfloat16 before multiplying by `c`, dropping bit 13.
print("=== TPU values: compiled with allow_excess_precision = False ===")
torch.compiler.reset()
torch.tpu.allow_excess_precision = False
mul_add_compiled_false = torch.compile(mul_add, backend="tpu")
print(f"{torch.tpu.allow_excess_precision=}")
res_false = mul_add_compiled_false(bit_6_tensor, bit_7_tensor, one_float32)[
    0, 0
]
print(f"{res_false=:.20f}")

try:
  assert res_false == bits_6_7_tensor[0, 0]
  print("  [PASSED] Successfully disabled excess precision.")
except AssertionError:
  print(
      "  [WARNING] Failed to disable excess precision (XLA compiled another"
      " graph with default 'True' earlier in this session)."
  )

# Changing allow_excess_precision after XLA compilation has already started in
# this process raises a RuntimeError because compiler options are locked on
# the first compilation.
#
# When `torch.tpu.allow_excess_precision = True` is set at the start of a
# fresh Python session before any TPU operations or compilations run, XLA
# keeps `a @ b` in float32 when fusing with `* c`, preserving the 13th
# mantissa bit (res_true == bits_6_7_13_tensor[0, 0] == 1.0235595703125).
print("=== TPU values: compiled with allow_excess_precision = True ===")
torch.compiler.reset()
try:
  torch.tpu.allow_excess_precision = True
  mul_add_compiled_true = torch.compile(mul_add, backend="tpu")
  print(f"{torch.tpu.allow_excess_precision=}")
  res_true = mul_add_compiled_true(bit_6_tensor, bit_7_tensor, one_float32)[
      0, 0
  ]
  print(f"{res_true=:.20f}")
  assert res_true == bits_6_7_13_tensor[0, 0]
  print("  [PASSED] Successfully enabled excess precision.")
except (RuntimeError, AssertionError) as e:
  print(
      "  [EXPECTED] Cannot change excess precision after first compilation:"
      f" {e}"
  )
```
