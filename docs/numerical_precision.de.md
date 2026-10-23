# Numerische Genauigkeit in TorchTPU steuern { #manage-numerical-precision-in-torchtpu }

Google TPUs verteilen Gleitkommaoperationen auf zwei Hardwareeinheiten, um
Durchsatz und numerische Genauigkeit auszubalancieren:

*   **Matrix Multiply Unit (MXU):** Führt kontrahierende Operationen (etwa
    `torch.matmul`, `torch.mm`, `torch.bmm` und Faltungen) mit nativen
    `bfloat16`-Eingaben (8-Bit-Signifikand) und 32-Bit-Akkumulation in
    `float32` aus.
*   **Vector Execution Unit (VPU):** Führt nicht kontrahierende Operationen
    (etwa elementweise Arithmetik, Aktivierungen, Layer-Normalisierung und
    Softmax) in vollem IEEE-754-`float32` (24-Bit-Signifikand) aus.

In diesem Leitfaden erfahren Sie, wie Sie die Genauigkeitsmodi für
`float32`-Matrixmultiplikationen konfigurieren, die numerische Parität zwischen
CPU und TPU prüfen, die Akkumulatortypen der Ausgabe und die Excess Precision
(überschüssige Genauigkeit) steuern und das bitweise Verhalten der MXU im Code
validieren.

### Genauigkeit von PyTorch auf CUDA und TPU vergleichen { #compare-pytorch-precision-on-cuda-and-tpu }

TorchTPU unterstützt die Standard-APIs von PyTorch für die Genauigkeit
(`torch.set_float32_matmul_precision` und `torch.get_float32_matmul_precision`)
sowie TPU-spezifische Steuerelemente (`torch.tpu.Precision` und
`torch.tpu.precision`):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Aspekt          | PyTorch auf CUDA                      | PyTorch auf TPU (TorchTPU)              |
| :----------------- | :----------------------------------- | :------------------------------------- |
| **Standard-Genauigkeitsmodus** | `"highest"` (reines `float32`) | `"medium"` (`Precision.DEFAULT`)       |
| **Beschleuniger-Primitive für Fast Math** | Tensor Cores (`TF32`) | Systolische MXUs (`bfloat16`)             |
| **Bits bei Fast Math** | 1 Vorzeichen- + 8 Exponenten- + 10 Mantissenbits (19-Bit-`TF32`) | 1 Vorzeichen- + 8 Exponenten- + 7 Mantissenbits (16-Bit-`bfloat16`) |
| **Genauigkeit der Akkumulation** | `float32` (`"highest"`, `"high"`), `TF32` (`"medium"`) | `float32` in allen Modi |
| **Mechanismus für höhere Genauigkeit** | Dedizierte Hardwareeinheiten für `float32`/`TF32` | Zerlegung in mehrere Durchläufe (3 oder 6 MXU-Durchläufe) |
| **Auswirkung von `high`/`highest` auf die Performance** | Basisgeschwindigkeit oder leichte Verlangsamung (~1–2x) | ~3x MXU-Rechenaufwand für `"high"`, ~6x für `"highest"` |
| **API für den Geltungsbereich** | Global `torch.set_float32_matmul_precision` | Global <br> `torch.set_float32_matmul_precision` + bereichsbezogen `torch.tpu.precision` |
<!-- mdformat on -->

## Genauigkeitsmodi festlegen { #specify-precision-modes }

Wenn `float32`-Tensoren auf der MXU multipliziert werden, ordnet TorchTPU die
Genauigkeitsstufen von PyTorch dem Enum `torch.tpu.Precision` zu:

*   `"medium"` / `Precision.DEFAULT` (1 Durchlauf): **Standard.** Rundet
    `float32`-Eingaben auf `bfloat16` (multipliziert die obersten 8 Bits des
    Signifikanden) und akkumuliert in `float32`. Empfohlen für fast alle
    Trainings- und Inferenz-Arbeitslasten.
*   `"high"` / `Precision.HIGH` (3 Durchläufe): Zerlegt `float32`-Eingaben in
    hohe und niedrige `bfloat16`-Komponenten und berechnet drei Kreuzterme
    (multipliziert die ersten 16 Bits des Signifikanden). Der
    MXU-Rechenaufwand steigt dabei etwa auf das Dreifache.
*   `"highest"` / `Precision.HIGHEST` (6 Durchläufe): Zerlegt
    `float32`-Eingaben in drei `bfloat16`-Komponenten und berechnet sechs
    Kreuzterme (multipliziert alle 24 Bits des Signifikanden), um die volle
    IEEE-754-Genauigkeit von `float32` anzunähern. Der MXU-Rechenaufwand steigt
    dabei etwa auf das Sechsfache.

Sie können die Genauigkeit der Matrixmultiplikation auf zwei Arten
konfigurieren:

*   **Globale Konfiguration:** Rufen Sie `torch.set_float32_matmul_precision("medium"
    | "high" | "highest")` auf, um den Standardmodus für den Prozess
    festzulegen.
*   **Bereichsbezogener Kontextmanager:** Verwenden Sie
    `with torch.tpu.precision(mode):`, um eine höhere Genauigkeit nur auf
    numerisch empfindliche Schichten oder Untermodule anzuwenden.
*   **Performance-Warnung:** Wenn Sie `"high"` oder `"highest"` auswählen, wird
    einmalig eine Warnung ausgegeben (`TORCH_WARN_ONCE`). Behalten Sie den
    Standardmodus `"medium"` bei, sofern Sie keine Konvergenzprobleme
    beobachten, die mit der Genauigkeit zusammenhängen.

> **Note:** Das Betreten oder Verlassen von `torch.tpu.precision` verursacht
> einen Graph-Break in `torch.compile`. Wenn Sie
> `torch.compile(..., fullgraph=True)` verwenden, legen Sie die Genauigkeit
> außerhalb der kompilierten Funktion mit
> `torch.set_float32_matmul_precision(...)` fest oder umschließen Sie den
> äußeren Aufruf mit `with
> torch.tpu.precision(...):`.

Mit `torch.tpu.precision` können Sie die Ausführungszeit im eingeschwungenen
Zustand für die drei Modi vergleichen:

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

### Genauigkeit der Modi vergleichen { #compare-accuracy-across-precision-modes }

Vergleichen Sie die TPU-Ausgaben mit einer `float32`-Referenz auf der CPU für
dieselben Eingaben, um zu sehen, wie stark jeder Genauigkeitsmodus den
numerischen Fehler verringert:

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

## Numerische Parität prüfen { #verify-numerical-parity }

Wenn Sie ein Modell auf die TPU portieren oder Unterschiede bei der Konvergenz
debuggen, prüfen Sie die numerische Parität: Führen Sie dieselbe Schicht oder
dasselbe Untermodul auf CPU und TPU mit synchronisierten Gewichten und
identischen Eingaben aus und vergleichen Sie die Ausgaben dann mit
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

### Detaillierte Abweichungen interpretieren { #interpret-detailed-mismatches }

Wenn Tensoren die Toleranzen überschreiten, gibt `torch.testing.assert_close`
eine Zusammenfassung aus. Damit können Sie normale Gleitkommaabweichungen von
einem Logikfehler unterscheiden:

```text
Tensor-likes are not close!

Mismatched elements: 1 / 50 (2.0%)
Greatest absolute difference: 0.006596 at index (3, 9) (up to 1e-05 allowed)
Greatest relative difference: 0.165375 at index (3, 9) (up to 0.016 allowed)
```

*   **Mismatched elements:** Ein hoher Prozentsatz von Elementen mit kleinen
    absoluten Abweichungen deutet meist auf übliche Rundungsunterschiede oder
    eine andere Akkumulationsreihenfolge durch Compiler-Fusionen hin, nicht auf
    einen Fehler.
*   **Greatest absolute / relative difference:** Gibt den größten Fehler und
    seinen Tensor-Index an. So können Sie Ausreißer untersuchen oder passende
    Schwellenwerte für `atol` und `rtol` wählen.

### Best Practices für die Parität { #best-practices-for-parity }

*   **Verwenden Sie auf beiden Geräten dieselben Datentypen:** Vergleichen Sie
    `bfloat16` auf der TPU mit `bfloat16` auf der CPU oder `float32` auf der
    TPU (mit `Precision.HIGHEST`) mit `float32` auf der CPU. Wenn Sie
    `bfloat16` mit `float32` vergleichen, messen Sie den Quantisierungsfehler
    statt der Abweichung durch die Hardware.
*   **Prüfen Sie neben Aktivierungen auch Gradienten:** Vergleichen Sie nach dem Aufruf
    von `loss.backward()` nach demselben Muster
    `model_tpu.weight.grad.cpu()` mit `model_cpu.weight.grad`.
*   **Beginnen Sie mit den Standardtoleranzen:** Verwenden Sie die
    Standardtoleranzen aus der
    [Testdokumentation von PyTorch](https://docs.pytorch.org/docs/stable/testing.html)
    als Ausgangspunkt und lockern Sie `atol` oder `rtol` nur dort, wo dies
    wegen einer Akkumulation mit geringerer Genauigkeit zu erwarten ist.

## Ausgabegenauigkeit festlegen { #specify-output-precision }

Wenn Sie `bfloat16`-Tensoren mit `torch.mm` oder `torch.bmm` multiplizieren,
akkumuliert die MXU die Produkte in `float32` und rundet die endgültige Ausgabe
dann wieder auf `bfloat16`. Übergeben Sie `out_dtype=torch.float32`, um den
`float32`-Akkumulator direkt und ohne Downcast zurückzugeben:

```python
a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device=device)
# Using out_dtype to force float32 output accumulator precision
res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
print(f"Input dtype: {a_bmm.dtype}")
print(f"Output dtype: {res_bmm.dtype}")
```

## Excess Precision steuern { #control-excess-precision }

Da die MXU der TPU `bfloat16`-Matrixmultiplikationen in `float32` akkumuliert,
kann der XLA-Compiler fusionierte Operationen optimieren, indem er
Zwischenwerte in `float32` behält, statt sie zwischen den Schritten auf
`bfloat16` zu runden. Das Beibehalten dieser zusätzlichen Zwischenbits heißt
*Excess Precision* (überschüssige Genauigkeit).

Sie können dieses Verhalten global mit `torch.tpu.allow_excess_precision`
konfigurieren. Diese Einstellung setzt das zugrunde liegende Compiler-Flag
`--xla_allow_excess_precision` und ist standardmäßig `False`:

*   **Excess Precision deaktivieren (`False`, Standard):** Setzen Sie
    `torch.tpu.allow_excess_precision = False`, damit der Compiler die
    Downcasts von Zwischenwerten auf `bfloat16` beibehält. So entsprechen
    fusionierte Operationen dem Rundungsverhalten von `bfloat16` im
    Eager-Modus.
*   **Excess Precision zulassen (`True`):** Setzen Sie `torch.tpu.allow_excess_precision =
    True`, damit der Compiler Downcasts von Zwischenwerten weglassen und
    fusionierte Zwischenergebnisse in `float32` behalten kann.

> **Important:** Legen Sie `torch.tpu.allow_excess_precision` ganz am Anfang
> Ihres Skripts fest, bevor Sie TPU-Operationen ausführen oder kompilieren. Die
> XLA-Laufzeit sperrt dieses Flag zu Beginn der ersten Kompilierung. Wenn Sie
> danach versuchen, einen anderen Wert festzulegen, wird ein `RuntimeError`
> ausgelöst.

## Genauigkeit der TPU-MXU verstehen { #understand-tpu-mxu-precision }

Wenn Sie verstehen, wie die MXU der TPU Gleitkommazahlen verarbeitet, wird
klar, warum Excess Precision und die `float32`-Modi mit mehreren Durchläufen so
funktionieren, wie sie es tun:

*   **Wertebereich und Genauigkeit (`bfloat16`):**
    [Untersuchungen von Google Cloud zu `bfloat16`](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
    haben gezeigt, dass neuronale Netze mit reduzierter Mantissengenauigkeit
    trainiert werden können, ohne an Modellgenauigkeit zu verlieren, solange
    sie den großen Dynamikbereich von `float32` beibehalten (entscheidend für
    Operationen wie Normen und Gradienten). `float32` hat 23 Mantissenbits
    (24 Bits Signifikandengenauigkeit einschließlich des impliziten führenden
    Bits). `bfloat16` behält denselben 8-Bit-Exponentenbereich, hat aber nur 7
    Mantissenbits (8 Bits Signifikandengenauigkeit, was einem
    Maschinen-Epsilon von `2**-7` oder ~0,78 % entspricht).
*   **Austauschformate und Rechenformate:** Gleitkommasysteme unterscheiden
    zwischen **Austauschformaten** (wie Zahlen im Speicher abgelegt werden) und
    **Rechenformaten** (wie die Hardware Zwischenoperationen berechnet und
    rundet, etwa mit den 3 zusätzlichen Guard-, Round- und Sticky-Bits in einer
    Standard-ALU). `bfloat16` kann beides bezeichnen.
*   **Akkumulation entlang der kontrahierten Dimension:** Die Multiplikation
    zweier Matrizen der Shape `[I, J] @ [J, K]` berechnet ein Raster von
    Skalarprodukten der Shape `[1, J] @ [J, 1]`. Jedes davon summiert `J`
    elementweise Produkte entlang der **kontrahierten Dimension** `J`. Da `bfloat16` ein Maschinen-Epsilon von
    `2**-7` (~`0.008`) hat, würde die Akkumulation von `J` Produkten in
    `bfloat16` große Rundungsfehler verursachen. Stattdessen nimmt die MXU
    `bfloat16`-Eingaben an (und `fp8` bei neueren Generationen wie TPU v7x) und
    akkumuliert Zwischensummen in `float32`. Das erfüllt dieselbe Funktion wie
    Guard-, Round- und Sticky-Bits in einer ALU vor dem Runden auf das
    Ausgabeformat.
*   **Unterschiedliche Ein- und Ausgabeformate:** Dass die Eingaben in
    `bfloat16` vorliegen und der Akkumulator in `float32` arbeitet, ist die
    Ursache sowohl für *Excess Precision* (Beibehalten von
    `float32`-Zwischenwerten über fusionierte Ops hinweg) als auch für eine
    *geringere Genauigkeit als erwartet* (eine `float32`-Matmul mit einem
    Durchlauf kürzt die Eingaben auf 8-Bit-Signifikanden von `bfloat16`). Im
    Gegensatz dazu arbeitet die Vector Execution Unit (VPU) der TPU bei
    `float32` mit allen 23 Mantissenbits (abgesehen von kleineren
    IEEE-754-Sonderfällen wie subnormalen Zahlen).

### Polynomentwicklung mit mehreren Durchläufen für `float32` { #multi-pass-polynomial-expansion-for-float32 }

Da eine MXU nur `bfloat16`-Eingaben (8-Bit-Signifikand) annimmt, kann sie eine
vollständige 24-Bit-`float32`-Matrixmultiplikation nicht in einem einzigen
Durchlauf berechnen. In der Praxis speichern die meisten modernen Modelle ihre
Gewichte in `bfloat16` und führen selten `float32`-Matmuls aus. Wenn die volle
`float32`-Genauigkeit für Matmuls benötigt wird, verwendet XLA den Algorithmus
aus
[Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376)
(Henry, Tang und Heinecke). Dabei wird jeder 24-Bit-Signifikand von `float32`
als Summe von drei 8-Bit-Werten in `bfloat16` (`High + Med + Low`) dargestellt
und ihr Produkt ausmultipliziert:

```text
(a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
```

Die Multiplikation zweier Summen mit je 3 Termen ergibt neun
`bfloat16`-Produkte. Die drei Terme mit der niedrigsten Ordnung unterlaufen in
`float32` immer:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|          | High        | Med                      | Low                      |
| :------- | :---------- | :----------------------- | :----------------------- |
| **High** | High x High *(1 Durchlauf)* | High x Med *(3 Durchläufe)*    | High x Low *(6 Durchläufe)*    |
| **Med**  | Med x High *(3 Durchläufe)* | Med x Med *(6 Durchläufe)*     | Med x Low *(Unterlauf)* |
| **Low**  | Low x High *(6 Durchläufe)* | Low x Med *(Unterlauf)* | Low x Low *(Unterlauf)* |
<!-- mdformat on -->

### Performance mit 3 und 6 Durchläufen bewerten { #evaluate-3-pass-and-6-pass-performance }

`Precision.HIGH` (3 Durchläufe) und `Precision.HIGHEST` (6 Durchläufe)
benötigen zwar dreimal bzw. sechsmal so viele MXU-Durchläufe, die
Gesamtauswirkung hängt aber von Ihrer Arbeitslast ab:

*   **Große kontrahierte Dimensionen:** In modernen Modellen, die mit
    `float32`-Akkumulation trainiert werden (etwa DeepSeek-V3) und große
    kontrahierte Dimensionen haben, überwiegen das Akkumulationsrauschen und
    der Nichtdeterminismus über Tausende von Elementen den Beitrag der
    Mantissen-Kreuzterme niedrigerer Ordnung. Ein Durchlauf
    (`Precision.DEFAULT`) reicht daher aus.
*   **Durch die Speicherbandbreite begrenzte Modelle:** Ältere `float32`-Modelle
    führen pro aus dem High Bandwidth Memory (HBM) übertragenem Byte oft nicht
    genug Rechenoperationen aus, um die MXU (oder die Tensor Cores einer GPU)
    auszulasten. In diesen speichergebundenen Arbeitslasten nutzen die zusätzlichen
    MXU-Durchläufe für 3 und 6 Durchläufe ansonsten ungenutzte MXU-Zyklen. Die
    Verlangsamung insgesamt liegt daher unter dem Faktor 3 bzw. 6.

## Bitweises Genauigkeitsverhalten validieren { #validate-bitwise-precision-behavior }

Sie können prüfen, wie die MXU mit Mantissenbits und Excess Precision umgeht,
indem Sie Eingaben nach zwei Regeln für binäre Gleitkommazahlen konstruieren:

*   `1 + 2**-n` erzeugt eine Zahl mit dem binären Signifikanden
    `1.000...1000...`, bei der eine gebrochene `1` am `n`-ten Mantissenbit
    steht.
*   `(1 + m) * (1 + n) = 1 + m + n + mn`: Wenn Sie zwei Zahlen mit gesetzten
    Bits an den Mantissenpositionen 6 und 7 multiplizieren, entsteht ein
    Ergebnis mit gesetzten Bits an den Positionen 6, 7 und `6 + 7 = 13`. Da
    `bfloat16` nur 7 Mantissenbits hat, passt Position 13 nicht mehr in
    `bfloat16` und bleibt nur erhalten, wenn sie im `float32`-Akkumulator
    behalten wird:

```py
value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
```

Wenn Sie diese Werte in `256 x 256`-Matrizen einbetten, wird die Ausführung auf
der MXU ausgelöst. So können Sie prüfen, ob das 13. Mantissenbit weggerundet
oder beibehalten wird:

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
