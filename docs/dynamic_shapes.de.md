# Dynamische Shapes { #dynamic-shapes }

<!-- Link to colab and to generated notebook -->

[![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/dynamic_shapes.ipynb)

Wenn Sie ein Modell mit `torch.compile(model)` kompilieren, lowert TorchTPU die
Berechnung nach StableHLO und kompiliert sie mit dem XLA-Compiler
(siehe [Funktionsweise von TorchTPU](./how_torchtpu_works.md)). Der
XLA-Compiler ist traditionell auf **statisch bekannte Tensor-Shapes** ausgelegt.
So lassen sich Puffer im High Bandwidth Memory (HBM) im Voraus anordnen,
Operator-Pipelines fusionieren und Matrixmultiplikationen für die systolischen
Arrays (MXUs) der TPU in Tiles zerlegen. Wenn sich die Dimensionen der
Eingabe-Tensoren in einer `torch.compile`-Arbeitslast von Aufruf zu Aufruf ändern,
würde eine Kompilierung für jede exakte Shape die Shape-Guards von Dynamo
ungültig machen und bei jeder neuen Shape eine aufwendige vollständige
Neukompilierung durch XLA auslösen.

Um **dynamische Shapes** ohne ständige Neukompilierungen zu unterstützen,
verwendet TorchTPU **Bounded Dynamism** (begrenzte Dynamik). Dabei sind die
Obergrenzen der dynamischen Dimensionen zur Kompilierzeit festgelegt, während
XLA die tatsächlichen Dimensionsgrößen zur Laufzeit dynamisch verfolgt.

## Wie TorchTPU dynamische Shapes verarbeitet { #how-dynamic-shapes-are-handled-by-torchtpu }

Wenn Sie ein Modell oder eine Funktion mit der Standarddynamik kompilieren, also
mit `torch.compile(model, dynamic=None)` (wenn sich die Eingabe-Shapes zwischen
Aufrufen ändern), oder dynamische Dimensionen explizit mit
`torch._dynamo.mark_dynamic` markieren, tract Dynamo die Berechnung und erzeugt
einen FX-Graphen. Darin werden sich ändernde Dimensionen als symbolische
Integer-Variablen (`torch.SymInt`) statt als feste Konstanten dargestellt.
TorchTPU verarbeitet diese symbolischen Dimensionen so:

### Kompilierung { #compilation }

*   TorchTPU fängt den getracten FX-Graphen ab und bestimmt für jede symbolische
    Dimension einen Gesamtbereich `[lower_bound, upper_bound]` (aus
    `mark_dynamic(..., min=..., max=...)` oder von Dynamo aus der anfänglichen
    Laufzeitgröße `S` abgeleitet). Diesen Bereich teilt TorchTPU in
    zusammenhängende Buckets auf, die an Vielfachen von `bucket_size` (Standard
    `128`, passend zur Tile-Dimension der TPU mit 128 Lanes) ausgerichtet sind,
    bis zu `num_dynamic_buckets` (Standard `1`, maximal `8`):
    *   Der erste Bucket reicht von `lower_bound` bis zum nächsten Vielfachen von
        `128` bei oder nach `S` (`U_1`), der zweite Bucket von `U_1 + 1` bis zum
        nächsten Vielfachen von `128` (`U_2`) und so weiter (zum Beispiel
        `[lower_bound, 128]`, `[129, 256]`, ...).
    *   TorchTPU registriert bei Dynamo einen einzigen Bereichs-Guard über die
        gesamte Spanne aller Buckets. Dynamo löst daher nur dann eine
        Neukompilierung aus, wenn die Dimension über die Gesamtobergrenze
        `upper_bound` hinauswächst.
*   Für jeden Bucket ersetzt TorchTPU die symbolische Dimension durch die
    statische Obergrenze `U` des Buckets, übergibt die tatsächliche
    Dimensionsgröße zur Laufzeit als Eingabe und lowert den Graphen nach
    StableHLO. Dabei werden begrenzte Tensor-Typen verwendet, die in allen
    Operationen sowohl die statische Obergrenze `U` als auch die aktive
    Laufzeitgröße verfolgen.
*   XLA kompiliert das resultierende Modul für jede konfigurierte
    Bucket-Obergrenze `U`.

### Ausführung { #execution }

*   Zur Laufzeit wählt TorchTPU anhand der tatsächlichen Größe der dynamischen
    Dimension die passende kompilierte Bucket-Executable aus.

*   Bei der Ausführung des kompilierten Moduls füllt XLA den dynamischen
    Eingabe-Tensor bis zur statischen Obergrenze `U` des Buckets auf (Padding).
    Die Werte im aufgefüllten Bereich jenseits der aktiven Laufzeitgröße setzt
    XLA auf algebraische Neutralelemente zurück. So beeinflussen die
    aufgefüllten Elemente das Ergebnis nicht.

*   Bei dynamischen Ausgabe-Tensoren bleibt das Padding am Ende des Puffers.
    Dadurch kann TorchTPU den Puffer mit einer einfachen Aktualisierung der
    Shape-Metadaten an PyTorch zurückgeben, statt den Tensor physisch zu
    slicen.

## Motivierender Anwendungsfall: autoregressives Decoding { #motivating-use-case-autoregressive-decoding }

Eine zentrale Arbeitslast, die Bounded Dynamism motiviert, ist **autoregressives
LLM-Decoding** mit einem dynamisch wachsenden KV-Cache. In einer
Standard-Decode-Schleife hängt jeder Schritt die Key- und Value-Zustände des
neuen Tokens mit `torch.cat` entlang der Sequenzdimension an den
gecachten Verlauf an:

*   **Prefill:** Der Eingabe-Prompt hat die Shape `[B, S]` (Batchgröße `B`,
    Sequenzlänge des Prompts `S`) und erzeugt einen KV-Cache der Shape
    `[B, H_kv, S, D]` (`H_kv` Key/Value-Heads, Head-Dimension `D`).
*   **Decode-Schritt 1:** Das Eingabe-Token hat die Shape `[B, 1]` und wird mit
    dem Cache verkettet, was die Shape `[B, H_kv, S + 1, D]` ergibt.
*   **Decode-Schritt `t`:** Die Sequenzlänge des KV-Cache wächst auf `S + t`.

Bei Kompilierung mit statischen Shapes (`dynamic=False`) setzt Dynamo einen
Guard auf die exakte Sequenzlänge des KV-Cache. Jeder Decode-Schritt ändert die
Tensor-Shape, macht den Guard ungültig und löst eine **vollständige
Neukompilierung durch XLA für jedes generierte Token** aus. Aus Decode-Schritten
im Millisekundenbereich werden so Kompilierungen von mehreren Sekunden, sofern
Sie das Modell nicht so umschreiben, dass es statische Puffer vorab alloziert
und In-place-Mutationen ausführt.

**Bounded Dynamism** löst dieses Problem, ohne dass Sie das Modell umschreiben
müssen: Während der KV-Cache über die Decode-Schritte wächst (`S + 1`, `S + 2`,
...), verwenden alle Shapes innerhalb des kompilierten Buckets (`<= 128`,
`<= 256`, ...) dieselbe XLA-Executable.

Eine praktische Anleitung, die Bounded Dynamism auf eine vollständige
LLM-Decode-Schleife anwendet (und mit der statischen Vorab-Allokation des
KV-Cache vergleicht), finden Sie unter
[LLM-Inferenz von Anfang bis Ende](./case_studies_muse_glimmer_30b_inference.md).

## Beispiel: autoregressive Schleife mit wachsendem Tensor { #example-autoregressive-loop-with-a-growing-tensor }

Wie sich statische Kompilierung (`dynamic=False`) und Bounded Dynamism mit der
Standarddynamik (`dynamic=None`) in der Praxis verhalten, zeigt eine minimale
autoregressive Schleife. Darin hängt jeder Schritt mit `torch.cat` ein neues
Token entlang der Sequenzdimension (`dim=2`) an:

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

### 1. Ausführung mit `dynamic=False` { #1-running-with-dynamicfalse }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()
torch._dynamo.config.recompile_limit = 64

step_static = torch.compile(decode_step, backend="tpu", dynamic=False)
run_decode_loop(step_static, prefix="[dynamic=False] ")
```

Beispielausgabe:

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

**Was passiert ist:** Mit `dynamic=False` setzt Dynamo einen Guard auf die
exakte Sequenzlänge von `kv_cache` (`16`, `17`, `18`, ...). Da `torch.cat`
`dim=2` bei jeder Iteration um `1` erhöht, schlägt der Shape-Guard bei jedem
Aufruf fehl und löst bei jedem Schritt eine vollständige Neukompilierung durch
XLA aus.

### 2. Ausführung mit `dynamic=None` (Bounded Dynamism) { #2-running-with-dynamicnone-bounded-dynamism }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()

step_dynamic = torch.compile(decode_step, backend="tpu", dynamic=None)
run_decode_loop(step_dynamic, prefix="[dynamic=None] ")
```

Beispielausgabe:

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

**Was passiert ist:** Mit `dynamic=None` fällt nur in den ersten 3 Iterationen
Kompilierungsaufwand an. Ab Schritt 4 sinkt die Ausführungszeit auf etwa 0,4 ms
pro Schritt (etwa 100-mal schneller), ohne jede Neukompilierung:

*   **Schritt 1 (`seq_len=16 -> 17`):** Dynamo sieht die Eingaben zum ersten
    Mal und kompiliert eine statische Executable für `seq_len=16`.
*   **Schritt 2 (`seq_len=17 -> 18`):** Da sich `dim=2` von `16` auf `17`
    geändert hat, markiert Dynamo `dim=2` automatisch als dynamisch (`SymInt`).
    TorchTPU kompiliert mit dem anfänglichen Layout des Eingabe-Tensors eine
    begrenzt dynamische Executable für den Bucket `[17, 128]`.
*   **Schritt 3 (`seq_len=18 -> 19`):** Der begrenzt dynamische Ausgabepuffer
    aus Schritt 2 hat das aufgefüllte Ausgabe-Layout von XLA. Deshalb kompiliert
    TorchTPU den Bucket `[17,
    128]` noch einmal für dieses stabile XLA-Layout.
*   **Schritte 4–10 (`seq_len=19 -> 26`):** Sowohl die Sequenzlänge (innerhalb
    des Buckets `<= 128`) als auch das Pufferlayout von XLA passen zur
    gecachten Executable. Jeder weitere Schritt ruft daher direkt die
    vorkompilierte XLA-Executable auf, ohne neu zu kompilieren.

> **Tip:** Wenn Sie schon vorher wissen, welche Dimension sich ändert, können
> Sie vor dem ersten Schritt
> `torch._dynamo.mark_dynamic(kv_cache, 2, min=16, max=128)` aufrufen. Damit
> weisen Sie Dynamo an, `dim=2` schon in Schritt 1 als dynamisch zu tracen. Die
> anfängliche statische Kompilierung in Schritt 1 entfällt, und Sie können
> explizite Grenzen `[min, max]` angeben.

## Aktuelle Einschränkungen { #current-limitations }

Bounded Dynamism in TorchTPU wird aktiv weiterentwickelt und ist derzeit auf
den Anwendungsfall **autoregressiver KV-Cache** optimiert. Beachten Sie die
folgenden Einschränkungen, wenn Sie es auf Ihre Modelle anwenden:

*   **Bucket-Limit und unbegrenzte Dynamik:** Pro Trace werden höchstens
    **8 Buckets** (`num_dynamic_buckets <= 8`) kompiliert. Unbegrenzte Dynamik
    wird noch nicht unterstützt. Wenn eine dynamische Dimension über die
    Obergrenze des höchsten kompilierten Buckets hinauswächst, wird eine
    Neukompilierung ausgelöst.
*   **Shape-Einschränkungen von Dynamo:** Die Bucket-Obergrenzen müssen die
    aktiven Shape-Einschränkungen von Dynamo erfüllen (etwa
    Teilbarkeitsanforderungen aus `reshape` oder `view`). Andernfalls greift
    TorchTPU auf einen einzelnen Bucket mit der Obergrenze `2 * S` zurück (das
    Doppelte der anfänglichen Laufzeitgröße `S`).
*   **Rechen- und HBM-Overhead bis zur Bucket-Obergrenze:** Da die Ausführung
    auf Puffern läuft, die bis zur statischen Obergrenze `U` des Buckets
    aufgefüllt sind, skalieren HBM-Bedarf und FLOPs mit `U` statt mit der
    tatsächlichen Laufzeitgröße. Wenn Sie eine sehr große `bucket_size` oder
    `mark_dynamic(..., max=U)` festlegen, die tatsächliche Größe aber viel
    kleiner als `U` ist, zahlen Sie die Rechen- und Speicherkosten von `U`.
*   **Änderungen am XLA-Layout:** Auch wenn die Shape eines Tensors in einen
    bereits kompilierten Bucket fällt, löst eine Änderung des zugrunde liegenden
    physischen XLA-Layouts eines Eingabepuffers eine Neukompilierung des Buckets
    aus.
*   **Kein persistenter Kompilierungscache (`AOTAutogradCache`):** Begrenzt
    dynamische Executables werden noch nicht in `AOTAutogradCache` serialisiert.
    Sie werden daher bei jedem Start eines neuen Prozesses neu kompiliert.
*   **Operator-Abdeckung:** Noch nicht alle PyTorch-Operatoren unterstützen
    begrenzt dynamische Shapes.
*   **Views als Modelleingaben:** Die Unterstützung für die direkte Übergabe von
    Tensor-Views als Eingaben an das kompilierte Modell ist eingeschränkt.
*   **Datenabhängige Dynamik:** Operationen, deren Ausgabe-Shapes von
    Tensorwerten zur Laufzeit abhängen (etwa `torch.nonzero`), werden nicht
    unterstützt.
*   **Eigene Kernels:** Begrenzt dynamische Shapes werden mit eigenen
    TPU-Kernels (etwa Pallas-Kernels, die mit `torch.tpu.pallas.jax_op`
    registriert wurden) nicht unterstützt.
*   **dynamic=True:** Der Modus torch.compile() *dynamic=True* wird nicht
    unterstützt.
