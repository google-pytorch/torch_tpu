# FLUX-Optimierung { #flux-optimization }

<!-- Link to colab and to generated notebook -->

[![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_flux.ipynb)

Diese Fallstudie zeigt, wie wir ausgehend von einer
PyTorch-Referenzimplementierung des `DoubleStreamBlock` aus dem Diffusion
Transformer [FLUX](https://github.com/black-forest-labs/flux) einen
Trainingsschritt mit TorchTPU auf Cloud TPU bei einer Sequenzlänge von 32.768
**1,48-mal schneller** gemacht haben.

Diese Seite ist eine gekürzte Begleitseite zum vollständigen Tutorial. Sie
stellt das Kernkonzept jeder Optimierung mit kurzen, eigenständigen Snippets
vor. Den vollständigen Modellcode, Benchmarks, XProf-Traces und Tests finden
Sie hier:

*   [FLUX-Optimierungstutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md):
    vollständige Anleitung zu jedem Meilenstein, einschließlich XProf-Traces und
    Anweisungen zum Reproduzieren aller Benchmarks.
*   [Eigene Pallas-Kernels für PyTorch schreiben](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md):
    Schritt-für-Schritt-Anleitung zum Pallas-Kernel SplashAttention, der in
    Meilenstein 4 verwendet wird.

Die Implementierung basiert auf der
[TorchTitan-Implementierung von FLUX](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)
und dem Repository
[black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/).

Wenn Sie mit den Beispielen noch nicht vertraut sind, sollten Sie sich zuerst
[in unseren anderen Dokumenten](./get_started_on_gke.md) mit dem Ausführen von
Beispielen auf GKE vertraut machen.

## Ergebnisse im Überblick { #results-at-a-glance }

Mittlere Dauer eines Trainingsschritts (Forward + Loss + Backward +
Optimizer-Schritt) für einen einzelnen `DoubleStreamBlock`:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Konfiguration | Mittlere Schrittdauer | Beschleunigung ggü. Baseline |
| :--- | ---: | ---: |
| Referenz-Baseline (Eager) | 178,61 ms | 1,00x |
| Nur Pallas SplashAttention (Eager) | 149,81 ms | 1,19x |
| Nur Zero-Copy-Views (Eager) | 178,30 ms | 1,00x |
| Nur `torch.compile` | 159,33 ms | 1,12x |
| Zero-Copy-Views + fusioniertes RoPE + Pallas SplashAttention (Eager) | 156,73 ms | 1,14x |
| Zero-Copy-Views + `torch.compile` | 159,08 ms | 1,12x |
| Zero-Copy-Views + fusioniertes RoPE + `torch.compile` | 153,60 ms | 1,16x |
| **Zero-Copy-Views + fusioniertes RoPE + Pallas SplashAttention + `torch.compile`** | **122,51 ms** | **1,46x** |
<!-- mdformat on -->

**Messaufbau:**

-   **Hardware:** ein einzelner TPU-v6e-Chip (32 GiB HBM).
-   **Software:** Python 3.14, PyTorch 2.15 (Vorabversion), JAX 0.12, einops
    0.8.2 und aus dem Quellcode erstelltes TorchTPU.
-   **Modell-Shape:** `batch_size=2`, `hidden_size=1024`, `num_heads=8`
    (`head_dim=128`), `img_seq_len=32256`, `txt_seq_len=512` (Gesamtlänge der
    Sequenz 32.768).
-   **Genauigkeit:** Blockparameter und Eingaben in bfloat16.
-   **Blockgrößen von SplashAttention:** 2048 für `block_q`, `block_kv` und die
    dK/dV-Backward-Tiles.
-   **Zeitmessung:** 3 Warm-up-Schritte, gefolgt von 10 gemessenen Schritten.
    Die gemessenen Schritte werden direkt nacheinander dispatcht, und
    `torch.accelerator.synchronize()` läuft einmal nach dem letzten Schritt. Die
    mittlere Schrittdauer (Gesamtzeit geteilt durch die Anzahl der Schritte)
    enthält also die Ausführung auf dem Gerät. Der Profiler ist während der
    gemessenen Schritte deaktiviert.

Bei dieser Sequenzlänge dominiert Attention den Schritt. Daher bringen
SplashAttention und Kompilierung die größten Gewinne. Bei kürzeren
Sequenzlängen ist die Kompilierung wichtiger (etwa 3x bei einer Sequenzlänge
von 640) und der Attention-Kernel weniger wichtig.

Die Erkenntnisse lassen sich auf die meisten Transformer-Modelle auf TPU
übertragen:

1.  **Versteckte Layout-Kopien vermeiden**: Bevorzugen Sie Zero-Copy-Views
    (`.view()`, `torch.unbind()`) gegenüber Umordnungen, die neue HBM-Puffer
    materialisieren.
2.  **Weniger speichern, mehr berechnen**: Drücken Sie Positions-Embeddings als
    kompakte elementweise Berechnung aus statt als materialisierte Matrizen.
3.  **Den ganzen Block kompilieren**: Mit `torch.compile()` kann XLA Ketten
    elementweiser Operationen fusionieren und Zwischen-Roundtrips über HBM
    entfernen.
4.  **Bei Bedarf eigene Kernels einsetzen**: Pallas-Kernels geben
    Ihnen explizite Kontrolle über das Tiling auf dem Chip für speichergebundene
    Operationen wie Attention bei langen Sequenzlängen.

### So lesen Sie diese Seite { #how-to-read-this-page }

Jeder Meilenstein wird als Paar von Abschnitten dargestellt:

-   **Vorher (Referenz)**: die ursprüngliche Implementierung und ihr Engpass.
-   **Nachher (optimiert)**: die optimierte Implementierung, gefolgt von einer
    Prüfung, dass sie dasselbe Ergebnis wie die Referenz liefert.

Codezellen sind in ihrer ersten Zeile mit dem passenden Kommentar
`# BEFORE (reference)` oder `# AFTER
(optimized)` gekennzeichnet.

**Voraussetzungen:** Die Referenzzelle in Meilenstein 1 verwendet
[einops](https://github.com/arogozhnikov/einops) (`pip install einops`).

```python
import torch

device = torch.device("tpu")
print(f"Running on {device}.")
```

## Meilenstein 1: Zero-Copy-Tensor-Views { #milestone-1-zero-copy-tensor-views }

Im `DoubleStreamBlock` projizieren der Bild- und der Textstream Hidden States
der Shape `(B, L, 3 * H * D)` auf Queries, Keys und Values. Wie diese Projektion
in Q, K und V aufgeteilt wird, entscheidet darüber, ob die TPU während des
Trainings Daten im HBM kopieren muss.

### Vorher (Referenz): `einops.rearrange` { #before-reference-einopsrearrange }

Die Referenzimplementierung verschiebt die Achse `3` (Q/K/V) nach vorn:

```python
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)
```

Im Forward-Pass gibt `einops.rearrange` Strided Views der Projektion zurück.
Es werden also noch keine Daten verschoben. Die Kosten entstehen im
**Backward-Pass**: Um Gradienten zurück durch die Achsenumordnung zu leiten,
materialisiert Autograd den Gradienten in einem neuen Puffer. Das läuft auf der
TPU in jedem Attention-Block als zusätzliche Kopie von HBM zu HBM (in XProf als
Operation `copy` sichtbar).

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

### Nachher (optimiert): `.view()` + `torch.unbind()` { #after-optimized-view-torchunbind }

Die optimierte Implementierung lässt die Achse `3` neben `(H, D)` und teilt sie
mit `torch.unbind` auf. Der Backward-Pass stapelt die Gradienten von Q, K und V
dann direkt in das ursprüngliche Layout, sodass keine zusätzliche Kopie nötig
ist. In der Ergebnistabelle ist die Auswirkung auf die Schrittdauer gering
(**178,61 ms auf 178,30 ms** im Eager-Modus, **159,33 ms auf 159,08 ms**
kompiliert), da die Kopie nur einen kleinen Teil des Schritts ausmacht. Sie
entfällt aber ohne zusätzliche Kosten.

> **Note:** Die Zellen in diesem Meilenstein zeigen nur, dass beide Ansätze
> dieselben Werte für Q, K und V liefern. Im Forward-Pass geben beide dieselben
> Strided Views zurück, daher sehen ihre Ausgaben identisch aus. Die
> zusätzliche HBM-Kopie durch `einops.rearrange` entsteht nur im Backward-Pass
> und ist in einem XProf-Trace eines vollständigen Trainingsschritts sichtbar,
> nicht in der Ausgabe der Zelle. Die Traces finden Sie in
> [Meilenstein 1 des FLUX-Optimierungstutorials](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-1).

```python
# AFTER (optimized): keep Q/K/V adjacent to (H, D) and unbind.
_b, _l, _ = qkv.shape
q_after, _k, _v = torch.unbind(
    qkv.view(_b, _l, 3, num_heads, head_dim), dim=2
)
print(f"q shape: {tuple(q_after.shape)}")
print(f"Matches reference: {torch.equal(q_after.cpu(), q_before.cpu())}")
```

## Meilenstein 2: Fusioniertes algebraisches RoPE { #milestone-2-fused-algebraic-rope }

FLUX wendet mehrachsige Rotary Position Embeddings (RoPE) auf Queries und Keys
an. Die folgenden Zellen implementieren beide Versionen für eine einzelne
RoPE-Achse.

### Vorher (Referenz): 2x2-Rotationsmatrizen { #before-reference-2x2-rotation-matrices }

Die Referenzimplementierung materialisiert für jedes Frequenzpaar eine
vollständige `2x2`-Rotationsmatrix `[[cos,
-sin], [sin, cos]]` und wendet sie als Matrix-Vektor-Produkt auf umgeformte
6D-Query/Key-Tensoren an. Dadurch werden `cos` und `sin` doppelt gespeichert,
und das zusätzliche Stapeln und Umformen erscheint in XProf als große
Kopierblöcke.

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

### Nachher (optimiert): kompakte `[cos, sin]`-Paare + elementweise Rotation { #after-optimized-compact-cos-sin-pairs-elementwise-rotation }

Die optimierte Implementierung speichert nur kompakte `[cos, sin]`-Paare und
wendet die Rotation direkt mit elementweiser Berechnung an:

```python
# (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
xq_out_0 = xq_0 * cos - xq_1 * sin
xq_out_1 = xq_0 * sin + xq_1 * cos
```

Das halbiert den Speicherbedarf der Positions-Embeddings, wird auf
Fused-Multiply-Add-Befehle abgebildet und bei der Kompilierung mit den
umgebenden Projektions- und Attention-Operationen fusioniert. Zusätzlich zu
Zero-Copy-Views und `torch.compile` verringert das die Schrittdauer von
**159,08 ms auf 153,60 ms (3,4 %)**.

Da beide Versionen dieselben Multiplikationen und Additionen mit denselben
Operanden ausführen, liefern sie identische Ergebnisse. Die folgende Zelle
führt die optimierte Version mit denselben Eingaben aus und vergleicht sie mit
der Referenz:

> **Note:** Diese Zellen zeigen den kleineren Frequenz-Tensor und dass beide
> Versionen identische Ergebnisse liefern. Die Beschleunigung zeigt sich erst,
> wenn RoPE im kompilierten `DoubleStreamBlock` läuft, wo die elementweise
> Rotation mit den umgebenden Operationen fusioniert wird. Die XProf-Traces
> finden Sie in
> [Meilenstein 2 des FLUX-Optimierungstutorials](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-2).

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

## Meilenstein 3: XLA-Kompilierung des gesamten Blocks { #milestone-3-whole-block-xla-compilation }

### Vorher (Referenz): Eager-Ausführung { #before-reference-eager-execution }

Im Eager-Modus startet jede PyTorch-Operation ein eigenes TPU-Programm. Dadurch
laufen Zwischenaktivierungen über HBM hin und zurück, und der Host blockiert
häufig beim Dispatch. In der Ergebnistabelle benötigt der `DoubleStreamBlock`
der Baseline im Eager-Modus **178,61 ms** pro Schritt.

Die folgende Zelle führt das fusionierte RoPE aus Meilenstein 2 im Eager-Modus
aus, Operation für Operation:

```python
# BEFORE (reference): eager execution, one TPU program per op.
q_eager, k_eager = apply_rope(rope_xq, rope_xk, freqs)
print("Ran fused RoPE eagerly.")
```

### Nachher (optimiert): `torch.compile()` { #after-optimized-torchcompile }

Wenn Sie den gesamten `DoubleStreamBlock` mit `torch.compile()` kompilieren,
wird er in einen einzigen XLA-Graphen gelowert. Der Compiler kann dann
elementweise Operationen, Normalisierungen und Residual-Additionen fusionieren:

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

Die Kompilierung senkt die Schrittdauer von **178,61 ms auf 159,33 ms
(1,12x)**. Bei einer Sequenzlänge von 32.768 ist der Gewinn moderat, weil
Attention den Schritt dominiert. Bei einer Sequenzlänge von 640 ist die
kompilierte Version etwa 3x schneller.

Dasselbe Muster gilt für jede Funktion. Die folgende Zelle kompiliert das
fusionierte RoPE und prüft, ob es mit der Eager-Ausführung übereinstimmt:

> **Note:** Diese Zellen prüfen nur, dass die Kompilierung das Ergebnis nicht
> ändert. Die Fusion und die Beschleunigung zeigen sich in XProf und in den
> Messungen der Schrittdauer des vollständigen `DoubleStreamBlock`, nicht in
> der Ausgabe der Zelle. Traces im Vergleich von Eager und kompiliert finden
> Sie in
> [Meilenstein 3 des FLUX-Optimierungstutorials](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-3).

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

## Meilenstein 4: Eigener Pallas-Kernel für SplashAttention { #milestone-4-custom-pallas-splashattention-kernel }

Bei langen Sequenzlängen (zum Beispiel `S = 2048` oder `S = 4096`) wird die
vollständige gemeinsame Attention speichergebunden. Dieser Meilenstein wird nur
in Snippets gezeigt. Die vollständige, ausführbare Implementierung finden Sie
in der
[Anleitung zum Pallas-Kernel](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md).

### Vorher (Referenz): Standard-SDPA von PyTorch { #before-reference-standard-pytorch-sdpa }

```python
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

Attention ohne Tiling materialisiert eine Score-Matrix der Größe `O(S^2)` im
HBM.

### Nachher (optimiert): Pallas SplashAttention { #after-optimized-pallas-splashattention }

```python
attn = splash_sdpa(q, k, v, is_causal=False)  # Drop-in replacement.
```

`splash_sdpa` ist ein eigener
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html)-Kernel für
SplashAttention, der Attention in Tiles auf dem Chip berechnet. Er basiert auf
der
[TorchTitan-Implementierung von SplashAttention](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py).

Der Kernel wird PyTorch über `torch.tpu.pallas.jax_op` bereitgestellt. Seine
Forward- und Backward-Funktionen sind mit `jax.Array`-Typen annotiert, aus
denen `jax_op` das Schema des eigenen PyTorch-Operators ableitet.
Der Forward-Pass speichert das Residuum `logsumexp` des Softmax, damit der
Backward-Kernel die Forward-Berechnung nicht erneut ausführen muss:

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

> **Important:** **JAX wird zur Laufzeit nicht ausgeführt.** Pallas wird im
> Paket `jax` ausgeliefert, daher wird der Kernel mit `jax`-Syntax geschrieben.
> `torch.tpu.pallas.jax_op` verwendet JAX aber nur einmal, beim Tracing, mit
> abstrakten Platzhaltern für Shape und dtype, um den Kernel als MLIR-Modul zu
> exportieren. Während Training und Inferenz läuft der Kernel sowohl im
> Eager-Modus als auch unter `torch.compile` direkt auf TPU-Puffern vom Typ
> `torch.Tensor`.

**Ergebnisse**: In der Ergebnistabelle ist SplashAttention allein **1,19x**
schneller als die Eager-Baseline (178,61 ms auf 149,81 ms). Mit aktivierten
Zero-Copy-Views und fusioniertem RoPE ist es sowohl im Eager-Modus
(**182,61 ms auf 156,73 ms, 1,17x**) als auch kompiliert (**153,60 ms auf
122,51 ms, 1,25x**) schneller als Standard-SDPA.

Diese Ergebnisse hängen davon ab, dass der Kernel für die Arbeitslast
konfiguriert ist. Mit den standardmäßigen Blöcken von 512 Tokens war
SplashAttention bei gleicher Sequenzlänge unter Kompilierung etwa 10 %
langsamer als Standard-SDPA, und Eingaben in float32 waren noch langsamer.
Führen Sie Benchmarks mit einigen Blockgrößen für Ihre Sequenzlänge aus,
betreiben Sie den Kernel in bfloat16 und erstellen Sie ein Profil, bevor Sie zu
einem eigenen Kernel greifen.

**Weitere Informationen:**

*   **Pallas-Beispiel für SplashAttention (in dieser Fallstudie verwendet)**:
    [`splash_attention.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention.py)
    enthält die vollständige Einbindung des Kernels, und
    [`splash_attention_test.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention_test.py)
    prüft die Übereinstimmung von Forward-Pass und Gradienten mit
    PyTorch-SDPA. Die
    [Anleitung zum Kernel](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
    erklärt Blockgrößen, Masken, die Registrierung für Autograd und das Testen
    Schritt für Schritt.
*   **Pallas-Grundlagen**: [Eigene Kernels mit Pallas](./custom_kernels.md)
    führt in das Schreiben von Pallas-Kernels mit TorchTPU ein.
