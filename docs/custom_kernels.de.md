# Eigene Kernels in TorchTPU { #torchtpu-custom-kernels }

TorchTPU beschleunigt Standardoperatoren des PyTorch-Frameworks nativ (zum
Beispiel
[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)).
Trotzdem kann es nötig sein, eigene Kernels einzubinden, um modellspezifische
Operationen zu beschleunigen, Operationen zu fusionieren und so den Overhead bei
der Speicherbandbreite zu senken, oder neue Architekturen zu implementieren,
die von den Standardoperatoren in PyTorch noch nicht abgedeckt werden.

## Integrationsansatz auswählen { #choosing-an-integration-approach }

TorchTPU bietet je nach Anwendungsfall drei Möglichkeiten, eigene Kernels auf
der TPU zu verwenden:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Ansatz | Status | Am besten geeignet für |
| :---- | :---- | :---- |
| **Tokamax** | Produktionsreif | Direkt einsetzbare Beschleunigung für gängige LLM- und MoE-Bausteine, ohne Kernel-Code zu schreiben. |
| **Pallas (`torch.tpu.pallas.jax_op`)** | Produktionsreif | Maßgeschneiderte TPU-Kernels mit maximaler Performance in JAX/Pallas schreiben und aus PyTorch aufrufen. |
| **Helion** | Experimentell | Kernels mit einer PyTorch-nativen DSL schreiben, die zwischen Beschleunigern (z. B. GPU, TPU) portierbar sind. |
<!-- mdformat on -->

## 1. Vorgefertigte Kernels aus Tokamax verwenden { #1-using-pre-built-kernels-from-tokamax }

Bevor Sie einen eigenen Kernel von Grund auf schreiben, prüfen Sie, ob der
Operator in [Tokamax](http://openxla.org/tokamax) verfügbar ist. Tokamax ist
eine Bibliothek mit hochperformanten TPU-Kernels, die nativ mit PyTorch
zusammenarbeiten.

### Einsatzbereich { #when-to-use }

Verwenden Sie Tokamax, wenn Ihr Modell gängige, hochperformante Bausteine
benötigt, darunter:

*   **Mixture-of-Experts (MoE):** gruppierte Matrixmultiplikation sowie Gather-
    und Scatter-Operationen für Tokens.
*   **Attention-Varianten:** Paged Attention, Gated DeltaNet, Kimi Delta
    Attention und Multi-head Latent Attention (MLA).
*   **Auswahl:** optimierte `TopK`-Berechnung.

### Funktionsweise { #how-it-works }

Tokamax-Kernels sind in JAX/Pallas geschrieben und werden durch integriertes
Autotuning (etwa eine automatische Suche nach der Tile-Größe), XProf-Benchmarks
mit geringem Overhead und Regressionstests über TPU-Generationen hinweg
unterstützt. Intern verwendet Tokamax `torch.tpu.pallas.jax_op`, um die
Interoperabilität zwischen den Frameworks transparent zu handhaben, und stellt
Ihrem Modell direkt native PyTorch-APIs zur Verfügung.

## 2. Eigene Pallas-Kernels schreiben { #2-writing-custom-pallas-kernels }

Wenn Sie eine maßgeschneiderte Operation brauchen, die weder PyTorch noch
Tokamax abdeckt, können Sie einen TPU-Kernel in
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) schreiben und ihn
mit `torch.tpu.pallas.jax_op` als PyTorch-Operator registrieren.

### Pallas-Kernel einbinden { #how-to-integrate-a-pallas-kernel }

1.  **Definieren Sie den Pallas-Kernel** mit Speicherreferenzen (`Ref`s) für
    Ein- und Ausgaben.
2.  **Umschließen Sie den Kernel mit einer JAX-Funktion**, die mit
    `pl.pallas_call` die Ausgabe-Shapes und dtypes angibt.
3.  **Registrieren Sie die JAX-Funktion bei TorchTPU** mit
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

## 3. Portierbare Kernels mit Helion schreiben (experimentell) { #3-writing-portable-kernels-with-helion-experimental }

Wenn Ihre Kernels zwischen Beschleunigern (z. B. TPU, GPU) portierbar sein
sollen, können Sie [Helion](https://helionlang.com/) verwenden, eine
PyTorch-native DSL zum Schreiben von Kernels.

> **Note:** Die Unterstützung für Helion in TorchTPU ist derzeit
> **experimentell**. Verwenden Sie für Produktionsarbeitslasten Tokamax oder
> `torch.tpu.pallas.jax_op`.

### Helion-Kernel schreiben { #how-to-write-a-helion-kernel }

Versehen Sie Ihre Funktion mit dem Decorator `@helion.kernel`. Verwenden Sie
für die Einrichtung auf dem Host Standardoperationen von PyTorch und definieren
Sie mit `hl.tile` die Ausführungsschleife auf dem Gerät über partitionierte
Tiles:

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

Weitere Informationen zu Helion finden Sie in der
[Helion-Dokumentation](https://helionlang.com/). Fragen können Sie im
[Diskussionsforum für PyTorch-Entwickler](https://dev-discuss.pytorch.org/c/helion/31)
stellen, und Probleme können Sie
[auf GitHub melden](https://github.com/pytorch/helion/issues).
