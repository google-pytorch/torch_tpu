# Muse Glimmer 30B: Training { #muse-glimmer-30b-training }

In dieser Fallstudie gehen wir unser eigenständiges Muse-Glimmer-30B-Beispiel
durch. Es zeigt eine einfache Trainingsschleife für das vollständige Modell mit
FSDP auf 4 Ironwood-TPUs.

Wie Sie dieses Beispiel auf TPUs in Ihrem GKE-Cluster (Google Kubernetes Engine)
ausführen, erfahren Sie in der
[README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop)
des Beispiels. Hier erläutern wir stattdessen einige Entscheidungen bei der
Implementierung.

Der Schwerpunkt dieses Beispiels liegt auf Einfachheit und Lesbarkeit. Es läuft
auf nur einem Host mit 4 Ironwood-TPUs. Das ist natürlich keine realistische
Größenordnung für Pre-Training. Unser Ziel ist aber ein funktionierendes
Beispiel, das Sie mit geringen Kosten ausführen, weiterentwickeln und studieren
können.

Dieser reine PyTorch-Code (ohne eigene Kernels) erreicht bei einer
Sequenzlänge von 5120 eine Trainingsgeschwindigkeit von etwa 2400 bis 2800 TPS
pro Rank (insgesamt also 19.000 bis 22.000 TPS). In künftigen Updates wollen
wir weitere Optimierungen und Best Practices für die Performance auf TPUs
beschreiben.

## Modelldefinition { #model-definition }
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) folgt
einer recht üblichen Transformer-Architektur.
Es besteht aus 52 Schichten mit Blöcken aus kausaler Self-Attention und
Feed-Forward (FFN). Jede vierte Schicht des Modells verwendet Full Attention,
alle anderen Schichten verwenden Sliding Window Attention (SWA).

Der Modellcode ist vollständig in sich geschlossen und befindet sich in
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)
(sowie
[rope.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/rope.py)
für das Rotary Positional Encoding).

Die Konfiguration wird in
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)
festgelegt:

```python
class Config:
  # General model config
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  # Details
  swa_size: int = 2048
  # ... omitted ...

  # Perf evaluation config
  batch_size: int = 8  # Local (per rank) batch size
  seq_len: int = 5120
```

Die Definition des Modell-Backbones lässt sich auf Folgendes reduzieren:

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
self.embed_norm = nn.RMSNorm(
    (config.dmodel,), eps=config.rmsnorm_eps, elementwise_affine=False
)
self.layers = nn.ModuleList(
    DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
)
self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)
```

Beachten Sie, dass der Modell-Head (auch „LM-Head“ genannt) hier fehlt. Beim
Training ist er meist besser bei der Loss-Funktion aufgehoben. Weitere
Informationen finden Sie unten im Abschnitt zum chunkweise fusionierten Loss.

## torch.compile pro Schicht { #per-layer-torchcompile }

TorchTPU erzielt die beste Performance mit `torch.compile`, weil der
XLA-Compiler verschiedene Optimierungen durchführen kann, etwa Op-Fusion,
Speicheroptimierung und das automatische Überlappen von Berechnung und
Kommunikation.

Dieses Beispiel zeigt, wie Sie `torch.compile` auf jede Modellschicht einzeln
anwenden können (statt auf das gesamte Modell). Beide Ansätze funktionieren gut,
aber die Kompilierung des gesamten Modells kann deutlich länger dauern. Bei der
Performance bringt es zumindest für dieses Modell keinen Vorteil, über eine
einzelne Schicht hinaus zu kompilieren.

```python
class MuseGlimmer30B(nn.Module):

  def __init__(self, config: Config):
    # ... omitted ...
    self.layers = nn.ModuleList(
        DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
    )

  def apply_torch_compile(self, backend):
    self.embed_norm = torch.compile(self.embed_norm, dynamic=False, backend=backend)
    self.layers = nn.ModuleList(
        torch.compile(layer, dynamic=False, backend=backend)
        for layer in self.layers
    )
    self.norm = torch.compile(self.norm, dynamic=False, backend=backend)
```

`dynamic=False` bedeutet hier, dass keine dynamischen Shapes verwendet werden
(alle Shapes sind während unseres Trainings statisch). Die Variable `backend`
dient nur dazu, einfacher zwischen den Backends `"tpu"` und `"inductor"` (für
CUDA) wechseln zu können.

Intern speichert TorchTPU kompilierte Schichten im Cache. Da es in diesem
Modell nur zwei Arten von Schichten gibt (Full und SWA), beschleunigt das die
Kompilierung erheblich.

## Activation Checkpointing { #activation-checkpointing }

Die Modellgewichte belegen etwa 60 GiB (in bf16) HBM und würden problemlos auf
einen einzelnen TensorCore einer Ironwood-TPU passen (der 96 GiB hat).

Beim Training brauchen wir aber zusätzlich Speicher für Gradienten und
Optimizer-Zustände (zum Beispiel Momentum und Varianz von AdamW). Noch
wichtiger ist, dass die Zwischenaktivierungen, die für Autograd benötigt
werden, den Speicherverbrauch leicht dominieren können. Activation
Checkpointing (auch Rematerialisierung genannt) ist eine sehr verbreitete
Technik für dieses Problem.

Da wir das Modell pro Schicht kompilieren, liegt es nahe, Activation
Checkpointing mit derselben Granularität anzuwenden. Dafür nutzen wir die
Standardfunktionalität
[`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint):

```python
class DecoderBlock(nn.Module):
  # ... omitted ...

  def forward(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    return torch.utils.checkpoint.checkpoint(
        self.forward_impl,
        h,
        rope_sin_cos,
        use_reentrant=False,
    )

  def forward_impl(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(self.input_layernorm(h), rope_sin_cos)
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h
```

> **Note:** Activation Checkpointing (AC) wird innerhalb der Methode `forward`
> der Modellschicht (Modul `DecoderBlock`) angewendet, `torch.compile` dagegen
> außerhalb, also auf die Schicht selbst. Diese Reihenfolge ist wichtig.
> Sie könnten AC innerhalb der Schicht belassen und `torch.compile` auf das
> gesamte Modell anwenden. Das würde gut funktionieren.
> Sie können die beiden aber nicht vertauschen. Wenn Sie zum Beispiel AC auf
> dieselbe Weise anwenden, `torch.compile` aber innerhalb separat auf die
> Attention- und FFN-Unterschichten, funktioniert es nicht wie erwartet.

`torch.compile` funktioniert in TorchTPU besonders gut mit AC: Während des
(globalen) Backward-Passes führt jede Schicht (aufgrund der Rematerialisierung)
einen kleinen gemeinsamen Forward- und Backward-Pass aus. Da der
TPU-Backend-Compiler (XLA) den gesamten gemeinsamen Berechnungsgraphen aus
Forward- und Backward-Pass sieht, kann er zusätzliche Optimierungen durchführen
und sogar den Speicherverbrauch für Zwischen-Tensoren senken.

## „Naives“ FSDP { #naive-fsdp }

Mit diesem Beispiel wollten wir die einfachste mögliche Art zeigen, FSDP-Training
(Fully Sharded Data Parallel) durchzuführen, und zwar direkt mit nativen
kollektiven Operationen von PyTorch.

> **Tip:** Eine noch einfachere Beschreibung dieses Ansatzes finden Sie in
> unserem
> [Beispiel für manuelles FSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual).

Wir setzen FSDP um, indem wir alle Projektionsgewichte von Attention und FFN
manuell sharden, zum Beispiel:

```python
# QKV projections are fully sharded:
self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
```

> **Note:** Die PyTorch-Schicht `nn.Linear` speichert ihre Gewichtsmatrix
> transponiert. Deshalb sharden wir hier entlang der 2. Dimension. Die
> tatsächliche Shape des Gewichts ist
> `[num_q_heads * dh / world_size, dm]`. Das ist wichtig, weil die
> Collectives All-Gather und Reduce-Scatter von PyTorch auf der 1. Dimension
> arbeiten.

Im Forward-Pass sammeln wir die Gewichte dann mit den funktionalen Collectives
von PyTorch ein, bevor wir sie verwenden:

```python
# Functional collectives
import torch.distributed._functional_collectives as func_col

# ... all-gather:
q_proj_w = func_col.all_gather_single(self.q_proj.weight, gather_dim=0, group=self.pg)

# ... use full weight directly:
q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
```

Dasselbe Prinzip wenden wir auf alle anderen Projektionen an (Attention
Q/K/V/Out/Gate und FFN Up/Gate/Down).

Entscheidend ist, dass funktionale Collectives wie `func_col.all_gather_single`
bei der Autograd-Engine von PyTorch registriert sind. Im `.backward()`-Pass
erhalten wir deshalb automatisch die entsprechenden Reduce-Scatter-Collectives
(wir müssen sie also nicht selbst schreiben).

Sie werden bemerken, dass diese Implementierung relativ naiv ist: Sie versucht
nicht, die Gewichte der nächsten Schicht vorab zu laden, während die aktuelle
Schicht läuft. Würde das nicht dazu führen, dass TPU-TensorCores blockiert sind
und untätig auf den Abschluss der Collectives warten?

Da wir `torch.compile` auf die gesamte Schicht anwenden, kann der Compiler von
TorchTPU (XLA) Collectives umordnen und auf den SparseCore auslagern, damit sie
asynchron laufen, und zwar sowohl im Forward-Pass als auch in den gemeinsamen
Forward- und Backward-Passes. In diesem Fall reicht das aus, um Kommunikation
und Berechnung nahezu perfekt zu überlappen.

## Chunkweise fusionierter LM-Head mit Cross-Entropy-Loss { #chunkwise-fused-lm-head-cross-entropy-loss }

Das Beispiel verwendet standardmäßig eine lokale Batchgröße von 8 und eine
Sequenzlänge von 5120. Würden wir den Cross-Entropy-Trainings-Loss naiv
berechnen, müssten wir einen Zwischen-Tensor mit Logits der Shape
`[8, 5120, 202048]` materialisieren. Die letzte Zahl ist die Vokabulargröße des
Modells. In bf16 bräuchte allein das schon etwa 16 GiB HBM-Speicher, was zu
verschwenderisch ist.

Eine gängige Lösung für dieses Problem ist, den Loss chunkweise zu berechnen
und dabei die Matmul mit der Vokabular-Gewichtsmatrix mit dem
Cross-Entropy-Loss zu fusionieren. Ein Ansatz ist das Chunking entlang der
Vokabulardimension (dafür ist Online-Softmax nötig, was sich in reinem PyTorch
schwer ausdrücken lässt). Ein anderer Ansatz ist das Chunking entlang der
Sequenzdimension. Wir verwenden hier Letzteres:

```python
class MuseGlimmerChunkwiseFusedLMHeadLoss(nn.Module):
  # ... omitted ...

  for i in range(0, h.shape[0], chunk_size):
    h_chunk.data.copy_(h[i : (i + chunk_size)])
    tgt_chunk.data.copy_(tgt[i : (i + chunk_size)])

    self.fused_loss_fn_fwd_bwd(
        h_chunk, tgt_chunk, self.lm_head.weight, alpha, cap, loss_scaler
    )

    h_chunk_grads.append(h_chunk.grad)
    h_chunk.grad = None

  return torch.cat(h_chunk_grads, dim=0).view(bsz, seq_len, dm)
```

`fused_loss_fn_fwd_bwd` bezeichnet die mit `torch.compile` kompilierte Version
der statischen Funktion, die für einen einzelnen Chunk der Tokens die
fusionierte Vokabular-Matmul, den Cross-Entropy-Loss und den Backward-Pass
berechnet:

```python
@staticmethod
def _chunk_fused_lmhead_loss_fwd_bwd(
    h, tgt, lm_head_w, alpha, cap, loss_scaler
):
  logits = F.tanh(F.linear(h, lm_head_w) * (alpha / cap)) * cap
  loss = F.cross_entropy(input=logits, target=tgt, reduction="sum")
  loss *= loss_scaler
  # Note: in recent versions of PyTorch, .backward is no longer a graph break!
  loss.backward()
  return
```

Um den größten Nutzen aus diesem Ansatz zu ziehen, führen wir sogar den
Backward-Pass auf dem Chunk aus. In neueren PyTorch-Versionen verursacht der
Aufruf von `loss.backward()` keinen Graph-Break mehr. Dadurch kann der Compiler
von TorchTPU (XLA) mehr Optimierungen und Fusionen durchführen.

## Was in diesem Beispiel fehlt { #whats-missing-in-this-example }

Dieses Beispiel zeigt nur die grundlegende Trainingsschleife. Der Übersicht
halber lässt es verschiedene Details weg, die für jedes praktische
Trainings-Setup wichtig sind:

 - Laden der Daten, Tokenisierung,
 - Laden und Speichern von Modell-Checkpoints,
 - Überwachung des Job-Zustands, Konvergenzdiagnose,
 - Fehlertoleranz.

Für ein realistisches, produktionsreifes Trainings-Setup mit TorchTPU empfehlen
wir einen Blick auf [TorchTitan](https://github.com/google-pytorch/torchtitan/).
