# Muse Glimmer 30B: addestramento { #muse-glimmer-30b-training }

In questo caso di studio esaminiamo il nostro esempio autonomo di Muse
Glimmer 30B, che mostra un semplice loop di addestramento per il modello
completo usando FSDP su 4 TPU Ironwood.

Per informazioni su come eseguire questo esempio sulle TPU del tuo cluster GKE
(Google Kubernetes Engine), consulta il
[README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop)
dell'esempio. Qui spieghiamo invece alcune delle scelte di implementazione.

Questo esempio punta su semplicità e leggibilità. Viene eseguito su un solo
host con 4 TPU Ironwood. Chiaramente non è una scala realistica per il
pre-addestramento, ma il nostro obiettivo è offrirti un esempio funzionante
che puoi eseguire, modificare e studiare a basso costo.

Questo codice in PyTorch puro (senza kernel personalizzati) raggiunge una
velocità di addestramento di circa 2400-2800 TPS per rank (quindi da 19k a
22k TPS in totale), con una lunghezza di sequenza di 5120. Nei prossimi
aggiornamenti prevediamo di descrivere altre ottimizzazioni e best practice
per le prestazioni sulle TPU.

## Definizione del modello { #model-definition }
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) segue
un'architettura transformer piuttosto standard.
È composto da 52 layer di blocchi di self-attention causale e feed-forward
(FFN). Un layer su 4 del modello usa l'attention completa, mentre tutti gli
altri layer usano la sliding-window attention (SWA).

Il codice del modello è interamente contenuto in
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)
(e in
[rope.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/rope.py)
per la codifica posizionale rotary).

La configurazione è specificata in
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py):

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

La definizione del backbone del modello si riduce a:

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

Nota che la head del modello (detta anche "LM head") non è presente qui:
nell'addestramento conviene di solito collocarla insieme alla funzione di
loss. Consulta la sezione sulla loss fusa a blocchi più avanti.

## torch.compile per layer { #per-layer-torchcompile }

TorchTPU offre le prestazioni migliori con `torch.compile`, grazie alla
capacità del compilatore XLA di eseguire varie ottimizzazioni, come la fusione
delle operazioni, l'ottimizzazione della memoria e la sovrapposizione
automatica del calcolo con le comunicazioni.

Questo esempio mostra come applicare `torch.compile` a ogni layer del modello
separatamente (invece che all'intero modello). Entrambi gli approcci
funzionano bene, ma la compilazione dell'intero modello può richiedere molto
più tempo e, in termini di prestazioni, almeno per questo modello, non c'è
alcun vantaggio nel compilare oltre l'ambito di un singolo layer.

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

Qui `dynamic=False` significa nessuna forma dinamica (tutte le forme sono
statiche durante il nostro addestramento), e la variabile `backend` serve
solo a semplificare il passaggio tra i backend `"tpu"` e `"inductor"` (per
CUDA).

Internamente TorchTPU memorizza nella cache i layer compilati e, poiché in
questo modello ci sono solo due tipi di layer (completo e SWA), questo
accelera notevolmente il processo di compilazione.

## Activation checkpointing { #activation-checkpointing }

I pesi del modello occupano circa 60 GiB (in bf16) di HBM e possono stare
comodamente su un singolo TensorCore di una TPU Ironwood (che ha 96 GiB).

Tuttavia, durante l'addestramento serve memoria anche per i gradienti e per
gli stati dell'ottimizzatore (ad esempio momentum e varianza di AdamW) e,
soprattutto, l'uso della memoria può essere facilmente dominato dalle
attivazioni intermedie necessarie per il funzionamento di autograd.
L'activation checkpointing (detto anche rematerializzazione) è una tecnica
molto comune per risolvere quest'ultimo problema.

Poiché compiliamo il modello per layer, è molto naturale applicare
l'activation checkpointing con la stessa granularità. Lo facciamo con la
funzionalità standard
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

> **Note:** l'activation checkpointing (AC) viene applicato all'interno del
> metodo `forward` del layer del modello (modulo `DecoderBlock`), mentre
> `torch.compile` viene applicato all'esterno, sul layer stesso: questo
> ordine è importante.
> Potresti mantenere l'AC all'interno del layer e applicare `torch.compile`
> all'intero modello: funzionerebbe bene.
> Ma non puoi invertirli: ad esempio, se applicassi l'AC nello stesso modo,
> ma applicassi `torch.compile` all'interno, separatamente sui sotto-layer di
> attention e FFN, non funzionerebbe come previsto.

`torch.compile` funziona particolarmente bene con l'AC in TorchTPU: durante il
backward pass (globale), ogni layer esegue un piccolo forward+backward pass
congiunto (a causa della rematerializzazione) e, poiché il compilatore del
backend TPU (XLA) vede l'intero grafo di calcolo congiunto forward+backward,
può eseguire ottimizzazioni aggiuntive e persino ridurre l'uso della memoria
per i tensori intermedi.

## FSDP "ingenuo" { #naive-fsdp }

In questo esempio volevamo mostrare il modo più semplice possibile di eseguire
un addestramento FSDP (fully-sharded data-parallel), usando direttamente le
operazioni collettive native di PyTorch.

> **Tip:** per una descrizione ancora più semplice di questo approccio,
> consulta il nostro
> [esempio di FSDP manuale](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual).

Implementiamo FSDP suddividendo manualmente in shard tutti i pesi delle
proiezioni di attention e FFN, ad esempio:

```python
# QKV projections are fully sharded:
self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
```

> **Note:** il layer `nn.Linear` di PyTorch memorizza la matrice dei pesi
> trasposta, ed è per questo che qui eseguiamo lo sharding sulla 2ª
> dimensione. La forma effettiva del peso sarà
> `[num_q_heads * dh / world_size, dm]`. Questo è importante perché le
> operazioni collettive all-gather / reduce-scatter di PyTorch lavorano sulla
> 1ª dimensione.

Poi, durante il forward pass, usiamo le operazioni collettive funzionali di
PyTorch per raccogliere i pesi prima di usarli:

```python
# Functional collectives
import torch.distributed._functional_collectives as func_col

# ... all-gather:
q_proj_w = func_col.all_gather_single(self.q_proj.weight, gather_dim=0, group=self.pg)

# ... use full weight directly:
q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
```

La stessa idea viene applicata a tutte le altre proiezioni (Q/K/V/out/gate
dell'attention e up/gate/down della FFN).

Un aspetto fondamentale è che le operazioni collettive funzionali come
`func_col.all_gather_single` sono registrate nel motore autograd di PyTorch,
quindi durante il pass `.backward()` otteniamo automaticamente le
corrispondenti operazioni collettive reduce-scatter (cioè non dobbiamo
scriverle noi).

Noterai che questa implementazione è relativamente ingenua, perché non prova
a eseguire il prefetch dei pesi del layer successivo mentre esegue il layer
corrente. Questo non causerebbe TensorCore della TPU bloccati o inattivi in
attesa del completamento delle operazioni collettive?

Poiché applichiamo `torch.compile` all'intero layer, il compilatore di
TorchTPU (XLA) è in grado di riordinare le operazioni collettive e di
delegarle allo SparseCore per eseguirle in modo asincrono, sia nel forward
pass sia nei forward+backward pass congiunti. In questo caso ciò basta per
ottenere una sovrapposizione quasi perfetta tra comunicazione e calcolo.

## LM head fusa a blocchi + loss di cross-entropy { #chunkwise-fused-lm-head-cross-entropy-loss }

Per impostazione predefinita l'esempio usa una batch size locale di 8 e una
lunghezza di sequenza di 5120. Se calcolassimo in modo ingenuo la loss di
cross-entropy per l'addestramento, dovremmo materializzare un tensore
intermedio di logit di forma `[8, 5120, 202048]`, dove l'ultimo numero è la
dimensione del vocabolario del modello. In bf16 questo da solo richiederebbe
già circa 16 GiB di memoria HBM, uno spreco eccessivo.

Una soluzione comune a questo problema è calcolare la loss a blocchi (chunk),
fondendo la matmul con la matrice dei pesi del vocabolario con la loss di
cross-entropy. Un approccio consiste nel suddividere in blocchi lungo la
dimensione del vocabolario (che richiede la softmax online ed è difficile da
esprimere in PyTorch puro), un altro consiste nel suddividere in blocchi
lungo la dimensione della sequenza. Qui usiamo il secondo:

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

`fused_loss_fn_fwd_bwd` è la versione compilata con `torch.compile` della
funzione statica che calcola, su un singolo blocco di token, la matmul del
vocabolario fusa + la loss di cross-entropy + il backward pass:

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

Per ottenere il massimo vantaggio da questo approccio, eseguiamo sul blocco
anche il backward pass. Nelle versioni più recenti di PyTorch, invocare
`loss.backward()` non causa più un graph break, il che permette al
compilatore di TorchTPU (XLA) di eseguire più ottimizzazioni e fusioni.

## Cosa manca in questo esempio { #whats-missing-in-this-example }

Questo esempio illustra solo il loop di addestramento di base e, per
chiarezza, omette vari dettagli importanti per qualsiasi configurazione di
addestramento reale:

 - caricamento dei dati, tokenizzazione,
 - caricamento e salvataggio dei checkpoint del modello,
 - monitoraggio dello stato dei job, diagnostica della convergenza,
 - tolleranza agli errori.

Per una configurazione di addestramento realistica e pronta per la produzione
con TorchTPU, ti consigliamo di consultare
[TorchTitan](https://github.com/google-pytorch/torchtitan/).
