# Ottimizzazione di FLUX { #flux-optimization }

<!-- Link to colab and to generated notebook -->

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_flux.ipynb)

Questo caso di studio mostra come abbiamo preso un'implementazione PyTorch di
riferimento del `DoubleStreamBlock` del diffusion transformer
[FLUX](https://github.com/black-forest-labs/flux) e abbiamo reso uno step di
addestramento **1,48 volte più veloce** su Cloud TPU con TorchTPU, con una
lunghezza di sequenza di 32.768.

Questa pagina è una versione condensata del tutorial completo. Mette in
evidenza il concetto chiave di ogni ottimizzazione con snippet brevi e
autonomi. Per il codice completo del modello, i benchmark, le tracce XProf e i
test, consulta:

*   [Tutorial sull'ottimizzazione di FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md):
    guida completa a ogni milestone, con le tracce XProf e le istruzioni per
    riprodurre ogni benchmark.
*   [Scrivere kernel Pallas personalizzati per PyTorch](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md):
    guida passo passo al kernel Pallas SplashAttention usato nella Milestone 4.

L'implementazione si basa
sull'[implementazione di FLUX in TorchTitan](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)
e sul repository
[black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/).

Se non hai mai usato gli esempi, puoi prima prendere familiarità con
l'esecuzione degli esempi su GKE
[tramite gli altri nostri documenti](./get_started_on_gke.md).

## Risultati in sintesi { #results-at-a-glance }

Tempo medio di uno step di addestramento (forward + loss + backward + step
dell'ottimizzatore) per un singolo `DoubleStreamBlock`:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Configurazione | Tempo medio per step | Speedup rispetto alla baseline |
| :--- | ---: | ---: |
| Riferimento baseline (eager) | 178,61 ms | 1,00x |
| Solo Pallas SplashAttention (eager) | 149,81 ms | 1,19x |
| Solo viste zero-copy (eager) | 178,30 ms | 1,00x |
| Solo `torch.compile` | 159,33 ms | 1,12x |
| Viste zero-copy + RoPE fusa + Pallas SplashAttention (eager) | 156,73 ms | 1,14x |
| Viste zero-copy + `torch.compile` | 159,08 ms | 1,12x |
| Viste zero-copy + RoPE fusa + `torch.compile` | 153,60 ms | 1,16x |
| **Viste zero-copy + RoPE fusa + Pallas SplashAttention + `torch.compile`** | **122,51 ms** | **1,46x** |
<!-- mdformat on -->

**Configurazione della misurazione:**

-   **Hardware:** un singolo chip TPU v6e (32 GiB di HBM).
-   **Software:** Python 3.14, PyTorch 2.15 (pre-release), JAX 0.12, einops
    0.8.2 e TorchTPU compilato dal codice sorgente.
-   **Forma del modello:** `batch_size=2`, `hidden_size=1024`, `num_heads=8`
    (`head_dim=128`), `img_seq_len=32256`, `txt_seq_len=512` (lunghezza totale
    della sequenza 32.768).
-   **Precisione:** parametri e input del blocco in bfloat16.
-   **Dimensioni dei blocchi di SplashAttention:** 2048 per `block_q`,
    `block_kv` e per i tile del backward dK/dV.
-   **Misurazione dei tempi:** 3 step di warmup seguiti da 10 step
    cronometrati. Gli step cronometrati vengono inviati uno dopo l'altro e
    `torch.accelerator.synchronize()` viene eseguito una sola volta dopo
    l'ultimo, quindi il tempo medio per step (tempo totale diviso per il numero
    di step) include l'esecuzione sul dispositivo. Il profiler è disattivato
    durante gli step cronometrati.

Con questa lunghezza di sequenza, l'attention domina lo step, quindi
SplashAttention e la compilazione danno i guadagni maggiori. Con lunghezze di
sequenza più brevi, la compilazione conta di più (circa 3 volte con una
lunghezza di sequenza di 640) e il kernel di attention conta di meno.

Queste lezioni valgono per la maggior parte dei modelli transformer su TPU:

1.  **Evita le copie di layout nascoste**: preferisci le viste zero-copy
    (`.view()`, `torch.unbind()`) ai riordinamenti che materializzano nuovi
    buffer in HBM.
2.  **Memorizza meno, calcola di più**: esprimi gli embedding posizionali come
    calcoli elemento per elemento compatti invece che come matrici
    materializzate.
3.  **Compila l'intero blocco**: `torch.compile()` permette a XLA di fondere le
    catene di operazioni elemento per elemento ed eliminare i round trip
    intermedi verso la HBM.
4.  **Usa kernel personalizzati quando servono**: i kernel Pallas danno un
    controllo esplicito sul tiling on-chip per le operazioni limitate dalla
    memoria, come l'attention con sequenze lunghe.

### Come leggere questa pagina { #how-to-read-this-page }

Ogni milestone è presentata come una coppia di sezioni:

-   **Prima (riferimento)**: l'implementazione originale e il suo collo di
    bottiglia.
-   **Dopo (ottimizzata)**: l'implementazione ottimizzata, seguita da una
    verifica che produca lo stesso risultato del riferimento.

Le celle di codice sono contrassegnate da un commento corrispondente,
`# BEFORE (reference)` o `# AFTER
(optimized)`, sulla prima riga.

**Prerequisiti:** la cella di riferimento della Milestone 1 usa
[einops](https://github.com/arogozhnikov/einops) (`pip install einops`).

```python
import torch

device = torch.device("tpu")
print(f"Running on {device}.")
```

## Milestone 1: viste dei tensori zero-copy { #milestone-1-zero-copy-tensor-views }

In `DoubleStreamBlock`, gli stream di immagine e di testo proiettano stati
nascosti di forma `(B, L, 3 * H * D)` in query, key e value. Il modo in cui
questa proiezione viene suddivisa in Q, K e V determina se la TPU deve copiare
dati nella HBM durante l'addestramento.

### Prima (riferimento): `einops.rearrange` { #before-reference-einopsrearrange }

L'implementazione di riferimento sposta l'asse `3` (Q/K/V) in prima posizione:

```python
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)
```

Nel forward pass, `einops.rearrange` restituisce viste strided della
proiezione, quindi i dati non vengono ancora spostati. Il costo si manifesta
nel **backward pass**: per riportare i gradienti attraverso il riordinamento
degli assi, autograd materializza il gradiente in un nuovo buffer, che su TPU
si traduce in una copia aggiuntiva da HBM a HBM in ogni blocco di attention
(visibile come operazione `copy` in XProf).

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

### Dopo (ottimizzata): `.view()` + `torch.unbind()` { #after-optimized-view-torchunbind }

L'implementazione ottimizzata mantiene l'asse `3` adiacente a `(H, D)` e lo
divide con `torch.unbind`. Il backward pass impila poi i gradienti di Q, K e
V direttamente nel layout originale, quindi non serve alcuna copia
aggiuntiva. Nella tabella dei risultati, l'effetto sul tempo per step è piccolo
(**da 178,61 ms a 178,30 ms** in modalità eager, **da 159,33 ms a 159,08 ms** con
la compilazione) perché la copia è una piccola frazione dello step, ma viene
eliminata senza alcun costo.

> **Note:** le celle di questa milestone mostrano solo che i due approcci
> producono gli stessi Q, K e V. Nel forward pass, entrambi restituiscono le
> stesse viste strided, quindi i loro output sembrano identici. La copia
> aggiuntiva in HBM di `einops.rearrange` avviene solo nel backward pass ed è
> visibile in una traccia XProf di uno step di addestramento completo, non
> nell'output della cella. Per le tracce, consulta la
> [Milestone 1 del tutorial sull'ottimizzazione di FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-1).

```python
# AFTER (optimized): keep Q/K/V adjacent to (H, D) and unbind.
_b, _l, _ = qkv.shape
q_after, _k, _v = torch.unbind(
    qkv.view(_b, _l, 3, num_heads, head_dim), dim=2
)
print(f"q shape: {tuple(q_after.shape)}")
print(f"Matches reference: {torch.equal(q_after.cpu(), q_before.cpu())}")
```

## Milestone 2: RoPE algebrica fusa { #milestone-2-fused-algebraic-rope }

FLUX applica i Rotary Position Embedding (RoPE) multiasse a query e key. Le
celle seguenti implementano entrambe le versioni per un singolo asse RoPE.

### Prima (riferimento): matrici di rotazione 2x2 { #before-reference-2x2-rotation-matrices }

L'implementazione di riferimento materializza una matrice di rotazione `2x2`
completa `[[cos,
-sin], [sin, cos]]` per ogni coppia di frequenze e la applica come prodotto
matrice-vettore su tensori query/key 6D rimodellati. Questo memorizza `cos` e
`sin` due volte, e le operazioni aggiuntive di stacking e reshaping appaiono in
XProf come grandi blocchi di copia.

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

### Dopo (ottimizzata): coppie `[cos, sin]` compatte + rotazione elemento per elemento { #after-optimized-compact-cos-sin-pairs-elementwise-rotation }

L'implementazione ottimizzata memorizza solo coppie `[cos, sin]` compatte e
applica la rotazione direttamente con calcoli elemento per elemento:

```python
# (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
xq_out_0 = xq_0 * cos - xq_1 * sin
xq_out_1 = xq_0 * sin + xq_1 * cos
```

Questo dimezza l'occupazione di memoria degli embedding posizionali, si mappa
su istruzioni fused multiply-add e, con la compilazione, si fonde con le
operazioni di proiezione e di attention circostanti. Applicata sopra le viste
zero-copy e `torch.compile`, riduce il tempo per step **da 159,08 ms a
153,60 ms (3,4%)**.

Poiché entrambe le versioni eseguono le stesse moltiplicazioni e addizioni sugli
stessi operandi, producono risultati identici. La cella seguente esegue la
versione ottimizzata sugli stessi input e la confronta con il riferimento:

> **Note:** queste celle mostrano il tensore delle frequenze più piccolo e che
> entrambe le versioni producono risultati identici. Lo speedup appare solo
> quando la RoPE viene eseguita all'interno del `DoubleStreamBlock` compilato,
> dove la rotazione elemento per elemento si fonde con le operazioni
> circostanti. Per le tracce XProf, consulta la
> [Milestone 2 del tutorial sull'ottimizzazione di FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-2).

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

## Milestone 3: compilazione XLA dell'intero blocco { #milestone-3-whole-block-xla-compilation }

### Prima (riferimento): esecuzione eager { #before-reference-eager-execution }

In modalità eager, ogni operazione PyTorch avvia un programma TPU indipendente,
quindi le attivazioni intermedie fanno avanti e indietro dalla HBM e l'host si
blocca spesso sul dispatch. Nella tabella dei risultati, il
`DoubleStreamBlock` di baseline richiede **178,61 ms** per step in modalità eager.

La cella seguente esegue in modalità eager la RoPE fusa della Milestone 2,
un'operazione alla volta:

```python
# BEFORE (reference): eager execution, one TPU program per op.
q_eager, k_eager = apply_rope(rope_xq, rope_xk, freqs)
print("Ran fused RoPE eagerly.")
```

### Dopo (ottimizzata): `torch.compile()` { #after-optimized-torchcompile }

Compilare l'intero `DoubleStreamBlock` con `torch.compile()` lo abbassa
(lowering) a un unico grafo XLA, permettendo al compilatore di fondere le
operazioni elemento per elemento, le normalizzazioni e le addizioni residuali:

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

La compilazione porta il tempo per step **da 178,61 ms a 159,33 ms (1,12x)**.
Il guadagno è modesto con una lunghezza di sequenza di 32.768 perché
l'attention domina lo step. Con una lunghezza di sequenza di 640, la
compilazione è circa 3 volte più veloce.

Lo stesso schema si applica a qualsiasi funzione. La cella seguente compila la
RoPE fusa e verifica che corrisponda all'esecuzione eager:

> **Note:** queste celle verificano solo che la compilazione non cambi il
> risultato. La fusione e lo speedup si vedono in XProf e nelle misurazioni del
> tempo per step dell'intero `DoubleStreamBlock`, non nell'output della cella.
> Per le tracce eager e compilate a confronto, consulta la
> [Milestone 3 del tutorial sull'ottimizzazione di FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-3).

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

## Milestone 4: kernel Pallas SplashAttention personalizzato { #milestone-4-custom-pallas-splashattention-kernel }

Con sequenze lunghe (ad esempio `S = 2048` o `S = 4096`), la joint attention
completa diventa limitata dalla memoria. Questa milestone è mostrata solo
tramite snippet; per l'implementazione completa ed eseguibile, consulta la
[guida al kernel Pallas](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md).

### Prima (riferimento): SDPA standard di PyTorch { #before-reference-standard-pytorch-sdpa }

```python
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

L'attention senza tiling materializza in HBM una matrice dei punteggi
`O(S^2)`.

### Dopo (ottimizzata): Pallas SplashAttention { #after-optimized-pallas-splashattention }

```python
attn = splash_sdpa(q, k, v, is_causal=False)  # Drop-in replacement.
```

`splash_sdpa` è un kernel SplashAttention
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) personalizzato che
calcola l'attention in tile on-chip, basato
sull'[implementazione di SplashAttention in TorchTitan](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py).

Il kernel viene esposto a PyTorch con `torch.tpu.pallas.jax_op`. Le sue funzioni
di forward e backward sono annotate con tipi `jax.Array`, da cui `jax_op`
ricava lo schema dell'operatore personalizzato PyTorch. Il forward pass salva il
residuo `logsumexp` della softmax, così il kernel di backward non deve
rieseguire il calcolo del forward:

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

> **Important:** **JAX non viene eseguito a runtime.** Pallas è distribuito
> all'interno del pacchetto `jax`, quindi il kernel è scritto con la sintassi di
> `jax`. Tuttavia, `torch.tpu.pallas.jax_op` usa JAX una sola volta, al momento
> del tracciamento, con segnaposto astratti di forma/dtype per esportare il kernel
> come modulo MLIR. Durante l'addestramento e l'inferenza, sia in modalità eager
> sia con `torch.compile`, il kernel viene eseguito direttamente sui buffer TPU di
> `torch.Tensor`.

**Risultati**: nella tabella dei risultati, la sola SplashAttention è **1,19
volte** più veloce della baseline eager (da 178,61 ms a 149,81 ms). Con le
viste zero-copy e la RoPE fusa abilitate, è più veloce della SDPA standard sia
in modalità eager (**da 182,61 ms a 156,73 ms, 1,17x**) sia in modalità
compilata (**da 153,60 ms a 122,51 ms, 1,25x**).

Questi risultati dipendono dalla configurazione del kernel per il carico di lavoro. Con
i blocchi predefiniti da 512 token, SplashAttention era circa il 10% più lenta
della SDPA standard con la compilazione alla stessa lunghezza di sequenza, e
con input float32 era ancora più lenta. Prova alcune dimensioni dei blocchi per
la tua lunghezza di sequenza, esegui il kernel in bfloat16 e profila prima di
ricorrere a un kernel personalizzato.

**Per saperne di più:**

*   **Esempio di Pallas SplashAttention (usato in questo caso di studio)**:
    [`splash_attention.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention.py)
    contiene l'integrazione completa del kernel e
    [`splash_attention_test.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention_test.py)
    verifica la parità del forward e dei gradienti rispetto alla SDPA di
    PyTorch. La
    [guida al kernel](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
    spiega passo passo le dimensioni dei blocchi, le maschere, la registrazione
    di autograd e i test.
*   **Nozioni di base su Pallas**:
    [Kernel personalizzati con Pallas](./custom_kernels.md) introduce la
    scrittura di kernel Pallas con TorchTPU.
