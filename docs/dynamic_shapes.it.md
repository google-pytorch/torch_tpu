# Forme dinamiche { #dynamic-shapes }

<!-- Link to colab and to generated notebook -->

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/dynamic_shapes.ipynb)

Quando compili un modello con `torch.compile(model)`, TorchTPU abbassa
(lowering) il calcolo in StableHLO e lo compila con il compilatore XLA
(consulta [Come funziona TorchTPU](./how_torchtpu_works.md)). Il compilatore
XLA è tradizionalmente progettato intorno a **forme dei tensori note
staticamente**, in modo che i buffer della High Bandwidth Memory (HBM) possano
essere disposti in anticipo, le pipeline di operatori possano essere fuse e le
moltiplicazioni di matrici possano essere suddivise in tile per gli array
sistolici della TPU (MXU). Quando le dimensioni dei tensori di input variano
da una chiamata all'altra in un carico di lavoro `torch.compile`, compilare
per ogni forma esatta invaliderebbe i guard sulle forme di Dynamo e
attiverebbe una costosa ricompilazione XLA completa per ogni nuova forma.

Per supportare le **forme dinamiche** senza ricompilazioni continue, TorchTPU
usa il **dinamismo limitato** (Bounded Dynamism): i limiti superiori in fase
di compilazione delle dimensioni dinamiche sono fissi, mentre le dimensioni
effettive a runtime vengono tracciate dinamicamente da XLA.

## Come TorchTPU gestisce le forme dinamiche { #how-dynamic-shapes-are-handled-by-torchtpu }

Quando compili un modello o una funzione con il dinamismo predefinito, ovvero
`torch.compile(model, dynamic=None)` (quando le forme di input variano tra le
chiamate), oppure contrassegni esplicitamente le dimensioni dinamiche con
`torch._dynamo.mark_dynamic`, Dynamo traccia il calcolo e produce un grafo FX
in cui le dimensioni che cambiano sono rappresentate come variabili intere
simboliche (`torch.SymInt`) invece che come costanti fisse. TorchTPU gestisce
queste dimensioni simboliche nel modo seguente:

### Compilazione { #compilation }

*   TorchTPU intercetta il grafo FX tracciato, determina un intervallo
    complessivo `[lower_bound, upper_bound]` per ogni dimensione simbolica (da
    `mark_dynamic(..., min=..., max=...)` oppure dedotto da Dynamo dalla
    dimensione iniziale a runtime `S`) e suddivide l'intervallo in bucket
    contigui allineati a multipli di `bucket_size` (valore predefinito `128`,
    corrispondente alla dimensione dei tile a 128 lane della TPU), fino a
    `num_dynamic_buckets` (valore predefinito `1`, massimo `8`):
    *   Il primo bucket va da `lower_bound` fino al successivo multiplo di
        `128` uguale o successivo a `S` (`U_1`), il secondo bucket va da
        `U_1 + 1` fino al successivo multiplo di `128` (`U_2`), e così via (ad
        esempio `[lower_bound, 128]`, `[129, 256]`, ...).
    *   TorchTPU registra in Dynamo un unico guard sull'intervallo che copre
        l'intera estensione di tutti i bucket, quindi Dynamo attiva una
        ricompilazione solo se la dimensione supera l'`upper_bound`
        complessivo.
*   Per ogni bucket, TorchTPU sostituisce la dimensione simbolica con il
    limite superiore statico `U` del bucket, passa come input la dimensione
    effettiva a runtime e abbassa il grafo in StableHLO con tipi di tensore
    limitati che tengono traccia, in tutte le operazioni, sia del limite
    superiore statico `U` sia della dimensione attiva a runtime.
*   XLA compila il modulo risultante per ogni limite superiore `U` dei bucket
    configurati.

### Esecuzione { #execution }

*   A runtime, in base alla dimensione effettiva della dimensione dinamica,
    TorchTPU seleziona l'eseguibile compilato del bucket corrispondente.

*   Quando esegue il modulo compilato, XLA aggiunge padding al tensore di input
    dinamico fino al limite superiore statico `U` del bucket e reimposta i
    valori nella regione di padding oltre la dimensione attiva a runtime su
    valori di identità algebrica, in modo che gli elementi di padding non
    influiscano sul risultato.

*   Per i tensori di output dinamici, il padding rimane alla fine del buffer,
    così TorchTPU può restituire il buffer a PyTorch con un semplice
    aggiornamento dei metadati della forma, invece di eseguire fisicamente lo
    slicing del tensore.

## Caso d'uso di riferimento: decodifica autoregressiva { #motivating-use-case-autoregressive-decoding }

Un carico di lavoro fondamentale che motiva il dinamismo limitato è la
**decodifica autoregressiva degli LLM** con una KV cache che cresce
dinamicamente. In un loop di decodifica standard, ogni step aggiunge gli stati
key e value del nuovo token alla cronologia memorizzata nella cache lungo la
dimensione della sequenza tramite `torch.cat`:

*   **Prefill:** il prompt di input ha forma `[B, S]` (batch size `B`,
    lunghezza della sequenza del prompt `S`) e produce una KV cache di forma
    `[B, H_kv, S, D]` (`H_kv` head key/value, dimensione della head `D`).
*   **Step di decodifica 1:** il token di input ha forma `[B, 1]` e viene
    concatenato con la cache per produrre la forma `[B, H_kv, S + 1, D]`.
*   **Step di decodifica `t`:** la lunghezza della sequenza della KV cache
    cresce fino a `S + t`.

Con la compilazione a forme statiche (`dynamic=False`), Dynamo imposta un
guard sulla lunghezza esatta della sequenza della KV cache. Ogni step di
decodifica cambia la forma del tensore, invalida il guard e attiva una
**ricompilazione XLA completa per ogni token generato**, trasformando step di
decodifica da millisecondi in compilazioni da diversi secondi, a meno che tu
non riscriva il modello per preallocare buffer statici ed eseguire modifiche
in-place.

Il **dinamismo limitato** risolve questo problema senza richiedere di
riscrivere il modello: mentre la KV cache cresce durante gli step di
decodifica (`S + 1`, `S + 2`, ...), tutte le forme all'interno del bucket
compilato (`<= 128`, `<= 256`, ...) riutilizzano lo stesso eseguibile XLA.

Per una procedura pratica che applica il dinamismo limitato a un loop di
decodifica LLM completo (e lo confronta con la preallocazione statica della KV
cache), consulta
[Inferenza LLM end-to-end](./case_studies_muse_glimmer_30b_inference.md).

## Esempio: loop autoregressivo con un tensore che cresce { #example-autoregressive-loop-with-a-growing-tensor }

Per vedere in pratica come si comportano la compilazione statica
(`dynamic=False`) e il dinamismo limitato con il dinamismo predefinito
(`dynamic=None`), considera un loop autoregressivo minimo in cui ogni step
aggiunge un nuovo token lungo la dimensione della sequenza (`dim=2`) con
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

### 1. Esecuzione con `dynamic=False` { #1-running-with-dynamicfalse }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()
torch._dynamo.config.recompile_limit = 64

step_static = torch.compile(decode_step, backend="tpu", dynamic=False)
run_decode_loop(step_static, prefix="[dynamic=False] ")
```

Output di esempio:

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

**Cosa è successo:** con `dynamic=False`, Dynamo imposta un guard sulla
lunghezza esatta della sequenza di `kv_cache` (`16`, `17`, `18`, ...). Poiché
`torch.cat` aumenta `dim=2` di `1` a ogni iterazione, il guard sulla forma
fallisce a ogni chiamata e attiva una ricompilazione XLA completa a ogni step.

### 2. Esecuzione con `dynamic=None` (dinamismo limitato) { #2-running-with-dynamicnone-bounded-dynamism }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()

step_dynamic = torch.compile(decode_step, backend="tpu", dynamic=None)
run_decode_loop(step_dynamic, prefix="[dynamic=None] ")
```

Output di esempio:

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

**Cosa è successo:** con `dynamic=None`, solo le prime 3 iterazioni hanno un
overhead di compilazione e, dallo step 4 in poi, l'esecuzione scende a circa
0,4 ms per step (circa 100 volte più veloce) senza alcuna ricompilazione:

*   **Step 1 (`seq_len=16 -> 17`):** Dynamo vede gli input per la prima volta
    e compila un eseguibile statico per `seq_len=16`.
*   **Step 2 (`seq_len=17 -> 18`):** poiché `dim=2` è cambiata da `16` a `17`,
    Dynamo contrassegna automaticamente `dim=2` come dinamica (`SymInt`) e
    TorchTPU compila un eseguibile a dinamismo limitato per il bucket
    `[17, 128]` usando il layout del tensore di input iniziale.
*   **Step 3 (`seq_len=18 -> 19`):** il buffer di output a dinamismo limitato
    prodotto dallo step 2 ha il layout di output con padding di XLA, quindi
    TorchTPU compila ancora una volta il bucket `[17,
    128]` per quel layout XLA a regime.
*   **Step 4–10 (`seq_len=19 -> 26`):** sia la lunghezza della sequenza
    (all'interno del bucket `<= 128`) sia il layout del buffer XLA
    corrispondono all'eseguibile nella cache, quindi ogni step successivo
    invia direttamente l'eseguibile XLA precompilato senza ricompilare.

> **Tip:** se sai già in anticipo quale dimensione varierà, puoi chiamare
> `torch._dynamo.mark_dynamic(kv_cache, 2, min=16, max=128)` prima del primo
> step. In questo modo Dynamo traccia `dim=2` come dinamica fin dallo step 1,
> saltando la compilazione statica iniziale dello step 1 e permettendoti di
> specificare limiti `[min, max]` espliciti.

## Limitazioni attuali { #current-limitations }

Il dinamismo limitato in TorchTPU è in continua evoluzione e al momento è
ottimizzato per il caso d'uso della **KV cache autoregressiva**. Tieni
presenti le seguenti limitazioni quando lo applichi ai tuoi modelli:

*   **Limite di bucket e dinamismo illimitato:** vengono compilati al massimo
    **8 bucket** (`num_dynamic_buckets <= 8`) per ogni tracciamento. Il
    dinamismo illimitato non è ancora supportato: se una dimensione dinamica
    supera il limite superiore del bucket compilato più alto, viene attivata
    una ricompilazione.
*   **Vincoli di forma di Dynamo:** i limiti superiori dei bucket devono
    rispettare i vincoli di forma attivi di Dynamo (come i requisiti di
    divisibilità di `reshape` o `view`). In caso contrario, TorchTPU ripiega
    su un unico bucket con limite superiore `2 * S` (il doppio della
    dimensione iniziale a runtime `S`).
*   **Overhead di calcolo e HBM fino al limite superiore del bucket:** poiché
    l'esecuzione avviene su buffer con padding fino al limite superiore
    statico `U` del bucket, l'occupazione della HBM e i FLOP crescono con `U`
    e non con la dimensione effettiva a runtime. Se imposti un `bucket_size`
    molto grande o `mark_dynamic(..., max=U)` quando la dimensione effettiva è
    molto più piccola di `U`, paghi il costo di calcolo e di memoria di `U`.
*   **Modifiche del layout XLA:** anche quando la forma di un tensore rientra
    in un bucket già compilato, una modifica del layout fisico XLA sottostante
    di un buffer di input attiva la ricompilazione del bucket.
*   **Nessuna cache di compilazione persistente (`AOTAutogradCache`):** gli
    eseguibili a dinamismo limitato non vengono ancora serializzati in
    `AOTAutogradCache`, quindi vengono ricompilati a ogni avvio di un nuovo
    processo.
*   **Copertura degli operatori:** non tutti gli operatori PyTorch supportano
    ancora le forme dinamiche limitate.
*   **Viste come input del modello:** il supporto per passare viste di tensori
    direttamente come input al modello compilato è limitato.
*   **Dinamismo dipendente dai dati:** le operazioni le cui forme di output
    dipendono dai valori dei tensori a runtime (come `torch.nonzero`) non sono
    supportate.
*   **Kernel personalizzati:** le forme dinamiche limitate non sono supportate
    con i kernel TPU personalizzati (come i kernel Pallas registrati tramite
    `torch.tpu.pallas.jax_op`).
*   **dynamic=True:** la modalità *dynamic=True* di torch.compile() non è
    supportata.
