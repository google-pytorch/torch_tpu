# Best practice per FSDP { #fsdp-best-practices }

Fully Sharded Data Parallel (FSDP) suddivide (sharding) i parametri del
modello, i gradienti e lo stato dell'ottimizzatore tra i rank data-parallel,
in modo che ogni core TPU memorizzi solo una porzione `1 / world_size` del
modello. Prima che un layer venga eseguito, i suoi pesi completi vengono
ricostruiti con un'operazione **all-gather**; durante il backward, i gradienti
vengono combinati e ripartiti di nuovo con un'operazione **reduce-scatter**.

Questa guida spiega il modo consigliato di usare FSDP su TorchTPU e rimanda a
esempi eseguibili in
[`examples/fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp).

In questa guida imparerai:

1.  **Quale approccio FSDP scegliere** su TorchTPU.
2.  **SimpleFSDP (consigliato)**: FSDP pensato per la compilazione, con una
    sola chiamata di wrapping.
3.  **FSDP manuale**: inserire le operazioni collettive direttamente nella
    definizione del modello.
4.  **Best practice** valide per tutti gli approcci.
5.  **Esecuzione degli esempi**: i comandi di avvio e cosa verifica ogni
    esempio.

## 1. Scegliere un approccio FSDP { #1-choosing-an-fsdp-approach }

```
On TPU, performance comes from compiling the whole training step into a single
XLA program with `torch.compile(backend="tpu")`. When the XLA compiler sees all
forward, backward, and collective operations at once, it can schedule
collectives to overlap with computation and combine small collectives when
profitable.

Choose an FSDP approach based on how well it composes with whole-graph
compilation:
```

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
    | Approccio | Quando usarlo | Esempio |
    | :------- | :------------- | :------ |
    | **SimpleFSDP** (`simple_fsdp.data_parallel`) | **Scelta predefinita consigliata.** Vuoi usare FSDP o HSDP senza modificare la definizione del modello.| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) |
    | **FSDP manuale** (operazioni collettive manuali) | Gestisci tu il codice del modello e vuoi tenere il codice di comunicazione accanto al calcolo per ottimizzare le prestazioni. | [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) |
    | **PyTorch FSDPv2** (`fully_shard`) | Codice esistente non ancora migrato. Funziona, ma i suoi hook di runtime causano interruzioni del grafo (graph break) e, per ottenere prestazioni competitive, richiede il supporto, in arrivo, per l'offload degli stream sullo SparseCore della TPU.
.| [Guida alla migrazione](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) |
<!-- mdformat on -->

```
SimpleFSDP and manual FSDP both express the collectives as part of the traced
graph, so XLA can optimize them together with the rest of the step. PyTorch
FSDPv2 (`fully_shard`) relies on Python pre-forward and post-backward hooks.
These hooks split the step into many smaller graphs, even when the model is
compiled.

For a side-by-side comparison of compiled FSDPv2 and compiled SimpleFSDP, see
the
[FSDPv2 to SimpleFSDP tutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md).
```

## 2. Consigliato: SimpleFSDP { #2-recommended-simplefsdp }

[`simple_fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/simple_fsdp.py)
è un'implementazione di FSDP progettata per funzionare con `torch.compile`. È
stata introdotta in TorchTitan e in `examples/fsdp/simple_fsdp.py` è
disponibile una copia autonoma che dipende solo da PyTorch.

SimpleFSDP memorizza ogni parametro come `DTensor` suddiviso in shard. Quando
si accede a un parametro, una parametrizzazione chiama
`DTensor.redistribute` per eseguirne l'all-gather. Il backward di autograd di
`redistribute` esegue il reduce-scatter dei gradienti. Non ci sono hook di
runtime, quindi l'intero step viene compilato senza graph break. L'intero
modello viene suddiviso in shard con una sola chiamata
`simple_fsdp.data_parallel(...)`, senza wrapping per singolo layer né
ottimizzazione del prefetch, e HSDP è supportato con `mode="hybrid_shard"` su
una `DeviceMesh` 2D.

Per scoprire come usare SimpleFSDP, consulta:

*   [README di SimpleFSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/README.md):
    guida rapida e opzioni disponibili.
*   [Tutorial da FSDPv2 a SimpleFSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md):
    come migrare il codice `fully_shard` esistente.

## 3. Alternativa: FSDP manuale { #3-alternative-manual-fsdp }

Se controlli la definizione del modello, puoi scrivere FSDP direttamente con
operazioni collettive manuali (`torch.distributed._functional_collectives`) e
`torch.compile`. Ogni layer memorizza solo il proprio shard dei pesi ed esegue
l'all-gather del peso completo nel forward. Le operazioni collettive
funzionali hanno formule di autograd, quindi il backward dell'all-gather è un
reduce-scatter e non devi scrivere una `torch.autograd.Function`
personalizzata.

In questo modo il codice di comunicazione resta accanto al codice di calcolo,
il che può semplificare l'ottimizzazione delle prestazioni. Lo svantaggio è
che devi modificare la definizione del modello.

Per scoprire come implementare FSDP manuale, consulta la
[guida a FSDP manuale](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual/manual_fsdp.md).

## 4. Best practice { #4-best-practices }

Questi consigli valgono per SimpleFSDP e per FSDP manuale.

1.  **Compila l'intero step di addestramento con
    `torch.compile(backend="tpu")`.** Se non puoi compilare l'intero step,
    compila almeno forward, loss e backward insieme. Compilare solo il modulo
    del modello non produce un unico grafo e questo limita la capacità di XLA
    di sovrapporre le operazioni collettive al calcolo.
2.  **Usa l'activation checkpointing su ogni layer suddiviso in shard.**
    Racchiudi il forward di ogni layer in
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. In questo
    modo i pesi non suddivisi vengono liberati dopo il forward pass di ogni
    layer, invece di essere mantenuti fino al backward, e ottieni il
    comportamento di memoria ZeRO-3 che ti aspetti da FSDP.
3.  **Rendi le dimensioni suddivise divisibili per la dimensione della mesh.**
    FSDP manuale lo richiede. SimpleFSDP replica i parametri troppo piccoli
    per essere suddivisi. Usa `shard_dim` per scegliere una dimensione che si
    divida in modo esatto.
4.  **Usa un seed diverso per ogni rank quando inizializzi i pesi suddivisi.**
    Se ogni rank inizializza il proprio shard con lo stesso seed, il peso
    completo è lo stesso blocco casuale ripetuto su tutti i rank, e questo può
    peggiorare la convergenza. Per ottenere shard riproducibili e distinti,
    inizializza ogni rank con un valore che dipende dal rank, ad esempio
    `torch.manual_seed(base_seed + dist.get_rank())`.
5.  **Fai eseguire lo stesso programma a tutti i rank.** Non inserire
    operazioni che forzano la materializzazione (come `.item()`) all'interno di
    rami specifici per rank come `if
    rank == 0`.
6.  **Usa HSDP quando lo sharding completo sull'intera mesh non scala.** Usa
    `mode="hybrid_shard"` con una mesh 2D quando la tua slice ICI è troppo
    grande per suddividere il modello in modo efficace o quando lo sharding su
    di essa genera troppo traffico di rete. Quando esegui su più slice, quasi
    sempre conviene replicare (data-parallel) sulla rete DCN, più lenta, ed
    eseguire lo sharding all'interno di ogni slice.
7.  **Verifica memoria e risultati numerici prima di scalare.** Esegui un
    confronto con un riferimento non suddiviso o FSDPv2 usando `assert_close` e
    misura la memoria mantenuta a cavallo del confine tra forward e backward.
    Gli esempi includono un controllo di parità
    (`fsdp_to_simple_fsdp_worker.py`) e un test di memoria
    (`manual_fsdp_memory_test.py`).

## 5. Esecuzione degli esempi { #5-running-the-examples }

Ogni esempio include uno script di avvio che viene eseguito su 8 core TPU (ad
esempio `v6e-8`):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Esempio | Comando di avvio | Cosa verifica |
:-------- | :------------- | :------------- |
| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) | `./run_fsdp_to_simple_fsdp.sh` | FSDPv2 compilato e SimpleFSDP <br> compilato hanno loss e previsioni <br> corrispondenti, e vengono riportati <br> i tempi dei loro step. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `./launch.sh` oppure `python3 launch.py` | I gradienti e le previsioni di FSDP manuale <br> corrispondono a un riferimento CPU su singolo dispositivo. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `python3 manual_fsdp_memory_test.py` | I pesi non suddivisi vengono liberati dopo il <br> forward pass. |
<!-- mdformat on -->
