# Riferimento API di TorchTPU { #torchtpu-api-reference }

Questo documento è la guida di riferimento per le API Python pubbliche e le
variabili di ambiente di TorchTPU. Ti aiuta a:

*   **Migrare da CUDA a TPU**: mappare le chiamate `torch.cuda` più comuni alle
    API
    [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
    di PyTorch, indipendenti dal dispositivo.
*   **Usare le funzionalità specifiche delle TPU**: configurare modalità di
    esecuzione, precisione numerica della MXU, topologia distribuita, operatori
    Pallas/TPU personalizzati, ottimizzatori tracciabili e profilazione con
    XProf tramite il modulo **`torch.tpu`**.
*   **Configurare il runtime**: regolare la cache di compilazione, i controlli
    di debug e le impostazioni di avvio multi-host tramite variabili di
    ambiente.

> **Note:** per l'elenco completo e generato automaticamente, che fa da fonte
> di riferimento, di tutte le API Python pubbliche e le variabili di ambiente,
> consulta
> [API pubbliche di TorchTPU (`public_apis.md`)](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/public_apis.md).

## **Query di base sui dispositivi** { #core-device-queries }

Quando importi `torch` con TorchTPU installato, il backend registra
automaticamente il dispositivo `"tpu"` e il modulo `torch.tpu`. Quando PyTorch
offre un'API
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
indipendente dal dispositivo, usala al posto di un'API specifica del backend,
così il codice resta portabile tra acceleratori diversi. Usa `torch.tpu` per le
proprietà hardware specifiche delle TPU che non hanno un equivalente generico.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API TPU / Accelerator | Descrizione |
| :--- | :--- | :--- |
| `torch.device("cuda")` | `torch.accelerator.current_accelerator()` | Restituisce `torch.device("tpu")` quando la TPU è l'acceleratore attivo. |
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Verifica se il backend TPU è disponibile. |
| `torch.cuda.is_initialized()` | `torch.tpu.is_initialized()` | Verifica se lo stato del runtime di TorchTPU è stato inizializzato. |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | Numero di dispositivi visibili al processo corrente. |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | Restituisce l'indice del dispositivo TPU attivo. |
| `torch.cuda.set_device(rank)` | `torch.accelerator.set_device_index(rank)` | Imposta l'indice del dispositivo TPU attivo per il processo corrente. |
| `torch.cuda.device(rank)` | `torch.accelerator.device_index(rank)` | Context manager che imposta temporaneamente l'indice del dispositivo attivo. |
| `torch.cuda.get_device_name(device=None)` | `torch.tpu.get_device_properties(device=None).name` | Restituisce il nome leggibile del modello della TPU collegata (ad esempio `"TPU v6e"`). |
| `torch.cuda.get_device_properties(device=None)` | `torch.tpu.get_device_properties(device=None)` | Restituisce le proprietà statiche del dispositivo TPU (tra cui `name`, `devices_per_chip`, `hbm_bytes_per_chip`, `hbm_bytes_per_device`, `bf16_flops_per_chip`, `bf16_flops_per_device`, `sparse_cores_per_device` e `sparse_core_simd_width`). |
<!-- mdformat on -->

Le API `torch.accelerator` di questa tabella richiedono PyTorch 2.11 o versioni
successive. Nel raro caso in cui usi una versione precedente di PyTorch, usa
invece l'API `torch.tpu` corrispondente (ad esempio `torch.tpu.device_count()`
o `torch.tpu.current_device()`). In PyTorch 2.11 e versioni successive, queste
API `torch.tpu` ridondanti sono interne a TorchTPU e chiamarle direttamente dal
codice utente genera un errore.

```python
import torch

# Initialize hardware
device = torch.device("tpu")

print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
print(f"torch.accelerator.device_count(): {torch.accelerator.device_count()}")
print(
    "torch.accelerator.current_device_index():"
    f" {torch.accelerator.current_device_index()}"
)
print(
    "torch.tpu.get_device_properties().name:"
    f" {torch.tpu.get_device_properties().name}"
)
```

## **Generazione di numeri casuali (RNG)** { #random-number-generation-rng }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API TPU / Accelerator | Descrizione |
| :--- | :--- | :--- |
| `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | Imposta il seed del generatore di numeri casuali del dispositivo TPU attivo. |
| `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | Imposta il seed dell'RNG su tutti i dispositivi TPU locali. |
| `torch.cuda.initial_seed()` | `torch.accelerator.random.initial_seed()` | Restituisce il seed casuale iniziale del dispositivo TPU corrente. |
| `torch.cuda.get_rng_state()` | `torch.accelerator.random.get_rng_state()` | Restituisce il tensore di stato dell'RNG per il dispositivo corrente. |
| `torch.cuda.set_rng_state(new_state, device=...)` | `torch.tpu.set_rng_state(new_state, device=...)` | Imposta il tensore di stato dell'RNG per il dispositivo TPU attivo del processo corrente. |
| `torch.cuda.default_generators` | `torch.tpu.default_generators` | Tupla, inizializzata in modo lazy, delle istanze `torch.Generator` predefinite, indicizzate per ordinale del dispositivo locale. |
<!-- mdformat on -->

Le API `torch.accelerator.random` di questa tabella richiedono PyTorch 2.13 o
versioni successive. Nel raro caso in cui usi una versione precedente di
PyTorch, usa invece `torch.tpu.initial_seed()` o `torch.tpu.get_rng_state()`.
In PyTorch 2.13 e versioni successive, queste API `torch.tpu` sono interne a
TorchTPU e chiamarle dal tuo codice genera un errore.

```python
# Set the seed across all cores
torch.tpu.manual_seed_all(42)
print("Seeded all TPU cores with value 42.")
```

## **Automatic Mixed Precision (AMP)** { #automatic-mixed-precision-amp }

TorchTPU si integra con il framework standard
[`torch.amp.autocast`](https://docs.pytorch.org/docs/stable/amp.html) di
PyTorch tramite la stringa di tipo di dispositivo `"tpu"`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API TPU | Descrizione |
| :--- | :--- | :--- |
| `torch.amp.autocast("cuda", ...)` | `torch.amp.autocast("tpu", ...)` | Context manager / decoratore che abilita la precisione mista automatica (`bfloat16` per impostazione predefinita su TPU). |
| `torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | Elenca i dtype supportati su TPU (`bfloat16`, `float16`, `float32`, `float64`, `float8_*`, tipi interi, booleani e complessi). |
<!-- mdformat on -->

```python
supported_dtypes = torch.tpu.get_amp_supported_dtype()
print(f"AMP supported dtypes: {supported_dtypes}")
```

## **Stream ed eventi** { #streams-and-events }

Per la gestione degli stream e la sincronizzazione dei dispositivi, usa le API
standard
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
invece delle API specifiche del backend. Usa `torch.tpu` solo quando costruisci
direttamente oggetti `Stream` o `Event` della TPU o configuri il pinning dei
core TPU.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API TPU / Accelerator | Descrizione |
| :--- | :--- | :--- |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | Restituisce lo stream attivo per un dispositivo. |
| `torch.cuda.set_stream(stream)` | `torch.accelerator.set_stream(stream)` | Imposta lo stream attivo. |
| `torch.cuda.default_stream(device=None)` | `torch.tpu.default_stream(device=None)` | Restituisce il `TpuStream` predefinito del dispositivo. |
| `torch.cuda.stream(stream)` | `torch.tpu.stream(stream)` | Context manager che seleziona un determinato `TpuStream`. |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` | Attende il completamento di tutto il lavoro in sospeso sul dispositivo TPU. |
| `torch.cuda.Stream` | `torch.tpu.Stream(device=None, priority=0, *, pinned=None)` | Sottoclasse di `torch.Stream`. Supporta `synchronize()`, `record_event()`, `wait_event()`, `wait_stream()` e il pinning dei core (`pinned="dense"` o `"sparse"`, verificabile tramite `is_dense` / `is_sparse`). |
| `torch.cuda.Event` | `torch.tpu.Event(enable_timing=False)` | Sottoclasse di `torch.Event`. Supporta `record()`, `synchronize()`, `query()` e `elapsed_time(end_event)`. |
<!-- mdformat on -->

Le API `torch.accelerator` di questa tabella richiedono PyTorch 2.11 o versioni
successive. Nel raro caso in cui usi una versione precedente di PyTorch, usa
invece `torch.tpu.current_stream()`, `torch.tpu.set_stream()` o
`torch.tpu.synchronize()`.

> **Note:** gli stream sono un concetto di CUDA senza un equivalente diretto
> sulle TPU. Nel modello di esecuzione differita di TorchTPU, il compilatore XLA
> e il runtime PjRt pianificano e sovrappongono automaticamente le operazioni in
> base alle dipendenze dei dati, quindi non serve gestire manualmente più
> stream.

## **API specifiche delle TPU** { #tpu-specific-apis }

Oltre agli equivalenti CUDA standard, `torch.tpu` espone API progettate
specificamente per le modalità di esecuzione delle TPU, la precisione della
MXU, la compilazione XLA, la topologia distribuita, i kernel Pallas
personalizzati, gli ottimizzatori tracciabili e la profilazione.

### **Modalità di esecuzione e opzioni del compilatore** { #execution-modes-and-compiler-options }

Controlla come le operazioni eager vengono accodate, fuse e compilate da XLA.
Per una panoramica dell'architettura, consulta
[Modalità di esecuzione di TorchTPU](./torchtpu_execution_model.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrizione |
| :--- | :--- |
| `torch.tpu.EagerMode` | Enum che controlla l'esecuzione eager: `DEFER_AND_FUSE`, `DEFER_NEVER` e `DEFER_NEVER_AND_LAUNCH_BLOCKING`. |
| `torch.tpu.eager_mode` | Legge o imposta globalmente l'`EagerMode` attiva. |
| `torch.tpu.set_eager_mode(mode)` | Context manager che imposta temporaneamente l'`EagerMode` attiva. |
| `torch.tpu.compiler.custom_compiler_options(options)` | Context manager che passa flag personalizzati del compilatore XLA alle operazioni eager materializzate all'interno del blocco. |
<!-- mdformat on -->

```python
print(f"Default eager_mode: {torch.tpu.eager_mode}")
with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
  x = torch.ones((4, 4), device="tpu") + 1
  print(
      f"Inside context manager: {torch.tpu.eager_mode}, sum={x.sum().item()}"
  )
```

### **Precisione numerica** { #numerical-precision }

Controlla la precisione della Matrix Multiply Unit (MXU) e l'accumulo
intermedio in `float32`. Per indicazioni dettagliate, consulta
[Precisione numerica](./numerical_precision.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrizione |
| :--- | :--- |
| `torch.tpu.Precision` | Enum che controlla la precisione della MXU per matmul e convoluzioni in `float32`: `Precision.DEFAULT`, `Precision.HIGH` e `Precision.HIGHEST`. |
| `torch.tpu.precision(level)` | Context manager che imposta il livello di `Precision` per le operazioni di matmul e convoluzione all'interno del blocco. |
| `torch.tpu.allow_excess_precision` | Legge o imposta globalmente il flag `--xla_allow_excess_precision` di XLA per mantenere in `float32` i valori fusi intermedi. |
<!-- mdformat on -->

```python
a = torch.randn(64, 64, device="tpu", dtype=torch.float32)
b = torch.randn(64, 64, device="tpu", dtype=torch.float32)
with torch.tpu.precision(torch.tpu.Precision.HIGH):
  c = torch.matmul(a, b)
print(f"Computed HIGH-precision matmul with output shape: {tuple(c.shape)}")
```

### **Addestramento distribuito e topologia** { #distributed-training-and-topology }

TorchTPU registra `"tpu_dist"` (con l'alias `"tpu"`) come backend
`torch.distributed` predefinito per i dispositivi `"tpu"`. Consulta
[Inizia a utilizzare TorchTPU su GKE](./get_started_on_gke.md) e
[Best practice per FSDP](./case_studies_fsdp.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrizione |
| :--- | :--- |
| `torch.tpu.topology_aware_mesh(mesh_shape, ...)` | Calcola un tensore di disposizione dei rank consapevole della topologia, di forma `mesh_shape`, per `DeviceMesh("tpu", mesh=...)`. |
| `torch.tpu.distributed.set_tpu_launch_env(...)` | Imposta le variabili di ambiente della topologia TPU e dell'avvio multi-host prima di generare i worker. |
| `torch.tpu.distributed.spmd_safe(func)` | Decoratore che dichiara che tutti i rank eseguono operazioni TPU identiche in `func`, abilitando la fusione delle operazioni collettive in modalità eager. |
| `torch.tpu.window_num_stripes` | Legge o imposta il numero massimo di stripe DMA concorrenti per i trasferimenti peer-to-peer. |
| `torch.tpu.window_stripe_chunk_mb` | Legge o imposta la dimensione minima del payload (in MB) per stripe DMA nei trasferimenti peer-to-peer. |
<!-- mdformat on -->

### **Operatori personalizzati TPU (`torch.tpu.ops`) e Pallas (`torch.tpu.pallas`)** { #tpu-custom-operators-torchtpuops-and-pallas-torchtpupallas }

TorchTPU espone operatori specializzati per l'hardware in `torch.tpu.ops` e un
wrapper per kernel JAX/Pallas personalizzati in `torch.tpu.pallas`. Consulta
[Aggiungere kernel personalizzati con Pallas](./custom_kernels.md).

#### **Operatori generali, di attention, MoE e collettivi** { #general-attention-moe-and-collective-operators }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrizione |
| :--- | :--- |
| `torch.tpu.ops.max_pool{1,2,3}d(...)` | Applica il max pooling 1D, 2D o 3D su TPU senza materializzare un tensore di indici. |
| `torch.tpu.ops.max_pool{1,2,3}d_backward(...)` | Calcola il backward pass di `max_pool{1,2,3}d` senza un tensore di indici. |
| `torch.tpu.ops.optimization_barrier(inputs)` | Impedisce a XLA di spostare, fondere o deduplicare i calcoli attraverso la barriera. |
| `torch.tpu.ops.splash_sdpa(q, k, v, ...)` | Scaled dot-product attention basata sul kernel Pallas Splash Attention per TPU. |
| `torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes, ...)` | Moltiplicazione di matrici raggruppata per i carichi di lavoro Mixture-of-Experts (MoE). |
| `torch.tpu.ops.ragged_dot_weight_grad(lhs, grad_output, group_sizes)` | Calcola il gradiente dei pesi per `ragged_dot`. |
| `torch.tpu.ops.align_jagged_to_multiple(nt, ...)` | Aggiunge padding a un buffer `NestedTensor` `torch.jagged` fino a un multiplo specificato, per l'allineamento su TPU. |
| `torch.tpu.ops.strip_jagged_padding(nt, ...)` | Rimuove il padding finale aggiunto da `align_jagged_to_multiple`. |
| `torch.tpu.ops.all_gather_dim(input, dim, group=None)` | Esegue l'all-gather di `input` su `group` e concatena lungo la dimensione `dim`. |
| `torch.tpu.ops.reduce_scatter_dim(input, dim, group=None, ...)` | Riduce `input` su `group` e distribuisce i blocchi lungo la dimensione `dim`. |
| `torch.tpu.ops.ragged_all_to_all(...)` | Scambia slice di tensori di lunghezza variabile tra i rank di un process group. |
| `torch.tpu.ops.send(tensors, dst, tag)` / `torch.tpu.ops.recv(tensors, src, tag)` | Invio e ricezione point-to-point di tensori sul process group TPU predefinito. |
| `torch.tpu.pallas.jax_op(name, fn=None, ...)` | Registra una funzione JAX/Pallas come operatore personalizzato PyTorch tracciabile. |
<!-- mdformat on -->

#### **Operatori di embedding SparseCore** { #sparsecore-embedding-operators }

Questi operatori richiedono una TPU con SparseCore. La proprietà
`sparse_cores_per_device` di `torch.tpu.get_device_properties()` indica il
numero di SparseCore per dispositivo (0 sulle TPU senza SparseCore), mentre
`sparse_core_simd_width` ne indica la larghezza SIMD.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrizione |
| :--- | :--- |
| `torch.tpu.ops.sparse_dense_matmul(...)` | Esegue lookup e aggregazione di embedding CSR su TPU SparseCore. |
| `torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(...)` / `_with_adagrad(...)` / `_with_adam(...)` | Calcola i gradienti del backward per `sparse_dense_matmul` e applica su SparseCore un aggiornamento fuso in-place della tabella con SGD, Adagrad o Adam. |
| `torch.tpu.ops.sparse_dense_matmul_activation_unstack(...)` | Separa le attivazioni di embedding SparseCore impilate in tensori per singola feature. |
| `torch.tpu.ops.sparse_dense_matmul_gradient_stack(...)` | Impila i gradienti delle attivazioni per singola feature in un unico tensore per SparseCore. |
| `torch.tpu.ops.sparse_gather(...)` | Raccoglie righe da un tensore 2D usando indici CSR su TPU SparseCore. |
| `torch.tpu.ops.sparse_gather_backward(...)` | Esegue lo scatter-add dei gradienti per `sparse_gather` su TPU SparseCore. |
| `torch.tpu.ops.sparse_iota(...)` | Genera gli indici di posizione all'interno della riga per gli elementi diversi da zero di una matrice CSR su SparseCore. |
| `torch.tpu.ops.sparse_map_row(...)` | Applica un calcolo MLIR personalizzato riga per riga su matrici CSR sparse su SparseCore. |
<!-- mdformat on -->

### **Step di addestramento fuso e ottimizzatori (`torch.tpu.optimizers`)** { #fused-training-step-and-optimizers-torchtpuoptimizers }

`torch.tpu.optimizers` traccia il forward pass, il calcolo della loss, il
backward pass e lo step funzionale dell'ottimizzatore in un unico grafo TPU
compilato.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrizione |
| :--- | :--- |
| `torch.tpu.optimizers.SingleTraceTrainer(model, optimizer=None, ...)` | Gestisce i parametri funzionali (`params`) e lo stato dell'ottimizzatore (`param_group`) e compila l'intero step di addestramento tramite `make_compiled_train_step(...)`. |
| `torch.tpu.optimizers.Optimizer(lr=..., weight_decay=...)` | Classe base per gli ottimizzatori funzionali stateless (espone `defaults`, `init_param_group` e `step`). |
| `torch.tpu.optimizers.AdamW(...)` | Ottimizzatore AdamW funzionale e tracciabile (`AdamW.ParamGroup`). |
| `torch.tpu.optimizers.Muon(...)` | Ottimizzatore Muon funzionale e tracciabile (`Muon.ParamGroup`). |
| `torch.tpu.optimizers.Sgd(...)` | Ottimizzatore SGD funzionale e tracciabile (`Sgd.ParamGroup`). |
<!-- mdformat on -->

### **Profilazione (`torch.tpu.profiler`)** { #profiling-torchtpuprofiler }

Configura il tracciamento dell'hardware TPU e di Python per XProf e TensorBoard.
Consulta [XProf per TorchTPU](./performance_profiling_with_xprof.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrizione |
| :--- | :--- |
| `torch.tpu.profiler.TpuProfilerConfig(...)` | Configurazione sperimentale passata a `torch.profiler.profile(..., experimental_config=...)` per configurare il tracciamento dell'host, del dispositivo TPU e di Python per XProf. |
<!-- mdformat on -->

## **Variabili di ambiente** { #environment-variables }

Il comportamento di TorchTPU è controllato da diverse variabili di ambiente che
configurano il compilatore XLA, il driver TPU (`libtpu`) e il sottosistema di
debug di PyTorch.

> **CRITICO: inizializzazione globale del processo** Le variabili di ambiente
> di TorchTPU vengono **bloccate** nel momento in cui il backend si inizializza
> (di solito durante `import torch` o `torch.device("tpu")`).

### **Visibilità dei dispositivi, runtime e avvio distribuito** { #device-visibility-runtime-and-distributed-launch }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variabile di ambiente | Descrizione |
| :--- | :--- |
| `TPU_VISIBLE_DEVICES` | Indici dei chip TPU, separati da virgole, visibili al processo corrente (analoga a `CUDA_VISIBLE_DEVICES`). |
| `RANK` | Rank globale del processo corrente in un job distribuito. |
| `LOCAL_RANK` | Rank locale all'host del processo corrente. |
| `WORLD_SIZE` | Numero totale di processi nel job distribuito. |
| `NNODES` | Numero totale di nodi host in un job multi-host. |
| `NODE_RANK` | Rank del nodo host corrente in un job multi-host. |
| `MASTER_ADDR` | Indirizzo del nodo coordinatore di rank 0 per `torch.distributed`. |
| `MASTER_PORT` | Porta su `MASTER_ADDR` usata per il rendezvous di `torch.distributed`. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | Imposta `"0"` per disattivare la registrazione automatica del backend `torch_tpu`. |
| `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS` | Se abilitata, consente di chiamare le API interne dei dispositivi `torch.tpu` (quelle con un equivalente `torch.accelerator`) dall'esterno del core di PyTorch, con un avviso emesso una sola volta invece di generare un `RuntimeError`. Usala solo come soluzione temporanea: è preferibile migrare a `torch.accelerator.*` e questa variabile verrà probabilmente rimossa in una release futura. |
| `LIBTPU_INIT_ARGS` | Flag di inizializzazione, separati da spazi, passati a `libtpu` e al runtime PJRT. |
| `TPU_PREMAPPED_BUFFER_SIZE` | Dimensione in byte del buffer di memoria host-dispositivo premappato per il client PJRT. |
| `TMPDIR` | Directory temporanea usata dal runtime e dal compilatore. |
<!-- mdformat on -->

### **Topologia hardware e coordinamento multi-host** { #hardware-topology-and-multi-host-coordination }

Queste variabili specificano la topologia hardware delle TPU e la rete dei
processi. Nella maggior parte dei casi vengono impostate automaticamente da
`torch.tpu.distributed.set_tpu_launch_env()`, dal launcher di TorchTPU o
dall'ambiente Google Cloud.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variabile di ambiente | Descrizione |
| :--- | :--- |
| `TORCH_TPU_TOPOLOGY` | Topologia fisica della slice TPU (ad esempio `2x2x1`, `2x2x4`). |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Indirizzi `host:port` dei worker, separati da virgole, per il servizio slice builder delle TPU. |
| `TORCH_TPU_DEVICES_PER_HOST` | Numero di chip TPU per host quando si ricavano gli indirizzi dello slice builder. |
| `TORCH_TPU_BASE_PORT` | Numero di porta iniziale quando si ricavano gli indirizzi dello slice builder. |
| `TORCH_TPU_SPMD_HANDSHAKE_STAGE` | Fase in cui i rank verificano che i grafi delle operazioni collettive coincidano tra i rank. |
| `TORCHCOMM_RANK` | Rank del processo per `torchcomm` (ha la precedenza su `RANK`). |
| `TORCHCOMM_SIZE` | World size per `torchcomm` (ha la precedenza su `WORLD_SIZE`). |
<!-- mdformat on -->

### **Modalità di esecuzione, compilazione e cache** { #execution-mode-compilation-and-caching }

Queste variabili controllano come le operazioni e i grafi PyTorch vengono
inviati, abbassati (lowering), ottimizzati da XLA e memorizzati nella cache tra
esecuzioni e rank.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variabile di ambiente | Descrizione |
| :--- | :--- |
| `TORCH_TPU_DEFER_AND_FUSE` | Se abilitata (`"1"`/`"true"`), imposta la modalità eager predefinita su `EagerMode.DEFER_AND_FUSE`. |
| `TORCH_TPU_LAUNCH_BLOCKING` | Se abilitata (`"1"`/`"true"`), imposta la modalità eager predefinita su `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING`. |
| `TORCH_TPU_TIER2_COMPILATION_CACHE` | Nome della directory in `/dev/shm/torch_tpu_cache/` per la cache di compilazione locale all'host (oppure `"disabled"`). |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | Percorso (ad esempio `gs://...` o un percorso locale) della cache di compilazione persistente di livello 3, condivisa tra host ed esecuzioni. |
| `XLA_FLAGS` | Flag del compilatore XLA separati da spazi (ad esempio `--xla_dump_hlo_as_text --xla_dump_to=...`). |
<!-- mdformat on -->

### **Debug, sicurezza, profilazione e tracciamento** { #debugging-safety-profiling-and-tracing }

Queste variabili controllano il contesto di debug, i controlli di sicurezza del
runtime e l'output del profiler per TorchTPU.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variabile di ambiente | Descrizione |
| :--- | :--- |
| `TORCH_SHOW_CPP_STACKTRACES` | Se impostata su `"1"`, arricchisce le eccezioni Python `RuntimeError` con i backtrace C++. |
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` | Se impostata su `"1"`, abilita i controlli di debug e dei limiti a runtime su TPU. |
| `TORCH_TPU_PROFILER_OUTPUT_DIR` | Directory di output di riserva per i file di traccia XProf quando `TpuProfilerConfig.run_dir` non è impostato. |
| `TORCH_TPU_XPROF_SESSION_ID` | ID della sessione di profilazione XProf condivisa tra i worker multi-host. |
| `TORCH_TRACE` | Directory di output per i log di traccia strutturati del compilatore e del dispatcher (`tlparse`). |
<!-- mdformat on -->

### **Esempi di configurazione delle variabili di ambiente** { #environment-variable-configuration-snippets }

#### **1. Stacktrace C++ (`TORCH_SHOW_CPP_STACKTRACES`)** { #1-c-stacktraces-torch_show_cpp_stacktraces }

Impostare `TORCH_SHOW_CPP_STACKTRACES=1` prima di importare `torch` rende molto
più facile diagnosticare gli errori che si verificano in profondità nel backend
C++ o in XLA. Invece di un semplice `RuntimeError` Python, ricevi un `C++
CapturedTraceback` dettagliato che mostra i frame di esecuzione esatti.

```python
import os
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
import torch

device = torch.device("tpu")
a = torch.randn(10, device=device)
b = torch.randn(20, device=device)
# Raises RuntimeError enriched with C++ Error Trace and Captured Traceback:
c = a + b
```

#### **2. Controlli di debug interni e convalida dei limiti (`TORCH_TPU_ENABLE_DEBUG_CHECKS`)** { #2-internal-debug-checks-boundary-validation-torch_tpu_enable_debug_checks }

Per impostazione predefinita, la TPU salta i controlli dei limiti per motivi di
velocità. Abilitare `TORCH_TPU_ENABLE_DEBUG_CHECKS=1` intercetta bug subdoli
legati a indici fuori dai limiti (ad esempio in `torch.take`), anche se
rallenta l'esecuzione a causa della sincronizzazione aggiuntiva tra CPU e TPU.
Le operazioni che altrimenti fallirebbero in modo silenzioso o causerebbero la
corruzione dei dati generano un'eccezione esplicita come `take(): expected indices to be
in range`:

```python
import os
os.environ["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "1"
import torch

device = torch.device("tpu")
x = torch.tensor([1.0, 2.0, 3.0], device=device)
indices = torch.tensor([5], device=device)  # Out of bounds
# Raises an explicit out-of-bounds exception when materialized:
y = torch.take(x, indices).cpu()
```

#### **3. Dump dell'IR / HLO di XLA (`XLA_FLAGS`)** { #3-dumping-xla-ir-hlo-xla_flags }

Impostando `XLA_FLAGS`, puoi forzare il compilatore a eseguire il dump del testo
High-Level Optimizer (HLO) in una directory. La directory di dump (ad esempio
`/tmp/hlo_dump`) conterrà:

*   **`*.after_optimizations.txt`**: il grafo HLO finale con le operazioni a
    livello hardware.
*   **`*.flagfile`**: l'elenco completo dei flag attivi del compilatore XLA
    (verifica `xla_optimization_level`).
*   **`*tpu_comp_env.txt`**: le impostazioni interne dell'ambiente TPU usate
    durante la compilazione.

```python
import os

dump_dir = "/tmp/hlo_dump"
os.makedirs(dump_dir, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"

import torch
device = torch.device("tpu")
x = torch.randn(10, device=device)
y = (x + 1).to("cpu")
```

#### **4. Cache di compilazione multilivello (`TORCH_TPU_TIER2_COMPILATION_CACHE` e `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** { #4-multi-tiered-compilation-caching-torch_tpu_tier2_compilation_cache-torch_tpu_tier3_compilation_cache_root }

TorchTPU usa una cache multilivello per evitare compilazioni ridondanti tra
esecuzioni e rank:

*   **Livello 2 (`TORCH_TPU_TIER2_COMPILATION_CACHE`)**: viene archiviato
    localmente nella memoria condivisa dell'host. Si aspetta il **nome** di una directory
    (ad esempio `"tpu_tier2_cache"`), che il backend colloca automaticamente in
    `/dev/shm/torch_tpu_cache/<name>`.
*   **Livello 3 (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)**: è una cache di
    compilazione persistente condivisa tra host e sessioni (in genere l'URI di
    un bucket `gs://` o il percorso di un file system condiviso).

Gli artefatti binari compilati (eseguibili `*.bin`) vengono salvati nelle
directory della cache in modo asincrono e riutilizzati nelle esecuzioni
successive, riducendo drasticamente la latenza della compilazione JIT:

```python
import os

tier2_name = "tpu_tier2_cache"
tier2_path = os.path.join("/dev/shm/torch_tpu_cache", tier2_name)
tier3_path = "/tmp/tpu_tier3_cache"  # Or "gs://<bucket_name>/<path_to_cache>"
os.makedirs(tier2_path, exist_ok=True)
os.makedirs(tier3_path, exist_ok=True)

# TORCH_TPU_TIER2_COMPILATION_CACHE expects just the directory NAME inside /dev/shm/torch_tpu_cache/
os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = tier2_name
os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = tier3_path

import torch
device = torch.device("tpu")
a = torch.randn(1024, 1024, device=device)
_ = (a @ a).to("cpu")
```
