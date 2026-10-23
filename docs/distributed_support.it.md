# Supporto per il calcolo distribuito in TorchTPU { #torchtpu-distributed-support }

## Supporto per il calcolo distribuito { #distributed-support }

TorchTPU implementa la comunicazione distribuita a livello di ProcessGroup di
PyTorch (`c10d::Backend`) con il nome di backend `tpu_dist`. Intercettando le
operazioni collettive a livello di ProcessGroup, TorchTPU supporta le API
standard `torch.distributed` sia in modalità eager sia con `torch.compile`, e
funziona direttamente con i framework distribuiti di PyTorch, tra cui DTensor,
DistributedDataParallel (DDP), FSDPv2 e
[SimpleFSDP](https://arxiv.org/abs/2411.00284).

Come PyTorch su GPU, TorchTPU impone un modello di esecuzione con un processo
per dispositivo. Internamente, la comunicazione da TPU a TPU non usa una
libreria di comunicazione di runtime separata (come NCCL); le operazioni
collettive vengono invece abbassate (lowering) direttamente in
[operazioni collettive StableHLO](https://openxla.org/stablehlo/spec#collective_ops)
(come `stablehlo.all_reduce`, `stablehlo.all_gather`,
`stablehlo.reduce_scatter` e `stablehlo.all_to_all`) e compilate da XLA
nell'eseguibile TPU.

### Principali differenze di architettura rispetto alle GPU { #key-architectural-differences-from-gpu }

TorchTPU mantiene la semantica standard di `torch.distributed`, ma programmare
le TPU tramite StableHLO e PjRT introduce tre differenze chiave rispetto ai
backend GPU:

*   **Rilevamento della topologia hardware prima dell'inizializzazione:** su un
    host multi-GPU, ogni processo avviato da torchrun può esaminare tutte le GPU
    collegate, selezionare il dispositivo assegnato tramite `cuda:{local_rank}`
    e spostare i modelli su quel dispositivo prima di chiamare
    `torch.distributed.init_process_group()`. Su un host TPU, ogni processo
    worker è associato a un singolo dispositivo TPU e non può possedere da solo
    l'intero insieme di dispositivi. Prima di inizializzare il runtime PjRT o di
    chiamare `torch.distributed.init_process_group()`, l'ambiente deve
    specificare la geometria fisica della mesh della
    [slice TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    (`TORCH_TPU_TOPOLOGY`), gli indirizzi dei coordinatori locali usati per
    costruire la mesh Inter-Chip Interconnect (ICI)
    (`TORCH_TPU_SLICEBUILDER_ADDRESSES`) e il numero totale di dispositivi TPU
    partecipanti (`WORLD_SIZE`).
*   **Risoluzione globale dei gruppi di repliche per i sottogruppi:** in
    PyTorch, una chiamata collettiva come
    [`torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
    riceve solo il ProcessGroup (sottogruppo) a cui partecipa il rank
    chiamante. Al contrario,
    [`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce)
    richiede una descrizione completa di tutti i sottogruppi partecipanti
    nell'intero process group predefinito (`replica_groups`), anche di quelli a
    cui il processo corrente non appartiene:

    ```
    %0 = "stablehlo.all_reduce"(%arg0) <{
      replica_groups = dense<[[4, 0, 1, 3], [5, 2, 6, 7]]> : tensor<2x4xi64>
    }> ({ ... })
    ```

    Il backend `tpu_dist` di TorchTPU scambia automaticamente l'appartenenza ai
    sottogruppi tra tutti i rank del process group predefinito durante la
    configurazione delle operazioni collettive e costruisce i valori globali
    `replica_groups` e `device_assignment` richiesti da XLA.
*   **Compilazione SPMD ed esecuzione MPMD:** il compilatore XLA e lo stack di
    comunicazione TPU sono progettati per l'esecuzione Single Program, Multiple
    Data (SPMD), in cui ogni rank compila ed esegue un modulo StableHLO
    identico che contiene le stesse operazioni collettive. I programmi PyTorch,
    però, possono avere un comportamento Multiple Program, Multiple Data (MPMD)
    tra i rank, ad esempio a causa di rami condizionati dal rank
    (`if dist.get_rank() == 0:`) o di uno sharding non uniforme dei tensori
    quando una dimensione non è divisibile per il numero di dispositivi. Se rank
    diversi eseguono moduli StableHLO diversi che contengono operazioni
    collettive, ID dei flag di sincronizzazione o offset dei buffer non
    corrispondenti tra i rank possono causare deadlock di runtime o
    corruzione della memoria.

### Modalità di esecuzione e fusione delle operazioni collettive { #collective-execution-and-fusion-modes }

TorchTPU esegue ogni operazione collettiva come skinny collective. Per permettere
a XLA di sovrapporre calcolo e comunicazione, puoi attivare uno o entrambi i
meccanismi che fondono le operazioni collettive con il calcolo circostante. I
due meccanismi sono indipendenti e si possono combinare: le operazioni
collettive all'interno di una funzione `@spmd_safe` vengono sempre fuse e, se
abilitato, il Compilation Handshake decide per le restanti operazioni
collettive di ogni grafo `torch.compile`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Modalità | Configurazione | Comportamento del grafo | Ideale per |
| :---- | :---- | :---- | :---- |
| Skinny collective (comportamento predefinito) | Sempre attiva per qualsiasi operazione collettiva che le opzioni facoltative seguenti non fondono | Compila ogni operazione collettiva in un modulo StableHLO autonomo ("skinny") inserendo graph break attorno alle operazioni collettive. | Sicurezza MPMD generale fin da subito, al costo di impedire a XLA di sovrapporre calcolo e comunicazione. |
| Decoratore `@spmd_safe` (facoltativo) | `@torch.tpu.distributed.spmd_safe` | Disattiva i graph break attorno alle operazioni collettive all'interno della funzione decorata, fondendo calcolo e operazioni collettive in un unico modulo StableHLO. | Funzioni o step di addestramento notoriamente SPMD (ad esempio SimpleFSDP o tensor parallelism manuale), per massimizzare la sovrapposizione tra calcolo e comunicazione in XLA. |
| Compilation Handshake (facoltativo) | `TORCH_TPU_SPMD_HANDSHAKE_STAGE=COMPILE_STAGE` o `DISPATCH_STAGE` (predefinito: OFF) | Nei grafi torch.compile, verifica che i grafi FX compilati coincidano su tutti i rank; se coincidono fonde le operazioni collettive, altrimenti torna alle skinny collective. | Abilitare automaticamente la fusione delle operazioni collettive in `torch.compile` proteggendosi da divergenze MPMD accidentali. |
<!-- mdformat on -->

#### Usare `@torch.tpu.distributed.spmd_safe` { #using-torchtpudistributedspmd_safe }

Quando sai che una funzione o uno step del modello è strettamente SPMD su tutti
i rank (senza flusso di controllo dipendente dal rank o forme dei tensori non
uniformi), decorala con `@torch.tpu.distributed.spmd_safe`. Con `torch.compile`
(e con la modalità eager `DEFER_AND_FUSE`), questo rimuove i graph break attorno
alle operazioni collettive, così XLA può pianificare la comunicazione asincrona
insieme al calcolo matriciale:

```python
import torch
import torch.distributed as dist

@torch.compile
@torch.tpu.distributed.spmd_safe
def spmd_train_step(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
  out = x @ weight
  dist.all_reduce(out, op=dist.ReduceOp.AVG)
  return torch.nn.functional.silu(out)
```

Applica `@torch.tpu.distributed.spmd_safe` solo al codice che esegue
esattamente la stessa sequenza di operazioni sui tensori e collettive su ogni
rank; applicarlo a codice MPMD può produrre moduli StableHLO diversi tra i rank
e causare deadlock nell'esecuzione su TPU.

#### Configurare il compilation handshake { #configuring-the-compilation-handshake }

Invece di annotare manualmente le funzioni con `@spmd_safe`, puoi abilitare il
Compilation Handshake di TorchTPU tramite la variabile di ambiente
`TORCH_TPU_SPMD_HANDSHAKE_STAGE`. Quando `torch.compile` traccia un grafo che
contiene operazioni collettive globali, TorchTPU esegue un controllo di
consenso tra tutti i rank per verificare che i loro grafi FX compilati
coincidano prima di fondere le operazioni collettive nel grafo:

*   `OFF` (predefinito):** disattiva l'handshake; le operazioni collettive usano
    le skinny collective predefinite, a meno che non siano contrassegnate con
    `@torch.tpu.distributed.spmd_safe`.
*   `COMPILE_STAGE`: verifica il consenso sul grafo tra i rank prima della
    compilazione PjRT. Se tutti i rank producono grafi identici, i graph break
    attorno alle operazioni collettive vengono eliminati. Consigliato per i
    carichi di lavoro `torch.compile` standard in cui tutti i rank compilano insieme.
*   `DISPATCH_STAGE`: compila le operazioni collettive senza graph break e
    verifica il consenso tra i rank al momento del dispatch, ricompilando con
    operazioni collettive separate se i rank divergono. Questa modalità è
    necessaria per i carichi di lavoro che possono causare ricompilazioni specifiche di
    un rank; ha un overhead maggiore di `COMPILE_STAGE`.

Di seguito trovi un esempio di ricompilazione specifica di un rank supportata da
`DISPATCH_STAGE`, mentre con `COMPILE_STAGE` si verificherebbe un timeout
perché solo `divergent_rank` ricompila all'iterazione 1:

```python
iteration = 0
divergent_rank = ...

def divergence_after_iteration(x, y):
  z = torch.add(x, y)
  if rank == divergent_rank and iteration == 1:
    z = torch.mul(z, 5.0)
  dist.all_reduce(z)
  return z

compiled_fn = torch.compile(divergence_after_iteration, backend=backend)
res0 = compiled_fn(x, y)
iteration += 1
# IMPORTANT: If Handshake is in COMPILE_STAGE a recompilation happens only on `divergent_rank`
# this would initiate a Handshake ONLY ON THAT RANK which would timeout which is why we provide
# a DISPATCH_STAGE mode.
res1 = compiled_fn(x, y)
```

## Inizializzazione distribuita { #distributed-initialization }

Per inizializzare un carico di lavoro distribuito in TorchTPU si usano le API standard
`torch.distributed`, insieme all'helper dell'ambiente di avvio di TorchTPU che
imposta le variabili di ambiente della topologia TPU e dello slice builder:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variabile di ambiente | Descrizione | Esempio |
| :---- | :---- | :---- |
| `TORCH_TPU_TOPOLOGY` | Geometria fisica della mesh della slice TPU. | `"2,2,1"`, `"2,2,1,2"`, `"4,4,4"` |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Indirizzi `host:port` separati da virgole (uno per dispositivo) usati per costruire la mesh ICI. | `"localhost:8471,localhost:8472,..."` |
<!-- mdformat on -->

Devi impostare queste variabili prima di avviare i worker. Nelle configurazioni
comuni (un singolo host o una slice TPU di GKE), TorchTPU può determinare i
valori corretti per te: esegui

```bash
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
```

nello script di avvio prima di torchrun, oppure chiama
`torch.tpu.distributed.set_tpu_launch_env()` nel processo Python di avvio
padre prima di generare i worker. Entrambi i metodi lasciano invariate le
variabili che hai già impostato, quindi puoi sostituire singoli valori.

Di seguito trovi il flusso di inizializzazione di un carico di lavoro
TorchTPU distribuito, seguito da un esempio completo di worker e launcher:

![Flusso di inizializzazione distribuita](./images/distributed.png)

### 1. Script del worker (`train.py`) { #1-worker-script-trainpy }

All'interno di ogni processo worker, inizializza il backend `tpu_dist` e
acquisisci il dispositivo TPU locale tramite `torch.device("tpu")`:

```python
import torch
import torch.distributed as dist

def worker_main():
  # 1. Initialize the distributed process group using the tpu_dist backend`
  dist.init_process_group(backend="tpu_dist")

  # 2. Query rank, world size, and acquire the local TPU device
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  device = torch.device("tpu")

  print(f"[Rank {rank}/{world_size}] Initialized on {device}")

  # 3. Perform a distributed collective operation across TPU workers
  x = torch.ones(4, 4, device=device) * (rank + 1)
  dist.all_reduce(x, op=dist.ReduceOp.SUM)

  print(f"[Rank {rank}] Reduced sum: {x[0, 0].item()}")

  # 4. Clean up process group resources
  dist.destroy_process_group()

if __name__ == "__main__":
  worker_main()
```

### 2. Avvio con torchrun { #2-launching-with-torchrun }

Usa il modulo CLI `torch.tpu.distributed` per rilevare la topologia hardware
della TPU ed esportare le variabili di avvio richieste prima di invocare
`torchrun`:

```bash
# Detect and export TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES,
# WORLD_SIZE, NODE_RANK, and multi-host coordination variables:
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
# Launch 1 worker process per TPU device on the host:
torchrun --nproc_per_node=$WORLD_SIZE train.py
```

In alternativa, se avvii i worker da Python (ad esempio con
`torch.distributed.launcher.elastic_launch`), chiama
`torch.tpu.distributed.set_tpu_launch_env()` nel processo padre prima di
generare i worker. Tieni presente che il process group predefinito comprende
tutti i dispositivi della
[slice TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices).

## Mappatura ottimale della device mesh { #optimal-device-mesh-mapping }

Per massimizzare la larghezza di banda delle operazioni collettive e ridurre al
minimo la latenza, allinea le dimensioni di parallelismo del modello con la
gerarchia fisica dell'interconnessione TPU:

*   **Inter-Chip Interconnect (ICI)**: collegamenti diretti ad alta velocità che
    connettono i chip TPU in una mesh a toro 2D/3D o a toro intrecciato
    (twisted torus) all'interno di una slice (da decine a centinaia di GB/s per
    collegamento).
*   **Data Center Network (DCN)**: rete Ethernet a livello di host che collega
    slice TPU separate nei carichi di lavoro multi-slice (larghezza di banda inferiore e
    latenza maggiore rispetto all'ICI).

Usa `torch.tpu.distributed.topology_aware_mesh()` per calcolare una
disposizione dei rank allineata con gli anelli ICI fisici. Ordina le dimensioni
della mesh logica dall'**intensità di comunicazione più bassa** (la più
esterna) all'**intensità di comunicazione più alta** (la più interna):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Tipo di parallelismo | Operazioni collettive tipiche | Frequenza di comunicazione | Posizionamento fisico consigliato |
| :---- | :---- | :---- | :---- |
| **Tensor Parallel (TP)** | `all_reduce`, `reduce_scatter`, `all_gather` | A ogni layer transformer (forward + backward) | Anello ICI locale più veloce (stesso tray/host, dispositivi contigui a 1 hop) |
| **Context / Sequence Parallel (CP/SP)** | `all_to_all`, Ring Attention | Per ogni layer di attention | Assi ICI locali (all'interno della slice) |
| **Expert Parallel (EP)** | `all_to_all`, `ragged_all_to_all` | Per ogni layer router MoE | Assi ICI locali (all'interno della slice) |
| **Fully Sharded Data Parallel (FSDP)** | `all_gather`, `reduce_scatter` | Per modulo / blocco transformer | Assi ICI (all'interno della slice) |
| **Replicated Data Parallel (DP)** | `all_reduce` (solo gradienti) | Una volta per step di addestramento | Asse ICI esterno o DCN (tra slice) |
<!-- mdformat on -->

### Esempio: costruire una DeviceMesh 2D consapevole della topologia { #example-constructing-a-topology-aware-2d-devicemesh }

```python
import torch
from torch.distributed.device_mesh import DeviceMesh

# Logical mesh shape: (dp=4, tp=2) on an 8-device slice,
# ordered from least communication-intensive (dp) to most intensive (tp):
mesh_shape = (4, 2)
dim_names = ("dp", "tp")

# Compute an optimal rank tensor aligned with physical ICI rings
optimal_rank_tensor = torch.tpu.distributed.topology_aware_mesh(
    mesh_shape=mesh_shape,
)

# Instantiate the PyTorch DeviceMesh
mesh_2d = DeviceMesh(
    "tpu",
    mesh=optimal_rank_tensor,
    mesh_dim_names=dim_names,
    preserve_rank_order=True,
)
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]
```

Per i carichi di lavoro multi-slice (ibridi DCN + ICI), passare `dcn_mesh_shape` insieme
a `mesh_shape` è facoltativo e serve solo a scegliere la suddivisione in modo
esplicito: per impostazione predefinita,
`torch.tpu.distributed.topology_aware_mesh()` rileva automaticamente le slice e
mappa la dimensione della mesh non banale più esterna (in genere il data
parallel) tra le slice tramite DCN e le dimensioni interne all'interno di ogni
slice tramite ICI.

## Kernel Pallas distribuiti { #distributed-pallas-kernels }

TorchTPU supporta kernel
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) personalizzati che
eseguono Remote DMA (RDMA) diretta tra dispositivi tramite ICI senza
coinvolgere la CPU dell'host. Questo ti permette di scrivere kernel di
comunicazione point-to-point o ad anello personalizzati, come Ring Attention,
shuffle All-to-All personalizzati, shift di pipeline parallel o scambi di halo,
e di invocarli come operatori personalizzati PyTorch nativi sia in modalità
eager sia con
`torch.compile`.

### Scrivere e registrare un kernel Pallas che comunica { #writing-and-registering-a-communicating-pallas-kernel }

L'integrazione di un kernel Pallas distribuito prevede tre passaggi:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Fase | Passaggi e operazioni principali |
| :---- | :---- |
| **1. Corpo del kernel Pallas** | Interroga le coordinate logiche con `jax.lax.axis_index` e `jax.lax.axis_size`, costruisci un descrittore di trasferimento asincrono tramite `pltpu.make_async_remote_copy(..., device_id=(target,))` e gestisci il completamento con `.start()` e `.wait()`. |
| **2. `pl.pallas_call` e `jax.shard_map`** | Alloca i semafori DMA (`pltpu.SemaphoreType.DMA`) in `scratch_shapes` e racchiudi il kernel con `jax.shard_map` su una `jax.sharding.Mesh` allineata ai rank e di dimensione pari al world size, con `PartitionSpec`. |
| **3. Registrazione in TorchTPU** | Registra una funzione wrapper JAX con annotazioni di tipo usando `torch.tpu.pallas.jax_op`, passando `mesh`, `input_partition_specs` e `output_partition_specs`. |
<!-- mdformat on -->

```python
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torch

P = jax.sharding.PartitionSpec

# 1. Pallas Kernel Body: Asynchronously sends local buffer to neighbor rank (rank + 1)
def pallas_send_to_next_kernel(x_ref, y_ref, send_sem, recv_sem):
  rank = jax.lax.axis_index("x")
  world_size = jax.lax.axis_size("x")
  dest_id = (rank + 1) % world_size
  # Trigger hardware Remote DMA over TPU interconnect
  async_copy = pltpu.make_async_remote_copy(
      src_ref=x_ref,
      dst_ref=y_ref,
      send_sem=send_sem,
      recv_sem=recv_sem,
      device_id=(dest_id,),
  )
  async_copy.start()
  async_copy.wait()

# 2. JAX Wrapper with ShardMap and DMA Semaphores
def build_jax_fn(mesh):
  def local_pallas_call(x: jax.Array) -> jax.Array:
    return pl.pallas_call(
        pallas_send_to_next_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(x.shape)],
            out_specs=pl.BlockSpec(x.shape),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),  # send & recv semaphores
        ),
    )(x)

  return jax.shard_map(
      local_pallas_call,
      mesh=mesh,
      in_specs=P("x"),
      out_specs=P("x"),
      check_vma=False,
  )

# 3. Create the TorchTPU callable wrapper
def get_send_to_next_op(world_size: int):
  mesh = torch.tpu.pallas.get_pallas_mesh(
      axis_names=("x",), mesh_shape=(world_size,)
  )
  jax_fn = build_jax_fn(mesh)
  return torch.tpu.pallas.jax_op(
      "pallas::send_to_next",
      jax_fn,
      mesh=mesh,
      input_partition_specs=[P("x")],
      output_partition_specs=[P("x")],
  )
```

### Mesh allineate ai rank e assegnazione dei dispositivi { #rank-aligned-meshes-and-device-assignment }

In molte architetture TPU gli ID dei dispositivi e i rank assegnati non hanno
una corrispondenza 1 a 1: i rank vengono assegnati in modo che i vicini siano i
dispositivi adiacenti (il rank 1 è accanto al rank 0 e al rank 2), ma in questo
scenario gli ID dei dispositivi su v7x sono 0, 1 e 4. Per semplificare la
conversione, TorchTPU fornisce `torch.tpu.pallas.get_pallas_mesh`, che
restituisce una mesh Pallas/JAX che corrisponde esattamente all'ID di rank
usato nel resto dell'applicazione Torch.

*   **Costruzione di una mesh allineata ai rank**:
    `torch.tpu.pallas.get_pallas_mesh(axis_names=..., mesh_shape=...,
    process_group=...)` legge la mappatura rank-dispositivo del `ProcessGroup`
    PyTorch attivo e costruisce una `jax.sharding.Mesh` di dispositivi astratti
    solo per la compilazione, senza inizializzare un backend JAX PjRT.
*   **Requisito di una mesh grande quanto il world size**: i kernel Pallas
    distribuiti richiedono una mesh il cui numero totale di dispositivi sia
    uguale a `dist.get_world_size()`. Per le topologie parallele
    multidimensionali (come data parallelism + tensor parallelism), passa un
    `mesh_shape` il cui prodotto sia uguale a `world_size`, ad esempio
    `torch.tpu.pallas.get_pallas_mesh(axis_names=("dp", "tp"), mesh_shape=(2,
    4))`.
*   **Assegnazione unificata dei dispositivi per operazioni collettive e
    kernel**: quando passi `mesh` a `torch.tpu.pallas.jax_op`, TorchTPU estrae
    la mappatura rank-dispositivo della mesh come `DeviceAssignment` XLA per il
    kernel e la riconcilia con eventuali operazioni collettive
    `torch.distributed` standard fuse nello stesso modulo compilato. Questo
    garantisce che `jax.lax.axis_index`, le DMA indirizzate tramite mesh
    (`device_id={axis: target}`) e le operazioni collettive PyTorch vengano
    instradati in modo coerente secondo l'ordine dei rank di PyTorch.
