# Unterstützung für verteiltes Rechnen in TorchTPU { #torchtpu-distributed-support }

## Unterstützung für verteiltes Rechnen { #distributed-support }

TorchTPU implementiert die verteilte Kommunikation auf der Ebene der
ProcessGroup von PyTorch (`c10d::Backend`) unter dem Backend-Namen `tpu_dist`.
Da TorchTPU Collectives auf der ProcessGroup-Ebene abfängt, unterstützt es die
Standard-APIs von `torch.distributed` sowohl im Eager-Modus als auch mit
`torch.compile`. Es funktioniert direkt mit verteilten PyTorch-Frameworks wie
DTensor, DistributedDataParallel (DDP), FSDPv2 und
[SimpleFSDP](https://arxiv.org/abs/2411.00284).

Wie PyTorch auf GPU erzwingt TorchTPU ein Ausführungsmodell mit einem Prozess
pro Gerät. Intern verwendet die Kommunikation zwischen TPUs keine separate
Kommunikationsbibliothek zur Laufzeit (wie NCCL). Stattdessen werden
kollektive Operationen direkt in
[kollektive StableHLO-Operationen](https://openxla.org/stablehlo/spec#collective_ops)
(z. B. `stablehlo.all_reduce`, `stablehlo.all_gather`,
`stablehlo.reduce_scatter` und `stablehlo.all_to_all`) gelowert und von
XLA in das ausführbare TPU-Programm kompiliert.

### Wichtige Architekturunterschiede zu GPU { #key-architectural-differences-from-gpu }

TorchTPU behält die Standardsemantik von `torch.distributed` bei. Die
Programmierung von TPUs über StableHLO und PjRT bringt im Vergleich zu
GPU-Backends aber drei wichtige Unterschiede mit sich:

*   **Erkennung der Hardwaretopologie vor der Initialisierung:** Auf einem Host
    mit mehreren GPUs kann jeder von torchrun gestartete Prozess alle
    angeschlossenen GPUs prüfen, sein zugewiesenes Gerät über
    `cuda:{local_rank}` auswählen und Modelle auf dieses Gerät verschieben,
    bevor er `torch.distributed.init_process_group()` aufruft. Auf einem
    TPU-Host ist jeder Worker-Prozess an ein einzelnes TPU-Gerät gebunden und
    kann nicht allein über alle Geräte verfügen. Bevor Sie die PjRT-Laufzeit
    initialisieren oder `torch.distributed.init_process_group()` aufrufen, muss
    die Umgebung Folgendes angeben: die physische Mesh-Geometrie des
    [TPU-Slice](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    (`TORCH_TPU_TOPOLOGY`), die lokalen Koordinatoradressen, mit denen das Mesh
    des Inter-Chip Interconnect (ICI) aufgebaut wird
    (`TORCH_TPU_SLICEBUILDER_ADDRESSES`), und die Gesamtzahl der beteiligten
    TPU-Geräte (`WORLD_SIZE`).
*   **Globale Auflösung von Replica Groups für Untergruppen:** In PyTorch
    erhält ein kollektiver Aufruf wie
    [`torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
    nur die ProcessGroup (Untergruppe), an der der aufrufende Rank beteiligt
    ist. Dagegen benötigt
    [`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce)
    eine vollständige Beschreibung aller beteiligten Untergruppen in der
    Standard-Prozessgruppe (`replica_groups`), auch derjenigen, zu denen der
    aktuelle Prozess nicht gehört:

    ```
    %0 = "stablehlo.all_reduce"(%arg0) <{
      replica_groups = dense<[[4, 0, 1, 3], [5, 2, 6, 7]]> : tensor<2x4xi64>
    }> ({ ... })
    ```

    Das Backend `tpu_dist` von TorchTPU tauscht beim Einrichten der Collectives
    automatisch die Mitgliedschaft in Untergruppen zwischen allen Ranks der
    Standard-Prozessgruppe aus. Daraus erstellt es die globalen
    `replica_groups` und das `device_assignment`, die XLA benötigt.
*   **SPMD-Kompilierung im Vergleich zu MPMD-Ausführung:** Der XLA-Compiler und
    der TPU-Kommunikationsstack sind für die Ausführung nach dem Prinzip Single
    Program, Multiple Data (SPMD) ausgelegt. Dabei kompiliert und führt jeder
    Rank ein identisches StableHLO-Modul mit übereinstimmenden kollektiven
    Operationen aus. PyTorch-Programme können sich jedoch zwischen den Ranks
    nach dem Prinzip Multiple Program, Multiple Data (MPMD) verhalten, etwa
    durch Rank-abhängige Verzweigungen (`if dist.get_rank() == 0:`) oder
    ungleichmäßiges Tensor-Sharding, wenn eine Dimension nicht gleichmäßig
    durch die Anzahl der Geräte teilbar ist. Wenn verschiedene Ranks
    unterschiedliche StableHLO-Module mit Collectives ausführen, können nicht
    übereinstimmende IDs von Synchronisierungs-Flags oder Puffer-Offsets
    zwischen den Ranks zu Deadlocks zur Laufzeit oder zu Speicherbeschädigung
    führen.

### Ausführung von Collectives und Fusionsmodi { #collective-execution-and-fusion-modes }

TorchTPU führt jedes Collective als Skinny Collective aus. Damit XLA Berechnung
und Kommunikation überlappen kann, können Sie einen oder beide von zwei
Mechanismen aktivieren, die Collectives mit der umgebenden Berechnung
fusionieren. Die beiden sind unabhängig und lassen sich kombinieren: Collectives
in einer Funktion mit `@spmd_safe` werden immer fusioniert. Wenn der
Compilation Handshake aktiviert ist, entscheidet er für die übrigen Collectives
in jedem Graphen von `torch.compile`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Modus | Konfiguration | Verhalten des Graphen | Am besten geeignet für |
| :---- | :---- | :---- | :---- |
| Skinny Collectives (Standardverhalten) | Immer aktiv für jedes Collective, das die folgenden Opt-ins nicht fusionieren | Kompiliert jede kollektive Operation in ein eigenes, eigenständiges („skinny“) StableHLO-Modul, indem Graph-Breaks um Collectives eingefügt werden. | Allgemeine MPMD-Sicherheit ohne weitere Konfiguration, allerdings kann XLA dann Berechnung und Kommunikation nicht überlappen. |
| Decorator `@spmd_safe` (Opt-in) | `@torch.tpu.distributed.spmd_safe` | Deaktiviert Graph-Breaks um Collectives in der dekorierten Funktion und fusioniert Berechnung und Collectives in einem einzigen StableHLO-Modul. | Funktionen oder Trainingsschritte, die bekanntermaßen SPMD sind (z. B. SimpleFSDP oder manuelle Tensor-Parallelität), um die Überlappung von Berechnung und Kommunikation in XLA zu maximieren. |
| Compilation Handshake (Opt-in) | `TORCH_TPU_SPMD_HANDSHAKE_STAGE=COMPILE_STAGE` oder `DISPATCH_STAGE` (Standard: OFF) | Prüft in Graphen von torch.compile, ob die kompilierten FX-Graphen auf allen Ranks übereinstimmen. Wenn ja, werden Collectives fusioniert, andernfalls wird auf Skinny Collectives zurückgegriffen. | Automatisches Aktivieren der Fusion von Collectives in `torch.compile` mit Schutz vor versehentlicher MPMD-Divergenz. |
<!-- mdformat on -->

#### `@torch.tpu.distributed.spmd_safe` verwenden { #using-torchtpudistributedspmd_safe }

Wenn Sie wissen, dass eine Funktion oder ein Modellschritt auf allen Ranks
strikt SPMD ist (kein Rank-abhängiger Kontrollfluss und keine ungleichmäßigen
Tensor-Shapes), dekorieren Sie sie mit `@torch.tpu.distributed.spmd_safe`. Unter
`torch.compile` (und im Eager-Modus `DEFER_AND_FUSE`) entfernt das die
Graph-Breaks um Collectives. So kann XLA asynchrone Kommunikation parallel zur
Matrixberechnung planen:

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

Verwenden Sie `@torch.tpu.distributed.spmd_safe` nur für Code, der auf jedem
Rank genau dieselbe Abfolge von Tensor- und kollektiven Operationen ausführt.
Bei MPMD-Code können dadurch auf den Ranks unterschiedliche StableHLO-Module
entstehen, die zu Deadlocks bei der TPU-Ausführung führen.

#### Compilation Handshake konfigurieren { #configuring-the-compilation-handshake }

Statt Funktionen manuell mit `@spmd_safe` zu annotieren, können Sie den
Compilation Handshake von TorchTPU über die Umgebungsvariable
`TORCH_TPU_SPMD_HANDSHAKE_STAGE` aktivieren. Wenn `torch.compile` beim Tracing
einen Graphen mit globalen Collectives erfasst, führt TorchTPU eine
Konsensprüfung über alle Ranks aus. Sie prüft, ob die kompilierten FX-Graphen
übereinstimmen, bevor die Collectives in den Graphen fusioniert werden:

*   `OFF` (Standard):** Deaktiviert den Handshake. Collectives verwenden
    standardmäßig Skinny Collectives, sofern sie nicht mit
    `@torch.tpu.distributed.spmd_safe` markiert sind.
*   `COMPILE_STAGE`: Prüft den Konsens der Graphen über die Ranks vor der
    PjRT-Kompilierung. Wenn alle Ranks identische Graphen erzeugen, entfallen
    die Graph-Breaks um Collectives. Empfohlen für Standard-Arbeitslasten mit
    `torch.compile`, bei denen alle Ranks gemeinsam kompilieren.
*   `DISPATCH_STAGE`: Kompiliert Collectives ohne Graph-Breaks und prüft den
    Konsens über die Ranks zum Zeitpunkt des Dispatch. Wenn die Ranks
    voneinander abweichen, wird mit aufgeteilten Collectives neu kompiliert.
    Dieser Modus ist für Arbeitslasten nötig, die Rank-spezifische Neukompilierungen
    auslösen können. Er hat einen höheren Overhead als `COMPILE_STAGE`.

Das folgende Beispiel zeigt eine Rank-spezifische Neukompilierung, die
`DISPATCH_STAGE` unterstützt. Mit `COMPILE_STAGE` käme es dagegen zu einem
Timeout, weil nur `divergent_rank` in Iteration 1 neu kompiliert:

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

## Verteilte Initialisierung { #distributed-initialization }

Zum Initialisieren einer verteilten Arbeitslast in TorchTPU verwenden Sie die
Standard-APIs von `torch.distributed`. Dazu kommt der Helper für die
Startumgebung von TorchTPU, der die Umgebungsvariablen für die TPU-Topologie und
den Slice-Builder befüllt:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Umgebungsvariable | Beschreibung | Beispiel |
| :---- | :---- | :---- |
| `TORCH_TPU_TOPOLOGY` | Physische Mesh-Geometrie des TPU-Slices. | `"2,2,1"`, `"2,2,1,2"`, `"4,4,4"` |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Kommagetrennte Adressen im Format `host:port` (eine pro Gerät), mit denen das ICI-Mesh aufgebaut wird. | `"localhost:8471,localhost:8472,..."` |
<!-- mdformat on -->

Sie müssen diese Variablen festlegen, bevor Sie die Worker starten. In
gängigen Setups (ein einzelner Host oder ein TPU-Slice auf GKE) kann TorchTPU
die richtigen Werte für Sie ermitteln. Führen Sie

```bash
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
```

in Ihrem Startskript vor torchrun aus oder rufen Sie
`torch.tpu.distributed.set_tpu_launch_env()` in Ihrem übergeordneten
Python-Launcher-Prozess auf, bevor Sie Worker starten. Beide Varianten lassen
bereits gesetzte Variablen unverändert. So können Sie einzelne Werte
überschreiben.

Im Folgenden sehen Sie den Initialisierungsablauf für eine verteilte
TorchTPU-Arbeitslast, gefolgt von einem vollständigen Beispiel für Worker und
Launcher:

![Ablauf der verteilten Initialisierung](./images/distributed.png)

### 1. Worker-Skript (`train.py`) { #1-worker-script-trainpy }

Initialisieren Sie in jedem Worker-Prozess das Backend `tpu_dist` und rufen Sie
das lokale TPU-Gerät über `torch.device("tpu")` ab:

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

### 2. Mit torchrun starten { #2-launching-with-torchrun }

Verwenden Sie das CLI-Modul `torch.tpu.distributed`, um die Hardwaretopologie
der TPU zu erkennen und die erforderlichen Startvariablen zu exportieren, bevor
Sie `torchrun` aufrufen:

```bash
# Detect and export TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES,
# WORLD_SIZE, NODE_RANK, and multi-host coordination variables:
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
# Launch 1 worker process per TPU device on the host:
torchrun --nproc_per_node=$WORLD_SIZE train.py
```

Wenn Sie Worker aus Python starten (z. B. mit
`torch.distributed.launcher.elastic_launch`), rufen Sie alternativ
`torch.tpu.distributed.set_tpu_launch_env()` im übergeordneten Prozess auf,
bevor Sie Worker starten. Die Standard-Prozessgruppe umfasst alle Geräte im
[TPU-Slice](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices).

## Optimale Zuordnung des Device Mesh { #optimal-device-mesh-mapping }

Um die Bandbreite von Collectives zu maximieren und die Latenz zu minimieren,
richten Sie die Parallelitätsdimensionen Ihres Modells an der physischen
Hierarchie der TPU-Verbindungen aus:

*   **Inter-Chip Interconnect (ICI)**: Schnelle Direktverbindungen zwischen
    TPU-Chips in einem 2D-/3D-Torus oder Twisted-Torus-Mesh innerhalb eines
    Slices (zehn bis mehrere hundert GB/s pro Verbindung).
*   **Data Center Network (DCN)**: Ethernet auf Host-Ebene, das separate
    TPU-Slices in Multi-Slice-Arbeitslasten verbindet (geringere Bandbreite und
    höhere Latenz als ICI).

Verwenden Sie `torch.tpu.distributed.topology_aware_mesh()`, um ein
Rank-Layout zu berechnen, das an den physischen ICI-Ringen ausgerichtet ist.
Ordnen Sie die Dimensionen Ihres logischen Mesh von der **geringsten
Kommunikationsintensität** (außen) bis zur **höchsten
Kommunikationsintensität** (innen) an:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Art der Parallelität | Typische Collectives | Kommunikationshäufigkeit | Empfohlene physische Platzierung |
| :---- | :---- | :---- | :---- |
| **Tensor Parallel (TP)** | `all_reduce`, `reduce_scatter`, `all_gather` | Jede Transformer-Schicht (Forward + Backward) | Schnellster lokaler ICI-Ring (gleicher Tray/Host, zusammenhängende Geräte mit 1 Hop) |
| **Context / Sequence Parallel (CP/SP)** | `all_to_all`, Ring Attention | Pro Attention-Schicht | Lokale ICI-Achsen (innerhalb des Slices) |
| **Expert Parallel (EP)** | `all_to_all`, `ragged_all_to_all` | Pro MoE-Router-Schicht | Lokale ICI-Achsen (innerhalb des Slices) |
| **Fully Sharded Data Parallel (FSDP)** | `all_gather`, `reduce_scatter` | Pro Modul / Transformer-Block | ICI-Achsen (innerhalb des Slices) |
| **Replicated Data Parallel (DP)** | `all_reduce` (nur Gradienten) | Einmal pro Trainingsschritt | Äußere ICI-Achse oder DCN (über Slices hinweg) |
<!-- mdformat on -->

### Beispiel: Topologiebewusstes 2D-DeviceMesh erstellen { #example-constructing-a-topology-aware-2d-devicemesh }

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

Bei Multi-Slice-Arbeitslasten (hybrid aus DCN und ICI) ist es optional,
`dcn_mesh_shape` zusätzlich zu `mesh_shape` zu übergeben. Das ist nur nötig, um
die Aufteilung explizit festzulegen. Standardmäßig erkennt
`torch.tpu.distributed.topology_aware_mesh()` die Slices automatisch. Die
äußerste nicht triviale Mesh-Dimension (typischerweise Datenparallelität) wird
über DCN auf die Slices verteilt, die inneren Dimensionen über ICI innerhalb
jedes Slices.

## Verteilte Pallas-Kernels { #distributed-pallas-kernels }

TorchTPU unterstützt eigene
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html)-Kernels, die direktes
Remote DMA (RDMA) zwischen Geräten über ICI ohne Beteiligung der Host-CPU
ausführen. Damit können Sie eigene Kommunikations-Kernels für Punkt-zu-Punkt-
oder Ring-Kommunikation schreiben, etwa Ring Attention, eigene
All-to-All-Shuffles, Verschiebungen für Pipeline-Parallelität oder
Halo-Austausch. Sie rufen diese als native eigene PyTorch-Ops
sowohl im Eager-Modus als auch mit `torch.compile` auf.

### Kommunizierenden Pallas-Kernel schreiben und registrieren { #writing-and-registering-a-communicating-pallas-kernel }

Die Einbindung eines verteilten Pallas-Kernels umfasst drei Schritte:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Phase | Wichtige Schritte und Operationen |
| :---- | :---- |
| **1. Pallas-Kernel-Body** | Fragen Sie logische Koordinaten mit `jax.lax.axis_index` und `jax.lax.axis_size` ab, erstellen Sie über `pltpu.make_async_remote_copy(..., device_id=(target,))` einen Deskriptor für die asynchrone Übertragung und steuern Sie den Abschluss mit `.start()` und `.wait()`. |
| **2. `pl.pallas_call` und `jax.shard_map`** | Weisen Sie DMA-Semaphoren (`pltpu.SemaphoreType.DMA`) in `scratch_shapes` zu und umschließen Sie den Kernel mit `jax.shard_map` über ein am Rank ausgerichtetes `jax.sharding.Mesh`, dessen Größe der World Size entspricht, und eine `PartitionSpec`. |
| **3. Registrierung in TorchTPU** | Registrieren Sie eine JAX-Wrapper-Funktion mit Typannotationen über `torch.tpu.pallas.jax_op` und übergeben Sie dabei `mesh`, `input_partition_specs` und `output_partition_specs`. |
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

### Am Rank ausgerichtete Meshes und Gerätezuweisung { #rank-aligned-meshes-and-device-assignment }

Bei vielen TPU-Architekturen entsprechen die Geräte-IDs nicht 1:1 den
zugewiesenen Ranks. Die Ranks werden so vergeben, dass Nachbarn benachbarte
Geräte sind (Rank 1 liegt neben Rank 0 und Rank 2). Die Geräte-IDs sind in
diesem Szenario bei v7x aber 0, 1 und 4. Um die Übersetzung zu vereinfachen,
bietet TorchTPU `torch.tpu.pallas.get_pallas_mesh`. Diese Funktion liefert ein
Pallas/JAX-Mesh, das genau den Rank-IDs entspricht, die im Rest der
Torch-Anwendung verwendet werden.

*   **Am Rank ausgerichteter Aufbau des Mesh**:
    `torch.tpu.pallas.get_pallas_mesh(axis_names=..., mesh_shape=...,
    process_group=...)` fragt die Zuordnung von Rank zu Gerät der aktiven
    PyTorch-`ProcessGroup` ab und erstellt ein `jax.sharding.Mesh` aus
    abstrakten, nur für die Kompilierung genutzten Geräten, ohne ein
    JAX-PjRT-Backend zu initialisieren.
*   **Mesh-Größe muss der World Size entsprechen**: Verteilte Pallas-Kernels
    benötigen ein Mesh, dessen Gesamtzahl an Geräten `dist.get_world_size()`
    entspricht. Für mehrdimensionale parallele Topologien (z. B. Daten- und
    Tensor-Parallelität) übergeben Sie eine `mesh_shape`, deren Produkt
    `world_size` ergibt, zum Beispiel
    `torch.tpu.pallas.get_pallas_mesh(axis_names=("dp", "tp"), mesh_shape=(2,
    4))`.
*   **Einheitliche Gerätezuweisung für Collectives und Kernels**: Wenn
    `mesh` an `torch.tpu.pallas.jax_op` übergeben wird, übernimmt TorchTPU die
    Zuordnung von Rank zu Gerät aus dem Mesh als XLA-`DeviceAssignment` für den
    Kernel. Diese gleicht es mit allen Standard-Collectives von
    `torch.distributed` ab, die in dasselbe kompilierte Modul fusioniert sind.
    So ist sichergestellt, dass `jax.lax.axis_index`, über das Mesh adressierte
    DMAs (`device_id={axis: target}`) und PyTorch-Collectives einheitlich nach
    der Rank-Reihenfolge von PyTorch geroutet werden.
