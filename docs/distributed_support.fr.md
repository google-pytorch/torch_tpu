# Prise en charge du calcul distribué dans TorchTPU { #torchtpu-distributed-support }

## Prise en charge du calcul distribué { #distributed-support }

TorchTPU implémente la communication distribuée au niveau du ProcessGroup de
PyTorch (`c10d::Backend`), sous le nom de backend `tpu_dist`. En interceptant
les collectives au niveau de la couche ProcessGroup, TorchTPU prend en charge
les API `torch.distributed` standards en mode eager comme avec `torch.compile`,
et fonctionne directement avec les frameworks distribués de PyTorch, notamment
DTensor, DistributedDataParallel (DDP), FSDPv2 et
[SimpleFSDP](https://arxiv.org/abs/2411.00284).

Comme PyTorch sur GPU, TorchTPU impose un modèle d'exécution à un processus par
appareil. En interne, la communication de TPU à TPU n'utilise pas de
bibliothèque de communication distincte au runtime (comme NCCL). Les opérations
collectives sont directement abaissées en
[opérations collectives StableHLO](https://openxla.org/stablehlo/spec#collective_ops)
(comme `stablehlo.all_reduce`, `stablehlo.all_gather`,
`stablehlo.reduce_scatter` et `stablehlo.all_to_all`), puis compilées par XLA
dans l'exécutable TPU.

### Principales différences d'architecture par rapport au GPU { #key-architectural-differences-from-gpu }

TorchTPU conserve la sémantique standard de `torch.distributed`, mais la
programmation des TPU via StableHLO et PjRT introduit trois différences
majeures par rapport aux backends GPU :

*   **Découverte de la topologie matérielle avant l'initialisation :** sur un
    hôte multi-GPU, chaque processus lancé par torchrun peut inspecter tous les
    GPU connectés, sélectionner l'appareil qui lui est attribué via
    `cuda:{local_rank}` et y déplacer les modèles avant d'appeler
    `torch.distributed.init_process_group()`. Sur un hôte TPU, chaque processus
    worker est lié à un seul appareil TPU et ne peut pas posséder seul
    l'ensemble des appareils. Avant d'initialiser le runtime PjRT ou d'appeler
    `torch.distributed.init_process_group()`, l'environnement doit spécifier
    la géométrie physique du maillage de la
    [tranche (slice) TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    (`TORCH_TPU_TOPOLOGY`), les adresses des coordinateurs locaux utilisées
    pour construire le maillage Inter-Chip Interconnect (ICI)
    (`TORCH_TPU_SLICEBUILDER_ADDRESSES`) et le nombre total d'appareils TPU
    participants (`WORLD_SIZE`).
*   **Résolution globale des groupes de réplicas pour les sous-groupes :** dans
    PyTorch, un appel collectif comme
    [`torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
    ne reçoit que le ProcessGroup (sous-groupe) auquel participe le rang
    appelant. À l'inverse,
    [`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce)
    exige une description complète de tous les sous-groupes participants dans
    le groupe de processus par défaut (`replica_groups`), y compris ceux
    auxquels le processus actuel n'appartient pas :

    ```
    %0 = "stablehlo.all_reduce"(%arg0) <{
      replica_groups = dense<[[4, 0, 1, 3], [5, 2, 6, 7]]> : tensor<2x4xi64>
    }> ({ ... })
    ```

    Le backend `tpu_dist` de TorchTPU échange automatiquement l'appartenance
    aux sous-groupes entre tous les rangs du groupe de processus par défaut
    lors de la configuration des collectives, puis construit les
    `replica_groups` et le `device_assignment` globaux qu'exige XLA.
*   **Compilation SPMD et exécution MPMD :** le compilateur XLA et la pile de
    communication TPU sont conçus pour une exécution SPMD (Single Program,
    Multiple Data), où chaque rang compile et exécute un module StableHLO
    identique contenant les mêmes opérations collectives. Cependant, les
    programmes PyTorch peuvent présenter un comportement MPMD (Multiple
    Program, Multiple Data) d'un rang à l'autre, par exemple à cause de
    branches conditionnelles selon le rang (`if dist.get_rank() == 0:`) ou d'un
    sharding inégal des tensors lorsqu'une dimension n'est pas divisible par le
    nombre d'appareils. Si différents rangs exécutent différents modules
    StableHLO contenant des collectives, des ID de flags de synchronisation ou
    des offsets de buffers incohérents entre les rangs peuvent provoquer des
    interblocages (deadlocks) ou une corruption de la mémoire au runtime.

### Modes d'exécution et de fusion des collectives { #collective-execution-and-fusion-modes }

TorchTPU exécute chaque collective sous forme de collective « skinny ». Pour
permettre à XLA de chevaucher calcul et communication, vous pouvez activer
l'un ou les deux mécanismes qui fusionnent les collectives avec le calcul
environnant. Ils sont indépendants et peuvent être combinés : les collectives
d'une fonction `@spmd_safe` sont toujours fusionnées et, s'il est activé, le
handshake de compilation décide pour les autres collectives de chaque graphe
`torch.compile`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Mode | Configuration | Comportement du graphe | Idéal pour |
| :---- | :---- | :---- | :---- |
| Collectives skinny (comportement par défaut) | Toujours actif pour toute collective que les options ci-dessous ne fusionnent pas | Compile chaque opération collective dans son propre module StableHLO autonome (« skinny ») en insérant des ruptures de graphe (graph breaks) autour des collectives. | Sécurité MPMD générale par défaut, au prix d'empêcher XLA de chevaucher calcul et communication. |
| Décorateur `@spmd_safe` (facultatif) | `@torch.tpu.distributed.spmd_safe` | Désactive les ruptures de graphe autour des collectives dans la fonction décorée, et fusionne calcul et collectives dans un seul module StableHLO. | Fonctions ou étapes d'entraînement dont on sait qu'elles sont SPMD (par exemple, SimpleFSDP ou parallélisme de tensors manuel), pour maximiser le chevauchement calcul-communication par XLA. |
| Handshake de compilation (facultatif) | `TORCH_TPU_SPMD_HANDSHAKE_STAGE=COMPILE_STAGE` ou `DISPATCH_STAGE` (par défaut : OFF) | Dans les graphes torch.compile, vérifie que les graphes FX compilés concordent sur tous les rangs ; fusionne les collectives si c'est le cas et revient aux collectives skinny sinon. | Activer automatiquement la fusion des collectives dans `torch.compile` tout en se protégeant contre une divergence MPMD accidentelle. |
<!-- mdformat on -->

#### Utiliser `@torch.tpu.distributed.spmd_safe` { #using-torchtpudistributedspmd_safe }

Si vous savez qu'une fonction ou une étape de modèle est strictement SPMD sur
tous les rangs (pas de flux de contrôle dépendant du rang ni de formes de
tensors inégales), décorez-la avec `@torch.tpu.distributed.spmd_safe`. Sous
`torch.compile` (et en mode eager `DEFER_AND_FUSE`), cela supprime les ruptures
de graphe autour des collectives : XLA peut alors planifier la communication
asynchrone en parallèle des calculs matriciels :

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

N'appliquez `@torch.tpu.distributed.spmd_safe` qu'à du code qui exécute
exactement la même séquence d'opérations sur les tensors et d'opérations
collectives sur chaque rang. L'appliquer à du code MPMD peut produire des
modules StableHLO différents selon les rangs et provoquer des interblocages
lors de l'exécution sur TPU.

#### Configurer le handshake de compilation { #configuring-the-compilation-handshake }

Au lieu d'annoter manuellement les fonctions avec `@spmd_safe`, vous pouvez
activer le handshake de compilation de TorchTPU via la variable
d'environnement `TORCH_TPU_SPMD_HANDSHAKE_STAGE`. Lorsque `torch.compile` trace
un graphe contenant des collectives globales, TorchTPU effectue une
vérification de consensus entre tous les rangs pour confirmer que leurs
graphes FX compilés concordent avant de fusionner les collectives dans le
graphe :

*   `OFF` (par défaut) :** désactive le handshake ; les collectives utilisent
    les collectives skinny par défaut, sauf si elles sont marquées
    `@torch.tpu.distributed.spmd_safe`.
*   `COMPILE_STAGE` : vérifie le consensus des graphes entre les rangs avant la
    compilation PjRT. Si tous les rangs produisent des graphes identiques, les
    ruptures de graphe autour des collectives sont supprimées. Recommandé pour
    les charges de travail `torch.compile` standards où tous les rangs
    compilent ensemble.
*   `DISPATCH_STAGE` : compile les collectives sans ruptures de graphe et
    vérifie le consensus entre les rangs au moment du dispatch, puis recompile
    avec des collectives séparées si les rangs divergent. Ce mode est
    nécessaire pour les charges de travail qui peuvent déclencher des
    recompilations propres à un rang ; son surcoût est plus élevé que celui de
    `COMPILE_STAGE`.

Voici un exemple de recompilation propre à un rang prise en charge par
`DISPATCH_STAGE`. Avec `COMPILE_STAGE`, le délai d'attente expirerait, car
seul `divergent_rank` recompile à l'itération 1 :

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

## Initialisation distribuée { #distributed-initialization }

L'initialisation d'une charge de travail distribuée dans TorchTPU utilise les
API `torch.distributed` standards, associées à l'outil d'environnement de
lancement de TorchTPU qui renseigne les variables d'environnement de topologie
TPU et de slice builder :

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable d'environnement | Description | Exemple |
| :---- | :---- | :---- |
| `TORCH_TPU_TOPOLOGY` | Géométrie physique du maillage de la tranche TPU. | `"2,2,1"`, `"2,2,1,2"`, `"4,4,4"` |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Adresses `host:port` séparées par des virgules (une par appareil), utilisées pour construire le maillage ICI. | `"localhost:8471,localhost:8472,..."` |
<!-- mdformat on -->

Vous devez définir ces variables avant de démarrer les workers. Dans les
configurations courantes (un seul hôte ou une tranche TPU GKE), TorchTPU peut
déterminer les bonnes valeurs pour vous : exécutez

```bash
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
```

dans votre script de lancement avant torchrun, ou appelez
`torch.tpu.distributed.set_tpu_launch_env()` dans votre processus de lancement
Python parent avant de démarrer les workers. Les deux méthodes laissent
inchangées les variables que vous avez déjà définies : vous pouvez donc
remplacer des valeurs individuellement.

Voici le flux d'initialisation d'une charge de travail TorchTPU distribuée,
suivi d'un exemple complet de worker et de lanceur :

![Flux d'initialisation distribuée](./images/distributed.png)

### 1. Script worker (`train.py`) { #1-worker-script-trainpy }

Dans chaque processus worker, initialisez le backend `tpu_dist` et obtenez
l'appareil TPU local via `torch.device("tpu")` :

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

### 2. Lancement avec torchrun { #2-launching-with-torchrun }

Utilisez le module CLI `torch.tpu.distributed` pour détecter la topologie
matérielle des TPU et exporter les variables de lancement requises avant
d'appeler `torchrun` :

```bash
# Detect and export TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES,
# WORLD_SIZE, NODE_RANK, and multi-host coordination variables:
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
# Launch 1 worker process per TPU device on the host:
torchrun --nproc_per_node=$WORLD_SIZE train.py
```

Si vous lancez plutôt les workers depuis Python (par exemple, avec
`torch.distributed.launcher.elastic_launch`), appelez
`torch.tpu.distributed.set_tpu_launch_env()` dans le processus parent avant de
démarrer les workers. Notez que le groupe de processus par défaut englobe tous
les appareils de la
[tranche TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices).

## Mappage optimal du maillage d'appareils { #optimal-device-mesh-mapping }

Pour maximiser la bande passante des collectives et réduire la latence, alignez
les dimensions de parallélisme de votre modèle sur la hiérarchie physique de
l'interconnexion TPU :

*   **Inter-Chip Interconnect (ICI)** : liaisons directes à haut débit qui
    relient les puces TPU d'une tranche dans un maillage en tore 2D/3D ou en
    tore torsadé (de quelques dizaines à quelques centaines de Go/s par
    liaison).
*   **Data Center Network (DCN)** : réseau Ethernet au niveau des hôtes qui
    relie des tranches TPU distinctes dans les charges de travail
    multitranches (bande passante plus faible et latence plus élevée que
    l'ICI).

Utilisez `torch.tpu.distributed.topology_aware_mesh()` pour calculer une
disposition des rangs alignée sur les anneaux ICI physiques. Ordonnez les
dimensions de votre maillage logique de l'**intensité de communication la plus
faible** (la plus externe) à l'**intensité de communication la plus élevée**
(la plus interne) :

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Type de parallélisme | Collectives typiques | Fréquence de communication | Placement physique recommandé |
| :---- | :---- | :---- | :---- |
| **Parallélisme de tensors (TP)** | `all_reduce`, `reduce_scatter`, `all_gather` | Chaque couche transformer (passe avant + passe arrière) | Anneau ICI local le plus rapide (même plateau/hôte, appareils contigus à 1 saut) |
| **Parallélisme de contexte / de séquence (CP/SP)** | `all_to_all`, Ring Attention | Par couche d'attention | Axes ICI locaux (dans la tranche) |
| **Parallélisme d'experts (EP)** | `all_to_all`, `ragged_all_to_all` | Par couche de routage MoE | Axes ICI locaux (dans la tranche) |
| **Parallélisme de données entièrement shardé (FSDP)** | `all_gather`, `reduce_scatter` | Par module / bloc transformer | Axes ICI (dans la tranche) |
| **Parallélisme de données répliqué (DP)** | `all_reduce` (gradients uniquement) | Une fois par étape d'entraînement | Axe ICI externe ou DCN (entre les tranches) |
<!-- mdformat on -->

### Exemple : construire un DeviceMesh 2D adapté à la topologie { #example-constructing-a-topology-aware-2d-devicemesh }

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

Pour les charges de travail multitranches (hybrides DCN + ICI), transmettre
`dcn_mesh_shape` avec `mesh_shape` est facultatif et ne sert qu'à choisir
explicitement la répartition. Par défaut,
`torch.tpu.distributed.topology_aware_mesh()` détecte automatiquement les
tranches et mappe la dimension non triviale la plus externe du maillage
(généralement le parallélisme de données) entre les tranches via le DCN, et les
dimensions internes dans chaque tranche via l'ICI.

## Kernels Pallas distribués { #distributed-pallas-kernels }

TorchTPU prend en charge les kernels
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) personnalisés qui
effectuent des transferts Remote DMA (RDMA) directs entre appareils via l'ICI,
sans intervention du CPU hôte. Vous pouvez ainsi écrire des kernels de
communication point à point ou en anneau personnalisés (par exemple Ring
Attention, des brassages All-to-All personnalisés, des décalages de
parallélisme de pipeline ou des échanges de halos) et les appeler comme des
opérations personnalisées PyTorch natives, en mode eager comme avec
`torch.compile`.

### Écrire et enregistrer un kernel Pallas communicant { #writing-and-registering-a-communicating-pallas-kernel }

L'intégration d'un kernel Pallas distribué comporte trois étapes :

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Étape | Étapes et opérations clés |
| :---- | :---- |
| **1. Corps du kernel Pallas** | Interrogez les coordonnées logiques avec `jax.lax.axis_index` et `jax.lax.axis_size`, construisez un descripteur de transfert asynchrone via `pltpu.make_async_remote_copy(..., device_id=(target,))` et gérez la fin du transfert avec `.start()` et `.wait()`. |
| **2. `pl.pallas_call` et `jax.shard_map`** | Allouez des sémaphores DMA (`pltpu.SemaphoreType.DMA`) dans `scratch_shapes` et encapsulez le kernel avec `jax.shard_map` sur un `jax.sharding.Mesh` aligné sur les rangs et de taille égale au world size, avec une `PartitionSpec`. |
| **3. Enregistrement dans TorchTPU** | Enregistrez une fonction wrapper JAX annotée avec des types à l'aide de `torch.tpu.pallas.jax_op`, en transmettant `mesh`, `input_partition_specs` et `output_partition_specs`. |
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

### Maillages alignés sur les rangs et attribution des appareils { #rank-aligned-meshes-and-device-assignment }

Dans de nombreuses architectures TPU, les ID d'appareils et les rangs attribués
ne correspondent pas un à un : les rangs sont attribués de sorte que les
voisins soient les appareils adjacents (le rang 1 est à côté des rangs 0 et 2),
mais sur v7x, les ID d'appareils sont alors 0, 1 et 4. Pour simplifier cette
conversion, TorchTPU fournit `torch.tpu.pallas.get_pallas_mesh`, qui renvoie un
maillage Pallas/JAX correspondant exactement à l'ID de rang utilisé dans le
reste de l'application Torch.

*   **Construction d'un maillage aligné sur les rangs** :
    `torch.tpu.pallas.get_pallas_mesh(axis_names=..., mesh_shape=...,
    process_group=...)` interroge le mappage rang-appareil du `ProcessGroup`
    PyTorch actif et construit un `jax.sharding.Mesh` d'appareils abstraits,
    réservés à la compilation, sans initialiser de backend JAX PjRT.
*   **Maillage de taille égale au world size** : les kernels Pallas
    distribués nécessitent un maillage dont le nombre total d'appareils est égal
    à `dist.get_world_size()`. Pour les topologies parallèles
    multidimensionnelles (comme le parallélisme de données + de tensors),
    transmettez un `mesh_shape` dont le produit est égal à `world_size`, par
    exemple `torch.tpu.pallas.get_pallas_mesh(axis_names=("dp", "tp"), mesh_shape=(2,
    4))`.
*   **Attribution unifiée des appareils pour les collectives et les kernels** :
    lorsque `mesh` est transmis à `torch.tpu.pallas.jax_op`, TorchTPU extrait
    le mappage rang-appareil du maillage comme `DeviceAssignment` XLA du kernel
    et le concilie avec les collectives `torch.distributed` standards fusionnées
    dans le même module compilé. Cela garantit que `jax.lax.axis_index`, les
    DMA adressés par maillage (`device_id={axis: target}`) et les collectives
    PyTorch suivent tous un routage cohérent, conforme à l'ordre des rangs de
    PyTorch.
