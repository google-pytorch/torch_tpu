# Compatibilidad de TorchTPU con la ejecución distribuida { #torchtpu-distributed-support }

## Compatibilidad con la ejecución distribuida { #distributed-support }

TorchTPU implementa la comunicación distribuida a nivel de ProcessGroup de
PyTorch (`c10d::Backend`) con el nombre de backend `tpu_dist`. Como intercepta
las operaciones colectivas en la capa de ProcessGroup, TorchTPU admite las APIs
estándar de `torch.distributed` tanto en modo eager como con `torch.compile`, y
funciona directamente con frameworks distribuidos de PyTorch, como DTensor,
DistributedDataParallel (DDP), FSDPv2 y
[SimpleFSDP](https://arxiv.org/abs/2411.00284).

Al igual que PyTorch en GPU, TorchTPU aplica un modelo de ejecución de un
proceso por dispositivo. Internamente, la comunicación de TPU a TPU no usa una
biblioteca de comunicación separada en el entorno de ejecución (como NCCL). En
cambio, las operaciones colectivas se reducen (lowering) directamente a
[operaciones colectivas de StableHLO](https://openxla.org/stablehlo/spec#collective_ops)
(como `stablehlo.all_reduce`, `stablehlo.all_gather`,
`stablehlo.reduce_scatter` y `stablehlo.all_to_all`), y XLA las compila en el
ejecutable de TPU.

### Diferencias clave de arquitectura respecto de GPU { #key-architectural-differences-from-gpu }

Aunque TorchTPU conserva la semántica estándar de `torch.distributed`,
programar TPUs mediante StableHLO y PjRT presenta tres diferencias clave en
comparación con los backends de GPU:

*   **Detección de la topología de hardware antes de la inicialización:** En un
    host con varias GPUs, cada proceso que inicia torchrun puede inspeccionar
    todas las GPUs conectadas, seleccionar su dispositivo asignado mediante
    `cuda:{local_rank}` y mover los modelos a ese dispositivo antes de llamar a
    `torch.distributed.init_process_group()`. En un host de TPU, cada proceso
    worker está vinculado a un solo dispositivo de TPU y no puede ser
    propietario del conjunto completo de dispositivos por sí solo. Antes de
    inicializar el entorno de ejecución de PjRT o de llamar a
    `torch.distributed.init_process_group()`, el entorno debe especificar la
    geometría física de la malla de la
    [porción de TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    (`TORCH_TPU_TOPOLOGY`), las direcciones de los coordinadores locales que se
    usan para construir la malla de Inter-Chip Interconnect (ICI)
    (`TORCH_TPU_SLICEBUILDER_ADDRESSES`) y la cantidad total de dispositivos de
    TPU que participan (`WORLD_SIZE`).
*   **Resolución global de grupos de réplicas para subgrupos:** En PyTorch, una
    llamada colectiva como
    [`torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
    solo recibe el ProcessGroup (subgrupo) en el que participa el rango que
    hace la llamada. En cambio,
    [`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce)
    requiere una descripción completa de todos los subgrupos que participan en
    el grupo de procesos predeterminado (`replica_groups`), incluso de aquellos
    a los que no pertenece el proceso actual:

    ```
    %0 = "stablehlo.all_reduce"(%arg0) <{
      replica_groups = dense<[[4, 0, 1, 3], [5, 2, 6, 7]]> : tensor<2x4xi64>
    }> ({ ... })
    ```

    El backend `tpu_dist` de TorchTPU intercambia automáticamente la membresía
    de los subgrupos entre todos los rangos del grupo de procesos
    predeterminado durante la configuración de la operación colectiva, y
    construye los valores globales `replica_groups` y `device_assignment` que
    requiere XLA.
*   **Compilación SPMD frente a ejecución MPMD:** El compilador XLA y la pila
    de comunicación de TPU están diseñados para la ejecución Single Program,
    Multiple Data (SPMD), en la que cada rango compila y ejecuta un módulo de
    StableHLO idéntico que contiene operaciones colectivas coincidentes. Sin
    embargo, los programas de PyTorch pueden mostrar un comportamiento Multiple
    Program, Multiple Data (MPMD) entre rangos, por ejemplo, debido a
    ramificaciones condicionadas por el rango (`if dist.get_rank() == 0:`) o a
    una fragmentación desigual de tensores cuando una dimensión no es divisible
    de manera uniforme por la cantidad de dispositivos. Si distintos rangos
    ejecutan distintos módulos de StableHLO que contienen operaciones
    colectivas, los IDs de indicadores de sincronización o los desplazamientos
    de búfer que no coinciden entre rangos pueden causar interbloqueos
    (deadlocks) en el entorno de ejecución o daños en la memoria.

### Modos de ejecución y fusión de operaciones colectivas { #collective-execution-and-fusion-modes }

TorchTPU ejecuta cada operación colectiva como una operación colectiva aislada
(skinny collective). Para que XLA superponga el cómputo y la comunicación,
puedes habilitar uno o ambos de los dos mecanismos que fusionan las operaciones
colectivas con el cómputo que las rodea. Los dos son independientes y se pueden
combinar: las operaciones colectivas dentro de una función `@spmd_safe` siempre
se fusionan y, cuando está habilitado, el protocolo de enlace de compilación
(Compilation Handshake) decide sobre las operaciones colectivas restantes de
cada grafo de `torch.compile`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Modo | Configuración | Comportamiento del grafo | Ideal para |
| :---- | :---- | :---- | :---- |
| Operaciones colectivas aisladas (comportamiento predeterminado) | Siempre activo para cualquier operación colectiva que no fusionen las opciones de abajo | Compila cada operación colectiva en su propio módulo de StableHLO independiente ("skinny") al insertar interrupciones de grafo alrededor de las operaciones colectivas. | Seguridad general para MPMD desde el principio, a costa de impedir que XLA superponga el cómputo y la comunicación. |
| Decorador `@spmd_safe` (opcional) | `@torch.tpu.distributed.spmd_safe` | Desactiva las interrupciones de grafo alrededor de las operaciones colectivas dentro de la función decorada, y fusiona el cómputo y las operaciones colectivas en un solo módulo de StableHLO. | Funciones o pasos de entrenamiento que se sabe que son SPMD (por ejemplo, SimpleFSDP o paralelismo de tensores manual) para maximizar la superposición de cómputo y comunicación en XLA. |
| Protocolo de enlace de compilación (opcional) | `TORCH_TPU_SPMD_HANDSHAKE_STAGE=COMPILE_STAGE` o `DISPATCH_STAGE` (predeterminado: OFF) | En los grafos de torch.compile, verifica que los grafos FX compilados coincidan en todos los rangos; si coinciden, fusiona las operaciones colectivas y, si no, recurre a operaciones colectivas aisladas. | Habilitar automáticamente la fusión de operaciones colectivas en `torch.compile` y, al mismo tiempo, protegerse contra divergencias MPMD accidentales. |
<!-- mdformat on -->

#### Usa `@torch.tpu.distributed.spmd_safe` { #using-torchtpudistributedspmd_safe }

Cuando sabes que una función o un paso del modelo es estrictamente SPMD en
todos los rangos (sin flujo de control que dependa del rango ni formas de
tensor desiguales), decóralo con `@torch.tpu.distributed.spmd_safe`. Con
`torch.compile` (y con el modo eager `DEFER_AND_FUSE`), esto quita las
interrupciones de grafo alrededor de las operaciones colectivas para que XLA
pueda programar la comunicación asíncrona junto con el cómputo de matrices:

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

Aplica `@torch.tpu.distributed.spmd_safe` solo al código que ejecuta
exactamente la misma secuencia de operaciones de tensores y operaciones
colectivas en cada rango. Si lo aplicas a código MPMD, se pueden producir
módulos de StableHLO distintos entre rangos y causar interbloqueos en la
ejecución de TPU.

#### Configura el protocolo de enlace de compilación { #configuring-the-compilation-handshake }

En lugar de anotar funciones manualmente con `@spmd_safe`, puedes habilitar el
protocolo de enlace de compilación de TorchTPU mediante la variable de entorno
`TORCH_TPU_SPMD_HANDSHAKE_STAGE`. Cuando `torch.compile` traza un grafo que
contiene operaciones colectivas globales, TorchTPU ejecuta una verificación de
consenso en todos los rangos para comprobar que sus grafos FX compilados
coincidan antes de fusionar las operaciones colectivas en el grafo:

*   `OFF` (predeterminado): Desactiva el protocolo de enlace; las operaciones
    colectivas usan las operaciones colectivas aisladas predeterminadas, a
    menos que estén marcadas con `@torch.tpu.distributed.spmd_safe`.
*   `COMPILE_STAGE`: Verifica el consenso del grafo entre los rangos antes de
    la compilación de PjRT. Si todos los rangos producen grafos idénticos, se
    omiten las interrupciones de grafo alrededor de las operaciones colectivas.
    Se recomienda para cargas de trabajo estándar de `torch.compile` en las que
    todos los rangos compilan juntos.
*   `DISPATCH_STAGE`: Compila las operaciones colectivas sin interrupciones de
    grafo y verifica el consenso entre los rangos en el momento del despacho;
    si los rangos divergen, vuelve a compilar con operaciones colectivas
    separadas. Este modo es necesario para cargas de trabajo que pueden activar
    recompilaciones específicas de un rango; tiene una sobrecarga mayor que
    `COMPILE_STAGE`.

A continuación, se muestra un ejemplo de recompilación específica de un rango
que admite `DISPATCH_STAGE`, mientras que con `COMPILE_STAGE` se agotaría el
tiempo de espera porque solo `divergent_rank` vuelve a compilar en la
iteración 1:

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

## Inicialización distribuida { #distributed-initialization }

Para inicializar una carga de trabajo distribuida en TorchTPU, se usan las APIs
estándar de `torch.distributed`, junto con el asistente del entorno de
lanzamiento de TorchTPU, que completa las variables de entorno de la topología
de TPU y del constructor de porciones (slice builder):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable de entorno | Descripción | Ejemplo |
| :---- | :---- | :---- |
| `TORCH_TPU_TOPOLOGY` | Geometría física de la malla de la porción de TPU. | `"2,2,1"`, `"2,2,1,2"`, `"4,4,4"` |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Direcciones `host:port` separadas por comas (una por dispositivo) que se usan para construir la malla de ICI. | `"localhost:8471,localhost:8472,..."` |
<!-- mdformat on -->

Debes configurar estas variables antes de iniciar los workers. En las
configuraciones comunes (un solo host o una porción de TPU de GKE), TorchTPU
puede determinar los valores correctos por ti: ejecuta

```bash
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
```

en tu secuencia de comandos de lanzamiento antes de torchrun, o llama a
`torch.tpu.distributed.set_tpu_launch_env()` en tu proceso principal de
lanzamiento de Python antes de generar los workers. Ambas opciones dejan sin
cambios cualquier variable que ya hayas configurado, por lo que puedes anular
valores individuales.

A continuación, se muestra el flujo de inicialización de una carga de trabajo
distribuida de TorchTPU, seguido de un ejemplo completo de worker y de
lanzador:

![Flujo de inicialización distribuida](./images/distributed.png)

### 1. Secuencia de comandos del worker (`train.py`) { #1-worker-script-trainpy }

Dentro de cada proceso worker, inicializa el backend `tpu_dist` y obtén el
dispositivo de TPU local mediante `torch.device("tpu")`:

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

### 2. Lanzamiento con torchrun { #2-launching-with-torchrun }

Usa el módulo de CLI `torch.tpu.distributed` para detectar la topología de
hardware de TPU y exportar las variables de lanzamiento necesarias antes de
invocar `torchrun`:

```bash
# Detect and export TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES,
# WORLD_SIZE, NODE_RANK, and multi-host coordination variables:
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
# Launch 1 worker process per TPU device on the host:
torchrun --nproc_per_node=$WORLD_SIZE train.py
```

Como alternativa, si inicias los workers desde Python (por ejemplo, con
`torch.distributed.launcher.elastic_launch`), llama a
`torch.tpu.distributed.set_tpu_launch_env()` en el proceso principal antes de
generar los workers. Ten en cuenta que el grupo de procesos predeterminado
abarca todos los dispositivos de la
[porción de TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices).

## Asignación óptima de la malla de dispositivos { #optimal-device-mesh-mapping }

Para maximizar el ancho de banda de las operaciones colectivas y minimizar la
latencia, alinea las dimensiones de paralelismo de tu modelo con la jerarquía
física de interconexión de TPU:

*   **Inter-Chip Interconnect (ICI)**: Enlaces directos de alta velocidad que
    conectan los chips de TPU en una malla de toro 2D/3D o de toro torcido
    (twisted torus) dentro de una porción (de decenas a cientos de GB/s por
    enlace).
*   **Data Center Network (DCN)**: Ethernet a nivel de host que conecta
    porciones de TPU separadas en cargas de trabajo de varias porciones (menor
    ancho de banda y mayor latencia que ICI).

Usa `torch.tpu.distributed.topology_aware_mesh()` para calcular una
distribución de rangos alineada con los anillos físicos de ICI. Ordena las
dimensiones de tu malla lógica de **menor intensidad de comunicación** (la más
externa) a **mayor intensidad de comunicación** (la más interna):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Tipo de paralelismo | Operaciones colectivas típicas | Frecuencia de comunicación | Ubicación física recomendada |
| :---- | :---- | :---- | :---- |
| **Paralelismo de tensores (TP)** | `all_reduce`, `reduce_scatter`, `all_gather` | En cada capa de transformer (pase hacia adelante + pase hacia atrás) | El anillo de ICI local más rápido (misma bandeja/host, dispositivos contiguos a 1 salto) |
| **Paralelismo de contexto/secuencia (CP/SP)** | `all_to_all`, Ring Attention | Por capa de atención | Ejes de ICI locales (dentro de la porción) |
| **Paralelismo de expertos (EP)** | `all_to_all`, `ragged_all_to_all` | Por capa de router de MoE | Ejes de ICI locales (dentro de la porción) |
| **Paralelismo de datos completamente fragmentado (FSDP)** | `all_gather`, `reduce_scatter` | Por módulo / bloque de transformer | Ejes de ICI (dentro de la porción) |
| **Paralelismo de datos replicado (DP)** | `all_reduce` (solo gradientes) | Una vez por paso de entrenamiento | Eje de ICI externo o DCN (entre porciones) |
<!-- mdformat on -->

### Ejemplo: Construye una DeviceMesh 2D que tenga en cuenta la topología { #example-constructing-a-topology-aware-2d-devicemesh }

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

En las cargas de trabajo de varias porciones (híbridas DCN + ICI), pasar
`dcn_mesh_shape` junto con `mesh_shape` es opcional y solo se necesita para
elegir la división de forma explícita: de forma predeterminada,
`torch.tpu.distributed.topology_aware_mesh()` detecta las porciones
automáticamente y asigna la dimensión de malla no trivial más externa
(normalmente, la de paralelismo de datos) entre porciones a través de DCN, y las
dimensiones internas dentro de cada porción a través de ICI.

## Kernels distribuidos de Pallas { #distributed-pallas-kernels }

TorchTPU admite kernels personalizados de
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) que realizan acceso
directo remoto a memoria (RDMA) entre dispositivos a través de ICI, sin
intervención de la CPU del host. Esto te permite crear kernels personalizados de
comunicación punto a punto o en anillo (como Ring Attention, reorganizaciones
All-to-All personalizadas, desplazamientos de paralelismo de canalización o
intercambios de halo) e invocarlos como operaciones personalizadas nativas de
PyTorch tanto en modo eager como con `torch.compile`.

### Escribe y registra un kernel de Pallas que se comunica { #writing-and-registering-a-communicating-pallas-kernel }

Integrar un kernel distribuido de Pallas implica tres pasos:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Etapa | Pasos y operaciones clave |
| :---- | :---- |
| **1. Cuerpo del kernel de Pallas** | Consulta las coordenadas lógicas con `jax.lax.axis_index` y `jax.lax.axis_size`, construye un descriptor de transferencia asíncrona mediante `pltpu.make_async_remote_copy(..., device_id=(target,))` y administra la finalización con `.start()` y `.wait()`. |
| **2. `pl.pallas_call` y `jax.shard_map`** | Asigna semáforos de DMA (`pltpu.SemaphoreType.DMA`) en `scratch_shapes` y encapsula el kernel con `jax.shard_map` sobre una `jax.sharding.Mesh` alineada con los rangos y del tamaño del mundo (world size), y una `PartitionSpec`. |
| **3. Registro en TorchTPU** | Registra una función wrapper de JAX con anotaciones de tipo mediante `torch.tpu.pallas.jax_op`, y pasa `mesh`, `input_partition_specs` y `output_partition_specs`. |
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

### Mallas alineadas con los rangos y asignación de dispositivos { #rank-aligned-meshes-and-device-assignment }

En muchas arquitecturas de TPU, los IDs de dispositivo y los rangos asignados no
tienen una correspondencia de 1 a 1: los rangos se asignan de modo que los
vecinos sean los dispositivos adyacentes (el rango 1 está junto al rango 0 y al
rango 2), pero en este caso los IDs de dispositivo son 0, 1 y 4 para v7x. Para
facilitar la traducción, TorchTPU proporciona
`torch.tpu.pallas.get_pallas_mesh`, que ofrece una malla de Pallas/JAX que
coincide exactamente con el ID de rango que se usa en el resto de la aplicación
de Torch.

*   **Construcción de mallas alineadas con los rangos**:
    `torch.tpu.pallas.get_pallas_mesh(axis_names=..., mesh_shape=...,
    process_group=...)` consulta la asignación de rangos a dispositivos del
    `ProcessGroup` activo de PyTorch y construye una `jax.sharding.Mesh` de
    dispositivos abstractos solo para compilación, sin inicializar un backend
    de PjRT de JAX.
*   **Requisito de malla del tamaño del mundo**: Los kernels distribuidos de
    Pallas requieren una malla cuya cantidad total de dispositivos sea igual a
    `dist.get_world_size()`. Para topologías paralelas multidimensionales (como
    paralelismo de datos + paralelismo de tensores), pasa un `mesh_shape` cuyo
    producto sea igual a `world_size`; por ejemplo,
    `torch.tpu.pallas.get_pallas_mesh(axis_names=("dp", "tp"), mesh_shape=(2,
    4))`.
*   **Asignación unificada de dispositivos para operaciones colectivas y
    kernels**: Cuando se proporciona `mesh` a `torch.tpu.pallas.jax_op`,
    TorchTPU extrae la asignación de rangos a dispositivos de la malla como el
    `DeviceAssignment` de XLA para el kernel y la concilia con cualquier
    operación colectiva estándar de `torch.distributed` fusionada en el mismo
    módulo compilado. Esto garantiza que `jax.lax.axis_index`, las DMAs
    dirigidas por malla (`device_id={axis: target}`) y las operaciones
    colectivas de PyTorch se enruten de forma coherente según el orden de
    rangos de PyTorch.
