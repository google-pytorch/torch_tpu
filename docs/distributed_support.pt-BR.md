# Suporte a execução distribuída no TorchTPU { #torchtpu-distributed-support }

## Suporte a execução distribuída { #distributed-support }

O TorchTPU implementa a comunicação distribuída no nível do ProcessGroup do
PyTorch (`c10d::Backend`) com o nome de backend `tpu_dist`. Ao interceptar as
coletivas na camada do ProcessGroup, o TorchTPU oferece suporte às APIs padrão
de `torch.distributed` tanto no modo eager quanto com `torch.compile` e
funciona diretamente com frameworks distribuídos do PyTorch, incluindo DTensor,
DistributedDataParallel (DDP), FSDPv2 e
[SimpleFSDP](https://arxiv.org/abs/2411.00284).

Assim como o PyTorch em GPU, o TorchTPU adota um modelo de execução com um
processo por dispositivo. Internamente, a comunicação entre TPUs não usa uma
biblioteca de comunicação de runtime separada (como a NCCL). Em vez disso, as
operações coletivas são convertidas (lowering) diretamente em
[operações coletivas do StableHLO](https://openxla.org/stablehlo/spec#collective_ops)
(como `stablehlo.all_reduce`, `stablehlo.all_gather`,
`stablehlo.reduce_scatter` e `stablehlo.all_to_all`) e compiladas pelo XLA no
executável da TPU.

### Principais diferenças de arquitetura em relação à GPU { #key-architectural-differences-from-gpu }

Embora o TorchTPU preserve a semântica padrão de `torch.distributed`, programar
TPUs via StableHLO e PjRT introduz três diferenças importantes em comparação
com os backends de GPU:

*   **Descoberta da topologia de hardware antes da inicialização:** em um host
    com várias GPUs, cada processo iniciado pelo torchrun pode inspecionar
    todas as GPUs conectadas, selecionar o dispositivo atribuído a ele via
    `cuda:{local_rank}` e mover os modelos para esse dispositivo antes de
    chamar `torch.distributed.init_process_group()`. Em um host de TPU, cada
    processo worker fica vinculado a um único dispositivo TPU e não pode
    controlar sozinho o conjunto completo de dispositivos. Antes de inicializar
    o runtime PjRT ou chamar `torch.distributed.init_process_group()`, o
    ambiente precisa especificar a geometria física da malha da
    [fração (slice) de TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    (`TORCH_TPU_TOPOLOGY`), os endereços dos coordenadores locais usados para
    construir a malha de Inter-Chip Interconnect (ICI)
    (`TORCH_TPU_SLICEBUILDER_ADDRESSES`) e o número total de dispositivos TPU
    participantes (`WORLD_SIZE`).
*   **Resolução global de grupos de réplicas para subgrupos:** no PyTorch, uma
    chamada coletiva como
    [`torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
    recebe apenas o ProcessGroup (subgrupo) do qual o rank que faz a chamada
    participa. Já
    [`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce)
    exige uma descrição completa de todos os subgrupos participantes em todo o
    grupo de processos padrão (`replica_groups`), inclusive daqueles aos quais
    o processo atual não pertence:

    ```
    %0 = "stablehlo.all_reduce"(%arg0) <{
      replica_groups = dense<[[4, 0, 1, 3], [5, 2, 6, 7]]> : tensor<2x4xi64>
    }> ({ ... })
    ```

    O backend `tpu_dist` do TorchTPU troca automaticamente as informações de
    participação nos subgrupos entre todos os ranks do grupo de processos
    padrão durante a configuração da coletiva e constrói os `replica_groups` e
    o `device_assignment` globais exigidos pelo XLA.
*   **Compilação SPMD versus execução MPMD:** o compilador XLA e a pilha de
    comunicação da TPU foram projetados para a execução Single Program, Multiple
    Data (SPMD), em que cada rank compila e executa um módulo StableHLO idêntico
    com operações coletivas correspondentes. No entanto, programas PyTorch podem
    apresentar comportamento Multiple Program, Multiple Data (MPMD) entre os
    ranks, por exemplo, devido a ramificações condicionadas ao rank
    (`if dist.get_rank() == 0:`) ou à fragmentação (sharding) desigual de
    tensores quando uma dimensão não é divisível igualmente pelo número de
    dispositivos. Se ranks diferentes executarem módulos StableHLO diferentes
    com coletivas, IDs de flags de sincronização ou offsets de buffer
    incompatíveis entre os ranks podem causar deadlocks em tempo de execução ou
    corrupção de memória.

### Modos de execução e fusão de coletivas { #collective-execution-and-fusion-modes }

O TorchTPU executa cada coletiva como uma coletiva isolada (skinny
collective). Para permitir que o XLA sobreponha computação e comunicação, você
pode ativar um ou ambos os mecanismos que fundem as coletivas com a computação
ao redor. Os dois são independentes e podem ser combinados: as coletivas
dentro de uma função `@spmd_safe` são sempre fundidas e, quando ativado, o
Compilation Handshake decide sobre as coletivas restantes em cada grafo do
`torch.compile`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Modo | Configuração | Comportamento do grafo | Ideal para |
| :---- | :---- | :---- | :---- |
| Coletivas isoladas (skinny collectives, comportamento padrão) | Sempre ativo para qualquer coletiva que as opções abaixo não fundem | Compila cada operação coletiva em um módulo StableHLO próprio e independente ("skinny"), inserindo quebras de grafo ao redor das coletivas. | Segurança geral para MPMD desde o início, ao custo de impedir que o XLA sobreponha computação e comunicação. |
| Decorador `@spmd_safe` (opcional) | `@torch.tpu.distributed.spmd_safe` | Desativa as quebras de grafo ao redor das coletivas dentro da função decorada, fundindo computação e coletivas em um único módulo StableHLO. | Funções ou etapas de treinamento sabidamente SPMD (por exemplo, SimpleFSDP ou paralelismo de tensor manual), para maximizar a sobreposição de computação e comunicação no XLA. |
| Compilation Handshake (opcional) | `TORCH_TPU_SPMD_HANDSHAKE_STAGE=COMPILE_STAGE` ou `DISPATCH_STAGE` (padrão: OFF) | Em grafos do torch.compile, verifica se os grafos FX compilados são iguais em todos os ranks. Funde as coletivas se forem e volta para coletivas isoladas se não forem. | Ativar automaticamente a fusão de coletivas no `torch.compile`, com proteção contra divergências MPMD acidentais. |
<!-- mdformat on -->

#### Como usar `@torch.tpu.distributed.spmd_safe` { #using-torchtpudistributedspmd_safe }

Quando você sabe que uma função ou etapa do modelo é estritamente SPMD em todos
os ranks (sem fluxo de controle dependente do rank nem formatos de tensor
desiguais), decore-a com `@torch.tpu.distributed.spmd_safe`. Com
`torch.compile` (e no modo eager `DEFER_AND_FUSE`), isso remove as quebras de
grafo ao redor das coletivas para que o XLA possa agendar a comunicação
assíncrona junto com a computação de matrizes:

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

Aplique `@torch.tpu.distributed.spmd_safe` somente a código que executa
exatamente a mesma sequência de operações de tensor e coletivas em todos os
ranks. Aplicá-lo a código MPMD pode gerar módulos StableHLO diferentes entre os
ranks e causar deadlocks na execução da TPU.

#### Como configurar o compilation handshake { #configuring-the-compilation-handshake }

Em vez de anotar funções manualmente com `@spmd_safe`, você pode ativar o
Compilation Handshake do TorchTPU com a variável de ambiente
`TORCH_TPU_SPMD_HANDSHAKE_STAGE`. Quando o `torch.compile` rastreia um grafo
que contém coletivas globais, o TorchTPU executa uma verificação de consenso
entre todos os ranks para confirmar que os grafos FX compilados são iguais
antes de fundir as coletivas no grafo:

*   `OFF` (padrão):** desativa o handshake. As coletivas usam as coletivas
    isoladas (Skinny Collectives) padrão, a menos que sejam marcadas com
    `@torch.tpu.distributed.spmd_safe`.
*   `COMPILE_STAGE`: verifica o consenso do grafo entre os ranks antes da
    compilação PjRT. Se todos os ranks produzirem grafos idênticos, as quebras
    de grafo ao redor das coletivas são omitidas. Recomendado para cargas de
    trabalho padrão de `torch.compile` em que todos os ranks compilam juntos.
*   `DISPATCH_STAGE`: compila as coletivas sem quebras de grafo e verifica o
    consenso entre os ranks no momento do despacho, recompilando com coletivas
    separadas se os ranks divergirem. Este modo é necessário para cargas de
    trabalho que podem disparar recompilações específicas de um rank. Ele tem
    uma sobrecarga maior que a de `COMPILE_STAGE`.

Veja abaixo um exemplo de recompilação específica de um rank com suporte em
`DISPATCH_STAGE`, enquanto `COMPILE_STAGE` excederia o tempo limite porque
somente `divergent_rank` recompila na iteração 1:

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

## Inicialização distribuída { #distributed-initialization }

A inicialização de uma carga de trabalho distribuída no TorchTPU usa as APIs
padrão de `torch.distributed`, junto com o auxiliar de ambiente de
inicialização do TorchTPU, que preenche as variáveis de ambiente de topologia
da TPU e do slice builder:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variável de ambiente | Descrição | Exemplo |
| :---- | :---- | :---- |
| `TORCH_TPU_TOPOLOGY` | Geometria física da malha da fração de TPU. | `"2,2,1"`, `"2,2,1,2"`, `"4,4,4"` |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Endereços `host:port` separados por vírgula (um por dispositivo) usados para construir a malha ICI. | `"localhost:8471,localhost:8472,..."` |
<!-- mdformat on -->

Você precisa definir essas variáveis antes de iniciar os workers. Em
configurações comuns (um único host ou uma fração de TPU no GKE), o TorchTPU
consegue determinar os valores corretos para você: execute

```bash
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
```

no seu script de inicialização antes do torchrun ou chame
`torch.tpu.distributed.set_tpu_launch_env()` no processo Python pai que
inicializa os workers, antes de iniciá-los. As duas opções mantêm inalteradas
as variáveis que você já definiu, então você pode substituir valores
individuais.

Veja abaixo o fluxo de inicialização de uma carga de trabalho distribuída do
TorchTPU, seguido de um exemplo completo de worker e de inicializador:

![Fluxo de inicialização distribuída](./images/distributed.png)

### 1. Script do worker (`train.py`) { #1-worker-script-trainpy }

Dentro de cada processo worker, inicialize o backend `tpu_dist` e obtenha o
dispositivo TPU local via `torch.device("tpu")`:

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

### 2. Como iniciar com o torchrun { #2-launching-with-torchrun }

Use o módulo de CLI `torch.tpu.distributed` para detectar a topologia de
hardware da TPU e exportar as variáveis de inicialização necessárias antes de
chamar o `torchrun`:

```bash
# Detect and export TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES,
# WORLD_SIZE, NODE_RANK, and multi-host coordination variables:
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
# Launch 1 worker process per TPU device on the host:
torchrun --nproc_per_node=$WORLD_SIZE train.py
```

Se preferir iniciar os workers pelo Python (por exemplo, com
`torch.distributed.launcher.elastic_launch`), chame
`torch.tpu.distributed.set_tpu_launch_env()` no processo pai antes de iniciar
os workers. O grupo de processos padrão abrange todos os dispositivos da
[fração de TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices).

## Mapeamento ideal da malha de dispositivos { #optimal-device-mesh-mapping }

Para maximizar a largura de banda das coletivas e minimizar a latência, alinhe
as dimensões de paralelismo do seu modelo à hierarquia física de interconexão
da TPU:

*   **Inter-Chip Interconnect (ICI)**: links diretos de alta velocidade que
    conectam os chips TPU em uma malha toroidal 2D/3D ou toroidal torcida
    (twisted torus) dentro de uma fração (de dezenas a centenas de GB/s por
    link).
*   **Data Center Network (DCN)**: Ethernet no nível do host que conecta
    frações de TPU separadas em cargas de trabalho multi-slice (largura de
    banda menor e latência maior que a ICI).

Use `torch.tpu.distributed.topology_aware_mesh()` para calcular um layout de
ranks alinhado aos anéis físicos de ICI. Ordene as dimensões da sua malha
lógica da **menor intensidade de comunicação** (mais externa) para a **maior
intensidade de comunicação** (mais interna):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Tipo de paralelismo | Coletivas típicas | Frequência de comunicação | Posicionamento físico recomendado |
| :---- | :---- | :---- | :---- |
| **Paralelismo de tensor (TP)** | `all_reduce`, `reduce_scatter`, `all_gather` | Toda camada do transformer (forward + backward) | Anel ICI local mais rápido (mesma bandeja/host, dispositivos contíguos a 1 salto) |
| **Paralelismo de contexto / sequência (CP/SP)** | `all_to_all`, Ring Attention | Por camada de atenção | Eixos ICI locais (dentro da fração) |
| **Paralelismo de especialistas (EP)** | `all_to_all`, `ragged_all_to_all` | Por camada de roteador MoE | Eixos ICI locais (dentro da fração) |
| **Paralelismo de dados totalmente fragmentado (FSDP)** | `all_gather`, `reduce_scatter` | Por módulo / bloco do transformer | Eixos ICI (dentro da fração) |
| **Paralelismo de dados replicado (DP)** | `all_reduce` (somente gradientes) | Uma vez por etapa de treinamento | Eixo ICI externo ou DCN (entre frações) |
<!-- mdformat on -->

### Exemplo: como construir um DeviceMesh 2D com reconhecimento de topologia { #example-constructing-a-topology-aware-2d-devicemesh }

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

Para cargas de trabalho multi-slice (híbridas DCN + ICI), passar
`dcn_mesh_shape` junto com `mesh_shape` é opcional e só é necessário para
escolher a divisão explicitamente. Por padrão,
`torch.tpu.distributed.topology_aware_mesh()` detecta as frações
automaticamente e mapeia a dimensão não trivial mais externa da malha
(geralmente o paralelismo de dados) entre as frações pela DCN e as dimensões
internas dentro de cada fração pela ICI.

## Kernels Pallas distribuídos { #distributed-pallas-kernels }

O TorchTPU oferece suporte a kernels
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) personalizados que
fazem Remote DMA (RDMA) direto entre dispositivos pela ICI, sem envolver a CPU
do host. Com isso, você pode criar kernels personalizados de comunicação ponto
a ponto ou em anel, como Ring Attention, embaralhamentos All-to-All
personalizados, deslocamentos de paralelismo de pipeline ou trocas de halo, e
chamá-los como operações personalizadas nativas do PyTorch tanto no modo eager
quanto com `torch.compile`.

### Como escrever e registrar um kernel Pallas com comunicação { #writing-and-registering-a-communicating-pallas-kernel }

A integração de um kernel Pallas distribuído envolve três etapas:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Etapa | Principais passos e operações |
| :---- | :---- |
| **1. Corpo do kernel Pallas** | Consulte as coordenadas lógicas com `jax.lax.axis_index` e `jax.lax.axis_size`, construa um descritor de transferência assíncrona via `pltpu.make_async_remote_copy(..., device_id=(target,))` e gerencie a conclusão com `.start()` e `.wait()`. |
| **2. `pl.pallas_call` e `jax.shard_map`** | Aloque semáforos de DMA (`pltpu.SemaphoreType.DMA`) em `scratch_shapes` e envolva o kernel com `jax.shard_map` sobre um `jax.sharding.Mesh` alinhado aos ranks e do tamanho do mundo (world size) e um `PartitionSpec`. |
| **3. Registro no TorchTPU** | Registre uma função wrapper JAX com anotações de tipo usando `torch.tpu.pallas.jax_op`, passando `mesh`, `input_partition_specs` e `output_partition_specs`. |
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

### Malhas alinhadas aos ranks e atribuição de dispositivos { #rank-aligned-meshes-and-device-assignment }

Em muitas arquiteturas de TPU, os IDs de dispositivo e os ranks atribuídos não
têm correspondência de 1 para 1: os ranks são atribuídos de modo que os
vizinhos sejam os dispositivos adjacentes (o rank 1 fica ao lado do rank 0 e
do rank 2), mas, nesse cenário, os IDs de dispositivo na v7x são 0, 1 e 4.
Para facilitar essa tradução, o TorchTPU oferece
`torch.tpu.pallas.get_pallas_mesh`, que fornece uma malha Pallas/JAX que
corresponde exatamente ao ID de rank usado no restante da aplicação Torch.

*   **Construção de malha alinhada aos ranks**:
    `torch.tpu.pallas.get_pallas_mesh(axis_names=..., mesh_shape=...,
    process_group=...)` consulta o mapeamento de rank para dispositivo do
    `ProcessGroup` ativo do PyTorch e constrói um `jax.sharding.Mesh` de
    dispositivos abstratos, apenas para compilação, sem inicializar um backend
    PjRT do JAX.
*   **Requisito de malha do tamanho do mundo**: kernels Pallas distribuídos
    exigem uma malha cujo número total de dispositivos seja igual a
    `dist.get_world_size()`. Para topologias de paralelismo multidimensionais
    (como paralelismo de dados + de tensor), passe um `mesh_shape` cujo
    produto seja igual a `world_size`, por exemplo,
    `torch.tpu.pallas.get_pallas_mesh(axis_names=("dp", "tp"), mesh_shape=(2,
    4))`.
*   **Atribuição unificada de dispositivos para coletivas e kernels**: quando
    `mesh` é fornecido para `torch.tpu.pallas.jax_op`, o TorchTPU extrai o
    mapeamento de rank para dispositivo da malha como o `DeviceAssignment` do
    XLA para o kernel e o concilia com quaisquer coletivas padrão de
    `torch.distributed` fundidas no mesmo módulo compilado. Isso garante que
    `jax.lax.axis_index`, os DMAs endereçados pela malha
    (`device_id={axis: target}`) e as coletivas do PyTorch sejam todos
    roteados de forma consistente, de acordo com a ordem de ranks do PyTorch.
