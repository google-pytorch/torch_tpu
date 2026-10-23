# TorchTPU 분산 지원 { #torchtpu-distributed-support }

## 분산 지원 { #distributed-support }

TorchTPU는 PyTorch ProcessGroup 수준(`c10d::Backend`)에서 `tpu_dist` 백엔드
이름으로 분산 통신을 구현합니다. ProcessGroup 계층에서 집합 통신(collective)을
가로채므로 TorchTPU는 eager 모드와 `torch.compile` 모두에서 표준
`torch.distributed` API를 지원하며, DTensor, DistributedDataParallel(DDP),
FSDPv2, [SimpleFSDP](https://arxiv.org/abs/2411.00284) 등 PyTorch 분산
프레임워크와 바로 함께 작동합니다.

GPU 기반 PyTorch와 마찬가지로 TorchTPU는 디바이스당 프로세스 1개 실행 모델을
적용합니다. 내부적으로 TPU 간 통신은 별도의 런타임 통신 라이브러리(예: NCCL)를
사용하지 않습니다. 대신 집합 통신은
[StableHLO 집합 통신 연산](https://openxla.org/stablehlo/spec#collective_ops)
(예: `stablehlo.all_reduce`, `stablehlo.all_gather`,
`stablehlo.reduce_scatter`, `stablehlo.all_to_all`)으로 직접 로어링되고
XLA가 이를 TPU 실행 파일로 컴파일합니다.

### GPU와의 주요 아키텍처 차이점 { #key-architectural-differences-from-gpu }

TorchTPU는 표준 `torch.distributed` 시맨틱스를 유지하지만, StableHLO와 PjRT를
통해 TPU를 프로그래밍하면 GPU 백엔드와 비교해 세 가지 주요 차이점이 생깁니다.

*   **초기화 전 하드웨어 토폴로지 검색:** 멀티 GPU 호스트에서는 torchrun이
    시작한 각 프로세스가 연결된 모든 GPU를 확인하고, `cuda:{local_rank}`로 할당된
    디바이스를 선택하고, `torch.distributed.init_process_group()`을 호출하기 전에
    모델을 해당 디바이스로 옮길 수 있습니다. TPU 호스트에서는 각 워커 프로세스가 TPU
    디바이스 1개에 바인딩되며 혼자서 전체 디바이스 세트를 소유할 수 없습니다. PjRT
    런타임을 초기화하거나 `torch.distributed.init_process_group()`을 호출하기
    전에 환경에서
    [TPU 슬라이스](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)의
    물리적 메시 구성(`TORCH_TPU_TOPOLOGY`), ICI(Inter-Chip Interconnect) 메시를
    구성하는 데 사용하는 로컬 코디네이터 주소(`TORCH_TPU_SLICEBUILDER_ADDRESSES`),
    참여하는 TPU 디바이스의 총수(`WORLD_SIZE`)를 지정해야 합니다.
*   **하위 그룹의 전역 복제본 그룹 결정:** PyTorch에서
    [`torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
    같은 집합 통신 호출은 호출하는 랭크가 참여하는 ProcessGroup(하위
    그룹)만 받습니다. 반면
    [`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce)는
    기본 프로세스 그룹 전체에서 참여하는 모든 하위 그룹에 대한 완전한
    설명(`replica_groups`)이 필요하며, 여기에는 현재 프로세스가 속하지 않은 하위
    그룹도 포함됩니다.

    ```
    %0 = "stablehlo.all_reduce"(%arg0) <{
      replica_groups = dense<[[4, 0, 1, 3], [5, 2, 6, 7]]> : tensor<2x4xi64>
    }> ({ ... })
    ```

    TorchTPU의 `tpu_dist` 백엔드는 집합 통신을 설정하는 동안 기본 프로세스 그룹의
    모든 랭크 간에 하위 그룹 멤버십을 자동으로 교환하고, XLA에 필요한 전역
    `replica_groups`와 `device_assignment`를 구성합니다.
*   **SPMD 컴파일과 MPMD 실행:** XLA 컴파일러와 TPU 통신 스택은 SPMD(Single
    Program, Multiple Data) 실행을 위해 설계되었습니다. SPMD에서는 모든 랭크가
    일치하는 집합 통신을 포함한 동일한 StableHLO 모듈을 컴파일하고 실행합니다.
    하지만 PyTorch 프로그램은 랭크별로 MPMD(Multiple Program, Multiple Data)
    동작을 보일 수 있습니다. 예를 들어 랭크 조건부 분기
    (`if dist.get_rank() == 0:`)가 있거나, 차원이 디바이스 수로 나누어떨어지지 않아
    텐서 샤딩이 고르지 않은 경우입니다. 여러 랭크가 집합 통신을 포함한 서로 다른
    StableHLO 모듈을 실행하면 랭크 간 동기화 플래그 ID나 버퍼 오프셋이 일치하지
    않아 런타임 교착 상태나 메모리 손상이 발생할 수 있습니다.

### 집합 통신 실행 및 퓨전 모드 { #collective-execution-and-fusion-modes }

TorchTPU는 각 집합 통신을 스키니(skinny) 집합 통신으로 실행합니다. XLA가 계산과
통신을 겹칠 수 있게 하려면 집합 통신을 주변 계산과 퓨전하는 두 가지 메커니즘 중
하나 또는 둘 다를 선택적으로 사용 설정할 수 있습니다. 두 메커니즘은 서로
독립적이며 함께 사용할 수 있습니다. `@spmd_safe` 함수 내부의 집합 통신은 항상
퓨전되며, 컴파일 핸드셰이크를 사용 설정하면 각 `torch.compile` 그래프의 나머지
집합 통신은 핸드셰이크가 결정합니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 모드 | 구성 | 그래프 동작 | 적합한 용도 |
| :---- | :---- | :---- | :---- |
| 스키니 집합 통신(기본 동작) | 아래 선택 기능이 퓨전하지 않는 모든 집합 통신에 항상 적용됨 | 집합 통신 주위에 그래프 중단을 삽입하여 각 집합 통신을 독립된("스키니") StableHLO 모듈로 컴파일합니다. | 별도 설정 없이 일반적인 MPMD 안전성을 제공하지만, XLA가 계산과 통신을 겹치지 못하는 단점이 있습니다. |
| `@spmd_safe` 데코레이터(선택) | `@torch.tpu.distributed.spmd_safe` | 데코레이터가 적용된 함수 내부의 집합 통신 주위에서 그래프 중단을 사용 중지하여 계산과 집합 통신을 단일 StableHLO 모듈로 퓨전합니다. | SPMD임이 확실한 함수나 학습 단계(예: SimpleFSDP 또는 수동 텐서 병렬 처리)에서 XLA의 계산-통신 오버랩을 극대화합니다. |
| 컴파일 핸드셰이크(선택) | `TORCH_TPU_SPMD_HANDSHAKE_STAGE=COMPILE_STAGE` 또는 `DISPATCH_STAGE`(기본값: OFF) | torch.compile 그래프에서 컴파일된 FX 그래프가 모든 랭크에서 일치하는지 확인합니다. 일치하면 집합 통신을 퓨전하고, 일치하지 않으면 스키니 집합 통신으로 대체합니다. | 의도치 않은 MPMD 분기를 방지하면서 `torch.compile`에서 집합 통신 퓨전을 자동으로 사용 설정합니다. |
<!-- mdformat on -->

#### `@torch.tpu.distributed.spmd_safe` 사용 { #using-torchtpudistributedspmd_safe }

함수나 모델 단계가 모든 랭크에서 엄격하게 SPMD임을 알고 있다면(랭크에 따른
제어 흐름이나 고르지 않은 텐서 형상이 없음)
`@torch.tpu.distributed.spmd_safe`로 데코레이트하세요. `torch.compile`(및
`DEFER_AND_FUSE` eager 모드)에서 이 데코레이터는 집합 통신 주위의 그래프
중단을 제거하므로 XLA가 비동기 통신을 행렬 계산과 함께 스케줄링할 수 있습니다.

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

`@torch.tpu.distributed.spmd_safe`는 모든 랭크에서 정확히 같은 순서의 텐서 연산과
집합 통신을 실행하는 코드에만 적용하세요. MPMD 코드에 적용하면 랭크마다 다른
StableHLO 모듈이 생성되어 TPU 실행 교착 상태가 발생할 수 있습니다.

#### 컴파일 핸드셰이크 구성 { #configuring-the-compilation-handshake }

함수에 `@spmd_safe`를 직접 지정하는 대신 `TORCH_TPU_SPMD_HANDSHAKE_STAGE` 환경
변수로 TorchTPU의 컴파일 핸드셰이크를 사용 설정할 수 있습니다. `torch.compile`이
전역 집합 통신을 포함한 그래프를 트레이싱하면 TorchTPU는 집합 통신을 그래프에
퓨전하기 전에 모든 랭크에서 합의 확인을 실행하여 컴파일된 FX 그래프가 일치하는지
검증합니다.

*   `OFF`(기본값): 핸드셰이크를 사용 중지합니다. 집합 통신은
    `@torch.tpu.distributed.spmd_safe`로 표시되지 않은 한 기본 스키니 집합 통신을
    사용합니다.
*   `COMPILE_STAGE`: PjRT 컴파일 전에 랭크 간 그래프 합의를 검증합니다. 모든
    랭크가 동일한 그래프를 생성하면 집합 통신 주위의 그래프 중단이 생략됩니다.
    모든 랭크가 함께 컴파일하는 표준 `torch.compile` 워크로드에 권장됩니다.
*   `DISPATCH_STAGE`: 그래프 중단 없이 집합 통신을 컴파일하고 디스패치 시점에
    랭크 간 합의를 검증하며, 랭크가 서로 달라지면 분할된 집합 통신으로 다시
    컴파일합니다. 이 모드는 랭크별 재컴파일을 일으킬 수 있는 워크로드에
    필요하며, `COMPILE_STAGE`보다 오버헤드가 큽니다.

다음은 `DISPATCH_STAGE`에서는 지원되지만 `COMPILE_STAGE`에서는 반복 1에서
`divergent_rank`만 재컴파일하므로 시간 초과가 발생하는 랭크별 재컴파일의
예입니다.

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

## 분산 초기화 { #distributed-initialization }

TorchTPU에서 분산 워크로드를 초기화할 때는 표준 `torch.distributed` API를
사용하며, TorchTPU의 실행 환경 도우미로 TPU 토폴로지와 슬라이스 빌더 환경 변수를
채웁니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 환경 변수 | 설명 | 예 |
| :---- | :---- | :---- |
| `TORCH_TPU_TOPOLOGY` | TPU 슬라이스의 물리적 메시 구성입니다. | `"2,2,1"`, `"2,2,1,2"`, `"4,4,4"` |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | ICI 메시를 구성하는 데 사용하는 쉼표로 구분된 `host:port` 주소(디바이스당 1개)입니다. | `"localhost:8471,localhost:8472,..."` |
<!-- mdformat on -->

워커를 시작하기 전에 이 변수를 설정해야 합니다. 일반적인 구성(단일 호스트 또는
GKE TPU 슬라이스)에서는 TorchTPU가 올바른 값을 대신 결정할 수 있습니다. 다음을

```bash
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
```

실행 스크립트에서 torchrun 전에 실행하거나, 워커를 생성하기 전에 상위 Python
런처 프로세스에서 `torch.tpu.distributed.set_tpu_launch_env()`를 호출하세요. 두
방법 모두 이미 설정한 변수는 변경하지 않으므로 개별 값을 재정의할 수 있습니다.

다음은 분산 TorchTPU 워크로드의 초기화 흐름이며, 그 뒤에 전체 워커 및 런처
예제가 이어집니다.

![분산 초기화 흐름](./images/distributed.png)

### 1. 워커 스크립트(`train.py`) { #1-worker-script-trainpy }

각 워커 프로세스 내부에서 `tpu_dist` 백엔드를 초기화하고
`torch.device("tpu")`로 로컬 TPU 디바이스를 가져오세요.

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

### 2. torchrun으로 실행 { #2-launching-with-torchrun }

`torchrun`을 호출하기 전에 `torch.tpu.distributed` CLI 모듈을 사용하여 TPU
하드웨어 토폴로지를 감지하고 필요한 실행 변수를 내보내세요.

```bash
# Detect and export TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES,
# WORLD_SIZE, NODE_RANK, and multi-host coordination variables:
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
# Launch 1 worker process per TPU device on the host:
torchrun --nproc_per_node=$WORLD_SIZE train.py
```

또는 Python에서 워커를 시작하는 경우(예:
`torch.distributed.launcher.elastic_launch` 사용) 워커를 생성하기 전에 상위
프로세스에서 `torch.tpu.distributed.set_tpu_launch_env()`를 호출하세요. 기본
프로세스 그룹은
[TPU 슬라이스](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)의
모든 디바이스를 포함합니다.

## 최적의 디바이스 메시 매핑 { #optimal-device-mesh-mapping }

집합 통신 대역폭을 극대화하고 지연 시간을 최소화하려면 모델의 병렬 처리 차원을
물리적 TPU 상호 연결 계층 구조에 맞추세요.

*   **ICI(Inter-Chip Interconnect)**: 슬라이스 내에서 2D/3D 토러스 또는 트위스티드
    토러스 메시로 TPU 칩을 연결하는 고속 직접 링크입니다(링크당 수십~수백 GB/s).
*   **DCN(Data Center Network)**: 멀티슬라이스 워크로드에서 서로 다른 TPU
    슬라이스를 연결하는 호스트 수준 이더넷입니다(ICI보다 대역폭이 낮고 지연
    시간이 김).

`torch.tpu.distributed.topology_aware_mesh()`를 사용하여 물리적 ICI 링에 맞춘
랭크 레이아웃을 계산하세요. 논리적 메시 차원은 **통신 강도가 가장 낮은 차원**
(가장 바깥쪽)부터 **통신 강도가 가장 높은 차원**(가장 안쪽) 순으로 정렬하세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 병렬 처리 유형 | 일반적인 집합 통신 | 통신 빈도 | 권장 물리적 배치 |
| :---- | :---- | :---- | :---- |
| **텐서 병렬(TP)** | `all_reduce`, `reduce_scatter`, `all_gather` | 모든 트랜스포머 레이어(순전파 + 역전파) | 가장 빠른 로컬 ICI 링(같은 트레이/호스트, 연속된 1홉 디바이스) |
| **컨텍스트/시퀀스 병렬(CP/SP)** | `all_to_all`, Ring Attention | 어텐션 레이어마다 | 로컬 ICI 축(슬라이스 내) |
| **전문가 병렬(EP)** | `all_to_all`, `ragged_all_to_all` | MoE 라우터 레이어마다 | 로컬 ICI 축(슬라이스 내) |
| **완전 샤딩 데이터 병렬(FSDP)** | `all_gather`, `reduce_scatter` | 모듈/트랜스포머 블록마다 | ICI 축(슬라이스 내) |
| **복제 데이터 병렬(DP)** | `all_reduce`(그래디언트만) | 학습 단계당 1회 | 바깥쪽 ICI 축 또는 DCN(슬라이스 간) |
<!-- mdformat on -->

### 예: 토폴로지 인식 2D DeviceMesh 구성 { #example-constructing-a-topology-aware-2d-devicemesh }

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

멀티슬라이스(하이브리드 DCN + ICI) 워크로드에서 `mesh_shape`와 함께
`dcn_mesh_shape`를 전달하는 것은 선택사항이며, 분할을 명시적으로 선택할 때만
필요합니다. 기본적으로 `torch.tpu.distributed.topology_aware_mesh()`는 슬라이스를
자동으로 감지하고, 가장 바깥쪽의 크기가 1이 아닌 메시 차원(일반적으로 데이터
병렬)을 DCN을 통해 슬라이스 간에 매핑하며, 안쪽 차원은 ICI를 통해 각 슬라이스
내에 매핑합니다.

## 분산 Pallas 커널 { #distributed-pallas-kernels }

TorchTPU는 호스트 CPU를 거치지 않고 ICI를 통해 디바이스 간 원격 DMA(RDMA)를 직접
수행하는 커스텀 [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) 커널을
지원합니다. 이를 통해 Ring Attention, 커스텀 All-to-All 셔플, 파이프라인 병렬
시프트, 헤일로 교환 같은 커스텀 점대점 또는 링 통신 커널을 작성하고, eager
모드와 `torch.compile` 모두에서 네이티브 PyTorch 커스텀 연산으로 호출할 수
있습니다.

### 통신하는 Pallas 커널 작성 및 등록 { #writing-and-registering-a-communicating-pallas-kernel }

분산 Pallas 커널 통합은 세 단계로 이루어집니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 단계 | 주요 단계 및 작업 |
| :---- | :---- |
| **1. Pallas 커널 본문** | `jax.lax.axis_index`와 `jax.lax.axis_size`로 논리 좌표를 조회하고, `pltpu.make_async_remote_copy(..., device_id=(target,))`로 비동기 전송 디스크립터를 구성하고, `.start()`와 `.wait()`로 완료를 관리합니다. |
| **2. `pl.pallas_call` 및 `jax.shard_map`** | `scratch_shapes`에 DMA 세마포어(`pltpu.SemaphoreType.DMA`)를 할당하고, 랭크에 맞춰 정렬된 월드 크기의 `jax.sharding.Mesh`와 `PartitionSpec`에 대해 `jax.shard_map`으로 커널을 래핑합니다. |
| **3. TorchTPU 등록** | `torch.tpu.pallas.jax_op`을 사용하여 타입 주석이 달린 JAX 래퍼 함수를 등록하고 `mesh`, `input_partition_specs`, `output_partition_specs`를 전달합니다. |
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

### 랭크 정렬 메시 및 디바이스 할당 { #rank-aligned-meshes-and-device-assignment }

많은 TPU 아키텍처에서 할당된 디바이스 ID와 랭크는 일대일로 대응하지 않습니다. 랭크는
인접한 디바이스가 이웃이 되도록 할당되지만(랭크 1은 랭크 0 및 랭크 2와 인접), 이
경우 v7x에서 디바이스 ID는 0, 1, 4입니다. 이 변환을 쉽게 하기 위해 torchTPU는
`torch.tpu.pallas.get_pallas_mesh`를 제공합니다. 이 함수는 Torch 애플리케이션의
나머지 부분에서 사용하는 랭크 ID와 정확히 일치하는 Pallas/JAX 메시를 제공합니다.

*   **랭크 정렬 메시 구성**:
    `torch.tpu.pallas.get_pallas_mesh(axis_names=..., mesh_shape=...,
    process_group=...)`는 활성 PyTorch `ProcessGroup`의 랭크-디바이스 매핑을 조회하고,
    JAX PjRT 백엔드를 초기화하지 않고 컴파일 전용 추상 디바이스로 구성된
    `jax.sharding.Mesh`를 구성합니다.
*   **월드 크기 메시 요구사항**: 분산 Pallas 커널에는 총 디바이스 수가
    `dist.get_world_size()`와 같은 메시가 필요합니다. 다차원 병렬 토폴로지(예:
    데이터 + 텐서 병렬 처리)의 경우 곱이 `world_size`와 같은 `mesh_shape`를
    전달하세요. 예:
    `torch.tpu.pallas.get_pallas_mesh(axis_names=("dp", "tp"), mesh_shape=(2,
    4))`.
*   **집합 통신과 커널의 통합 디바이스 할당**: `torch.tpu.pallas.jax_op`에 `mesh`를
    제공하면 TorchTPU는 메시의 랭크-디바이스 매핑을 커널의 XLA `DeviceAssignment`로
    추출하고, 같은 컴파일된 모듈에 퓨전된 표준 `torch.distributed` 집합 통신과
    조정합니다. 이렇게 하면 `jax.lax.axis_index`, 메시 주소 지정 DMA
    (`device_id={axis: target}`), PyTorch 집합 통신이 모두 PyTorch 랭크 순서에
    따라 일관되게 라우팅됩니다.
