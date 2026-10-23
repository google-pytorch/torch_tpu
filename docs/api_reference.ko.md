# TorchTPU API 참조 { #torchtpu-api-reference }

이 문서는 TorchTPU의 공개 Python API와 환경 변수에 대한 참조 가이드입니다. 이
문서를 사용하여 다음 작업을 할 수 있습니다.

*   **CUDA에서 TPU로 마이그레이션**: 자주 사용하는 `torch.cuda` 호출을 PyTorch의
    디바이스 독립적인
    [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
    API에 매핑합니다.
*   **TPU 전용 기능 사용**: **`torch.tpu`** 모듈로 실행 모드, MXU 수치 정밀도,
    분산 토폴로지, 커스텀 Pallas/TPU 연산자, 트레이싱 가능한 옵티마이저, XProf
    프로파일링을 구성합니다.
*   **런타임 구성**: 환경 변수로 컴파일 캐싱, 디버깅 검사, 멀티 호스트 실행
    설정을 조정합니다.

> **Note:** 모든 공개 Python API와 환경 변수를 빠짐없이 담은, 자동 생성되는 기준
> 목록은
> [TorchTPU 공개 API(`public_apis.md`)](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/public_apis.md)를
> 참고하세요.

## **핵심 디바이스 쿼리** { #core-device-queries }

TorchTPU가 설치된 상태에서 `torch`를 가져오면 백엔드가 `"tpu"` 디바이스와
`torch.tpu` 모듈을 자동으로 등록합니다. PyTorch가 디바이스 독립적인
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
API를 제공하는 경우 백엔드 전용 API 대신 이를 사용하세요. 그러면 코드를 여러
가속기 간에 이식할 수 있습니다. 범용 대응 API가 없는 TPU 전용 하드웨어 속성에는
`torch.tpu`를 사용하세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU / 가속기 API | 설명 |
| :--- | :--- | :--- |
| `torch.device("cuda")` | `torch.accelerator.current_accelerator()` | TPU가 활성 가속기이면 `torch.device("tpu")`를 반환합니다. |
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | TPU 백엔드를 사용할 수 있는지 확인합니다. |
| `torch.cuda.is_initialized()` | `torch.tpu.is_initialized()` | TorchTPU의 런타임 상태가 초기화되었는지 확인합니다. |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | 현재 프로세스에 표시되는 디바이스 수입니다. |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | 활성 TPU 디바이스 인덱스를 반환합니다. |
| `torch.cuda.set_device(rank)` | `torch.accelerator.set_device_index(rank)` | 현재 프로세스의 활성 TPU 디바이스 인덱스를 설정합니다. |
| `torch.cuda.device(rank)` | `torch.accelerator.device_index(rank)` | 활성 디바이스 인덱스를 일시적으로 설정하는 컨텍스트 관리자입니다. |
| `torch.cuda.get_device_name(device=None)` | `torch.tpu.get_device_properties(device=None).name` | 연결된 TPU의 사람이 읽을 수 있는 모델 이름(예: `"TPU v6e"`)을 반환합니다. |
| `torch.cuda.get_device_properties(device=None)` | `torch.tpu.get_device_properties(device=None)` | 정적 TPU 디바이스 속성(`name`, `devices_per_chip`, `hbm_bytes_per_chip`, `hbm_bytes_per_device`, `bf16_flops_per_chip`, `bf16_flops_per_device`, `sparse_cores_per_device`, `sparse_core_simd_width` 포함)을 반환합니다. |
<!-- mdformat on -->

이 표의 `torch.accelerator` API에는 PyTorch 2.11 이상이 필요합니다. 드물게 이전
PyTorch 버전을 사용하는 경우에는 대응하는 `torch.tpu` API(예:
`torch.tpu.device_count()` 또는 `torch.tpu.current_device()`)를 대신 사용하세요.
PyTorch 2.11 이상에서는 이러한 중복 `torch.tpu` API가 TorchTPU 내부용이며, 사용자
코드에서 직접 호출하면 오류가 발생합니다.

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

## **난수 생성(RNG)** { #random-number-generation-rng }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU / 가속기 API | 설명 |
| :--- | :--- | :--- |
| `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | 활성 TPU 디바이스의 난수 생성기에 시드를 설정합니다. |
| `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | 모든 로컬 TPU 디바이스의 RNG에 시드를 설정합니다. |
| `torch.cuda.initial_seed()` | `torch.accelerator.random.initial_seed()` | 현재 TPU 디바이스의 초기 난수 시드를 반환합니다. |
| `torch.cuda.get_rng_state()` | `torch.accelerator.random.get_rng_state()` | 현재 디바이스의 RNG 상태 텐서를 반환합니다. |
| `torch.cuda.set_rng_state(new_state, device=...)` | `torch.tpu.set_rng_state(new_state, device=...)` | 현재 프로세스의 활성 TPU 디바이스에 대한 RNG 상태 텐서를 설정합니다. |
| `torch.cuda.default_generators` | `torch.tpu.default_generators` | 로컬 디바이스 서수로 인덱싱되는 기본 `torch.Generator` 인스턴스의 튜플이며, 처음 사용할 때 초기화됩니다. |
<!-- mdformat on -->

이 표의 `torch.accelerator.random` API에는 PyTorch 2.13 이상이 필요합니다.
드물게 이전 PyTorch 버전을 사용하는 경우에는 `torch.tpu.initial_seed()` 또는
`torch.tpu.get_rng_state()`를 대신 사용하세요. PyTorch 2.13 이상에서는 이러한
`torch.tpu` API가 TorchTPU 내부용이며, 코드에서 호출하면 오류가 발생합니다.

```python
# Set the seed across all cores
torch.tpu.manual_seed_all(42)
print("Seeded all TPU cores with value 42.")
```

## **자동 혼합 정밀도(AMP)** { #automatic-mixed-precision-amp }

TorchTPU는 `"tpu"` 디바이스 유형 문자열을 사용하여 PyTorch의 표준
[`torch.amp.autocast`](https://docs.pytorch.org/docs/stable/amp.html) 프레임워크와
통합됩니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU API | 설명 |
| :--- | :--- | :--- |
| `torch.amp.autocast("cuda", ...)` | `torch.amp.autocast("tpu", ...)` | 자동 혼합 정밀도(TPU의 기본값은 `bfloat16`)를 사용 설정하는 컨텍스트 관리자/데코레이터입니다. |
| `torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | TPU에서 지원되는 dtype(`bfloat16`, `float16`, `float32`, `float64`, `float8_*`, 정수, 불리언, 복소수 유형)을 나열합니다. |
<!-- mdformat on -->

```python
supported_dtypes = torch.tpu.get_amp_supported_dtype()
print(f"AMP supported dtypes: {supported_dtypes}")
```

## **스트림 및 이벤트** { #streams-and-events }

스트림 관리와 디바이스 동기화에는 백엔드 전용 API 대신 표준
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
API를 사용하세요. `torch.tpu`는 TPU `Stream` 또는 `Event` 객체를 직접 생성하거나
TPU 코어 고정을 구성할 때만 사용하세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU / 가속기 API | 설명 |
| :--- | :--- | :--- |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | 디바이스의 활성 스트림을 반환합니다. |
| `torch.cuda.set_stream(stream)` | `torch.accelerator.set_stream(stream)` | 활성 스트림을 설정합니다. |
| `torch.cuda.default_stream(device=None)` | `torch.tpu.default_stream(device=None)` | 디바이스의 기본 `TpuStream`을 반환합니다. |
| `torch.cuda.stream(stream)` | `torch.tpu.stream(stream)` | 지정한 `TpuStream`을 선택하는 컨텍스트 관리자입니다. |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` | TPU 디바이스에서 대기 중인 모든 작업이 완료될 때까지 기다립니다. |
| `torch.cuda.Stream` | `torch.tpu.Stream(device=None, priority=0, *, pinned=None)` | `torch.Stream`의 서브클래스입니다. `synchronize()`, `record_event()`, `wait_event()`, `wait_stream()`, 코어 고정(`pinned="dense"` 또는 `"sparse"`, `is_dense` / `is_sparse`로 조회)을 지원합니다. |
| `torch.cuda.Event` | `torch.tpu.Event(enable_timing=False)` | `torch.Event`의 서브클래스입니다. `record()`, `synchronize()`, `query()`, `elapsed_time(end_event)`를 지원합니다. |
<!-- mdformat on -->

이 표의 `torch.accelerator` API에는 PyTorch 2.11 이상이 필요합니다. 드물게 이전
PyTorch 버전을 사용하는 경우에는 `torch.tpu.current_stream()`,
`torch.tpu.set_stream()` 또는 `torch.tpu.synchronize()`를 대신 사용하세요.

> **Note:** 스트림은 CUDA 개념으로, TPU에는 이에 직접 대응하는 개념이 없습니다.
> TorchTPU의 지연 실행 모델에서는 XLA 컴파일러와 PjRt 런타임이 데이터 종속성에
> 따라 연산을 자동으로 스케줄링하고 오버랩하므로 여러 스트림을 직접 관리할 필요가
> 없습니다.

## **TPU 전용 API** { #tpu-specific-apis }

`torch.tpu`는 표준 CUDA 대응 API 외에도 TPU 실행 모드, MXU 정밀도, XLA 컴파일,
분산 토폴로지, 커스텀 Pallas 커널, 트레이싱 가능한 옵티마이저, 프로파일링을 위해
특별히 설계된 API를 제공합니다.

### **실행 모드 및 컴파일러 옵션** { #execution-modes-and-compiler-options }

eager 연산이 큐에 추가되고, 퓨전되고, XLA로 컴파일되는 방식을
제어합니다. 아키텍처 설명은
[TorchTPU 실행 모드](./torchtpu_execution_model.md)를 참고하세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 설명 |
| :--- | :--- |
| `torch.tpu.EagerMode` | eager 실행을 제어하는 열거형으로, `DEFER_AND_FUSE`, `DEFER_NEVER`, `DEFER_NEVER_AND_LAUNCH_BLOCKING` 값이 있습니다. |
| `torch.tpu.eager_mode` | 활성 `EagerMode`를 전역으로 가져오거나 설정합니다. |
| `torch.tpu.set_eager_mode(mode)` | 활성 `EagerMode`를 일시적으로 설정하는 컨텍스트 관리자입니다. |
| `torch.tpu.compiler.custom_compiler_options(options)` | 블록 내부에서 구체화되는 eager 연산에 커스텀 XLA 컴파일러 플래그를 전달하는 컨텍스트 관리자입니다. |
<!-- mdformat on -->

```python
print(f"Default eager_mode: {torch.tpu.eager_mode}")
with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
  x = torch.ones((4, 4), device="tpu") + 1
  print(
      f"Inside context manager: {torch.tpu.eager_mode}, sum={x.sum().item()}"
  )
```

### **수치 정밀도** { #numerical-precision }

MXU(Matrix Multiply Unit) 정밀도와 중간 `float32` 누적을 제어합니다. 자세한
안내는 [수치 정밀도](./numerical_precision.md)를 참고하세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 설명 |
| :--- | :--- |
| `torch.tpu.Precision` | `float32` 행렬 곱셈과 컨볼루션의 MXU 정밀도를 제어하는 열거형으로, `Precision.DEFAULT`, `Precision.HIGH`, `Precision.HIGHEST` 값이 있습니다. |
| `torch.tpu.precision(level)` | 블록 내 행렬 곱셈 및 컨볼루션 연산의 `Precision` 수준을 설정하는 컨텍스트 관리자입니다. |
| `torch.tpu.allow_excess_precision` | 퓨전된 중간 값을 `float32`로 유지하기 위해 XLA의 `--xla_allow_excess_precision` 플래그를 전역으로 가져오거나 설정합니다. |
<!-- mdformat on -->

```python
a = torch.randn(64, 64, device="tpu", dtype=torch.float32)
b = torch.randn(64, 64, device="tpu", dtype=torch.float32)
with torch.tpu.precision(torch.tpu.Precision.HIGH):
  c = torch.matmul(a, b)
print(f"Computed HIGH-precision matmul with output shape: {tuple(c.shape)}")
```

### **분산 학습 및 토폴로지** { #distributed-training-and-topology }

TorchTPU는 `"tpu"` 디바이스의 기본 `torch.distributed` 백엔드로 `"tpu_dist"`(별칭
`"tpu"`)를 등록합니다.
[GKE에서 TorchTPU 시작하기](./get_started_on_gke.md) 및
[FSDP 권장사항](./case_studies_fsdp.md)을 참고하세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 설명 |
| :--- | :--- |
| `torch.tpu.topology_aware_mesh(mesh_shape, ...)` | `DeviceMesh("tpu", mesh=...)`에 사용할 형상이 `mesh_shape`인 토폴로지 인식 랭크 레이아웃 텐서를 계산합니다. |
| `torch.tpu.distributed.set_tpu_launch_env(...)` | 워커를 생성하기 전에 TPU 토폴로지 및 멀티 호스트 실행 환경 변수를 채웁니다. |
| `torch.tpu.distributed.spmd_safe(func)` | 모든 랭크가 `func`에서 동일한 TPU 연산을 실행함을 선언하여 eager 집합 통신 퓨전을 사용 설정하는 데코레이터입니다. |
| `torch.tpu.window_num_stripes` | 피어 투 피어 전송에 사용하는 동시 DMA 스트라이프의 최대 수를 가져오거나 설정합니다. |
| `torch.tpu.window_stripe_chunk_mb` | 피어 투 피어 전송에서 DMA 스트라이프당 최소 페이로드 크기(MB)를 가져오거나 설정합니다. |
<!-- mdformat on -->

### **TPU 커스텀 연산자(`torch.tpu.ops`) 및 Pallas(`torch.tpu.pallas`)** { #tpu-custom-operators-torchtpuops-and-pallas-torchtpupallas }

TorchTPU는 `torch.tpu.ops` 아래에 하드웨어 특화 연산자를, `torch.tpu.pallas`
아래에 커스텀 JAX/Pallas 커널 래퍼를 제공합니다.
[Pallas를 사용하여 커스텀 커널 추가](./custom_kernels.md)를 참고하세요.

#### **일반, 어텐션, MoE, 집합 통신 연산자** { #general-attention-moe-and-collective-operators }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 설명 |
| :--- | :--- |
| `torch.tpu.ops.max_pool{1,2,3}d(...)` | 인덱스 텐서를 구체화하지 않고 TPU에서 1D, 2D 또는 3D 최대 풀링을 적용합니다. |
| `torch.tpu.ops.max_pool{1,2,3}d_backward(...)` | 인덱스 텐서 없이 `max_pool{1,2,3}d`의 역전파를 계산합니다. |
| `torch.tpu.ops.optimization_barrier(inputs)` | XLA가 배리어를 넘어 연산을 이동, 퓨전 또는 중복 제거하지 못하게 합니다. |
| `torch.tpu.ops.splash_sdpa(q, k, v, ...)` | TPU Pallas Splash Attention 커널 기반의 스케일된 내적 어텐션입니다. |
| `torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes, ...)` | MoE(Mixture-of-Experts) 워크로드를 위한 그룹 행렬 곱셈입니다. |
| `torch.tpu.ops.ragged_dot_weight_grad(lhs, grad_output, group_sizes)` | `ragged_dot`의 가중치 그래디언트를 계산합니다. |
| `torch.tpu.ops.align_jagged_to_multiple(nt, ...)` | TPU 정렬을 위해 `torch.jagged` `NestedTensor` 버퍼를 지정한 배수로 패딩합니다. |
| `torch.tpu.ops.strip_jagged_padding(nt, ...)` | `align_jagged_to_multiple`이 추가한 후행 패딩을 제거합니다. |
| `torch.tpu.ops.all_gather_dim(input, dim, group=None)` | `group` 전체에서 `input`을 all-gather하고 차원 `dim`을 따라 연결합니다. |
| `torch.tpu.ops.reduce_scatter_dim(input, dim, group=None, ...)` | `group` 전체에서 `input`을 리듀스하고 차원 `dim`을 따라 청크를 분산합니다. |
| `torch.tpu.ops.ragged_all_to_all(...)` | 프로세스 그룹의 랭크 간에 가변 길이 텐서 슬라이스를 교환합니다. |
| `torch.tpu.ops.send(tensors, dst, tag)` / `torch.tpu.ops.recv(tensors, src, tag)` | 기본 TPU 프로세스 그룹에서 점대점 텐서 송수신을 수행합니다. |
| `torch.tpu.pallas.jax_op(name, fn=None, ...)` | JAX/Pallas 함수를 트레이싱 가능한 PyTorch 커스텀 연산자로 등록합니다. |
<!-- mdformat on -->

#### **SparseCore 임베딩 연산자** { #sparsecore-embedding-operators }

이 연산자에는 SparseCore가 있는 TPU가 필요합니다.
`torch.tpu.get_device_properties()`의 `sparse_cores_per_device` 속성은 디바이스당
SparseCore 수(SparseCore가 없는 TPU에서는 0)이며, `sparse_core_simd_width`는
SparseCore의 SIMD 너비입니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 설명 |
| :--- | :--- |
| `torch.tpu.ops.sparse_dense_matmul(...)` | TPU SparseCore에서 CSR 임베딩 조회와 집계를 수행합니다. |
| `torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(...)` / `_with_adagrad(...)` / `_with_adam(...)` | `sparse_dense_matmul`의 역전파 그래디언트를 계산하고 SparseCore에서 퓨전된 인플레이스 SGD, Adagrad 또는 Adam 테이블 업데이트를 적용합니다. |
| `torch.tpu.ops.sparse_dense_matmul_activation_unstack(...)` | 스택된 SparseCore 임베딩 활성화를 특성별 텐서로 분할합니다. |
| `torch.tpu.ops.sparse_dense_matmul_gradient_stack(...)` | 특성별 활성화 그래디언트를 SparseCore용 단일 텐서로 스택합니다. |
| `torch.tpu.ops.sparse_gather(...)` | TPU SparseCore에서 CSR 인덱스를 사용하여 2D 텐서의 행을 수집합니다. |
| `torch.tpu.ops.sparse_gather_backward(...)` | TPU SparseCore에서 `sparse_gather`의 그래디언트를 scatter-add합니다. |
| `torch.tpu.ops.sparse_iota(...)` | SparseCore에서 CSR 행렬의 0이 아닌 요소에 대한 행 내 위치 인덱스를 생성합니다. |
| `torch.tpu.ops.sparse_map_row(...)` | SparseCore에서 희소 CSR 행렬에 커스텀 MLIR 행 단위 연산을 매핑합니다. |
<!-- mdformat on -->

### **퓨전된 학습 단계 및 옵티마이저(`torch.tpu.optimizers`)** { #fused-training-step-and-optimizers-torchtpuoptimizers }

`torch.tpu.optimizers`는 순전파, 손실 계산, 역전파, 함수형 옵티마이저
단계를 단일 컴파일된 TPU 그래프로 트레이싱합니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 설명 |
| :--- | :--- |
| `torch.tpu.optimizers.SingleTraceTrainer(model, optimizer=None, ...)` | 함수형 파라미터(`params`)와 옵티마이저 상태(`param_group`)를 관리하고, `make_compiled_train_step(...)`으로 전체 학습 단계를 컴파일합니다. |
| `torch.tpu.optimizers.Optimizer(lr=..., weight_decay=...)` | 상태 비저장 함수형 옵티마이저의 기본 클래스입니다(`defaults`, `init_param_group`, `step` 제공). |
| `torch.tpu.optimizers.AdamW(...)` | 트레이싱 가능한 함수형 AdamW 옵티마이저입니다(`AdamW.ParamGroup`). |
| `torch.tpu.optimizers.Muon(...)` | 트레이싱 가능한 함수형 Muon 옵티마이저입니다(`Muon.ParamGroup`). |
| `torch.tpu.optimizers.Sgd(...)` | 트레이싱 가능한 함수형 SGD 옵티마이저입니다(`Sgd.ParamGroup`). |
<!-- mdformat on -->

### **프로파일링(`torch.tpu.profiler`)** { #profiling-torchtpuprofiler }

XProf 및 TensorBoard용 TPU 하드웨어 및 Python 트레이싱을 구성합니다.
[TorchTPU용 XProf](./performance_profiling_with_xprof.md)를 참고하세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 설명 |
| :--- | :--- |
| `torch.tpu.profiler.TpuProfilerConfig(...)` | XProf용 호스트, TPU 디바이스, Python 트레이싱을 구성하기 위해 `torch.profiler.profile(..., experimental_config=...)`에 전달하는 실험용 구성입니다. |
<!-- mdformat on -->

## **환경 변수** { #environment-variables }

TorchTPU의 동작은 XLA 컴파일러, TPU 드라이버(`libtpu`), PyTorch 자체 디버그
하위 시스템을 구성하는 여러 환경 변수로 제어합니다.

> **중요: 프로세스 전역 초기화** TorchTPU 환경 변수는 백엔드가 초기화되는
> 순간(일반적으로 `import torch` 또는 `torch.device("tpu")` 중) **고정**됩니다.

### **디바이스 가시성, 런타임, 분산 실행** { #device-visibility-runtime-and-distributed-launch }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 환경 변수 | 설명 |
| :--- | :--- |
| `TPU_VISIBLE_DEVICES` | 현재 프로세스에 표시되는 쉼표로 구분된 TPU 칩 인덱스입니다(`CUDA_VISIBLE_DEVICES`와 유사). |
| `RANK` | 분산 작업에서 현재 프로세스의 전역 랭크입니다. |
| `LOCAL_RANK` | 현재 프로세스의 호스트 로컬 랭크입니다. |
| `WORLD_SIZE` | 분산 작업의 총 프로세스 수입니다. |
| `NNODES` | 멀티 호스트 작업의 총 호스트 노드 수입니다. |
| `NODE_RANK` | 멀티 호스트 작업에서 현재 호스트 노드의 랭크입니다. |
| `MASTER_ADDR` | `torch.distributed`의 랭크 0 코디네이터 노드 주소입니다. |
| `MASTER_PORT` | `torch.distributed` 랑데부에 사용하는 `MASTER_ADDR`의 포트입니다. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `torch_tpu` 백엔드의 자동 등록을 사용 중지하려면 `"0"`으로 설정합니다. |
| `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS` | 사용 설정하면 PyTorch 코어 외부에서 내부 `torch.tpu` 디바이스 API(`torch.accelerator` 대응 API가 있는 API)를 호출할 때 `RuntimeError`를 발생시키는 대신 일회성 경고를 표시합니다. 임시 해결책으로만 사용하세요. `torch.accelerator.*`로 마이그레이션하는 것이 좋으며, 이 변수는 향후 릴리스에서 삭제될 가능성이 높습니다. |
| `LIBTPU_INIT_ARGS` | `libtpu` 및 PJRT 런타임에 전달되는 공백으로 구분된 초기화 플래그입니다. |
| `TPU_PREMAPPED_BUFFER_SIZE` | PJRT 클라이언트의 사전 매핑된 호스트-디바이스 메모리 버퍼 크기(바이트)입니다. |
| `TMPDIR` | 런타임과 컴파일러가 사용하는 임시 디렉터리입니다. |
<!-- mdformat on -->

### **하드웨어 토폴로지 및 멀티 호스트 조정** { #hardware-topology-and-multi-host-coordination }

이 변수는 TPU 하드웨어 토폴로지와 프로세스 네트워킹을 지정합니다. 대부분의 경우
`torch.tpu.distributed.set_tpu_launch_env()`, TorchTPU 런처 또는 Google Cloud
환경이 자동으로 설정합니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 환경 변수 | 설명 |
| :--- | :--- |
| `TORCH_TPU_TOPOLOGY` | TPU 슬라이스의 물리적 토폴로지입니다(예: `2x2x1`, `2x2x4`). |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | TPU 슬라이스 빌더 서비스의 쉼표로 구분된 `host:port` 워커 주소입니다. |
| `TORCH_TPU_DEVICES_PER_HOST` | 슬라이스 빌더 주소를 도출할 때 사용하는 호스트당 TPU 칩 수입니다. |
| `TORCH_TPU_BASE_PORT` | 슬라이스 빌더 주소를 도출할 때 사용하는 시작 포트 번호입니다. |
| `TORCH_TPU_SPMD_HANDSHAKE_STAGE` | 랭크 간에 집합 통신 그래프가 일치하는지 검증하는 단계입니다. |
| `TORCHCOMM_RANK` | `torchcomm`의 프로세스 랭크입니다(`RANK`보다 우선). |
| `TORCHCOMM_SIZE` | `torchcomm`의 월드 크기입니다(`WORLD_SIZE`보다 우선). |
<!-- mdformat on -->

### **실행 모드, 컴파일, 캐싱** { #execution-mode-compilation-and-caching }

이 변수는 PyTorch 연산과 그래프가 디스패치되고, 로어링되고, XLA로
최적화되고, 실행 및 랭크 간에 캐시되는 방식을 제어합니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 환경 변수 | 설명 |
| :--- | :--- |
| `TORCH_TPU_DEFER_AND_FUSE` | 사용 설정하면(`"1"`/`"true"`) 기본 eager 모드를 `EagerMode.DEFER_AND_FUSE`로 설정합니다. |
| `TORCH_TPU_LAUNCH_BLOCKING` | 사용 설정하면(`"1"`/`"true"`) 기본 eager 모드를 `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING`으로 설정합니다. |
| `TORCH_TPU_TIER2_COMPILATION_CACHE` | 호스트 로컬 컴파일 캐시에 사용할 `/dev/shm/torch_tpu_cache/` 아래의 디렉터리 이름입니다(또는 `"disabled"`). |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | 호스트와 실행 간에 공유되는 영구 Tier-3 컴파일 캐시의 경로입니다(예: `gs://...` 또는 로컬 경로). |
| `XLA_FLAGS` | 공백으로 구분된 XLA 컴파일러 플래그입니다(예: `--xla_dump_hlo_as_text --xla_dump_to=...`). |
<!-- mdformat on -->

### **디버깅, 안전성, 프로파일링, 트레이싱** { #debugging-safety-profiling-and-tracing }

이 변수는 TorchTPU의 디버깅 컨텍스트, 런타임 안전성 검사, 프로파일러 출력을
제어합니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 환경 변수 | 설명 |
| :--- | :--- |
| `TORCH_SHOW_CPP_STACKTRACES` | `"1"`로 설정하면 Python `RuntimeError` 예외에 C++ 백트레이스를 추가합니다. |
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` | `"1"`로 설정하면 TPU에서 런타임 디버그 및 경계 검사를 사용 설정합니다. |
| `TORCH_TPU_PROFILER_OUTPUT_DIR` | `TpuProfilerConfig.run_dir`이 설정되지 않은 경우 XProf 트레이스 파일에 사용하는 대체 출력 디렉터리입니다. |
| `TORCH_TPU_XPROF_SESSION_ID` | 멀티 호스트 워커 간에 공유되는 XProf 프로파일링 세션 ID입니다. |
| `TORCH_TRACE` | 구조화된 컴파일러 및 디스패처 트레이스 로그(`tlparse`)의 출력 디렉터리입니다. |
<!-- mdformat on -->

### **환경 변수 구성 스니펫** { #environment-variable-configuration-snippets }

#### **1. C++ 스택 트레이스(`TORCH_SHOW_CPP_STACKTRACES`)** { #1-c-stacktraces-torch_show_cpp_stacktraces }

`torch`를 가져오기 전에 `TORCH_SHOW_CPP_STACKTRACES=1`을 설정하면 C++ 백엔드나
XLA 깊은 곳에서 발생하는 오류를 훨씬 쉽게 진단할 수 있습니다. 단순한 Python
`RuntimeError` 대신 정확한 실행 프레임을 보여 주는 자세한 `C++
CapturedTraceback`을 받습니다.

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

#### **2. 내부 디버그 검사 및 경계 검증(`TORCH_TPU_ENABLE_DEBUG_CHECKS`)** { #2-internal-debug-checks-boundary-validation-torch_tpu_enable_debug_checks }

기본적으로 TPU는 속도를 위해 경계 검사를 건너뜁니다.
`TORCH_TPU_ENABLE_DEBUG_CHECKS=1`을 사용 설정하면 범위를 벗어난 인덱스(예:
`torch.take`)과 관련된 미묘한 버그를 잡을 수 있지만, 추가 CPU/TPU 동기화로 인해
실행 속도가 느려집니다. 그렇지 않으면 아무 표시 없이 실패하거나 데이터 손상을
일으킬 연산이 `take(): expected indices to be
in range` 같은 명시적 예외를 발생시킵니다.

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

#### **3. XLA IR / HLO 덤프(`XLA_FLAGS`)** { #3-dumping-xla-ir-hlo-xla_flags }

`XLA_FLAGS`를 설정하면 컴파일러가 HLO(High-Level Optimizer) 텍스트를 디렉터리에
덤프하도록 강제할 수 있습니다. 덤프 디렉터리(예: `/tmp/hlo_dump`)에는 다음
파일이 포함됩니다.

*   **`*.after_optimizations.txt`**: 하드웨어 수준 연산을 보여 주는 최종 HLO
    그래프입니다.
*   **`*.flagfile`**: 활성 XLA 컴파일러 플래그의 전체 목록입니다
    (`xla_optimization_level` 확인).
*   **`*tpu_comp_env.txt`**: 컴파일 중에 사용된 내부 TPU 환경 설정입니다.

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

#### **4. 다중 계층 컴파일 캐싱(`TORCH_TPU_TIER2_COMPILATION_CACHE` 및 `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** { #4-multi-tiered-compilation-caching-torch_tpu_tier2_compilation_cache-torch_tpu_tier3_compilation_cache_root }

TorchTPU는 다중 계층 캐싱을 사용하여 실행 및 랭크 간의 중복 컴파일을 방지합니다.

*   **Tier 2(`TORCH_TPU_TIER2_COMPILATION_CACHE`)**는 호스트 내 공유 메모리에
    로컬로 저장됩니다. 디렉터리 **이름**(예: `"tpu_tier2_cache"`)을 받으며,
    백엔드가 이를 자동으로 `/dev/shm/torch_tpu_cache/<name>` 안에 배치합니다.
*   **Tier 3(`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)**은 호스트와 세션 간에
    공유되는 영구 컴파일 캐시입니다(일반적으로 `gs://` 버킷 URI 또는 공유 파일
    시스템 경로).

컴파일된 바이너리 아티팩트(`*.bin` 실행 파일)는 캐시 디렉터리에 비동기적으로
저장되고 이후 실행에서 재사용되므로 JIT 컴파일 지연 시간이 크게 줄어듭니다.

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
