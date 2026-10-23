# 동적 형상 { #dynamic-shapes }

<!-- Link to colab and to generated notebook -->

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/dynamic_shapes.ipynb)

`torch.compile(model)`로 모델을 컴파일하면 TorchTPU는 계산을 StableHLO로
로어링하고 XLA 컴파일러로 컴파일합니다([TorchTPU 작동 방식](./how_torchtpu_works.md)
참조). XLA 컴파일러는 전통적으로 **정적으로 알려진 텐서 형상**을 중심으로
설계되어 있어, HBM(High Bandwidth Memory) 버퍼를 미리 배치하고, 연산자
파이프라인을 퓨전하고, TPU의 시스톨릭 배열(MXU)에 맞게 행렬 곱셈을 타일링할 수
있습니다. `torch.compile` 워크로드에서 입력 텐서 차원이 호출마다 달라질 때
정확한 형상마다 컴파일하면 Dynamo의 형상 가드가 무효화되고, 새로운 형상마다
비용이 큰 전체 XLA 재컴파일이 발생합니다.

계속되는 재컴파일 없이 **동적 형상**을 지원하기 위해 TorchTPU는 **제한된
동적성(Bounded Dynamism)**을 사용합니다. 제한된 동적성에서는 동적 차원의 컴파일
시간 상한이 고정되고, 실제 런타임 차원 크기는 XLA가 동적으로 추적합니다.

## TorchTPU가 동적 형상을 처리하는 방식 { #how-dynamic-shapes-are-handled-by-torchtpu }

기본 동적성, 즉 `torch.compile(model, dynamic=None)`으로(입력 형상이 호출마다
달라지는 경우) 모델이나 함수를 컴파일하거나 `torch._dynamo.mark_dynamic`으로
동적 차원을 명시적으로 표시하면, Dynamo는 계산을 트레이싱하여 변하는 차원을 고정
상수 대신 기호 정수 변수(`torch.SymInt`)로 표현한 FX 그래프를 생성합니다.
TorchTPU는 이러한 기호 차원을 다음과 같이 처리합니다.

### 컴파일 { #compilation }

*   TorchTPU는 트레이싱된 FX 그래프를 가로채서 각 기호 차원의 전체
    `[lower_bound, upper_bound]` 범위를 결정하고
    (`mark_dynamic(..., min=..., max=...)`에서 가져오거나 Dynamo가 초기 런타임
    크기 `S`에서 추론),
    그 범위를 `bucket_size`(기본값 `128`, TPU의 128-레인 타일 차원과 일치)의
    배수에 맞춘 연속 버킷으로 최대 `num_dynamic_buckets`개(기본값 `1`, 최대
    `8`)까지 나눕니다.
    *   첫 번째 버킷은 `lower_bound`부터 `S` 이상인 다음 `128`의 배수(`U_1`)까지,
        두 번째 버킷은 `U_1 + 1`부터 다음 `128`의 배수(`U_2`)까지를 포함하며,
        이후도 같은 방식입니다(예: `[lower_bound, 128]`, `[129, 256]`, ...).
    *   TorchTPU는 모든 버킷의 전체 범위에 걸친 단일 범위 가드를 Dynamo에
        등록하므로, Dynamo는 차원이 전체 `upper_bound`를 넘어설 때만 재컴파일을
        트리거합니다.
*   각 버킷에 대해 TorchTPU는 기호 차원을 버킷의 정적 상한 `U`로 대체하고,
    실제 런타임 차원 크기를 입력으로 전달하며, 모든 연산에서 정적 상한 `U`와
    활성 런타임 크기를 모두 추적하는 제한된 텐서 타입으로 그래프를 StableHLO로
    로어링합니다.
*   그 결과로 생성된 모듈을 XLA가 구성된 각 버킷 상한 `U`에 대해 컴파일합니다.

### 실행 { #execution }

*   런타임에 TorchTPU는 동적 차원의 실제 크기를 기준으로 해당하는 컴파일된
    버킷 실행 파일을 선택합니다.

*   컴파일된 모듈을 실행할 때 XLA는 동적 입력 텐서를 버킷의 정적 상한 `U`까지
    패딩하고, 활성 런타임 크기를 넘는 패딩 영역의 값을 대수적 항등값으로
    재설정하여 패딩된 요소가 출력 결과에 영향을 주지 않도록 합니다.

*   동적 출력 텐서의 경우 패딩이 버퍼 끝에 남아 있으므로, TorchTPU는 텐서를
    물리적으로 슬라이스하지 않고 가벼운 형상 메타데이터 업데이트만으로 버퍼를
    PyTorch에 반환할 수 있습니다.

## 주요 사용 사례: 자기회귀 디코딩 { #motivating-use-case-autoregressive-decoding }

제한된 동적성이 필요한 주요 워크로드는 동적으로 증가하는 KV 캐시를 사용하는
**자기회귀 LLM 디코딩**입니다. 표준 디코딩 루프에서는 각 단계마다 새 토큰의
key 및 value 상태를 `torch.cat`으로 시퀀스 차원을 따라 캐시된 기록에
추가합니다.

*   **프리필:** 입력 프롬프트의 형상은 `[B, S]`(배치 크기 `B`, 프롬프트
    시퀀스 길이 `S`)이며, 형상이 `[B, H_kv, S, D]`(`H_kv`개의 key/value 헤드,
    헤드 차원 `D`)인 KV 캐시를 생성합니다.
*   **디코딩 단계 1:** 입력 토큰의 형상은 `[B, 1]`이며, 캐시와 연결되어
    형상 `[B, H_kv, S + 1, D]`를 생성합니다.
*   **디코딩 단계 `t`:** KV 캐시 시퀀스 길이가 `S + t`로 늘어납니다.

정적 형상 컴파일(`dynamic=False`)에서는 Dynamo가 KV 캐시의 정확한 시퀀스 길이에
가드를 설정합니다. 디코딩 단계마다 텐서 형상이 바뀌어 가드가 무효화되고, **생성되는
토큰마다 전체 XLA 재컴파일**이 트리거됩니다. 정적 버퍼를 미리 할당하고 인플레이스
변경을 수행하도록 모델을 다시 작성하지 않으면, 밀리초 단위의 디코딩 단계가 수 초
단위의 컴파일로 바뀝니다.

**제한된 동적성**은 모델을 다시 작성하지 않고도 이 문제를 해결합니다. 디코딩
단계를 거치며 KV 캐시가 커져도(`S + 1`, `S + 2`, ...) 컴파일된 버킷(`<= 128`,
`<= 256`, ...) 안의 모든 형상은 같은 XLA 실행 파일을 재사용합니다.

전체 LLM 디코딩 루프에 제한된 동적성을 적용하는 실습 안내(정적 KV 캐시 사전
할당과의 비교 포함)는
[엔드 투 엔드 LLM 추론](./case_studies_muse_glimmer_30b_inference.md)을
참조하세요.

## 예: 텐서가 커지는 자기회귀 루프 { #example-autoregressive-loop-with-a-growing-tensor }

정적 컴파일(`dynamic=False`)과 기본 동적성(`dynamic=None`)을 사용하는 제한된
동적성이 실제로 어떻게 동작하는지 보려면, 각 단계에서 `torch.cat`으로 시퀀스
차원(`dim=2`)을 따라 새 토큰을 추가하는 최소한의 자기회귀 루프를 살펴보세요.

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

### 1. `dynamic=False`로 실행 { #1-running-with-dynamicfalse }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()
torch._dynamo.config.recompile_limit = 64

step_static = torch.compile(decode_step, backend="tpu", dynamic=False)
run_decode_loop(step_static, prefix="[dynamic=False] ")
```

샘플 출력:

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

**결과 설명:** `dynamic=False`를 사용하면 Dynamo는 `kv_cache`의 정확한 시퀀스
길이(`16`, `17`, `18`, ...)에 가드를 설정합니다. `torch.cat`이 반복할 때마다
`dim=2`를 `1`씩 늘리므로 호출할 때마다 형상 가드가 실패하고, 단계마다 전체 XLA
재컴파일이 트리거됩니다.

### 2. `dynamic=None`으로 실행(제한된 동적성) { #2-running-with-dynamicnone-bounded-dynamism }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()

step_dynamic = torch.compile(decode_step, backend="tpu", dynamic=None)
run_decode_loop(step_dynamic, prefix="[dynamic=None] ")
```

샘플 출력:

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

**결과 설명:** `dynamic=None`을 사용하면 처음 3번의 반복에서만 컴파일 오버헤드가
발생하고, 4단계부터는 재컴파일 없이 실행 시간이 단계당 약 0.4ms로
줄어듭니다(약 100배 빠름).

*   **1단계(`seq_len=16 -> 17`):** Dynamo가 입력을 처음 확인하고
    `seq_len=16`에 대한 정적 실행 파일을 컴파일합니다.
*   **2단계(`seq_len=17 -> 18`):** `dim=2`가 `16`에서 `17`로 바뀌었으므로
    Dynamo가 `dim=2`를 자동으로 동적(`SymInt`)으로 표시하고, TorchTPU는 초기
    입력 텐서 레이아웃을 사용하여 `[17, 128]` 버킷에 대한 제한된 동적 실행
    파일을 컴파일합니다.
*   **3단계(`seq_len=18 -> 19`):** 2단계에서 생성된 제한된 동적 출력 버퍼는
    XLA의 패딩된 출력 레이아웃을 가지므로, TorchTPU는 그 정상 상태 XLA
    레이아웃에 맞춰 `[17,
    128]` 버킷을 한 번 더 컴파일합니다.
*   **4~10단계(`seq_len=19 -> 26`):** 시퀀스 길이(`<= 128` 버킷 내)와 XLA 버퍼
    레이아웃이 모두 캐시된 실행 파일과 일치하므로, 이후 모든 단계는 재컴파일
    없이 미리 컴파일된 XLA 실행 파일을 바로 디스패치합니다.

> **Tip:** 어떤 차원이 달라질지 미리 알고 있다면 첫 단계 전에
> `torch._dynamo.mark_dynamic(kv_cache, 2, min=16, max=128)`을 호출할 수
> 있습니다. 이렇게 하면 Dynamo가 1단계부터 즉시 `dim=2`를 동적으로 트레이싱하므로,
> 1단계의 초기 정적 컴파일을 건너뛰고 명시적인 `[min, max]` 경계를 지정할 수
> 있습니다.

## 현재 제한사항 { #current-limitations }

TorchTPU의 제한된 동적성은 활발히 발전하고 있으며, 현재는 **자기회귀 KV 캐시**
사용 사례를 중심으로 최적화되어 있습니다. 모델에 적용할 때는 다음 제한사항에
유의하세요.

*   **버킷 한도와 제한 없는 동적성:** 트레이스당 최대 **8개의
    버킷**(`num_dynamic_buckets <= 8`)이 컴파일됩니다. 제한 없는 동적성은 아직
    지원되지 않으므로, 동적 차원이 컴파일된 가장 큰 버킷의 상한을 넘어서면
    재컴파일이 트리거됩니다.
*   **Dynamo 형상 제약조건:** 버킷 상한은 활성 Dynamo 형상 제약조건(예:
    `reshape` 또는 `view`의 나누어떨어짐 요구사항)을 충족해야 합니다. 그렇지
    않으면 TorchTPU는 상한이 `2 * S`(초기 런타임 크기 `S`의 두 배)인 단일
    버킷으로 대체합니다.
*   **버킷 상한까지의 계산 및 HBM 오버헤드:** 실행은 버킷의 정적 상한 `U`까지
    패딩된 버퍼에서 이루어지므로, HBM 사용량과 FLOP은 실제 런타임 크기가 아닌
    `U`에 비례합니다. 실제 크기가 `U`보다 훨씬 작을 때 매우 큰 `bucket_size`
    또는 `mark_dynamic(..., max=U)`를 설정하면 `U`만큼의 계산 및 메모리 비용을
    치르게 됩니다.
*   **XLA 레이아웃 변경:** 텐서의 형상이 이미 컴파일된 버킷 안에 있더라도
    입력 버퍼의 기본 XLA 물리 레이아웃이 바뀌면 버킷 재컴파일이 트리거됩니다.
*   **영구 컴파일 캐시(`AOTAutogradCache`) 없음:** 제한된 동적 실행 파일은 아직
    `AOTAutogradCache`에 직렬화되지 않으므로, 새 프로세스가 시작될 때마다 다시
    컴파일됩니다.
*   **연산자 지원 범위:** 아직 모든 PyTorch 연산자가 제한된 동적 형상을
    지원하지는 않습니다.
*   **모델 입력으로서의 뷰:** 텐서 뷰를 컴파일된 모델의 입력으로 직접 전달하는
    기능은 제한적으로 지원됩니다.
*   **데이터 종속 동적성:** 출력 형상이 런타임 텐서 값에 따라 달라지는
    연산(예: `torch.nonzero`)은 지원되지 않습니다.
*   **커스텀 커널:** 제한된 동적 형상은 커스텀 TPU 커널(예:
    `torch.tpu.pallas.jax_op`를 통해 등록된 Pallas 커널)에서 지원되지 않습니다.
*   **dynamic=True:** torch.compile()의 *dynamic=True* 모드는 지원되지 않습니다.
