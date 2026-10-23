# TorchTPU에서 수치 정밀도 관리 { #manage-numerical-precision-in-torchtpu }

Google TPU는 처리량과 수치 정확도의 균형을 맞추기 위해 부동 소수점 연산을 두
하드웨어 유닛에 나누어 처리합니다.

*   **행렬 곱셈 유닛(MXU):** 축약 연산(예: `torch.matmul`, `torch.mm`,
    `torch.bmm`, 컨볼루션)을 네이티브 `bfloat16` 입력(8비트 유효숫자)과 32비트
    `float32` 누적으로 실행합니다.
*   **벡터 실행 유닛(VPU):** 비축약 연산(예: 요소별 산술, 활성화 함수, 레이어
    정규화, softmax)을 완전한 IEEE-754 `float32`(24비트 유효숫자)로 실행합니다.

이 가이드에서는 `float32` 행렬 곱셈 정밀도 모드를 구성하고, CPU와 TPU 간 수치
일치성을 확인하고, 출력 누산기 타입과 초과 정밀도를 제어하고, 코드에서 비트 단위
MXU 동작을 검증하는 방법을 설명합니다.

### CUDA와 TPU의 PyTorch 정밀도 비교 { #compare-pytorch-precision-on-cuda-and-tpu }

TorchTPU는 표준 PyTorch 정밀도 API(`torch.set_float32_matmul_precision` 및
`torch.get_float32_matmul_precision`)와 함께 TPU 전용 제어(`torch.tpu.Precision`
및 `torch.tpu.precision`)를 지원합니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 항목          | CUDA 기반 PyTorch                      | TPU 기반 PyTorch(TorchTPU)              |
| :----------------- | :----------------------------------- | :------------------------------------- |
| **기본 정밀도 모드** | `"highest"`(순수 `float32`) | `"medium"`(`Precision.DEFAULT`)       |
| **빠른 수학 연산 가속기 프리미티브** | Tensor Core(`TF32`) | 시스톨릭 MXU(`bfloat16`)             |
| **빠른 수학 연산 비트** | 부호 1 + 지수 8 + 가수 10비트(19비트 `TF32`) | 부호 1 + 지수 8 + 가수 7비트(16비트 `bfloat16`) |
| **누적 정밀도** | `float32`(`"highest"`, `"high"`), `TF32`(`"medium"`) | 모든 모드에서 `float32` |
| **더 높은 정밀도 구현 방식** | 전용 하드웨어 `float32`/`TF32` 유닛 | 다중 패스 분해(MXU 3회 또는 6회 패스) |
| **`high`/`highest`의 성능 영향** | 기준 속도 또는 약간의 속도 저하(~1–2배) | `"high"`는 MXU 계산 비용 ~3배, `"highest"`는 ~6배 |
| **범위 지정 API** | 전역 `torch.set_float32_matmul_precision` | 전역 <br> `torch.set_float32_matmul_precision` + 범위 지정 `torch.tpu.precision` |
<!-- mdformat on -->

## 정밀도 모드 지정 { #specify-precision-modes }

MXU에서 `float32` 텐서를 곱할 때 TorchTPU는 PyTorch의 정밀도 수준을
`torch.tpu.Precision` enum에 매핑합니다.

*   `"medium"` / `Precision.DEFAULT`(1회 패스): **기본값입니다.** `float32`
    입력을 `bfloat16`으로 반올림하고(유효숫자의 상위 8비트를 곱함) `float32`로
    누적합니다. 거의 모든 학습 및 추론 워크로드에 권장됩니다.
*   `"high"` / `Precision.HIGH`(3회 패스): `float32` 입력을 상위 및 하위
    `bfloat16` 구성요소로 나누고 세 개의 교차항을 계산하며(유효숫자의 처음
    16비트를 곱함), MXU 계산 비용은 약 3배입니다.
*   `"highest"` / `Precision.HIGHEST`(6회 패스): `float32` 입력을 세 개의
    `bfloat16` 구성요소로 나누고 여섯 개의 교차항을 계산하여(유효숫자 24비트
    전체를 곱함) 완전한 IEEE-754 `float32` 정확도에 근접하며, MXU 계산 비용은 약
    6배입니다.

행렬 곱셈 정밀도는 두 가지 방법으로 구성할 수 있습니다.

*   **전역 구성:** `torch.set_float32_matmul_precision("medium"
    | "high" | "highest")`를 호출하여 프로세스의 기본 모드를 설정합니다.
*   **범위 지정 컨텍스트 관리자:** `with torch.tpu.precision(mode):`를 사용하여
    수치적으로 민감한 레이어나 하위 모듈에만 더 높은 정밀도를 적용합니다.
*   **성능 경고:** `"high"` 또는 `"highest"`를 선택하면 일회성
    경고(`TORCH_WARN_ONCE`)가 표시됩니다. 정밀도 관련 수렴 문제가 관찰되지 않는
    한 기본 `"medium"` 모드를 유지하세요.

> **Note:** `torch.tpu.precision`에 들어가거나 나오면 `torch.compile` 그래프
> 중단이 발생합니다. `torch.compile(..., fullgraph=True)`를 사용할 때는
> `torch.set_float32_matmul_precision(...)`로 컴파일된 함수 외부에서 정밀도를
> 설정하거나 외부 호출을 `with
> torch.tpu.precision(...):`으로 래핑하세요.

`torch.tpu.precision`을 사용하여 세 모드의 정상 상태 실행 시간을 비교하세요.

```python
import time
import torch

device = torch.device("tpu")
precision = torch.tpu.precision  # Context manager
Precision = torch.tpu.Precision  # Enum class

print(f"Default matmul precision: {torch.get_float32_matmul_precision()}")

size = (8, 2048, 2048)
a = torch.randn(size, dtype=torch.float32, device=device)
b = torch.randn(size, dtype=torch.float32, device=device)

def run_matmul():
  res = torch.matmul(a, b)
  torch.accelerator.synchronize()
  return res

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    # Warm up to compile the graph before timing steady-state execution
    run_matmul()
    start = time.perf_counter()
    run_matmul()
    print(f"{mode} time: {time.perf_counter() - start:.4f}s")
```

### 정밀도 모드별 정확도 비교 { #compare-accuracy-across-precision-modes }

같은 입력에 대해 TPU 출력을 CPU `float32` 참조와 비교하여 각 정밀도 모드가 수치
오차를 얼마나 줄이는지 확인하세요.

```python
# CPU float32 reference result
c_ref = torch.matmul(a.cpu(), b.cpu())

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    res = torch.matmul(a, b)
    diff = res.cpu() - c_ref
    print(f"Mode: {mode}")
    print(f"  Max Absolute Error: {diff.abs().max():.6e}")
    print(
        "  Mean Relative Error: "
        f" {(diff.abs() / (c_ref.abs() + 1e-8)).mean():.6e}\n"
    )
```

## 수치 일치성 확인 { #verify-numerical-parity }

모델을 TPU로 이식하거나 수렴 차이를 디버깅할 때는 동기화된 가중치와 동일한 입력으로
같은 레이어나 하위 모듈을 CPU와 TPU에서 실행한 다음
`torch.testing.assert_close`로 출력을 비교하여 수치 일치성을 확인하세요.

```python
import torch.nn as nn

# 1. Set up identical models in bfloat16 for a fair comparison
model_cpu = nn.Linear(10, 10).cpu().to(torch.bfloat16)
model_tpu = nn.Linear(10, 10).to(device).to(torch.bfloat16)

# 2. Synchronize weights (mandatory for parity checks)
model_tpu.load_state_dict(model_cpu.state_dict())

# 3. Create identical input data for both devices
input_data = torch.randn(5, 10, dtype=torch.bfloat16)
input_tpu = input_data.to(device)

# 4. Run the forward pass
out_cpu = model_cpu(input_data)
out_tpu = model_tpu(input_tpu)

# 5. Compare results
# Note: TorchTPU's bfloat16 accumulation and fusion differences can exceed
# default tolerances, generating the `assert_close` diagnostic report below.
try:
  torch.testing.assert_close(out_tpu.cpu(), out_cpu)
  print("Parity check passed!")
except AssertionError as e:
  print("Parity check failed (expected for hardware precision drift):\n")
  print(e)
```

### 상세 불일치 해석 { #interpret-detailed-mismatches }

텐서가 허용 오차를 넘으면 `torch.testing.assert_close`가 요약을 출력합니다. 이
요약은 일반적인 부동 소수점 편차와 로직 버그를 구분하는 데 도움이 됩니다.

```text
Tensor-likes are not close!

Mismatched elements: 1 / 50 (2.0%)
Greatest absolute difference: 0.006596 at index (3, 9) (up to 1e-05 allowed)
Greatest relative difference: 0.165375 at index (3, 9) (up to 0.016 allowed)
```

*   **Mismatched elements:** 절대 차이가 작은 요소의 비율이 높다면 일반적으로
    버그가 아니라 컴파일러 퓨전으로 인한 표준 반올림 또는 누적 순서 차이를
    나타냅니다.
*   **Greatest absolute / relative difference:** 가장 큰 오차와 그 텐서
    인덱스를 보고하므로 이상값을 조사하거나 적절한 `atol` 및 `rtol` 임곗값을
    선택할 수 있습니다.

### 일치성 권장사항 { #best-practices-for-parity }

*   **두 디바이스에서 데이터 타입을 맞추세요:** TPU의 `bfloat16`은 CPU의
    `bfloat16`과, TPU의 `float32`(`Precision.HIGHEST` 사용)는 CPU의 `float32`와
    비교하세요. `bfloat16`을 `float32`와 비교하면 하드웨어 편차가 아니라 양자화
    오차를 측정하게 됩니다.
*   **활성화뿐 아니라 그래디언트도 확인하세요:** `loss.backward()`를 호출한 후
    같은 패턴으로 `model_tpu.weight.grad.cpu()`를 `model_cpu.weight.grad`와
    비교하세요.
*   **표준 허용 오차에서 시작하세요:**
    [PyTorch 테스트 문서](https://docs.pytorch.org/docs/stable/testing.html)의
    기본 허용 오차를 기준으로 사용하고, 낮은 정밀도 누적으로 인해 예상되는
    경우에만 `atol` 또는 `rtol`을 완화하세요.

## 출력 정밀도 지정 { #specify-output-precision }

`torch.mm` 또는 `torch.bmm`으로 `bfloat16` 텐서를 곱할 때 MXU는 곱을 `float32`로
누적한 후 최종 출력을 다시 `bfloat16`으로 반올림합니다. 다운캐스트 없이
`float32` 누산기를 바로 반환하려면 `out_dtype=torch.float32`를 전달하세요.

```python
a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device=device)
# Using out_dtype to force float32 output accumulator precision
res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
print(f"Input dtype: {a_bmm.dtype}")
print(f"Output dtype: {res_bmm.dtype}")
```

## 초과 정밀도 제어 { #control-excess-precision }

TPU MXU는 `bfloat16` 행렬 곱셈을 `float32`로 누적하므로, XLA 컴파일러는 중간
값을 단계 사이에서 `bfloat16`으로 반올림하지 않고 `float32`로 유지하여 퓨전된
연산을 최적화할 수 있습니다. 이렇게 추가 중간 비트를 유지하는 것을 *초과
정밀도(excess precision)*라고 합니다.

이 동작은 `torch.tpu.allow_excess_precision`(기본 `--xla_allow_excess_precision`
컴파일러 플래그를 설정하며 기본값은 `False`)으로 전역 구성할 수 있습니다.

*   **초과 정밀도 사용 중지(`False`, 기본값):**
    `torch.tpu.allow_excess_precision = False`를 설정하면 컴파일러가 중간
    `bfloat16` 다운캐스트를 유지하므로 퓨전된 연산이 eager `bfloat16` 반올림
    동작과 일치합니다.
*   **초과 정밀도 허용(`True`):** `torch.tpu.allow_excess_precision =
    True`를 설정하면 컴파일러가 중간 다운캐스트를 생략하고 퓨전된 중간 결과를
    `float32`로 유지할 수 있습니다.

> **Important:** TPU 연산을 실행하거나 컴파일하기 전에 스크립트의 맨 처음에
> `torch.tpu.allow_excess_precision`을 설정하세요. XLA 런타임은 첫 번째 컴파일이
> 시작될 때 이 플래그를 잠그며, 이후 다른 값으로 변경하려고 하면 `RuntimeError`가
> 발생합니다.

## TPU MXU 정밀도 이해 { #understand-tpu-mxu-precision }

TPU MXU가 부동 소수점 숫자를 처리하는 방식을 이해하면 초과 정밀도와 다중 패스
`float32` 모드가 왜 그렇게 작동하는지 알 수 있습니다.

*   **범위와 정밀도(`bfloat16`):**
    [`bfloat16`에 관한 Google Cloud 연구](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)에
    따르면, 신경망은 `float32`의 넓은 동적 범위(노름 및 그래디언트 같은 연산에
    중요)를 유지하기만 하면 가수 정밀도를 줄여도 모델 정확도를 잃지 않고 학습할 수
    있습니다. `float32`는 23개의 가수 비트(암시적 선행 비트를 포함하면 24비트
    유효숫자 정밀도)를 가지는 반면, `bfloat16`은 같은 8비트 지수 범위를
    유지하면서 가수 비트는 7개(8비트 유효숫자 정밀도, 머신 엡실론 `2**-7` 또는
    ~0.78%)만 가집니다.
*   **교환 형식과 산술 형식:** 부동 소수점 시스템은 **교환 형식**(숫자가
    메모리에 저장되는 방식)과 **산술 형식**(표준 ALU의 추가 guard, round, sticky
    비트 3개처럼 하드웨어가 중간 연산을 계산하고 반올림하는 방식)을 구분합니다.
    `bfloat16`은 둘 중 어느 것이든 의미할 수 있습니다.
*   **축약 차원을 따른 누적:** 형상이 `[I, J] @ [J, K]`인 두 행렬을 곱하면
    형상이 `[1,
    J] @ [J, 1]`인 내적의 격자가 계산됩니다. 각 내적은 **축약 차원** `J`를
    따라 `J`개의 요소별 곱을 합산합니다. `bfloat16`의 머신 엡실론은
    `2**-7`(~`0.008`)이므로 `bfloat16`으로 `J`개의 곱을 누적하면 큰 반올림
    오차가 발생합니다. 대신 MXU는 `bfloat16` 입력(TPU v7x 같은 최신
    세대에서는 `fp8`도)을 받아 중간 합을 `float32`로 누적하며, 이는 출력
    형식으로 반올림하기 전 ALU의 guard, round, sticky 비트와 같은 역할을
    합니다.
*   **입력/출력 형식 불일치:** `bfloat16` 입력과 `float32` 누산기를 함께 사용하는
    것이 *초과 정밀도*(퓨전된 연산 전반에서 `float32` 중간값 유지)와 *예상보다
    낮은 정밀도*(1회 패스 `float32` matmul이 입력을 8비트 `bfloat16` 유효숫자로
    잘라냄)의 근본 원인입니다. 반면 TPU 벡터 실행 유닛(VPU)은 `float32`의 가수
    비트 23개 전체로 연산합니다(서브노멀 같은 사소한 IEEE-754 예외 사례 제외).

### `float32`를 위한 다중 패스 다항식 전개 { #multi-pass-polynomial-expansion-for-float32 }

MXU는 `bfloat16` 입력(8비트 유효숫자)만 받으므로 완전한 24비트 `float32` 행렬
곱셈을 한 번의 패스로 계산할 수 없습니다. 실제로 대부분의 최신 모델은 가중치를
`bfloat16`으로 저장하며 `float32` matmul을 거의 실행하지 않습니다. 완전한
`float32` matmul 정밀도가 필요한 경우 XLA는
[Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376)
(Henry, Tang, Heinecke)의 알고리즘을 사용하여 각 24비트 `float32` 유효숫자를
세 개의 8비트 `bfloat16` 값의 합(`High + Med + Low`)으로 표현하고 그 곱을
전개합니다.

```text
(a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
```

두 개의 3항 합을 곱하면 아홉 개의 `bfloat16` 곱이 생기며, 이 중 가장 낮은 차수의
세 항은 `float32`에서 항상 언더플로됩니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|          | High        | Med                      | Low                      |
| :------- | :---------- | :----------------------- | :----------------------- |
| **High** | High x High *(1회 패스)* | High x Med *(3회 패스)*    | High x Low *(6회 패스)*    |
| **Med**  | Med x High *(3회 패스)* | Med x Med *(6회 패스)*     | Med x Low *(언더플로)* |
| **Low**  | Low x High *(6회 패스)* | Low x Med *(언더플로)* | Low x Low *(언더플로)* |
<!-- mdformat on -->

### 3회 패스 및 6회 패스 성능 평가 { #evaluate-3-pass-and-6-pass-performance }

`Precision.HIGH`(3회 패스)와 `Precision.HIGHEST`(6회 패스)는 MXU 패스가 각각
3배와 6배 필요하지만, 엔드 투 엔드 영향은 워크로드에 따라 다릅니다.

*   **큰 축약 차원:** 축약 차원이 큰 최신 모델 중 `float32` 누적으로 학습된
    모델(예: DeepSeek-V3)에서는 수천 개 요소에 걸친 누적 노이즈와 비결정성이 하위
    가수 교차항의 기여보다 크므로 1회 패스(`Precision.DEFAULT`)로 충분합니다.
*   **메모리 대역폭에 제한되는 모델:** 오래된 `float32` 모델은 HBM(High
    Bandwidth Memory)에서 전송되는 바이트당 산술 연산이 MXU(또는 GPU Tensor
    Core)를 포화시킬 만큼 충분하지 않은 경우가 많습니다. 이러한 메모리 제한
    워크로드에서는 3회 패스 및 6회 패스 실행의 추가 MXU 패스가 유휴 상태였을 MXU
    사이클을 사용하므로, 엔드 투 엔드 속도 저하는 3배 또는 6배보다 작습니다.

## 비트 단위 정밀도 동작 검증 { #validate-bitwise-precision-behavior }

두 가지 이진 부동 소수점 규칙을 사용하여 입력을 구성하면 MXU가 가수 비트와 초과
정밀도를 어떻게 처리하는지 확인할 수 있습니다.

*   `1 + 2**-n`은 이진 유효숫자가 `1.000...1000...`인 숫자를 만들어
    `n`번째 가수 비트에 소수 `1`을 배치합니다.
*   `(1 + m) * (1 + n) = 1 + m + n + mn`: 가수 위치 6과 7에 비트가 설정된 두
    숫자를 곱하면 위치 6, 7, `6 + 7 = 13`에 비트가 설정된 결과가 생성됩니다.
    `bfloat16`은 가수 비트를 7개만 보유하므로 위치 13은 `bfloat16`을 넘어서며,
    `float32` 누산기에 유지될 때만 보존됩니다.

```py
value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
```

이 값을 `256 x 256` 행렬로 감싸면 MXU에서 실행이 트리거되며, 13번째 가수 비트가
반올림으로 사라지는지 유지되는지 확인할 수 있습니다.

```python
# Python uses float64 by default.
bit_6 = 1 + 2**-6
bit_7 = 1 + 2**-7
bits_6_7 = 1.0 + 2**-6 + 2**-7
bits_6_7_13 = 1.0 + 2**-6 + 2**-7 + 2**-13

print("=== Python CPU values ===")
print(f"{bit_6=:.20f}")
print(f"{bit_7=:.20f}")
print(f"{bits_6_7=:.20f}")
print(f"{bits_6_7_13=:.20f}")
print(f"{(bit_6 * bit_7)=:.20f}")
assert bit_6 * bit_7 == bits_6_7_13

print("=== PyTorch TPU values ===")

eye = torch.eye(256, dtype=torch.bfloat16, device=device)
bit_6_tensor = eye.clone() * bit_6
bit_7_tensor = eye.clone() * bit_7
bits_6_7_tensor = eye.float().clone() * bits_6_7
bits_6_7_13_tensor = eye.float().clone() * bits_6_7_13

# Use a 1-element vector rather than a 0D scalar because PyTorch's type
# promotion rules ignore 0D scalar tensor dtypes when paired with 2D tensors.
one_float32 = torch.tensor([1.0], dtype=torch.float32, device=device)

print(f"{bit_6_tensor[0,0]=:.20f}")
print(f"{bit_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_13_tensor[0,0]=:.20f}")

def mul_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
  return a @ b * c

# In eager execution, `a @ b` rounds to bfloat16 before `* c`, dropping bit 13.
print("=== TPU values: Eager ===")
eager_res = mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
print(f"{eager_res=:.20f}")
assert eager_res == bits_6_7_tensor[0, 0]

# With allow_excess_precision = False, the compiled graph also rounds `a @ b`
# to bfloat16 before multiplying by `c`, dropping bit 13.
print("=== TPU values: compiled with allow_excess_precision = False ===")
torch.compiler.reset()
torch.tpu.allow_excess_precision = False
mul_add_compiled_false = torch.compile(mul_add, backend="tpu")
print(f"{torch.tpu.allow_excess_precision=}")
res_false = mul_add_compiled_false(bit_6_tensor, bit_7_tensor, one_float32)[
    0, 0
]
print(f"{res_false=:.20f}")

try:
  assert res_false == bits_6_7_tensor[0, 0]
  print("  [PASSED] Successfully disabled excess precision.")
except AssertionError:
  print(
      "  [WARNING] Failed to disable excess precision (XLA compiled another"
      " graph with default 'True' earlier in this session)."
  )

# Changing allow_excess_precision after XLA compilation has already started in
# this process raises a RuntimeError because compiler options are locked on
# the first compilation.
#
# When `torch.tpu.allow_excess_precision = True` is set at the start of a
# fresh Python session before any TPU operations or compilations run, XLA
# keeps `a @ b` in float32 when fusing with `* c`, preserving the 13th
# mantissa bit (res_true == bits_6_7_13_tensor[0, 0] == 1.0235595703125).
print("=== TPU values: compiled with allow_excess_precision = True ===")
torch.compiler.reset()
try:
  torch.tpu.allow_excess_precision = True
  mul_add_compiled_true = torch.compile(mul_add, backend="tpu")
  print(f"{torch.tpu.allow_excess_precision=}")
  res_true = mul_add_compiled_true(bit_6_tensor, bit_7_tensor, one_float32)[
      0, 0
  ]
  print(f"{res_true=:.20f}")
  assert res_true == bits_6_7_13_tensor[0, 0]
  print("  [PASSED] Successfully enabled excess precision.")
except (RuntimeError, AssertionError) as e:
  print(
      "  [EXPECTED] Cannot change excess precision after first compilation:"
      f" {e}"
  )
```
