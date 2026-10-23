# TorchTPU 커스텀 커널 { #torchtpu-custom-kernels }

TorchTPU는 표준 PyTorch 프레임워크 연산자(예:
[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html))를
네이티브로 가속하지만, 모델별 연산을 가속하거나, 연산을 퓨전하여 메모리 대역폭
오버헤드를 줄이거나, 표준 PyTorch 연산자가 아직 지원하지 않는 새로운 아키텍처를
구현하려면 커스텀 커널을 통합해야 할 수 있습니다.

## 통합 방식 선택 { #choosing-an-integration-approach }

TorchTPU는 사용 사례에 따라 TPU에서 커스텀 커널을 사용하는 세 가지 방법을
제공합니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 방식 | 상태 | 적합한 용도 |
| :---- | :---- | :---- |
| **Tokamax** | 프로덕션 준비 완료 | 커널 코드를 작성하지 않고 일반적인 LLM/MoE 빌딩 블록을 바로 가속합니다. |
| **Pallas (`torch.tpu.pallas.jax_op`)** | 프로덕션 준비 완료 | JAX/Pallas로 맞춤형 최고 성능 TPU 커널을 작성하고 PyTorch에서 호출합니다. |
| **Helion** | 실험적 | PyTorch 네이티브 DSL을 사용하여 여러 가속기(예: GPU, TPU)에서 이식 가능한 커널을 작성합니다. |
<!-- mdformat on -->

## 1. Tokamax의 사전 빌드 커널 사용 { #1-using-pre-built-kernels-from-tokamax }

커스텀 커널을 처음부터 작성하기 전에 해당 연산자를
[Tokamax](http://openxla.org/tokamax)에서 사용할 수 있는지 확인하세요. Tokamax는
PyTorch와 네이티브로 상호 운용되는 고성능 TPU 커널 라이브러리입니다.

### 사용 시기 { #when-to-use }

모델에 다음과 같은 표준 고성능 빌딩 블록이 필요하면 Tokamax를 사용하세요.

*   **MoE(Mixture-of-Experts):** 그룹 행렬 곱셈, 토큰 gather 및 scatter
*   **Attention 변형:** Paged attention, Gated DeltaNet, Kimi Delta Attention,
    MLA(Multi-head Latent Attention)
*   **선택:** 최적화된 `TopK` 계산

### 작동 방식 { #how-it-works }

Tokamax 커널은 JAX/Pallas로 작성되며, 통합 자동 조정(예: 자동 타일 크기 탐색),
오버헤드가 낮은 XProf 벤치마킹, TPU 세대 전반의 회귀 테스트가 뒷받침됩니다.
내부적으로 Tokamax는 `torch.tpu.pallas.jax_op`를 사용하여 프레임워크 상호
운용성을 투명하게 처리하고, 네이티브 PyTorch API를 모델에 직접 노출합니다.

## 2. 커스텀 Pallas 커널 작성 { #2-writing-custom-pallas-kernels }

PyTorch나 Tokamax가 지원하지 않는 맞춤형 연산이 필요하다면
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html)로 TPU 커널을 작성하고
`torch.tpu.pallas.jax_op`를 사용하여 PyTorch 연산자로 등록할 수 있습니다.

### Pallas 커널 통합 방법 { #how-to-integrate-a-pallas-kernel }

1.  입력 및 출력 메모리 참조(`Ref`)를 사용하여 **Pallas 커널을 정의합니다**.
2.  `pl.pallas_call`을 사용하여 출력 형상과 dtype을 지정하고 **커널을 JAX
    함수로 래핑합니다**.
3.  `torch.tpu.pallas.jax_op`를 통해 **JAX 함수를 TorchTPU에 등록합니다**.

```python
import jax
from jax.experimental import pallas as pl
import torch

# 1. Define the Pallas kernel
def add_vectors_pallas_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y

# 2. Wrap the Pallas kernel into a JAX function
def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x, y)

# 3. Wrap it for PyTorch using the functional interface
add_vectors_fn = torch.tpu.pallas.jax_op(
    "pallas::add_vectors",
    add_vectors_jax,
)

# 4. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors_fn(x, y)
```

## 3. Helion으로 이식 가능한 커널 작성(실험적) { #3-writing-portable-kernels-with-helion-experimental }

커널을 여러 가속기(예: TPU, GPU)에서 이식 가능하게 만들려면 커널 작성을 위한
PyTorch 네이티브 DSL인 [Helion](https://helionlang.com/)을 사용할 수 있습니다.

> **Note:** TorchTPU의 Helion 지원은 현재 **실험적**입니다. 프로덕션
> 워크로드에는 Tokamax 또는 `torch.tpu.pallas.jax_op`를 사용하세요.

### Helion 커널 작성 방법 { #how-to-write-a-helion-kernel }

함수에 `@helion.kernel` 데코레이터를 지정하세요. 호스트 측 설정에는 표준
PyTorch 연산을 사용하고, 분할된 타일에 대한 디바이스 실행 루프를 정의하려면
`hl.tile`을 사용하세요.

```python
import helion
import helion.language as hl
import torch

# 1. Define the kernel in Helion
@helion.kernel
def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  # Standard PyTorch host-side setup
  x, y = torch.broadcast_tensors(x, y)
  out = torch.empty_like(x)

  # Device execution loop over automatically partitioned tiles
  for tile in hl.tile(out.size()):
    out[tile] = x[tile] + y[tile]

  return out

# 2. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors(x, y)
```

Helion에 관한 자세한 내용은 [Helion 문서](https://helionlang.com/)를 참조하거나,
[PyTorch 개발자 토론 포럼](https://dev-discuss.pytorch.org/c/helion/31)에서
질문하거나, [GitHub에서 문제를 신고](https://github.com/pytorch/helion/issues)하세요.
