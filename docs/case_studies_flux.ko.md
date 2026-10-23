# FLUX 최적화 { #flux-optimization }

<!-- Link to colab and to generated notebook -->

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_flux.ipynb)

이 사례 연구에서는 [FLUX](https://github.com/black-forest-labs/flux) 디퓨전
트랜스포머 `DoubleStreamBlock`의 참조 PyTorch 구현을 가져와 시퀀스 길이 32,768에서
TorchTPU를 사용하여 Cloud TPU에서 학습 단계를 **1.48배 빠르게** 만든 과정을
살펴봅니다.

이 페이지는 전체 튜토리얼을 요약한 안내서입니다. 짧고 독립적인 스니펫으로 각
최적화의 핵심 개념을 소개합니다. 전체 모델 코드, 벤치마크, XProf 트레이스,
테스트는 다음을 참고하세요.

*   [FLUX 최적화 튜토리얼](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md):
    XProf 트레이스와 모든 벤치마크 재현 방법을 포함하여 모든 마일스톤을 자세히
    설명합니다.
*   [PyTorch용 커스텀 Pallas 커널 작성](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md):
    마일스톤 4에서 사용한 Pallas SplashAttention 커널을 단계별로 설명합니다.

이 구현은
[FLUX의 TorchTitan 구현](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)과
[black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/)
저장소를 기반으로 합니다.

예제를 처음 사용한다면
[다른 문서](./get_started_on_gke.md)를 통해 GKE에서 예제를 실행하는 방법을 먼저
익혀 보세요.

## 결과 요약 { #results-at-a-glance }

단일 `DoubleStreamBlock`의 평균 학습 단계 시간(순전파 + 손실 + 역전파 +
옵티마이저 단계):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 구성 | 평균 단계 시간 | 기준 대비 속도 향상 |
| :--- | ---: | ---: |
| 기준 참조 구현(eager) | 178.61 ms | 1.00x |
| Pallas SplashAttention만 적용(eager) | 149.81 ms | 1.19x |
| 제로 카피 뷰만 적용(eager) | 178.30 ms | 1.00x |
| `torch.compile`만 적용 | 159.33 ms | 1.12x |
| 제로 카피 뷰 + 퓨전된 RoPE + Pallas SplashAttention(eager) | 156.73 ms | 1.14x |
| 제로 카피 뷰 + `torch.compile` | 159.08 ms | 1.12x |
| 제로 카피 뷰 + 퓨전된 RoPE + `torch.compile` | 153.60 ms | 1.16x |
| **제로 카피 뷰 + 퓨전된 RoPE + Pallas SplashAttention + `torch.compile`** | **122.51 ms** | **1.46x** |
<!-- mdformat on -->

**측정 환경:**

-   **하드웨어:** 단일 TPU v6e 칩(32GiB HBM)
-   **소프트웨어:** Python 3.14, PyTorch 2.15(사전 출시 버전), JAX 0.12, einops
    0.8.2, 소스에서 빌드한 TorchTPU
-   **모델 형상:** `batch_size=2`, `hidden_size=1024`, `num_heads=8`
    (`head_dim=128`), `img_seq_len=32256`, `txt_seq_len=512`(총 시퀀스 길이
    32,768)
-   **정밀도:** bfloat16 블록 파라미터 및 입력
-   **SplashAttention 블록 크기:** `block_q`, `block_kv`, dK/dV 역전파 타일 모두
    2048
-   **시간 측정:** 워밍업 단계 3회 후 시간 측정 단계 10회. 시간 측정 단계는 연이어
    디스패치되며 마지막 단계 후에 `torch.accelerator.synchronize()`가 한 번
    실행되므로, 평균 단계 시간(총 시간을 단계 수로 나눈 값)에는 디바이스 실행 시간이
    포함됩니다. 시간 측정 단계 중에는 프로파일러가 사용 중지됩니다.

이 시퀀스 길이에서는 어텐션이 단계 시간의 대부분을 차지하므로 SplashAttention과
컴파일의 효과가 가장 큽니다. 더 짧은 시퀀스 길이에서는 컴파일의 효과가
커지고(시퀀스 길이 640에서 약 3배) 어텐션 커널의 효과는 작아집니다.

이 교훈은 TPU의 대부분의 트랜스포머 모델에 일반적으로 적용됩니다.

1.  **숨겨진 레이아웃 복사 피하기**: 새 HBM 버퍼를 구체화하는 재배열보다 제로
    카피 뷰(`.view()`, `torch.unbind()`)를 사용하세요.
2.  **저장은 적게, 계산은 많이**: 위치 임베딩을 구체화된 행렬 대신 간결한 요소별
    연산으로 표현하세요.
3.  **블록 전체 컴파일**: `torch.compile()`을 사용하면 XLA가 요소별 연산 체인을
    퓨전하고 중간 HBM 왕복을 제거할 수 있습니다.
4.  **필요할 때 커스텀 커널 사용**: Pallas 커널을 사용하면 긴 시퀀스 길이의
    어텐션처럼 메모리 바운드인 연산에서 온칩 타일링을 명시적으로 제어할 수
    있습니다.

### 이 페이지를 읽는 방법 { #how-to-read-this-page }

각 마일스톤은 두 섹션으로 구성됩니다.

-   **이전(참조)**: 원래 구현과 병목 지점
-   **이후(최적화)**: 최적화된 구현과, 참조 구현과 같은 결과를 생성하는지 확인하는
    검사

코드 셀의 첫 줄에는 이에 맞는 `# BEFORE (reference)` 또는 `# AFTER
(optimized)` 주석이 있습니다.

**기본 요건:** 마일스톤 1의 참조 셀은
[einops](https://github.com/arogozhnikov/einops)(`pip install einops`)를
사용합니다.

```python
import torch

device = torch.device("tpu")
print(f"Running on {device}.")
```

## 마일스톤 1: 제로 카피 텐서 뷰 { #milestone-1-zero-copy-tensor-views }

`DoubleStreamBlock`에서 이미지 스트림과 텍스트 스트림은 형상이
`(B, L, 3 * H * D)`인 은닉 상태를 쿼리, 키, 값으로 프로젝션합니다. 이 프로젝션을
Q, K, V로 분할하는 방식에 따라 학습 중에 TPU가 HBM에서 데이터를 복사해야 하는지가
결정됩니다.

### 이전(참조): `einops.rearrange` { #before-reference-einopsrearrange }

참조 구현은 `3`(Q/K/V) 축을 맨 앞으로 이동합니다.

```python
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)
```

순전파에서 `einops.rearrange`는 프로젝션의 스트라이드 뷰를 반환하므로 아직
데이터가 이동하지 않습니다. 비용은 **역전파**에서 발생합니다. 축 재정렬을
거쳐 그래디언트를 되돌려 보내기 위해 autograd가 그래디언트를 새 버퍼에
구체화하며, 이는 TPU에서 모든 어텐션 블록마다 추가 HBM 간 복사로
실행됩니다(XProf에서 `copy` 연산으로 표시됨).

```python
# BEFORE (reference): einops.rearrange moves the Q/K/V axis to the front.
import einops

num_heads, head_dim = 4, 32
qkv = torch.randn(2, 16, 3 * num_heads * head_dim, device=device)

q_before, _k, _v = einops.rearrange(
    qkv, "B L (K H D) -> K B L H D", K=3, H=num_heads
)
print(f"q shape: {tuple(q_before.shape)}")
```

### 이후(최적화): `.view()` + `torch.unbind()` { #after-optimized-view-torchunbind }

최적화된 구현은 `3` 축을 `(H, D)`에 인접하게 유지하고 `torch.unbind`로
분할합니다. 그러면 역전파에서 Q, K, V 그래디언트가 원래 레이아웃으로 바로
쌓이므로 추가 복사가 필요하지 않습니다. 결과 표에서 단계 시간에 미치는 효과는
작지만(eager 모드에서 **178.61 ms에서 178.30 ms**, 컴파일 시 **159.33 ms에서
159.08 ms**), 이는 복사가 단계에서 차지하는 비중이 작기 때문이며 비용 없이 복사를
제거할 수 있습니다.

> **Note:** 이 마일스톤의 셀은 두 방식이 같은 Q, K, V를 생성한다는 것만
> 보여 줍니다. 순전파에서는 두 방식 모두 같은 스트라이드 뷰를 반환하므로
> 출력이 동일해 보입니다. `einops.rearrange`로 인한 추가 HBM 복사는 역전파에서만
> 발생하며, 셀 출력이 아니라 전체 학습 단계의 XProf 트레이스에서
> 확인할 수 있습니다. 트레이스는
> [FLUX 최적화 튜토리얼의 마일스톤 1](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-1)을
> 참고하세요.

```python
# AFTER (optimized): keep Q/K/V adjacent to (H, D) and unbind.
_b, _l, _ = qkv.shape
q_after, _k, _v = torch.unbind(
    qkv.view(_b, _l, 3, num_heads, head_dim), dim=2
)
print(f"q shape: {tuple(q_after.shape)}")
print(f"Matches reference: {torch.equal(q_after.cpu(), q_before.cpu())}")
```

## 마일스톤 2: 퓨전된 대수적 RoPE { #milestone-2-fused-algebraic-rope }

FLUX는 쿼리와 키에 다축 RoPE(Rotary Position Embeddings)를 적용합니다. 아래
셀은 단일 RoPE 축에 대해 두 버전을 모두 구현합니다.

### 이전(참조): 2x2 회전 행렬 { #before-reference-2x2-rotation-matrices }

참조 구현은 모든 주파수 쌍에 대해 전체 `2x2` 회전 행렬 `[[cos,
-sin], [sin, cos]]`을 구체화하고, 재구성된 6D 쿼리/키 텐서에 행렬-벡터 곱으로
적용합니다. 이 방식은 `cos`와 `sin`을 두 번 저장하며, 추가 스택 및 재구성 작업은
XProf에서 큰 복사 블록으로 나타납니다.

```python
# BEFORE (reference): materialize 2x2 rotation matrices, then contract.
def rope_ref(pos, dim, theta=10000):
  """Returns rotation matrices of shape (B, S, D/2, 2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  out = torch.stack(
      [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
      dim=-1,
  )
  return out.reshape(*out.shape[:-1], 2, 2).float()

def apply_rope_ref(xq, xk, freqs_cis):
  """6D matrix-vector contraction against the rotation matrices."""
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return (
      xq_out.reshape(*xq.shape).type_as(xq),
      xk_out.reshape(*xk.shape).type_as(xk),
  )

_b, _s, _h, rope_head_dim = 2, 64, 4, 32
positions = (
    torch.arange(_s, dtype=torch.float32, device=device)
    .unsqueeze(0)
    .expand(_b, _s)
)
rope_xq = torch.randn(_b, _s, _h, rope_head_dim, device=device)
rope_xk = torch.randn(_b, _s, _h, rope_head_dim, device=device)

# Add a broadcast head dimension: (B, S, 1, D/2, 2, 2).
_freqs_ref = rope_ref(positions, rope_head_dim).unsqueeze(2)
q_rope_ref, k_rope_ref = apply_rope_ref(rope_xq, rope_xk, _freqs_ref)

print(f"Rotation matrix freqs shape: {tuple(_freqs_ref.shape)}")
```

### 이후(최적화): 간결한 `[cos, sin]` 쌍 + 요소별 회전 { #after-optimized-compact-cos-sin-pairs-elementwise-rotation }

최적화된 구현은 간결한 `[cos, sin]` 쌍만 저장하고 요소별 연산으로 회전을 직접
적용합니다.

```python
# (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
xq_out_0 = xq_0 * cos - xq_1 * sin
xq_out_1 = xq_0 * sin + xq_1 * cos
```

이렇게 하면 위치 임베딩이 차지하는 공간이 절반으로 줄고, 연산이 FMA(fused
multiply-add) 명령어에 매핑되며, 컴파일 시 주변의 프로젝션 및 어텐션 연산과
퓨전됩니다. 제로 카피 뷰와 `torch.compile`에 더해 이 방식을 적용하면 단계 시간이
**159.08 ms에서 153.60 ms로(3.4%)** 줄어듭니다.

두 버전은 같은 피연산자에 대해 같은 곱셈과 덧셈을 수행하므로 동일한 결과를
생성합니다. 아래 셀은 같은 입력으로 최적화된 버전을 실행하고 참조 구현과
비교합니다.

> **Note:** 이 셀은 주파수 텐서가 더 작다는 점과 두 버전이 동일한 결과를
> 생성한다는 점을 보여 줍니다. 속도 향상은 RoPE가 컴파일된 `DoubleStreamBlock`
> 내부에서 실행되어 요소별 회전이 주변 연산과 퓨전될 때만 나타납니다. XProf
> 트레이스는
> [FLUX 최적화 튜토리얼의 마일스톤 2](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-2)를
> 참고하세요.

```python
# AFTER (optimized): store compact [cos, sin] pairs, rotate elementwise.
def rope(pos, dim, theta=10000):
  """Returns compact [cos, sin] pairs of shape (B, S, D/2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  return torch.stack([torch.cos(out).float(), torch.sin(out).float()], dim=-1)

def apply_rope(xq, xk, freqs_cis):
  """Direct 2D algebraic rotation with elementwise math."""
  cos = freqs_cis[..., 0]
  sin = freqs_cis[..., 1]
  b, s, h, d = xq.shape
  xq_ = xq.float().view(b, s, h, d // 2, 2)
  xk_ = xk.float().view(b, s, h, d // 2, 2)
  xq_0, xq_1 = xq_[..., 0], xq_[..., 1]
  xk_0, xk_1 = xk_[..., 0], xk_[..., 1]
  xq_out = torch.stack(
      [xq_0 * cos - xq_1 * sin, xq_0 * sin + xq_1 * cos], dim=-1
  ).view(b, s, h, d)
  xk_out = torch.stack(
      [xk_0 * cos - xk_1 * sin, xk_0 * sin + xk_1 * cos], dim=-1
  ).view(b, s, h, d)
  return xq_out.type_as(xq), xk_out.type_as(xk)

# Add a broadcast head dimension: (B, S, 1, D/2, 2).
freqs = rope(positions, rope_head_dim).unsqueeze(2)
_q_new, _k_new = apply_rope(rope_xq, rope_xk, freqs)

print(f"Compact freqs shape:         {tuple(freqs.shape)}")
print(
    "Max |q_ref - q_new|:        "
    f" {(q_rope_ref.cpu() - _q_new.cpu()).abs().max().item():.3e}"
)
print(
    "Max |k_ref - k_new|:        "
    f" {(k_rope_ref.cpu() - _k_new.cpu()).abs().max().item():.3e}"
)
torch.testing.assert_close(
    _q_new.cpu(), q_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_new.cpu(), k_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
```

## 마일스톤 3: 블록 전체 XLA 컴파일 { #milestone-3-whole-block-xla-compilation }

### 이전(참조): eager 실행 { #before-reference-eager-execution }

eager 모드에서는 모든 PyTorch 연산이 독립적인 TPU 프로그램을 시작하므로 중간
활성화가 HBM을 왕복하고 호스트가 디스패치 때문에 자주 멈춥니다. 결과 표에서 기준
`DoubleStreamBlock`은 eager 모드에서 단계당 **178.61 ms**가 걸립니다.

아래 셀은 마일스톤 2의 퓨전된 RoPE를 eager 모드에서 연산별로 실행합니다.

```python
# BEFORE (reference): eager execution, one TPU program per op.
q_eager, k_eager = apply_rope(rope_xq, rope_xk, freqs)
print("Ran fused RoPE eagerly.")
```

### 이후(최적화): `torch.compile()` { #after-optimized-torchcompile }

`torch.compile()`로 `DoubleStreamBlock` 전체를 컴파일하면 블록이 단일 XLA
그래프로 로어링되므로 컴파일러가 요소별 연산, 정규화, 잔차 덧셈을 퓨전할
수 있습니다.

```python
block = DoubleStreamBlock(
    hidden_size=1024,
    num_heads=8,
    mlp_ratio=4.0,
    config=OptimizationConfig(
        use_zero_copy_view=True,
        use_fused_rope=True,
        compile=True,
    ),
)
compiled_block = block.maybe_compile()  # Wraps the block in torch.compile().
```

컴파일하면 단계 시간이 **178.61 ms에서 159.33 ms로(1.12배)** 줄어듭니다. 시퀀스
길이 32,768에서는 어텐션이 단계 시간의 대부분을 차지하므로 개선 폭이 크지
않습니다. 시퀀스 길이 640에서는 컴파일하면 약 3배 빨라집니다.

같은 패턴을 모든 함수에 적용할 수 있습니다. 아래 셀은 퓨전된 RoPE를 컴파일하고
eager 실행 결과와 일치하는지 확인합니다.

> **Note:** 이 셀은 컴파일해도 결과가 바뀌지 않는지만 확인합니다. 퓨전과 속도
> 향상은 셀 출력이 아니라 전체 `DoubleStreamBlock`의 XProf 및 단계 시간 측정에서
> 나타납니다. eager 실행과 컴파일 트레이스 비교는
> [FLUX 최적화 튜토리얼의 마일스톤 3](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-3)을
> 참고하세요.

```python
# AFTER (optimized): compile into a single fused XLA graph.
compiled_apply_rope = torch.compile(apply_rope, backend="tpu")
_q_compiled, _k_compiled = compiled_apply_rope(rope_xq, rope_xk, freqs)

torch.testing.assert_close(
    _q_compiled.cpu(), q_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_compiled.cpu(), k_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
print("Compiled fused RoPE matches eager execution.")
```

## 마일스톤 4: 커스텀 Pallas SplashAttention 커널 { #milestone-4-custom-pallas-splashattention-kernel }

긴 시퀀스 길이(예: `S = 2048` 또는 `S = 4096`)에서는 전체 결합 어텐션이 메모리
바운드가 됩니다. 이 마일스톤은 스니펫으로만 보여 줍니다. 실행 가능한 전체 구현은
[Pallas 커널 안내](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)를
참고하세요.

### 이전(참조): 표준 PyTorch SDPA { #before-reference-standard-pytorch-sdpa }

```python
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

타일링되지 않은 어텐션은 HBM에 `O(S^2)` 점수 행렬을 구체화합니다.

### 이후(최적화): Pallas SplashAttention { #after-optimized-pallas-splashattention }

```python
attn = splash_sdpa(q, k, v, is_causal=False)  # Drop-in replacement.
```

`splash_sdpa`는 온칩 타일 단위로 어텐션을 계산하는 커스텀
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) SplashAttention
커널이며,
[SplashAttention의 TorchTitan 구현](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py)을
기반으로 합니다.

이 커널은 `torch.tpu.pallas.jax_op`으로 PyTorch에 노출됩니다. 순전파 함수와
역전파 함수에는 `jax.Array` 타입 주석이 달려 있으며, `jax_op`은 이를 바탕으로
PyTorch 커스텀 연산자 스키마를 추론합니다. 순전파는 softmax
`logsumexp` 잔차를 저장하므로 역전파 커널이 순전파 연산을 다시 실행할 필요가
없습니다.

```python
torch_fwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_fwd_name}", splash_fn)
torch_bwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_bwd_name}", splash_bwd_fn)


def setup_context(ctx, inputs, output):
  q, k, v = inputs
  out, lse = output
  ctx.save_for_backward(q, k, v, out, lse)


def backward(ctx, grad_out, grad_lse):
  del grad_lse
  q, k, v, out, lse = ctx.saved_tensors
  return torch_bwd_fn(q, k, v, out, lse, grad_out)


torch_fwd_fn.register_autograd(backward, setup_context=setup_context)
```

> **Important:** **런타임에는 JAX가 실행되지 않습니다.** Pallas는 `jax` 패키지에
> 포함되어 배포되므로 커널은 `jax` 문법으로 작성합니다. 하지만
> `torch.tpu.pallas.jax_op`은 트레이스 시점에 추상 형상/dtype 자리표시자로 커널을
> MLIR 모듈로 내보낼 때 JAX를 한 번만 사용합니다. 학습과 추론 중에는 eager
> 모드와 `torch.compile` 모두에서 커널이 `torch.Tensor` TPU 버퍼에서 직접
> 실행됩니다.

**결과**: 결과 표에서 SplashAttention만 적용하면 eager 기준보다
**1.19배** 빠릅니다(178.61 ms에서 149.81 ms). 제로 카피 뷰와 퓨전된 RoPE를
사용 설정하면 eager 모드(**182.61 ms에서 156.73 ms, 1.17배**)와 컴파일
모드(**153.60 ms에서 122.51 ms, 1.25배**) 모두에서 표준 SDPA보다 빠릅니다.

이 결과는 워크로드에 맞게 커널을 구성해야 얻을 수 있습니다. 기본 512토큰 블록을
사용하면 같은 시퀀스 길이에서 컴파일 시 SplashAttention이 표준 SDPA보다 약 10%
느렸고, float32 입력에서는 더 느렸습니다. 커스텀 커널을 도입하기 전에 시퀀스
길이에 맞춰 몇 가지 블록 크기로 벤치마크하고, 커널을 bfloat16으로 실행하고,
프로파일링하세요.

**자세히 알아보기:**

*   **Pallas SplashAttention 예제(이 사례 연구에서 사용)**:
    [`splash_attention.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention.py)에는
    전체 커널 통합 코드가 있고,
    [`splash_attention_test.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention_test.py)는
    PyTorch SDPA와 비교하여 순전파 및 그래디언트 일치성을 검증합니다.
    [커널 안내](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)에서는
    블록 크기, 마스크, autograd 등록, 테스트를 단계별로 설명합니다.
*   **Pallas 기본사항**: [Pallas를 사용한 커스텀 커널](./custom_kernels.md)에서
    TorchTPU로 Pallas 커널을 작성하는 방법을 소개합니다.
