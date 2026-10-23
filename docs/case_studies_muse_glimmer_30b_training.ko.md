# Muse Glimmer 30B: 학습 { #muse-glimmer-30b-training }

이 사례 연구에서는 4개의 Ironwood TPU에서 FSDP를 사용하여 전체 모델의 간단한
학습 루프를 보여 주는 독립형 Muse Glimmer 30B 예제를 살펴봅니다.

GKE(Google Kubernetes Engine) 클러스터의 TPU에서 이 예제를 실행하는 방법은 예제
[README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop)를
참조하세요. 여기서는 대신 몇 가지 구현 선택 사항을 설명합니다.

이 예제는 단순성과 가독성에 중점을 둡니다. 4개의 Ironwood TPU가 있는 단일
호스트에서만 실행됩니다. 물론 이는 현실적인 사전 학습 규모가 아니지만, 적은
비용으로 실행하고, 반복 개선하고, 연구할 수 있는 작동하는 예제를 제공하는 것이
목표입니다.

이 순수 PyTorch(커스텀 커널 없음) 코드는 시퀀스 길이 5120에서 랭크당 약
2400~2800 TPS(전체 19k~22k TPS)의 학습 속도를 달성합니다. 향후 업데이트에서는
TPU에서 실행할 때의 추가 최적화와 성능 권장사항을 설명할 예정입니다.

## 모델 정의 { #model-definition }
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)는
상당히 표준적인 transformer 아키텍처를 따릅니다.
인과적 셀프 어텐션과 피드포워드(FFN) 블록으로 이루어진 52개의 레이어로
구성됩니다. 모델의 4번째 레이어마다 전체 어텐션(full attention)을 사용하고, 나머지 모든
레이어는 슬라이딩 윈도우 어텐션(SWA)을 사용합니다.

모델 코드는
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)
(그리고 회전 위치 인코딩을 위한
[rope.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/rope.py))에
완전히 독립적으로 포함되어 있습니다.

구성은
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)에
지정되어 있습니다.

```python
class Config:
  # General model config
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  # Details
  swa_size: int = 2048
  # ... omitted ...

  # Perf evaluation config
  batch_size: int = 8  # Local (per rank) batch size
  seq_len: int = 5120
```

모델 백본의 정의는 다음과 같이 요약됩니다.

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
self.embed_norm = nn.RMSNorm(
    (config.dmodel,), eps=config.rmsnorm_eps, elementwise_affine=False
)
self.layers = nn.ModuleList(
    DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
)
self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)
```

여기에는 모델 헤드("LM 헤드"라고도 함)가 없다는 점에 유의하세요. 학습의 경우
일반적으로 손실 함수와 함께 두는 것이 더 좋습니다. 아래의 청크 단위 퓨전 손실
섹션을 참조하세요.

## 레이어별 torch.compile { #per-layer-torchcompile }

TorchTPU는 연산 퓨전, 메모리 최적화, 계산과 통신의 자동 오버랩 등 다양한 최적화를
수행하는 XLA 컴파일러 덕분에 `torch.compile`에서 최고의 성능을 냅니다.

이 예제에서는 (전체 모델이 아니라) 각 모델 레이어에 개별적으로 `torch.compile`을
적용하는 방법을 보여 줍니다. 두 방법 모두 잘 작동하지만, 전체 모델 컴파일은
시간이 꽤 더 오래 걸릴 수 있으며, 적어도 이 모델에서는 성능 면에서 단일 레이어
범위를 넘어 컴파일해도 이점이 없습니다.

```python
class MuseGlimmer30B(nn.Module):

  def __init__(self, config: Config):
    # ... omitted ...
    self.layers = nn.ModuleList(
        DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
    )

  def apply_torch_compile(self, backend):
    self.embed_norm = torch.compile(self.embed_norm, dynamic=False, backend=backend)
    self.layers = nn.ModuleList(
        torch.compile(layer, dynamic=False, backend=backend)
        for layer in self.layers
    )
    self.norm = torch.compile(self.norm, dynamic=False, backend=backend)
```

여기서 `dynamic=False`는 동적 형상을 사용하지 않는다는 의미이며(학습 중 모든
형상이 정적임), `backend` 변수는 `"tpu"`와 `"inductor"`(CUDA용) 백엔드 간
전환을 쉽게 하기 위한 것입니다.

내부적으로 TorchTPU는 컴파일된 레이어를 캐시하며, 이 모델에는 레이어 종류가 두
가지(전체 어텐션 및 SWA)뿐이므로 컴파일 과정이 크게 빨라집니다.

## 활성화 체크포인팅 { #activation-checkpointing }

모델 가중치는 HBM에서 약 60GiB(bf16 기준)를 차지하며, 단일 Ironwood TPU
TensorCore(96GiB)에 쉽게 들어갑니다.

그러나 학습 중에는 그래디언트와 옵티마이저 상태(예: AdamW momentum 및
variance)를 위한 메모리도 필요하며, 더 중요한 점은 autograd 기능에 필요한 중간
활성화가 메모리 사용량의 대부분을 쉽게 차지할 수 있다는 것입니다. 활성화
체크포인팅(재구체화라고도 함)은 후자를 해결하는 매우 일반적인 기법입니다.

모델을 레이어별로 컴파일하므로 같은 단위로 활성화 체크포인팅을 적용하는 것이
매우 자연스럽습니다. 표준
[`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint)
기능을 사용하여 이를 수행합니다.

```python
class DecoderBlock(nn.Module):
  # ... omitted ...

  def forward(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    return torch.utils.checkpoint.checkpoint(
        self.forward_impl,
        h,
        rope_sin_cos,
        use_reentrant=False,
    )

  def forward_impl(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(self.input_layernorm(h), rope_sin_cos)
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h
```

> **Note:** 활성화 체크포인팅(AC)은 모델 레이어(`DecoderBlock` 모듈)의 `forward`
> 메서드 안에서 적용되고, `torch.compile`은 레이어 외부 / 레이어 자체에
> 적용됩니다. 이 순서가 중요합니다.
> AC를 레이어 안에 두고 전체 모델에 `torch.compile`을 적용해도 잘 작동합니다.
> 하지만 둘의 위치를 바꿀 수는 없습니다. 예를 들어 AC를 같은 방식으로 적용하면서
> `torch.compile`을 내부의 어텐션 및 FFN 하위 레이어에 개별적으로 적용하면
> 예상대로 작동하지 않습니다.

TorchTPU에서 `torch.compile`은 AC와 특히 잘 작동합니다. (전역) 역전파 중에 각
레이어는 (재구체화로 인해) 작은 순전파+역전파 결합 패스를 수행하며, TPU 백엔드
컴파일러(XLA)는 순전파+역전파 결합 계산 그래프 전체를 보므로 추가 최적화를
수행하고 중간 텐서의 메모리 사용량까지 줄일 수 있습니다.

## "단순한" FSDP { #naive-fsdp }

이 예제에서는 네이티브 PyTorch 집합 통신 연산을 직접 사용하여 FSDP(fully-sharded
data-parallel) 학습을 수행하는 가장 간단한 방법을 보여 주고자 했습니다.

> **Tip:** 이 접근 방식에 대한 더 간단한 설명은
> [수동 FSDP 예제](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual)를 참조하세요.

모든 어텐션 및 FFN 프로젝션 가중치를 수동으로 샤딩하여 FSDP를 수행합니다. 예를
들면 다음과 같습니다.

```python
# QKV projections are fully sharded:
self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
```

> **Note:** PyTorch `nn.Linear` 레이어는 가중치 행렬을 전치하여 저장하므로
> 여기서는 두 번째 차원에서 샤딩합니다. 가중치의 실제 형상은
> `[num_q_heads * dh / world_size, dm]`입니다. PyTorch all-gather /
> reduce-scatter 집합 통신은 첫 번째 차원에서 작동하므로 이 점이 중요합니다.

그런 다음 순전파 중에 PyTorch의 함수형 집합 통신을 사용하여 가중치를 사용하기
전에 수집합니다.

```python
# Functional collectives
import torch.distributed._functional_collectives as func_col

# ... all-gather:
q_proj_w = func_col.all_gather_single(self.q_proj.weight, gather_dim=0, group=self.pg)

# ... use full weight directly:
q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
```

같은 방식이 다른 모든 프로젝션(어텐션 Q/K/V/out/gate, FFN up/gate/down)에도
적용됩니다.

중요한 점은 `func_col.all_gather_single`과 같은 함수형 집합 통신이 PyTorch
autograd 엔진에 등록되어 있어, `.backward()` 패스 중에 해당하는 reduce-scatter
집합 통신이 자동으로 생성된다는 것입니다(즉, 직접 작성할 필요가 없습니다).

이 구현은 현재 레이어를 실행하는 동안 다음 레이어의 가중치를 프리페치하려고
시도하지 않는다는 점에서 비교적 단순합니다. 그렇다면 TPU TensorCore가 집합 통신이
완료되기를 기다리며 정체/유휴 상태가 되지 않을까요?

전체 레이어에 `torch.compile`을 적용하므로 TorchTPU의 컴파일러(XLA)는 순전파와
순전파+역전파 결합 패스 모두에서 집합 통신의 순서를 바꾸고 SparseCore로
오프로드하여 비동기적으로 실행할 수 있습니다. 이 경우 이것만으로도 통신 / 계산
오버랩을 거의 완벽하게 달성할 수 있습니다.

## 청크 단위 퓨전 LM 헤드 + 교차 엔트로피 손실 { #chunkwise-fused-lm-head-cross-entropy-loss }

이 예제는 기본적으로 로컬 배치 크기 8과 시퀀스 길이 5120을 사용합니다. 교차
엔트로피 학습 손실을 단순하게 계산하면 형상이 `[8, 5120, 202048]`인 중간
logits 텐서를 구체화해야 합니다. 여기서 마지막 숫자는 모델의 어휘 크기입니다.
bf16에서는 이것만으로도 약 16GiB의 HBM 메모리가 필요하므로 너무 낭비입니다.

이 문제의 일반적인 해결책은 어휘 행렬 가중치 matmul과 교차 엔트로피 손실을
퓨전하여 손실 계산을 청크 단위로 수행하는 것입니다. 한 가지 방법은 어휘 차원을
따라 청크로 나누는 것(online-softmax가 필요하며 순수 PyTorch로 표현하기
어려움)이고, 다른 방법은 시퀀스 차원을 따라 청크로 나누는 것입니다. 여기서는
후자를 사용합니다.

```python
class MuseGlimmerChunkwiseFusedLMHeadLoss(nn.Module):
  # ... omitted ...

  for i in range(0, h.shape[0], chunk_size):
    h_chunk.data.copy_(h[i : (i + chunk_size)])
    tgt_chunk.data.copy_(tgt[i : (i + chunk_size)])

    self.fused_loss_fn_fwd_bwd(
        h_chunk, tgt_chunk, self.lm_head.weight, alpha, cap, loss_scaler
    )

    h_chunk_grads.append(h_chunk.grad)
    h_chunk.grad = None

  return torch.cat(h_chunk_grads, dim=0).view(bsz, seq_len, dm)
```

`fused_loss_fn_fwd_bwd`는 토큰의 단일 청크에서 퓨전된 어휘 matmul + 교차
엔트로피 손실 + 역전파를 계산하는 정적 함수의 `torch.compile`된 버전을
가리킵니다.

```python
@staticmethod
def _chunk_fused_lmhead_loss_fwd_bwd(
    h, tgt, lm_head_w, alpha, cap, loss_scaler
):
  logits = F.tanh(F.linear(h, lm_head_w) * (alpha / cap)) * cap
  loss = F.cross_entropy(input=logits, target=tgt, reduction="sum")
  loss *= loss_scaler
  # Note: in recent versions of PyTorch, .backward is no longer a graph break!
  loss.backward()
  return
```

이 방법의 이점을 최대한 얻기 위해 청크에서 역전파까지 수행합니다. 최신 버전의
PyTorch에서는 `loss.backward()`를 호출해도 더 이상 그래프 중단이 발생하지
않으므로, TorchTPU의 컴파일러(XLA)가 더 많은 최적화와 퓨전을 수행할 수 있습니다.

## 이 예제에서 빠진 내용 { #whats-missing-in-this-example }

이 예제는 기본 학습 루프만 보여 주며, 명확성을 위해 실제 학습 설정에서 중요한
다음과 같은 세부 사항을 생략합니다.

 - 데이터 로드, 토큰화
 - 모델 체크포인트 로드/저장
 - 작업 상태 모니터링, 수렴 진단
 - 내결함성

TorchTPU를 사용한 현실적인 프로덕션 수준 학습 설정은
[TorchTitan](https://github.com/google-pytorch/torchtitan/)을 참조하세요.
