# Muse Glimmer 30B: 자기회귀 디코딩 { #muse-glimmer-30b-autoregressive-decoding }

<!-- Link to colab and to generated notebook -->

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_muse_glimmer_30b_inference.ipynb)

이 가이드에서는 **Muse Glimmer** 아키텍처와 사전 학습된 Hugging Face 가중치를
사용하여 TorchTPU로 TPU에서 자기회귀 LLM 추론을 실행하는 과정을 살펴봅니다. 하나의
모델을 공유하는 두 가지 KV 캐시를 비교합니다. 하나는 정적 형상(`dynamic=False`)으로
컴파일되는 사전 할당된 **`StaticCache`**이고, 다른 하나는 동적
형상(`dynamic=None`)으로 컴파일되는, 크기가 커지는 **`DynamicCache`**입니다.

정적 캐시와 동적 캐시용 실행기가 분리된 이 예제의 독립 실행형 스크립트 버전은
[`examples/muse_glimmer_30b/llm_inference`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference)를
참고하세요.

> **메모리 요구사항:** 기본적으로(`USE_FULL_MODEL = False`) 노트북은 디코더
> 레이어 52개 중 처음 4개만 로드하므로(가중치 \~9.3GB) HBM이 적은 TPU에도 맞지만,
> 생성된 텍스트는 의미가 없습니다. 전체 모델(TPU HBM에 가중치 \~55.7GB)을 로드하고
> 의미 있는 텍스트를 생성하려면 다음 셀에서 `USE_FULL_MODEL = True`를 설정하세요.

```python
# True loads all 52 decoder layers (~55.7 GB). False loads only the first 4
# layers (~9.3 GB): the model still compiles and runs, but the generated text
# isn't meaningful.
USE_FULL_MODEL = False
```

## 1. XLA 컴파일을 위한 KV 캐싱 { #1-kv-caching-for-xla-compilation }

자기회귀 LLM 추론에서 생성은 두 단계로 진행됩니다.

*   **프리필(Prefill):** 모델이 전체 프롬프트 `[B, S]`를 병렬로 처리하고 KV 캐시
    `[B, H_kv, S, D]`를 채웁니다(배치 `B`, 프롬프트 길이 `S`, KV 헤드 `H_kv`개,
    헤드 차원 `D`).
*   **디코딩(Decode):** 각 단계 `t`에서 새 토큰 하나 `[B, 1]`을 입력하고, 그
    키와 값을 캐시에 추가하고, 모든 `S + t` 위치에 대해 어텐션을 계산합니다.

**`StaticCache`**는 디코딩 단계 전반에서 모든 텐서 형상을 일정하게 유지하여
`dynamic=False`에서 디코딩 재컴파일을 방지합니다.

1.  **고정 크기 버퍼:** `keys`와 `values`는 `[B, H_kv,
    max_cache_len, D]`로 사전 할당됩니다.
2.  **디바이스 내 길이:** 캐시 길이는 Dynamo가 가드하는 Python `int`가 아니라,
    인플레이스로 업데이트되는 0차원 TPU 텐서입니다.
3.  **인플레이스 쓰기, 고정 형상 마스크:** 새 항목은 `index_copy_`로 쓰고, 아직
    쓰지 않은 위치는 `-inf`로 마스킹합니다.
4.  **두 번의 컴파일:** 프리필용 한 번과 디코딩용 한 번이며, 이후 모든 단계에서
    재사용됩니다.

**`DynamicCache`**는 반대 방식을 사용하며 Hugging Face 모델의 기본 캐시입니다.
각 단계의 키와 값을 `torch.cat`으로 추가하므로 캐시는 정확히 `S + t`개의 위치를
보관합니다. 이 방식은 더 간단하고 패딩 메모리를 사용하지 않지만, 캐시 크기가 매
단계 바뀌므로 토큰마다 재컴파일하지 않으려면 동적 형상(`dynamic=None`)에
의존해야 합니다.

## 2. 모델 및 KV 캐시 구현 { #2-model-and-kv-cache-implementation }

다음 셀은 `StaticCache`, `DynamicCache`, `MuseGlimmer30B` 디코더 아키텍처를
구현합니다. 이 아키텍처는 블록 4개 단위로, RoPE를 사용하는 슬라이딩 윈도우
어텐션(SWA) 레이어 3개와 NoPE를 사용하는 전체 인과적 어텐션 레이어 1개를
교대로 배치합니다.

같은 모델과 캐시는
[`model.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/model.py)
및
[`kv_cache.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/kv_cache.py)에서
재사용 가능한 모듈로도 제공됩니다.

<section class="zippy" markdown="1">

**가져오기 및 디바이스 설정**

```python
from concurrent import futures
import dataclasses
import json
import os
import time
import warnings
from etils import epath
import huggingface_hub
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

torch._logging.set_logs()  # pylint: disable=protected-access
transformers.logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*FLASH_ATTENTION SDPBackend.*",
)
device = torch.device("tpu")
print(f"Using device: {device}")
```

</section>

<section class="zippy" markdown="1">

**KV 캐시 및 모델 구현**

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
  dh = x.shape[-1]
  x1 = x[..., : dh // 2]
  x2 = x[..., dh // 2 :]
  return torch.cat([-x2, x1], dim=-1)

def apply_rope(
    x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
  dtype = x.dtype
  x_f32 = x.to(dtype=torch.float32)
  return ((x_f32 * cos) + (rotate_half(x_f32) * sin)).to(dtype=dtype)

def precompute_rope_sin_cos(
    seq_len: int, dim_head: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
  inv_freq = 1.0 / (
      rope_theta
      ** (torch.arange(0, dim_head, 2, dtype=torch.float32) / dim_head)
  )
  inv_freq = inv_freq.repeat(2).view(1, 1, 1, dim_head)
  pos_idx = torch.arange(0, seq_len, dtype=torch.float32).view(
      1, 1, seq_len, 1
  )
  return torch.sin(pos_idx * inv_freq), torch.cos(pos_idx * inv_freq)

class StaticCache:
  """Fixed-size preallocated KV cache mutated in-place for static compilation.

  SWA layers are not capped at `swa_size`: `max_cache_len` here is far below
  `swa_size`, so no token ever leaves the window, and the SWA mask alone keeps
  attention correct.
  """

  def __init__(
      self,
      num_layers: int,
      batch_size: int,
      num_kv_heads: int,
      max_cache_len: int,
      head_dim: int,
      dtype: torch.dtype,
      device: torch.device | str,
  ):
    shape = (batch_size, num_kv_heads, max_cache_len, head_dim)
    self.max_cache_len = max_cache_len
    self.keys = [
        torch.zeros(shape, dtype=dtype, device=device)
        for _ in range(num_layers)
    ]
    self.values = [
        torch.zeros(shape, dtype=dtype, device=device)
        for _ in range(num_layers)
    ]
    # Stored as a 0-D tensor on device so in-place updates do not trigger
    # torch.compile(dynamic=False) recompilations across decode steps.
    self.cumulative_length = [
        torch.zeros((), dtype=torch.long, device=device)
        for _ in range(num_layers)
    ]

  def get_seq_length(self) -> torch.Tensor:
    return self.cumulative_length[0].clone()

  def get_kv_length(self, seq_len: int) -> int:
    """Returns the sequence length attention sees: always the full buffer."""
    del seq_len  # Unused: the buffer size is fixed.
    return self.max_cache_len

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      layer_idx: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    kv_len = key_states.shape[-2]
    past_len = self.cumulative_length[layer_idx]
    cache_position = torch.arange(kv_len, device=key_states.device) + past_len
    self.keys[layer_idx].index_copy_(2, cache_position, key_states)
    self.values[layer_idx].index_copy_(2, cache_position, value_states)
    past_len.add_(kv_len)
    return self.keys[layer_idx], self.values[layer_idx]

class DynamicCache:
  """KV cache that grows along the sequence dimension via torch.cat."""

  def __init__(self):
    self.keys: list[torch.Tensor] = []
    self.values: list[torch.Tensor] = []

  def get_seq_length(self) -> int:
    return self.keys[0].shape[-2] if self.keys else 0

  def get_kv_length(self, seq_len: int) -> int:
    """Returns the sequence length attention sees: past plus new tokens."""
    return self.get_seq_length() + seq_len

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      layer_idx: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if layer_idx == len(self.keys):
      self.keys.append(key_states)
      self.values.append(value_states)
    else:
      self.keys[layer_idx] = torch.cat(
          [self.keys[layer_idx], key_states], dim=-2
      )
      self.values[layer_idx] = torch.cat(
          [self.values[layer_idx], value_states], dim=-2
      )
    return self.keys[layer_idx], self.values[layer_idx]

# Both caches expose the same interface, so the model works with either.
KVCache = StaticCache | DynamicCache

@dataclasses.dataclass
class Config:
  # General model config
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  swa_size: int = 2048
  rope_freq_base: float = 500000.0
  rmsnorm_eps: float = 1e-5
  post_norm_eps: float = 1e-8
  qk_scale_factor: float = 3.87
  final_logit_softcapping: float = 20.0
  output_multiplier: float = 0.19611613513818404

  prefill_seq_len: int = 128
  num_decode_steps: int = 20

class Attention(nn.Module):
  """Grouped-query attention that works with StaticCache or DynamicCache."""

  def __init__(self, config: Config, is_swa: bool, layer_idx: int = 0):
    super().__init__()
    self.config = config
    self.is_swa = is_swa
    self.layer_idx = layer_idx

    dm, dh = config.dmodel, config.dhead
    num_q_heads = config.num_q_heads
    num_kv_heads = config.num_kv_heads

    self.q_proj = nn.Linear(dm, num_q_heads * dh, bias=False)
    self.k_proj = nn.Linear(dm, num_kv_heads * dh, bias=False)
    self.v_proj = nn.Linear(dm, num_kv_heads * dh, bias=False)
    self.gate_proj = nn.Linear(dm, num_q_heads * dh, bias=False)
    self.o_proj = nn.Linear(num_q_heads * dh, dm, bias=False)

    self.qk_norm = nn.RMSNorm(
        normalized_shape=(dh,),
        eps=config.rmsnorm_eps,
        dtype=torch.float32,
        elementwise_affine=False,
    )

  def forward(
      self,
      h: torch.Tensor,
      rope_sin_cos: tuple[torch.Tensor, torch.Tensor],
      past_key_values: KVCache,
      attn_mask: torch.Tensor,
  ) -> torch.Tensor:
    local_bsz, seq_len, _ = h.shape
    dh = self.config.dhead
    num_q_heads = self.config.num_q_heads
    num_kv_heads = self.config.num_kv_heads

    q = self.q_proj(h).view(local_bsz, seq_len, num_q_heads, dh)
    k = self.k_proj(h).view(local_bsz, seq_len, num_kv_heads, dh)
    v = self.v_proj(h).view(local_bsz, seq_len, num_kv_heads, dh)

    qk_scale = self.config.qk_scale_factor
    q = self.qk_norm(q.transpose(1, 2).contiguous()) * qk_scale
    k = self.qk_norm(k.transpose(1, 2).contiguous())
    v = v.transpose(1, 2).contiguous()

    if self.is_swa:
      # In Muse Glimmer 30B, RoPE is only applied in SWA layers; full attention
      # layers use NoPE.
      rs, rc = rope_sin_cos
      q = apply_rope(q, rs, rc)
      k = apply_rope(k, rs, rc)

    k, v = past_key_values.update(k, v, self.layer_idx)

    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, enable_gqa=True
    )
    attn_out = attn_out.transpose(1, 2).reshape(
        local_bsz, seq_len, num_q_heads * dh
    )

    gate = F.sigmoid(self.gate_proj(h))
    return self.o_proj(attn_out * gate)

class FeedForwardSwiGLU(nn.Module):
  """Simple FFN with SwiGLU activation."""

  def __init__(self, config: Config):
    super().__init__()
    dm, df = config.dmodel, config.dff
    self.up_proj = nn.Linear(dm, df, bias=False)
    self.gate_proj = nn.Linear(dm, df, bias=False)
    self.down_proj = nn.Linear(df, dm, bias=False)

  def forward(self, h: torch.Tensor) -> torch.Tensor:
    return self.down_proj(self.up_proj(h) * F.silu(self.gate_proj(h)))

class DecoderBlock(nn.Module):

  def __init__(self, config: Config, layer_idx: int):
    super().__init__()
    self.self_attn = Attention(
        config, is_swa=((layer_idx % 4) != 3), layer_idx=layer_idx
    )
    self.mlp = FeedForwardSwiGLU(config)

    dm = config.dmodel
    f32 = torch.float32
    eps_pre, eps_post = config.rmsnorm_eps, config.post_norm_eps
    self.input_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
    self.post_attention_layernorm = nn.RMSNorm((dm,), eps=eps_post, dtype=f32)
    self.pre_feedforward_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
    self.post_feedforward_layernorm = nn.RMSNorm(
        (dm,), eps=eps_post, dtype=f32
    )

  def forward(
      self,
      h: torch.Tensor,
      rope_sin_cos: tuple[torch.Tensor, torch.Tensor],
      past_key_values: KVCache,
      attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(
            self.input_layernorm(h),
            rope_sin_cos,
            past_key_values=past_key_values,
            attn_mask=attn_mask,
        )
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h

class MuseGlimmer30B(nn.Module):

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    # Allocate parameter shapes on the meta device first and materialize
    # uninitialized storage via to_empty() so constructing the model on TPU
    # does not compile and execute random initialization kernels for weights
    # that will be overwritten by load_weights_from_checkpoint(). Every
    # parameter must therefore be loaded from the checkpoint.
    target_device = torch.get_default_device()
    with torch.device("meta"):
      self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
      self.embed_norm = nn.RMSNorm(
          (config.dmodel,), eps=config.rmsnorm_eps, elementwise_affine=False
      )
      self.layers = nn.ModuleList(
          DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
      )
      self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)
      self.lm_head = nn.Linear(config.dmodel, config.vocab_size, bias=False)
    self.to_empty(device=target_device)

    max_len = config.prefill_seq_len + config.num_decode_steps
    rope_sin, rope_cos = precompute_rope_sin_cos(
        seq_len=max_len,
        dim_head=config.dhead,
        rope_theta=config.rope_freq_base,
    )
    self.rope_sin = nn.Buffer(rope_sin, persistent=False)
    self.rope_cos = nn.Buffer(rope_cos, persistent=False)

  def forward(
      self,
      tok: torch.Tensor,
      past_key_values: KVCache,
  ) -> tuple[torch.Tensor, KVCache]:
    # past_len is a 0-D device tensor for StaticCache and a Python int for
    # DynamicCache; both work below.
    past_len = past_key_values.get_seq_length()
    seq_len = tok.shape[1]
    # Number of key positions attention sees after this step's cache update.
    kv_len = past_key_values.get_kv_length(seq_len)

    cache_position = torch.arange(seq_len, device=tok.device) + past_len
    rope_sin_cos = (
        self.rope_sin.index_select(2, cache_position),
        self.rope_cos.index_select(2, cache_position),
    )
    # Compute the masks each step by comparing key and query positions,
    # so the mask's last dimension is exactly kv_len. Keys after the query
    # position are masked with -inf; SWA layers also mask keys swa_size
    # or more positions behind.
    kv_pos = torch.arange(kv_len, device=tok.device)  # [kv_len]
    q_pos = cache_position[:, None]  # [seq_len, 1]
    visible = kv_pos[None, :] <= q_pos  # [seq_len, kv_len]
    in_window = visible & (kv_pos[None, :] > q_pos - self.config.swa_size)
    causal_mask = torch.where(visible, 0.0, float("-inf"))[None, None]
    swa_mask = torch.where(in_window, 0.0, float("-inf"))[None, None]

    h = self.embed_norm(self.embed_tokens(tok))
    for layer in self.layers:
      attn_mask = swa_mask if layer.self_attn.is_swa else causal_mask
      h = layer(
          h,
          rope_sin_cos,
          past_key_values=past_key_values,
          attn_mask=attn_mask,
      )
    h = self.norm(h)

    alpha = self.config.output_multiplier
    c = self.config.final_logit_softcapping
    logits = F.tanh(self.lm_head(h[:, -1:, :]) * (alpha / c)) * c
    return logits, past_key_values
```

</section>

## 3. 사전 학습된 가중치 및 토크나이저 로드 { #3-loading-pretrained-weights-and-tokenizer }

이 섹션에서는 `Muse-Glimmer-30B` 토크나이저와 사전 학습된 가중치를
`MuseGlimmer30B`에 로드하는 방법을 정의합니다. 체크포인트는 다음 두 위치 중 한
곳에서 가져올 수 있습니다.

1.  **체크포인트의 로컬 사본:** `MUSE_GLIMMER_CHECKPOINT_DIR` 환경 변수를
    체크포인트 파일이 있는 디렉터리로 설정하세요. 노트북에서는 다음 셀 전에
    `os.environ["MUSE_GLIMMER_CHECKPOINT_DIR"] =
    "/path/to/checkpoint"`를 실행하면 됩니다. 변수가 설정되었지만 디렉터리가
    없으면 다운로드로 대체하지 않고 로드가 실패합니다.
2.  **Hugging Face Hub(기본값):** `MUSE_GLIMMER_CHECKPOINT_DIR`이 설정되지
    않으면 Hub 저장소 `MODEL_NAME`(`meta-models/Muse-Glimmer-30B`)에서 파일을
    다운로드합니다. 다운로드한 파일은 캐시되므로 이후 실행에서 재사용됩니다.

어느 경우든 가중치는 같은 방식으로 로드됩니다.

*   **구성된 레이어만 읽습니다.** 텐서는 `Config(num_layers=...)`의 레이어에
    대해서만 로드되므로 `USE_FULL_MODEL = False`(노트북 상단 참고)이면 레이어
    52개 대신 4개를 로드합니다.
*   **텐서를 모델에 바로 복사합니다.** 텍스트 디코더 텐서는 Hugging Face 모델을
    빌드하지 않고 `.safetensors` 파일에서 `m.state_dict()`로 직접 복사됩니다.
*   **텐서를 병렬로 읽습니다.** 텐서를 작은 배치 단위로 동시에 읽고 다음 배치를
    읽기 전에 TPU로 복사하므로 호스트 메모리 사용량이 적게 유지됩니다.

<section class="zippy" markdown="1">

**체크포인트 및 토크나이저 로드**

```python
MODEL_NAME = "meta-models/Muse-Glimmer-30B"  # Hugging Face Hub repo ID.
# Optional directory with a local copy of the checkpoint. If set, all files
# are read from it (and it must exist); otherwise they are downloaded from
# the Hub.
LOCAL_CHECKPOINT_DIR = os.environ.get("MUSE_GLIMMER_CHECKPOINT_DIR") or None
# Text decoder weights live under this prefix in the multimodal checkpoint.
CHECKPOINT_PREFIX = "model.language_model."
INDEX_FILENAME = "model.safetensors.index.json"

def use_local_checkpoint() -> bool:
  """Returns True if the checkpoint should be read from LOCAL_CHECKPOINT_DIR."""
  if LOCAL_CHECKPOINT_DIR is None:
    return False
  if not epath.Path(LOCAL_CHECKPOINT_DIR).exists():
    # Fail instead of silently downloading the full checkpoint from the Hub.
    raise FileNotFoundError(
        f"MUSE_GLIMMER_CHECKPOINT_DIR={LOCAL_CHECKPOINT_DIR} is set but"
        " does not exist or is not readable."
    )
  return True

def checkpoint_source() -> str:
  """Describes where checkpoint files are read from."""
  if use_local_checkpoint():
    return str(LOCAL_CHECKPOINT_DIR)
  return f"Hugging Face Hub ({MODEL_NAME})"

def checkpoint_file(filename: str) -> epath.Path:
  """Returns a checkpoint file, downloading it from the Hub if needed."""
  if use_local_checkpoint():
    return epath.Path(LOCAL_CHECKPOINT_DIR) / filename
  return epath.Path(huggingface_hub.hf_hub_download(MODEL_NAME, filename))

def checkpoint_key(name: str) -> str:
  """Maps a MuseGlimmer30B parameter name to its HF checkpoint key."""
  return name if name == "lm_head.weight" else CHECKPOINT_PREFIX + name

def load_tokenizer() -> transformers.PreTrainedTokenizerBase:
  """Loads the Hugging Face Muse Glimmer tokenizer."""
  source = LOCAL_CHECKPOINT_DIR if use_local_checkpoint() else MODEL_NAME
  print(f"Loading tokenizer from {checkpoint_source()}...", flush=True)
  return transformers.AutoTokenizer.from_pretrained(source)

# Tensors larger than this are streamed in chunks when their shard can't be
# memory-mapped (see `load_weights_from_checkpoint`).
CHUNK_BYTES = 256 * 1024 * 1024
SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}

def fix_layernorm(name: str, tensor: torch.Tensor) -> torch.Tensor:
  """Converts HF's centered RMSNorm weights to `nn.RMSNorm` weights."""
  if name.endswith("_layernorm.weight"):
    # HF computes x_norm * (1 + w); nn.RMSNorm computes x_norm * w.
    return tensor.float() + 1.0
  return tensor

def read_tensor(path: epath.Path, key: str) -> torch.Tensor:
  """Reads one tensor from a safetensors file into host memory."""
  with safetensors.safe_open(path, framework="pt", device="cpu") as f:
    return f.get_tensor(key)

def copy_tensor_in_chunks(
    path: epath.Path, key: str, dst: torch.Tensor
) -> None:
  """Copies one tensor from a safetensors file into `dst`, a few rows at a time.

  A safetensors file starts with an 8-byte header length, then a JSON header
  giving each tensor's dtype, shape, and byte offsets, then the raw data.
  """
  with path.open("rb") as f:
    header_size = int.from_bytes(f.read(8), "little")
    info = json.loads(f.read(header_size))[key]
    dtype = SAFETENSORS_DTYPES[info["dtype"]]
    num_rows, *row_shape = info["shape"]
    row_bytes = dtype.itemsize * torch.Size(row_shape).numel()
    data_start = 8 + header_size + info["data_offsets"][0]
    rows_per_chunk = max(1, CHUNK_BYTES // row_bytes)
    for start in range(0, num_rows, rows_per_chunk):
      rows = min(rows_per_chunk, num_rows - start)
      f.seek(data_start + start * row_bytes)
      data = bytearray(f.read(rows * row_bytes))
      chunk = torch.frombuffer(data, dtype=dtype).view(rows, *row_shape)
      dst[start : start + rows].copy_(chunk)

@torch.no_grad()
def load_weights_from_checkpoint(m: MuseGlimmer30B) -> None:
  """Loads Muse Glimmer text weights from a safetensors checkpoint into `m`."""
  index = json.loads(checkpoint_file(INDEX_FILENAME).read_text())
  weight_map: dict[str, str] = index["weight_map"]
  state_dict = m.state_dict()

  # Find the file holding each parameter (downloading it if needed).
  shard_paths = {
      shard: checkpoint_file(shard)
      for shard in {weight_map[checkpoint_key(n)] for n in state_dict}
  }
  sources = {
      name: (
          shard_paths[weight_map[checkpoint_key(name)]],
          checkpoint_key(name),
      )
      for name in state_dict
  }

  # Local files are memory-mapped, so `get_tensor` needs little extra host
  # memory. Other storage is read into several full-size host buffers per
  # tensor, so large tensors (~2.7 GB for embed_tokens and lm_head) are
  # streamed from it in chunks instead.
  def stream_in_chunks(name: str) -> bool:
    path, _ = sources[name]
    return state_dict[name].nbytes > CHUNK_BYTES and not os.path.isfile(path)

  print(
      f"Loading {len(state_dict)} tensors from {checkpoint_source()}...",
      flush=True,
  )
  t0 = time.perf_counter()

  # Step 1: stream the large tensors, one at a time.
  streamed = [name for name in state_dict if stream_in_chunks(name)]
  for name in streamed:
    copy_tensor_in_chunks(*sources[name], state_dict[name])

  # Step 2: read the remaining tensors in parallel batches. Each batch is
  # copied to the TPU before the next one is read, so at most `batch_size`
  # tensors are in host memory at once. `copy_` also casts the dtype.
  remaining = [name for name in state_dict if name not in streamed]
  batch_size = 4
  with futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
    for i in range(0, len(remaining), batch_size):
      batch = remaining[i : i + batch_size]
      tensors = pool.map(lambda name: read_tensor(*sources[name]), batch)
      for name, tensor in zip(batch, tensors):
        state_dict[name].copy_(fix_layernorm(name, tensor))

  torch.accelerator.synchronize()
  print(f"Loaded checkpoint in {time.perf_counter() - t0:.2f} s", flush=True)
```

</section>

## 4. 프리필 및 자기회귀 디코딩 루프 { #4-prefill-and-autoregressive-decode-loop }

다음 셀은 `model_generate`를 정의합니다. 이 함수는 입력 프롬프트에 대해 프리필
패스를 실행한 후 토큰 단위 그리디 디코딩을 실행합니다. KV 캐시를 인수로 받으므로
`StaticCache`와 `DynamicCache` 중 어느 것으로든 같은 루프를 실행할 수 있습니다.
이 셀은 토크나이저와 사전 학습된 가중치도 한 번 로드하며, 아래의 두 실행은 같은
`muse_model`을 공유합니다.

```python
PROMPT = "The future of artificial intelligence is"

def model_generate(
    compiled_model,
    config: Config,
    initial_inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizerBase,
    past_key_values,
    prefix: str = "",
) -> str:
  """Runs prefill and a greedy decode loop with the given KV cache."""
  num_decode_steps = config.num_decode_steps
  with torch.inference_mode():
    prefill_start = time.perf_counter()
    logits, past_key_values = compiled_model(
        initial_inputs,
        past_key_values=past_key_values,
    )
    torch.accelerator.synchronize()
    prefill_time_ms = (time.perf_counter() - prefill_start) * 1000
    print(f"{prefix}Prefill done in {prefill_time_ms:.2f} ms", flush=True)

    generated_tokens = [initial_inputs]

    for step in range(num_decode_steps):
      next_token = torch.argmax(logits, dim=-1)
      generated_tokens.append(next_token)

      step_start = time.perf_counter()
      logits, past_key_values = compiled_model(
          next_token,
          past_key_values=past_key_values,
      )
      torch.accelerator.synchronize()
      step_time_ms = (time.perf_counter() - step_start) * 1000

      print(
          f"{prefix}Decode Step {step + 1}/{num_decode_steps}:"
          f" {step_time_ms:.2f} ms",
          flush=True,
      )

    output_tokens = torch.cat([t.to("cpu") for t in generated_tokens], dim=1)

  output_text = str(
      tokenizer.decode(output_tokens[0], skip_special_tokens=True)
  )
  if USE_FULL_MODEL:
    print(f"{prefix}Prompt: {PROMPT}", flush=True)
    print(f"{prefix}Output: {output_text}", flush=True)
  else:
    print(
        f"{prefix}Skipping generated text: only {config.num_layers} layers"
        " are loaded.",
        flush=True,
    )
  return output_text

torch.manual_seed(42)
torch.set_default_dtype(torch.bfloat16)

tokenizer = load_tokenizer()
prompt_ids = tokenizer(PROMPT, add_special_tokens=False).input_ids
if tokenizer.bos_token_id is not None:
  prompt_ids = [tokenizer.bos_token_id] + prompt_ids
inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)

config = Config(
    num_layers=52 if USE_FULL_MODEL else 4,
    prefill_seq_len=inputs.shape[1],
)
with torch.device(device):
  muse_model = MuseGlimmer30B(config)
load_weights_from_checkpoint(muse_model)
```

## 5. `StaticCache`를 사용한 자기회귀 디코딩 { #5-autoregressive-decoding-with-staticcache }

`StaticCache`와 `torch.compile(muse_model, dynamic=False)`를 사용하면 다음과
같이 동작합니다.

*   **프리필**에서 `seq_len = prefill_seq_len`에 대한 그래프를 컴파일합니다.
*   **디코딩 단계 1**에서 `seq_len = 1`에 대한 그래프를 컴파일합니다.
*   **디코딩 단계 2..N**에서는 컴파일된 디코딩 실행 파일을 재사용하며 **재컴파일이
    전혀 없습니다**.

`backend="tpu"`를 전달할 필요는 없습니다. 컴파일된 모델의 입력이 TPU에 있으면
`torch.compile`이 자동으로 TorchTPU 백엔드를 사용합니다.

```python
static_model = torch.compile(muse_model, dynamic=False)
static_cache = StaticCache(
    num_layers=config.num_layers,
    batch_size=inputs.shape[0],
    num_kv_heads=config.num_kv_heads,
    max_cache_len=config.prefill_seq_len + config.num_decode_steps,
    head_dim=config.dhead,
    dtype=torch.bfloat16,
    device=inputs.device,
)
_ = model_generate(
    static_model,
    config,
    inputs,
    tokenizer,
    past_key_values=static_cache,
    prefix="[StaticCache, dynamic=False] ",
)
```

## 6. `DynamicCache`를 사용한 자기회귀 디코딩 { #6-autoregressive-decoding-with-dynamiccache }

`DynamicCache`는 새 키와 값을 `torch.cat`으로 추가하므로 KV 캐시 길이가 디코딩
단계마다 바뀝니다. `dynamic=False`를 사용하면 새 길이마다 재컴파일이 발생합니다.
이 섹션에서는 대신 같은 모델을 동적 형상으로 컴파일하여 컴파일된 그래프 하나로
커지는 캐시를 처리합니다.

### TorchTPU에서 동적 형상 사용 설정 { #enable-dynamic-shapes-on-torchtpu }

TorchTPU는 PyTorch
[동적 형상](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)을
지원하지만, 현재 이 지원은 실험용이므로 직접 사용 설정해야 합니다. TorchTPU는
`torch.compile` 기본값을 `dynamic=None`에서 `dynamic=False`로 변경하므로, 사용
설정하지 않으면 모든 그래프가 정확한 형상에 맞게 특수화됩니다.

*   **사용 설정하려면 `dynamic=None`을 전달하세요.** `dynamic=None`을 사용하면
    Dynamo는 먼저 정적 형상으로 컴파일합니다. 이후 호출에서 크기가 바뀌어
    재컴파일되면 Dynamo가 해당 차원을 동적으로 표시합니다. 이를 *자동 동적
    형상*이라고 합니다. 첫 번째 컴파일부터 특정 차원을 동적으로 만들려면
    `torch._dynamo.mark_dynamic`을 사용하세요.
*   **`dynamic=True`는 사용하지 마세요.** `dynamic=True`는 첫 번째 컴파일부터
    모든 크기를 동적으로 취급합니다. PyTorch는 이 모드를 권장하지 않으므로
    TorchTPU는 자동 동적 형상에 중점을 둡니다. `torch.compile(...,
    dynamic=True)`를 호출하면 `NotImplementedError`가 발생하며, 대신
    `dynamic=None`을 사용하라는 메시지가 표시됩니다.

내부적으로 XLA는 각 그래프를 고정 버퍼 크기의 TPU 프로그램으로 컴파일하므로
메모리 레이아웃을 완전히 최적화하고 연산자 퓨전을 수행할 수 있습니다. 동적 차원이
있는 그래프를 컴파일하기 위해 TorchTPU는 해당 차원에 상한을 지정하고, 상한까지의
모든 크기를 처리하는 단일 프로그램을 컴파일합니다. 실제 크기는 런타임에
전달되므로 상한 이내의 크기는 컴파일된 프로그램을 재사용하고, 상한을 넘는 크기는
상한을 다시 설정하여 재컴파일을 트리거합니다. 여기서 KV 캐시 길이처럼 실제로
바뀌는 차원에만 동적 특성을 제한하면 XLA가 나머지는 모두 정적으로 최적화할 수
있습니다.

다음 셀은 `dynamic=None`으로 모델을 컴파일합니다. 처음 몇 디코딩 단계에서 워밍업
컴파일이 몇 번 발생하며, 그 이후에는 캐시가 커져도 더 이상 재컴파일이 트리거되지
않습니다.

```python
# Reset Dynamo so that this run doesn't reuse the graphs compiled for the
# static run in the previous section.
torch._dynamo.reset()  # pylint: disable=protected-access
dynamic_model = torch.compile(muse_model, dynamic=None)
_ = model_generate(
    dynamic_model,
    config,
    inputs,
    tokenizer,
    past_key_values=DynamicCache(),
    prefix="[DynamicCache, dynamic=None] ",
)
```

## 7. 벤치마크 결과 { #7-benchmark-results }

전체 52레이어 `MuseGlimmer30B` 모델(`num_layers=52`, `num_decode_steps=20`)과
사전 학습된 체크포인트 가중치로 단일 TPU v7x 칩에서 이 노트북을 실행하면, 컴파일
캐시가 비어 있는 상태(콜드 스타트)에서 측정한 지연 시간 프로필은 다음과 같습니다.
52레이어 체크포인트(텐서 627개)를 로드하는 데는 측정 시 스토리지 읽기 처리량에 따라
\~100초에서 \~420초가 걸렸습니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| 측정항목 | `StaticCache` (`dynamic=False`) | `DynamicCache` (`dynamic=None`) |
| --- | --- | --- |
| **프리필 시간(ms)** | 21024.81 | 18256.18 |
| **디코딩 단계 1(ms)** | 19756.86 | 19969.41 |
| **디코딩 단계 2(ms)** | 24.66 | 21985.62 |
| **디코딩 단계 3(ms)** | 24.29 | 4908.23 |
| **디코딩 단계 4(ms)** | 23.69 | 25.57 |
| **디코딩 단계 5(ms)** | 23.59 | 24.83 |
| **디코딩 단계 6(ms)** | 23.58 | 24.15 |
| **디코딩 단계 7(ms)** | 23.56 | 24.10 |
| **디코딩 단계 8(ms)** | 23.43 | 23.97 |
| **디코딩 단계 9(ms)** | 23.38 | 24.04 |
| **디코딩 단계 10(ms)** | 23.42 | 24.03 |
| **디코딩 단계 11(ms)** | 23.48 | 24.06 |
| **디코딩 단계 12(ms)** | 23.52 | 24.07 |
| **디코딩 단계 13(ms)** | 23.53 | 24.04 |
| **디코딩 단계 14(ms)** | 23.41 | 24.05 |
| **디코딩 단계 15(ms)** | 23.43 | 24.00 |
| **디코딩 단계 16(ms)** | 23.42 | 24.01 |
| **디코딩 단계 17(ms)** | 23.42 | 23.95 |
| **디코딩 단계 18(ms)** | 23.42 | 23.96 |
| **디코딩 단계 19(ms)** | 23.34 | 23.99 |
| **디코딩 단계 20(ms)** | 23.41 | 24.05 |
| **토큰당 평균 디코딩 시간(워밍업 제외)(ms)** | **23.58**(단계 2–20) | **24.17**(단계 4–20) |

<!-- mdformat on -->

*   **`StaticCache`: 단계 1 이후 디코딩 재컴파일 없음.** 프리필(\~21.0초)과
    디코딩 단계 1(\~19.8초)에는 각각 프롬프트 길이(이 프롬프트와 토크나이저의
    경우 `seq_len = 7` 토큰)와 `seq_len = 1`에 대한 일회성 XLA 컴파일이
    포함됩니다. **디코딩 단계 2부터는** 모든 토큰이 재컴파일 없이
    **\~23.6 ms**에 실행됩니다.
*   **`DynamicCache`: `StaticCache`보다 워밍업 단계가 2개 더 많고, 그 이후에는
    재컴파일 없음.** 프리필과 디코딩 단계 1은 정적 형상으로 컴파일됩니다. 디코딩
    단계 2(\~22.0초)는 KV 캐시 길이가 처음으로 바뀌는 단계이므로 Dynamo가 이를
    동적으로 표시하고 가변 캐시 길이를 처리하는 그래프를 컴파일합니다. 디코딩
    단계 3(\~4.9초)에도 아직 일부 컴파일이 포함됩니다. **디코딩 단계 4부터는**
    모든 토큰이 **\~24.2 ms**에 실행됩니다.
*   **반복 실행 시 컴파일이 더 빠릅니다.** TorchTPU 백엔드는 컴파일된 프로그램을
    PyTorch의 디스크 기반 컴파일 캐시(`AOTAutogradCache`)에 저장하며, 이 캐시는
    같은 머신에서 커널을 다시 시작해도 유지됩니다. 모델 코드를 변경하지 않고 이
    노트북을 다시 실행하면 프리필과 디코딩 단계 1은 재컴파일하지 않고 캐시에서
    로드됩니다. 토큰당 디코딩 시간은 변하지 않습니다. 위 표처럼 콜드 스타트
    컴파일을 측정하려면 `torch.compile` 전에
    `torch._dynamo.reset()` 및
    `torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache.clear()`를
    실행하세요.
*   **샘플 출력(전체 52레이어 모델, 디코딩 단계 100회):**

```text
Prompt: The future of artificial intelligence is
Output: The future of artificial intelligence is here, and it’s changing the way we work, learn, and live. From self-driving cars to virtual assistants, AI is transforming industries and revolutionizing the way we interact with technology. But what does the future hold for AI? In this article, we’ll explore the latest trends and predictions for the future of artificial intelligence.

## The Rise of AI

Artificial intelligence has been around for decades, but it’s only in recent years that we’ve seen a significant increase in its use and development.
```

> **Note:** `USE_FULL_MODEL = False`(노트북 상단 참고)이면 디코더 레이어 52개 중
> 처음 4개만 로드되므로 노트북에서 생성된 텍스트를 출력하지 않습니다.
