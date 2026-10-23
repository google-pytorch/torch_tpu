# Muse Glimmer 30B: 自己回帰デコード { #muse-glimmer-30b-autoregressive-decoding }

<!-- Link to colab and to generated notebook -->

[![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_muse_glimmer_30b_inference.ipynb)

このガイドでは、**Muse Glimmer** アーキテクチャと Hugging Face の事前トレーニング済みの重みを使って、TorchTPU で TPU 上の LLM の自己回帰推論を行う方法を説明します。1 つのモデルを共有する 2 つの KV キャッシュを比較します。静的なシェイプ（`dynamic=False`）でコンパイルする事前割り当て型の **`StaticCache`** と、動的シェイプ（`dynamic=None`）でコンパイルする、サイズが増えていく **`DynamicCache`** です。

静的キャッシュと動的キャッシュ用に別々のランナーを備えた、このサンプルのスタンドアロン スクリプト版については、[`examples/muse_glimmer_30b/llm_inference`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference) をご覧ください。

> **メモリ要件:** デフォルト（`USE_FULL_MODEL = False`）では、ノートブックは 52 個のデコーダ レイヤのうち最初の 4 個（重みは \~9.3 GB）だけを読み込むため、HBM が少ない TPU にも収まりますが、生成されるテキストには意味がありません。モデル全体（TPU HBM 上の重みは \~55.7 GB）を読み込んで意味のあるテキストを生成するには、次のセルで `USE_FULL_MODEL = True` を設定してください。

```python
# True loads all 52 decoder layers (~55.7 GB). False loads only the first 4
# layers (~9.3 GB): the model still compiles and runs, but the generated text
# isn't meaningful.
USE_FULL_MODEL = False
```

## 1. XLA コンパイルのための KV キャッシュ { #1-kv-caching-for-xla-compilation }

LLM の自己回帰推論では、生成は 2 つのフェーズで進みます。

*   **プレフィル:** モデルはプロンプト全体 `[B, S]` を並列に処理し、KV キャッシュ `[B, H_kv, S, D]`（バッチ `B`、プロンプト長 `S`、KV ヘッド数 `H_kv`、ヘッドの次元 `D`）を埋めます。
*   **デコード:** 各ステップ `t` では新しいトークン `[B, 1]` を 1 つ入力し、そのキーとバリューをキャッシュに追加して、`S + t` 個のすべての位置にアテンションを適用します。

**`StaticCache`** は、デコード ステップ間ですべてのテンソルのシェイプを一定に保つことで、`dynamic=False` でのデコード時の再コンパイルを回避します。

1.  **固定サイズのバッファ:** `keys` と `values` は `[B, H_kv,
    max_cache_len, D]` として事前に割り当てられます。
2.  **デバイス上の長さ:** キャッシュの長さは、Dynamo がガードする Python の `int` ではなく、インプレースで更新される 0 次元の TPU テンソルです。
3.  **インプレース書き込みと固定シェイプのマスク:** 新しいエントリは `index_copy_` で書き込まれ、まだ書き込まれていない位置は `-inf` でマスクされます。
4.  **2 回のコンパイル:** プレフィル用とデコード用に 1 回ずつコンパイルし、以降のすべてのステップで再利用します。

**`DynamicCache`** は逆のアプローチを取り、Hugging Face のモデルではこれがデフォルトのキャッシュです。各ステップのキーとバリューを `torch.cat` で追加するため、キャッシュには正確に `S + t` 個の位置が保持されます。この方法はシンプルで、パディング用のメモリも使いませんが、キャッシュのサイズがステップごとに変わるため、トークンごとの再コンパイルを避けるには動的シェイプ（`dynamic=None`）が必要です。

## 2. モデルと KV キャッシュの実装 { #2-model-and-kv-cache-implementation }

次のセルでは、`StaticCache`、`DynamicCache`、`MuseGlimmer30B` デコーダ アーキテクチャを実装します。このアーキテクチャでは、4 ブロックごとに、RoPE を使う 3 つのスライディング ウィンドウ アテンション（SWA）レイヤと、1 つの NoPE の完全な因果的アテンション レイヤが交互に配置されます。

同じモデルとキャッシュは、[`model.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/model.py) と [`kv_cache.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/kv_cache.py) で再利用可能なモジュールとしても提供されています。

<section class="zippy" markdown="1">

**インポートとデバイスのセットアップ**

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

**KV キャッシュとモデルの実装**

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

## 3. 事前トレーニング済みの重みとトークナイザの読み込み { #3-loading-pretrained-weights-and-tokenizer }

このセクションでは、`Muse-Glimmer-30B` のトークナイザと事前トレーニング済みの重みを `MuseGlimmer30B` に読み込む方法を定義します。チェックポイントは、次の 2 つのいずれかから取得できます。

1.  **チェックポイントのローカルコピー:** `MUSE_GLIMMER_CHECKPOINT_DIR` 環境変数に、チェックポイント ファイルを含むディレクトリを設定します。ノートブックでは、次のセルの前に `os.environ["MUSE_GLIMMER_CHECKPOINT_DIR"] =
    "/path/to/checkpoint"` を実行できます。変数が設定されているのにディレクトリが存在しない場合、ダウンロードにフォールバックせずに読み込みが失敗します。
2.  **Hugging Face Hub（デフォルト）:** `MUSE_GLIMMER_CHECKPOINT_DIR` が設定されていない場合、ファイルは Hub のリポジトリ `MODEL_NAME`（`meta-models/Muse-Glimmer-30B`）からダウンロードされます。ダウンロードはキャッシュされるため、以降の実行で再利用されます。

どちらの場合も、重みは同じ方法で読み込まれます。

*   **設定したレイヤだけを読み込みます。** テンソルは `Config(num_layers=...)` のレイヤについてのみ読み込まれるため、`USE_FULL_MODEL = False`（ノートブックの冒頭を参照）では 52 個ではなく 4 個のレイヤが読み込まれます。
*   **テンソルをモデルに直接コピーします。** テキスト デコーダのテンソルは、Hugging Face のモデルを構築せずに、`.safetensors` ファイルから `m.state_dict()` に直接コピーされます。
*   **テンソルを並列に読み込みます。** テンソルの小さなバッチを同時に読み込み、次のバッチを読み込む前に TPU にコピーするため、ホストのメモリ使用量を低く抑えられます。

<section class="zippy" markdown="1">

**チェックポイントとトークナイザの読み込み**

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

## 4. プレフィルと自己回帰デコードのループ { #4-prefill-and-autoregressive-decode-loop }

次のセルでは `model_generate` を定義します。この関数は、入力プロンプトに対してプレフィル パスを実行し、その後トークンごとの貪欲（greedy）デコードを行います。KV キャッシュを引数として受け取るため、`StaticCache` と `DynamicCache` のどちらでも同じループを実行できます。このセルではトークナイザと事前トレーニング済みの重みも 1 回だけ読み込みます。以下の 2 つの実行は、同じ `muse_model` を共有します。

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

## 5. `StaticCache` を使った自己回帰デコード { #5-autoregressive-decoding-with-staticcache }

`StaticCache` と `torch.compile(muse_model, dynamic=False)` を使う場合:

*   **プレフィル**では、`seq_len = prefill_seq_len` 用のグラフをコンパイルします。
*   **デコード ステップ 1** では、`seq_len = 1` 用のグラフをコンパイルします。
*   **デコード ステップ 2..N** では、コンパイル済みのデコード用の実行可能プログラムを再利用し、**再コンパイルは一切発生しません**。

`backend="tpu"` を渡す必要はありません。コンパイル済みモデルへの入力が TPU 上にある場合、`torch.compile` は自動的に TorchTPU バックエンドを使います。

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

## 6. `DynamicCache` を使った自己回帰デコード { #6-autoregressive-decoding-with-dynamiccache }

`DynamicCache` は新しいキーとバリューを `torch.cat` で追加するため、KV キャッシュの長さはデコード ステップごとに変わります。`dynamic=False` では、新しい長さになるたびに再コンパイルが発生します。このセクションでは、代わりに動的シェイプで同じモデルをコンパイルし、1 つのコンパイル済みグラフで増えていくキャッシュを処理します。

### TorchTPU で動的シェイプを有効にする { #enable-dynamic-shapes-on-torchtpu }

TorchTPU は PyTorch の[動的シェイプ](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)をサポートしていますが、このサポートは現在試験運用版のため、オプトイン方式です。TorchTPU は `torch.compile` のデフォルトを `dynamic=None` から `dynamic=False` に変更しているため、オプトインしない限り、すべてのグラフは正確なシェイプに特化されます。

*   **オプトインするには `dynamic=None` を渡します。** `dynamic=None` では、Dynamo はまず静的なシェイプでコンパイルします。後の呼び出しでサイズが変わって再コンパイルが発生すると、Dynamo はその次元を動的としてマークします。これを*自動動的シェイプ*（automatic dynamic shapes）と呼びます。特定の次元を最初のコンパイルから動的にするには、`torch._dynamo.mark_dynamic` を使います。
*   **`dynamic=True` は使わないでください。** `dynamic=True` は、最初のコンパイルからすべてのサイズを動的として扱います。PyTorch はこのモードを推奨していないため、TorchTPU は自動動的シェイプに重点を置いています。`torch.compile(...,
    dynamic=True)`
    を使うと、代わりに `dynamic=None` を使うよう求める `NotImplementedError` が発生します。

内部的には、XLA は各グラフを固定のバッファサイズを持つ TPU プログラムにコンパイルします。これにより、メモリ レイアウトを完全に最適化し、演算子の融合を行えます。動的な次元を持つグラフをコンパイルするために、TorchTPU はその次元に上限を設け、上限までのすべてのサイズを処理する単一のプログラムをコンパイルします。実際のサイズは実行時に渡されるため、上限内のサイズではコンパイル済みのプログラムが再利用され、上限を超えるサイズでは上限を設定し直して再コンパイルが行われます。動的性をここでの KV キャッシュの長さのように実際に変化する次元だけに限定すると、XLA はそれ以外のすべてを静的なものとして最適化できます。

次のセルでは、`dynamic=None` でモデルをコンパイルします。最初のいくつかのデコード ステップでは、ウォームアップのコンパイルが数回発生します。その後は、キャッシュが増えても再コンパイルは発生しません。

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

## 7. ベンチマーク結果 { #7-benchmark-results }

52 レイヤの完全な `MuseGlimmer30B` モデル（`num_layers=52`、`num_decode_steps=20`）と事前トレーニング済みのチェックポイントの重みを使い、単一の TPU v7x チップでこのノートブックを実行すると、次のレイテンシ プロファイルが得られます。測定はコンパイル キャッシュが空の状態（コールド スタート）で行いました。52 レイヤのチェックポイント（627 個のテンソル）の読み込みには、ストレージの読み取りスループットに応じて、私たちの実行では \~100 秒から \~420 秒かかりました。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| 指標 | `StaticCache`（`dynamic=False`） | `DynamicCache`（`dynamic=None`） |
| --- | --- | --- |
| **プレフィル時間（ms）** | 21024.81 | 18256.18 |
| **デコード ステップ 1（ms）** | 19756.86 | 19969.41 |
| **デコード ステップ 2（ms）** | 24.66 | 21985.62 |
| **デコード ステップ 3（ms）** | 24.29 | 4908.23 |
| **デコード ステップ 4（ms）** | 23.69 | 25.57 |
| **デコード ステップ 5（ms）** | 23.59 | 24.83 |
| **デコード ステップ 6（ms）** | 23.58 | 24.15 |
| **デコード ステップ 7（ms）** | 23.56 | 24.10 |
| **デコード ステップ 8（ms）** | 23.43 | 23.97 |
| **デコード ステップ 9（ms）** | 23.38 | 24.04 |
| **デコード ステップ 10（ms）** | 23.42 | 24.03 |
| **デコード ステップ 11（ms）** | 23.48 | 24.06 |
| **デコード ステップ 12（ms）** | 23.52 | 24.07 |
| **デコード ステップ 13（ms）** | 23.53 | 24.04 |
| **デコード ステップ 14（ms）** | 23.41 | 24.05 |
| **デコード ステップ 15（ms）** | 23.43 | 24.00 |
| **デコード ステップ 16（ms）** | 23.42 | 24.01 |
| **デコード ステップ 17（ms）** | 23.42 | 23.95 |
| **デコード ステップ 18（ms）** | 23.42 | 23.96 |
| **デコード ステップ 19（ms）** | 23.34 | 23.99 |
| **デコード ステップ 20（ms）** | 23.41 | 24.05 |
| **トークンあたりの平均デコード時間（ウォームアップを除く）（ms）** | **23.58**（ステップ 2～20） | **24.17**（ステップ 4～20） |

<!-- mdformat on -->

*   **`StaticCache`: ステップ 1 以降はデコード時の再コンパイルがありません。** プレフィル（\~21.0 秒）とデコード ステップ 1（\~19.8 秒）には、それぞれプロンプト長（このプロンプトとトークナイザでは `seq_len = 7` トークン）と `seq_len = 1` 用の 1 回限りの XLA コンパイルが含まれます。**デコード ステップ 2 以降**は、すべてのトークンが再コンパイルなしで **\~23.6 ms** で実行されます。
*   **`DynamicCache`: `StaticCache` よりウォームアップ ステップが 2 つ多く、その後は再コンパイルがありません。** プレフィルとデコード ステップ 1 は静的なシェイプでコンパイルされます。デコード ステップ 2（\~22.0 秒）は KV キャッシュの長さが変わる最初のステップであるため、Dynamo はその次元を動的としてマークし、可変長のキャッシュを処理するグラフをコンパイルします。デコード ステップ 3（\~4.9 秒）にもまだ一部のコンパイルが含まれます。**デコード ステップ 4 以降**は、すべてのトークンが **\~24.2 ms** で実行されます。
*   **2 回目以降の実行ではコンパイルが速くなります。** TorchTPU バックエンドは、コンパイル済みのプログラムを PyTorch のディスク上のコンパイル キャッシュ（`AOTAutogradCache`）に保存します。このキャッシュは、同じマシン上でカーネルを再起動しても保持されます。モデルのコードを変更せずにこのノートブックを再実行すると、プレフィルとデコード ステップ 1 は再コンパイルせずにキャッシュから読み込まれます。トークンあたりのデコード時間は変わりません。上の表のようにコールド スタートでのコンパイルを測定するには、`torch.compile` の前に `torch._dynamo.reset()` と `torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache.clear()` を実行してください。
*   **出力例（52 レイヤの完全なモデル、100 デコード ステップ）:**

```text
Prompt: The future of artificial intelligence is
Output: The future of artificial intelligence is here, and it’s changing the way we work, learn, and live. From self-driving cars to virtual assistants, AI is transforming industries and revolutionizing the way we interact with technology. But what does the future hold for AI? In this article, we’ll explore the latest trends and predictions for the future of artificial intelligence.

## The Rise of AI

Artificial intelligence has been around for decades, but it’s only in recent years that we’ve seen a significant increase in its use and development.
```

> **Note:** `USE_FULL_MODEL = False`（ノートブックの冒頭を参照）では、52 個のデコーダ レイヤのうち最初の 4 個だけが読み込まれるため、ノートブックは生成されたテキストの出力を省略します。
