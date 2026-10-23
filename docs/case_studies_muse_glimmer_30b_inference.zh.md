# Muse Glimmer 30B：自回归解码 { #muse-glimmer-30b-autoregressive-decoding }

<!-- Link to colab and to generated notebook -->

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_muse_glimmer_30b_inference.ipynb)

本指南介绍如何使用 TorchTPU 在 TPU 上进行自回归 LLM 推理，所用的是 **Muse Glimmer** 架构和预训练的 Hugging Face 权重。本指南比较共享同一个模型的两种 KV 缓存：一种是预分配的 **`StaticCache`**，使用静态形状（`dynamic=False`）编译；另一种是不断增长的 **`DynamicCache`**，使用动态形状（`dynamic=None`）编译。

如需查看此示例的独立脚本版本（其中为静态缓存和动态缓存提供了单独的运行程序），请参阅 [`examples/muse_glimmer_30b/llm_inference`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference)。

> **内存要求**：默认情况下（`USE_FULL_MODEL = False`），该笔记本只加载 52 个解码器层中的前 4 层（权重为 \~9.3 GB），因此可以在 HBM 较小的 TPU 上运行，但生成的文本没有实际意义。如需加载完整模型（TPU HBM 中的权重为 \~55.7 GB）并生成有意义的文本，请在下一个代码单元中设置 `USE_FULL_MODEL = True`。

```python
# True loads all 52 decoder layers (~55.7 GB). False loads only the first 4
# layers (~9.3 GB): the model still compiles and runs, but the generated text
# isn't meaningful.
USE_FULL_MODEL = False
```

## 1. 用于 XLA 编译的 KV 缓存 { #1-kv-caching-for-xla-compilation }

在自回归 LLM 推理中，生成过程分为两个阶段：

*   **预填充（prefill）**：模型并行处理整个提示 `[B, S]`，并填充 KV 缓存 `[B, H_kv, S, D]`（批次大小 `B`、提示长度 `S`、`H_kv` 个 KV 头、头维度 `D`）。
*   **解码（decode）**：每个步骤 `t` 输入一个新 token `[B, 1]`，将其 key 和 value 追加到缓存中，并对全部 `S + t` 个位置计算注意力。

**`StaticCache`** 通过让所有张量形状在各解码步骤之间保持不变，避免在 `dynamic=False` 下进行解码重新编译：

1.  **固定大小的缓冲区**：`keys` 和 `values` 预分配为 `[B, H_kv,
    max_cache_len, D]`。
2.  **设备端长度**：缓存长度是一个原地更新的 0 维 TPU 张量，而不是 Dynamo 会对其设置守卫的 Python `int`。
3.  **原地写入，固定形状的掩码**：新条目通过 `index_copy_` 写入；尚未写入的位置使用 `-inf` 进行掩码。
4.  **两次编译**：一次用于预填充，一次用于解码，之后的每个步骤都会重复使用它们。

**`DynamicCache`** 采用相反的方法，它也是 Hugging Face 模型中的默认缓存：它使用 `torch.cat` 追加每个步骤的 key 和 value，因此缓存恰好包含 `S + t` 个位置。这种方法更简单，也不占用填充内存，但缓存大小在每个步骤都会变化，因此它依赖动态形状（`dynamic=None`）来避免为每个 token 重新编译。

## 2. 模型和 KV 缓存实现 { #2-model-and-kv-cache-implementation }

下一个代码单元实现了 `StaticCache`、`DynamicCache` 和 `MuseGlimmer30B` 解码器架构。该架构每四个块中交替使用三个带 RoPE 的滑动窗口注意力（SWA）层和一个完整的 NoPE 因果注意力层。

同样的模型和缓存也以可重用模块的形式提供，位于 [`model.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/model.py) 和 [`kv_cache.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/kv_cache.py) 中。

<section class="zippy" markdown="1">

**导入和设备设置**

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

**KV 缓存和模型实现**

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

## 3. 加载预训练权重和分词器 { #3-loading-pretrained-weights-and-tokenizer }

本部分定义如何将 `Muse-Glimmer-30B` 分词器和预训练权重加载到 `MuseGlimmer30B` 中。检查点（checkpoint）可以来自以下两个位置之一：

1.  **检查点的本地副本**：将 `MUSE_GLIMMER_CHECKPOINT_DIR` 环境变量设置为包含检查点文件的目录。在笔记本中，您可以在运行下一个代码单元之前运行 `os.environ["MUSE_GLIMMER_CHECKPOINT_DIR"] =
    "/path/to/checkpoint"`。如果设置了该变量但目录不存在，加载会失败，而不会回退为下载。
2.  **Hugging Face Hub（默认）**：如果未设置 `MUSE_GLIMMER_CHECKPOINT_DIR`，则从 Hub 代码库 `MODEL_NAME`（`meta-models/Muse-Glimmer-30B`）下载文件。下载的文件会被缓存，因此之后的运行会重复使用这些文件。

无论使用哪种方式，权重的加载方式都相同：

*   **只读取已配置的层。** 只为 `Config(num_layers=...)` 中的层加载张量，因此 `USE_FULL_MODEL = False`（请参阅笔记本顶部）会加载 4 层，而不是 52 层。
*   **张量直接复制到模型中。** 文本解码器的张量直接从 `.safetensors` 文件复制到 `m.state_dict()` 中，无需构建 Hugging Face 模型。
*   **并行读取张量。** 系统会并发读取小批量的张量，并在读取下一批之前将其复制到 TPU，从而降低主机内存用量。

<section class="zippy" markdown="1">

**检查点和分词器加载**

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

## 4. 预填充和自回归解码循环 { #4-prefill-and-autoregressive-decode-loop }

下一个代码单元定义了 `model_generate`，它会对输入提示运行预填充，然后逐个 token 进行贪婪解码。它将 KV 缓存作为参数，因此同一个循环既可以使用 `StaticCache`，也可以使用 `DynamicCache` 运行。该代码单元还会一次性加载分词器和预训练权重；下面的两次运行共享同一个 `muse_model`。

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

## 5. 使用 `StaticCache` 进行自回归解码 { #5-autoregressive-decoding-with-staticcache }

使用 `StaticCache` 和 `torch.compile(muse_model, dynamic=False)` 时：

*   **预填充**会为 `seq_len = prefill_seq_len` 编译计算图。
*   **解码步骤 1** 会为 `seq_len = 1` 编译计算图。
*   **解码步骤 2..N** 会重复使用已编译的解码可执行文件，**无需任何重新编译**。

您无需传递 `backend="tpu"`：当已编译模型的输入位于 TPU 上时，`torch.compile` 会自动使用 TorchTPU 后端。

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

## 6. 使用 `DynamicCache` 进行自回归解码 { #6-autoregressive-decoding-with-dynamiccache }

`DynamicCache` 使用 `torch.cat` 追加每个新的 key 和 value，因此 KV 缓存长度在每个解码步骤都会变化。在 `dynamic=False` 下，每个新长度都会触发一次重新编译。因此，本部分改为使用动态形状编译同一个模型，让一个已编译的计算图即可处理不断增长的缓存。

### 在 TorchTPU 上启用动态形状 { #enable-dynamic-shapes-on-torchtpu }

TorchTPU 支持 PyTorch [动态形状](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)，但这项支持目前处于实验阶段，因此需要选择启用。TorchTPU 将 `torch.compile` 的默认值从 `dynamic=None` 更改为 `dynamic=False`，因此除非您选择启用，否则每个计算图都会针对其确切形状进行特化：

*   **传递 `dynamic=None` 以选择启用。** 使用 `dynamic=None` 时，Dynamo 首先使用静态形状进行编译。之后如果某次调用因为某个大小发生变化而重新编译，Dynamo 会将该维度标记为动态维度。这称为*自动动态形状*。如需从第一次编译起就让特定维度成为动态维度，请使用 `torch._dynamo.mark_dynamic`。
*   **不要使用 `dynamic=True`。** `dynamic=True` 从第一次编译起就将每个大小都视为动态。PyTorch 不建议使用此模式，因此 TorchTPU 专注于自动动态形状。`torch.compile(...,
    dynamic=True)`
    会引发 `NotImplementedError`，提示您改用 `dynamic=None`。

在底层，XLA 会将每个计算图编译为具有固定缓冲区大小的 TPU 程序，从而可以全面优化内存布局并执行算子融合。为了编译带有动态维度的计算图，TorchTPU 会为该维度设定一个上限，并编译单个程序来处理不超过该上限的所有大小。实际大小在运行时传入，因此上限以内的大小会重复使用已编译的程序，而超出上限的大小会触发重新编译，并重新确定上限。只让实际发生变化的维度（例如此处的 KV 缓存长度）保持动态，可以让 XLA 将其他所有部分作为静态部分进行优化。

下一个代码单元使用 `dynamic=None` 编译模型。在最初的几个解码步骤中，预计会有几次预热编译；之后，不断增长的缓存不会再触发任何重新编译。

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

## 7. 基准测试结果 { #7-benchmark-results }

在单个 TPU v7x 芯片上，使用完整的 52 层 `MuseGlimmer30B` 模型（`num_layers=52`、`num_decode_steps=20`）和预训练检查点权重运行此笔记本，会得到以下延迟分布情况。测量时编译缓存为空（冷启动）。在我们的运行中，加载 52 层检查点（627 个张量）耗时 \~100 秒到 \~420 秒，具体取决于存储读取吞吐量。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| 指标 | `StaticCache`（`dynamic=False`） | `DynamicCache`（`dynamic=None`） |
| --- | --- | --- |
| **预填充时间（ms）** | 21024.81 | 18256.18 |
| **解码步骤 1（ms）** | 19756.86 | 19969.41 |
| **解码步骤 2（ms）** | 24.66 | 21985.62 |
| **解码步骤 3（ms）** | 24.29 | 4908.23 |
| **解码步骤 4（ms）** | 23.69 | 25.57 |
| **解码步骤 5（ms）** | 23.59 | 24.83 |
| **解码步骤 6（ms）** | 23.58 | 24.15 |
| **解码步骤 7（ms）** | 23.56 | 24.10 |
| **解码步骤 8（ms）** | 23.43 | 23.97 |
| **解码步骤 9（ms）** | 23.38 | 24.04 |
| **解码步骤 10（ms）** | 23.42 | 24.03 |
| **解码步骤 11（ms）** | 23.48 | 24.06 |
| **解码步骤 12（ms）** | 23.52 | 24.07 |
| **解码步骤 13（ms）** | 23.53 | 24.04 |
| **解码步骤 14（ms）** | 23.41 | 24.05 |
| **解码步骤 15（ms）** | 23.43 | 24.00 |
| **解码步骤 16（ms）** | 23.42 | 24.01 |
| **解码步骤 17（ms）** | 23.42 | 23.95 |
| **解码步骤 18（ms）** | 23.42 | 23.96 |
| **解码步骤 19（ms）** | 23.34 | 23.99 |
| **解码步骤 20（ms）** | 23.41 | 24.05 |
| **每个 token 的平均解码时间（不含预热）（ms）** | **23.58**（步骤 2–20） | **24.17**（步骤 4–20） |

<!-- mdformat on -->

*   **`StaticCache`：步骤 1 之后解码不再重新编译。** 预填充（\~21.0 秒）和解码步骤 1（\~19.8 秒）分别包含针对提示长度（对于此提示和分词器，`seq_len = 7` 个 token）和 `seq_len = 1` 的一次性 XLA 编译。从**解码步骤 2 开始**，每个 token 的执行时间为 **\~23.6 ms**，且不会重新编译。
*   **`DynamicCache`：比 `StaticCache` 多两个预热步骤，之后不再重新编译。** 预填充和解码步骤 1 针对静态形状进行编译。解码步骤 2（\~22.0 秒）是 KV 缓存长度第一次发生变化的步骤，因此 Dynamo 会将其标记为动态，并编译一个可以处理可变缓存长度的计算图；解码步骤 3（\~4.9 秒）仍包含部分编译。从**解码步骤 4 开始**，每个 token 的执行时间为 **\~24.2 ms**。
*   **重复运行时编译更快。** TorchTPU 后端会将已编译的程序保存到 PyTorch 的磁盘编译缓存（`AOTAutogradCache`）中，该缓存在同一台机器上的内核重启后仍会保留。如果您在模型代码未更改的情况下重新运行此笔记本，预填充和解码步骤 1 会从缓存中加载，而不会重新编译；每个 token 的解码时间保持不变。如需像上表那样测量冷启动编译，请在运行 `torch.compile` 之前运行 `torch._dynamo.reset()` 和 `torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache.clear()`。
*   **示例输出（完整的 52 层模型，100 个解码步骤）：**

```text
Prompt: The future of artificial intelligence is
Output: The future of artificial intelligence is here, and it’s changing the way we work, learn, and live. From self-driving cars to virtual assistants, AI is transforming industries and revolutionizing the way we interact with technology. But what does the future hold for AI? In this article, we’ll explore the latest trends and predictions for the future of artificial intelligence.

## The Rise of AI

Artificial intelligence has been around for decades, but it’s only in recent years that we’ve seen a significant increase in its use and development.
```

> **Note:** 当 `USE_FULL_MODEL = False`（请参阅笔记本顶部）时，只会加载 52 个解码器层中的前 4 层，因此笔记本会跳过输出生成的文本。
