# Muse Glimmer 30B: decodificação autorregressiva { #muse-glimmer-30b-autoregressive-decoding }

<!-- Link to colab and to generated notebook -->

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_muse_glimmer_30b_inference.ipynb)

Este guia mostra a inferência autorregressiva de LLM na TPU com o TorchTPU,
usando a arquitetura **Muse Glimmer** e pesos pré-treinados do Hugging Face.
Ele compara dois KV caches que compartilham o mesmo modelo: um
**`StaticCache`** pré-alocado, compilado com formatos estáticos
(`dynamic=False`), e um **`DynamicCache`** que cresce, compilado com formatos
dinâmicos (`dynamic=None`).

Para uma versão deste exemplo como script independente, com executores
separados para os caches estático e dinâmico, consulte
[`examples/muse_glimmer_30b/llm_inference`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference).

> **Requisitos de memória:** por padrão (`USE_FULL_MODEL = False`), o notebook
> carrega apenas as 4 primeiras das 52 camadas do decodificador (\~9,3 GB de
> pesos), então ele cabe em TPUs com menos HBM, mas o texto gerado não faz
> sentido. Para carregar o modelo completo (\~55,7 GB de pesos na HBM da TPU) e
> gerar texto com sentido, defina `USE_FULL_MODEL = True` na próxima célula.

```python
# True loads all 52 decoder layers (~55.7 GB). False loads only the first 4
# layers (~9.3 GB): the model still compiles and runs, but the generated text
# isn't meaningful.
USE_FULL_MODEL = False
```

## 1. KV cache para compilação com XLA { #1-kv-caching-for-xla-compilation }

Na inferência autorregressiva de LLM, a geração acontece em duas fases:

*   **Prefill:** o modelo processa o prompt inteiro `[B, S]` em paralelo e
    preenche o KV cache `[B, H_kv, S, D]` (batch `B`, comprimento do prompt
    `S`, `H_kv` heads de KV, dimensão do head `D`).
*   **Decode:** cada etapa `t` recebe um novo token `[B, 1]`, adiciona a key e
    o value dele ao cache e aplica a atenção sobre todas as `S + t` posições.

O **`StaticCache`** evita recompilações no decode com `dynamic=False` ao manter
constantes todos os formatos de tensor entre as etapas de decode:

1.  **Buffers de tamanho fixo:** `keys` e `values` são pré-alocados como
    `[B, H_kv,
    max_cache_len, D]`.
2.  **Comprimento no dispositivo:** o comprimento do cache é um tensor 0-D na
    TPU atualizado in-place, e não um `int` do Python sobre o qual o Dynamo
    cria guardas.
3.  **Gravações in-place, máscaras de formato fixo:** as novas entradas são
    gravadas com `index_copy_`. As posições ainda não gravadas são mascaradas
    com `-inf`.
4.  **Duas compilações:** uma para o prefill e outra para o decode, reutilizadas
    por todas as etapas seguintes.
O **`DynamicCache`** segue a abordagem oposta e é o cache padrão nos modelos do
Hugging Face: ele adiciona as keys e os values de cada etapa com `torch.cat`,
então o cache contém exatamente `S + t` posições. Isso é mais simples e não usa
memória com preenchimento (padding), mas o tamanho do cache muda a cada etapa,
então ele depende de formatos dinâmicos (`dynamic=None`) para evitar recompilar
a cada token.

## 2. Implementação do modelo e do KV cache { #2-model-and-kv-cache-implementation }

A próxima célula implementa `StaticCache`, `DynamicCache` e a arquitetura de
decodificador `MuseGlimmer30B`, que intercala três camadas de atenção com
janela deslizante (sliding window attention, SWA) com RoPE e uma camada de
atenção causal completa NoPE a cada quatro blocos.

O mesmo modelo e os mesmos caches também estão disponíveis como módulos
reutilizáveis em
[`model.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/model.py)
e
[`kv_cache.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/kv_cache.py).

<section class="zippy" markdown="1">

**Importações e configuração do dispositivo**

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

**KV caches e implementação do modelo**

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

## 3. Carregamento dos pesos pré-treinados e do tokenizador { #3-loading-pretrained-weights-and-tokenizer }

Esta seção define como carregar o tokenizador e os pesos pré-treinados do
`Muse-Glimmer-30B` no `MuseGlimmer30B`. O checkpoint pode vir de um destes dois
lugares:

1.  **Uma cópia local do checkpoint:** defina a variável de ambiente
    `MUSE_GLIMMER_CHECKPOINT_DIR` como um diretório que contenha os arquivos
    do checkpoint. Em um notebook, você pode executar
    `os.environ["MUSE_GLIMMER_CHECKPOINT_DIR"] =
    "/path/to/checkpoint"` antes da próxima célula. Se a variável estiver
    definida, mas o diretório não existir, o carregamento falha em vez de
    recorrer a um download.
2.  **O Hugging Face Hub (padrão):** se `MUSE_GLIMMER_CHECKPOINT_DIR` não
    estiver definida, os arquivos são baixados do repositório do Hub
    `MODEL_NAME` (`meta-models/Muse-Glimmer-30B`). Os downloads ficam em cache,
    então as execuções seguintes os reutilizam.

Em ambos os casos, os pesos são carregados da mesma forma:

*   **Somente as camadas configuradas são lidas.** Os tensores são carregados
    apenas para as camadas em `Config(num_layers=...)`, então
    `USE_FULL_MODEL = False` (veja o início do notebook) carrega 4 camadas em
    vez de 52.
*   **Os tensores são copiados diretamente para o modelo.** Os tensores do
    decodificador de texto são copiados dos arquivos `.safetensors`
    diretamente para `m.state_dict()`, sem construir o modelo do Hugging Face.
*   **Os tensores são lidos em paralelo.** Pequenos lotes de tensores são lidos
    simultaneamente e copiados para a TPU antes da leitura do próximo lote, o
    que mantém baixo o uso de memória do host.

<section class="zippy" markdown="1">

**Carregamento do checkpoint e do tokenizador**

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

## 4. Prefill e loop de decode autorregressivo { #4-prefill-and-autoregressive-decode-loop }

A próxima célula define `model_generate`, que executa a etapa de prefill no
prompt de entrada e, em seguida, a decodificação gulosa (greedy), token por
token. Ela recebe o KV cache como argumento, então o mesmo loop funciona com
`StaticCache` ou com `DynamicCache`. A célula também carrega o tokenizador e os
pesos pré-treinados uma única vez. As duas execuções abaixo compartilham o
mesmo `muse_model`.

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

## 5. Decodificação autorregressiva com `StaticCache` { #5-autoregressive-decoding-with-staticcache }

Com `StaticCache` e `torch.compile(muse_model, dynamic=False)`:

*   O **prefill** compila o grafo para `seq_len = prefill_seq_len`.
*   A **etapa de decode 1** compila o grafo para `seq_len = 1`.
*   As **etapas de decode 2..N** reutilizam o executável de decode compilado,
    com **zero recompilações**.

Você não precisa passar `backend="tpu"`: quando as entradas do modelo compilado
estão na TPU, o `torch.compile` usa o backend do TorchTPU automaticamente.

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

## 6. Decodificação autorregressiva com `DynamicCache` { #6-autoregressive-decoding-with-dynamiccache }

O `DynamicCache` adiciona cada nova key e cada novo value com `torch.cat`, então
o comprimento do KV cache muda a cada etapa de decode. Com `dynamic=False`, cada
novo comprimento dispararia uma recompilação. Por isso, esta seção compila o
mesmo modelo com formatos dinâmicos, para que um único grafo compilado lide com
o cache que cresce.

### Ativar formatos dinâmicos no TorchTPU { #enable-dynamic-shapes-on-torchtpu }

O TorchTPU oferece suporte aos
[formatos dinâmicos (dynamic shapes)](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
do PyTorch, mas esse suporte ainda é experimental e, por isso, precisa ser
ativado explicitamente. O TorchTPU muda o padrão do `torch.compile` de
`dynamic=None` para `dynamic=False`, então cada grafo é especializado para os
formatos exatos dele, a menos que você ative o recurso:

*   **Passe `dynamic=None` para ativar.** Com `dynamic=None`, o Dynamo compila
    primeiro com formatos estáticos. Quando uma chamada posterior recompila
    porque um tamanho mudou, o Dynamo marca essa dimensão como dinâmica. Isso
    se chama *formatos dinâmicos automáticos* (automatic dynamic shapes). Para
    tornar dimensões específicas dinâmicas desde a primeira compilação, use
    `torch._dynamo.mark_dynamic`.
*   **Não use `dynamic=True`.** `dynamic=True` trata todos os tamanhos como
    dinâmicos desde a primeira compilação. O PyTorch não recomenda esse modo,
    então o TorchTPU se concentra nos formatos dinâmicos automáticos.
    `torch.compile(...,
    dynamic=True)` gera um `NotImplementedError` que pede para você usar
    `dynamic=None`.

Internamente, o XLA compila cada grafo em um programa de TPU com tamanhos de
buffer fixos, o que permite otimizar totalmente o layout da memória e fazer a
fusão de operadores. Para compilar um grafo com uma dimensão dinâmica, o
TorchTPU define um limite superior para essa dimensão e compila um único
programa que lida com todos os tamanhos até esse limite. O tamanho real é
passado em tempo de execução, então tamanhos dentro do limite reutilizam o
programa compilado, e um tamanho acima do limite dispara uma recompilação com
um novo limite superior. Ao restringir o dinamismo às dimensões que realmente
mudam, como o comprimento do KV cache neste caso, o XLA pode otimizar todo o
resto como estático.

A próxima célula compila o modelo com `dynamic=None`. Espere algumas
compilações de aquecimento nas primeiras etapas de decode. Depois disso, o
crescimento do cache não dispara mais recompilações.

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

## 7. Resultados de benchmark { #7-benchmark-results }

Executar este notebook com o modelo `MuseGlimmer30B` completo, de 52 camadas
(`num_layers=52`, `num_decode_steps=20`), e os pesos do checkpoint pré-treinado
em um único chip TPU v7x gera o perfil de latência a seguir, medido com o cache
de compilação vazio (cold start). O carregamento do checkpoint de 52 camadas
(627 tensores) levou entre \~100 s e \~420 s nas nossas execuções, dependendo
da taxa de leitura do armazenamento.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| Métrica | `StaticCache` (`dynamic=False`) | `DynamicCache` (`dynamic=None`) |
| --- | --- | --- |
| **Tempo de prefill (ms)** | 21.024,81 | 18.256,18 |
| **Etapa de decode 1 (ms)** | 19.756,86 | 19.969,41 |
| **Etapa de decode 2 (ms)** | 24,66 | 21.985,62 |
| **Etapa de decode 3 (ms)** | 24,29 | 4.908,23 |
| **Etapa de decode 4 (ms)** | 23,69 | 25,57 |
| **Etapa de decode 5 (ms)** | 23,59 | 24,83 |
| **Etapa de decode 6 (ms)** | 23,58 | 24,15 |
| **Etapa de decode 7 (ms)** | 23,56 | 24,10 |
| **Etapa de decode 8 (ms)** | 23,43 | 23,97 |
| **Etapa de decode 9 (ms)** | 23,38 | 24,04 |
| **Etapa de decode 10 (ms)** | 23,42 | 24,03 |
| **Etapa de decode 11 (ms)** | 23,48 | 24,06 |
| **Etapa de decode 12 (ms)** | 23,52 | 24,07 |
| **Etapa de decode 13 (ms)** | 23,53 | 24,04 |
| **Etapa de decode 14 (ms)** | 23,41 | 24,05 |
| **Etapa de decode 15 (ms)** | 23,43 | 24,00 |
| **Etapa de decode 16 (ms)** | 23,42 | 24,01 |
| **Etapa de decode 17 (ms)** | 23,42 | 23,95 |
| **Etapa de decode 18 (ms)** | 23,42 | 23,96 |
| **Etapa de decode 19 (ms)** | 23,34 | 23,99 |
| **Etapa de decode 20 (ms)** | 23,41 | 24,05 |
| **Tempo médio de decode / token (sem aquecimento) (ms)** | **23,58** (etapas 2–20) | **24,17** (etapas 4–20) |

<!-- mdformat on -->

*   **`StaticCache`: zero recompilações no decode após a etapa 1.** O prefill
    (\~21,0 s) e a etapa de decode 1 (\~19,8 s) incluem uma compilação XLA
    única para o comprimento do prompt (`seq_len = 7` tokens para este prompt
    e tokenizador) e para `seq_len = 1`, respectivamente. **A partir da etapa
    de decode 2**, cada token é executado em **\~23,6 ms**, com zero
    recompilações.
*   **`DynamicCache`: duas etapas de aquecimento a mais que o `StaticCache` e,
    depois, nenhuma recompilação.** O prefill e a etapa de decode 1 compilam
    para formatos estáticos. A etapa de decode 2 (\~22,0 s) é a primeira em
    que o comprimento do KV cache muda, então o Dynamo o marca como dinâmico e
    compila um grafo que lida com um comprimento de cache variável. A etapa de
    decode 3 (\~4,9 s) ainda inclui alguma compilação. **A partir da etapa de
    decode 4**, cada token é executado em **\~24,2 ms**.
*   **Execuções repetidas compilam mais rápido.** O backend do TorchTPU salva
    os programas compilados no cache de compilação em disco do PyTorch
    (`AOTAutogradCache`), que persiste entre reinicializações do kernel na
    mesma máquina. Quando você executa este notebook novamente sem alterar o
    código do modelo, o prefill e a etapa de decode 1 são carregados do cache
    em vez de recompilar. Os tempos de decode por token não mudam. Para medir
    a compilação em cold start, como na tabela acima, execute
    `torch._dynamo.reset()` e
    `torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache.clear()`
    antes de `torch.compile`.
*   **Exemplo de saída (modelo completo de 52 camadas, 100 etapas de
    decode):**

```text
Prompt: The future of artificial intelligence is
Output: The future of artificial intelligence is here, and it’s changing the way we work, learn, and live. From self-driving cars to virtual assistants, AI is transforming industries and revolutionizing the way we interact with technology. But what does the future hold for AI? In this article, we’ll explore the latest trends and predictions for the future of artificial intelligence.

## The Rise of AI

Artificial intelligence has been around for decades, but it’s only in recent years that we’ve seen a significant increase in its use and development.
```

> **Note:** com `USE_FULL_MODEL = False` (veja o início do notebook), apenas as
> 4 primeiras das 52 camadas do decodificador são carregadas, então o notebook
> não imprime o texto gerado.
