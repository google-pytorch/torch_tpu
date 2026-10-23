# Muse Glimmer 30B: decodificación autorregresiva { #muse-glimmer-30b-autoregressive-decoding }

<!-- Link to colab and to generated notebook -->

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_muse_glimmer_30b_inference.ipynb)

En esta guía, se explica la inferencia autorregresiva de LLM en TPU con
TorchTPU, con la arquitectura **Muse Glimmer** y pesos previamente entrenados de
Hugging Face. Se comparan dos cachés de KV que comparten un mismo modelo: una
**`StaticCache`** preasignada que se compila con formas estáticas
(`dynamic=False`) y una **`DynamicCache`** que crece y se compila con formas
dinámicas (`dynamic=None`).

Para ver una versión de este ejemplo como secuencia de comandos independiente,
con ejecutores separados para las cachés estática y dinámica, consulta
[`examples/muse_glimmer_30b/llm_inference`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference).

> **Requisitos de memoria:** De forma predeterminada (`USE_FULL_MODEL = False`),
> el notebook carga solo las primeras 4 de las 52 capas del decodificador
> (\~9.3 GB de pesos), por lo que cabe en TPUs con menos HBM, pero el texto
> generado no tiene sentido. Para cargar el modelo completo (\~55.7 GB de pesos
> en la HBM de la TPU) y generar texto con sentido, configura
> `USE_FULL_MODEL = True` en la siguiente celda.

```python
# True loads all 52 decoder layers (~55.7 GB). False loads only the first 4
# layers (~9.3 GB): the model still compiles and runs, but the generated text
# isn't meaningful.
USE_FULL_MODEL = False
```

## 1. Almacenamiento en caché de KV para la compilación de XLA { #1-kv-caching-for-xla-compilation }

En la inferencia autorregresiva de LLM, la generación avanza en dos fases:

*   **Prefill:** El modelo procesa todo el prompt `[B, S]` en paralelo y llena
    la caché de KV `[B, H_kv, S, D]` (batch `B`, longitud del prompt `S`,
    `H_kv` cabezas de KV, dimensión de cabeza `D`).
*   **Decodificación:** Cada paso `t` recibe un token nuevo `[B, 1]`, agrega su
    clave y su valor a la caché, y aplica atención sobre las `S + t`
    posiciones.

**`StaticCache`** evita las recompilaciones durante la decodificación con
`dynamic=False`, ya que mantiene constantes todas las formas de los tensores en
los pasos de decodificación:

1.  **Búferes de tamaño fijo:** `keys` y `values` se preasignan como `[B, H_kv,
    max_cache_len, D]`.
2.  **Longitud en el dispositivo:** La longitud de la caché es un tensor de TPU
    0-D que se actualiza in situ, no un `int` de Python sobre el que Dynamo
    aplica guardas.
3.  **Escrituras in situ, máscaras de forma fija:** Las entradas nuevas se
    escriben con `index_copy_`; las posiciones sin escribir se enmascaran con
    `-inf`.
4.  **Dos compilaciones:** Una para el prefill y otra para la decodificación,
    que reutilizan todos los pasos posteriores.

**`DynamicCache`** adopta el enfoque opuesto y es la caché predeterminada en los
modelos de Hugging Face: agrega las claves y los valores de cada paso con
`torch.cat`, por lo que la caché contiene exactamente `S + t` posiciones. Esto
es más simple y no usa memoria de relleno, pero el tamaño de la caché cambia en
cada paso, por lo que depende de las formas dinámicas (`dynamic=None`) para
evitar volver a compilar en cada token.

## 2. Implementación del modelo y de la caché de KV { #2-model-and-kv-cache-implementation }

La siguiente celda implementa `StaticCache`, `DynamicCache` y la arquitectura
de decodificador `MuseGlimmer30B`, que intercala tres capas de atención de
ventana deslizante (SWA) con RoPE y una capa de atención causal completa NoPE
cada cuatro bloques.

El mismo modelo y las mismas cachés también están disponibles como módulos
reutilizables en
[`model.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/model.py)
y
[`kv_cache.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/kv_cache.py).

<section class="zippy" markdown="1">

**Importaciones y configuración del dispositivo**

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

**Cachés de KV e implementación del modelo**

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

## 3. Carga de los pesos previamente entrenados y del tokenizador { #3-loading-pretrained-weights-and-tokenizer }

En esta sección, se define cómo cargar el tokenizador y los pesos previamente
entrenados de `Muse-Glimmer-30B` en `MuseGlimmer30B`. El checkpoint puede
provenir de uno de dos lugares:

1.  **Una copia local del checkpoint:** Configura la variable de entorno
    `MUSE_GLIMMER_CHECKPOINT_DIR` con un directorio que contenga los archivos
    del checkpoint. En un notebook, puedes ejecutar
    `os.environ["MUSE_GLIMMER_CHECKPOINT_DIR"] =
    "/path/to/checkpoint"` antes de la siguiente celda. Si la variable está
    configurada, pero el directorio no existe, la carga falla en lugar de
    recurrir a una descarga.
2.  **Hugging Face Hub (opción predeterminada):** Si
    `MUSE_GLIMMER_CHECKPOINT_DIR` no está configurada, los archivos se
    descargan del repositorio `MODEL_NAME` del Hub
    (`meta-models/Muse-Glimmer-30B`). Las descargas se almacenan en caché, por
    lo que las ejecuciones posteriores las reutilizan.

En ambos casos, los pesos se cargan de la misma manera:

*   **Solo se leen las capas configuradas.** Los tensores se cargan solo para
    las capas de `Config(num_layers=...)`, por lo que `USE_FULL_MODEL = False`
    (consulta el principio del notebook) carga 4 capas en lugar de 52.
*   **Los tensores se copian directamente en el modelo.** Los tensores del
    decodificador de texto se copian desde los archivos `.safetensors`
    directamente en `m.state_dict()`, sin construir el modelo de Hugging Face.
*   **Los tensores se leen en paralelo.** Se leen lotes pequeños de tensores de
    forma simultánea y se copian a la TPU antes de leer el siguiente lote, lo
    que mantiene bajo el uso de memoria del host.

<section class="zippy" markdown="1">

**Carga del checkpoint y del tokenizador**

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

## 4. Prefill y bucle de decodificación autorregresiva { #4-prefill-and-autoregressive-decode-loop }

La siguiente celda define `model_generate`, que ejecuta el pase de prefill sobre
el prompt de entrada y, luego, la decodificación voraz (greedy) token por
token. Recibe la caché de KV como argumento, por lo que el mismo bucle se
ejecuta con `StaticCache` o con `DynamicCache`. La celda también carga el
tokenizador y los pesos previamente entrenados una sola vez; las dos ejecuciones
de abajo comparten el mismo `muse_model`.

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

## 5. Decodificación autorregresiva con `StaticCache` { #5-autoregressive-decoding-with-staticcache }

Con `StaticCache` y `torch.compile(muse_model, dynamic=False)`:

*   El **prefill** compila el grafo para `seq_len = prefill_seq_len`.
*   El **paso de decodificación 1** compila el grafo para `seq_len = 1`.
*   Los **pasos de decodificación 2..N** reutilizan el ejecutable de
    decodificación compilado **sin ninguna recompilación**.

No necesitas pasar `backend="tpu"`: cuando las entradas del modelo compilado
están en la TPU, `torch.compile` usa el backend de TorchTPU automáticamente.

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

## 6. Decodificación autorregresiva con `DynamicCache` { #6-autoregressive-decoding-with-dynamiccache }

`DynamicCache` agrega cada clave y cada valor nuevos con `torch.cat`, por lo que
la longitud de la caché de KV cambia en cada paso de decodificación. Con
`dynamic=False`, cada longitud nueva activaría una recompilación. En cambio, en
esta sección se compila el mismo modelo con formas dinámicas, de modo que un
solo grafo compilado maneja la caché que crece.

### Habilita las formas dinámicas en TorchTPU { #enable-dynamic-shapes-on-torchtpu }

TorchTPU admite las
[formas dinámicas](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
de PyTorch, pero, como esta compatibilidad es experimental por ahora, debes
habilitarlo de forma explícita. TorchTPU cambia el valor predeterminado de
`torch.compile` de `dynamic=None` a `dynamic=False`, por lo que cada grafo se
especializa en sus formas exactas, a menos que lo habilites:

*   **Pasa `dynamic=None` para habilitarlo.** Con `dynamic=None`, Dynamo
    primero compila con formas estáticas. Cuando una llamada posterior vuelve a
    compilar porque cambió un tamaño, Dynamo marca esa dimensión como dinámica.
    Esto se denomina *formas dinámicas automáticas*. Para que dimensiones
    específicas sean dinámicas desde la primera compilación, usa
    `torch._dynamo.mark_dynamic`.
*   **No uses `dynamic=True`.** `dynamic=True` trata todos los tamaños como
    dinámicos desde la primera compilación. PyTorch no recomienda este modo,
    por lo que TorchTPU se enfoca en las formas dinámicas automáticas.
    `torch.compile(...,
    dynamic=True)` genera un `NotImplementedError` que te pide usar
    `dynamic=None` en su lugar.

Internamente, XLA compila cada grafo en un programa de TPU con tamaños de búfer
fijos, lo que le permite optimizar por completo el diseño de la memoria y
realizar la fusión de operadores. Para compilar un grafo con una dimensión
dinámica, TorchTPU le asigna un límite superior a esa dimensión y compila un
solo programa que maneja todos los tamaños hasta ese límite. El tamaño real se
pasa en el entorno de ejecución, por lo que los tamaños dentro del límite
reutilizan el programa compilado, y un tamaño que supera el límite activa una
recompilación con un nuevo límite superior. Si limitas el dinamismo a las
dimensiones que realmente cambian, como la longitud de la caché de KV en este
caso, XLA puede optimizar todo lo demás como estático.

La siguiente celda compila el modelo con `dynamic=None`. Es normal que haya
algunas compilaciones de calentamiento en los primeros pasos de decodificación;
después de eso, la caché que crece no activa más recompilaciones.

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

## 7. Resultados de las comparativas de rendimiento { #7-benchmark-results }

Si ejecutas este notebook con el modelo `MuseGlimmer30B` completo de 52 capas
(`num_layers=52`, `num_decode_steps=20`) y los pesos del checkpoint previamente
entrenado en un solo chip TPU v7x, se obtiene el siguiente perfil de latencia,
medido con una caché de compilación vacía (inicio en frío). En nuestras
ejecuciones, cargar el checkpoint de 52 capas (627 tensores) tardó entre
\~100 s y \~420 s, según la capacidad de procesamiento de lectura del
almacenamiento.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| Métrica | `StaticCache` (`dynamic=False`) | `DynamicCache` (`dynamic=None`) |
| --- | --- | --- |
| **Tiempo de prefill (ms)** | 21024.81 | 18256.18 |
| **Paso de decodificación 1 (ms)** | 19756.86 | 19969.41 |
| **Paso de decodificación 2 (ms)** | 24.66 | 21985.62 |
| **Paso de decodificación 3 (ms)** | 24.29 | 4908.23 |
| **Paso de decodificación 4 (ms)** | 23.69 | 25.57 |
| **Paso de decodificación 5 (ms)** | 23.59 | 24.83 |
| **Paso de decodificación 6 (ms)** | 23.58 | 24.15 |
| **Paso de decodificación 7 (ms)** | 23.56 | 24.10 |
| **Paso de decodificación 8 (ms)** | 23.43 | 23.97 |
| **Paso de decodificación 9 (ms)** | 23.38 | 24.04 |
| **Paso de decodificación 10 (ms)** | 23.42 | 24.03 |
| **Paso de decodificación 11 (ms)** | 23.48 | 24.06 |
| **Paso de decodificación 12 (ms)** | 23.52 | 24.07 |
| **Paso de decodificación 13 (ms)** | 23.53 | 24.04 |
| **Paso de decodificación 14 (ms)** | 23.41 | 24.05 |
| **Paso de decodificación 15 (ms)** | 23.43 | 24.00 |
| **Paso de decodificación 16 (ms)** | 23.42 | 24.01 |
| **Paso de decodificación 17 (ms)** | 23.42 | 23.95 |
| **Paso de decodificación 18 (ms)** | 23.42 | 23.96 |
| **Paso de decodificación 19 (ms)** | 23.34 | 23.99 |
| **Paso de decodificación 20 (ms)** | 23.41 | 24.05 |
| **Tiempo promedio de decodificación por token (sin calentamiento) (ms)** | **23.58** (pasos 2–20) | **24.17** (pasos 4–20) |

<!-- mdformat on -->

*   **`StaticCache`: ninguna recompilación durante la decodificación después
    del paso 1.** El prefill (\~21.0 s) y el paso de decodificación 1
    (\~19.8 s) incluyen una compilación única de XLA para la longitud del
    prompt (`seq_len = 7` tokens para este prompt y este tokenizador) y para
    `seq_len = 1`, respectivamente. **A partir del paso de
    decodificación 2**, cada token se ejecuta en **\~23.6 ms** sin ninguna
    recompilación.
*   **`DynamicCache`: dos pasos de calentamiento más que `StaticCache` y,
    luego, ninguna recompilación.** El prefill y el paso de decodificación 1
    compilan para formas estáticas. El paso de decodificación 2 (\~22.0 s) es
    el primer paso en el que cambia la longitud de la caché de KV, por lo que
    Dynamo la marca como dinámica y compila un grafo que maneja una longitud de
    caché variable; el paso de decodificación 3 (\~4.9 s) todavía incluye algo
    de compilación. **A partir del paso de decodificación 4**, cada token se
    ejecuta en **\~24.2 ms**.
*   **Las ejecuciones repetidas compilan más rápido.** El backend de TorchTPU
    guarda los programas compilados en la caché de compilación en disco de
    PyTorch (`AOTAutogradCache`), que persiste entre reinicios del kernel en la
    misma máquina. Cuando vuelves a ejecutar este notebook sin cambiar el
    código del modelo, el prefill y el paso de decodificación 1 se cargan desde
    la caché en lugar de volver a compilarse; los tiempos de decodificación por
    token no cambian. Para medir la compilación en un inicio en frío, como en la
    tabla anterior, ejecuta `torch._dynamo.reset()` y
    `torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache.clear()`
    antes de `torch.compile`.
*   **Resultado de ejemplo (modelo completo de 52 capas, 100 pasos de
    decodificación):**

```text
Prompt: The future of artificial intelligence is
Output: The future of artificial intelligence is here, and it’s changing the way we work, learn, and live. From self-driving cars to virtual assistants, AI is transforming industries and revolutionizing the way we interact with technology. But what does the future hold for AI? In this article, we’ll explore the latest trends and predictions for the future of artificial intelligence.

## The Rise of AI

Artificial intelligence has been around for decades, but it’s only in recent years that we’ve seen a significant increase in its use and development.
```

> **Note:** Con `USE_FULL_MODEL = False` (consulta el principio del notebook),
> solo se cargan las primeras 4 de las 52 capas del decodificador, por lo que
> el notebook omite mostrar el texto generado.
