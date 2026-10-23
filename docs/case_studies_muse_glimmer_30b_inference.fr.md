# Muse Glimmer 30B : décodage autorégressif { #muse-glimmer-30b-autoregressive-decoding }

<!-- Link to colab and to generated notebook -->

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_muse_glimmer_30b_inference.ipynb)

Ce guide présente l'inférence autorégressive d'un LLM sur TPU avec TorchTPU, à
l'aide de l'architecture **Muse Glimmer** et de poids pré-entraînés Hugging
Face. Il compare deux caches KV qui partagent un même modèle : un
**`StaticCache`** préalloué, compilé avec des formes statiques
(`dynamic=False`), et un **`DynamicCache`** qui grandit, compilé avec des formes
dynamiques (`dynamic=None`).

Pour une version de cet exemple sous forme de script autonome, avec des
scripts d'exécution distincts pour les caches statique et dynamique, consultez
[`examples/muse_glimmer_30b/llm_inference`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference).

> **Mémoire requise :** par défaut (`USE_FULL_MODEL = False`), le notebook ne
> charge que les 4 premières des 52 couches du décodeur (\~9,3 Go de poids). Il
> tient ainsi sur des TPU dotés de moins de HBM, mais le texte généré n'a pas de
> sens. Pour charger le modèle complet (\~55,7 Go de poids dans la HBM du TPU) et
> générer un texte cohérent, définissez `USE_FULL_MODEL = True` dans la cellule
> suivante.

```python
# True loads all 52 decoder layers (~55.7 GB). False loads only the first 4
# layers (~9.3 GB): the model still compiles and runs, but the generated text
# isn't meaningful.
USE_FULL_MODEL = False
```

## 1. Cache KV pour la compilation XLA { #1-kv-caching-for-xla-compilation }

Dans l'inférence autorégressive d'un LLM, la génération se déroule en deux
phases :

*   **Prefill :** le modèle traite l'ensemble du prompt `[B, S]` en parallèle et
    remplit le cache KV `[B, H_kv, S, D]` (batch `B`, longueur du prompt `S`,
    `H_kv` têtes KV, dimension de tête `D`).
*   **Décodage :** chaque étape `t` fournit un nouveau token `[B, 1]`, ajoute sa
    clé et sa valeur au cache, et applique l'attention sur les `S + t`
    positions.

**`StaticCache`** évite les recompilations pendant le décodage avec
`dynamic=False` en gardant toutes les formes de tensors constantes d'une étape
de décodage à l'autre :

1.  **Buffers de taille fixe :** `keys` et `values` sont préalloués sous la forme `[B, H_kv,
    max_cache_len, D]`.
2.  **Longueur sur l'appareil :** la longueur du cache est un tensor TPU 0-D mis
    à jour sur place, et non un `int` Python sur lequel Dynamo pose des gardes
    (guards).
3.  **Écritures sur place, masques de forme fixe :** les nouvelles entrées sont
    écrites avec `index_copy_` ; les positions non écrites sont masquées avec
    `-inf`.
4.  **Deux compilations :** une pour le prefill et une pour le décodage,
    réutilisées par toutes les étapes suivantes.

**`DynamicCache`** adopte l'approche inverse. C'est le cache par défaut des
modèles Hugging Face : il ajoute les clés et les valeurs de chaque étape avec
`torch.cat`, de sorte que le cache contient exactement `S + t` positions. Cette
approche est plus simple et n'utilise pas de mémoire de padding, mais la taille
du cache change à chaque étape. Elle s'appuie donc sur les formes dynamiques
(`dynamic=None`) pour éviter une recompilation à chaque token.

## 2. Implémentation du modèle et du cache KV { #2-model-and-kv-cache-implementation }

La cellule suivante implémente `StaticCache`, `DynamicCache` et l'architecture
de décodeur `MuseGlimmer30B`, qui alterne, tous les quatre blocs, trois couches
d'attention à fenêtre glissante (SWA) avec RoPE et une couche d'attention
causale complète NoPE.

Le même modèle et les mêmes caches sont également disponibles sous forme de
modules réutilisables dans
[`model.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/model.py)
et
[`kv_cache.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/llm_inference/kv_cache.py).

<section class="zippy" markdown="1">

**Imports et configuration de l'appareil**

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

**Caches KV et implémentation du modèle**

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

## 3. Chargement des poids pré-entraînés et du tokenizer { #3-loading-pretrained-weights-and-tokenizer }

Cette section définit comment charger le tokenizer et les poids pré-entraînés
de `Muse-Glimmer-30B` dans `MuseGlimmer30B`. Le checkpoint peut provenir de
deux sources :

1.  **Une copie locale du checkpoint :** définissez la variable d'environnement
    `MUSE_GLIMMER_CHECKPOINT_DIR` sur un répertoire qui contient les fichiers du
    checkpoint. Dans un notebook, vous pouvez exécuter `os.environ["MUSE_GLIMMER_CHECKPOINT_DIR"] =
    "/path/to/checkpoint"` avant la cellule suivante. Si la variable est
    définie mais que le répertoire n'existe pas, le chargement échoue au lieu de
    se rabattre sur un téléchargement.
2.  **Le Hugging Face Hub (par défaut) :** si `MUSE_GLIMMER_CHECKPOINT_DIR`
    n'est pas définie, les fichiers sont téléchargés depuis le dépôt du Hub
    `MODEL_NAME` (`meta-models/Muse-Glimmer-30B`). Les téléchargements sont mis
    en cache : les exécutions suivantes les réutilisent.

Dans les deux cas, les poids sont chargés de la même façon :

*   **Seules les couches configurées sont lues.** Les tensors ne sont chargés
    que pour les couches de `Config(num_layers=...)` : avec
    `USE_FULL_MODEL = False` (voir le début du notebook), 4 couches sont donc
    chargées au lieu de 52.
*   **Les tensors sont copiés directement dans le modèle.** Les tensors du
    décodeur de texte sont copiés depuis les fichiers `.safetensors` directement
    dans `m.state_dict()`, sans construire le modèle Hugging Face.
*   **Les tensors sont lus en parallèle.** De petits lots de tensors sont lus
    simultanément et copiés sur le TPU avant la lecture du lot suivant, ce qui
    limite l'utilisation de la mémoire de l'hôte.

<section class="zippy" markdown="1">

**Chargement du checkpoint et du tokenizer**

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

## 4. Prefill et boucle de décodage autorégressif { #4-prefill-and-autoregressive-decode-loop }

La cellule suivante définit `model_generate`, qui exécute la passe de prefill
sur le prompt d'entrée, puis un décodage glouton (greedy) token par token. Elle
prend le cache KV en argument : la même boucle s'exécute donc avec
`StaticCache` comme avec `DynamicCache`. La cellule charge également le
tokenizer et les poids pré-entraînés une seule fois ; les deux exécutions
ci-dessous partagent le même `muse_model`.

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

## 5. Décodage autorégressif avec `StaticCache` { #5-autoregressive-decoding-with-staticcache }

Avec `StaticCache` et `torch.compile(muse_model, dynamic=False)` :

*   Le **prefill** compile le graphe pour `seq_len = prefill_seq_len`.
*   L'**étape de décodage 1** compile le graphe pour `seq_len = 1`.
*   Les **étapes de décodage 2 à N** réutilisent l'exécutable de décodage
    compilé, **sans aucune recompilation**.

Vous n'avez pas besoin de transmettre `backend="tpu"` : lorsque les entrées du
modèle compilé se trouvent sur le TPU, `torch.compile` utilise automatiquement
le backend TorchTPU.

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

## 6. Décodage autorégressif avec `DynamicCache` { #6-autoregressive-decoding-with-dynamiccache }

`DynamicCache` ajoute chaque nouvelle clé et valeur avec `torch.cat` : la
longueur du cache KV change donc à chaque étape de décodage. Avec
`dynamic=False`, chaque nouvelle longueur déclencherait une recompilation.
Cette section compile plutôt le même modèle avec des formes dynamiques, de
sorte qu'un seul graphe compilé gère le cache qui grandit.

### Activer les formes dynamiques dans TorchTPU { #enable-dynamic-shapes-on-torchtpu }

TorchTPU prend en charge les
[formes dynamiques](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
de PyTorch, mais cette prise en charge est actuellement expérimentale : vous
devez donc l'activer explicitement. TorchTPU fait passer la valeur par défaut
de `torch.compile` de `dynamic=None` à `dynamic=False`. Chaque graphe est donc
spécialisé pour ses formes exactes, sauf si vous activez les formes
dynamiques :

*   **Transmettez `dynamic=None` pour les activer.** Avec `dynamic=None`, Dynamo
    compile d'abord avec des formes statiques. Lorsqu'un appel ultérieur
    entraîne une recompilation parce qu'une taille a changé, Dynamo marque
    cette dimension comme dynamique. C'est ce qu'on appelle les *formes
    dynamiques automatiques* (automatic dynamic shapes). Pour rendre certaines
    dimensions dynamiques dès la première compilation, utilisez
    `torch._dynamo.mark_dynamic`.
*   **N'utilisez pas `dynamic=True`.** `dynamic=True` traite chaque taille comme
    dynamique dès la première compilation. PyTorch déconseille ce mode : TorchTPU
    se concentre donc sur les formes dynamiques automatiques. `torch.compile(...,
    dynamic=True)` lève une `NotImplementedError` qui vous invite à utiliser
    `dynamic=None` à la place.

En interne, XLA compile chaque graphe en un programme TPU avec des tailles de
buffers fixes, ce qui lui permet d'optimiser entièrement la disposition de la
mémoire et d'effectuer la fusion d'opérateurs. Pour compiler un graphe avec une
dimension dynamique, TorchTPU attribue une borne supérieure à cette dimension
et compile un seul programme qui gère toutes les tailles jusqu'à cette borne.
La taille réelle est transmise au runtime : les tailles inférieures à la borne
réutilisent le programme compilé, et une taille supérieure déclenche une
recompilation avec une nouvelle borne supérieure. Limiter le dynamisme aux
dimensions qui changent réellement, comme ici la longueur du cache KV, permet à
XLA d'optimiser tout le reste comme statique.

La cellule suivante compile le modèle avec `dynamic=None`. Attendez-vous à
quelques compilations de préchauffage lors des premières étapes de décodage ;
ensuite, le cache qui grandit ne déclenche plus de recompilation.

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

## 7. Résultats du benchmark { #7-benchmark-results }

L'exécution de ce notebook avec le modèle `MuseGlimmer30B` complet à 52 couches
(`num_layers=52`, `num_decode_steps=20`) et les poids du checkpoint
pré-entraîné sur une seule puce TPU v7x produit le profil de latence suivant,
mesuré avec un cache de compilation vide (démarrage à froid). Le chargement du
checkpoint à 52 couches (627 tensors) a pris entre \~100 s et \~420 s lors de
nos exécutions, selon le débit de lecture du stockage.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| Métrique | `StaticCache` (`dynamic=False`) | `DynamicCache` (`dynamic=None`) |
| --- | --- | --- |
| **Temps de prefill (ms)** | 21 024,81 | 18 256,18 |
| **Étape de décodage 1 (ms)** | 19 756,86 | 19 969,41 |
| **Étape de décodage 2 (ms)** | 24,66 | 21 985,62 |
| **Étape de décodage 3 (ms)** | 24,29 | 4 908,23 |
| **Étape de décodage 4 (ms)** | 23,69 | 25,57 |
| **Étape de décodage 5 (ms)** | 23,59 | 24,83 |
| **Étape de décodage 6 (ms)** | 23,58 | 24,15 |
| **Étape de décodage 7 (ms)** | 23,56 | 24,10 |
| **Étape de décodage 8 (ms)** | 23,43 | 23,97 |
| **Étape de décodage 9 (ms)** | 23,38 | 24,04 |
| **Étape de décodage 10 (ms)** | 23,42 | 24,03 |
| **Étape de décodage 11 (ms)** | 23,48 | 24,06 |
| **Étape de décodage 12 (ms)** | 23,52 | 24,07 |
| **Étape de décodage 13 (ms)** | 23,53 | 24,04 |
| **Étape de décodage 14 (ms)** | 23,41 | 24,05 |
| **Étape de décodage 15 (ms)** | 23,43 | 24,00 |
| **Étape de décodage 16 (ms)** | 23,42 | 24,01 |
| **Étape de décodage 17 (ms)** | 23,42 | 23,95 |
| **Étape de décodage 18 (ms)** | 23,42 | 23,96 |
| **Étape de décodage 19 (ms)** | 23,34 | 23,99 |
| **Étape de décodage 20 (ms)** | 23,41 | 24,05 |
| **Temps moyen de décodage par token (hors préchauffage) (ms)** | **23,58** (étapes 2 à 20) | **24,17** (étapes 4 à 20) |

<!-- mdformat on -->

*   **`StaticCache` : aucune recompilation pendant le décodage après l'étape 1.**
    Le prefill (\~21,0 s) et l'étape de décodage 1 (\~19,8 s) incluent une
    compilation XLA unique, respectivement pour la longueur du prompt
    (`seq_len = 7` tokens pour ce prompt et ce tokenizer) et pour
    `seq_len = 1`. **À partir de l'étape de décodage 2**, chaque token
    s'exécute en **\~23,6 ms**, sans aucune recompilation.
*   **`DynamicCache` : deux étapes de préchauffage de plus qu'avec
    `StaticCache`, puis aucune recompilation.** Le prefill et l'étape de
    décodage 1 compilent pour des formes statiques. L'étape de décodage 2
    (\~22,0 s) est la première où la longueur du cache KV change : Dynamo la
    marque donc comme dynamique et compile un graphe qui gère une longueur de
    cache variable. L'étape de décodage 3 (\~4,9 s) inclut encore une part de
    compilation. **À partir de l'étape de décodage 4**, chaque token s'exécute
    en **\~24,2 ms**.
*   **Les exécutions répétées compilent plus vite.** Le backend TorchTPU
    enregistre les programmes compilés dans le cache de compilation sur disque
    de PyTorch (`AOTAutogradCache`), qui persiste entre les redémarrages du
    kernel sur la même machine. Lorsque vous réexécutez ce notebook sans
    modifier le code du modèle, le prefill et l'étape de décodage 1 sont
    chargés depuis le cache au lieu d'être recompilés ; les temps de décodage
    par token restent inchangés. Pour mesurer la compilation à froid, comme dans
    le tableau ci-dessus, exécutez
    `torch._dynamo.reset()` et
    `torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache.clear()`
    avant `torch.compile`.
*   **Exemple de sortie (modèle complet à 52 couches, 100 étapes de
    décodage) :**

```text
Prompt: The future of artificial intelligence is
Output: The future of artificial intelligence is here, and it’s changing the way we work, learn, and live. From self-driving cars to virtual assistants, AI is transforming industries and revolutionizing the way we interact with technology. But what does the future hold for AI? In this article, we’ll explore the latest trends and predictions for the future of artificial intelligence.

## The Rise of AI

Artificial intelligence has been around for decades, but it’s only in recent years that we’ve seen a significant increase in its use and development.
```

> **Note:** Avec `USE_FULL_MODEL = False` (voir le début du notebook), seules
> les 4 premières des 52 couches du décodeur sont chargées : le notebook
> n'affiche donc pas le texte généré.
