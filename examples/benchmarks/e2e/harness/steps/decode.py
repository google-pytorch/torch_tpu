# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import functools
import inspect
from typing import Any
import torch
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness.steps import common
from transformers import cache_utils

# Architectures whose state is recurrent / SSM based rather than a standard
# attention KV cache. These never use `StaticCache`, regardless of dynamism.
_RECURRENT_MODEL_TYPES = frozenset({
    "decision_transformer",
    "falcon_mamba",
    "mamba",
    "mamba2",
    "recurrent_gemma",
    "rwkv",
    "xlstm",
})

# A layer's `conv_states`/`recurrent_states`. transformers used to expose a
# single tensor per layer and now exposes a mapping of state index to tensor.
_LinearState = torch.Tensor | dict[int, torch.Tensor]


def _safe_get(obj: Any, *attr_names: str) -> Any:
  """Safely retrieves the first non-None attribute or dict key from obj."""
  if obj is None:
    return None
  for name in attr_names:
    val = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    if val is not None:
      return val
  return None


def _apply_projection(
    module: Any, hidden_state: torch.Tensor
) -> torch.Tensor | None:
  """Projects hidden_state using a module or embedding weights."""
  if module is None or hidden_state is None:
    return None
  if isinstance(module, torch.nn.Embedding):
    return torch.nn.functional.linear(hidden_state, module.weight)
  if callable(module):
    return module(hidden_state)
  if hasattr(module, "weight"):
    return torch.nn.functional.linear(hidden_state, module.weight)
  return None


@functools.lru_cache(maxsize=256)
def _inspect_params(fn: Any) -> tuple[frozenset[str], bool] | None:
  """Returns (parameter names, accepts **kwargs) for fn, or None if unknown."""
  try:
    sig = inspect.signature(fn)
  except (ValueError, TypeError):
    return None
  return frozenset(sig.parameters), any(
      p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
  )


def _forward_param_spec(target_fn: Any) -> tuple[frozenset[str], bool] | None:
  """Memoized `_inspect_params`, safe to call on every decode step.

  The decode loop calls this once per generated token, so the underlying
  `inspect.signature` call is cached. The cache is keyed on the *unbound*
  function (`__func__`) rather than the bound method so that it does not keep
  model instances alive and stays shared across instances of the same class.
  Dropping the implicit `self` parameter is harmless here because callers only
  test for the presence of named keyword parameters.

  Args:
    target_fn: The model `forward` callable to introspect.

  Returns:
    A `(parameter names, accepts **kwargs)` tuple, or None if `target_fn` has
    no introspectable signature.
  """
  key = getattr(target_fn, "__func__", target_fn)
  try:
    return _inspect_params(key)
  except TypeError:
    # Unhashable callable (e.g. an exotic wrapper); fall back to an uncached
    # lookup rather than failing the step.
    return _inspect_params.__wrapped__(key)


def _config_supports_static_cache(cfg: Any) -> bool:
  """Returns whether transformers can build a `StaticCache` for every layer.

  `config.layer_types` is authoritative: `StaticCache` dispatches each entry
  through `cache_utils.STATIC_LAYER_TYPE_MAPPING`, which covers full/sliding/
  chunked attention as well as linear-attention and hybrid (SSM) layers. Testing
  membership there rather than against a hand-maintained allowlist keeps hybrid
  architectures (Jamba, FalconH1, Bamba, Zamba, Nemotron-H, ...) on statically
  shaped caches, and stops this check from rotting as transformers grows new
  layer types.

  Args:
    cfg: The (text) model config to inspect.

  Returns:
    True if a static cache layer exists for every declared layer type.
  """
  layer_types = getattr(cfg, "layer_types", None)
  if not isinstance(layer_types, (list, tuple)) or not layer_types:
    return True
  mapping = getattr(cache_utils, "STATIC_LAYER_TYPE_MAPPING", None)
  if not mapping:
    return True
  return all(layer_type in mapping for layer_type in layer_types)


def _make_dynamic_cache(model_config: Any) -> cache_utils.DynamicCache:
  """Builds a `DynamicCache`, preserving per-layer types when possible.

  Passing the config matters: without it `DynamicCache` replicates a plain
  `DynamicLayer` for every layer, which silently produces the wrong layer class
  for sliding-window, linear-attention and hybrid architectures.

  Args:
    model_config: The model config to derive layer types from.

  Returns:
    A `DynamicCache` built from the config when the installed transformers
    version supports it, otherwise a default one.
  """
  try:
    return cache_utils.DynamicCache(config=model_config)
  except (TypeError, AttributeError):
    return cache_utils.DynamicCache()


def _supports_early_initialization(cache: Any) -> bool:
  """Returns whether `early_initialization` is safe for every layer of `cache`.

  `early_initialization` hands each layer a zero-length key/value tensor.
  Releases that predate `CacheLayerMixin.supports_early_init` do this
  unconditionally, which lazily initializes a linear-attention layer's
  `conv_states` with an attention-shaped buffer and makes the first real
  `update_conv_state` fail on a shape mismatch. Newer releases skip those
  layers themselves, so the call is safe there.

  Args:
    cache: The cache about to be early-initialized.

  Returns:
    True if `early_initialization` can be called without corrupting a layer.
  """
  layers = getattr(cache, "layers", None) or []
  if all(hasattr(layer, "supports_early_init") for layer in layers):
    return True
  return not any(
      isinstance(layer, cache_utils.LinearAttentionCacheLayerMixin)
      for layer in layers
  )


def _activate_past_recording(cache: Any) -> None:
  """Arms rollback recording on a `DynamicCache`, which `crop` requires.

  Deliberately skipped for `StaticCache`: there, past recording makes linear
  attention layers grow `conv_states` instead of mutating a fixed-width buffer,
  which would defeat the static shapes we want on TPU. Static caches are instead
  rewound from an explicit snapshot.

  Args:
    cache: The cache to arm.
  """
  if isinstance(cache, cache_utils.DynamicCache) and hasattr(
      cache, "activate_past_recording"
  ):
    cache.activate_past_recording()


def _clone_linear_attention_state(states: Any) -> _LinearState | None:
  """Clones one `conv_states`/`recurrent_states` container, or None if empty.

  transformers changed this container from a single tensor to a mapping of
  state index to tensor (a layer may cache several states). Both layouts are
  supported so that this harness keeps working against older pinned releases.

  Args:
    states: The raw container read off a cache layer.

  Returns:
    A detached copy in the same layout, or None if there is nothing to clone.
  """
  if isinstance(states, torch.Tensor):
    return states.clone()
  if isinstance(states, dict):
    cloned = {
        state_idx: state.clone()
        for state_idx, state in states.items()
        if isinstance(state, torch.Tensor)
    }
    return cloned or None
  return None


def _snapshot_linear_attention_states(
    cache: Any, include_conv: bool
) -> dict[int, dict[str, _LinearState]] | None:
  """Clones the per-layer linear-attention (SSM) states of a cache.

  Both `conv_states` and `recurrent_states` are mutated in place via `copy_()`
  (see `LinearAttentionCacheLayerMixin` in transformers `cache_utils`), so
  nothing else can rewind them after warmup decode steps.

  Args:
    cache: The cache to snapshot.
    include_conv: Whether to also capture `conv_states`. Only needed for
      `StaticCache`; on a `DynamicCache`, `crop()` already rewinds them, and
      re-imposing the post-prefill width could change tensor shapes and trigger
      a recompile.

  Returns:
    A mapping of layer index to state name to cloned state, or None if the
    cache holds no linear-attention state.
  """
  names = (
      ("conv_states", "recurrent_states")
      if include_conv
      else ("recurrent_states",)
  )
  snapshot = {}
  for layer_idx, layer in enumerate(getattr(cache, "layers", None) or []):
    entry = {}
    for name in names:
      cloned = _clone_linear_attention_state(getattr(layer, name, None))
      if cloned is not None:
        entry[name] = cloned
    if entry:
      snapshot[layer_idx] = entry
  return snapshot or None


def _copy_or_replace(current: Any, want: torch.Tensor) -> torch.Tensor:
  """Returns `want` written into `current` in place when the shapes allow.

  Copying in place keeps the static address that `mark_static_address` and the
  compiled decode step rely on; reassigning is only a fallback.

  Args:
    current: The tensor currently held by the cache layer, if any.
    want: The snapshotted tensor to restore.

  Returns:
    The tensor the cache layer should now hold.
  """
  if isinstance(current, torch.Tensor) and current.shape == want.shape:
    current.copy_(want)
    return current
  return want.clone()


def _restore_linear_attention_states(
    cache: Any, snapshot: dict[int, dict[str, _LinearState]]
) -> None:
  """Restores states captured by `_snapshot_linear_attention_states`."""
  layers = getattr(cache, "layers", None) or []
  for layer_idx, entry in snapshot.items():
    if layer_idx >= len(layers):
      continue
    layer = layers[layer_idx]
    for name, want in entry.items():
      current = getattr(layer, name, None)
      if isinstance(want, torch.Tensor):
        setattr(layer, name, _copy_or_replace(current, want))
      elif isinstance(current, dict):
        for state_idx, want_state in want.items():
          current[state_idx] = _copy_or_replace(
              current.get(state_idx), want_state
          )


def _snapshot_module_recurrent_states(
    model: Any,
) -> list[tuple[Any, str, torch.Tensor]]:
  """Clones recurrent state tensors stored directly on model submodules.

  Some architectures (notably RecurrentGemma's RG-LRU and 1D-conv blocks) keep
  their recurrent state as tensor attributes on the `nn.Module` instances
  (`conv1d_state`, `recurrent_states`) rather than inside the `Cache` object
  passed through `past_key_values`.

  Args:
    model: The model whose submodules should be scanned.

  Returns:
    A list of `(submodule, attr_name, cloned_tensor)` triples.
  """
  if not hasattr(model, "modules"):
    return []
  snapshots = []
  for submodule in model.modules():
    for attr in ("conv1d_state", "recurrent_states"):
      val = getattr(submodule, attr, None)
      if isinstance(val, torch.Tensor):
        snapshots.append((submodule, attr, val.clone()))
  return snapshots


def _restore_module_recurrent_states(
    snapshots: list[tuple[Any, str, torch.Tensor]] | None,
) -> None:
  """Restores submodule recurrent state tensors saved by `_snapshot_module_recurrent_states`."""
  if not snapshots:
    return
  for submodule, attr, want in snapshots:
    current = getattr(submodule, attr, None)
    setattr(submodule, attr, _copy_or_replace(current, want))


def _reset_static_attention_layer(layer: Any, prompt_len: int) -> None:
  """Rewinds one static attention cache layer to its post-prefill state."""
  keys = getattr(layer, "keys", None)
  if keys is not None:
    keys[:, :, prompt_len:, :].zero_()
  values = getattr(layer, "values", None)
  if values is not None:
    values[:, :, prompt_len:, :].zero_()

  cumulative_length = getattr(layer, "cumulative_length", None)
  if isinstance(cumulative_length, torch.Tensor):
    cumulative_length.fill_(prompt_len)
  elif cumulative_length is not None:
    layer.cumulative_length = prompt_len
  # `StaticSlidingWindowLayer` keeps a parallel python int (used to decide
  # eviction and mask sizes) alongside the tensor counter. Leaving it advanced
  # would corrupt sliding-window behaviour for the measured decode steps.
  if hasattr(layer, "cumulative_length_int"):
    layer.cumulative_length_int = prompt_len


class DecodeStepper(common.BaseStepper):

  def __init__(self, output_tokens: int = 16, dynamism: bool = False):
    super().__init__()
    self.output_tokens = output_tokens
    self.dynamism = dynamism
    self.input_ids = None
    self.batch_size = None
    self.prompt_len = None
    self.head_dim = None
    self.num_heads = None
    self.vocab_size = None
    self.device = None
    self.dtype = None
    self.cache = None
    self.next_token = None
    self.prefill_next_token = None
    self.current_pos = None
    self._decode_model = None
    self._prefill_cache_snapshot = None
    self._prefill_linear_states = None
    self._prefill_module_states = None

  def init_with_benchmark_args(
      self, model, input_args, input_kwargs, *args, **kwargs
  ) -> None:
    del input_args
    super().init_with_benchmark_args(model, *args, **kwargs)
    assert self._model is not None
    self.input_ids = input_kwargs["input_ids"]
    self.batch_size = self.input_ids.shape[0]
    self.prompt_len = self.input_ids.shape[1]

    model_config = getattr(self._model, "config")  # pytype: disable=attribute-error
    cfg = _safe_get(model_config, "text_config") or model_config

    self.vocab_size = _safe_get(cfg, "vocab_size")
    num_attn_heads = (
        _safe_get(cfg, "num_attention_heads", "n_head", "num_heads") or 1
    )
    hidden_size = _safe_get(cfg, "hidden_size", "n_embd", "d_model") or 0
    head_dim = _safe_get(cfg, "head_dim", "head_size", "d_kv")
    if head_dim is None and num_attn_heads > 0:
      head_dim = hidden_size // num_attn_heads
    self.head_dim = head_dim or 1

    model_type = (getattr(cfg, "model_type", "") or "").lower()
    if (
        model_type == "falcon"
        and getattr(cfg, "multi_query", False)
        and not getattr(cfg, "new_decoder_architecture", False)
    ):
      # Legacy Falcon multi-query attention shares a single KV head across all
      # query heads, so num_key_value_heads is absent from the config.
      num_heads = 1
    else:
      num_heads = _safe_get(
          cfg, "num_key_value_heads", "num_kv_heads", "n_head_kv"
      )
      if num_heads is None:
        num_heads = num_attn_heads
    self.num_heads = num_heads or 1
    self.device = self.input_ids.device
    self.dtype = (
        self._model.dtype if hasattr(self._model, "dtype") else torch.bfloat16
    )

    # Determine if the model architecture natively supports StaticCache.
    # Models with recurrent / SSM state (e.g. Mamba, RecurrentGemma, RWKV, xLSTM)
    # or non-attention layers do not use standard StaticCache.
    supports_static = getattr(self._model, "_supports_static_cache", None)
    if supports_static is None and hasattr(self._model, "config"):
      supports_static = getattr(cfg, "_supports_static_cache", None)

    if model_type in _RECURRENT_MODEL_TYPES:
      # Pure recurrent / SSM stacks carry no attention KV cache to preallocate,
      # so a StaticCache buys nothing; DynamicCache with crop-based rollback is
      # the cheaper and better-exercised path for them.
      supports_static = False
    elif supports_static is None:
      supports_static = _config_supports_static_cache(cfg)

    if not self.dynamism and supports_static:
      try:
        self.cache = cache_utils.StaticCache(
            config=model_config,
            max_cache_len=self.prompt_len + self.output_tokens,
            batch_size=self.batch_size,
            dtype=self.dtype,
            device=self.device,
        )
        if _supports_early_initialization(self.cache):
          try:
            self.cache.early_initialization(
                batch_size=self.batch_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
          except (ValueError, TypeError, AttributeError):
            # Early initialization may fail for heterogeneous architectures;
            # allow lazy init.
            pass
      except Exception:  # pylint: disable=broad-except
        # Fall back to a dynamic cache rather than failing the benchmark, but
        # keep the config so per-layer types are still honoured.
        self.cache = _make_dynamic_cache(model_config)
    else:
      self.cache = _make_dynamic_cache(model_config)

    _activate_past_recording(self.cache)

    self.next_token = None
    self.prefill_next_token = None
    self.current_pos = None
    self._decode_model = self._model
    self._prefill_cache_snapshot = None
    self._prefill_linear_states = None
    self._prefill_module_states = None

  # Compiles only the inner model used for decode.
  def compile(
      self,
      compile_config: compile_lib.CompileConfig,
      target: target_lib.Target,
  ) -> None:
    # We ignore compile_config.scope and always compile the model for decoding
    self._decode_model = compile_config.apply(self._model, target)

  @staticmethod
  def _prepare_forward_kwargs(
      m: Any,
      cache: Any,
      base_kwargs: dict[str, Any],
  ) -> dict[str, Any]:
    """Prepares kwargs matching the model forward signature."""
    call_kwargs = dict(base_kwargs)
    if cache is None:
      return call_kwargs

    target_mod = m
    visited = set()
    while target_mod is not None and id(target_mod) not in visited:
      visited.add(id(target_mod))
      orig = getattr(target_mod, "_orig_mod", None)
      if orig is None or orig is target_mod:
        break
      target_mod = orig
    target_fn = getattr(target_mod, "forward", target_mod)
    spec = _forward_param_spec(target_fn) if callable(target_fn) else None

    if spec is not None:
      params, has_var_kwargs = spec
      if "past_key_values" in params:
        call_kwargs["past_key_values"] = cache
      elif "cache_params" in params:
        # Some recurrent models (e.g. xLSTM, or Mamba in older transformers
        # releases) expect a bespoke cache object (`xLSTMCache`, `MambaCache`)
        # rather than a `DynamicCache` without linear-attention layers. When
        # our initial placeholder `DynamicCache` has no linear-attention
        # layers, omit `cache_params` on prefill so the model constructs its
        # native cache and returns it in the output.
        if not isinstance(cache, cache_utils.DynamicCache) or any(
            isinstance(layer, cache_utils.LinearAttentionCacheLayerMixin)
            for layer in getattr(cache, "layers", [])
        ):
          call_kwargs["cache_params"] = cache
      elif "state" in params:
        # State-list recurrent models (e.g. RWKV) expect `state=None` on
        # prefill so the model allocates its initial state tensor list, then
        # `state=<list>` on subsequent decode steps.
        if not isinstance(cache, cache_utils.Cache):
          call_kwargs["state"] = cache
      elif has_var_kwargs:
        call_kwargs["past_key_values"] = cache

      if (
          "cache_position" in call_kwargs
          and "cache_position" not in params
          and not has_var_kwargs
      ):
        del call_kwargs["cache_position"]
    else:
      call_kwargs["past_key_values"] = cache

    return call_kwargs

  @staticmethod
  def _step_function(
      m,
      next_token,
      step_kwargs,
      cache,
  ):
    call_kwargs = DecodeStepper._prepare_forward_kwargs(m, cache, step_kwargs)
    return m(
        input_ids=next_token,
        **call_kwargs,
    )

  def _project_hidden_state_to_logits(
      self, hidden_state: torch.Tensor
  ) -> torch.Tensor | None:
    """Projects hidden_state to vocabulary logits using model output or input embeddings."""
    if self._model is None or hidden_state is None:
      return None

    get_out = getattr(self._model, "get_output_embeddings", None)
    out_embed = get_out() if callable(get_out) else None
    res = _apply_projection(out_embed, hidden_state)
    if res is not None:
      return res

    lm_head = getattr(self._model, "lm_head", None)
    res = _apply_projection(lm_head, hidden_state)
    if res is not None:
      return res

    get_in = getattr(self._model, "get_input_embeddings", None)
    in_embed = get_in() if callable(get_in) else None
    if in_embed is None:
      in_embed = _safe_get(
          self._model, "embed_tokens", "wte", "word_embeddings"
      )
    return _apply_projection(in_embed, hidden_state)

  def _unpack_output_and_update_cache(self, output: Any) -> torch.Tensor:
    """Extracts logits and updates self.cache from model output."""
    vocab_size = self.vocab_size
    if vocab_size is None and self._model is not None:
      model_config = getattr(self._model, "config", None)
      cfg = _safe_get(model_config, "text_config") or model_config
      vocab_size = _safe_get(cfg, "vocab_size")

    logits = None
    if isinstance(output, (tuple, list)):
      logits = output[0] if output else None
      if len(output) > 1 and output[1] is not None:
        candidate = output[1]
        # Accept known cache containers, plus custom cache objects (which some
        # models return instead of a `Cache`). Scalars and tensors are never a
        # cache: latching one would make the next step pass it as
        # `past_key_values` and crash inside the model.
        if isinstance(
            candidate, (cache_utils.Cache, tuple, list, dict)
        ) or not isinstance(
            candidate, (torch.Tensor, bool, int, float, complex, str, bytes)
        ):
          self.cache = candidate
      if (
          isinstance(logits, torch.Tensor)
          and vocab_size is not None
          and logits.shape[-1] != vocab_size
      ):
        projected = self._project_hidden_state_to_logits(logits)
        if projected is not None:
          logits = projected
    else:
      cache = _safe_get(output, "past_key_values", "cache_params", "state")
      if cache is not None:
        self.cache = cache

      logits = _safe_get(output, "logits")
      if logits is None:
        hidden_state = _safe_get(output, "last_hidden_state")
        if hidden_state is not None:
          projected = self._project_hidden_state_to_logits(hidden_state)
          logits = projected if projected is not None else hidden_state
        elif isinstance(output, torch.Tensor):
          if vocab_size is not None and output.shape[-1] == vocab_size:
            logits = output
          else:
            projected = self._project_hidden_state_to_logits(output)
            logits = projected if projected is not None else output
        elif hasattr(output, "__getitem__"):
          try:
            first_item = output[0]
            if isinstance(first_item, torch.Tensor):
              if vocab_size is not None and first_item.shape[-1] == vocab_size:
                logits = first_item
              else:
                projected = self._project_hidden_state_to_logits(first_item)
                logits = projected if projected is not None else first_item
          except (IndexError, KeyError, TypeError):
            pass

    # The model may hand back a different cache object than the one we passed
    # in, so re-arm rollback recording on whatever we now hold.
    _activate_past_recording(self.cache)

    assert (
        logits is not None
    ), f"Could not extract logits from model output of type {type(output)}"
    return logits

  def _save_cache_snapshot(self) -> None:
    """Saves the post-prefill cache state that cannot be recomputed by a reset.

    Attention key/value state is cheap to rewind in place (zeroing the decode
    slice for `StaticCache`, `crop()` for `DynamicCache`), so it is never
    copied. Linear-attention (SSM) layers are the exception: their `conv_states`
    and `recurrent_states` are mutated in place, so the parts that the rewind
    cannot reconstruct are snapshotted here. Any other (custom) cache class is
    deep-copied wholesale.
    """
    self._prefill_cache_snapshot = None
    self._prefill_linear_states = None
    self._prefill_module_states = _snapshot_module_recurrent_states(self._model)
    if self.cache is None:
      return
    if isinstance(self.cache, cache_utils.StaticCache):
      # Nothing rewinds conv/recurrent buffers on a static cache.
      self._prefill_linear_states = _snapshot_linear_attention_states(
          self.cache, include_conv=True
      )
      return
    if isinstance(self.cache, cache_utils.DynamicCache):
      # `crop()` rewinds keys/values and conv states; recurrent states it
      # cannot, because they are overwritten rather than appended to.
      self._prefill_linear_states = _snapshot_linear_attention_states(
          self.cache, include_conv=False
      )
      return
    try:
      self._prefill_cache_snapshot = copy.deepcopy(self.cache)
    except Exception:  # pylint: disable=broad-except
      self._prefill_cache_snapshot = None

  # Always eager prefill and save state for future decode.
  def pre_warmup_init(self):
    assert self._model is not None
    _activate_past_recording(self.cache)
    prefill_kwargs = {"use_cache": True}
    prefill_kwargs = self._prepare_forward_kwargs(
        self._model, self.cache, prefill_kwargs
    )
    with torch.no_grad():
      prefill_output = self._model(
          input_ids=self.input_ids,
          **prefill_kwargs,
      )
    prefill_logits = self._unpack_output_and_update_cache(prefill_output)
    self.next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1).unsqueeze(
        -1
    )
    self.prefill_next_token = self.next_token.clone()
    self.current_pos = self.prompt_len
    self._save_cache_snapshot()

  # In-place reset of cache to return it to its post-prefill state.
  # This is used to facilitate warmup decode runs to warm the compilation cache w/o post warmup recompilation.
  def post_warmup_hook(self):
    assert self.prefill_next_token is not None
    assert self.prompt_len is not None
    self.next_token = self.prefill_next_token.clone()

    if isinstance(self.cache, cache_utils.StaticCache):
      for layer in getattr(self.cache, "layers", []):
        _reset_static_attention_layer(layer, self.prompt_len)
    elif isinstance(self.cache, cache_utils.DynamicCache):
      tokens_to_crop = (
          (self.current_pos - self.prompt_len)
          if self.current_pos is not None
          else 0
      )
      if hasattr(self.cache, "crop") and tokens_to_crop >= 0:
        try:
          self.cache.crop(-tokens_to_crop)
        except (TypeError, ValueError, AttributeError):
          pass
    elif self._prefill_cache_snapshot is not None:
      self.cache = copy.deepcopy(self._prefill_cache_snapshot)

    # Hybrid / SSM layers keep state that neither rewind above can reconstruct.
    if self._prefill_linear_states is not None:
      _restore_linear_attention_states(self.cache, self._prefill_linear_states)
    _restore_module_recurrent_states(self._prefill_module_states)

    self.current_pos = self.prompt_len

  def get_cache(self):
    return self.cache

  def get_step_fn(self) -> step_lib.StepFn:

    def step_fn():
      with torch.no_grad():
        decode_output = None
        for _ in range(self.output_tokens):
          cache_position = torch.tensor([self.current_pos], device=self.device)
          decode_output = self._inner_stepper(
              self._decode_model,
              self.next_token,
              {"use_cache": True, "cache_position": cache_position},
              self.cache,
          )
          logits = self._unpack_output_and_update_cache(decode_output)
          self.next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
          assert self.current_pos is not None
          self.current_pos += 1
        return decode_output

    return step_fn


@step_lib.register_stepper(step_lib.StepperType.DECODER_ONLY_DECODE)
def decode(
    output_tokens: int = 16,
    dynamism: bool = False,
) -> step_lib.Stepper:
  """Stateful decode step: prefill, reset, and decode iterations."""
  return DecodeStepper(output_tokens=output_tokens, dynamism=dynamism)
