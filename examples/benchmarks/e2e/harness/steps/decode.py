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

from typing import Any
import torch
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness.steps import common
from transformers import cache_utils


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

    if not self.dynamism:
      self.cache = cache_utils.StaticCache(
          config=model_config,
          max_cache_len=self.prompt_len + self.output_tokens,
          batch_size=self.batch_size,
          dtype=self.dtype,
          device=self.device,
      )
      self.cache.early_initialization(
          batch_size=self.batch_size,
          num_heads=self.num_heads,
          head_dim=self.head_dim,
          dtype=self.dtype,
          device=self.device,
      )
    else:
      self.cache = cache_utils.DynamicCache(config=model_config)
      if hasattr(self.cache, "activate_past_recording"):
        self.cache.activate_past_recording()

    self.next_token = None
    self.prefill_next_token = None
    self.current_pos = None
    self._decode_model = self._model

  # Compiles only the inner model used for decode.
  def compile(
      self,
      compile_config: compile_lib.CompileConfig,
      target: target_lib.Target,
  ) -> None:
    # We ignore compile_config.scope and always compile the model for decoding
    self._decode_model = compile_config.apply(self._model, target)

  @staticmethod
  def _step_function(
      m,
      next_token,
      step_kwargs,
      cache,
  ):
    return m(
        input_ids=next_token,
        past_key_values=cache,
        **step_kwargs,
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

    if isinstance(output, (tuple, list)):
      logits = output[0] if output else None
      if len(output) > 1 and isinstance(output[1], cache_utils.Cache):
        self.cache = output[1]
      if (
          isinstance(logits, torch.Tensor)
          and vocab_size is not None
          and logits.shape[-1] != vocab_size
      ):
        projected = self._project_hidden_state_to_logits(logits)
        if projected is not None:
          logits = projected
    else:
      cache = _safe_get(output, "past_key_values")
      if cache is not None:
        self.cache = cache

      logits = _safe_get(output, "logits")
      if logits is None:
        if isinstance(output, torch.Tensor):
          if vocab_size is not None and output.shape[-1] == vocab_size:
            logits = output
          else:
            projected = self._project_hidden_state_to_logits(output)
            logits = projected if projected is not None else output
        else:
          hidden_state = _safe_get(output, "last_hidden_state")
          if hidden_state is not None:
            projected = self._project_hidden_state_to_logits(hidden_state)
            logits = projected if projected is not None else hidden_state
          elif hasattr(output, "__getitem__"):
            try:
              first_item = output[0]
              if isinstance(first_item, torch.Tensor):
                if (
                    vocab_size is not None
                    and first_item.shape[-1] == vocab_size
                ):
                  logits = first_item
                else:
                  projected = self._project_hidden_state_to_logits(first_item)
                  logits = projected if projected is not None else first_item
            except (IndexError, KeyError, TypeError):
              pass

    if self.dynamism and hasattr(self.cache, "activate_past_recording"):
      self.cache.activate_past_recording()

    assert (
        logits is not None
    ), f"Could not extract logits from model output of type {type(output)}"
    return logits

  # Always eager prefill and save state for future decode.
  def pre_warmup_init(self):
    assert self._model is not None
    if self.dynamism and hasattr(self.cache, "activate_past_recording"):
      self.cache.activate_past_recording()
    with torch.no_grad():
      prefill_output = self._model(
          input_ids=self.input_ids,
          past_key_values=self.cache,
          use_cache=True,
      )
    prefill_logits = self._unpack_output_and_update_cache(prefill_output)
    self.next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1).unsqueeze(
        -1
    )
    self.prefill_next_token = self.next_token.clone()
    self.current_pos = self.prompt_len

  # In-place reset of cache to return it to its post-prefill state.
  # This is used to facilitate warmup decode runs to warm the compilation cache w/o post warmup recompilation.
  def post_warmup_hook(self):
    assert self.prefill_next_token is not None
    assert self.cache is not None
    assert self.prompt_len is not None
    self.next_token = self.prefill_next_token.clone()

    if self.dynamism:
      tokens_to_crop = (
          (self.current_pos - self.prompt_len)
          if self.current_pos is not None
          else 0
      )
      if hasattr(self.cache, "crop"):
        if tokens_to_crop >= 0:
          self.cache.crop(-tokens_to_crop)
    else:

      def _reset_internal(cache, prompt_len):
        for layer in cache.layers:
          if hasattr(layer, "keys"):
            layer.keys[:, :, prompt_len:, :].zero_()
          if hasattr(layer, "values"):
            layer.values[:, :, prompt_len:, :].zero_()
          if hasattr(layer, "cumulative_length"):
            if isinstance(layer.cumulative_length, int):
              layer.cumulative_length = prompt_len
            else:
              layer.cumulative_length.fill_(prompt_len)

      _reset_internal(self.cache, self.prompt_len)

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
