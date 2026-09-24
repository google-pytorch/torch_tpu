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

"""Cross-architecture coverage for `DecodeStepper`.

Each case resolves a real Hugging Face config through `module_registry`,
shrinks it to a couple of layers, instantiates the architecture with random
weights, and drives it through the full decode lifecycle:

  init_with_benchmark_args -> pre_warmup_init -> warmup steps ->
  post_warmup_hook -> measured steps

The matrix deliberately mixes recurrent (Mamba, RWKV, ...), hybrid
attention+SSM (Jamba, Falcon-H1, Nemotron-H, ...) and plain attention
architectures. The recurrent and hybrid families are the ones whose cache
handling is easy to regress; the dense ones are controls that catch collateral
damage.

This test is marked `notap` (which also sets `notest_oss` in `build_defs.bzl`):
  1. It resolves configs from `module_registry`'s weights mirror, which is not
     hermetic for TAP.
  2. The OSS build pins an older `transformers` release whose cache layer
     classes differ from the internal one (single-tensor SSM state, no
     `STATIC_LAYER_TYPE_MAPPING`), so the cache expectations asserted here are
     verified against internal `transformers`.
"""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import torch
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness import torch_device_ops
from examples.benchmarks.e2e.harness.steps import decode
from tests import module_registry
from tests import seed_test_utils
import transformers

_PROMPT_LEN = 4
_OUTPUT_TOKENS = 4

# Keep instantiation cheap: these are real production configs, so without
# shrinking a single case would allocate several billion parameters.
_MAX_LAYERS = 2
_MAX_VOCAB = 128
_MAX_INTERMEDIATE = 256
_MAX_EXPERTS = 2

# Per-layer schedules that must stay consistent with `num_hidden_layers`.
# Hybrid architectures describe their attention/SSM interleaving with one of
# these, so truncating the layer count without preserving both a recurrent/conv
# layer and an attention/hybrid layer leaves the config self-inconsistent.
_LAYER_SCHEDULE_SEQUENCE_ATTRS = (
    "block_types",
    "layer_types",
    "layers_block_type",
)
_LAYER_SCHEDULE_STRING_ATTRS = ("hybrid_override_pattern",)
_ATTN_SCHEDULE_TYPES = frozenset({
    "attention",
    "chunked_attention",
    "full_attention",
    "hybrid",
    "hybrid_sliding",
    "sliding_attention",
})
_LINEAR_SCHEDULE_TYPES = frozenset({
    "conv",
    "linear_attention",
    "mamba",
    "recurrent",
})

_SIZE_CAPS = (
    ("intermediate_size", _MAX_INTERMEDIATE),
    ("moe_intermediate_size", _MAX_INTERMEDIATE),
    ("shared_expert_intermediate_size", _MAX_INTERMEDIATE),
    ("num_experts", _MAX_EXPERTS),
    ("num_local_experts", _MAX_EXPERTS),
    ("num_experts_per_tok", 1),
)

# (case name, HF repo id, expected model_type, expected cache class name).
#
# `expected_cache` is a golden: `DecodeStepper` must give recurrent stacks a
# `DynamicCache` (they have no attention KV cache worth preallocating) and
# everything else a `StaticCache` when dynamism is off.
_RECURRENT_MODELS = (
    ("mamba", "state-spaces/mamba-130m-hf", "mamba", "DynamicCache"),
    ("mamba2", "AntonV/mamba2-130m-hf", "mamba2", "DynamicCache"),
    ("falcon_mamba", "tiiuae/falcon-mamba-7b", "falcon_mamba", "DynamicCache"),
    ("rwkv", "RWKV/rwkv-4-169m-pile", "rwkv", "DynamicCache"),
    (
        "recurrent_gemma",
        "google/recurrentgemma-2b",
        "recurrent_gemma",
        "DynamicCache",
    ),
)

_HYBRID_MODELS = (
    ("jamba", "ai21labs/Jamba-tiny-random", "jamba", "StaticCache"),
    ("falcon_h1", "tiiuae/Falcon-H1-0.5B-Base", "falcon_h1", "StaticCache"),
    ("bamba", "ibm-ai-platform/Bamba-9B-v1", "bamba", "StaticCache"),
    ("zamba2", "Zyphra/Zamba2-1.2B-instruct", "zamba2", "StaticCache"),
    (
        "nemotron_h",
        "nvidia/Nemotron-H-4B-Base-8K",
        "nemotron_h",
        "StaticCache",
    ),
    ("lfm2", "LiquidAI/LFM2-350M", "lfm2", "StaticCache"),
    (
        "granitemoehybrid",
        "ibm-granite/granite-4.0-tiny-preview",
        "granitemoehybrid",
        "StaticCache",
    ),
)

_DENSE_MODELS = (
    ("llama", "meta-llama/Llama-3.2-1B", "llama", "StaticCache"),
    ("gpt2", "openai-community/gpt2", "gpt2", "StaticCache"),
    ("qwen3", "Qwen/Qwen3-0.6B", "qwen3", "StaticCache"),
    (
        "qwen3_moe",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-MINI",
        "qwen3_moe",
        "StaticCache",
    ),
    ("mistral", "mistralai/Mistral-7B-Instruct-v0.3", "mistral", "StaticCache"),
    ("gemma3_text", "google/gemma-3-270m", "gemma3_text", "StaticCache"),
    ("phi3", "microsoft/Phi-3-mini-4k-instruct", "phi3", "StaticCache"),
    ("gpt_oss", "openai/gpt-oss-20b", "gpt_oss", "StaticCache"),
    ("mixtral", "TitanML/tiny-mixtral", "mixtral", "StaticCache"),
)

_ALL_MODELS = _RECURRENT_MODELS + _HYBRID_MODELS + _DENSE_MODELS


def _try_set(cfg: Any, name: str, value: Any) -> None:
  """Sets `cfg.name` when it is a writable, already-present attribute.

  Several configs expose derived fields such as `layer_types` as read-only
  properties, so a plain `setattr` would raise.

  Args:
    cfg: The config to mutate.
    name: The attribute to set.
    value: The value to set it to.
  """
  if getattr(cfg, name, None) is None:
    return
  try:
    setattr(cfg, name, value)
  except AttributeError:
    pass


def _shrink_schedule_sequence(
    schedule: list[Any], num_layers: int
) -> list[Any]:
  """Truncates a layer-type schedule while keeping both linear and attention layers."""
  first_linear = next(
      (t for t in schedule if t in _LINEAR_SCHEDULE_TYPES), None
  )
  first_attn = next((t for t in schedule if t in _ATTN_SCHEDULE_TYPES), None)
  if first_linear is not None and first_attn is not None and num_layers >= 2:
    return [first_linear, first_attn] + list(schedule[: num_layers - 2])
  return list(schedule[:num_layers])


def _shrink_config(cfg: Any) -> None:
  """Shrinks one config in place so the architecture stays cheap to build."""
  num_layers = getattr(cfg, "num_hidden_layers", None)
  if isinstance(num_layers, int) and num_layers > _MAX_LAYERS:
    num_layers = _MAX_LAYERS
    _try_set(cfg, "num_hidden_layers", num_layers)

    for name in _LAYER_SCHEDULE_SEQUENCE_ATTRS:
      schedule = getattr(cfg, name, None)
      if isinstance(schedule, (list, tuple)) and len(schedule) > num_layers:
        _try_set(
            cfg, name, _shrink_schedule_sequence(list(schedule), num_layers)
        )
    for name in _LAYER_SCHEDULE_STRING_ATTRS:
      schedule = getattr(cfg, name, None)
      if isinstance(schedule, str) and len(schedule) > num_layers:
        _try_set(cfg, name, schedule[:num_layers])

    indices = getattr(cfg, "attn_layer_indices", None)
    if isinstance(indices, (list, tuple)) and indices:
      kept = [i for i in indices if i < num_layers]
      _try_set(cfg, "attn_layer_indices", kept or [num_layers - 1])

    if isinstance(getattr(cfg, "hybrid_layer_ids", None), (list, tuple)):
      block_types = getattr(cfg, "layers_block_type", None) or []
      _try_set(
          cfg,
          "hybrid_layer_ids",
          [i for i, t in enumerate(block_types) if t == "hybrid"],
      )

    attn_period = getattr(cfg, "attn_layer_period", None)
    if isinstance(attn_period, int) and attn_period > num_layers:
      _try_set(cfg, "attn_layer_period", num_layers)
      _try_set(cfg, "attn_layer_offset", num_layers - 1)
    expert_period = getattr(cfg, "expert_layer_period", None)
    if isinstance(expert_period, int) and expert_period > num_layers:
      _try_set(cfg, "expert_layer_period", num_layers)
      expert_offset = getattr(cfg, "expert_layer_offset", 0)
      if isinstance(expert_offset, int) and expert_offset >= num_layers:
        _try_set(cfg, "expert_layer_offset", num_layers - 1)

  vocab_size = getattr(cfg, "vocab_size", None)
  if isinstance(vocab_size, int) and vocab_size > _MAX_VOCAB:
    _try_set(cfg, "vocab_size", _MAX_VOCAB)
    for tok_attr, fallback in (
        ("pad_token_id", 0),
        ("bos_token_id", 1),
        ("eos_token_id", 2),
    ):
      tok_val = getattr(cfg, tok_attr, None)
      if isinstance(tok_val, int) and tok_val >= _MAX_VOCAB:
        _try_set(cfg, tok_attr, fallback)
      elif isinstance(tok_val, list) and any(
          isinstance(x, int) and x >= _MAX_VOCAB for x in tok_val
      ):
        _try_set(cfg, tok_attr, [fallback])

  for name, cap in _SIZE_CAPS:
    value = getattr(cfg, name, None)
    if isinstance(value, int) and value > cap:
      _try_set(cfg, name, cap)


def _shrink(config: Any) -> Any:
  """`modify_config_hook` that shrinks a config and any nested text config."""
  _shrink_config(config)
  text_config = getattr(config, "text_config", None)
  if text_config is not None and text_config is not config:
    _shrink_config(text_config)
  return config


def _linear_state_tensors(layer: Any, name: str) -> dict[int, torch.Tensor]:
  """Returns a layer's `conv_states`/`recurrent_states` as a dict of tensors."""
  states = getattr(layer, name, None)
  if isinstance(states, torch.Tensor):
    return {0: states}
  if isinstance(states, dict):
    return {k: v for k, v in states.items() if isinstance(v, torch.Tensor)}
  return {}


def _snapshot_linear_states(
    cache: Any, model: Any
) -> dict[tuple[str, int, str, int], torch.Tensor]:
  """Clones every recurrent/SSM state tensor held by `cache` or `model`."""
  snapshot: dict[tuple[str, int, str, int], torch.Tensor] = {}
  for layer_idx, layer in enumerate(getattr(cache, "layers", None) or []):
    for name in ("conv_states", "recurrent_states"):
      for state_idx, tensor in _linear_state_tensors(layer, name).items():
        snapshot[("cache_layer", layer_idx, name, state_idx)] = tensor.clone()
  if isinstance(cache, (list, tuple)):
    for idx, item in enumerate(cache):
      if isinstance(item, torch.Tensor):
        snapshot[("state_list", idx, "state", 0)] = item.clone()
  if hasattr(model, "modules"):
    for mod_idx, submodule in enumerate(model.modules()):
      for attr in ("conv1d_state", "recurrent_states"):
        val = getattr(submodule, attr, None)
        if isinstance(val, torch.Tensor):
          snapshot[("module", mod_idx, attr, 0)] = val.clone()
  return snapshot


def _current_linear_state(
    cache: Any, model: Any, key: tuple[str, int, str, int]
) -> torch.Tensor:
  """Reads back the live tensor identified by a `_snapshot_linear_states` key."""
  kind, idx, name, state_idx = key
  if kind == "cache_layer":
    return _linear_state_tensors(cache.layers[idx], name)[state_idx]
  if kind == "state_list":
    return cache[idx]
  modules = list(model.modules())
  return getattr(modules[idx], name)


class DecodeStepperModelCoverageTest(seed_test_utils.RepeatableTest):
  """Runs `DecodeStepper` end to end across many causal-LM architectures."""

  def setUp(self):
    super().setUp()
    self.ops = torch_device_ops.TorchDeviceOps(
        target_lib.make_target(target_lib.Platform.V5E_1X1)
    )
    self.registry = module_registry.ModuleRegistry()

  def _build_model(self, repo: str, expected_model_type: str):
    """Instantiates a shrunken, randomly initialized model from `repo`."""
    spec = self.registry.get_module_spec(
        "transformers",
        repo,
        load_weights=False,
        modify_config_hook=_shrink,
    )
    self.assertEqual(
        getattr(spec.config, "model_type", None),
        expected_model_type,
        f"{repo} no longer declares model_type={expected_model_type!r}; the"
        " case needs updating",
    )
    model = spec.module_factory().to(self.ops.device)
    model.eval()
    return model

  def _make_input_ids(self, model) -> torch.Tensor:
    vocab_size = model.config.get_text_config().vocab_size
    return torch.randint(
        0, vocab_size, (1, _PROMPT_LEN), device=self.ops.device
    )

  @parameterized.named_parameters(*_ALL_MODELS)
  def test_decode_lifecycle(self, repo, expected_model_type, expected_cache):
    model = self._build_model(repo, expected_model_type)
    stepper = decode.decode(output_tokens=_OUTPUT_TOKENS, dynamism=False)
    stepper.init_with_benchmark_args(
        model, (), {"input_ids": self._make_input_ids(model)}
    )

    cache = stepper.get_cache()
    self.assertEqual(
        type(cache).__name__,
        expected_cache,
        f"{repo} selected {type(cache).__name__}, expected {expected_cache}",
    )

    stepper.pre_warmup_init()
    torch.accelerator.synchronize()
    self.assertIsNotNone(stepper.next_token)
    self.assertEqual(stepper.current_pos, _PROMPT_LEN)
    prefill_states = _snapshot_linear_states(stepper.get_cache(), model)

    # Warmup decode advances the cache past the prefill position.
    step_fn = stepper.get_step_fn()
    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, _PROMPT_LEN + _OUTPUT_TOKENS)
    if prefill_states:
      # Guards the restore assertion below from being vacuous.
      self.assertTrue(
          any(
              not torch.equal(
                  _current_linear_state(stepper.get_cache(), model, key), want
              )
              for key, want in prefill_states.items()
          ),
          f"{repo}: warmup decode did not mutate any SSM state, so the"
          " rollback assertion below would be vacuous",
      )

    # Rewinding must put the stepper back exactly where prefill left it.
    stepper.post_warmup_hook()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, _PROMPT_LEN)
    self.assertTrue(torch.equal(stepper.next_token, stepper.prefill_next_token))
    for key, want in prefill_states.items():
      kind, idx, name, state_idx = key
      self.assertTrue(
          torch.equal(
              _current_linear_state(stepper.get_cache(), model, key), want
          ),
          f"{repo}: {kind}[{idx}].{name}[{state_idx}] was not restored to its"
          " post-prefill value",
      )

    # Decoding must still work after the rollback, otherwise the measured
    # steps would run against a corrupted cache.
    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, _PROMPT_LEN + _OUTPUT_TOKENS)

  @parameterized.named_parameters(*_ALL_MODELS)
  def test_decode_lifecycle_with_dynamism(
      self, repo, expected_model_type, expected_cache
  ):
    del expected_cache  # Dynamism always forces a DynamicCache.
    model = self._build_model(repo, expected_model_type)
    stepper = decode.decode(output_tokens=_OUTPUT_TOKENS, dynamism=True)
    stepper.init_with_benchmark_args(
        model, (), {"input_ids": self._make_input_ids(model)}
    )
    self.assertIsInstance(
        stepper.get_cache(), transformers.cache_utils.DynamicCache
    )

    stepper.pre_warmup_init()
    torch.accelerator.synchronize()
    prefill_states = _snapshot_linear_states(stepper.get_cache(), model)

    step_fn = stepper.get_step_fn()
    _ = step_fn()
    torch.accelerator.synchronize()

    stepper.post_warmup_hook()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, _PROMPT_LEN)
    for key, want in prefill_states.items():
      kind, idx, name, state_idx = key
      if name == "conv_states":
        # `crop()` renormalizes conv states back to the kernel width, so the
        # post-prefill (prompt-width) value is deliberately not restored.
        continue
      self.assertTrue(
          torch.equal(
              _current_linear_state(stepper.get_cache(), model, key), want
          ),
          f"{repo}: {kind}[{idx}].{name}[{state_idx}] was not restored to its"
          " post-prefill value",
      )

    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, _PROMPT_LEN + _OUTPUT_TOKENS)


if __name__ == "__main__":
  absltest.main()
