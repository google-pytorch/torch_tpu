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

"""Tests the sliding-window layers of the Muse Glimmer KV caches.

Sliding-window (SWA) layers keep a fixed window of key/value slots in both
caches, so only the full-attention layers of `DynamicCache` change shape while
decoding. The tests check the window bookkeeping against pure growth, that
cached decoding reproduces cache-free recompute across the window boundary, and
that `torch.compile(dynamic=None)` compiles the decode step once, with or
without `torch._dynamo.mark_dynamic` on the full-attention layers. The same
source runs on CPU and, as `swa_cache_tpu_test`, on TPU.
"""

from collections.abc import Callable, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._dynamo import testing as dynamo_testing
import kv_cache  # from muse_glimmer_30b.llm_inference
import model as model_lib  # from muse_glimmer_30b.llm_inference

_BATCH_SIZE = 1
_NUM_KV_HEADS = 2
_HEAD_DIM = 8
_SWA_SIZE = 4
# Observed differences are ~1e-9 (float32, highest matmul precision), while a
# window that is off by one token moves the logits by ~1e-3.
_RTOL = 1e-5
_ATOL = 1e-5

_Forward = Callable[..., tuple[torch.Tensor, kv_cache.Cache]]


def _device() -> torch.device:
  accelerator = torch.accelerator.current_accelerator(check_available=True)
  return accelerator or torch.device("cpu")


def _config(prefill_seq_len: int, num_decode_steps: int) -> model_lib.Config:
  """A four-layer model: layers 0-2 use sliding-window attention, layer 3 is full."""
  return model_lib.Config(
      num_layers=4,
      dmodel=32,
      dff=64,
      dhead=_HEAD_DIM,
      num_q_heads=4,
      num_kv_heads=_NUM_KV_HEADS,
      vocab_size=32,
      swa_size=_SWA_SIZE,
      prefill_seq_len=prefill_seq_len,
      num_decode_steps=num_decode_steps,
  )


class WindowCacheTest(parameterized.TestCase):
  """Checks the window bookkeeping of both caches against pure growth."""

  def setUp(self):
    super().setUp()
    self.device = _device()

  def _states(self, seq_len: int) -> torch.Tensor:
    return torch.randn(
        _BATCH_SIZE, _NUM_KV_HEADS, seq_len, _HEAD_DIM, device=self.device
    )

  def _run_steps(
      self, cache: kv_cache.Cache, num_layers: int, seq_lens: Sequence[int]
  ) -> None:
    """Feeds `seq_lens[t]` tokens per step and checks every returned window.

    The first slots of a returned window hold the newest tokens in order; a
    `DynamicCache` returns the whole prompt for SWA layers in the prefill and
    all tokens for full-attention layers.
    """
    has_swa = any(cache.window(i) is not None for i in range(num_layers))
    all_keys = [[] for _ in range(num_layers)]
    all_values = [[] for _ in range(num_layers)]
    total = 0
    for step, seq_len in enumerate(seq_lens):
      if step > 0 and has_swa:
        positions = cache.swa_key_positions(seq_len)
        offset = max(0, total + seq_len - cache.num_swa_slots)
        self.assertEqual(
            positions.tolist(),
            [offset + i for i in range(cache.num_swa_slots)],
            f"step {step}",
        )
      total += seq_len
      for layer_idx in range(num_layers):
        msg = f"step {step}, layer {layer_idx}"
        key_states, value_states = self._states(seq_len), self._states(seq_len)
        all_keys[layer_idx].append(key_states)
        all_values[layer_idx].append(value_states)
        keys, values = cache.update(key_states, value_states, layer_idx)
        is_swa = cache.window(layer_idx) is not None
        dynamic = isinstance(cache, kv_cache.DynamicCache)
        if is_swa and (step > 0 or not dynamic):
          self.assertEqual(keys.shape[-2], cache.num_swa_slots, msg)
        elif dynamic:
          self.assertEqual(keys.shape[-2], total, msg)
        kept = min(total, keys.shape[-2])
        for actual, expected in (
            (keys, torch.cat(all_keys[layer_idx], dim=-2)),
            (values, torch.cat(all_values[layer_idx], dim=-2)),
        ):
          self.assertTrue(
              torch.equal(actual[:, :, :kept], expected[:, :, -kept:]), msg
          )
      self.assertEqual(int(cache.get_seq_length()), total, f"step {step}")

  @parameterized.named_parameters(
      ("prefill_below_window", 2, None),
      ("prefill_at_window", 4, None),
      ("prefill_above_window", 6, None),
      ("longer_cache", 2, 16),
  )
  def test_dynamic_cache_rolls_window(
      self, prefill_len: int, max_cache_len: int | None
  ):
    cache = kv_cache.DynamicCache(
        swa_size=_SWA_SIZE,
        swa_layers=(True, True, False),
        max_cache_len=max_cache_len,
    )
    self.assertEqual(cache.num_swa_slots, _SWA_SIZE)
    self._run_steps(cache, num_layers=3, seq_lens=[prefill_len] + [1] * 6)
    for layer_idx in (0, 1):
      self.assertEqual(cache.keys[layer_idx].shape[-2], _SWA_SIZE)
      self.assertEqual(cache.values[layer_idx].shape[-2], _SWA_SIZE)
    self.assertEqual(cache.keys[2].shape[-2], prefill_len + 6)

  def test_dynamic_cache_window_covering_cache_never_rolls(self):
    cache = kv_cache.DynamicCache(
        swa_size=16, swa_layers=(True, False), max_cache_len=8
    )
    self.assertEqual(cache.num_swa_slots, 8)
    self.assertFalse(cache.swa_rolls)
    self._run_steps(cache, num_layers=2, seq_lens=[3] + [1] * 5)

  def test_dynamic_cache_without_swa_layers_grows(self):
    cache = kv_cache.DynamicCache()
    self._run_steps(cache, num_layers=2, seq_lens=[3, 1, 2, 1])

  def test_dynamic_cache_requires_full_attention_last_layer(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Cache contract unit test.
        ValueError, "last layer"
    ):
      kv_cache.DynamicCache(swa_size=_SWA_SIZE, swa_layers=(False, True))

  def test_dynamic_cache_rejects_chunked_decoding(self):
    cache = kv_cache.DynamicCache(swa_size=_SWA_SIZE, swa_layers=(True, False))
    for layer_idx in range(2):
      cache.update(self._states(2), self._states(2), layer_idx)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Cache contract unit test.
        ValueError, "single-token"
    ):
      cache.update(self._states(2), self._states(2), 0)

  def _static_cache(
      self, swa_size: int, max_cache_len: int
  ) -> kv_cache.StaticCache:
    return kv_cache.StaticCache(
        num_layers=3,
        batch_size=_BATCH_SIZE,
        num_kv_heads=_NUM_KV_HEADS,
        max_cache_len=max_cache_len,
        head_dim=_HEAD_DIM,
        dtype=torch.float32,
        device=self.device,
        swa_size=swa_size,
        swa_layers=(True, True, False),
    )

  @parameterized.named_parameters(
      ("prefill_below_window", 2),
      ("prefill_at_window", 4),
  )
  def test_static_cache_rolls_window(self, prefill_len: int):
    cache = self._static_cache(swa_size=_SWA_SIZE, max_cache_len=12)
    self.assertEqual(cache.num_swa_slots, _SWA_SIZE)
    self.assertEqual(cache.rolls, [True, True, False])
    self._run_steps(cache, num_layers=3, seq_lens=[prefill_len] + [1] * 6)
    for layer_idx in (0, 1):
      self.assertEqual(cache.keys[layer_idx].shape[-2], _SWA_SIZE)
    self.assertEqual(cache.keys[2].shape[-2], 12)

  def test_static_cache_window_covering_cache_never_rolls(self):
    cache = self._static_cache(swa_size=16, max_cache_len=8)
    self.assertEqual(cache.num_swa_slots, 8)
    self.assertEqual(cache.rolls, [False, False, False])
    self._run_steps(cache, num_layers=3, seq_lens=[3] + [1] * 5)

  def test_static_cache_rejects_prefill_longer_than_window(self):
    cache = self._static_cache(swa_size=_SWA_SIZE, max_cache_len=12)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Cache contract unit test.
        ValueError, "4 slots"
    ):
      cache.update(self._states(5), self._states(5), 0)

  def test_static_cache_rejects_chunked_decoding(self):
    cache = self._static_cache(swa_size=_SWA_SIZE, max_cache_len=12)
    for layer_idx in range(3):
      cache.update(self._states(2), self._states(2), layer_idx)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Cache contract unit test.
        ValueError, "single-token"
    ):
      cache.update(self._states(2), self._states(2), 0)


class _ModelTest(parameterized.TestCase):
  """Shared setup: a tiny model and cache-free reference logits."""

  def setUp(self):
    super().setUp()
    torch._dynamo.reset()
    torch.set_float32_matmul_precision("highest")
    self.device = _device()

  def _build(
      self, prefill_len: int, num_decode_steps: int = 5
  ) -> tuple[model_lib.Config, model_lib.MuseGlimmer30B, torch.Tensor]:
    config = _config(prefill_len, num_decode_steps)
    model = model_lib.MuseGlimmer30B(config)
    model_lib.init_nn_module(model)
    model.to(self.device).eval()
    tokens = torch.randint(
        0,
        config.vocab_size,
        (_BATCH_SIZE, prefill_len + num_decode_steps),
        device=self.device,
    )
    return config, model, tokens

  def _reference_logits(
      self,
      model: model_lib.MuseGlimmer30B,
      tokens: torch.Tensor,
      prefill_len: int,
  ) -> Sequence[torch.Tensor]:
    """Last-token logits of every prefix, each from a fresh cache."""
    with torch.inference_mode():
      return [
          model(tokens[:, :length])[0]
          for length in range(prefill_len, tokens.shape[1] + 1)
      ]

  def _decode(
      self,
      forward: _Forward,
      tokens: torch.Tensor,
      prefill_len: int,
      cache: kv_cache.Cache | None = None,
      before_step: Callable[[kv_cache.Cache], None] | None = None,
  ) -> tuple[Sequence[torch.Tensor], kv_cache.Cache]:
    """Prefills, then feeds the remaining tokens one at a time."""
    logits = []
    with torch.inference_mode():
      out, cache = forward(tokens[:, :prefill_len], past_key_values=cache)
      logits.append(out)
      for pos in range(prefill_len, tokens.shape[1]):
        if before_step is not None:
          before_step(cache)
        # Decode inputs are fresh tensors, like the sampled token in the
        # runners; a sliced view would make the TPU backend recompile for
        # every storage offset.
        out, cache = forward(
            tokens[:, pos : pos + 1].clone(), past_key_values=cache
        )
        logits.append(out)
    return logits, cache

  def _assert_logits_match(
      self, actual: Sequence[torch.Tensor], expected: Sequence[torch.Tensor]
  ) -> None:
    self.assertLen(actual, len(expected))
    for step, (actual_step, expected_step) in enumerate(zip(actual, expected)):
      torch.testing.assert_close(
          actual_step,
          expected_step,
          rtol=_RTOL,
          atol=_ATOL,
          msg=f"logits at step {step}",
      )


class ModelParityTest(_ModelTest):
  """Cached decoding must match cache-free recompute across the window."""

  @parameterized.named_parameters(
      ("prefill_below_window", 2),
      ("prefill_at_window", 4),
      ("prefill_above_window", 6),
  )
  def test_dynamic_cache_matches_recompute(self, prefill_len: int):
    config, model, tokens = self._build(prefill_len)
    expected = self._reference_logits(model, tokens, prefill_len)
    logits, cache = self._decode(model, tokens, prefill_len)
    self._assert_logits_match(logits, expected)
    self.assertIsInstance(cache, kv_cache.DynamicCache)
    self.assertEqual(cache.num_swa_slots, config.swa_size)
    for layer_idx, is_swa in enumerate(config.swa_layers()):
      expected_len = config.swa_size if is_swa else tokens.shape[1]
      self.assertEqual(cache.keys[layer_idx].shape[-2], expected_len)

  @parameterized.named_parameters(
      ("prefill_below_window", 2),
      ("prefill_at_window", 4),
  )
  def test_static_cache_matches_recompute(self, prefill_len: int):
    config, model, tokens = self._build(prefill_len)
    expected = self._reference_logits(model, tokens, prefill_len)
    cache = kv_cache.StaticCache(
        num_layers=config.num_layers,
        batch_size=_BATCH_SIZE,
        num_kv_heads=config.num_kv_heads,
        max_cache_len=tokens.shape[1],
        head_dim=config.dhead,
        dtype=torch.float32,
        device=self.device,
        swa_size=config.swa_size,
        swa_layers=config.swa_layers(),
    )
    logits, _ = self._decode(model, tokens, prefill_len, cache=cache)
    self._assert_logits_match(logits, expected)
    dynamic_logits, _ = self._decode(model, tokens, prefill_len)
    self._assert_logits_match(dynamic_logits, logits)


class CompileTest(_ModelTest):
  """`torch.compile(dynamic=None)` compiles decoding once across the window."""

  _PREFILL_LEN = 2
  _NUM_DECODE_STEPS = 6

  def setUp(self):
    super().setUp()
    self.backend = "tpu" if self.device.type == "tpu" else "aot_eager"

  def _decode_compiled(
      self, mark: Callable[[torch.Tensor], None] | None
  ) -> int:
    """Decodes across the window boundary and returns the number of graphs."""
    _, model, tokens = self._build(self._PREFILL_LEN, self._NUM_DECODE_STEPS)
    expected = self._reference_logits(model, tokens, self._PREFILL_LEN)
    counter = dynamo_testing.CompileCounterWithBackend(self.backend)
    compiled = torch.compile(model, backend=counter, dynamic=None)

    def mark_full_layers(cache: kv_cache.Cache) -> None:
      # Only the full-attention entries change shape; the SWA windows and the
      # token counters are static and need no marking. The entries are new
      # tensors after every step, so they are marked before each one.
      if mark is None:
        return
      for layer_idx, is_swa in enumerate(cache.swa_layers):
        if not is_swa:
          mark(cache.keys[layer_idx])
          mark(cache.values[layer_idx])

    logits, cache = self._decode(
        compiled, tokens, self._PREFILL_LEN, before_step=mark_full_layers
    )
    self._assert_logits_match(logits, expected)
    self.assertIsInstance(cache, kv_cache.DynamicCache)
    for layer_idx, is_swa in enumerate(cache.swa_layers):
      expected_len = _SWA_SIZE if is_swa else tokens.shape[1]
      self.assertEqual(cache.keys[layer_idx].shape[-2], expected_len)
    return counter.frame_count

  def test_automatic_dynamic_shapes(self):
    # Graphs: prefill, the first decode step with static shapes, and the second
    # one after which the full-attention entries are dynamic. Crossing the
    # window boundary does not recompile.
    self.assertEqual(self._decode_compiled(mark=None), 3)

  def test_mark_dynamic(self):
    # Marking the full-attention entries dynamic compiles decoding once.
    self.assertEqual(
        self._decode_compiled(lambda t: torch._dynamo.mark_dynamic(t, 2)), 2
    )

  def test_mark_dynamic_with_bounds(self):
    max_len = self._PREFILL_LEN + self._NUM_DECODE_STEPS

    def mark(t: torch.Tensor) -> None:
      torch._dynamo.mark_dynamic(t, 2, min=2, max=max_len - 1)

    self.assertEqual(self._decode_compiled(mark), 2)


if __name__ == "__main__":
  absltest.main()
