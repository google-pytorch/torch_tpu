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

"""Unit tests for signature-driven input synthesis."""

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import torch
from tests.model_registry.core import inputs
from tests.model_registry.types import Modality
import transformers


def _tiny_models() -> dict[str, torch.nn.Module]:
  """Tiny randomly initialized HF models, one per input family."""
  return {
      "bert": transformers.BertModel(
          transformers.BertConfig(
              vocab_size=100,
              hidden_size=16,
              num_hidden_layers=1,
              num_attention_heads=2,
              intermediate_size=32,
              max_position_embeddings=32,
          )
      ),
      "vit": transformers.ViTModel(
          transformers.ViTConfig(
              image_size=16,
              patch_size=4,
              hidden_size=16,
              num_hidden_layers=1,
              num_attention_heads=2,
              intermediate_size=32,
          )
      ),
      "t5": transformers.T5Model(
          transformers.T5Config(
              vocab_size=100,
              d_model=16,
              d_kv=8,
              d_ff=32,
              num_layers=1,
              num_heads=2,
          )
      ),
      "whisper": transformers.WhisperModel(
          transformers.WhisperConfig(
              vocab_size=100,
              d_model=16,
              encoder_layers=1,
              decoder_layers=1,
              encoder_attention_heads=2,
              decoder_attention_heads=2,
              encoder_ffn_dim=32,
              decoder_ffn_dim=32,
              num_mel_bins=8,
              max_source_positions=16,
              max_target_positions=16,
              pad_token_id=0,
              bos_token_id=1,
              eos_token_id=1,
              decoder_start_token_id=2,
          )
      ),
  }


def _generate(config, *arg_names: str, **ctx_kwargs) -> inputs.Inputs:
  """Generates inputs for a synthetic `forward(*, <arg_names>=None)`.

  Test helper only: production code passes the model's real `forward`. The
  signature is never opaque, so the modality does not affect the result.
  """

  def forward(**kwargs):
    del kwargs

  setattr(
      forward,
      "__signature__",
      inspect.Signature([
          inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=None)
          for n in arg_names
      ]),
  )
  return inputs.generate(
      forward,
      inputs.InputContext(config=config, **ctx_kwargs),
      modality=Modality.TEXT,
  )


# Test classes use absltest / parameterized directly: seed_test_utils depends
# on torch_tpu C++ extensions, which model_registry tests must not, and the
# code under test draws random tensors from its own seeded CPU generator.
class SignatureInputsTest(
    absltest.TestCase,  # ABSLTEST_OK=seed_test_utils needs C++ deps
):
  """Tier 1 on real HF models: keys, shapes, and a forward pass."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.models = _tiny_models()

  _MODALITIES = {
      "bert": Modality.TEXT,
      "vit": Modality.VISION,
      "t5": Modality.SEQ2SEQ,
      "whisper": Modality.AUDIO,
  }

  def _generate(self, name: str) -> inputs.Inputs:
    model = self.models[name]
    ctx = inputs.InputContext(config=model.config)
    return inputs.generate(model.forward, ctx, modality=self._MODALITIES[name])

  def test_text_encoder(self):
    out = self._generate("bert")
    # Exactly these keys: unlisted arguments (token_type_ids, position_ids,
    # inputs_embeds, ...) are left to HF defaults, as in legacy.
    self.assertCountEqual(out, ["input_ids", "attention_mask"])
    # min(512, max_position_embeddings=32).
    self.assertEqual(out["input_ids"].shape, (1, 32))
    self.assertLess(int(out["input_ids"].max()), 100)

  def test_vision_encoder(self):
    out = self._generate("vit")
    self.assertCountEqual(out, ["pixel_values"])
    self.assertEqual(out["pixel_values"].shape, (1, 3, 16, 16))

  def test_seq2seq(self):
    out = self._generate("t5")
    self.assertCountEqual(
        out, ["input_ids", "attention_mask", "decoder_input_ids"]
    )

  def test_speech_seq2seq(self):
    out = self._generate("whisper")
    # ViT and Whisper accept attention_mask but no input_ids: no token mask.
    self.assertCountEqual(out, ["input_features", "decoder_input_ids"])
    # Whisper requires exactly 2 * max_source_positions mel frames.
    self.assertEqual(out["input_features"].shape, (1, 8, 32))

  def test_generated_inputs_run_forward(self):
    for name, model in self.models.items():
      with self.subTest(name), torch.no_grad():
        model.eval()(**self._generate(name))


class RuleTableTest(
    absltest.TestCase,  # ABSLTEST_OK=seed_test_utils needs C++ deps
):
  """Each Tier 1 rule, exercised through a minimal `forward` signature."""

  def test_rule_table_is_the_documented_set(self):
    # token_type_ids and pixel_mask are intentionally absent: legacy never
    # generates them and HF defaults them to zeros / ones.
    self.assertCountEqual(
        inputs._RULES,  # pylint: disable=protected-access
        [
            "input_ids",
            "decoder_input_ids",
            "attention_mask",
            "pixel_values",
            "input_values",
            "input_features",
            "sample",
            "timestep",
        ],
    )

  def test_attention_mask_requires_input_ids(self):
    for anchor in ("pixel_values", "input_values"):
      with self.subTest(anchor):
        out = _generate({}, anchor, "attention_mask")
        self.assertCountEqual(out, [anchor])

  def test_input_values(self):
    out = _generate({}, "input_values")
    self.assertEqual(out["input_values"].shape, (1, 16000))

  def test_input_values_follow_shape_hint(self):
    out = _generate({}, "input_values", shape=(2, 128))
    self.assertEqual(out["input_values"].shape, (2, 128))

  def test_input_values_cover_feature_encoder_receptive_field(self):
    # wav2vec2 defaults: one frame needs 400 samples.
    config = {
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
        "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    }
    out = _generate(config, "input_values", shape=(1, 128))
    self.assertEqual(out["input_values"].shape, (1, 400))

  def test_decoder_input_ids_use_target_vocab_and_length(self):
    # FSMT-style asymmetric vocabularies: decoder ids index the target vocab.
    config = {
        "src_vocab_size": 50000,
        "tgt_vocab_size": 10,
        "max_target_positions": 64,
    }
    out = _generate(config, "input_ids", "decoder_input_ids")
    self.assertLess(int(out["decoder_input_ids"].max()), 10)
    self.assertEqual(out["decoder_input_ids"].shape, (1, 64))
    self.assertEqual(out["input_ids"].shape, (1, 64))

  def test_decoder_vocab_from_decoder_config(self):
    # Composite configs may key the decoder as `decoder_config`, as legacy.
    config = {"vocab_size": 50000, "decoder_config": {"vocab_size": 10}}
    self.assertEqual(
        inputs.InputContext(config=config).decoder_vocab_size(), 10
    )

  def test_diffusion_sample_and_timestep(self):
    out = _generate({"sample_size": 8, "in_channels": 4}, "sample", "timestep")
    self.assertEqual(out["sample"].shape, (1, 4, 8, 8))
    self.assertEqual(out["timestep"].shape, (1,))


class ModalityDefaultTest(
    absltest.TestCase,  # ABSLTEST_OK=seed_test_utils needs C++ deps
):
  """Tier 2: only opaque signatures fall back to modality defaults."""

  def test_kwargs_only_forward(self):
    def forward(*args, **kwargs):
      del args, kwargs

    ctx = inputs.InputContext(config={"image_size": 8})
    out = inputs.generate(forward, ctx, modality=Modality.VISION)
    self.assertCountEqual(out, ["pixel_values"])
    self.assertEqual(out["pixel_values"].shape, (1, 3, 8, 8))

  def test_unknown_modality_with_opaque_signature_raises(self):
    def forward(*args, **kwargs):
      del args, kwargs

    ctx = inputs.InputContext(config={})
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing input validation.
        ValueError, "No default inputs"
    ):
      inputs.generate(forward, ctx, modality=Modality.UNKNOWN)

  def test_known_signature_without_rules_yields_nothing(self):
    # Models such as PatchTST (`past_values`) need a quirk builder instead.
    self.assertEmpty(_generate({}, "past_values"))


class DefaultsTest(
    absltest.TestCase,  # ABSLTEST_OK=seed_test_utils needs C++ deps
):
  """Seed, vocabulary and EOS defaults shared with transformers_inputs.py."""

  def test_seeded_and_reproducible(self):
    config = {"vocab_size": 1000, "image_size": 8}
    first = _generate(config, "input_ids", "pixel_values")
    torch.manual_seed(1234)  # Global RNG state must not matter.
    second = _generate(config, "input_ids", "pixel_values")
    for key in first:
      self.assertTrue(torch.equal(first[key], second[key]), key)

  def test_src_vocab_size_preferred_for_input_ids(self):
    config = {"src_vocab_size": 10, "vocab_size": 50000, "eos_token_id": 99999}
    out = _generate(config, "input_ids", shape=(1, 64))
    self.assertLess(int(out["input_ids"].max()), 10)

  def test_default_vocab_size(self):
    self.assertEqual(inputs.InputContext(config={}).vocab_size(), 32000)

  def test_boolean_config_values_are_ignored(self):
    # Like _safe_int: a boolean is not a size.
    self.assertEqual(
        inputs.InputContext(config={"vocab_size": True}).vocab_size(), 32000
    )

  def test_input_ids_end_with_eos(self):
    out = _generate({"vocab_size": 100, "eos_token_id": [7, 8]}, "input_ids")
    self.assertTrue(torch.all(out["input_ids"][:, -1] == 7))


class DimensionHeuristicsTest(
    parameterized.TestCase,  # ABSLTEST_OK=seed_test_utils needs C++ deps
):
  """Lookups that mirror _get_max_seq_len / _parse_image_size / channels."""

  @parameterized.named_parameters(
      ("defaults_to_512", {}, None, 512),
      ("capped_at_512", {"max_position_embeddings": 4096}, None, 512),
      ("shape_override", {"max_position_embeddings": 64}, (2, 16), 16),
      ("shape_capped_by_config", {"max_position_embeddings": 64}, (1, 128), 64),
      # RoBERTa-style: position ids start at pad_token_id + 1.
      (
          "padded_positions",
          {"max_position_embeddings": 514, "pad_token_id": 1},
          None,
          512,
      ),
      (
          "min_over_text_config",
          {
              "max_position_embeddings": 4096,
              "text_config": {"max_position_embeddings": 128},
          },
          None,
          128,
      ),
      (
          "min_over_encoder_decoder",
          {
              "encoder": {"max_position_embeddings": 64},
              "decoder": {"max_position_embeddings": 100},
          },
          None,
          64,
      ),
  )
  def test_text_len(self, config, shape, expected):
    ctx = inputs.InputContext(config=config, shape=shape)
    self.assertEqual(ctx.text_len(), expected)

  @parameterized.named_parameters(
      ("default", {}, (224, 224)),
      (
          "vision_config_first",
          {"image_size": 32, "vision_config": {"image_size": 16}},
          (16, 16),
      ),
      ("backbone_config", {"backbone_config": {"image_size": 48}}, (48, 48)),
      ("non_square_kept", {"image_size": [512, 864]}, (512, 864)),
      (
          "timm_input_size",
          {"pretrained_cfg": {"input_size": [3, 288, 288]}},
          (288, 288),
      ),
  )
  def test_image_hw(self, config, expected):
    self.assertEqual(inputs.image_hw(config), expected)

  @parameterized.named_parameters(
      ("default", {}, 3),
      (
          "vision_config_first",
          {"num_channels": 3, "vision_config": {"num_channels": 1}},
          1,
      ),
      ("encoder", {"encoder": {"num_channels": 4}}, 4),
  )
  def test_num_channels(self, config, expected):
    self.assertEqual(inputs.num_channels(config), expected)


if __name__ == "__main__":
  absltest.main()
