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

"""Tests for transformers_inputs."""

from typing import Any

from absl.testing import absltest
import torch
from tests import seed_test_utils
from tests import transformers_inputs

# The convolutional feature encoder of wav2vec2, HuBERT and WavLM. Each frame
# that it outputs sees 400 waveform samples.
_FEATURE_ENCODER = {
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
}


def _speech_config(**kwargs: Any) -> dict[str, Any]:
  """Returns a minimal config of a wav2vec2 speech recognition model."""
  return {
      "model_type": "wav2vec2",
      "architectures": ["Wav2Vec2ForCTC"],
      **kwargs,
  }


class AudioInputsTest(seed_test_utils.RepeatableTest):

  def test_short_waveform_covers_feature_encoder_receptive_field(self):
    inputs = transformers_inputs.generate_transformers_inputs(
        _speech_config(**_FEATURE_ENCODER), "audio", shape=(2, 128)
    )

    self.assertEqual(inputs["input_values"].shape, (2, 400))

  def test_long_waveform_kept(self):
    inputs = transformers_inputs.generate_transformers_inputs(
        _speech_config(**_FEATURE_ENCODER), "audio", shape=(1, 2048)
    )

    self.assertEqual(inputs["input_values"].shape, (1, 2048))

  def test_waveform_kept_without_feature_encoder(self):
    inputs = transformers_inputs.generate_transformers_inputs(
        _speech_config(), "audio", shape=(1, 128)
    )

    self.assertEqual(inputs["input_values"].shape, (1, 128))


class MultimodalInputsTest(seed_test_utils.RepeatableTest):

  def test_vilt_pixel_mask_is_an_integer_mask_of_the_whole_image(self):
    config = {
        "model_type": "vilt",
        "architectures": ["ViltForQuestionAnswering"],
        "image_size": 384,
    }
    inputs = transformers_inputs.generate_transformers_inputs(
        config, "multimodal", shape=(2, 40)
    )

    self.assertEqual(inputs["pixel_values"].shape, (2, 3, 384, 384))
    self.assertEqual(inputs["pixel_mask"].shape, (2, 384, 384))
    self.assertEqual(inputs["pixel_mask"].dtype, torch.long)
    self.assertTrue(bool(inputs["pixel_mask"].all()))

  def test_fuyu_pixel_values_are_patches_matching_image_tokens(self):
    config = {
        "model_type": "fuyu",
        "architectures": ["FuyuForCausalLM"],
        "image_size": 300,
        "patch_size": 30,
        "num_channels": 3,
        "image_token_id": 71011,
        "vocab_size": 262144,
    }
    inputs = transformers_inputs.generate_transformers_inputs(
        config, "multimodal", shape=(2, 128)
    )

    self.assertEqual(inputs["pixel_values"].shape, (2, 100, 2700))
    self.assertEqual(int((inputs["input_ids"] == 71011).sum()), 2 * 100)

  def test_hunyuan_vl_pixel_values_are_patches_matching_image_tokens(self):
    config = {
        "model_type": "hunyuan_vl",
        "architectures": ["HunYuanVLForConditionalGeneration"],
        "image_token_id": 120120,
        "vision_config": {
            "patch_size": 16,
            "spatial_merge_size": 2,
            "num_channels": 3,
        },
        "text_config": {"vocab_size": 120818},
    }
    inputs = transformers_inputs.generate_transformers_inputs(
        config, "multimodal", shape=(2, 128)
    )

    self.assertEqual(inputs["pixel_values"].shape, (2 * 14 * 14, 768))
    self.assertEqual(inputs["image_grid_thw"].tolist(), [[1, 14, 14]] * 2)
    self.assertEqual(int((inputs["input_ids"] == 120120).sum()), 2 * 58)
    self.assertEqual(int(inputs["mm_token_type_ids"].sum()), 2 * 58)


if __name__ == "__main__":
  absltest.main()
