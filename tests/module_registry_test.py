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

"""Tests for the module_registry.

This test suite verifies the functionality of the `ModuleRegistry` class and its
underlying providers (Torchvision, TIMM, and Transformers). It ensures that:
1.  All providers can list their available modules.
2.  The registry can aggregate these lists correctly.
3.  Specific modules from each provider can be instantiated and executed
    successfully with their generated sample inputs.
"""

from absl.testing import absltest
from etils import epath
from PIL import Image
import torch
from torch_tpu._internal import testing as tt_testing
from tests import module_registry

_GOLDFISH_IMG_PATH = (
    epath.resource_path("torch_tpu") / "tests/compile/data/goldfish.jpg"
)


class ModuleRegistryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.module_registry = module_registry.ModuleRegistry()

  def test_non_existent_path(self):
    with self.assertRaises(FileNotFoundError):
      module_registry.TimmProvider(base_path="/non/existent/path")

  def test_empty_base_path(self):
    registry = module_registry.ModuleRegistry(base_path="")

    self.assertEqual(registry.list_modules("torchvision"), [])
    self.assertEqual(registry.list_modules("timm"), [])
    self.assertEqual(registry.list_modules("transformers"), [])
    self.assertEqual(registry.list_modules("diffusers"), [])

    self.assertEqual(registry.list_all_modules(), [])

  def test_list_all_modules(self):
    modules = self.module_registry.list_all_modules()

    self.assertIn("torchvision/resnet50", modules)
    self.assertIn("timm/convnext_small.in12k_ft_in1k", modules)
    self.assertIn("timm/resnet50d.ra2_in1k", modules)
    self.assertIn("timm/vgg16.tv_in1k", modules)
    self.assertIn("transformers/openai/gpt-oss-120b", modules)
    self.assertIn("transformers/meta-llama/Llama-3.2-3B", modules)
    self.assertIn("transformers/google/gemma-3-270m", modules)
    self.assertIn("diffusers/stabilityai/stable-diffusion-xl-base-1.0", modules)

  def test_torchvision_list_modules(self):
    modules = self.module_registry.list_modules("torchvision")

    self.assertIn("resnet50", modules)

  def test_timm_list_modules(self):
    modules = self.module_registry.list_modules("timm")

    self.assertIn("convnext_small.in12k_ft_in1k", modules)
    self.assertIn("resnet50d.ra2_in1k", modules)
    self.assertIn("vgg16.tv_in1k", modules)

  def test_transformers_list_modules(self):
    modules = self.module_registry.list_modules("transformers")

    self.assertIn("openai/gpt-oss-120b", modules)
    self.assertIn("meta-llama/Llama-3.2-3B", modules)
    self.assertIn("google/gemma-3-270m", modules)

  def test_torchvision_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "torchvision", "convnext_small"
    )
    model = module_spec.module_factory()
    args, _ = module_spec.sample_inputs_factory()
    expected_output_shape = (args[0].shape[0], 1000)
    model.eval()

    out = model(*args)

    self.assertEqual(out.shape, expected_output_shape)

  def test_timm_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "timm", "mobilenetv3_small_050"
    )
    model = module_spec.module_factory()
    args, _ = module_spec.sample_inputs_factory()
    expected_output_shape = (args[0].shape[0], 1000)
    model.eval()

    out = model(*args)

    self.assertEqual(out.shape, expected_output_shape)

  def test_timm_get_module_spec_pretrained(self):
    module_spec = self.module_registry.get_module_spec(
        "timm", "convnext_small.in12k_ft_in1k", load_weights=True
    )
    model = module_spec.module_factory()
    args, _ = module_spec.sample_inputs_factory()
    expected_output_shape = (args[0].shape[0], 1000)
    model.eval()

    out = model(*args)

    self.assertEqual(out.shape, expected_output_shape)

  def test_timm_get_module_spec_pretrained_is_deterministic(self):
    model_name = "convnext_small.in12k_ft_in1k"

    module_spec_1 = self.module_registry.get_module_spec(
        "timm", model_name, load_weights=True
    )
    model_1 = module_spec_1.module_factory()

    module_spec_2 = self.module_registry.get_module_spec(
        "timm", model_name, load_weights=True
    )
    model_2 = module_spec_2.module_factory()

    # Compare the first layer of weights from two pretrained models.
    # They should be equal.
    weight_p = next(model_1.parameters())
    weight_r = next(model_2.parameters())

    self.assertTrue(torch.equal(weight_p, weight_r))

  def test_timm_preprocessor_loads_image_correctly(self):
    module_spec = self.module_registry.get_module_spec(
        "timm", "convnext_small.in12k_ft_in1k"
    )
    preprocessor = module_spec.preprocessor_factory()
    with _GOLDFISH_IMG_PATH.open("rb") as f:
      img = Image.open(f).convert("RGB")
    image_tensor = preprocessor(img)
    self.assertEqual(image_tensor.shape, (1, 3, 224, 224))

  def test_transformers_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "google/gemma-3-270m"
    )
    model = module_spec.module_factory()
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_ids", kwargs)
    self.assertIn("attention_mask", kwargs)
    self.assertNotIn("pixel_values", kwargs)
    self.assertNotIn("input_features", kwargs)
    expected_logits_shape = (
        *kwargs["input_ids"].shape,
        module_spec.config.vocab_size,
    )
    model.eval()

    out = model(**kwargs)

    self.assertEqual(out.logits.shape, expected_logits_shape)

  def test_transformers_get_module_spec_qwen3_vl(self):
    def scale_down_qwen3_vl_config(config):
      if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = 2
        config.text_config.hidden_size = 128
        config.text_config.intermediate_size = 256
        config.text_config.num_attention_heads = 4
        config.text_config.num_key_value_heads = 2
      if hasattr(config, "vision_config"):
        config.vision_config.depth = 2
        config.vision_config.hidden_size = 64
        config.vision_config.intermediate_size = 128
        config.vision_config.num_heads = 2
        config.vision_config.out_hidden_size = 128
        config.vision_config.deepstack_visual_indexes = [1]
      return config

    for model_name in [
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "Qwen/Qwen3-VL-Embedding-8B",
    ]:
      with self.subTest(model_name=model_name):
        module_spec = self.module_registry.get_module_spec(
            "transformers",
            model_name,
            load_weights=False,
            modify_config_hook=scale_down_qwen3_vl_config,
        )
        model = module_spec.module_factory()
        _, kwargs = module_spec.sample_inputs_factory()
        self.assertIn("input_ids", kwargs)
        self.assertIn("attention_mask", kwargs)
        model.eval()
        out = model(**kwargs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        vocab_size = (
            module_spec.config.text_config.vocab_size
            if hasattr(module_spec.config, "text_config")
            else module_spec.config.vocab_size
        )
        expected_logits_shape = (
            *kwargs["input_ids"].shape,
            vocab_size,
        )
        self.assertEqual(logits.shape, expected_logits_shape)

  def test_transformers_with_modify_config_hook(self):
    def modify_config(config):
      config.vocab_size = 99999
      return config

    module_spec = self.module_registry.get_module_spec(
        "transformers",
        "google/gemma-3-270m",
        load_weights=False,
        modify_config_hook=modify_config,
    )

    self.assertEqual(module_spec.config.vocab_size, 99999)

  def test_transformers_modify_config_hook_and_load_weights_raises(self):
    def modify_config(config):
      return config

    with self.assertRaises(NotImplementedError):
      self.module_registry.get_module_spec(
          "transformers",
          "google/gemma-3-270m",
          load_weights=True,
          modify_config_hook=modify_config,
      )

  def test_torchvision_with_modify_config_hook_raises(self):
    def modify_config(config):
      return config

    with self.assertRaises(ValueError):
      self.module_registry.get_module_spec(
          "torchvision",
          "convnext_small",
          modify_config_hook=modify_config,
      )

  def test_timm_with_modify_config_hook(self):
    def modify_config(config):
      config.input_size = (3, 999, 999)
      return config

    module_spec = self.module_registry.get_module_spec(
        "timm",
        "mobilenetv3_small_050",
        load_weights=False,
        modify_config_hook=modify_config,
    )

    self.assertEqual(module_spec.config.input_size, (3, 999, 999))

  def test_timm_with_modify_config_hook_and_invalid_config(self):
    def modify_config(config):
      config.input_size = (3, 999, 999)
      return config

    module_spec = self.module_registry.get_module_spec(
        "timm",
        "random_model_name",
        load_weights=False,
        modify_config_hook=modify_config,
    )
    self.assertIsNone(module_spec.config)

  def test_diffusers_with_modify_config_hook(self):
    def modify_config(config):
      config["sample_size"] = 999
      return config

    module_spec = self.module_registry.get_module_spec(
        "diffusers",
        "stabilityai/stable-diffusion-xl-base-1.0",
        load_weights=False,
        subfolder="unet",
        modify_config_hook=modify_config,
    )

    self.assertEqual(module_spec.config.get("sample_size"), 999)

  def test_diffusers_with_modify_config_hook_and_load_weights_raises(self):
    def modify_config(config):
      return config

    with self.assertRaises(NotImplementedError):
      self.module_registry.get_module_spec(
          "diffusers",
          "stabilityai/stable-diffusion-xl-base-1.0",
          load_weights=True,
          subfolder="unet",
          modify_config_hook=modify_config,
      )

  def test_transformers_get_module_spec_pretrained_using_sample_inputs(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "google/gemma-3-270m", load_weights=True
    )
    model = module_spec.module_factory()
    model.eval()
    _, kwargs = module_spec.sample_inputs_factory()
    expected_logits_shape = (
        *kwargs["input_ids"].shape,
        module_spec.config.vocab_size,
    )

    out = model(**kwargs)

    self.assertEqual(out.logits.shape, expected_logits_shape)

  def test_transformers_get_module_spec_pretrained_using_tokenizer(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "google/gemma-3-270m", load_weights=True
    )
    model = module_spec.module_factory()
    model.eval()
    tokenizer = module_spec.preprocessor_factory()
    prompt = "Write a haiku about logits."
    max_tokens = 300
    inputs = tokenizer(prompt, return_tensors="pt")
    batch_size, input_token_count = inputs.input_ids.shape
    expected_logits_shape = (
        batch_size,
        module_spec.config.vocab_size,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens - input_token_count,
        do_sample=False,
        temperature=0,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_logits=True,
    )
    output_text = tokenizer.decode(
        outputs.sequences[0], skip_special_tokens=True
    )

    self.assertEqual(outputs.logits[0].shape, expected_logits_shape)
    self.assertGreater(len(output_text), len(prompt))

  def test_diffusers_get_module_spec_random_weights(self):
    module_spec = self.module_registry.get_module_spec(
        "diffusers",
        "stabilityai/stable-diffusion-xl-base-1.0",
        load_weights=False,
        subfolder="unet",
    )
    model = module_spec.module_factory()

    _, kwargs = module_spec.sample_inputs_factory()

    self.assertIn("sample", kwargs)
    self.assertIn("timestep", kwargs)
    self.assertIn("encoder_hidden_states", kwargs)
    # SDXL should have added_cond_kwargs
    self.assertIn("added_cond_kwargs", kwargs)

    self.assertEqual(kwargs["sample"].shape[1], model.config.in_channels)
    self.assertEqual(kwargs["sample"].shape[2], model.config.sample_size)
    self.assertEqual(
        kwargs["encoder_hidden_states"].shape[-1],
        model.config.cross_attention_dim,
    )

  def test_diffusers_get_module_spec_pretrained(self):
    module_spec = self.module_registry.get_module_spec(
        "diffusers",
        "stabilityai/stable-diffusion-xl-base-1.0",
        load_weights=True,
        subfolder="unet",
    )

    # Check some of the config values for the pretrained model without loading
    # the entire model
    self.assertEqual(module_spec.config.get("sample_size"), 128)
    self.assertEqual(module_spec.config.get("cross_attention_dim"), 2048)

    _, kwargs = module_spec.sample_inputs_factory()

    self.assertIn("sample", kwargs)
    self.assertIn("timestep", kwargs)
    self.assertIn("encoder_hidden_states", kwargs)
    # SDXL should have added_cond_kwargs
    self.assertIn("added_cond_kwargs", kwargs)

    self.assertEqual(
        kwargs["encoder_hidden_states"].shape[-1],
        module_spec.config.get("cross_attention_dim"),
    )

  def test_transformers_vision_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "facebook/detr-resnet-50"
    )
    # Instantiate the model to verify it loads with config
    model = module_spec.module_factory()
    self.assertIsNotNone(model)
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("pixel_values", kwargs)
    self.assertLen(kwargs["pixel_values"].shape, 4)  # (batch, channels, H, W)
    self.assertEqual(kwargs["pixel_values"].shape[1], 3)  # RGB channels

  def test_transformers_audio_seq2seq_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "openai/whisper-large-v3"
    )
    model = module_spec.module_factory()
    self.assertIsNotNone(model)
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_features", kwargs)
    self.assertIn("decoder_input_ids", kwargs)

  def test_transformers_gemma4_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "google/gemma-4-31b"
    )
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_ids", kwargs)
    self.assertIn("attention_mask", kwargs)
    self.assertNotIn("pixel_values", kwargs)
    self.assertNotIn("image_position_ids", kwargs)

  def test_transformers_qwen3_5_moe_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "Qwen/Qwen3.5-397B-A17B"
    )
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_ids", kwargs)
    self.assertIn("attention_mask", kwargs)
    self.assertNotIn("pixel_values", kwargs)
    self.assertNotIn("image_grid_thw", kwargs)
    self.assertNotIn("mm_token_type_ids", kwargs)


if __name__ == "__main__":
  absltest.main()
