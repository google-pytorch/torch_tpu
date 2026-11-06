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

from collections.abc import Iterator
import contextlib
import copy
import json
import pathlib
import tempfile
import threading
import time
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from etils import epath
from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage
from PIL import Image
import torch
from torch_tpu._internal import testing as tt_testing
from tests import module_cache
from tests import module_registry
from tests import seed_test_utils
from tests import transformers_inputs
import transformers

_GOLDFISH_IMG_PATH = (
    epath.resource_path("torch_tpu") / "tests/compile/data/goldfish.jpg"
)


_TINY_QWEN3_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "vocab_size": 64,
    "max_position_embeddings": 64,
}

# Sets a dtype only on `text_config`.
_TINY_LLAVA_CONFIG = {
    "model_type": "llava",
    "architectures": ["LlavaForConditionalGeneration"],
    "text_config": {
        **_TINY_QWEN3_CONFIG,
        "architectures": None,
        "torch_dtype": "bfloat16",
    },
    "vision_config": {
        "model_type": "clip_vision_model",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "image_size": 8,
        "patch_size": 4,
    },
}


@contextlib.contextmanager
def _recording_parameter_dtypes() -> Iterator[list[torch.dtype]]:
  """Records the dtypes of the parameters registered within the context.

  Yields:
    The dtypes of the parameters in the order they were registered.
  """
  registered_dtypes = []
  handle = torch.nn.modules.module.register_module_parameter_registration_hook(
      lambda module, name, param: registered_dtypes.append(param.dtype)
  )
  try:
    yield registered_dtypes
  finally:
    handle.remove()


class ModuleRegistryTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    module_cache.clear_config_cache()
    self.addCleanup(module_cache.clear_config_cache)
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

  def _create_mock_registry(self, temp_dir):
    path = epath.Path(temp_dir)
    (path / "torchvision" / "resnet50").mkdir(parents=True)
    (path / "timm" / "convnext_small.in12k_ft_in1k").mkdir(parents=True)
    (path / "timm" / "resnet50d.ra2_in1k").mkdir(parents=True)
    (path / "timm" / "vgg16.tv_in1k").mkdir(parents=True)
    (path / "transformers" / "openai" / "gpt-oss-120b").mkdir(parents=True)
    (path / "transformers" / "meta-llama" / "Llama-3.2-3B").mkdir(parents=True)
    (path / "transformers" / "google" / "gemma-3-270m").mkdir(parents=True)
    (path / "diffusers" / "stabilityai" / "stable-diffusion-xl-base-1.0").mkdir(
        parents=True
    )
    return module_registry.ModuleRegistry(base_path=temp_dir)

  def test_list_all_modules(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      registry = self._create_mock_registry(temp_dir)
      modules = registry.list_all_modules()

      self.assertIn("torchvision/resnet50", modules)
      self.assertIn("timm/convnext_small.in12k_ft_in1k", modules)
      self.assertIn("timm/resnet50d.ra2_in1k", modules)
      self.assertIn("timm/vgg16.tv_in1k", modules)
      self.assertIn("transformers/openai/gpt-oss-120b", modules)
      self.assertIn("transformers/meta-llama/Llama-3.2-3B", modules)
      self.assertIn("transformers/google/gemma-3-270m", modules)
      self.assertIn(
          "diffusers/stabilityai/stable-diffusion-xl-base-1.0", modules
      )

  def test_torchvision_list_modules(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      registry = self._create_mock_registry(temp_dir)
      modules = registry.list_modules("torchvision")

      self.assertIn("resnet50", modules)

  def test_timm_list_modules(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      registry = self._create_mock_registry(temp_dir)
      modules = registry.list_modules("timm")

      self.assertIn("convnext_small.in12k_ft_in1k", modules)
      self.assertIn("resnet50d.ra2_in1k", modules)
      self.assertIn("vgg16.tv_in1k", modules)

  def test_transformers_list_modules(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      registry = self._create_mock_registry(temp_dir)
      modules = registry.list_modules("transformers")

      self.assertIn("openai/gpt-oss-120b", modules)
      self.assertIn("meta-llama/Llama-3.2-3B", modules)
      self.assertIn("google/gemma-3-270m", modules)

  def test_sentence_transformers_alias(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      registry = self._create_mock_registry(temp_dir)
      modules = registry.list_modules("sentence-transformers")

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

  def test_torchvision_get_module_spec_pretrained(self):
    module_spec = self.module_registry.get_module_spec(
        "torchvision", "resnet50", load_weights=True
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

  def test_get_module_spec_qualified_name(self):
    module_spec = self.module_registry.get_module_spec(
        "timm/mobilenetv3_small_050"
    )
    model = module_spec.module_factory()
    args, _ = module_spec.sample_inputs_factory()
    expected_output_shape = (args[0].shape[0], 1000)
    model.eval()

    out = model(*args)

    self.assertEqual(out.shape, expected_output_shape)

  def test_get_module_spec_with_redundant_provider_prefix(self):
    module_spec = self.module_registry.get_module_spec(
        "timm", "timm/mobilenetv3_small_050"
    )
    model = module_spec.module_factory()
    args, _ = module_spec.sample_inputs_factory()
    expected_output_shape = (args[0].shape[0], 1000)
    model.eval()

    out = model(*args)

    self.assertEqual(out.shape, expected_output_shape)

  def test_get_module_spec_missing_model_name_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        r"When 'name' is omitted, 'source' must be formatted as"
        r" '{source}/{model_name}', got: 'timm'",
    ):
      self.module_registry.get_module_spec("timm")

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
    self.assertIsNotNone(module_spec.preprocessor_factory)
    tokenizer = module_spec.preprocessor_factory()
    self.assertIsNotNone(tokenizer)

  def test_transformers_causal_lm_architecture_casing_resolution(self):
    def modify_config(config):
      # Simulate AdaptLLM / ChatQA casing: "LLaMAForCausalLM" instead of
      # "LlamaForCausalLM"
      config.architectures = ["LLaMAForCausalLM"]
      return config

    module_spec = self.module_registry.get_module_spec(
        "transformers",
        "meta-llama/Llama-3.2-tiny",
        load_weights=False,
        modify_config_hook=modify_config,
    )
    model = module_spec.module_factory()
    _, kwargs = module_spec.sample_inputs_factory()
    model.eval()
    out = model(**kwargs)
    self.assertTrue(hasattr(out, "logits"))
    self.assertIsNotNone(out.logits)
    self.assertEqual(
        out.logits.shape,
        (*kwargs["input_ids"].shape, module_spec.config.vocab_size),
    )

  def test_transformers_causal_lm_missing_architectures_resolution(self):
    def modify_config(config):
      # Simulate TheBloke / AntonV missing architectures field
      config.architectures = []
      return config

    module_spec = self.module_registry.get_module_spec(
        "transformers",
        "meta-llama/Llama-3.2-tiny",
        load_weights=False,
        modify_config_hook=modify_config,
    )
    model = module_spec.module_factory()
    _, kwargs = module_spec.sample_inputs_factory()
    model.eval()
    out = model(**kwargs)
    self.assertTrue(hasattr(out, "logits"))
    self.assertIsNotNone(out.logits)
    self.assertEqual(
        out.logits.shape,
        (*kwargs["input_ids"].shape, module_spec.config.vocab_size),
    )

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
    self.assertIn("pixel_values", kwargs)
    self.assertIn("image_position_ids", kwargs)

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

  def test_transformers_dinov2_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "facebook/dinov2-base"
    )
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("pixel_values", kwargs)
    self.assertNotIn("input_ids", kwargs)

  def test_transformers_encoder_decoder_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "Ayham/bert_gpt2_summarization_cnndm"
    )
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_ids", kwargs)
    self.assertIn("decoder_input_ids", kwargs)

  def test_transformers_clip_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "openai/clip-vit-base-patch16"
    )
    model = module_spec.module_factory()
    self.assertIsNotNone(model)
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_ids", kwargs)
    self.assertIn("pixel_values", kwargs)

  def test_transformers_llava_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "liuhaotian/llava-v1.5-7b"
    )
    # Skip instantiating 7B model to avoid OOM in tests
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_ids", kwargs)
    self.assertIn("pixel_values", kwargs)

  def test_transformers_git_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "microsoft/git-base-vqav2"
    )
    self.assertEqual(module_spec.modality, module_registry.Modality.MULTIMODAL)
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_ids", kwargs)
    self.assertIn("pixel_values", kwargs)
    self.assertIn("attention_mask", kwargs)

  def test_transformers_mllama_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers",
        "meta-llama/Llama-3.2-11B-Vision",
    )
    # Skip instantiating 11B model to avoid OOM in tests
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_ids", kwargs)
    self.assertIn("pixel_values", kwargs)
    self.assertIn("aspect_ratio_ids", kwargs)
    self.assertIn("aspect_ratio_mask", kwargs)
    self.assertLen(kwargs["pixel_values"].shape, 6)
    self.assertLen(kwargs["aspect_ratio_ids"].shape, 2)
    self.assertLen(kwargs["aspect_ratio_mask"].shape, 3)

  def test_transformers_glm_ocr_get_module_spec(self):
    spec = self.module_registry.get_module_spec(
        "transformers", "zai-org/GLM-OCR"
    )
    self.assertEqual(spec.modality, module_registry.Modality.MULTIMODAL)
    _, kwargs = spec.sample_inputs_factory((1, 2048), "cpu")
    self.assertIn("input_ids", kwargs)
    self.assertIn("pixel_values", kwargs)
    self.assertIn("image_grid_thw", kwargs)
    self.assertIn("mm_token_type_ids", kwargs)

  def test_transformers_kosmos2_get_module_spec(self):
    spec = self.module_registry.get_module_spec(
        "transformers", "microsoft/kosmos-2-patch14-224"
    )
    self.assertEqual(spec.modality, module_registry.Modality.MULTIMODAL)
    _, kwargs = spec.sample_inputs_factory((1, 2048), "cpu")
    self.assertIn("input_ids", kwargs)
    self.assertIn("pixel_values", kwargs)
    self.assertIn("image_embeds_position_mask", kwargs)

  def test_transformers_glm_ocr_generate_inputs(self):
    config = mock.MagicMock(
        model_type="glm_ocr",
        architectures=["GlmOcrForConditionalGeneration"],
        vision_config=mock.MagicMock(
            patch_size=14, temporal_patch_size=2, spatial_merge_size=2
        ),
        image_size=336,
        vocab_size=59281,
        max_position_embeddings=8192,
    )
    kwargs = transformers_inputs.generate_transformers_inputs(
        config,
        module_registry.Modality.MULTIMODAL,
        shape=(1, 2048),
        device="cpu",
    )
    self.assertIn("input_ids", kwargs)
    self.assertIn("pixel_values", kwargs)
    self.assertIn("image_grid_thw", kwargs)
    self.assertIn("mm_token_type_ids", kwargs)
    self.assertEqual(kwargs["image_grid_thw"].shape, (1, 3))
    self.assertEqual(kwargs["pixel_values"].dim(), 2)
    self.assertEqual(kwargs["input_ids"][0, 0].item(), 59256)
    self.assertEqual(kwargs["input_ids"][0, 145].item(), 59257)
    self.assertEqual(kwargs["mm_token_type_ids"][0, 1].item(), 1)

  def test_transformers_kosmos2_generate_inputs(self):
    config = mock.MagicMock(
        model_type="kosmos-2",
        architectures=["Kosmos2ForConditionalGeneration"],
        vision_config=mock.MagicMock(patch_size=14),
        image_size=224,
        latent_query_num=64,
        vocab_size=64007,
        max_position_embeddings=2048,
    )
    kwargs = transformers_inputs.generate_transformers_inputs(
        config,
        module_registry.Modality.MULTIMODAL,
        shape=(1, 2048),
        device="cpu",
    )
    self.assertIn("input_ids", kwargs)
    self.assertIn("pixel_values", kwargs)
    self.assertIn("image_embeds_position_mask", kwargs)
    self.assertEqual(
        kwargs["image_embeds_position_mask"][:, 1:65].sum().item(), 64
    )

  def test_transformers_sequence_length_clamping(self):
    config = mock.MagicMock(
        model_type="electra",
        architectures=["ElectraForPreTraining"],
        max_position_embeddings=512,
        vocab_size=30522,
    )
    kwargs = transformers_inputs.generate_transformers_inputs(
        config, module_registry.Modality.TEXT, shape=(1, 2048), device="cpu"
    )
    self.assertEqual(kwargs["input_ids"].shape, (1, 512))

  def test_transformers_segformer_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "nvidia/segformer-b2-finetuned-ade-512-512"
    )
    model = module_spec.module_factory()
    self.assertIsNotNone(model)
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("pixel_values", kwargs)
    self.assertNotIn("input_ids", kwargs)

  def test_transformers_tapas_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "Anonymous/ReasonBERT-TAPAS"
    )
    model = module_spec.module_factory()
    self.assertIsNotNone(model)
    _, kwargs = module_spec.sample_inputs_factory()
    self.assertIn("input_ids", kwargs)
    self.assertIn("attention_mask", kwargs)
    self.assertIn("token_type_ids", kwargs)
    self.assertEqual(kwargs["token_type_ids"].shape[-1], 7)
    self.assertLen(kwargs["token_type_ids"].shape, 3)

  def test_transformers_gemma4_unified_assistant_get_module_spec(self):
    def modify_config(config):
      # Scale down vocab size for fast unit test execution
      text_cfg = config.text_config
      text_cfg.vocab_size = 1000
      config.vocab_size = 1000
      return config

    module_spec = self.module_registry.get_module_spec(
        "transformers",
        "google/gemma-4-12B-it-assistant",
        load_weights=False,
        modify_config_hook=modify_config,
    )
    model = module_spec.module_factory()
    self.assertIsNotNone(model)
    _, kwargs = module_spec.sample_inputs_factory(shape=(1, 16))
    self.assertIn("inputs_embeds", kwargs)
    self.assertIn("shared_kv_states", kwargs)
    self.assertNotIn("input_ids", kwargs)
    self.assertIn("sliding_attention", kwargs["shared_kv_states"])
    self.assertIn("full_attention", kwargs["shared_kv_states"])
    self.assertEqual(
        kwargs["inputs_embeds"].shape,
        (1, 16, 2 * module_spec.config.backbone_hidden_size),
    )
    model.eval()
    out = model(**kwargs)
    self.assertEqual(out.logits.shape, (1, 16, 1000))

  def test_transformers_gemma4_assistant_get_module_spec(self):
    def modify_config(config):
      # Scale down vocab size for fast unit test execution
      text_cfg = config.text_config
      text_cfg.vocab_size = 1000
      config.vocab_size = 1000
      return config

    module_spec = self.module_registry.get_module_spec(
        "transformers",
        "google/gemma-4-31B-it-assistant",
        load_weights=False,
        modify_config_hook=modify_config,
    )
    model = module_spec.module_factory()
    self.assertIsNotNone(model)
    _, kwargs = module_spec.sample_inputs_factory(shape=(1, 16))
    self.assertIn("inputs_embeds", kwargs)
    self.assertIn("shared_kv_states", kwargs)
    self.assertNotIn("input_ids", kwargs)
    self.assertIn("sliding_attention", kwargs["shared_kv_states"])
    self.assertIn("full_attention", kwargs["shared_kv_states"])
    self.assertEqual(
        kwargs["inputs_embeds"].shape,
        (1, 16, 2 * module_spec.config.backbone_hidden_size),
    )
    model.eval()
    out = model(**kwargs)
    self.assertEqual(out.logits.shape, (1, 16, 1000))

  def test_transformers_model_and_inputs_dtype_alignment(self):
    def modify_config(config):
      config.torch_dtype = "bfloat16"
      return config

    module_spec = self.module_registry.get_module_spec(
        "transformers",
        "google/gemma-3-270m",
        load_weights=False,
        modify_config_hook=modify_config,
    )
    model = module_spec.module_factory()
    _, kwargs = module_spec.sample_inputs_factory()

    # Check model parameter dtypes
    for param in model.parameters():
      self.assertEqual(param.dtype, torch.bfloat16)

    # Check input floating-point tensor dtypes
    for _, v in kwargs.items():
      if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
        self.assertEqual(v.dtype, torch.bfloat16)

  def test_base_provider_cloud_bucket_path(self):
    provider = module_registry.TransformersProvider(base_path="")
    self.assertEqual(provider._cloud_bucket_path, "weights/transformers")

    with flagsaver.flagsaver(gcs_weights_prefix=""):
      self.assertEqual(provider._cloud_bucket_path, "transformers")

    with flagsaver.flagsaver(gcs_weights_bucket=""):
      self.assertIsNone(provider._cloud_bucket_path)

  def test_base_provider_fetch_gcs_file(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch("tempfile.gettempdir", return_value=temp_dir):
        provider = module_registry.TransformersProvider(base_path="")

        def fake_download(bucket, blob, dest_path):
          self.assertEqual(bucket, "torchtpu-test")
          self.assertEqual(
              blob, "weights/transformers/test-org/model/config.json"
          )
          dest_path.parent.mkdir(parents=True, exist_ok=True)
          dest_path.write_bytes(b'{"key": "value"}')
          return True

        with mock.patch(
            "torch_tpu.tests.module_registry._download_gcs_blob",
            side_effect=fake_download,
        ):
          dest = provider.fetch_gcs_file("test-org/model/config.json")
          self.assertIsNotNone(dest)
          self.assertTrue(dest.exists())
          self.assertEqual(dest.read_bytes(), b'{"key": "value"}')

          # Second call should use cache and not trigger download
          with mock.patch(
              "torch_tpu.tests.module_registry._download_gcs_blob"
          ) as mock_dl:
            dest2 = provider.fetch_gcs_file("test-org/model/config.json")
            self.assertEqual(dest, dest2)
            mock_dl.assert_not_called()

  def test_download_gcs_blob_success(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      dest = pathlib.Path(temp_dir) / "sub" / "config.json"
      mock_client = mock.MagicMock(spec=storage.Client)
      mock_bucket = mock.MagicMock(spec=storage.Bucket)
      mock_blob = mock.MagicMock(spec=storage.Blob)

      mock_client.bucket.return_value = mock_bucket
      mock_bucket.blob.return_value = mock_blob
      mock_blob.exists.return_value = True

      def fake_download(filename, timeout=None):
        del timeout  # Unused.
        pathlib.Path(filename).write_bytes(b'{"model_type": "bert"}')

      mock_blob.download_to_filename.side_effect = fake_download

      with mock.patch("google.cloud.storage.Client", return_value=mock_client):
        success = module_registry._download_gcs_blob(
            "torchtpu-test",
            "weights/transformers/test/config.json",
            dest,
        )
      self.assertTrue(success)
      self.assertTrue(dest.exists())
      self.assertEqual(dest.read_bytes(), b'{"model_type": "bert"}')
      mock_client.bucket.assert_called_once_with("torchtpu-test")
      mock_bucket.blob.assert_called_once_with(
          "weights/transformers/test/config.json"
      )
      mock_blob.download_to_filename.assert_called_once_with(
          mock.ANY, timeout=module_registry._GCS_DOWNLOAD_TIMEOUT_SECONDS
      )

  def test_download_gcs_blob_not_found(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      dest = pathlib.Path(temp_dir) / "config.json"
      mock_client = mock.MagicMock(spec=storage.Client)
      mock_bucket = mock.MagicMock(spec=storage.Bucket)
      mock_blob = mock.MagicMock(spec=storage.Blob)

      mock_client.bucket.return_value = mock_bucket
      mock_bucket.blob.return_value = mock_blob
      mock_blob.download_to_filename.side_effect = gcp_exceptions.NotFound(
          "Object not found"
      )

      with mock.patch("google.cloud.storage.Client", return_value=mock_client):
        success = module_registry._download_gcs_blob(
            "torchtpu-test",
            "weights/transformers/missing/config.json",
            dest,
        )
      self.assertFalse(success)
      self.assertFalse(dest.exists())

  def test_download_gcs_blob_api_error(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      dest = pathlib.Path(temp_dir) / "config.json"
      mock_client = mock.MagicMock(spec=storage.Client)
      mock_bucket = mock.MagicMock(spec=storage.Bucket)
      mock_blob = mock.MagicMock(spec=storage.Blob)

      mock_client.bucket.return_value = mock_bucket
      mock_bucket.blob.return_value = mock_blob
      mock_blob.exists.return_value = True
      mock_blob.download_to_filename.side_effect = (
          gcp_exceptions.GoogleAPICallError("Permission denied")
      )

      with mock.patch("google.cloud.storage.Client", return_value=mock_client):
        success = module_registry._download_gcs_blob(
            "torchtpu-test",
            "weights/transformers/error/config.json",
            dest,
        )
      self.assertFalse(success)
      self.assertFalse(dest.exists())

  def test_transformers_get_module_spec_fallback_to_gcs(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch("tempfile.gettempdir", return_value=temp_dir):
        provider = module_registry.TransformersProvider(base_path="")

        # Choose a dummy model name not in local resources
        model_name = "test-org/custom-test-bert"
        minimal_config_json = b"""{
          "model_type": "bert",
          "architectures": ["BertModel"],
          "hidden_size": 32,
          "num_attention_heads": 2,
          "num_hidden_layers": 1,
          "vocab_size": 100
        }"""

        def fake_download(bucket, blob, dest_path):
          del bucket, blob
          dest_path.parent.mkdir(parents=True, exist_ok=True)
          dest_path.write_bytes(minimal_config_json)
          return True

        with mock.patch(
            "torch_tpu.tests.module_registry._download_gcs_blob",
            side_effect=fake_download,
        ) as mock_dl:
          module_spec = provider.get_module_spec(model_name)
          self.assertIsNotNone(module_spec)
          self.assertEqual(module_spec.config.model_type, "bert")
          model = module_spec.module_factory()
          self.assertIsNotNone(model)
          _, kwargs = module_spec.sample_inputs_factory()
          self.assertIn("input_ids", kwargs)
          self.assertIn("attention_mask", kwargs)
          self.assertTrue(mock_dl.called)

  def _get_tiny_transformers_module_spec(
      self, config: dict[str, Any]
  ) -> module_registry.ModuleSpec:
    """Returns the module spec of a tiny model with random weights.

    Args:
      config: The model config.

    Returns:
      The module spec of the model.
    """
    # Each call writes its own config under the same model name.
    module_cache.clear_config_cache()
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch("tempfile.gettempdir", return_value=temp_dir):
        provider = module_registry.TransformersProvider(base_path="")
        model_name = "test-org/tiny-model"
        cache_dir = (
            pathlib.Path(temp_dir)
            / "torch_tpu_cache"
            / "transformers"
            / model_name
        )
        cache_dir.mkdir(parents=True)
        (cache_dir / "config.json").write_text(json.dumps(config))
        return provider.get_module_spec(model_name)

  def _construct_tiny_transformers_model(
      self, config: dict[str, Any], *, dtype: torch.dtype | None = None
  ) -> tuple[list[torch.dtype], torch.nn.Module]:
    """Constructs a tiny model with random weights from the registry.

    Args:
      config: The model config.
      dtype: The dtype to pass to the module factory.

    Returns:
      The dtypes of the parameters in the order they were registered during
      construction, and the constructed model.
    """
    module_spec = self._get_tiny_transformers_module_spec(config)
    with _recording_parameter_dtypes() as registered_dtypes:
      model = module_spec.module_factory(dtype=dtype)
    return registered_dtypes, model

  def _construct_tiny_qwen3(
      self, *, dtype: torch.dtype | None = None, **config_overrides: Any
  ) -> tuple[list[torch.dtype], torch.nn.Module]:
    """Constructs a tiny Qwen3 model with random weights from the registry.

    Args:
      dtype: The dtype to pass to the module factory.
      **config_overrides: Entries to add to the model config.

    Returns:
      The dtypes of the parameters in the order they were registered during
      construction, and the constructed model.
    """
    return self._construct_tiny_transformers_model(
        {**_TINY_QWEN3_CONFIG, **config_overrides}, dtype=dtype
    )

  def test_transformers_dtype_overrides_config_dtype(self):
    for config_overrides, dtype in (
        ({"torch_dtype": "bfloat16"}, torch.float32),
        ({"torch_dtype": "float32"}, torch.bfloat16),
        ({}, torch.bfloat16),
    ):
      with self.subTest(config_overrides=config_overrides, dtype=dtype):
        registered_dtypes, model = self._construct_tiny_qwen3(
            dtype=dtype, **config_overrides
        )

        self.assertNotEmpty(registered_dtypes)
        self.assertEqual(set(registered_dtypes), {dtype})
        self.assertEqual({p.dtype for p in model.parameters()}, {dtype})
        self.assertEqual(torch.get_default_dtype(), torch.float32)

  def test_transformers_dtype_leaves_module_spec_config_unchanged(self):
    # `_from_config` writes its dtype to the config, which later calls without
    # a `dtype` would otherwise construct in.
    module_spec = self._get_tiny_transformers_module_spec(_TINY_QWEN3_CONFIG)

    module_spec.module_factory(dtype=torch.bfloat16)
    model = module_spec.module_factory()

    self.assertEqual({p.dtype for p in model.parameters()}, {torch.float32})

  def test_transformers_config_without_dtype_constructs_in_default_dtype(self):
    # Some callers construct the model under their run dtype as the default
    # dtype instead of passing a `dtype`.
    module_spec = self._get_tiny_transformers_module_spec(_TINY_QWEN3_CONFIG)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
      model = module_spec.module_factory()
    finally:
      torch.set_default_dtype(original_dtype)

    self.assertEqual({p.dtype for p in model.parameters()}, {torch.bfloat16})

  def test_transformers_module_factory_leaves_sub_config_dtype_unchanged(self):
    # Without a `dtype`, the model is constructed in the `text_config` dtype,
    # which `_from_config` would otherwise write to the root config and the
    # other sub-configs of the module spec.
    module_spec = self._get_tiny_transformers_module_spec(_TINY_LLAVA_CONFIG)
    expected_config = module_spec.config.to_dict()

    model = module_spec.module_factory()

    self.assertEqual({p.dtype for p in model.parameters()}, {torch.bfloat16})
    self.assertEqual(module_spec.config.to_dict(), expected_config)

  def test_transformers_constructs_in_dtype_on_meta_device(self):
    # Benchmarks construct models on the meta device and materialize them on
    # the TPU later, so the meta parameters must already have the run dtype.
    module_spec = self._get_tiny_transformers_module_spec(_TINY_QWEN3_CONFIG)

    with _recording_parameter_dtypes() as registered_dtypes:
      with torch.device("meta"):
        model = module_spec.module_factory(dtype=torch.bfloat16)

    self.assertNotEmpty(registered_dtypes)
    self.assertEqual(set(registered_dtypes), {torch.bfloat16})
    self.assertEqual({p.device.type for p in model.parameters()}, {"meta"})
    self.assertEqual({p.dtype for p in model.parameters()}, {torch.bfloat16})

  def test_transformers_tapas_disables_relative_position_embeddings(self):
    # The relative position embeddings per table cell call `Tensor.item()`,
    # which fails on the meta device. The module spec config disables them, so
    # that models constructed from a copy of it do too.
    module_spec = self._get_tiny_transformers_module_spec({
        "model_type": "tapas",
        "architectures": ["TapasModel"],
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 64,
        "max_position_embeddings": 64,
    })
    expected_config = module_spec.config.to_dict()

    with torch.device("meta"):
      model = module_spec.module_factory()
    module_spec.sample_inputs_factory(None, "meta")

    self.assertFalse(model.config.reset_position_index_per_cell)
    self.assertFalse(model.config.init_cell_selection_weights_to_zero)
    # Generating inputs leaves the module spec config unchanged.
    self.assertEqual(module_spec.config.to_dict(), expected_config)

  def test_default_dtype_sets_and_restores_default_dtype(self):
    with module_registry._default_dtype(torch.bfloat16):
      self.assertEqual(torch.get_default_dtype(), torch.bfloat16)

    self.assertEqual(torch.get_default_dtype(), torch.float32)

  def test_default_dtype_restores_default_dtype_on_error(self):
    with self.assertRaises(  # ASSERT_RAISES_OK=Test infrastructure cleanup.
        RuntimeError
    ):
      with module_registry._default_dtype(torch.bfloat16):
        raise RuntimeError("construction failed")

    self.assertEqual(torch.get_default_dtype(), torch.float32)

  def test_default_dtype_leaves_default_dtype_for_none_and_non_floating(self):
    for dtype in (None, torch.int8):
      with self.subTest(dtype=dtype):
        with module_registry._default_dtype(dtype):
          self.assertEqual(torch.get_default_dtype(), torch.float32)

  def test_transformers_constructs_class_without_from_config_in_dtype(self):
    # E.g. `AutoModel*` classes named in `architectures` have `from_config`
    # but not `_from_config`, and other classes may have neither.

    class _ModelWithFromConfig(torch.nn.Module):

      def __init__(self, config: Any):
        del config  # Unused.
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

      @classmethod
      def from_config(cls, config: Any) -> "_ModelWithFromConfig":
        return cls(config)

    class _ModelWithoutFromConfig(torch.nn.Module):

      def __init__(self, config: Any):
        del config  # Unused.
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    for model_cls in (_ModelWithFromConfig, _ModelWithoutFromConfig):
      with self.subTest(model_cls=model_cls.__name__):
        with _recording_parameter_dtypes() as registered_dtypes:
          model = module_registry._instantiate_transformers_model(
              model_cls, transformers.PretrainedConfig(), torch.bfloat16
          )

        self.assertNotEmpty(registered_dtypes)
        self.assertEqual(set(registered_dtypes), {torch.bfloat16})
        self.assertEqual(
            {p.dtype for p in model.parameters()}, {torch.bfloat16}
        )
        self.assertEqual(torch.get_default_dtype(), torch.float32)

  def test_transformers_non_floating_dtype_raises(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Test infrastructure input.
        ValueError, "floating point"
    ):
      self._construct_tiny_qwen3(dtype=torch.int8)

  def test_transformers_composite_model_constructs_in_dtype(self):
    # Composite models construct their sub-models through
    # `AutoModel.from_config(sub_config)`, which constructs in the sub-config
    # dtype.
    registered_dtypes, model = self._construct_tiny_transformers_model(
        {
            **_TINY_LLAVA_CONFIG,
            "vision_config": {
                **_TINY_LLAVA_CONFIG["vision_config"],
                "torch_dtype": "bfloat16",
            },
        },
        dtype=torch.float32,
    )

    self.assertNotEmpty(registered_dtypes)
    self.assertEqual(set(registered_dtypes), {torch.float32})
    self.assertEqual({p.dtype for p in model.parameters()}, {torch.float32})

  def test_transformers_initializes_random_weights_in_bfloat16(self):
    # Configs without a dtype initialize their random weights directly in the
    # benchmark dtype, e.g. with `trunc_normal_` for ViT-style models.
    registered_dtypes, model = self._construct_tiny_transformers_model(
        {
            "model_type": "dinov2",
            "architectures": ["Dinov2Model"],
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
        },
        dtype=torch.bfloat16,
    )

    self.assertNotEmpty(registered_dtypes)
    self.assertEqual(set(registered_dtypes), {torch.bfloat16})
    for name, param in model.named_parameters():
      with self.subTest(name=name):
        self.assertEqual(param.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(param).all())

  def test_transformers_pretrained_loads_in_dtype(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      model_dir = pathlib.Path(temp_dir) / "transformers" / "test-org" / "tiny"
      model_dir.mkdir(parents=True)
      (model_dir / "config.json").write_text(
          json.dumps({**_TINY_QWEN3_CONFIG, "torch_dtype": "bfloat16"})
      )
      provider = module_registry.TransformersProvider(base_path=temp_dir)

      for spec_kwargs, dtype, expected_dtype in (
          ({}, None, torch.bfloat16),
          ({}, torch.float32, torch.float32),
          ({"torch_dtype": torch.float16}, None, torch.float16),
          ({"torch_dtype": torch.float16}, torch.float32, torch.float32),
      ):
        with self.subTest(spec_kwargs=spec_kwargs, dtype=dtype):
          with mock.patch.object(
              transformers.AutoModelForCausalLM, "from_pretrained"
          ) as mock_from_pretrained:
            module_spec = provider.get_module_spec(
                "test-org/tiny", load_weights=True, **spec_kwargs
            )
            module_spec.module_factory(dtype=dtype)

          mock_from_pretrained.assert_called_once_with(
              str(model_dir),
              low_cpu_mem_usage=True,
              ignore_mismatched_sizes=True,
              dtype=expected_dtype,
          )

  def _assert_module_factory_matches_cast(
      self, module_spec: module_registry.ModuleSpec, dtype: torch.dtype
  ) -> None:
    """Asserts that `module_factory(dtype=dtype)` matches a cast afterwards.

    Args:
      module_spec: The module spec of a provider that casts after construction.
      dtype: The dtype to pass to the module factory.
    """
    torch.manual_seed(0)
    model = module_spec.module_factory(dtype=dtype)
    torch.manual_seed(0)
    expected_model = module_spec.module_factory().to(dtype)

    self.assertEqual({p.dtype for p in model.parameters()}, {dtype})
    state_dict = model.state_dict()
    expected_state_dict = expected_model.state_dict()
    self.assertEqual(state_dict.keys(), expected_state_dict.keys())
    for name, tensor in state_dict.items():
      with self.subTest(name=name):
        self.assertTrue(torch.equal(tensor, expected_state_dict[name]))

  def test_torchvision_module_factory_casts_to_dtype(self):
    module_spec = self.module_registry.get_module_spec(
        "torchvision", "mobilenet_v3_small"
    )

    self._assert_module_factory_matches_cast(module_spec, torch.bfloat16)

  def test_timm_module_factory_casts_to_dtype(self):
    # Values that TIMM computes at construction and that are not parameters
    # (e.g. stochastic depth rates) must not depend on `dtype` either.
    module_spec = self.module_registry.get_module_spec(
        "timm", "mobilenetv3_small_050"
    )

    self._assert_module_factory_matches_cast(module_spec, torch.bfloat16)

  def test_diffusers_module_factory_casts_to_dtype(self):
    def tiny_unet_config(config):
      del config  # Unused.
      return {
          "_class_name": "UNet2DConditionModel",
          "block_out_channels": [32, 64],
          "layers_per_block": 1,
          "sample_size": 8,
          "in_channels": 4,
          "out_channels": 4,
          "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D"],
          "up_block_types": ["CrossAttnUpBlock2D", "UpBlock2D"],
          "cross_attention_dim": 32,
      }

    module_spec = self.module_registry.get_module_spec(
        "diffusers",
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        modify_config_hook=tiny_unet_config,
    )

    for dtype, expected_registered_dtype, expected_dtype in (
        (None, torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32, torch.float32),
    ):
      with self.subTest(dtype=dtype):
        with _recording_parameter_dtypes() as registered_dtypes:
          model = module_spec.module_factory(dtype=dtype)

        self.assertNotEmpty(registered_dtypes)
        self.assertEqual(set(registered_dtypes), {expected_registered_dtype})
        self.assertEqual(
            {p.dtype for p in model.parameters()}, {expected_dtype}
        )

  def test_transformers_get_module_spec_uses_cached_gcs_config(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch("tempfile.gettempdir", return_value=temp_dir):
        provider = module_registry.TransformersProvider(base_path="")
        model_name = "cached-org/cached-test-model"

        # Pre-populate temp cache
        gcs_temp_dir = (
            pathlib.Path(temp_dir)
            / "torch_tpu_cache"
            / "transformers"
            / model_name
        )
        gcs_temp_dir.mkdir(parents=True, exist_ok=True)
        (gcs_temp_dir / "config.json").write_bytes(b"""{
          "model_type": "gpt2",
          "architectures": ["GPT2Model"],
          "n_embd": 32,
          "n_head": 2,
          "n_layer": 1,
          "vocab_size": 100
        }""")

        with mock.patch(
            "torch_tpu.tests.module_registry._download_gcs_blob"
        ) as mock_dl:
          module_spec = provider.get_module_spec(model_name)
          self.assertIsNotNone(module_spec)
          self.assertEqual(module_spec.config.model_type, "gpt2")
          self.assertFalse(mock_dl.called)

  def test_transformers_get_module_spec_gcs_failure_raises(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch("tempfile.gettempdir", return_value=temp_dir):
        provider = module_registry.TransformersProvider(base_path="")
        model_name = "nonexistent-org/missing-model"

        with mock.patch(
            "torch_tpu.tests.module_registry._download_gcs_blob",
            return_value=False,
        ):
          with self.assertRaisesRegex(
              ValueError, "could not be loaded from cache or GCS"
          ):
            provider.get_module_spec(model_name)

  def test_module_spec_modality_attributes(self):
    # Vision models
    tv_spec = self.module_registry.get_module_spec(
        "torchvision", "convnext_small"
    )
    self.assertEqual(tv_spec.modality, module_registry.Modality.VISION)

    timm_spec = self.module_registry.get_module_spec(
        "timm", "mobilenetv3_small_050"
    )
    self.assertEqual(timm_spec.modality, module_registry.Modality.VISION)

    detr_spec = self.module_registry.get_module_spec(
        "transformers", "facebook/detr-resnet-50"
    )
    self.assertEqual(detr_spec.modality, module_registry.Modality.VISION)

    resnet_spec = self.module_registry.get_module_spec(
        "transformers", "microsoft/resnet-50"
    )
    self.assertEqual(resnet_spec.modality, module_registry.Modality.VISION)

    # Diffusion models
    diffusers_spec = self.module_registry.get_module_spec(
        "diffusers",
        "stabilityai/stable-diffusion-xl-base-1.0",
        load_weights=False,
        subfolder="unet",
    )
    self.assertEqual(
        diffusers_spec.modality, module_registry.Modality.DIFFUSION
    )

    # Causal LM models
    gemma_spec = self.module_registry.get_module_spec(
        "transformers", "google/gemma-3-270m"
    )
    self.assertEqual(gemma_spec.modality, module_registry.Modality.CAUSAL_LM)

    qwen_spec = self.module_registry.get_module_spec(
        "transformers", "Qwen/Qwen3-4B"
    )
    self.assertEqual(qwen_spec.modality, module_registry.Modality.CAUSAL_LM)

    llama_spec = self.module_registry.get_module_spec(
        "transformers", "meta-llama/Llama-3.2-tiny"
    )
    self.assertEqual(llama_spec.modality, module_registry.Modality.CAUSAL_LM)

    # Audio models
    whisper_spec = self.module_registry.get_module_spec(
        "transformers", "openai/whisper-large-v3"
    )
    self.assertEqual(whisper_spec.modality, module_registry.Modality.AUDIO)

    # Multimodal models
    gemma4_spec = self.module_registry.get_module_spec(
        "transformers", "google/gemma-4-31b"
    )
    self.assertEqual(gemma4_spec.modality, module_registry.Modality.MULTIMODAL)

    qwen_vl_spec = self.module_registry.get_module_spec(
        "transformers", "Qwen/Qwen3-VL-2B-Instruct"
    )
    self.assertEqual(qwen_vl_spec.modality, module_registry.Modality.MULTIMODAL)

    qwen_moe_spec = self.module_registry.get_module_spec(
        "transformers", "Qwen/Qwen3.5-397B-A17B"
    )
    self.assertEqual(
        qwen_moe_spec.modality, module_registry.Modality.MULTIMODAL
    )

    # Verify None text_config and vision_config are not treated as active
    def _add_none_subconfigs(cfg):
      cfg.text_config = None
      cfg.vision_config = None
      return cfg

    none_cfg_spec = self.module_registry.get_module_spec(
        "transformers",
        "google/gemma-3-270m",
        modify_config_hook=_add_none_subconfigs,
    )
    self.assertEqual(none_cfg_spec.modality, module_registry.Modality.CAUSAL_LM)

    # Custom ModuleSpec direct constructor across all modalities
    for mod in module_registry.Modality:
      custom_spec = module_registry.ModuleSpec(
          lambda: None,
          lambda: ((), {}),
          modality=mod,
      )
      self.assertEqual(custom_spec.modality, mod)

  def test_torchvision_weights_backbone_none(self):
    with mock.patch(
        "torchvision.models.get_model", return_value=mock.MagicMock()
    ) as mock_get_model:
      module_spec = self.module_registry.get_module_spec(
          "torchvision", "fasterrcnn_resnet50_fpn"
      )
      model = module_spec.module_factory()
      self.assertIsNotNone(model)
      mock_get_model.assert_called_once_with(
          "fasterrcnn_resnet50_fpn", weights=None, weights_backbone=None
      )

  def test_torchvision_without_weights_backbone_parameter(self):
    with mock.patch(
        "torchvision.models.get_model", return_value=mock.MagicMock()
    ) as mock_get_model:
      module_spec = self.module_registry.get_module_spec(
          "torchvision", "resnet50"
      )
      model = module_spec.module_factory()
      self.assertIsNotNone(model)
      mock_get_model.assert_called_once_with("resnet50", weights=None)

  def test_diffusers_dynamic_subfolder_resolution_via_model_index(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch("tempfile.gettempdir", return_value=temp_dir):
        provider = module_registry.DiffusersProvider(base_path="")
        model_name = "test-diffuser-org/auto-discovered-model"

        model_index_json = json.dumps({
            "_class_name": "StableDiffusionPipeline",
            "transformer": ["diffusers", "Transformer2DModel"],
        }).encode("utf-8")

        config_json = json.dumps({
            "_class_name": "Transformer2DModel",
            "sample_size": 32,
            "in_channels": 4,
            "cross_attention_dim": 1024,
        }).encode("utf-8")

        def fake_download(bucket, blob, dest_path):
          del bucket
          dest_path.parent.mkdir(parents=True, exist_ok=True)
          if blob.endswith("model_index.json"):
            dest_path.write_bytes(model_index_json)
            return True
          if blob.endswith("transformer/config.json"):
            dest_path.write_bytes(config_json)
            return True
          return False

        with mock.patch(
            "torch_tpu.tests.module_registry._download_gcs_blob",
            side_effect=fake_download,
        ):
          spec = provider.get_module_spec(model_name)
          self.assertIsNotNone(spec)
          self.assertEqual(spec.config.get("_class_name"), "Transformer2DModel")
          _, kwargs = spec.sample_inputs_factory()
          self.assertIn("hidden_states", kwargs)
          self.assertIn("encoder_hidden_states", kwargs)
          self.assertEqual(kwargs["hidden_states"].shape, (1, 4, 32, 32))
          self.assertEqual(kwargs["encoder_hidden_states"].shape, (1, 77, 1024))

  def test_diffusers_video_latent_and_cross_attention_dim_fallbacks(self):
    # 3D patch_size adds frames dimension
    def modify_3d(cfg):
      cfg["patch_size"] = [1, 2, 2]
      return cfg

    spec_3d = self.module_registry.get_module_spec(
        "diffusers",
        "stabilityai/stable-diffusion-xl-base-1.0",
        load_weights=False,
        subfolder="unet",
        modify_config_hook=modify_3d,
    )
    _, kwargs_3d = spec_3d.sample_inputs_factory()
    self.assertEqual(len(kwargs_3d["sample"].shape), 5)
    self.assertEqual(kwargs_3d["sample"].shape[2], 2)

    # Tuple/list cross_attention_dim unrolls
    def modify_tuple_cross_attn(cfg):
      cfg["cross_attention_dim"] = [512, 512]
      return cfg

    spec_tuple = self.module_registry.get_module_spec(
        "diffusers",
        "stabilityai/stable-diffusion-xl-base-1.0",
        load_weights=False,
        subfolder="unet",
        modify_config_hook=modify_tuple_cross_attn,
    )
    _, kwargs_tuple = spec_tuple.sample_inputs_factory()
    self.assertEqual(kwargs_tuple["encoder_hidden_states"].shape[-1], 512)

    # Fallback joint_attention_dim when cross_attention_dim is None
    def modify_joint_attn(cfg):
      cfg.pop("cross_attention_dim", None)
      cfg["joint_attention_dim"] = 768
      return cfg

    spec_joint = self.module_registry.get_module_spec(
        "diffusers",
        "stabilityai/stable-diffusion-xl-base-1.0",
        load_weights=False,
        subfolder="unet",
        modify_config_hook=modify_joint_attn,
    )
    _, kwargs_joint = spec_joint.sample_inputs_factory()
    self.assertEqual(kwargs_joint["encoder_hidden_states"].shape[-1], 768)

  def test_transformers_video_5d_tensor_generation(self):
    configs = [
        (
            "videomae",
            {
                "model_type": "videomae",
                "num_frames": 16,
                "image_size": 224,
                "patch_size": 16,
                "tubelet_size": 2,
            },
        ),
        (
            "vivit",
            {"model_type": "vivit", "num_frames": 8, "image_size": 224},
        ),
        (
            "timesformer",
            {"model_type": "timesformer", "num_frames": 8, "image_size": 224},
        ),
        (
            "vjepa2",
            {"model_type": "vjepa2", "num_frames": 8, "image_size": 224},
        ),
        (
            "videoprism",
            {"model_type": "videoprism", "num_frames": 8, "image_size": 224},
        ),
    ]
    for name, cfg_dict in configs:
      mock_cfg = mock.MagicMock()
      for k, v in cfg_dict.items():
        setattr(mock_cfg, k, v)
      inputs = transformers_inputs.generate_transformers_inputs(
          mock_cfg, module_registry.Modality.VISION
      )
      if name == "videoprism":
        self.assertIn("pixel_values_videos", inputs, f"Failed for {name}")
        self.assertEqual(
            inputs["pixel_values_videos"].shape,
            (1, 8, 3, 224, 224),
            f"Failed for {name}",
        )
      elif name == "vjepa2":
        self.assertIn("pixel_values_videos", inputs, f"Failed for {name}")
        self.assertIn("pixel_values", inputs, f"Failed for {name}")
        self.assertEqual(
            inputs["pixel_values"].shape,
            (1, 8, 3, 224, 224),
            f"Failed for {name}",
        )
      else:
        self.assertIn("pixel_values", inputs, f"Failed for {name}")
        self.assertEqual(
            inputs["pixel_values"].shape,
            (1, cfg_dict["num_frames"], 3, 224, 224),
            f"Failed for {name}",
        )
      if name == "videomae":
        self.assertIn("bool_masked_pos", inputs)
        expected_patches = ((224 // 16) ** 2) * (16 // 2)
        self.assertEqual(inputs["bool_masked_pos"].shape, (1, expected_patches))
        self.assertTrue(torch.any(inputs["bool_masked_pos"]))
        self.assertFalse(torch.all(inputs["bool_masked_pos"]))

  def test_transformers_videomae_classification_no_mask(self):
    mock_cfg = mock.MagicMock()
    mock_cfg.model_type = "videomae"
    mock_cfg.architectures = ["VideoMAEForVideoClassification"]
    mock_cfg.num_frames = 16
    mock_cfg.image_size = 224
    mock_cfg.patch_size = 16
    mock_cfg.tubelet_size = 2
    inputs = transformers_inputs.generate_transformers_inputs(
        mock_cfg, module_registry.Modality.VISION
    )
    self.assertIn("pixel_values", inputs)
    self.assertNotIn("bool_masked_pos", inputs)

  def test_transformers_audio_3d_codecs_tensor_generation(self):
    codecs = [
        "dac",
        "encodec",
        "mimi",
        "vibevoice_acoustic_tokenizer",
        "xcodec2",
    ]
    for codec in codecs:
      mock_cfg = mock.MagicMock()
      mock_cfg.model_type = codec
      mock_cfg.architectures = [codec]
      inputs = transformers_inputs.generate_transformers_inputs(
          mock_cfg, module_registry.Modality.AUDIO, shape=(2, 8000)
      )
      if codec == "xcodec2":
        self.assertIn("input_features", inputs, f"Failed for {codec}")
        self.assertEqual(
            inputs["input_features"].shape, (2, 1, 8000), f"Failed for {codec}"
        )
      else:
        self.assertIn("input_values", inputs, f"Failed for {codec}")
        self.assertEqual(
            inputs["input_values"].shape, (2, 1, 8000), f"Failed for {codec}"
        )

  def test_transformers_audio_kyutai_stt_tensor_generation(self):
    mock_cfg = mock.MagicMock()
    mock_cfg.model_type = "kyutai_speech_to_text"
    mock_cfg.architectures = ["KyutaiSpeechToTextForConditionalGeneration"]
    mock_cfg.num_codebooks = 32
    inputs = transformers_inputs.generate_transformers_inputs(
        mock_cfg, module_registry.Modality.AUDIO, shape=(2, 16)
    )
    self.assertIn("input_ids", inputs)
    self.assertEqual(inputs["input_ids"].shape, (2, 16, 33))
    self.assertIn("attention_mask", inputs)

  def test_transformers_audio_moshi_tensor_generation(self):
    mock_cfg = mock.MagicMock()
    mock_cfg.model_type = "moshi"
    mock_cfg.architectures = ["MoshiForConditionalGeneration"]
    mock_cfg.num_codebooks = 8
    mock_cfg.vocab_size = 32000
    mock_cfg.audio_vocab_size = 2048
    inputs = transformers_inputs.generate_transformers_inputs(
        mock_cfg, module_registry.Modality.AUDIO, shape=(2, 1)
    )
    self.assertIn("input_ids", inputs)
    self.assertIn("user_audio_codes", inputs)
    self.assertIn("moshi_audio_codes", inputs)
    self.assertEqual(inputs["user_audio_codes"].shape, (2, 8, 1))
    self.assertEqual(inputs["moshi_audio_codes"].shape, (2, 8, 1))

  def test_transformers_audio_glmasr_tensor_generation(self):
    mock_cfg = mock.MagicMock()
    mock_cfg.model_type = "glmasr"
    mock_cfg.architectures = ["GlmAsrForConditionalGeneration"]
    inputs = transformers_inputs.generate_transformers_inputs(
        mock_cfg, module_registry.Modality.AUDIO, shape=(2, 8)
    )
    self.assertIn("input_ids", inputs)
    self.assertEqual(inputs["input_ids"].shape, (2, 8))
    self.assertIn("attention_mask", inputs)

  def test_transformers_audio_voxtral_realtime_tensor_generation(self):
    mock_cfg = mock.MagicMock()
    mock_cfg.model_type = "voxtral_realtime"
    mock_cfg.architectures = ["VoxtralRealtimeForConditionalGeneration"]
    mock_cfg.downsample_factor = 4
    mock_cfg.audio_length_per_tok = 8
    inputs = transformers_inputs.generate_transformers_inputs(
        mock_cfg, module_registry.Modality.AUDIO, shape=(2, 16)
    )
    self.assertIn("input_ids", inputs)
    self.assertEqual(inputs["input_ids"].shape, (2, 16))
    self.assertIn("input_features", inputs)
    self.assertEqual(inputs["input_features"].shape, (2, 128, 128))

  def test_transformers_multimodal_clvp_tensor_generation(self):
    mock_cfg = mock.MagicMock()
    mock_cfg.model_type = "clvp"
    mock_cfg.architectures = ["ClvpForConditionalGeneration"]
    mock_cfg.decoder_config = {"feature_size": 80}
    inputs = transformers_inputs.generate_transformers_inputs(
        mock_cfg, module_registry.Modality.MULTIMODAL, shape=(2, 16)
    )
    self.assertIn("input_features", inputs)
    self.assertEqual(inputs["input_features"].shape, (2, 80, 100))
    self.assertNotIn("pixel_values", inputs)

  def test_transformers_audio_parakeet_tensor_generation(self):
    mock_cfg = mock.MagicMock()
    mock_cfg.model_type = "parakeet_tdt"
    mock_cfg.architectures = ["ParakeetTdtForCTC"]
    inputs = transformers_inputs.generate_transformers_inputs(
        mock_cfg, module_registry.Modality.AUDIO, shape=(2, 128)
    )
    self.assertIn("input_features", inputs)
    self.assertIn("decoder_input_ids", inputs)
    self.assertEqual(inputs["decoder_input_ids"].shape, (2, 1))

  def test_transformers_audio_clap_dynamic_spec_size(self):
    mock_cfg = mock.MagicMock()
    mock_cfg.model_type = "clap"
    mock_cfg.architectures = ["ClapModel"]
    mock_cfg.audio_config = {"num_mel_bins": 64, "spec_size": 512}
    inputs = transformers_inputs.generate_transformers_inputs(
        mock_cfg, module_registry.Modality.AUDIO, shape=(2, 128)
    )
    self.assertIn("input_features", inputs)
    self.assertEqual(inputs["input_features"].shape, (2, 1, 1001, 64))

  def test_transformers_timeseries_tensor_generation(self):
    ts_models = [
        "autoformer",
        "informer",
        "time_series_transformer",
        "patchtst",
        "patchtsmixer",
        "timesfm",
    ]
    for ts in ts_models:
      mock_cfg = mock.MagicMock()
      mock_cfg.model_type = ts
      mock_cfg.architectures = [ts]
      mock_cfg.context_length = 32
      mock_cfg.input_size = 1
      mock_cfg.num_time_features = 4
      mock_cfg.lags_sequence = [1, 2, 3]
      inputs = transformers_inputs.generate_transformers_inputs(
          mock_cfg, module_registry.Modality.CAUSAL_LM, shape=(2, 32)
      )
      self.assertNotIn("input_ids", inputs, f"Failed for {ts}")
      self.assertNotIn("attention_mask", inputs, f"Failed for {ts}")
      self.assertIn("past_values", inputs, f"Failed for {ts}")
      if ts == "timesfm":
        self.assertIn("freq", inputs)
        self.assertEqual(inputs["past_values"].shape, (2, 32))
      elif ts in ("autoformer", "informer", "time_series_transformer"):
        self.assertIn("past_time_features", inputs, f"Failed for {ts}")
        self.assertIn("past_observed_mask", inputs, f"Failed for {ts}")
        self.assertEqual(
            inputs["past_values"].shape, (2, 35), f"Failed for {ts}"
        )
        self.assertEqual(
            inputs["past_time_features"].shape, (2, 35, 4), f"Failed for {ts}"
        )
      else:
        self.assertIn("past_time_features", inputs, f"Failed for {ts}")
        self.assertIn("past_observed_mask", inputs, f"Failed for {ts}")
        self.assertEqual(
            inputs["past_values"].shape, (2, 64, 1), f"Failed for {ts}"
        )
        self.assertEqual(
            inputs["past_time_features"].shape, (2, 64, 4), f"Failed for {ts}"
        )

  def test_transformers_model_specific_kwargs(self):
    # SigLIP2
    cfg_siglip2 = mock.MagicMock(
        model_type="siglip2",
        architectures=["Siglip2VisionModel"],
        image_size=224,
        patch_size=16,
    )
    inputs_siglip2 = transformers_inputs.generate_transformers_inputs(
        cfg_siglip2, module_registry.Modality.VISION
    )
    self.assertIn("pixel_attention_mask", inputs_siglip2)
    self.assertIn("spatial_shapes", inputs_siglip2)
    self.assertEqual(inputs_siglip2["spatial_shapes"].tolist(), [[14, 14]])

    # OneFormer
    cfg_oneformer = mock.MagicMock(
        model_type="oneformer",
        architectures=["OneFormerModel"],
        image_size=224,
    )
    inputs_oneformer = transformers_inputs.generate_transformers_inputs(
        cfg_oneformer, module_registry.Modality.VISION
    )
    self.assertIn("task_inputs", inputs_oneformer)

    # VitMatte (4 channels)
    cfg_vitmatte = mock.MagicMock(
        model_type="vitmatte",
        architectures=["VitMatteForImageMatting"],
        image_size=224,
        num_channels=4,
    )
    inputs_vitmatte = transformers_inputs.generate_transformers_inputs(
        cfg_vitmatte, module_registry.Modality.VISION
    )
    self.assertEqual(inputs_vitmatte["pixel_values"].shape[1], 4)

    # VitPose
    cfg_vitpose = mock.MagicMock(
        model_type="vitpose",
        architectures=["VitPoseModel"],
        image_size=[256, 192],
        num_channels=3,
    )
    inputs_vitpose = transformers_inputs.generate_transformers_inputs(
        cfg_vitpose, module_registry.Modality.VISION
    )
    self.assertEqual(inputs_vitpose["pixel_values"].shape, (1, 3, 256, 192))
    self.assertIn("dataset_index", inputs_vitpose)

    # Bros (bbox)
    cfg_bros = mock.MagicMock(
        model_type="bros",
        architectures=["BrosModel"],
        vocab_size=30522,
    )
    inputs_bros = transformers_inputs.generate_transformers_inputs(
        cfg_bros, module_registry.Modality.CAUSAL_LM, shape=(1, 32)
    )
    self.assertIn("bbox", inputs_bros)
    self.assertEqual(inputs_bros["bbox"].shape, (1, 32, 4))

    # Pix2Struct (flattened_patches)
    cfg_p2s = mock.MagicMock(
        model_type="pix2struct",
        architectures=["Pix2StructForConditionalGeneration"],
        max_patches=16,
        text_config=mock.MagicMock(hidden_size=768),
    )
    inputs_p2s = transformers_inputs.generate_transformers_inputs(
        cfg_p2s, module_registry.Modality.MULTIMODAL, shape=(1, 32)
    )
    self.assertNotIn("pixel_values", inputs_p2s)
    self.assertIn("flattened_patches", inputs_p2s)
    self.assertEqual(inputs_p2s["flattened_patches"].shape, (1, 16, 770))
    self.assertIn("decoder_input_ids", inputs_p2s)

    # Qwen2-VL / Qwen3-VL / Holo (image_grid_thw)
    cfg_qwenvl = mock.MagicMock(
        model_type="qwen2_5_vl",
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        image_size=224,
    )
    inputs_qwenvl = transformers_inputs.generate_transformers_inputs(
        cfg_qwenvl, module_registry.Modality.MULTIMODAL
    )
    self.assertIn("image_grid_thw", inputs_qwenvl)
    self.assertEqual(inputs_qwenvl["image_grid_thw"].tolist(), [[1, 16, 16]])

    # InstructBlip / Blip
    cfg_iblip = mock.MagicMock(
        model_type="instructblip",
        architectures=["InstructBlipForConditionalGeneration"],
        image_size=224,
    )
    inputs_iblip = transformers_inputs.generate_transformers_inputs(
        cfg_iblip, module_registry.Modality.MULTIMODAL
    )
    self.assertIn("qformer_input_ids", inputs_iblip)
    self.assertIn("qformer_attention_mask", inputs_iblip)
    self.assertIn("decoder_input_ids", inputs_iblip)

    # LXMERT
    cfg_lxmert = mock.MagicMock(
        model_type="lxmert",
        architectures=["LxmertForQuestionAnswering"],
        vocab_size=30522,
        visual_feat_dim=2048,
    )
    inputs_lxmert = transformers_inputs.generate_transformers_inputs(
        cfg_lxmert, module_registry.Modality.CAUSAL_LM, shape=(1, 32)
    )
    self.assertIn("visual_feats", inputs_lxmert)
    self.assertIn("visual_pos", inputs_lxmert)
    self.assertEqual(inputs_lxmert["visual_feats"].shape, (1, 36, 2048))
    self.assertEqual(inputs_lxmert["visual_pos"].shape, (1, 36, 4))

    # SpeechT5 (TextToSpeech)
    cfg_speecht5_tts = mock.MagicMock(
        model_type="speecht5",
        architectures=["SpeechT5ForTextToSpeech"],
        vocab_size=81,
        num_mel_bins=80,
    )
    inputs_tts = transformers_inputs.generate_transformers_inputs(
        cfg_speecht5_tts, module_registry.Modality.AUDIO
    )
    self.assertIn("input_ids", inputs_tts)
    self.assertIn("decoder_input_values", inputs_tts)
    self.assertIn("speaker_embeddings", inputs_tts)
    self.assertIn("attention_mask", inputs_tts)
    self.assertIn("decoder_attention_mask", inputs_tts)
    self.assertNotIn("labels", inputs_tts)
    self.assertNotIn("input_values", inputs_tts)
    self.assertNotIn("decoder_input_ids", inputs_tts)
    self.assertEqual(inputs_tts["attention_mask"].shape, (1, 8))
    self.assertEqual(inputs_tts["decoder_input_values"].shape, (1, 8, 80))
    self.assertEqual(inputs_tts["speaker_embeddings"].shape, (1, 512))

    # SpeechT5 (SpeechToText)
    cfg_speecht5_asr = mock.MagicMock(
        model_type="speecht5",
        architectures=["SpeechT5ForSpeechToText"],
        vocab_size=81,
        num_mel_bins=80,
    )
    inputs_asr = transformers_inputs.generate_transformers_inputs(
        cfg_speecht5_asr, module_registry.Modality.AUDIO
    )
    self.assertIn("input_values", inputs_asr)
    self.assertIn("decoder_input_ids", inputs_asr)
    self.assertIn("attention_mask", inputs_asr)
    self.assertIn("decoder_attention_mask", inputs_asr)
    self.assertNotIn("labels", inputs_asr)
    self.assertNotIn("input_ids", inputs_asr)
    self.assertNotIn("decoder_input_values", inputs_asr)
    self.assertNotIn("speaker_embeddings", inputs_asr)
    self.assertEqual(inputs_asr["attention_mask"].shape, (1, 16000))

    # SpeechT5 (SpeechToSpeech)
    cfg_speecht5_vc = mock.MagicMock(
        model_type="speecht5",
        architectures=["SpeechT5ForSpeechToSpeech"],
        vocab_size=81,
        num_mel_bins=80,
    )
    inputs_vc = transformers_inputs.generate_transformers_inputs(
        cfg_speecht5_vc, module_registry.Modality.AUDIO
    )
    self.assertIn("input_values", inputs_vc)
    self.assertIn("decoder_input_values", inputs_vc)
    self.assertIn("speaker_embeddings", inputs_vc)
    self.assertIn("attention_mask", inputs_vc)
    self.assertIn("decoder_attention_mask", inputs_vc)
    self.assertNotIn("labels", inputs_vc)
    self.assertNotIn("decoder_input_ids", inputs_vc)
    self.assertEqual(inputs_vc["attention_mask"].shape, (1, 16000))

  def test_transformers_get_max_seq_len_bounds(self):
    cfg_normal = mock.MagicMock(max_position_embeddings=2048)
    self.assertEqual(module_registry._get_max_seq_len(cfg_normal), 2048)

    cfg_neg = mock.MagicMock(max_position_embeddings=-1)
    self.assertEqual(module_registry._get_max_seq_len(cfg_neg), 512)

    cfg_oversized = mock.MagicMock(max_position_embeddings=1000000)
    self.assertEqual(module_registry._get_max_seq_len(cfg_oversized), 512)

    cfg_empty = mock.MagicMock(spec=[])
    self.assertEqual(module_registry._get_max_seq_len(cfg_empty), 512)
    self.assertEqual(
        module_registry._get_max_seq_len(cfg_empty, default=256), 256
    )

  def test_safe_int(self):
    self.assertEqual(module_registry._safe_int(42), 42)
    self.assertEqual(module_registry._safe_int("128"), 128)
    self.assertEqual(module_registry._safe_int(None, default=10), 10)
    self.assertEqual(module_registry._safe_int(True, default=5), 5)
    self.assertEqual(module_registry._safe_int(False, default=5), 5)
    self.assertEqual(module_registry._safe_int("invalid", default=0), 0)
    self.assertEqual(module_registry._safe_int(0, default=10, min_val=1), 10)
    self.assertEqual(module_registry._safe_int(5, default=10, min_val=1), 5)
    self.assertEqual(
        module_registry._safe_int(mock.MagicMock(), default=10), 10
    )

  def test_parse_image_size(self):
    cfg_depth_pro = {
        "model_type": "depth_pro",
        "patch_size": 384,
        "scaled_images_ratios": [0, -1, 0.5],
    }
    self.assertEqual(
        module_registry._parse_image_size(cfg_depth_pro),
        max(int(384 / 0.5), 1536),
    )
    cfg_zero_ratios = {
        "model_type": "depth_pro",
        "patch_size": 384,
        "scaled_images_ratios": [0],
    }
    self.assertEqual(
        module_registry._parse_image_size(cfg_zero_ratios),
        max(int(384 / 0.25), 1536),
    )

  def test_extract_diffusers_subfolder_candidates(self):
    idx_dict = {
        "_class_name": "StableDiffusionPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    candidates = module_registry._extract_diffusers_subfolder_candidates(
        idx_dict
    )
    self.assertEqual(candidates, ["unet"])

    idx_transformer = {
        "_class_name": "FluxPipeline",
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    candidates_transformer = (
        module_registry._extract_diffusers_subfolder_candidates(idx_transformer)
    )
    self.assertEqual(candidates_transformer, ["transformer"])

    self.assertEqual(
        module_registry._extract_diffusers_subfolder_candidates({}), []
    )

  def test_determine_modality_vision_and_multimodal(self):
    self.assertEqual(
        module_registry._determine_modality({
            "model_type": "clip_vision_model",
        }),
        module_registry.Modality.VISION,
    )
    self.assertEqual(
        module_registry._determine_modality({
            "model_type": "siglip",
            "architectures": ["SiglipForImageClassification"],
            "text_config": {},
            "vision_config": {"image_size": 224},
        }),
        module_registry.Modality.VISION,
    )
    self.assertEqual(
        module_registry._determine_modality({
            "model_type": "chmv2",
            "architectures": ["CHMv2ForDepthEstimation"],
            "backbone_config": {"image_size": 224},
        }),
        module_registry.Modality.VISION,
    )
    # Config object with dict backbone_config
    cfg_obj = mock.MagicMock(
        _class_name="",
        _name_or_path="",
        model_type="chmv2",
        architectures=["CHMv2ForDepthEstimation"],
        is_encoder_decoder=False,
        text_config=None,
        vision_config=None,
        image_size=None,
        num_channels=None,
        backbone_config={"image_size": 224},
        vocab_size=None,
    )
    self.assertEqual(
        module_registry._determine_modality(cfg_obj),
        module_registry.Modality.VISION,
    )
    self.assertEqual(
        module_registry._determine_modality({
            "model_type": "clip",
            "text_config": {},
            "vision_config": {},
        }),
        module_registry.Modality.MULTIMODAL,
    )
    # Multimodal architecture with 'Image' in name should remain MULTIMODAL
    self.assertEqual(
        module_registry._determine_modality({
            "model_type": "blip",
            "architectures": ["BlipForImageTextRetrieval"],
        }),
        module_registry.Modality.MULTIMODAL,
    )
    # GitForCausalLM (vision_config + vocab_size) is MULTIMODAL
    self.assertEqual(
        module_registry._determine_modality({
            "model_type": "git",
            "architectures": ["GitForCausalLM"],
            "vision_config": {"image_size": 224},
            "vocab_size": 30522,
        }),
        module_registry.Modality.MULTIMODAL,
    )
    # Omni models with audio_config + text_config are MULTIMODAL
    self.assertEqual(
        module_registry._determine_modality({
            "model_type": "qwen2_5_omni_thinker",
            "architectures": ["Qwen2_5OmniThinkerForConditionalGeneration"],
            "audio_config": {"num_mel_bins": 128},
            "text_config": {"vocab_size": 152064},
        }),
        module_registry.Modality.MULTIMODAL,
    )
    # Gemma4 text models with audio_config and text_config but
    # vision_config=None are CAUSAL_LM.
    self.assertEqual(
        module_registry._determine_modality({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "audio_config": {"model_type": "gemma4_audio"},
            "text_config": {"model_type": "gemma4_text"},
            "vision_config": None,
        }),
        module_registry.Modality.CAUSAL_LM,
    )
    # MusicGen Melody CausalLM is AUDIO, not CAUSAL_LM
    self.assertEqual(
        module_registry._determine_modality({
            "model_type": "musicgen_melody_decoder",
            "architectures": ["MusicgenMelodyForCausalLM"],
        }),
        module_registry.Modality.AUDIO,
    )

  def test_sanitize_transformers_config_dict(self):
    raw_config = {
        "model_type": "layoutlmv3",
        "id2label": {"0": 1, "1": 2, "2": 3},
        "hidden_dropout_prob": 0,
        "attention_probs_dropout_prob": 0,
        "classifier_dropout": 0,
        "use_pretrained_backbone": True,
        "backbone": "resnet50",
        "rope_scaling": False,
    }
    sanitized = module_registry._sanitize_transformers_config_dict(
        raw_config, load_weights=False
    )
    self.assertNotIn("rope_scaling", sanitized)
    # id2label values must be str
    self.assertEqual(sanitized["id2label"], {"0": "1", "1": "2", "2": "3"})
    # dropout probs must be float
    self.assertIsInstance(sanitized["hidden_dropout_prob"], float)
    self.assertEqual(sanitized["hidden_dropout_prob"], 0.0)
    self.assertIsInstance(sanitized["attention_probs_dropout_prob"], float)
    self.assertIsInstance(sanitized["classifier_dropout"], float)
    # use_pretrained_backbone must be False when offline
    self.assertFalse(sanitized["use_pretrained_backbone"])
    # backbone_config must be synthesized when offline
    self.assertEqual(
        sanitized["backbone_config"],
        {"model_type": "timm_backbone", "backbone": "resnet50"},
    )

  def test_config_wrapper_dict(self):
    raw = {
        "model_type": "bert",
        "hidden_size": 768,
        "vision_config": {"image_size": 224, "num_channels": 3},
        "id2label": {"0": "cat"},
    }
    cfg = module_registry._ConfigWrapper(raw)

    # Attribute access
    self.assertEqual(cfg.model_type, "bert")
    self.assertEqual(cfg.hidden_size, 768)
    self.assertEqual(getattr(cfg, "model_type", ""), "bert")
    self.assertEqual(getattr(cfg, "nonexistent", "fallback"), "fallback")
    self.assertTrue(hasattr(cfg, "model_type"))
    self.assertFalse(hasattr(cfg, "nonexistent"))

    # Nested attribute access
    self.assertIsInstance(cfg.vision_config, module_registry._ConfigWrapper)
    self.assertEqual(cfg.vision_config.image_size, 224)
    self.assertEqual(cfg.vision_config.num_channels, 3)

    # Dictionary / Item access
    self.assertEqual(cfg["model_type"], "bert")
    self.assertEqual(cfg.get("hidden_size"), 768)
    self.assertEqual(cfg.get("hidden_size", ""), "768")
    self.assertEqual(cfg.get("nonexistent", 100), 100)
    self.assertEqual(cfg.get("nonexistent", ""), "")
    self.assertEqual(cfg.get("_class_name", "").lower(), "")
    self.assertEqual(cfg.get("architectures", []), [])
    self.assertIn("model_type", cfg)
    self.assertNotIn("nonexistent", cfg)

    # Mutation via attribute
    cfg.use_pretrained_backbone = False
    self.assertFalse(cfg.use_pretrained_backbone)
    self.assertFalse(raw["use_pretrained_backbone"])

    # Unwrap
    self.assertIs(cfg.unwrap(), raw)

    # Deepcopy and copy isolation
    cfg_deep = copy.deepcopy(cfg)
    self.assertIsInstance(cfg_deep, module_registry._ConfigWrapper)
    self.assertEqual(cfg_deep.model_type, "bert")
    self.assertIsNot(cfg_deep.unwrap(), raw)
    cfg_deep.hidden_size = 1024
    self.assertEqual(cfg.hidden_size, 768)
    self.assertEqual(cfg_deep.hidden_size, 1024)

  def test_config_wrapper_object(self):
    obj = types.SimpleNamespace(
        model_type="resnet",
        hidden_size=512,
        backbone_config={"image_size": 224},
    )
    cfg = module_registry._ConfigWrapper(obj)

    # Attribute delegation
    self.assertEqual(cfg.model_type, "resnet")
    self.assertEqual(cfg.hidden_size, 512)
    self.assertEqual(getattr(cfg, "nonexistent", "default"), "default")
    self.assertTrue(hasattr(cfg, "model_type"))
    self.assertFalse(hasattr(cfg, "nonexistent"))

    # Nested dict attribute delegation
    self.assertIsInstance(cfg.backbone_config, module_registry._ConfigWrapper)
    self.assertEqual(cfg.backbone_config.image_size, 224)

    # Item access delegation
    self.assertEqual(cfg["model_type"], "resnet")
    self.assertEqual(cfg.get("hidden_size"), 512)
    self.assertEqual(cfg.get("nonexistent", 99), 99)
    self.assertIn("model_type", cfg)
    self.assertNotIn("nonexistent", cfg)

    # Unwrap
    self.assertIs(cfg.unwrap(), obj)

  def test_load_and_sanitize_transformers_config_returns_wrapper(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      config_path = pathlib.Path(tmp_dir) / "config.json"
      config_path.write_text(
          json.dumps({
              "model_type": "bert",
              "hidden_size": 768,
              "vision_config": {"image_size": 224},
          })
      )
      cfg = module_registry._load_and_sanitize_transformers_config(
          config_path, load_weights=False
      )
      self.assertIsInstance(cfg, module_registry._ConfigWrapper)
      self.assertEqual(cfg.model_type, "bert")
      self.assertEqual(cfg.hidden_size, 768)
      self.assertEqual(cfg.vision_config.image_size, 224)

  def test_five_stage_transformers_pipeline(self):
    provider = module_registry.TransformersProvider(base_path="")
    with provider._resolve_transformers_config_path(
        "google/gemma-3-270m", load_weights=False
    ) as (cfg_path, model_dir):
      self.assertTrue(cfg_path.exists())
      self.assertIsNone(model_dir)
      cfg = module_registry._load_and_sanitize_transformers_config(
          cfg_path, load_weights=False
      )
    self.assertIsInstance(cfg, module_registry._ConfigWrapper)
    modality = module_registry._determine_modality(cfg)
    self.assertEqual(modality, module_registry.Modality.CAUSAL_LM)
    model_cls = module_registry._resolve_transformers_model_class(cfg, modality)
    self.assertIsNotNone(model_cls)
    with torch.device("meta"):
      model = module_registry._instantiate_transformers_model(
          model_cls, cfg, torch.bfloat16
      )
    self.assertIsInstance(model, torch.nn.Module)

  def test_transformers_config_caching_and_hook_isolation(self):
    module_cache.clear_config_cache()
    model_name = "google/gemma-3-270m"

    # First call loads config and caches it
    spec1 = self.module_registry.get_module_spec("transformers", model_name)
    orig_vocab = spec1.config.vocab_size

    # Mutating spec1.config directly must not contaminate the cache
    spec1.config.vocab_size = 777777

    # Second call with modify_config_hook should mutate only the returned config
    def modify_hook(cfg):
      cfg.vocab_size = 999999
      return cfg

    spec2 = self.module_registry.get_module_spec(
        "transformers", model_name, modify_config_hook=modify_hook
    )
    self.assertEqual(spec2.config.vocab_size, 999999)

    # Third call without hook should still see the original cached config
    spec3 = self.module_registry.get_module_spec("transformers", model_name)
    self.assertEqual(spec3.config.vocab_size, orig_vocab)

  def test_transformers_config_cache_avoids_repeated_load(self):
    module_cache.clear_config_cache()
    model_name = "google/gemma-3-270m"

    spec1 = self.module_registry.get_module_spec("transformers", model_name)
    self.assertIsNotNone(spec1.config)

    # Mock _load_and_sanitize_transformers_config to verify cache hit avoids I/O
    with mock.patch.object(
        module_registry, "_load_and_sanitize_transformers_config"
    ) as mock_load:
      spec2 = self.module_registry.get_module_spec("transformers", model_name)
      mock_load.assert_not_called()
      self.assertEqual(spec1.config.vocab_size, spec2.config.vocab_size)

  def test_diffusers_config_caching_and_hook_isolation(self):
    module_cache.clear_config_cache()
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"

    spec1 = self.module_registry.get_module_spec(
        "diffusers", model_name, subfolder="unet"
    )
    orig_sample_size = spec1.config.get("sample_size")

    # Mutating spec1.config directly must not contaminate the cache
    spec1.config["sample_size"] = 77777

    def modify_hook(cfg):
      cfg["sample_size"] = 12345
      return cfg

    spec2 = self.module_registry.get_module_spec(
        "diffusers",
        model_name,
        subfolder="unet",
        modify_config_hook=modify_hook,
    )
    self.assertEqual(spec2.config.get("sample_size"), 12345)

    spec3 = self.module_registry.get_module_spec(
        "diffusers", model_name, subfolder="unet"
    )
    self.assertEqual(spec3.config.get("sample_size"), orig_sample_size)

  def test_diffusers_config_cache_avoids_repeated_load(self):
    module_cache.clear_config_cache()
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"

    spec1 = self.module_registry.get_module_spec(
        "diffusers", model_name, subfolder="unet"
    )
    self.assertIsNotNone(spec1.config)

    with mock.patch.object(
        module_registry.auto_model.AutoModel, "load_config"
    ) as mock_load:
      spec2 = self.module_registry.get_module_spec(
          "diffusers", model_name, subfolder="unet"
      )
      mock_load.assert_not_called()
      self.assertEqual(
          spec1.config.get("sample_size"), spec2.config.get("sample_size")
      )

  def test_weights_are_not_cached(self):
    module_cache.clear_config_cache()
    model_name = "google/gemma-3-270m"

    spec = self.module_registry.get_module_spec("transformers", model_name)

    # Verify no torch.Tensor or torch.nn.Module stored in config cache
    for entry in module_cache.CONFIG_CACHE.values():
      self.assertNotIsInstance(entry.config, torch.nn.Module)
      self.assertNotIsInstance(entry.config, torch.Tensor)

    # Calling module_factory() must create distinct model instances
    m1 = spec.module_factory()
    m2 = spec.module_factory()
    self.assertIsNot(m1, m2)

  def test_clear_config_cache(self):
    self.module_registry.get_module_spec("transformers", "google/gemma-3-270m")
    self.assertNotEmpty(module_cache.CONFIG_CACHE)

    module_cache.clear_config_cache()
    self.assertEmpty(module_cache.CONFIG_CACHE)

  def test_concurrent_cache_access(self):
    module_cache.clear_config_cache()
    model_name = "google/gemma-3-270m"
    num_threads = 10
    results: list[module_registry.ModuleSpec | None] = [None] * num_threads
    errors: list[Exception] = []

    original_load = module_registry._load_and_sanitize_transformers_config
    call_count = [0]
    count_lock = threading.Lock()

    def instrumented_load(*args, **kwargs):
      with count_lock:
        call_count[0] += 1
      time.sleep(0.01)  # Increase likelihood of thread race
      return original_load(*args, **kwargs)

    with mock.patch.object(
        module_registry,
        "_load_and_sanitize_transformers_config",
        side_effect=instrumented_load,
    ):

      def worker(idx):
        try:
          results[idx] = self.module_registry.get_module_spec(
              "transformers", model_name
          )
        except Exception as exc:  # pylint: disable=broad-except
          errors.append(exc)

      threads = [
          threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

    self.assertEmpty(errors)
    # The per-key lock ensures from_pretrained was called exactly once
    self.assertEqual(call_count[0], 1)
    self.assertEqual(len(results), num_threads)
    for res in results:
      self.assertIsNotNone(res)
      self.assertIsNotNone(res.config)
      self.assertEqual(res.config.vocab_size, results[0].config.vocab_size)

    # Mutating one thread's config must not affect another thread's config
    results[0].config.vocab_size = 123456
    self.assertNotEqual(
        results[0].config.vocab_size, results[1].config.vocab_size
    )

  def test_diffusers_concurrent_cache_access(self):
    module_cache.clear_config_cache()
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    num_threads = 10
    results: list[module_registry.ModuleSpec | None] = [None] * num_threads
    errors: list[Exception] = []

    original_load_config = module_registry.auto_model.AutoModel.load_config
    call_count = [0]
    count_lock = threading.Lock()

    def instrumented_load(*args, **kwargs):
      with count_lock:
        call_count[0] += 1
      time.sleep(0.01)
      return original_load_config(*args, **kwargs)

    with mock.patch.object(
        module_registry.auto_model.AutoModel,
        "load_config",
        side_effect=instrumented_load,
    ):

      def worker(idx):
        try:
          results[idx] = self.module_registry.get_module_spec(
              "diffusers", model_name, subfolder="unet"
          )
        except Exception as exc:  # pylint: disable=broad-except
          errors.append(exc)

      threads = [
          threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

    self.assertEmpty(errors)
    self.assertEqual(call_count[0], 1)
    self.assertEqual(len(results), num_threads)
    for res in results:
      self.assertIsNotNone(res)
      self.assertIsNotNone(res.config)
      self.assertEqual(
          res.config.get("sample_size"), results[0].config.get("sample_size")
      )

    results[0].config["sample_size"] = 99999
    self.assertNotEqual(
        results[0].config.get("sample_size"),
        results[1].config.get("sample_size"),
    )

  def test_transformers_config_cache_load_weights_true_then_false_backbone(
      self,
  ):
    model_name = "facebook/detr-resnet-50"

    spec_pretrained = self.module_registry.get_module_spec(
        "transformers", model_name, load_weights=True
    )
    self.assertIsNotNone(spec_pretrained.config)

    spec_random = self.module_registry.get_module_spec(
        "transformers", model_name, load_weights=False
    )
    self.assertFalse(
        getattr(spec_random.config, "use_pretrained_backbone", False)
    )
    self.assertIsNotNone(spec_random.config.get("backbone_config"))
    model = spec_random.module_factory()
    self.assertIsNotNone(model)

  def test_transformers_preprocessor_cache_and_text_modality_priority(self):
    module_cache.clear_config_cache()
    model_name = "google/gemma-3-270m"

    fake_tokenizer = mock.MagicMock(name="FakeTokenizer")
    fake_processor = mock.MagicMock(name="FakeProcessor")
    with (
        mock.patch.object(
            module_registry.transformers.AutoTokenizer,
            "from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok_load,
        mock.patch.object(
            module_registry.transformers.AutoProcessor,
            "from_pretrained",
            return_value=fake_processor,
        ) as mock_proc_load,
    ):
      spec1 = self.module_registry.get_module_spec("transformers", model_name)
      spec2 = self.module_registry.get_module_spec("transformers", model_name)

      p1 = spec1.preprocessor_factory()
      p2 = spec1.preprocessor_factory()
      p3 = spec2.preprocessor_factory()

      self.assertIs(p1, fake_tokenizer)
      self.assertIs(p2, fake_tokenizer)
      self.assertIs(p3, fake_tokenizer)
      mock_tok_load.assert_called_once()
      mock_proc_load.assert_not_called()

      # Text modalities (CAUSAL_LM, TEXT, SEQ2SEQ) share the same cache entry
      for text_mod in (
          module_registry.Modality.CAUSAL_LM,
          module_registry.Modality.TEXT,
          module_registry.Modality.SEQ2SEQ,
      ):
        loaded = module_registry._load_transformers_preprocessor(
            "/tmp/fake_text_model", modality=text_mod
        )
        self.assertIs(loaded, fake_tokenizer)
      # 1 call for gemma-3-270m + 1 call for /tmp/fake_text_model across all 3
      # text modalities
      self.assertEqual(mock_tok_load.call_count, 2)

      # Non-text modalities (VISION, MULTIMODAL, None) share the same cache
      # entry
      for non_text_mod in (
          module_registry.Modality.VISION,
          module_registry.Modality.MULTIMODAL,
          None,
      ):
        loaded_proc = module_registry._load_transformers_preprocessor(
            "/tmp/fake_vlm_model", modality=non_text_mod
        )
        self.assertIs(loaded_proc, fake_processor)
      mock_proc_load.assert_called_once()

      # Vision input generation reuses the cached processor via
      # preprocessor_loader
      fake_processor.size = {"height": 384}
      vision_cfg = mock.MagicMock(
          model_type="vit",
          architectures=["ViTModel"],
          image_size=224,
          num_channels=3,
      )
      vision_inputs = transformers_inputs.generate_transformers_inputs(
          vision_cfg,
          module_registry.Modality.VISION,
          model_dir="/tmp/fake_vlm_model",
          preprocessor_loader=lambda d: (
              module_registry._load_transformers_preprocessor(
                  d, modality=module_registry.Modality.VISION
              )
          ),
      )
      self.assertEqual(vision_inputs["pixel_values"].shape, (1, 3, 384, 384))
      mock_proc_load.assert_called_once()

    self.assertNotEmpty(module_cache.PREPROCESSOR_CACHE)
    module_cache.clear_preprocessor_cache()
    self.assertEmpty(module_cache.PREPROCESSOR_CACHE)

  def test_transformers_preprocessor_deterministic_file_selection(self):
    module_cache.clear_config_cache()
    fake_processor = mock.MagicMock(name="FakeProcessor")
    with tempfile.TemporaryDirectory() as tmp_dir:
      (pathlib.Path(tmp_dir) / "preprocessor_config.json").write_text("{}")
      with (
          mock.patch.object(
              module_registry.transformers.AutoTokenizer,
              "from_pretrained",
          ) as mock_tok_load,
          mock.patch.object(
              module_registry.transformers.AutoProcessor,
              "from_pretrained",
              return_value=fake_processor,
          ) as mock_proc_load,
      ):
        loaded = module_registry._load_transformers_preprocessor(
            tmp_dir, modality=module_registry.Modality.CAUSAL_LM
        )
        self.assertIs(loaded, fake_processor)
        mock_tok_load.assert_not_called()
        mock_proc_load.assert_called_once()

  def test_concurrent_preprocessor_cache_access(self):
    module_cache.clear_config_cache()
    model_name = "google/gemma-3-270m"
    spec = self.module_registry.get_module_spec("transformers", model_name)
    module_cache.clear_preprocessor_cache()

    num_threads = 10
    results: list[object] = [None] * num_threads
    errors: list[Exception] = []
    call_count = [0]
    count_lock = threading.Lock()
    fake_tokenizer = mock.MagicMock(name="ConcurrentTokenizer")

    def instrumented_load(*unused_args, **unused_kwargs):
      with count_lock:
        call_count[0] += 1
      time.sleep(0.01)
      return fake_tokenizer

    with mock.patch.object(
        module_registry.transformers.AutoTokenizer,
        "from_pretrained",
        side_effect=instrumented_load,
    ):

      def worker(idx):
        try:
          local_spec = self.module_registry.get_module_spec(
              "transformers", model_name
          )
          results[idx] = local_spec.preprocessor_factory()
        except Exception as exc:  # pylint: disable=broad-except
          errors.append(exc)

      threads = [
          threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

    self.assertEmpty(errors)
    self.assertEqual(call_count[0], 1)
    for res in results:
      self.assertIs(res, fake_tokenizer)
    self.assertIs(spec.preprocessor_factory(), fake_tokenizer)

  def test_resolve_diffusers_model_class(self):
    cls_name, model_cls = module_registry._resolve_diffusers_model_class(
        "stabilityai/stable-diffusion-xl-base-1.0",
        {"_class_name": "UNet2DConditionModel"},
    )
    self.assertEqual(cls_name, "UNet2DConditionModel")
    self.assertIs(model_cls, module_registry.diffusers.UNet2DConditionModel)

    cls_name_tr, model_cls_tr = module_registry._resolve_diffusers_model_class(
        "nvidia/cosmos-transformer", {}
    )
    self.assertEqual(cls_name_tr, "Transformer2DModel")
    self.assertIs(model_cls_tr, module_registry.diffusers.Transformer2DModel)

    cls_name_remap, model_cls_remap = (
        module_registry._resolve_diffusers_model_class(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            {
                "_class_name": "Transformer2DModel",
                "norm_type": "ada_norm_single",
            },
        )
    )
    self.assertEqual(cls_name_remap, "PixArtTransformer2DModel")
    self.assertIs(
        model_cls_remap, module_registry.diffusers.PixArtTransformer2DModel
    )

  def test_transformers_composite_model_constructed_in_target_dtype(self):
    # Qwen3.5-397B-A17B's root config has no dtype while its text_config is
    # bfloat16. The model must be constructed in bfloat16 rather than in float32
    # and cast afterwards, which doubles peak memory during initialization.
    def modify_config(config):
      text_config = config.get_text_config()
      text_config.num_hidden_layers = 1
      text_config.layer_types = text_config.layer_types[:1]
      return config

    module_spec = self.module_registry.get_module_spec(
        "transformers",
        "Qwen/Qwen3.5-397B-A17B",
        load_weights=False,
        modify_config_hook=modify_config,
    )
    self.assertIsNone(module_spec.config.get("dtype"))
    self.assertEqual(module_spec.config.get_text_config().dtype, torch.bfloat16)

    # Records the dtype of each parameter as it is registered, so that a cast
    # after construction does not hide a float32 construction.
    registered_dtypes = set()

    def record_dtype(unused_module, unused_name, param):
      if param is not None and param.is_floating_point():
        registered_dtypes.add(param.dtype)

    handle = (
        torch.nn.modules.module.register_module_parameter_registration_hook(
            record_dtype
        )
    )
    self.addCleanup(handle.remove)
    with torch.device("meta"):
      model = module_spec.module_factory()
    handle.remove()

    self.assertEqual(registered_dtypes, {torch.bfloat16})
    for param in model.parameters():
      self.assertEqual(param.dtype, torch.bfloat16)

  def test_modernbert_dict_default_theta_and_custom_global_local_rope_theta(
      self,
  ):
    raw_cfg = {
        "model_type": "modernbert",
        "architectures": ["ModernBertModel"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 256,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "cls_token_id": 3,
        "sep_token_id": 4,
    }
    cfg = module_registry._instantiate_transformers_config_class(
        module_registry.transformers.ModernBertConfig, raw_cfg
    )
    module_registry._sanitize_config(cfg)

    self.assertIsInstance(cfg.rope_parameters, dict)
    self.assertIn("full_attention", cfg.rope_parameters)
    self.assertIn("sliding_attention", cfg.rope_parameters)
    self.assertNotIn("rope_theta", cfg.rope_parameters)
    self.assertNotIn("rope_type", cfg.rope_parameters)
    self.assertEqual(
        cfg.rope_parameters["full_attention"]["rope_theta"], 160000.0
    )
    self.assertEqual(
        cfg.rope_parameters["sliding_attention"]["rope_theta"], 10000.0
    )

    model = module_registry.transformers.ModernBertModel(cfg)
    self.assertEqual(
        model.config.rope_parameters["full_attention"]["rope_theta"], 160000.0
    )
    self.assertEqual(
        model.config.rope_parameters["sliding_attention"]["rope_theta"], 10000.0
    )

    custom_theta_cfg = {
        **raw_cfg,
        "global_rope_theta": 250000.0,
        "local_rope_theta": 20000.0,
    }
    cfg_custom = module_registry._instantiate_transformers_config_class(
        module_registry.transformers.ModernBertConfig, custom_theta_cfg
    )
    module_registry._sanitize_config(cfg_custom)
    self.assertEqual(
        cfg_custom.rope_parameters["full_attention"]["rope_theta"], 250000.0
    )
    self.assertEqual(
        cfg_custom.rope_parameters["sliding_attention"]["rope_theta"], 20000.0
    )

  def test_nested_rope_parameters_with_scalar_top_level_rope_theta(self):
    gemma3_cfg = module_registry._instantiate_transformers_config_class(
        module_registry.transformers.Gemma3TextConfig,
        {
            "model_type": "gemma3_text",
            "rope_theta": 500000.0,
            "rope_parameters": {
                "full_attention": {"type": "linear", "factor": 2.0},
                "sliding_attention": {"type": "default"},
            },
        },
    )
    module_registry._sanitize_config(gemma3_cfg)
    self.assertNotIn("rope_theta", gemma3_cfg.rope_parameters)
    self.assertEqual(
        gemma3_cfg.rope_parameters["full_attention"]["rope_type"], "linear"
    )
    self.assertEqual(
        gemma3_cfg.rope_parameters["full_attention"]["rope_theta"], 500000.0
    )
    self.assertEqual(
        gemma3_cfg.rope_parameters["sliding_attention"]["rope_theta"], 10000.0
    )

  def test_hybrid_rope_configs_with_empty_rope_parameters(self):
    for cfg_cls, model_type, expected_global_theta in (
        (module_registry.transformers.ModernBertConfig, "modernbert", 160000.0),
        (
            module_registry.transformers.Gemma3TextConfig,
            "gemma3_text",
            1000000.0,
        ),
    ):
      with self.subTest(model_type=model_type):
        cfg = module_registry._instantiate_transformers_config_class(
            cfg_cls,
            {"model_type": model_type, "rope_parameters": {}},
        )
        module_registry._sanitize_config(cfg)
        self.assertNotIn("rope_theta", cfg.rope_parameters)
        self.assertNotIn("rope_type", cfg.rope_parameters)
        self.assertEqual(
            cfg.rope_parameters["full_attention"]["rope_theta"],
            expected_global_theta,
        )
        self.assertEqual(
            cfg.rope_parameters["sliding_attention"]["rope_theta"], 10000.0
        )

  def test_hybrid_rope_configs_with_flat_rope_scaling_and_null_rope_theta(self):
    for model_type, expected_global_theta, expected_sliding_theta in (
        ("gemma3_text", 1000000.0, 10000.0),
        ("olmo3", 500000.0, 500000.0),
    ):
      with self.subTest(model_type=model_type):
        scaling_raw_cfg = {
            "model_type": model_type,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "rope_theta": None,
            "rope_scaling": {"type": "linear", "factor": 8.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
          cfg_path = pathlib.Path(tmpdir) / "config.json"
          cfg_path.write_text(json.dumps(scaling_raw_cfg))
          cfg = module_registry._load_and_sanitize_transformers_config(cfg_path)
        rope_params = cfg.rope_parameters.unwrap()
        self.assertIsInstance(rope_params, dict)
        self.assertNotIn("rope_theta", rope_params)
        self.assertIn("full_attention", rope_params)
        self.assertIn("sliding_attention", rope_params)
        self.assertEqual(rope_params["full_attention"]["rope_type"], "linear")
        self.assertEqual(rope_params["full_attention"]["factor"], 8.0)
        self.assertEqual(
            rope_params["full_attention"]["rope_theta"],
            expected_global_theta,
        )
        self.assertEqual(
            rope_params["sliding_attention"]["rope_type"], "default"
        )
        self.assertEqual(
            rope_params["sliding_attention"]["rope_theta"],
            expected_sliding_theta,
        )

  def test_nested_rope_parameters_round_trip_and_wrapper_sanitize(self):
    round_trip_cfg = module_registry._instantiate_transformers_config_class(
        module_registry.transformers.ModernBertConfig,
        {
            "model_type": "modernbert",
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 200000.0,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 15000.0,
                },
            },
        },
    )
    module_registry._sanitize_config(round_trip_cfg)
    self.assertNotIn("rope_theta", round_trip_cfg.rope_parameters)
    self.assertEqual(
        round_trip_cfg.rope_parameters["full_attention"]["rope_theta"], 200000.0
    )
    self.assertEqual(
        round_trip_cfg.rope_parameters["sliding_attention"]["rope_theta"],
        15000.0,
    )

    nested_wrapper = module_registry._ConfigWrapper({
        "rope_parameters": {
            "full_attention": {"type": "linear", "rope_theta": 160000.0},
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": None,
            },
        },
    })
    module_registry._sanitize_rope_parameters(nested_wrapper)
    nested_params = nested_wrapper.unwrap()["rope_parameters"]
    self.assertNotIn("rope_theta", nested_params)
    self.assertEqual(nested_params["full_attention"]["rope_type"], "linear")
    self.assertEqual(nested_params["full_attention"]["rope_theta"], 160000.0)
    self.assertEqual(nested_params["sliding_attention"]["rope_type"], "default")
    self.assertNotIn("rope_theta", nested_params["sliding_attention"])

  def test_nested_rope_scaling_promoted_to_rope_parameters(self):
    nested_scaling_cfg = module_registry._instantiate_transformers_config_class(
        module_registry.transformers.ModernBertConfig,
        {
            "model_type": "modernbert",
            "rope_scaling": {
                "full_attention": {"type": "linear", "factor": 2.0},
                "sliding_attention": {"type": "default"},
            },
        },
    )
    module_registry._sanitize_config(nested_scaling_cfg)
    self.assertEqual(
        nested_scaling_cfg.rope_parameters["full_attention"]["rope_type"],
        "linear",
    )
    self.assertEqual(
        nested_scaling_cfg.rope_parameters["full_attention"]["rope_theta"],
        160000.0,
    )
    self.assertEqual(
        nested_scaling_cfg.rope_parameters["sliding_attention"]["rope_theta"],
        10000.0,
    )

  def test_non_hybrid_rope_scaling_and_custom_converter_configs(self):
    llama_cfg = module_registry._instantiate_transformers_config_class(
        module_registry.transformers.LlamaConfig,
        {
            "model_type": "llama",
            "rope_theta": 500000.0,
            "rope_scaling": {"type": "linear", "factor": 2.0},
        },
    )
    module_registry._sanitize_config(llama_cfg)
    self.assertEqual(llama_cfg.rope_parameters["rope_type"], "linear")
    self.assertEqual(llama_cfg.rope_parameters["factor"], 2.0)
    self.assertEqual(llama_cfg.rope_parameters["rope_theta"], 500000.0)

    empty_scaling_cfg = module_registry._instantiate_transformers_config_class(
        module_registry.transformers.LlamaConfig,
        {
            "model_type": "llama",
            "rope_scaling": {},
            "rope_parameters": {
                "rope_type": "linear",
                "factor": 4.0,
                "rope_theta": None,
            },
        },
    )
    module_registry._sanitize_config(empty_scaling_cfg)
    self.assertEqual(empty_scaling_cfg.rope_parameters["rope_type"], "linear")
    self.assertEqual(empty_scaling_cfg.rope_parameters["factor"], 4.0)
    self.assertEqual(
        empty_scaling_cfg.rope_parameters["rope_theta"],
        empty_scaling_cfg.default_theta,
    )

    neox_cfg = module_registry._instantiate_transformers_config_class(
        module_registry.transformers.GPTNeoXConfig,
        {
            "model_type": "gpt_neox",
            "rotary_emb_base": 25000,
        },
    )
    module_registry._sanitize_config(neox_cfg)
    self.assertEqual(neox_cfg.rope_parameters["rope_theta"], 25000)

  def test_gemma4_vision_inputs_and_qwen_omni_audio_inputs(self):
    gemma4_vis_cfg = {
        "model_type": "gemma4_vision",
        "architectures": ["Gemma4VisionModel"],
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "num_channels": 3,
    }
    vis_inputs = transformers_inputs.generate_transformers_inputs(
        gemma4_vis_cfg,
        module_registry.Modality.VISION,
        shape=(2, 3, 224, 224),
    )
    self.assertEqual(vis_inputs["pixel_values"].shape, (2, 225, 3 * 16 * 16))
    self.assertEqual(vis_inputs["pixel_position_ids"].shape, (2, 225, 2))

    qwen_audio_cfg = {
        "model_type": "qwen3_omni_moe_audio_encoder",
        "architectures": ["Qwen3OmniMoeAudioEncoder"],
        "num_mel_bins": 128,
    }
    audio_inputs = transformers_inputs.generate_transformers_inputs(
        qwen_audio_cfg,
        module_registry.Modality.AUDIO,
        shape=(2, 128),
    )
    self.assertEqual(audio_inputs["input_features"].shape, (128, 256))
    self.assertEqual(audio_inputs["feature_lens"].tolist(), [128, 128])

  def test_xlstm_and_musicgen_melody_decoder_inputs(self):
    xlstm_cfg = {
        "model_type": "xlstm",
        "architectures": ["xLSTMForCausalLM"],
        "vocab_size": 1000,
        "chunk_size": 64,
    }
    xlstm_inputs = transformers_inputs.generate_transformers_inputs(
        xlstm_cfg,
        module_registry.Modality.TEXT,
        shape=(1, 16),
    )
    self.assertEqual(xlstm_inputs["input_ids"].shape, (1, 64))
    self.assertNotIn("attention_mask", xlstm_inputs)

    melody_dec_cfg = {
        "model_type": "musicgen_melody_decoder",
        "architectures": ["MusicgenMelodyForCausalLM"],
        "vocab_size": 2048,
        "num_codebooks": 4,
    }
    melody_inputs = transformers_inputs.generate_transformers_inputs(
        melody_dec_cfg,
        module_registry.Modality.AUDIO,
        shape=(2, 16),
    )
    self.assertEqual(melody_inputs["input_ids"].shape, (8, 16))
    self.assertNotIn("attention_mask", melody_inputs)
    self.assertNotIn("decoder_input_ids", melody_inputs)

  def test_sanitize_xlstm_and_qwen_omni_configs(self):
    xlstm_dict = {"model_type": "xlstm"}
    module_registry._sanitize_model_specific_config_dict(
        xlstm_dict, load_weights=False
    )
    self.assertEqual(
        xlstm_dict["chunkwise_kernel"], "chunkwise--native_autograd"
    )
    self.assertEqual(xlstm_dict["sequence_kernel"], "native_sequence__native")
    self.assertEqual(xlstm_dict["step_kernel"], "native")

    qwen_omni_dict = {"model_type": "qwen3_omni_moe"}
    module_registry._sanitize_model_specific_config_dict(
        qwen_omni_dict, load_weights=False
    )
    self.assertIs(qwen_omni_dict["enable_audio_output"], False)


if __name__ == "__main__":
  absltest.main()
