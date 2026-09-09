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

import json
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from etils import epath
from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage
from PIL import Image
import torch
from torch_tpu._internal import testing as tt_testing
from tests import module_registry
from tests import seed_test_utils

_GOLDFISH_IMG_PATH = (
    epath.resource_path("torch_tpu") / "tests/compile/data/goldfish.jpg"
)


class ModuleRegistryTest(seed_test_utils.RepeatableTest):

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

      def fake_download(filename):
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
          "torchvision", "resnet50"
      )
      model = module_spec.module_factory()
      self.assertIsNotNone(model)
      mock_get_model.assert_called_once_with(
          "resnet50", weights=None, weights_backbone=None
      )

  def test_torchvision_weights_backbone_type_error_fallback(self):
    def fake_get_model(name, weights=None, **kwargs):
      del name, weights
      if "weights_backbone" in kwargs:
        raise TypeError("unexpected keyword argument 'weights_backbone'")
      return mock.MagicMock()

    with mock.patch(
        "torchvision.models.get_model", side_effect=fake_get_model
    ) as mock_get_model:
      module_spec = self.module_registry.get_module_spec(
          "torchvision", "simple_model"
      )
      model = module_spec.module_factory()
      self.assertIsNotNone(model)
      self.assertEqual(mock_get_model.call_count, 2)

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
      inputs = module_registry._generate_transformers_inputs(
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
      inputs = module_registry._generate_transformers_inputs(
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

  def test_transformers_timeseries_tensor_generation(self):
    ts_models = [
        "autoformer",
        "informer",
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
      inputs = module_registry._generate_transformers_inputs(
          mock_cfg, module_registry.Modality.CAUSAL_LM, shape=(2, 32)
      )
      self.assertNotIn("input_ids", inputs, f"Failed for {ts}")
      self.assertNotIn("attention_mask", inputs, f"Failed for {ts}")
      self.assertIn("past_values", inputs, f"Failed for {ts}")
      if ts == "timesfm":
        self.assertIn("freq", inputs)
        self.assertEqual(inputs["past_values"].shape, (2, 32))
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
    inputs_siglip2 = module_registry._generate_transformers_inputs(
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
    inputs_oneformer = module_registry._generate_transformers_inputs(
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
    inputs_vitmatte = module_registry._generate_transformers_inputs(
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
    inputs_vitpose = module_registry._generate_transformers_inputs(
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
    inputs_bros = module_registry._generate_transformers_inputs(
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
    inputs_p2s = module_registry._generate_transformers_inputs(
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
    inputs_qwenvl = module_registry._generate_transformers_inputs(
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
    inputs_iblip = module_registry._generate_transformers_inputs(
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
    inputs_lxmert = module_registry._generate_transformers_inputs(
        cfg_lxmert, module_registry.Modality.CAUSAL_LM, shape=(1, 32)
    )
    self.assertIn("visual_feats", inputs_lxmert)
    self.assertIn("visual_pos", inputs_lxmert)
    self.assertEqual(inputs_lxmert["visual_feats"].shape, (1, 36, 2048))
    self.assertEqual(inputs_lxmert["visual_pos"].shape, (1, 36, 4))

  def test_transformers_get_max_seq_len_bounds(self):
    cfg_normal = mock.MagicMock(max_position_embeddings=2048)
    self.assertEqual(module_registry._get_max_seq_len(cfg_normal), 2048)

    cfg_neg = mock.MagicMock(max_position_embeddings=-1)
    self.assertEqual(module_registry._get_max_seq_len(cfg_neg), 512)

    cfg_oversized = mock.MagicMock(max_position_embeddings=100000)
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

  def test_get_config_attr(self):
    cfg_dict = {"foo": 1, "bar": "val"}
    self.assertEqual(module_registry._get_config_attr(cfg_dict, "foo"), 1)
    self.assertEqual(module_registry._get_config_attr(cfg_dict, "bar"), "val")
    self.assertIsNone(module_registry._get_config_attr(cfg_dict, "baz"))
    self.assertEqual(
        module_registry._get_config_attr(cfg_dict, "baz", default=42), 42
    )

    cfg_obj = mock.MagicMock(foo=2, bar="obj_val")
    self.assertEqual(module_registry._get_config_attr(cfg_obj, "foo"), 2)
    self.assertEqual(
        module_registry._get_config_attr(cfg_obj, "bar"), "obj_val"
    )
    self.assertIsNone(module_registry._get_config_attr(None, "foo"))

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


if __name__ == "__main__":
  absltest.main()
