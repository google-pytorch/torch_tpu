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
from torch_tpu.csrc.internal import testing as tt_testing
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


if __name__ == "__main__":
  absltest.main()
