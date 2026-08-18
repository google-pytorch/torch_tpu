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

"""End-to-end TPU tests for Qwen-Image TorchTPU prototype modules and workflows."""

import json
import os
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
from PIL import Image
from safetensors.torch import load_file
import torch
import torch_tpu
from prototypes.qwen_image import config
from prototypes.qwen_image import infer_qwen20b_tpu
from prototypes.qwen_image import tpu_sharding
from prototypes.qwen_image import train_torch_tpu
from prototypes.qwen_image import utils


def _create_tiny_qwen_image_pipeline(save_dir: str):
  """Builds and saves a miniature real QwenImagePipeline for TPU testing."""
  import diffusers
  import transformers

  torch.manual_seed(0)
  transformer = diffusers.QwenImageTransformer2DModel(
      patch_size=2,
      in_channels=64,
      out_channels=16,
      num_layers=2,
      attention_head_dim=16,
      num_attention_heads=2,
      joint_attention_dim=32,
      axes_dims_rope=(4, 4, 8),
  )

  z_dim = 16
  vae = diffusers.AutoencoderKLQwenImage(
      base_dim=z_dim * 2,
      z_dim=z_dim,
      dim_mult=[1, 2],
      num_res_blocks=1,
      temperal_downsample=[False],
      latents_mean=[0.0] * z_dim,
      latents_std=[1.0] * z_dim,
  )

  scheduler = diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps=20)

  text_Cfg = transformers.Qwen2_5_VLConfig(
      text_config={
          "bos_token_id": 0,
          "eos_token_id": 1,
          "hidden_size": 32,
          "intermediate_size": 64,
          "max_position_embeddings": 512,
          "max_window_layers": 2,
          "num_attention_heads": 2,
          "num_hidden_layers": 2,
          "num_key_value_heads": 2,
          "vocab_size": 1000,
          "rope_parameters": {
              "mrope_section": [2, 2, 4],
              "rope_type": "default",
              "rope_theta": 10000.0,
          },
      },
      vision_config={
          "depth": 2,
          "embed_dim": 16,
          "hidden_size": 32,
          "mlp_ratio": 2,
          "num_heads": 2,
          "in_channels": 3,
          "out_hidden_size": 32,
          "patch_size": 14,
          "spatial_merge_size": 2,
          "spatial_patch_size": 14,
          "temporal_patch_size": 2,
      },
      hidden_size=32,
      vocab_size=1000,
      vision_start_token_id=2,
      vision_end_token_id=3,
      vision_token_id=4,
      image_token_id=5,
      video_token_id=6,
  )
  text_encoder = transformers.Qwen2_5_VLForConditionalGeneration(text_Cfg)

  tok_dir = os.path.join(save_dir, "_tokenizer_init")
  os.makedirs(tok_dir, exist_ok=True)
  vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
  bs = (
      list(range(ord("!"), ord("~") + 1))
      + list(range(ord("¡"), ord("¬") + 1))
      + list(range(ord("®"), ord("ÿ") + 1))
  )
  cs = bs[:]
  n = 0
  for b in range(256):
    if b not in bs:
      bs.append(b)
      cs.append(256 + n)
      n += 1
  for ch in (chr(c) for c in cs):
    if ch not in vocab:
      vocab[ch] = len(vocab)
  for i in range(len(vocab), 1000):
    vocab[f"tok_{i}"] = i
  vocab_path = os.path.join(tok_dir, "vocab.json")
  merges_path = os.path.join(tok_dir, "merges.txt")
  with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f)
  with open(merges_path, "w", encoding="utf-8") as f:
    f.write("#version: 0.2\n")

  import inspect

  tok_params = inspect.signature(
      transformers.Qwen2Tokenizer.__init__
  ).parameters
  if "vocab" in tok_params:
    tokenizer = transformers.Qwen2Tokenizer(
        vocab=vocab,
        merges=[],
        unk_token="<|endoftext|>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
  else:
    tokenizer = transformers.Qwen2Tokenizer(
        vocab_file=vocab_path,
        merges_file=merges_path,
        unk_token="<|endoftext|>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )

  pipe = diffusers.QwenImagePipeline(
      scheduler=scheduler,
      vae=vae,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      transformer=transformer,
  )
  pipe.save_pretrained(save_dir)
  return pipe


class QwenImagePrototypeTest(absltest.TestCase):  # ABSLTEST_OK=prototype test

  def setUp(self):
    super().setUp()
    self.assertIsNotNone(torch_tpu)
    self.assertTrue(
        hasattr(torch, "tpu") and torch.tpu.is_available(),
        "Expected TPU device to be available for TPU test.",
    )
    self.assertGreaterEqual(torch.tpu.device_count(), 1)
    torch.manual_seed(42)

  def test_configs_and_tpu_sharding_on_tpu(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "tiny_qwen_model")
      pipe = _create_tiny_qwen_image_pipeline(model_dir)
      pipe.transformer.to(dtype=torch.bfloat16)

      tpu_cfg = config.TPUConfig(
          sharding_strategy="layer",
          num_devices=torch.tpu.device_count(),
          tpu_accelerator_type="v5litepod-1",
      )
      model_cfg = config.ModelConfig(model_id=model_dir)
      gen_cfg = config.GenerationConfig(
          num_inference_steps=2, height=32, width=32
      )
      self.assertEqual(model_cfg.model_id, model_dir)
      self.assertEqual(gen_cfg.height, 32)

      manager = tpu_sharding.TPUShardingManager(tpu_cfg)
      self.assertEqual(manager.primary_device.type, "tpu")
      sharded = manager.shard_model(pipe)
      self.assertIsNotNone(sharded)

      # Verify transformer blocks reside on TPU and execute a forward pass on TPU
      block_param_dev = next(
          pipe.transformer.transformer_blocks[0].parameters()
      ).device
      self.assertEqual(block_param_dev.type, "tpu")

      tpu_dev = manager.primary_device
      hidden_states = torch.randn(
          1, 16, 64, device=tpu_dev, dtype=torch.bfloat16
      )
      encoder_hidden_states = torch.randn(
          1, 8, 32, device=tpu_dev, dtype=torch.bfloat16
      )
      timestep = torch.tensor([0.5], device=tpu_dev, dtype=torch.bfloat16)
      with torch.no_grad():
        out = pipe.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            timestep=timestep,
            img_shapes=[[(1, 4, 4)]],
            return_dict=False,
        )[0]
      tpu_sharding.mark_step_tpu()
      self.assertEqual(out.device.type, "tpu")
      self.assertEqual(tuple(out.shape), (1, 16, 64))

      manager_spmd = tpu_sharding.TPUShardingManager(
          config.TPUConfig(sharding_strategy="spmd", num_devices=1)
      )
      self.assertIsNotNone(manager_spmd.shard_model(pipe))
      tpu_sharding.mark_step_tpu()

  def test_dataset_loading_utilities(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      img_path = os.path.join(tmpdir, "sample.png")
      Image.new("RGB", (32, 32), color=(255, 0, 0)).save(img_path)
      with open(os.path.join(tmpdir, "sample.txt"), "w", encoding="utf-8") as f:
        f.write("anime portrait")

      ds = utils.load_extensible_dataset(tmpdir, trigger_phrase="qwen_style")
      self.assertLen(ds, 1)
      item = ds[0]
      self.assertEqual(item["caption"], "qwen_style anime portrait")
      self.assertEqual(item["image"].size, (32, 32))

      jsonl_path = os.path.join(tmpdir, "metadata.jsonl")
      with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"image": "sample.png", "text": "jsonl caption"}))
      ds_jsonl = utils.GenericImageCaptionDataset(tmpdir)
      self.assertEqual(ds_jsonl[0]["caption"], "jsonl caption")

  def test_end_to_end_training_and_inference_workflow_on_tpu(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "tiny_qwen_model")
      _create_tiny_qwen_image_pipeline(model_dir)

      cache_path = os.path.join(tmpdir, "dataset_cache.pt")
      adapter_dir = os.path.join(tmpdir, "lora_output")
      infer_dir = os.path.join(tmpdir, "infer_output")

      synthetic_samples = [
          {
              "latent": torch.randn(16, 64, 64, dtype=torch.bfloat16),
              "prompt_embeds": torch.randn(256, 32, dtype=torch.bfloat16),
              "caption": "synthetic anime portrait",
          }
          for _ in range(2)
      ]
      torch.save(synthetic_samples, cache_path)

      self.assertTrue(train_torch_tpu.USE_TPU)
      self.assertEqual(train_torch_tpu.device.type, "tpu")

      train_argv = [
          "train_torch_tpu.py",
          "--dataset_cache",
          cache_path,
          "--max_samples",
          "2",
          "--steps",
          "2",
          "--dim",
          "32",
          "--num_blocks",
          "2",
          "--rank",
          "4",
          "--output_dir",
          adapter_dir,
      ]
      with mock.patch.object(sys, "argv", train_argv):
        train_torch_tpu.main()

      peft_safetensors = os.path.join(adapter_dir, "adapter_model.safetensors")
      diff_safetensors = os.path.join(
          adapter_dir, "clean_diffusers_adapter.safetensors"
      )
      self.assertTrue(os.path.exists(peft_safetensors))
      self.assertTrue(os.path.exists(diff_safetensors))
      diff_weights = load_file(diff_safetensors)
      self.assertNotEmpty(diff_weights)

      infer_argv = [
          "infer_qwen20b_tpu.py",
          "--model_id",
          model_dir,
          "--lora_weights_path",
          adapter_dir,
          "--output_dir",
          infer_dir,
          "--custom_prompts",
          "test prompt 1;test prompt 2",
          "--resolution",
          "32",
          "--num_inference_steps",
          "2",
          "--sharding_strategy",
          "layer",
          "--num_devices",
          "1",
      ]
      with mock.patch.object(sys, "argv", infer_argv):
        infer_qwen20b_tpu.main()

      img1_path = os.path.join(infer_dir, "qwen20b_tpu_prompt_1.png")
      img2_path = os.path.join(infer_dir, "qwen20b_tpu_prompt_2.png")
      self.assertTrue(os.path.exists(img1_path))
      self.assertTrue(os.path.exists(img2_path))
      with Image.open(img1_path) as img1:
        self.assertEqual(img1.size, (32, 32))


if __name__ == "__main__":
  absltest.main()
