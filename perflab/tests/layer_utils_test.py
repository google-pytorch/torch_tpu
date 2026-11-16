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

"""Tests for perflab.layer_utils configs and builders."""

import ast
import importlib
import sys

from absl.testing import absltest
import torch
from perflab.layer_utils import builders
from perflab.layer_utils import configs
from tests import seed_test_utils

_GOOGLE_PREFIX = "torch_tpu.google"
_CPU = torch.device("cpu")


def _google_modules() -> set[str]:
  return {
      name
      for name in sys.modules
      if name == _GOOGLE_PREFIX or name.startswith(_GOOGLE_PREFIX + ".")
  }


def _imported_module_names(module) -> set[str]:
  """Returns the module names that `module`'s source imports directly."""
  with open(module.__file__, encoding="utf-8") as f:
    tree = ast.parse(f.read())
  names = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      names.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
      names.add(node.module)
  return names


class MicrobenchmarkNameTest(seed_test_utils.RepeatableTest):
  """The expected strings are what the legacy `get_microbenchmark_name` yields."""

  def test_linear_config(self):
    config = configs.LinearConfig(
        batch_size=1, seq_len=128, in_features=128, out_features=128
    )
    self.assertEqual(
        configs.microbenchmark_name(config),
        "batch_size_1_seq_len_128_in_features_128_out_features_128",
    )

  def test_layer_norm_config(self):
    config = configs.LayerNormConfig(
        batch_size=1, seq_len=128, num_features=128
    )
    self.assertEqual(
        configs.microbenchmark_name(config),
        "batch_size_1_seq_len_128_num_features_128_num_normalized_dims_1",
    )

  def test_sdpa_config_with_backend(self):
    config = configs.SdpaConfig(
        batch_size=1,
        embed_dim=4096,
        q_seq_len=128,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=True,
        enable_gqa=True,
        dtype=torch.bfloat16,
        backend=torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    )
    self.assertEqual(
        configs.microbenchmark_name(config),
        "batch_size_1_embed_dim_4096_q_seq_len_128_q_num_heads_8"
        "_kv_num_heads_8_qk_head_dim_64_v_head_dim_64_is_causal_True"
        "_enable_gqa_True_dtype_bfloat16"
        "_backend_SDPBackend.FLASH_ATTENTION",
    )

  def test_configs_with_backends_matches_first_base_config(self):
    first = next(
        configs.SdpaConfig.configs_with_backends(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        )
    )
    self.assertEqual(
        configs.microbenchmark_name(first),
        "batch_size_1_embed_dim_4096_q_seq_len_128_q_num_heads_8"
        "_kv_num_heads_8_qk_head_dim_64_v_head_dim_64_is_causal_True"
        "_enable_gqa_True_dtype_bfloat16"
        "_backend_SDPBackend.FLASH_ATTENTION",
    )


class BuildersTest(seed_test_utils.RepeatableTest):

  def test_linear(self):
    config = configs.LINEAR_CONFIGS[0]
    result = builders.ml_layer_model_builder(
        builders.LayerArgs(
            model_name="nn.Linear",
            batch_size=config.batch_size,
            sequence_length=config.seq_len,
            custom_kwargs={
                "in_features": config.in_features,
                "out_features": config.out_features,
            },
        ),
        _CPU,
        torch.float32,
        is_training=True,
    )

    self.assertIsInstance(result.model.linear, torch.nn.Linear)
    self.assertTrue(result.model.training)
    self.assertEqual(
        tuple(result.example_inputs.shape),
        (config.batch_size, config.seq_len, config.in_features),
    )
    self.assertEqual(result.example_inputs.dtype, torch.float32)
    self.assertEqual(result.example_inputs.device.type, "cpu")

  def test_layer_norm(self):
    config = configs.LAYER_NORM_CONFIGS[0]
    result = builders.ml_layer_model_builder(
        builders.LayerArgs(
            model_name="nn.LayerNorm",
            batch_size=config.batch_size,
            custom_kwargs={
                "normalized_shape": config.normalized_shape,
                "shape": config.shape,
            },
        ),
        _CPU,
        torch.float32,
        is_training=False,
    )

    self.assertIsInstance(result.model.layernorm, torch.nn.LayerNorm)
    self.assertFalse(result.model.training)
    self.assertEqual(
        result.model.layernorm.normalized_shape, (config.num_features,)
    )
    self.assertEqual(
        tuple(result.example_inputs.shape),
        (config.batch_size, config.seq_len, config.num_features),
    )

  def test_sdpa_math_backend(self):
    config = configs.SdpaConfig(
        batch_size=2,
        embed_dim=64,
        q_seq_len=16,
        q_num_heads=4,
        kv_num_heads=2,
        qk_head_dim=16,
        v_head_dim=16,
        is_causal=True,
        enable_gqa=True,
        dtype=torch.float32,
        backend=torch.nn.attention.SDPBackend.MATH,
    )
    result = builders.ml_layer_model_builder(
        builders.LayerArgs(
            model_name="nn.f.scaled_dot_product_attention",
            batch_size=config.batch_size,
            custom_kwargs={
                "embed_dim": config.embed_dim,
                "q_seq_len": config.q_seq_len,
                "q_num_heads": config.q_num_heads,
                "kv_num_heads": config.kv_num_heads,
                "qk_head_dim": config.qk_head_dim,
                "v_head_dim": config.v_head_dim,
                "is_causal": config.is_causal,
                "enable_gqa": config.enable_gqa,
                "backend": config.backend,
            },
        ),
        _CPU,
        config.dtype,
        is_training=False,
    )

    expected_shape = (config.batch_size, config.q_seq_len, config.embed_dim)
    self.assertIsInstance(result.model.q_proj, torch.nn.Linear)
    self.assertIsInstance(result.model.out_proj, torch.nn.Linear)
    self.assertEqual(tuple(result.example_inputs.shape), expected_shape)
    with torch.no_grad():
      output = result.model(result.example_inputs)
    self.assertEqual(tuple(output.shape), expected_shape)

  def test_shape_from_kwargs_without_batch_size(self):
    result = builders.ml_layer_model_builder(
        builders.LayerArgs(
            model_name="nn.Tanh",
            batch_size=None,
            custom_kwargs={"shape": (4, 8)},
        ),
        _CPU,
        torch.float32,
        is_training=True,
    )

    self.assertIsInstance(result.model.tanh, torch.nn.Tanh)
    self.assertEqual(tuple(result.example_inputs.shape), (4, 8))
    self.assertTrue(result.example_inputs.requires_grad)


class NoGoogleImportsTest(seed_test_utils.RepeatableTest):

  def test_sources_do_not_import_google(self):
    for module in (configs, builders):
      with self.subTest(module=module.__name__):
        google_imports = {
            name
            for name in _imported_module_names(module)
            if name.startswith(_GOOGLE_PREFIX)
        }
        self.assertEmpty(google_imports)

  def test_import_loads_no_google_modules(self):
    # Derive module names from the imported modules: the OSS export moves
    # perflab to a different package path, and string literals are not
    # rewritten.
    package = configs.__name__.rpartition(".")[0]
    reloaded = [
        name
        for name in sys.modules
        if name == package or name.startswith(package + ".")
    ]
    saved = {name: sys.modules.pop(name) for name in reloaded}
    try:
      before = _google_modules()
      importlib.import_module(configs.__name__)
      importlib.import_module(builders.__name__)
      self.assertEmpty(_google_modules() - before)
    finally:
      sys.modules.update(saved)


if __name__ == "__main__":
  absltest.main()
