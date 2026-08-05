# Copyright 2025 Google LLC
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

"""Functional tests of HuggingFace Transformers.

These tests verify behavior of the transformers package.
These tests do not test any local code.
"""

from unittest import mock
from absl import flags
from absl import logging
from absl.testing import absltest
import torch
from torch.nn import attention
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples.huggingface_transformers import model_configs
from tests import seed_test_utils
import transformers

log_utils.log_to_stderr()


_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["tpu", "cuda"],
    required=True,
    help="Accelerator to test.",
)


class AllTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    seed = absltest.FLAGS.test_random_seed

    torch.manual_seed(seed)
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    device = torch.device(_DEVICE.value)
    torch.set_default_device(device)

  def test_default_llama3_sdpa_is_torch_nn_functional_sdpa(self):
    """Test that HF defaults to torch.nn.functional.sdpa."""
    # Arrange
    model_id = "meta-llama/Llama-3.2-1B"
    config = model_configs.create_config_loader(model_id)()

    # Act
    model = transformers.AutoModelForCausalLM.from_config(config)

    with mock.patch(
        "transformers.models.llama.modeling_llama.eager_attention_forward",
        wraps=transformers.models.llama.modeling_llama.eager_attention_forward,
    ) as mock_eager_attention, mock.patch(
        "torch.nn.functional.scaled_dot_product_attention",
        wraps=torch.nn.functional.scaled_dot_product_attention,
    ) as mock_sdpa:

      model(torch.zeros((1, 1), dtype=torch.int64))

      # Assert
      mock_eager_attention.assert_not_called()
      mock_sdpa.assert_called()

  def test_llama3_eager_attention_is_not_torch_nn_functional_sdpa(self):
    """Test that Llama3 with attn_implementation="eager" uses eager attention."""
    # Arrange
    model_id = "meta-llama/Llama-3.2-1B"
    config = model_configs.create_config_loader(model_id)()

    # Act
    model = transformers.AutoModelForCausalLM.from_config(
        config, attn_implementation="eager"
    )

    with mock.patch(
        "transformers.models.llama.modeling_llama.eager_attention_forward",
        wraps=transformers.models.llama.modeling_llama.eager_attention_forward,
    ) as mock_eager_attention, mock.patch(
        "torch.nn.functional.scaled_dot_product_attention",
        wraps=torch.nn.functional.scaled_dot_product_attention,
    ) as mock_sdpa:

      model(torch.zeros((1, 1), dtype=torch.int64))

      # Assert
      mock_eager_attention.assert_called()
      mock_sdpa.assert_not_called()

  def test_llama3_sdpa_default_is_flash_attention(self):
    """Test that flash attention shows up in OpTracer by default."""

    # Arrange
    model_id = "meta-llama/Llama-3.2-1B"
    config = model_configs.create_config_loader(model_id)()
    model = transformers.AutoModelForCausalLM.from_config(config)

    # Act
    with utils.OpTracer() as tracer:
      model(torch.zeros((1, 1024), dtype=torch.int64))

    # Assert
    self.assertIn(
        "aten._scaled_dot_product_flash_attention.default",
        tracer.ops_log["aten"],
    )

  def test_llama3_sdpa_context_manager_to_math_works(self):
    """Test that flash attn not there when using MATH backend.

    The math backend decomposes SDPA to its component
    matmuls and views.
    """
    # Arrange
    model_id = "meta-llama/Llama-3.2-1B"
    config = model_configs.create_config_loader(model_id)()
    model = transformers.AutoModelForCausalLM.from_config(config)

    # Act
    with utils.OpTracer() as tracer:
      with attention.sdpa_kernel([attention.SDPBackend.MATH]):
        model(torch.zeros((1, 1), dtype=torch.int64))

    # Assert
    self.assertNotIn(
        "aten._scaled_dot_product_flash_attention.default",
        tracer.ops_log["aten"],
    )


if __name__ == "__main__":
  absltest.main()
