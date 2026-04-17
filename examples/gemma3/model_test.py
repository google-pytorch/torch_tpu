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

"""Tests for Gemma3 model implementation.

Run with:
```
blaze test -c opt \
//examples/gemma3:model_test
```
"""

import json
from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal import sync
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples.gemma3 import model
import transformers

log_utils.log_to_stderr()


class TestModel(absltest.TestCase):

  def setUp(self):
    super().setUp()

    seed = absltest.FLAGS.test_random_seed
    torch.manual_seed(seed)
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    self.device = torch.device("tpu")
    logging.info("Using device: %s", self.device)

    # TODO(b/498567957): fix OOMs from parallel test execution
    miniconfig = json.loads(model.GEMMA3_27B_TEXT)
    miniconfig["num_hidden_layers"] = 6
    with torch.device(self.device):
      self.hf_model = transformers.Gemma3TextModel(
          transformers.Gemma3TextConfig(**miniconfig)
      )
      self.custom_model = model.Gemma3TextModel(
          model.Gemma3TextConfig(**miniconfig)
      )
    self.custom_model.load_state_dict(self.hf_model.state_dict())
    sync.synchronize(wait=True)

  def test_numel_equal_hf_and_custom(self):
    # Assert
    self.assertEqual(
        next(iter(self.custom_model.parameters())).device.type, self.device.type
    )

    def count_params(module: torch.nn.Module) -> int:
      return sum(p.numel() for p in module.parameters())

    self.assertEqual(
        count_params(self.hf_model), count_params(self.custom_model)
    )

  def test_equivalence_hf_and_custom_rope(self):
    """Tests just the rope transform.

    This test is necessary because rope transform
    is not a nn.module tracked by
    utils.ActivationTracer.
    """
    # TODO: Fuzz the weight on RMSNorm layers. They are initialized to zeros.

    # Act
    hf_rope = self.hf_model.rotary_emb
    custom_rope = self.custom_model.rotary_emb

    x = torch.randn(2, 32, 128).to(self.device)
    position_ids = (
        torch.arange(32, device=self.device).unsqueeze(0).expand(2, 32)
    )

    hf_output = hf_rope(x, position_ids)
    custom_output = custom_rope(x, position_ids)

    utils.assert_close(hf_output, custom_output)

  # @absltest.skip("Skipping test until no divergences from reference impl.")
  def test_equivalence_hf_and_custom_modules(self):
    """Tests equivalence of hf and custom model."""
    # Arrange
    x = torch.randint(
        0, self.hf_model.config.vocab_size, (2, 32), device=self.device
    )

    # TODO: Fuzz the weight on RMSNorm layers. They are initialized to zeros.

    # Act
    with utils.ActivationTracer(self.hf_model) as hf_trace:
      _ = self.hf_model(x)

    with utils.ActivationTracer(self.custom_model) as custom_trace:
      _ = self.custom_model(x)

    # Log the series of modules first to assist in debugging.
    logging.info("Listing all forward modules, collated.")
    logging.info("len(hf_trace.forward_log)=%s", len(hf_trace.forward_log))
    for idx, (hf, custom) in enumerate(
        zip(hf_trace.forward_log, custom_trace.forward_log, strict=False)
    ):
      hf_name = type(hf["module"]).__name__
      custom_name = type(custom["module"]).__name__
      logging.info(
          "{idx=%s}: {hf_name=%s}, {custom_name=%s}", idx, hf_name, custom_name
      )

    # Assert
    for idx, (hf, custom) in enumerate(
        zip(hf_trace.forward_log, custom_trace.forward_log, strict=True)
    ):
      hf_name = type(hf["module"]).__name__
      custom_name = type(custom["module"]).__name__
      logging.info(
          "{idx=%s} Checking {hf_name=%s} vs {custom_name=%s}...",
          idx,
          hf_name,
          custom_name,
      )

      # Ignore the output of the full model. Datastructures too different.
      if isinstance(
          hf["output"],
          transformers.modeling_outputs.BaseModelOutputWithPast,
      ):
        hf["output"] = hf["output"].last_hidden_state

      utils.assert_close(
          hf["output"],
          custom["output"],
          preamble=f"{idx=}: output of {hf_name=} vs {custom_name=}",
          rtol=1.0,
          atol=0.2,
          check_value=utils.CheckValueMode.LOOSE,
      )

      # If we made it here, the assert passed.
      logging.info(
          "{idx=%s}: Output of {hf_name=%s} vs {custom_name=%s} passed.",
          idx,
          hf_name,
          custom_name,
      )


if __name__ == "__main__":
  absltest.main()
