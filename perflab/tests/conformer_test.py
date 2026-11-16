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

"""Unit tests for the Conformer audio encoder benchmark."""

from unittest import mock

from absl.testing import absltest
import torch
from perflab import context as context_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import target as target_lib
from perflab.models import conformer
from tests import seed_test_utils


class ConformerTest(seed_test_utils.RepeatableTest):

  def test_lengths_to_padding_mask_infers_and_respects_max_length(self):
    lengths = torch.tensor([2, 4, 1])
    inferred = conformer.Conformer.lengths_to_padding_mask(lengths, "cpu")
    expected_inferred = torch.tensor([
        [False, False, True, True],
        [False, False, False, False],
        [False, True, True, True],
    ])
    self.assertTrue(torch.equal(inferred, expected_inferred))

    explicit = conformer.Conformer.lengths_to_padding_mask(
        lengths, "cpu", max_length=5
    )
    expected_explicit = torch.tensor([
        [False, False, True, True, True],
        [False, False, False, False, True],
        [False, True, True, True, True],
    ])
    self.assertTrue(torch.equal(explicit, expected_explicit))

  def test_conformer_forward_and_training_in_fake_tensor_mode(self):
    spec = registry_lib.REGISTRY["conformer_forward"]
    presubmit_ctx = registry_lib.resolve_context(
        spec,
        context_lib.Context(
            target=target_lib.make_target(target_lib.Platform.CPU),
            run_scope=context_lib.RunScope.PRESUBMIT,
        ),
    )
    full_ctx = registry_lib.resolve_context(
        spec,
        context_lib.Context(
            target=target_lib.make_target(target_lib.Platform.CPU),
            run_scope=context_lib.RunScope.FULL,
        ),
    )
    with mode_lib.fake_tensor_context():
      with mock.patch.object(
          torch, "randint", wraps=torch.randint
      ) as mock_randint:
        model, (inputs, padding_mask), kwargs = conformer.conformer_forward(
            presubmit_ctx
        )
        mock_randint.assert_called_once_with(1, 33, (2,))
      self.assertEqual(tuple(inputs.shape), (2, 32, 512))
      self.assertEqual(tuple(padding_mask.shape), (2, 32))
      self.assertEqual(padding_mask.dtype, torch.bool)
      out = model(inputs, padding_mask, **kwargs)
      self.assertEqual(tuple(out.shape), (2, 32, 512))

      train_model, train_args, train_kwargs, opt = conformer.conformer_training(
          presubmit_ctx
      )
      train_out = train_model(*train_args, **train_kwargs)
      train_out.sum().backward()
      opt.step()

      with mock.patch.object(
          torch, "randint", wraps=torch.randint
      ) as mock_randint:
        _, (full_inputs, full_mask), _ = conformer.conformer_forward(full_ctx)
        mock_randint.assert_called_once_with(1, 151, (32,))
      self.assertEqual(tuple(full_inputs.shape), (32, 150, 512))
      self.assertEqual(tuple(full_mask.shape), (32, 150))


if __name__ == "__main__":
  absltest.main()
