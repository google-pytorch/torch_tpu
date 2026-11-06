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

from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from perflab.model_utils import torchtitan_muse_glimmer
from tests import seed_test_utils


class TorchtitanMuseGlimmerTest(seed_test_utils.RepeatableTest):

  def test_text_config_and_build(self):
    config = torchtitan_muse_glimmer.text_config(
        "debugmodel", num_layers=2, seq_len=128
    )
    self.assertLen(config.layers, 2)
    self.assertIsNone(config.vision_encoder)

    torch.manual_seed(123)
    meta_param_dtypes: list[torch.dtype] = []
    to_empty = torch.nn.Module.to_empty

    def record_dtypes_and_materialize(module, *args, **kwargs):
      meta_param_dtypes.extend(
          p.dtype for p in module.parameters() if p.is_floating_point()
      )
      return to_empty(module, *args, **kwargs)

    with mock.patch.object(
        torch.nn.Module,
        "to_empty",
        autospec=True,
        side_effect=record_dtypes_and_materialize,
    ):
      model = torchtitan_muse_glimmer.build_model(config, "cpu", torch.bfloat16)

    # The parameters are materialized on the device in bfloat16, without a
    # float32 copy.
    self.assertEqual(torch.get_default_dtype(), torch.float32)
    self.assertNotEmpty(meta_param_dtypes)
    self.assertEqual(set(meta_param_dtypes), {torch.bfloat16})
    for p in model.parameters():
      self.assertEqual(p.dtype, torch.bfloat16)

    # Check that complex rope cache is preserved and matches fp32.
    torch.manual_seed(123)
    fp32_model = config.build()
    fp32_model.init_states(buffer_device=torch.device("cpu"))

    has_complex = False
    for (name, buf), (fp32_name, fp32_buf) in zip(
        model.named_buffers(), fp32_model.named_buffers()
    ):
      self.assertEqual(name, fp32_name)
      if buf.is_complex():
        has_complex = True
        test_utils.assert_close(buf, fp32_buf)
    self.assertTrue(has_complex)

  def test_sliding_window_validation(self):
    with self.assertRaisesRegex(ValueError, "exceeds the sliding window"):
      torchtitan_muse_glimmer.text_config(
          "debugmodel", num_layers=2, seq_len=256
      )

  def test_batch_size_2_forward_and_backward_equivalence(self):
    seq_len = 64
    config = torchtitan_muse_glimmer.text_config(
        "debugmodel", num_layers=4, seq_len=seq_len
    )
    torch.manual_seed(123)
    model = torchtitan_muse_glimmer.build_model(
        config, "cpu", torch.float32, seq_len=seq_len
    )

    seq0, labels0 = torchtitan_muse_glimmer.make_inputs(
        config.vocab_size, seq_len, rank=0, device="cpu", batch_size=1
    )
    seq1, labels1 = torchtitan_muse_glimmer.make_inputs(
        config.vocab_size, seq_len, rank=1, device="cpu", batch_size=1
    )

    model.zero_grad()
    logits0 = model(seq0)
    loss0 = torchtitan_muse_glimmer.cross_entropy_loss(
        model, (seq0,), {"labels": labels0}
    )
    loss0.backward()
    grads0 = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
    }

    model.zero_grad()
    logits1 = model(seq1)
    loss1 = torchtitan_muse_glimmer.cross_entropy_loss(
        model, (seq1,), {"labels": labels1}
    )
    loss1.backward()
    grads1 = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
    }

    model.zero_grad()
    tokens_b2 = torch.cat([seq0, seq1], dim=0)
    labels_b2 = torch.cat([labels0, labels1], dim=0)
    positions_b2 = torch.arange(seq_len, dtype=torch.long).repeat(2)
    logits_b2 = model(tokens_b2, positions=positions_b2)
    self.assertEqual(tuple(logits_b2.shape), (2 * seq_len, config.vocab_size))

    # Forward equivalence: each sequence slice in B=2 matches its B=1 forward.
    # Calibrated to 1e-5 for fp32 matmul/SDPA batch-tiling reduction order.
    logits_b2_unpacked = logits_b2.reshape(2, seq_len, config.vocab_size)
    test_utils.assert_close(
        logits_b2_unpacked[0], logits0, atol=1e-5, rtol=1e-5
    )
    test_utils.assert_close(
        logits_b2_unpacked[1], logits1, atol=1e-5, rtol=1e-5
    )

    loss_b2 = torchtitan_muse_glimmer.cross_entropy_loss(
        model,
        (tokens_b2,),
        {"labels": labels_b2, "positions": positions_b2},
    )
    test_utils.assert_close(
        loss_b2, 0.5 * (loss0 + loss1), atol=1e-5, rtol=1e-5
    )
    loss_b2.backward()

    # Gradient equivalence: B=2 mean-reduced CE gradient equals the average of
    # the two B=1 gradients. Calibrated to 1e-5 for fp32 backward accumulation
    # across 2*S tokens vs two S-token reductions.
    for name, param in model.named_parameters():
      self.assertIsNotNone(param.grad, name)
      expected_grad = 0.5 * (grads0[name] + grads1[name])
      test_utils.assert_close(param.grad, expected_grad, atol=1e-5, rtol=1e-5)

  def test_make_inputs_and_divisibility_validation(self):
    seq_len = 64
    config = torchtitan_muse_glimmer.text_config(
        "debugmodel", num_layers=2, seq_len=seq_len
    )
    tokens, labels = torchtitan_muse_glimmer.make_inputs(
        config.vocab_size, seq_len, rank=0, device="cpu", batch_size=2
    )
    self.assertEqual(tuple(tokens.shape), (2 * seq_len,))
    self.assertEqual(tuple(labels.shape), (2 * seq_len,))
    with self.assertRaisesRegex(ValueError, "batch_size must be >= 1"):
      torchtitan_muse_glimmer.make_inputs(
          config.vocab_size, seq_len, rank=0, device="cpu", batch_size=0
      )

    model = torchtitan_muse_glimmer.build_model(
        config, "cpu", torch.float32, seq_len=seq_len
    )
    # T < S delegates to the single-sequence path without raising.
    short_tokens = torch.randint(0, config.vocab_size, (seq_len // 2,))
    with torch.no_grad():
      short_logits = model(short_tokens)
    self.assertEqual(
        tuple(short_logits.shape), (seq_len // 2, config.vocab_size)
    )

    with self.assertRaisesRegex(ValueError, "must be divisible by seq_len"):
      model(
          torch.randint(0, config.vocab_size, (seq_len + 1,)),
          positions=torch.zeros(seq_len + 1, dtype=torch.long),
      )
    with self.assertRaisesRegex(ValueError, "seq_len must be >= 1"):
      torchtitan_muse_glimmer.use_batched_sdpa(model, seq_len=0)


if __name__ == "__main__":
  absltest.main()
