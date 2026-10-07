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

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from perflab.model_utils import torchtitan_qwen3_5
from tests import seed_test_utils
from transformers.models.qwen3_5 import modeling_qwen3_5


def _recurrent_gated_delta(q, k, v, g, beta):
  """Per-token reference (HF `torch_recurrent_gated_delta_rule` math)."""
  q = torchtitan_qwen3_5._l2norm(q.float()) * q.shape[-1] ** -0.5  # pylint: disable=protected-access
  k = torchtitan_qwen3_5._l2norm(k.float())  # pylint: disable=protected-access
  v, g, beta = v.float(), g.float(), beta.float()
  bs, seq_len, n_heads, d_k = k.shape
  state = torch.zeros(bs, n_heads, d_k, v.shape[-1])
  outs = []
  for t in range(seq_len):
    state = state * g[:, t].exp()[..., None, None]
    kv_mem = (state * k[:, t, :, :, None]).sum(-2)
    delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
    state = state + k[:, t, :, :, None] * delta[:, :, None, :]
    outs.append((state * q[:, t, :, :, None]).sum(-2))
  return torch.stack(outs, dim=1)


class TorchtitanQwen35Test(seed_test_utils.RepeatableTest):

  def test_chunked_matches_recurrent(self):
    bs, seq_len, n_heads, d_k, d_v = 2, 100, 3, 16, 8
    q = torch.randn(bs, seq_len, n_heads, d_k)
    k = torch.randn(bs, seq_len, n_heads, d_k)
    v = torch.randn(bs, seq_len, n_heads, d_v)
    g = -torch.rand(bs, seq_len, n_heads)
    beta = torch.rand(bs, seq_len, n_heads)
    # seq_len 100 with chunk 32 exercises the padding of the last chunk.
    out = torchtitan_qwen3_5.chunked_gated_delta(
        q, k, v, g, beta, chunk_size=32
    )
    test_utils.assert_close(
        out, _recurrent_gated_delta(q, k, v, g, beta), atol=1e-4, rtol=1e-4
    )

  def test_grouped_kernel_matches_hf(self):
    seq_len, n_key_heads, n_value_heads, d_k, d_v = 100, 2, 4, 16, 8
    q = torch.randn(seq_len, n_key_heads, d_k)
    k = torch.randn(seq_len, n_key_heads, d_k)
    v = torch.randn(seq_len, n_value_heads, d_v)
    g = -torch.rand(seq_len, n_value_heads)
    beta = torch.rand(seq_len, n_value_heads)
    out = torchtitan_qwen3_5._pure_torch_gated_delta_kernel_forward(  # pylint: disable=protected-access
        None, q, k, v, g, beta
    )
    # HF expects (bs, seqlen, heads, dim) with key heads already repeated.
    repeats = n_value_heads // n_key_heads
    expected, _ = modeling_qwen3_5.torch_chunk_gated_delta_rule(
        q.repeat_interleave(repeats, dim=1).unsqueeze(0),
        k.repeat_interleave(repeats, dim=1).unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        use_qk_l2norm_in_kernel=True,
    )
    test_utils.assert_close(out, expected.squeeze(0), atol=1e-4, rtol=1e-4)

  def test_debug_model_forward_backward(self):
    seq_len = 96
    config = torchtitan_qwen3_5.text_config(
        "debugmodel", num_layers=4, seq_len=seq_len
    )
    self.assertIsNone(config.vision_encoder)
    self.assertLen(config.layers, 4)
    model = torchtitan_qwen3_5.build_model(config, "cpu", torch.float32)
    # 3 GatedDeltaNet layers + 1 attention layer, all swapped to pure torch.
    kernels = [
        m
        for m in model.modules()
        if isinstance(m, torchtitan_qwen3_5.qwen3_5_gdn.GatedDeltaKernel)
    ]
    self.assertLen(kernels, 3)
    for kernel in kernels:
      self.assertEqual(
          kernel.forward.__func__,
          torchtitan_qwen3_5._pure_torch_gated_delta_kernel_forward,
      )

    tokens = torch.randint(0, config.vocab_size, (seq_len,))
    logits = model(tokens)
    self.assertEqual(tuple(logits.shape), (seq_len, config.vocab_size))
    loss = torch.nn.functional.cross_entropy(logits.float(), tokens)
    loss.backward()
    self.assertTrue(torch.isfinite(loss))
    for name, param in model.named_parameters():
      self.assertIsNotNone(param.grad, name)
      self.assertTrue(torch.isfinite(param.grad).all(), name)


if __name__ == "__main__":
  absltest.main()
