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

"""Tests for the glimmer-prototype Muse Glimmer perflab helpers."""

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from perflab.model_utils import muse_glimmer
from tests import seed_test_utils


class MuseGlimmerTest(seed_test_utils.RepeatableTest):

  def test_text_config_is_30b_hybrid_layout(self):
    config = muse_glimmer.text_config("30B", num_layers=4, doc_len=2048)
    rope_lib = muse_glimmer._submodule("models.common.rope")  # pylint: disable=protected-access
    self.assertEqual(config.dim, 6656)
    self.assertEqual(config.vocab_size, 202048)
    self.assertLen(config.layers, 4)
    self.assertEqual(
        [type(layer.attention.rope) for layer in config.layers],
        [rope_lib.CosSinRoPE.Config] * 3 + [type(None)],
    )
    self.assertEqual(
        [layer.attention.window_size for layer in config.layers],
        [2048, 2048, 2048, None],
    )

  def test_build_model_uses_dtype(self):
    bundle = muse_glimmer.build_model(
        "debugmodel", 4, doc_len=32, device="cpu", dtype=torch.bfloat16
    )
    self.assertIsNone(bundle.parallel_dims)
    self.assertLen(bundle.model.layers, 4)
    for name, p in bundle.model.named_parameters():
      self.assertEqual(p.dtype, torch.bfloat16, name)

  def test_training_step(self):
    # Eager flex attention has no backward on CPU.
    muse_glimmer.use_sdpa_flex_attention()
    num_docs, doc_len = 2, 32
    bundle = muse_glimmer.build_model(
        "debugmodel", 4, doc_len=doc_len, device="cpu", dtype=torch.float32
    )
    tokens, kwargs = muse_glimmer.make_batch(
        bundle, num_docs, doc_len, rank=0, world_size=1, device="cpu"
    )
    self.assertEqual(tokens.shape, (num_docs * doc_len,))
    self.assertEqual(kwargs["positions"].tolist(), list(range(doc_len)) * 2)
    for mask in kwargs["attention_masks"].values():
      self.assertEqual(mask._tpu_doc_len, doc_len)  # pylint: disable=protected-access

    opt = muse_glimmer.make_optimizer(bundle.model)
    before = [p.detach().clone() for p in bundle.model.parameters()]
    loss = muse_glimmer.cross_entropy_loss(bundle.model, (tokens,), kwargs)
    loss.backward()
    opt.step()
    self.assertTrue(torch.isfinite(loss))
    self.assertTrue(
        any(
            not torch.equal(b, a)
            for b, a in zip(before, bundle.model.parameters())
        )
    )

  def test_packed_documents_are_independent(self):
    # Longer than the debug model's 128-token window, so the sliding-window
    # masks are exercised too.
    num_docs, doc_len = 2, 192
    bundle = muse_glimmer.build_model(
        "debugmodel", 4, doc_len=doc_len, device="cpu", dtype=torch.float32
    )
    model = bundle.model
    tokens, kwargs = muse_glimmer.make_batch(
        bundle, num_docs, doc_len, rank=0, world_size=1, device="cpu"
    )
    del kwargs["labels"]
    positions = torch.arange(doc_len)
    with torch.no_grad():
      packed = model(tokens, **kwargs)
      separate = torch.cat([
          model(
              tokens[i * doc_len : (i + 1) * doc_len],
              positions=positions,
              attention_masks=model.get_attention_masks(positions=positions),
          )
          for i in range(num_docs)
      ])
    test_utils.assert_close(packed, separate, atol=1e-4, rtol=1e-4)

  def test_optimizer_clips_gradients(self):
    param = torch.nn.Parameter(torch.zeros(4))
    param.grad = torch.full((4,), 10.0)
    norm = muse_glimmer._clip_grad_norm_([param], max_norm=5.0)  # pylint: disable=protected-access
    test_utils.assert_close(norm, torch.tensor(20.0))
    test_utils.assert_close(
        param.grad.norm(), torch.tensor(5.0), atol=1e-4, rtol=1e-4
    )


if __name__ == "__main__":
  absltest.main()
