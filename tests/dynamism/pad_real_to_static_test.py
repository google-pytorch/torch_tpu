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

"""TPU tests for in-graph PadRealToStatic on wide and mixed dynamic inputs."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import oss_utils
from tests import seed_test_utils


@oss_utils.skip_if_libtpu_version(
    lambda v: v < (0, 0, 54),
    "Requires libtpu >= 0.0.54 for 64-bit and 128-bit PadRealToStatic.",
)
class PadRealToStaticTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  @parameterized.named_parameters(
      ("int64", torch.int64),
      ("float64", torch.float64),
      ("complex64", torch.complex64),
      ("complex128", torch.complex128),
  )
  def test_wide_dynamic_inputs_use_pad_real_to_static(self, dtype: torch.dtype):
    def fn(x):
      return x + x

    def make_input(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
      if dtype == torch.int64:
        cpu_x = torch.arange(rows * 8, dtype=dtype).reshape(rows, 8)
      else:
        cpu_x = torch.randn(rows, 8, dtype=dtype)
      tpu_x = cpu_x.to(self.device)
      torch._dynamo.mark_dynamic(tpu_x, 0, min=2, max=16)
      return cpu_x, tpu_x

    debugs = []
    compiled = torch.compile(
        fn,
        backend="tpu",
        fullgraph=True,
        options={"debug_callback": debugs.append},
    )
    cpu_3, tpu_3 = make_input(3)
    cpu_7, tpu_7 = make_input(7)
    cpu_5, tpu_5 = make_input(5)

    out_3 = compiled(tpu_3)
    out_7 = compiled(tpu_7)
    stats_before = torch.tpu._get_cache_stats()
    out_5 = compiled(tpu_5)
    stats_after = torch.tpu._get_cache_stats()

    self.assertEqual(
        stats_after.num_compilations - stats_before.num_compilations, 0
    )
    mlir_text = debugs[0].stablehlo_forward_text[0]
    self.assertEqual(mlir_text.count("@PadRealToStatic"), 1)
    utils.assert_close(out_3.cpu(), fn(cpu_3))
    utils.assert_close(out_7.cpu(), fn(cpu_7))
    utils.assert_close(out_5.cpu(), fn(cpu_5))

  def test_mixed_int64_and_bfloat16_attention_pad_real_to_static(self):
    vocab_size, hidden_dim, num_heads, head_dim = 128, 64, 4, 16
    embed_w = (
        torch.randn(
            vocab_size, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )
    wq, wk, wv = [
        torch.randn(
            hidden_dim, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
        for _ in range(3)
    ]

    def attn_step(input_ids, k_cache, v_cache):
      hidden = torch.nn.functional.embedding(input_ids, embed_w)
      batch_size, seq_len, _ = hidden.shape
      query = (
          (hidden @ wq)
          .view(batch_size, seq_len, num_heads, head_dim)
          .transpose(1, 2)
      )
      key_new = (
          (hidden @ wk)
          .view(batch_size, seq_len, num_heads, head_dim)
          .transpose(1, 2)
      )
      val_new = (
          (hidden @ wv)
          .view(batch_size, seq_len, num_heads, head_dim)
          .transpose(1, 2)
      )
      k_full = torch.cat([k_cache, key_new], dim=2)
      v_full = torch.cat([v_cache, val_new], dim=2)
      return torch.nn.functional.scaled_dot_product_attention(
          query, k_full, v_full, is_causal=False
      )

    def make_inputs(seq_len: int, kv_len: int):
      ids = torch.randint(
          0, vocab_size, (2, seq_len), dtype=torch.int64, device=self.device
      )
      kc = torch.randn(
          2,
          num_heads,
          kv_len,
          head_dim,
          dtype=torch.bfloat16,
          device=self.device,
      )
      vc = torch.randn(
          2,
          num_heads,
          kv_len,
          head_dim,
          dtype=torch.bfloat16,
          device=self.device,
      )
      torch._dynamo.mark_dynamic(ids, 1, min=2, max=32)
      torch._dynamo.mark_dynamic(kc, 2, min=8, max=64)
      torch._dynamo.mark_dynamic(vc, 2, min=8, max=64)
      return ids, kc, vc

    debugs = []
    compiled = torch.compile(
        attn_step,
        backend="tpu",
        fullgraph=True,
        options={"debug_callback": debugs.append},
    )
    in_7, in_19, in_13 = (
        make_inputs(7, 24),
        make_inputs(19, 40),
        make_inputs(13, 48),
    )
    out_7 = compiled(*in_7)
    out_19 = compiled(*in_19)
    stats_before = torch.tpu._get_cache_stats()
    out_13 = compiled(*in_13)
    stats_after = torch.tpu._get_cache_stats()

    self.assertEqual(
        stats_after.num_compilations - stats_before.num_compilations, 0
    )
    mlir_text = debugs[0].stablehlo_forward_text[0]
    self.assertEqual(mlir_text.count("@PadRealToStatic"), 3)
    for got, args in ((out_7, in_7), (out_19, in_19), (out_13, in_13)):
      utils.assert_close(
          got.float().cpu(),
          attn_step(*args).float().cpu(),
          rtol=2e-2,
          atol=2e-2,
      )

  def test_growing_int64_attention_mask_decode_loop_no_recompile(self):
    vocab_size, hidden_dim, num_heads, head_dim = 128, 64, 4, 16
    embed_w = (
        torch.randn(
            vocab_size, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )
    wk, wv = [
        torch.randn(
            hidden_dim, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
        for _ in range(2)
    ]

    def decode_step(next_token, k_cache, v_cache, attention_mask):
      hidden = torch.nn.functional.embedding(next_token, embed_w)
      batch_size, seq_len, _ = hidden.shape
      key_new = (
          (hidden @ wk)
          .view(batch_size, seq_len, num_heads, head_dim)
          .transpose(1, 2)
      )
      val_new = (
          (hidden @ wv)
          .view(batch_size, seq_len, num_heads, head_dim)
          .transpose(1, 2)
      )
      k_full = torch.cat([k_cache, key_new], dim=2)
      v_full = torch.cat([v_cache, val_new], dim=2)
      scores = torch.matmul(key_new, k_full.transpose(-2, -1)) * head_dim**-0.5
      mask_bias = (1 - attention_mask[:, None, None, :]).to(
          torch.float32
      ) * -1e4
      probs = torch.softmax(scores.float() + mask_bias, dim=-1)
      attn = torch.matmul(probs, v_full.float()).to(torch.bfloat16)
      return attn, k_full, v_full

    batch, kv_len = 2, 16
    tok = torch.randint(
        0, vocab_size, (batch, 1), dtype=torch.int64, device=self.device
    )
    k_cur = torch.randn(
        batch,
        num_heads,
        kv_len,
        head_dim,
        dtype=torch.bfloat16,
        device=self.device,
    )
    v_cur = torch.randn(
        batch,
        num_heads,
        kv_len,
        head_dim,
        dtype=torch.bfloat16,
        device=self.device,
    )
    ref_k, ref_v = k_cur.clone(), v_cur.clone()
    masks_tpu = []
    for step in range(6):
      mask_cpu = torch.ones((batch, kv_len + step + 1), dtype=torch.int64)
      mask_cpu[0, :2] = 0
      masks_tpu.append(mask_cpu.to(self.device))

    debugs = []
    compiled = torch.compile(
        decode_step,
        backend="tpu",
        dynamic=None,
        fullgraph=True,
        options={"debug_callback": debugs.append},
    )
    step_outputs = []
    compilations_after_warmup = 0
    for step, mask_cur in enumerate(masks_tpu):
      before_step = torch.tpu._get_cache_stats().num_compilations
      attn, k_cur, v_cur = compiled(tok, k_cur, v_cur, mask_cur)
      after_step = torch.tpu._get_cache_stats().num_compilations
      step_outputs.append((attn, k_cur, v_cur))
      # Step 0 compiles static, step 1 compiles dynamic initial layout, and
      # step 2 compiles steady-state dynamic KV output layout.
      if step == 2:
        compilations_after_warmup = after_step
      elif step > 2:
        self.assertEqual(after_step, before_step)
    compilations_after_loop = torch.tpu._get_cache_stats().num_compilations

    self.assertEqual(compilations_after_loop, compilations_after_warmup)
    self.assertTrue(
        any(
            mlir_text.count("@PadRealToStatic") == 3
            for mlir_text in itertools.chain.from_iterable(
                debug_info.stablehlo_forward_text for debug_info in debugs
            )
        )
    )
    for mask_cur, (attn, k_got, v_got) in zip(
        masks_tpu, step_outputs, strict=True
    ):
      ref_attn, ref_k, ref_v = decode_step(tok, ref_k, ref_v, mask_cur)
      utils.assert_close(
          attn.float().cpu(), ref_attn.float().cpu(), rtol=2e-2, atol=2e-2
      )
      utils.assert_close(
          k_got.float().cpu(), ref_k.float().cpu(), rtol=2e-2, atol=2e-2
      )
      utils.assert_close(
          v_got.float().cpu(), ref_v.float().cpu(), rtol=2e-2, atol=2e-2
      )

  def test_mixed_int64_and_bfloat16_multibucket(self):
    embed_w = (
        torch.randn(64, 16, dtype=torch.bfloat16, device=self.device) * 0.1
    )

    def mb_fn(sym_offset, kv_transposed_view, input_ids):
      tok_emb = torch.nn.functional.embedding(input_ids, embed_w)
      pooled = kv_transposed_view.mean(dim=1, keepdim=True)
      return tok_emb * 0.5 + pooled + sym_offset * 0.01

    def make_mb_args(sym_off: int, seq_len: int, kv_len: int):
      kv_view = torch.randn(
          2, 16, kv_len, dtype=torch.bfloat16, device=self.device
      ).transpose(1, 2)
      ids = torch.randint(
          0, 64, (2, seq_len), dtype=torch.int64, device=self.device
      )
      torch._dynamo.mark_dynamic(kv_view, 1, min=4, max=256)
      torch._dynamo.mark_dynamic(ids, 1, min=4, max=256)
      return sym_off, kv_view, ids

    debugs = []
    compiled_mb = torch.compile(
        mb_fn,
        backend="tpu",
        fullgraph=True,
        options={"num_dynamic_buckets": 2, "debug_callback": debugs.append},
    )
    args_b0, args_b1, args_b0_unseen = (
        make_mb_args(3, 12, 40),
        make_mb_args(5, 150, 200),
        make_mb_args(4, 20, 48),
    )
    with torch.compiler.config.patch({"dynamic_sources": ".*sym_offset.*"}):
      out_b0 = compiled_mb(*args_b0)
      out_b1 = compiled_mb(*args_b1)
      stats_before = torch.tpu._get_cache_stats()
      out_b0_unseen = compiled_mb(*args_b0_unseen)
      stats_after = torch.tpu._get_cache_stats()

    self.assertEqual(
        stats_after.num_compilations - stats_before.num_compilations, 0
    )
    self.assertLen(debugs, 1)
    self.assertEqual(
        debugs[0].stablehlo_forward_text[0].count("@PadRealToStatic"), 2
    )
    for args, got in (
        (args_b0, out_b0),
        (args_b1, out_b1),
        (args_b0_unseen, out_b0_unseen),
    ):
      utils.assert_close(
          got.float().cpu(), mb_fn(*args).float().cpu(), rtol=2e-2, atol=2e-2
      )

  def test_mixed_int64_and_bfloat16_autograd(self):
    def train_step(input_ids, hidden_act, proj_w):
      emb = torch.nn.functional.embedding(input_ids, proj_w)
      return ((emb + hidden_act) ** 2).sum()

    debugs = []
    compiled_train = torch.compile(
        train_step,
        backend="tpu",
        fullgraph=True,
        options={"debug_callback": debugs.append},
    )
    proj_w_cpu = torch.randn(64, 16, dtype=torch.bfloat16)
    proj_w = proj_w_cpu.to(self.device).requires_grad_(True)

    def run_train_step(seq_len: int) -> int:
      ids_cpu = torch.randint(0, 64, (2, seq_len), dtype=torch.int64)
      act_cpu = torch.randn(2, seq_len, 16, dtype=torch.bfloat16)
      ids = ids_cpu.to(self.device)
      act = act_cpu.to(self.device).requires_grad_(True)
      torch._dynamo.mark_dynamic(ids, 1, min=2, max=32)
      torch._dynamo.mark_dynamic(act, 1, min=2, max=32)
      proj_w.grad = None

      before = torch.tpu._get_cache_stats().num_compilations
      loss = compiled_train(ids, act, proj_w)
      loss.backward()
      compilations_delta = (
          torch.tpu._get_cache_stats().num_compilations - before
      )

      ref_w = proj_w_cpu.clone().float().requires_grad_(True)
      ref_act = act_cpu.clone().float().requires_grad_(True)
      ref_loss = train_step(ids_cpu, ref_act, ref_w)
      ref_loss.backward()
      utils.assert_close(loss.float().cpu(), ref_loss, rtol=2e-2, atol=2e-2)
      utils.assert_close(
          act.grad.float().cpu(), ref_act.grad, rtol=3e-2, atol=3e-2
      )
      utils.assert_close(
          proj_w.grad.float().cpu(), ref_w.grad, rtol=3e-2, atol=3e-2
      )
      return compilations_delta

    run_train_step(6)
    run_train_step(14)
    unseen_delta = run_train_step(10)

    self.assertEqual(unseen_delta, 0)
    self.assertLen(debugs, 1)
    self.assertIn("@PadRealToStatic", debugs[0].stablehlo_forward_text[0])
    self.assertIn("@PadRealToStatic", debugs[0].stablehlo_backward_text[0])

  def test_inplace_mutated_wide_dynamic_input_not_donated(self):
    def fn(ids, kv):
      ids.add_(1)
      return kv.mul_(2.0)

    debugs = []
    compiled = torch.compile(
        fn,
        backend="tpu",
        dynamic=None,
        fullgraph=True,
        options={"debug_callback": debugs.append},
    )

    def run_step(seq_len: int) -> None:
      ids_cpu = torch.arange(2 * seq_len, dtype=torch.int64).reshape(2, seq_len)
      kv_cpu = torch.randn(2, seq_len, 8, dtype=torch.bfloat16)
      ids_tpu = ids_cpu.to(self.device)
      kv_tpu = kv_cpu.to(self.device)
      torch._dynamo.mark_dynamic(ids_tpu, 1, min=2, max=32)
      torch._dynamo.mark_dynamic(kv_tpu, 1, min=2, max=32)
      got_out = compiled(ids_tpu, kv_tpu)
      ref_out = fn(ids_cpu, kv_cpu)
      utils.assert_close(got_out.cpu(), ref_out)
      utils.assert_close(ids_tpu.cpu(), ids_cpu)
      utils.assert_close(kv_tpu.cpu(), kv_cpu)

    run_step(5)
    run_step(11)
    stats_before = torch.tpu._get_cache_stats()
    run_step(7)
    stats_after = torch.tpu._get_cache_stats()

    self.assertEqual(
        stats_after.num_compilations - stats_before.num_compilations, 0
    )
    mlir_text = debugs[0].stablehlo_forward_text[0]
    self.assertEqual(mlir_text.count("@PadRealToStatic"), 2)
    self.assertNotIn("tf.aliasing_output", mlir_text)


if __name__ == "__main__":
  absltest.main()
