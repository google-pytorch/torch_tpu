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

"""Numerical parity tests for FLUX 2D Algebraic RoPE optimizations on TPU.

================================================================================
OVERVIEW: WHY 2D ALGEBRAIC RoPE ACHIEVES BIT-EXACT PARITY (atol=0.0, rtol=0.0)
================================================================================
In reference FLUX (`rope_ref` / `apply_rope_ref`), 2D rotary embeddings are
constructed as explicit 2x2 rotation matrices:
  R(theta) = [[cos(theta), -sin(theta)],
              [sin(theta),  cos(theta)]]
reshaped into 5D tensors `(B, S, 1, D/2, 2, 2)`. Applying RoPE contracts these
matrices against 6D reshaped queries/keys `x = [x_0, x_1]^T`:
  x_out_0 = R[..., 0, 0] * x_0 + R[..., 0, 1] * x_1 = cos * x_0 + (-sin) * x_1
  x_out_1 = R[..., 1, 0] * x_0 + R[..., 1, 1] * x_1 = sin * x_0 +   cos  * x_1

In optimized FLUX (`rope` / `apply_rope`), we store only the compact `[cos, sin]`
tuple in a 4D tensor `(B, S, 1, D/2, 2)` and directly compute:
  x_out_0 = x_0 * cos - x_1 * sin
  x_out_1 = x_0 * sin + x_1 * cos

Because both formulations execute the exact same IEEE-754 floating-point
multiplications and additions on identical operands, `rope` and `apply_rope`
achieve bit-exact mathematical equivalence (`atol=0.0, rtol=0.0`) while
halving positional embedding HBM bandwidth.

================================================================================
HOW TO RUN
================================================================================
Via Bazel:
  bazel test \\
    //examples/tutorials/FLUX_optimization:flux_rope_parity_test
"""

import logging
from absl.testing import absltest
import torch
import torch_tpu  # Required to initialize TPU distributed device runtime
from examples.tutorials.FLUX_optimization import flux

# ==============================================================================
# STEP 1: BIT-EXACT MATHEMATICAL & MODULE-LEVEL PARITY VERIFICATION
# ==============================================================================


class FluxRopeParityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.device = torch.device("tpu")
    self.batch_size = 2
    self.img_seq_len = 256
    self.txt_seq_len = 64
    self.hidden_size = 512
    self.num_heads = 8
    self.head_dim = self.hidden_size // self.num_heads
    self.axes_dim = (16, 24, 24)

  def test_rope_frequencies_parity(self):
    """Verifies that 2D compact frequency generator matches 5D reference with bit-exact parity."""
    pos = torch.randint(
        0, 1000, (self.batch_size, self.img_seq_len), device=self.device
    )
    dim = 32
    theta = 10000

    # 1. Generate 5D reference rotation matrices (BEFORE) and 4D compact pairs (AFTER)
    freqs_ref = flux.rope_ref(pos, dim, theta)
    freqs_opt = flux.rope(pos, dim, theta)

    # Reference shape: (B, S, D/2, 2, 2)
    # Optimized shape: (B, S, D/2, 2) where [..., 0] is cos and [..., 1] is sin
    self.assertEqual(
        freqs_ref.shape, (self.batch_size, self.img_seq_len, dim // 2, 2, 2)
    )
    self.assertEqual(
        freqs_opt.shape, (self.batch_size, self.img_seq_len, dim // 2, 2)
    )

    # 2. Extract cos and sin from the 2x2 matrix [[cos, -sin], [sin, cos]]
    cos_ref = freqs_ref[..., 0, 0]
    sin_ref = freqs_ref[..., 1, 0]
    cos_opt = freqs_opt[..., 0]
    sin_opt = freqs_opt[..., 1]

    # 3. Verify bit-exact IEEE-754 parity (atol=0.0, rtol=0.0)
    torch.testing.assert_close(cos_ref, cos_opt, atol=0.0, rtol=0.0)
    torch.testing.assert_close(sin_ref, sin_opt, atol=0.0, rtol=0.0)
    logging.info("RoPE frequencies parity verified (atol=0.0, rtol=0.0).")

  def test_apply_rope_mathematical_parity(self):
    """Verifies that `apply_rope` matches `apply_rope_ref` with bit-exact parity (`atol=0.0, rtol=0.0`)."""
    seq_len = self.img_seq_len + self.txt_seq_len
    xq = torch.randn(
        self.batch_size,
        seq_len,
        self.num_heads,
        self.head_dim,
        device=self.device,
        dtype=torch.float32,
    )
    xk = torch.randn(
        self.batch_size,
        seq_len,
        self.num_heads,
        self.head_dim,
        device=self.device,
        dtype=torch.float32,
    )

    ids = torch.randint(
        0, 1000, (self.batch_size, seq_len, 3), device=self.device
    )
    embed_ref = flux.EmbedNDRef(
        dim=self.head_dim, theta=10000, axes_dim=self.axes_dim
    )
    embed_opt = flux.EmbedND(
        dim=self.head_dim, theta=10000, axes_dim=self.axes_dim
    )

    # Generate 5D reference embeddings vs 4D compact embeddings across 3 coordinate axes
    pe_ref = embed_ref(ids).to(device=self.device)
    pe_opt = embed_opt(ids).to(device=self.device)

    xq_out_ref, xk_out_ref = flux.apply_rope_ref(xq, xk, pe_ref)
    xq_out_opt, xk_out_opt = flux.apply_rope(xq, xk, pe_opt)

    # Verify bit-exact equivalence between 5D matrix contraction and 2D algebraic rotation
    torch.testing.assert_close(xq_out_ref, xq_out_opt, atol=0.0, rtol=0.0)
    torch.testing.assert_close(xk_out_ref, xk_out_opt, atol=0.0, rtol=0.0)
    logging.info(
        "apply_rope bit-exact mathematical parity verified (atol=0.0,"
        " rtol=0.0)."
    )

  def test_double_stream_block_rope_parity(self):
    """Verifies end-to-end `DoubleStreamBlock` forward pass parity with and without `use_fused_rope`."""
    torch.manual_seed(42)
    img, txt, vec, pe_ref = flux.generate_baseline_inputs(
        batch_size=self.batch_size,
        img_seq_len=self.img_seq_len,
        txt_seq_len=self.txt_seq_len,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        axes_dim=self.axes_dim,
        device=self.device,
        use_fused_rope=False,
    )
    torch.manual_seed(42)
    _, _, _, pe_opt = flux.generate_baseline_inputs(
        batch_size=self.batch_size,
        img_seq_len=self.img_seq_len,
        txt_seq_len=self.txt_seq_len,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        axes_dim=self.axes_dim,
        device=self.device,
        use_fused_rope=True,
    )

    # Use zero-copy view for both blocks to isolate the effect of 2D algebraic RoPE
    block_ref = flux.build_double_stream_block(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        config=flux.OptimizationConfig(
            use_zero_copy_view=True, use_fused_rope=False, compile=False
        ),
    ).to(self.device)

    block_opt = flux.build_double_stream_block(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        config=flux.OptimizationConfig(
            use_zero_copy_view=True, use_fused_rope=True, compile=False
        ),
    ).to(self.device)

    block_opt.load_state_dict(block_ref.state_dict())

    with torch.no_grad():
      out_img_ref, out_txt_ref = block_ref(img=img, txt=txt, vec=vec, pe=pe_ref)
      out_img_opt, out_txt_opt = block_opt(img=img, txt=txt, vec=vec, pe=pe_opt)

    torch.testing.assert_close(out_img_ref, out_img_opt, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_txt_ref, out_txt_opt, atol=1e-5, rtol=1e-5)
    logging.info(
        "DoubleStreamBlock full module RoPE parity verified (atol=1e-5,"
        " rtol=1e-5)."
    )

  def test_double_stream_block_fused_rope_compiled_parity(self):
    """Verifies that `use_fused_rope=True` compiles cleanly under `torch.compile(backend='tpu')`."""
    img, txt, vec, pe = flux.generate_baseline_inputs(
        batch_size=self.batch_size,
        img_seq_len=self.img_seq_len,
        txt_seq_len=self.txt_seq_len,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        axes_dim=self.axes_dim,
        device=self.device,
        use_fused_rope=True,
    )

    block_eager = flux.build_double_stream_block(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        config=flux.OptimizationConfig(
            use_zero_copy_view=True, use_fused_rope=True, compile=False
        ),
    ).to(self.device)

    block_compiled = flux.build_double_stream_block(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        config=flux.OptimizationConfig(
            use_zero_copy_view=True, use_fused_rope=True, compile=True
        ),
    ).to(self.device)

    orig_mod = getattr(block_compiled, "_orig_mod", block_compiled)
    orig_mod.load_state_dict(block_eager.state_dict())

    with torch.no_grad():
      out_img_eager, out_txt_eager = block_eager(
          img=img, txt=txt, vec=vec, pe=pe
      )
      out_img_compiled, out_txt_compiled = block_compiled(
          img=img, txt=txt, vec=vec, pe=pe
      )

    # Relaxed tolerance accounts for XLA fused multiply-add (FMA) and associative
    # floating-point reduction tree reordering during whole-graph compilation.
    torch.testing.assert_close(
        out_img_eager, out_img_compiled, atol=2e-3, rtol=1e-2
    )
    torch.testing.assert_close(
        out_txt_eager, out_txt_compiled, atol=2e-3, rtol=1e-2
    )
    logging.info("DoubleStreamBlock compiled fused RoPE parity verified.")


if __name__ == "__main__":
  absltest.main()
