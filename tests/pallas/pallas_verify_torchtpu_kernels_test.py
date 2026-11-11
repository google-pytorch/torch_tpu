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

"""End-to-end verification tests on TorchTPU's shipped Pallas TPU kernels."""

from __future__ import annotations

from absl.testing import absltest
import jax
import jax.numpy as jnp
import torch
from torch.tpu import ops as tpu_ops
from torch.tpu import pallas
from torch.tpu.pallas import verifier
from torch_tpu._internal.ops import splash_attention
from torch_tpu._internal.ops.scaled_grouped_mm.kernels import rowwise_scaled_grouped_mm_kernel
from torch_tpu._internal.ops.scaled_mm.kernels import blockwise128_scaled_mm_kernel
from torch_tpu._internal.ops.scaled_mm.kernels import blockwise_scaled_mm_kernel
from tests import seed_test_utils

jax_op = pallas.jax_op


def _make_fp8_tensor(shape: tuple[int, ...]) -> torch.Tensor:
  numel = 1
  for dim in shape:
    numel *= dim
  vals = torch.linspace(-1.0, 1.0, numel, dtype=torch.float32).reshape(shape)
  return vals.to(torch.float8_e4m3fn)


@jax_op("pallas_verify_torchtpu::blockwise_scaled_mm_bf16")
def _blockwise_scaled_mm_bf16_op(
    a: jax.Array,
    b: jax.Array,
    scale_a: jax.Array,
    scale_b: jax.Array,
) -> jax.Array:
  return blockwise_scaled_mm_kernel.blockwise_scaled_mm(
      a,
      b,
      scale_a,
      scale_b,
      block_size=32,
      out_dtype=jnp.bfloat16,
      tile_m=128,
      tile_n=128,
      tile_k=256,
      transpose_rhs=False,
  )


@jax_op("pallas_verify_torchtpu::blockwise_scaled_mm_transposed_f32")
def _blockwise_scaled_mm_transposed_f32_op(
    a: jax.Array,
    b: jax.Array,
    scale_a: jax.Array,
    scale_b: jax.Array,
) -> jax.Array:
  return blockwise_scaled_mm_kernel.blockwise_scaled_mm(
      a,
      b,
      scale_a,
      scale_b,
      block_size=32,
      out_dtype=jnp.float32,
      tile_m=128,
      tile_n=128,
      tile_k=256,
      transpose_rhs=True,
  )


@jax_op("pallas_verify_torchtpu::scaled_mm_1x128_128x128_bias_bf16")
def _scaled_mm_1x128_128x128_bias_bf16_op(
    a: jax.Array,
    b: jax.Array,
    scale_a: jax.Array,
    scale_b: jax.Array,
    bias: jax.Array | None = None,
) -> jax.Array:
  return blockwise128_scaled_mm_kernel.scaled_mm_1x128_128x128(
      a,
      b,
      scale_a,
      scale_b,
      out_dtype=jnp.bfloat16,
      tile_m=128,
      tile_n=128,
      tile_k=256,
      bias=bias,
  )


@jax_op("pallas_verify_torchtpu::scaled_mm_1x128_128x128_f32")
def _scaled_mm_1x128_128x128_f32_op(
    a: jax.Array,
    b: jax.Array,
    scale_a: jax.Array,
    scale_b: jax.Array,
) -> jax.Array:
  return blockwise128_scaled_mm_kernel.scaled_mm_1x128_128x128(
      a,
      b,
      scale_a,
      scale_b,
      out_dtype=jnp.float32,
      tile_m=128,
      tile_n=128,
      tile_k=256,
  )


@jax_op("pallas_verify_torchtpu::rowwise_scaled_grouped_mm_bf16")
def _rowwise_scaled_grouped_mm_bf16_op(
    a: jax.Array,
    b: jax.Array,
    scale_a: jax.Array,
    scale_b: jax.Array,
    offs: jax.Array,
) -> jax.Array:
  return rowwise_scaled_grouped_mm_kernel.rowwise_scaled_grouped_mm(
      a,
      b,
      scale_a,
      scale_b,
      offs,
      out_dtype=jnp.bfloat16,
      transpose_rhs=False,
      tile_m=128,
      tile_k=128,
      tile_n=128,
  )


@jax_op("pallas_verify_torchtpu::rowwise_scaled_grouped_mm_transposed_f32")
def _rowwise_scaled_grouped_mm_transposed_f32_op(
    a: jax.Array,
    b: jax.Array,
    scale_a: jax.Array,
    scale_b: jax.Array,
    offs: jax.Array,
) -> jax.Array:
  return rowwise_scaled_grouped_mm_kernel.rowwise_scaled_grouped_mm(
      a,
      b,
      scale_a,
      scale_b,
      offs,
      out_dtype=jnp.float32,
      transpose_rhs=True,
      tile_m=128,
      tile_k=128,
      tile_n=128,
  )


class PallasVerifyTorchTpuKernelsTest(seed_test_utils.RepeatableTest):

  def _assert_clean_report(
      self,
      report: verifier.VerificationReport,
      *,
      expected_phases: tuple[verifier.Phase, ...],
  ) -> None:
    self.assertTrue(report.passed, msg=report.summary())
    self.assertEmpty(
        report.findings_by_severity(verifier.Severity.ERROR),
        msg=report.summary(),
    )
    self.assertEmpty(
        report.findings_by_severity(verifier.Severity.WARNING),
        msg=report.summary(),
    )
    for phase in expected_phases:
      self.assertIn(phase, report.checked_phases, msg=report.summary())

  def test_splash_sdpa_forward_and_backward_kernels_pass_verification(self):
    opts = verifier.VerificationOptions(tpu_version=torch.tpu.TpuVersion.V7)
    batch, seq_len, head_dim = 1, 128, 128

    # 1. Causal MHA forward + packed segment_ids.
    q = torch.randn((batch, 2, seq_len, head_dim), dtype=torch.bfloat16)
    k = torch.randn((batch, 2, seq_len, head_dim), dtype=torch.bfloat16)
    v = torch.randn((batch, 2, seq_len, head_dim), dtype=torch.bfloat16)
    seg_row = [0] * (seq_len // 2) + [1] * (seq_len // 2)
    segment_ids = torch.tensor([seg_row], dtype=torch.int32)

    causal_rep = verifier.verify(
        tpu_ops.splash_sdpa,
        (q, k, v),
        kwargs={"is_causal": True, "segment_ids": segment_ids},
        options=opts,
    )
    self._assert_clean_report(
        causal_rep,
        expected_phases=(
            verifier.Phase.FX_GRAPH,
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )

    # 2. Non-causal GQA and MQA forwards.
    q_gqa = torch.randn((batch, 4, seq_len, head_dim), dtype=torch.bfloat16)
    k_gqa = torch.randn((batch, 2, seq_len, head_dim), dtype=torch.bfloat16)
    v_gqa = torch.randn((batch, 2, seq_len, head_dim), dtype=torch.bfloat16)
    gqa_rep = verifier.verify(
        tpu_ops.splash_sdpa,
        (q_gqa, k_gqa, v_gqa),
        kwargs={"is_causal": False, "enable_gqa": True},
        options=opts,
    )
    self._assert_clean_report(
        gqa_rep,
        expected_phases=(
            verifier.Phase.FX_GRAPH,
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )

    k_mqa = torch.randn((batch, 1, seq_len, head_dim), dtype=torch.bfloat16)
    v_mqa = torch.randn((batch, 1, seq_len, head_dim), dtype=torch.bfloat16)
    mqa_rep = verifier.verify(
        tpu_ops.splash_sdpa,
        (q_gqa, k_mqa, v_mqa),
        kwargs={"is_causal": True, "enable_gqa": True},
        options=opts,
    )
    self._assert_clean_report(
        mqa_rep,
        expected_phases=(
            verifier.Phase.FX_GRAPH,
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )

    # 3. Direct custom backward ops (fused vmap, fused lax.scan, and split dQ/dKV).
    out = torch.randn_like(q)
    lse = torch.randn((batch, 2, seq_len), dtype=torch.float32)
    grad_out = torch.randn_like(q)
    for use_fused_bwd, use_vmap_bwd, seg_arg in (
        (True, True, segment_ids),
        (True, False, None),
        (False, True, None),
    ):
      with self.subTest(
          use_fused_bwd=use_fused_bwd,
          use_vmap_bwd=use_vmap_bwd,
          segmented=seg_arg is not None,
      ):
        _, op_bwd_name = splash_attention._get_splash_op_names(
            seq_len=seq_len,
            n_heads=2,
            n_kv_heads=2,
            scale=None,
            is_causal=True,
            local_window_size=None,
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=None if use_fused_bwd else 128,
            block_kv_dq=None if use_fused_bwd else 128,
            use_fused_bwd_kernel=use_fused_bwd,
            use_vmap_bwd=use_vmap_bwd,
            q_layout="HEAD_DIM_MINOR",
            k_layout="HEAD_DIM_MINOR",
            v_layout="HEAD_DIM_MINOR",
        )
        bwd_op = getattr(torch.ops.pallas, op_bwd_name)
        bwd_args = (
            (q, k, v, out, lse, grad_out, seg_arg)
            if seg_arg is not None
            else (q, k, v, out, lse, grad_out)
        )
        bwd_rep = verifier.verify(bwd_op, bwd_args, options=opts)
        self._assert_clean_report(
            bwd_rep,
            expected_phases=(
                verifier.Phase.CONTRACTS,
                verifier.Phase.LOWERING,
            ),
        )

    # 4. Multi-batch (batch > 1) causal forward + autograd backward in STATIC
    # mode (exercises jax.vmap vmapped_dims on Pallas grid mappings).
    q2 = torch.randn(
        (2, 2, seq_len, head_dim), dtype=torch.bfloat16, requires_grad=True
    )
    k2 = torch.randn(
        (2, 2, seq_len, head_dim), dtype=torch.bfloat16, requires_grad=True
    )
    v2 = torch.randn(
        (2, 2, seq_len, head_dim), dtype=torch.bfloat16, requires_grad=True
    )
    seg2 = torch.tensor([seg_row, seg_row], dtype=torch.int32)
    batched_static_rep = verifier.verify(
        tpu_ops.splash_sdpa,
        (q2, k2, v2),
        kwargs={"is_causal": True, "segment_ids": seg2},
        mode=verifier.VerificationMode.STATIC,
        options=opts,
    )
    self._assert_clean_report(
        batched_static_rep,
        expected_phases=(
            verifier.Phase.FX_GRAPH,
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )

  def test_blockwise_scaled_mm_kernel_passes_verification(self):
    opts = verifier.VerificationOptions(tpu_version=torch.tpu.TpuVersion.V7)

    # Tile-aligned MXFP8 (M=128, N=128, K=256, block_size=32).
    a = _make_fp8_tensor((128, 256))
    b = _make_fp8_tensor((256, 128))
    scale_a = torch.ones((128, 8), dtype=torch.float32)
    scale_b = torch.ones((8, 128), dtype=torch.float32)
    aligned_rep = verifier.verify(
        _blockwise_scaled_mm_bf16_op,
        (a, b, scale_a, scale_b),
        options=opts,
    )
    self._assert_clean_report(
        aligned_rep,
        expected_phases=(
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )

    # Unaligned shapes with transposed RHS inside an FX-traced PyTorch program.
    a_unaligned = _make_fp8_tensor((100, 192))
    b_unaligned_t = _make_fp8_tensor((100, 192))
    sa_unaligned = torch.ones((100, 6), dtype=torch.float32)
    sb_unaligned = torch.ones((6, 100), dtype=torch.float32)
    prog_rep = verifier.verify(
        lambda lhs, rhs, sa, sb: _blockwise_scaled_mm_transposed_f32_op(
            lhs, rhs, sa, sb
        ),
        (a_unaligned, b_unaligned_t, sa_unaligned, sb_unaligned),
        options=opts,
    )
    self._assert_clean_report(
        prog_rep,
        expected_phases=(
            verifier.Phase.FX_GRAPH,
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )

  def test_blockwise128_scaled_mm_kernel_passes_verification(self):
    opts = verifier.VerificationOptions(tpu_version=torch.tpu.TpuVersion.V7)

    # Tile-aligned 1x128 / 128x128 FP8 matmul with bias.
    a = _make_fp8_tensor((128, 256))
    b = _make_fp8_tensor((256, 128))
    scale_a = torch.ones((128, 2), dtype=torch.float32)
    scale_b = torch.ones((2, 1), dtype=torch.float32)
    bias = torch.linspace(-0.5, 0.5, 128, dtype=torch.float32)
    bias_rep = verifier.verify(
        _scaled_mm_1x128_128x128_bias_bf16_op,
        (a, b, scale_a, scale_b, bias),
        options=opts,
    )
    self._assert_clean_report(
        bias_rep,
        expected_phases=(
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )

    # Unaligned M and N without bias inside an FX-traced PyTorch program.
    a_unaligned = _make_fp8_tensor((100, 128))
    b_unaligned = _make_fp8_tensor((128, 100))
    sa_unaligned = torch.ones((100, 1), dtype=torch.float32)
    sb_unaligned = torch.ones((1, 1), dtype=torch.float32)
    prog_rep = verifier.verify(
        lambda lhs, rhs, sa, sb: _scaled_mm_1x128_128x128_f32_op(
            lhs, rhs, sa, sb
        ),
        (a_unaligned, b_unaligned, sa_unaligned, sb_unaligned),
        options=opts,
    )
    self._assert_clean_report(
        prog_rep,
        expected_phases=(
            verifier.Phase.FX_GRAPH,
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )

  def test_rowwise_scaled_grouped_mm_kernel_passes_verification(self):
    opts = verifier.VerificationOptions(tpu_version=torch.tpu.TpuVersion.V7)

    # Uneven group boundaries, an empty middle group, and trailing unowned rows.
    a = _make_fp8_tensor((256, 256))
    b = _make_fp8_tensor((3, 256, 128))
    scale_a = torch.ones((256,), dtype=torch.float32)
    scale_b = torch.ones((3, 128), dtype=torch.float32)
    offs = torch.tensor([100, 100, 220], dtype=torch.int32)
    grouped_rep = verifier.verify(
        _rowwise_scaled_grouped_mm_bf16_op,
        (a, b, scale_a, scale_b, offs),
        options=opts,
    )
    self._assert_clean_report(
        grouped_rep,
        expected_phases=(
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )

    # Unaligned M with transposed RHS inside an FX-traced PyTorch program.
    a_unaligned = _make_fp8_tensor((200, 128))
    b_transposed = _make_fp8_tensor((2, 128, 128))
    sa_unaligned = torch.ones((200,), dtype=torch.float32)
    sb_transposed = torch.ones((2, 128), dtype=torch.float32)
    offs_two = torch.tensor([80, 200], dtype=torch.int32)
    prog_rep = verifier.verify(
        lambda lhs, rhs, sa, sb, group_offs: (
            _rowwise_scaled_grouped_mm_transposed_f32_op(
                lhs, rhs, sa, sb, group_offs
            )
        ),
        (a_unaligned, b_transposed, sa_unaligned, sb_transposed, offs_two),
        options=opts,
    )
    self._assert_clean_report(
        prog_rep,
        expected_phases=(
            verifier.Phase.FX_GRAPH,
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
        ),
    )


if __name__ == "__main__":
  absltest.main()
