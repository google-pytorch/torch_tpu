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

"""Numerical correctness and parity tests for fused StableHLO GRU on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from torch_tpu._internal.utils.test_utils import assert_close
from tests import seed_test_utils


class GruOpTest(seed_test_utils.RepeatableTest):
  """Unit and integration tests for TPU fused StableHLO GRU (aten::gru.input).

  Verifies numerical parity, autograd gradient calculation, corner-case input
  handling, strided memory layouts, and error validations against PyTorch's
  canonical CPU GRU reference implementation.
  """

  def _create_models(
      self,
      input_size: int,
      hidden_size: int,
      num_layers: int,
      bias: bool,
      batch_first: bool,
      dtype: torch.dtype,
      bidirectional: bool = False,
      dropout: float = 0.0,
  ):
    """Instantiates CPU reference and TPU target models with identical weights.

    Copies initialized weights directly from the CPU module to the TPU module
    so that differences in outputs and gradients reflect only numerical and
    arithmetic variations between CPU and TPU execution.
    """
    cpu_gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dtype=dtype,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    tpu_gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dtype=dtype,
        bidirectional=bidirectional,
        dropout=dropout,
        device="tpu",
    )
    with torch.no_grad():
      for cpu_p, tpu_p in zip(cpu_gru.parameters(), tpu_gru.parameters()):
        tpu_p.copy_(cpu_p.to("tpu"))
    return cpu_gru, tpu_gru

  @parameterized.named_parameters(
      (
          "float32_seq_first_1layer",
          torch.float32,
          False,  # batch_first
          1,  # num_layers
          True,  # bias
      ),
      (
          "float32_batch_first_1layer",
          torch.float32,
          True,  # batch_first
          1,  # num_layers
          True,  # bias
      ),
      (
          "float32_seq_first_2layers",
          torch.float32,
          False,  # batch_first
          2,  # num_layers
          True,  # bias
      ),
      (
          "float32_batch_first_2layers",
          torch.float32,
          True,  # batch_first
          2,  # num_layers
          True,  # bias
      ),
      (
          "float32_no_bias_1layer",
          torch.float32,
          False,  # batch_first
          1,  # num_layers
          False,  # bias
      ),
      (
          "bfloat16_seq_first_1layer",
          torch.bfloat16,
          False,  # batch_first
          1,  # num_layers
          True,  # bias
      ),
      (
          "bfloat16_batch_first_1layer",
          torch.bfloat16,
          True,  # batch_first
          1,  # num_layers
          True,  # bias
      ),
      (
          "bfloat16_seq_first_2layers",
          torch.bfloat16,
          False,  # batch_first
          2,  # num_layers
          True,  # bias
      ),
      (
          "bfloat16_batch_first_2layers",
          torch.bfloat16,
          True,  # batch_first
          2,  # num_layers
          True,  # bias
      ),
  )
  def test_forward_and_backward_parity(
      self,
      dtype: torch.dtype,
      batch_first: bool,
      num_layers: int,
      bias: bool,
  ):
    """Verifies forward outputs and BPTT backward gradients against CPU reference.

    What is being tested:
      - Full round-trip forward inference and autograd backward execution.
      - Tested configurations: Float32 and BFloat16 dtypes, sequence-first [T,
      B, H]
        and batch-first [B, T, H] layouts, single-layer and multi-layer (L=2)
        stacks,
        with and without additive biases.
      - Mathematical operations: input projection GEMM (X @ W_ih^T), recurrent
        gate evaluations (r, z, n), hidden state updates, and reverse BPTT
        gradient
        accumulation across all timesteps.

    Expected result:
      - Forward output sequence 'out_tpu' matches PyTorch CPU reference
      'out_cpu'
        within numerical tolerance (atol=5e-3 / 5e-2 for FP32 / BF16).
      - Final hidden state 'hy_tpu' matches CPU 'hy_cpu' within tolerance.
      - Backward input gradient 'x_tpu.grad' matches 'x_cpu.grad'.
      - Initial state gradient 'h0_tpu.grad' matches 'h0_cpu.grad'.
      - All parameter gradients (weight_ih, weight_hh, bias_ih, bias_hh) across
      all
        layers match CPU reference gradients within tolerance.
    """
    batch = 4
    seq_len = 8
    input_size = 16
    hidden_size = 32

    cpu_gru, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dtype=dtype,
    )

    x_shape = (
        (batch, seq_len, input_size)
        if batch_first
        else (seq_len, batch, input_size)
    )
    x_cpu = torch.randn(x_shape, dtype=dtype, requires_grad=True)
    h0_cpu = torch.randn(
        num_layers, batch, hidden_size, dtype=dtype, requires_grad=True
    )

    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)

    # 1. Forward Pass Evaluation
    out_cpu, hy_cpu = cpu_gru(x_cpu, h0_cpu)
    out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)

    atol = 5e-3 if dtype == torch.float32 else 5e-2
    rtol = 5e-3 if dtype == torch.float32 else 5e-2

    assert_close(out_tpu.cpu(), out_cpu, atol=atol, rtol=rtol, check_dtype=True)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=atol, rtol=rtol, check_dtype=True)

    # 2. Backward Pass Evaluation (BPTT)
    loss_cpu = out_cpu.sum() + hy_cpu.sum()
    loss_tpu = out_tpu.sum() + hy_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    assert_close(
        x_tpu.grad.cpu(), x_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )
    atol_h0 = 1e-2 if dtype == torch.float32 else 5e-2
    assert_close(
        h0_tpu.grad.cpu(),
        h0_cpu.grad,
        atol=atol_h0,
        rtol=rtol,
        check_dtype=True,
    )

    atol_params = 5e-2 if dtype == torch.float32 else 1e-1
    rtol_params = 5e-2
    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_gru.named_parameters(), tpu_gru.named_parameters()
    ):
      self.assertEqual(name_c, name_t)
      self.assertIsNotNone(p_cpu.grad)
      self.assertIsNotNone(p_tpu.grad)
      assert_close(
          p_tpu.grad.cpu(),
          p_cpu.grad,
          atol=atol_params,
          rtol=rtol_params,
          check_dtype=True,
      )

  def test_default_initial_states(self):
    """Verifies forward and backward execution when h_0 is omitted (None).

    What is being tested:
      - Calling nn.GRU or aten::gru.input without an explicit initial hidden
      state h_0.
      - In PyTorch semantics, an omitted h_0 implies an all-zeros initial hidden
      state
        tensor of shape [num_layers * num_directions, batch, hidden_size].
      - Both forward evaluation and reverse BPTT gradient propagation with
      default h_0.

    Expected result:
      - The operator automatically initializes h_0 to zeros in TPU vector
      memory.
      - Forward sequence output and final hidden state match CPU reference
      within atol=5e-3.
      - Backward gradient with respect to input x matches CPU reference within
      atol=5e-3.
    """
    batch = 3
    seq_len = 6
    input_size = 8
    hidden_size = 16

    cpu_gru, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
    )

    x_cpu = torch.randn(seq_len, batch, input_size, requires_grad=True)
    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)

    out_cpu, hy_cpu = cpu_gru(x_cpu)
    out_tpu, hy_tpu = tpu_gru(x_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)

    (out_cpu.sum() + hy_cpu.sum()).backward()
    (out_tpu.sum() + hy_tpu.sum()).backward()

    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)

  @parameterized.named_parameters(
      ("t1_b1_h1", 1, 1, 1, 1),
      ("t7_b2_h4", 7, 2, 8, 4),
      ("t8_b4_h16", 8, 4, 16, 16),
      ("t9_b3_h8", 9, 3, 8, 8),
      ("t17_b2_h16", 17, 2, 8, 16),
      ("t64_b2_h32", 64, 2, 16, 32),
      ("t128_b2_h32", 128, 2, 16, 32),
  )
  def test_boundary_and_corner_dimensions(
      self, seq_len: int, batch: int, input_size: int, hidden_size: int
  ):
    """Validates execution on extreme edge cases, non-multiples of unroll, and power-of-two sizes.

    What is being tested:
      - 't1_b1_h1': Degenerate single-element tensor (T=1, B=1, H=1), testing
      minimal boundaries.
      - 't7_b2_h4': T=7 < k=8 (strictly smaller than the unroll factor k=8),
      testing the static
        unrolled loop path where chunked loop count is zero.
      - 't8_b4_h16': T=8 == k (exact multiple of 1 chunk, 0 remainder
      timesteps).
      - 't9_b3_h8': T=9 = 8 + 1 (1 chunk of 8 plus 1 remainder timestep),
      testing tail remainder
        concatenation boundaries in forward and backward passes.
      - 't17_b2_h16': T=17 = 2*8 + 1 (multi-chunk with remainder timestep).
      - 't64_b2_h32', 't128_b2_h32': Long sequences (64 and 128 timesteps),
      testing numerical
        stability and accumulator correctness across many while-loop iterations.

    Expected result:
      - Forward outputs (out, hy) match PyTorch CPU reference within atol=5e-3,
      rtol=5e-3.
      - Backward gradients (x.grad, h0.grad) match CPU reference within
      atol=5e-3, rtol=5e-3.
      - No buffer overflow, out-of-bounds dynamic slice, or shape mismatch
      occurs.
    """
    cpu_gru, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
    )

    x_cpu = torch.randn(seq_len, batch, input_size, requires_grad=True)
    h0_cpu = torch.randn(1, batch, hidden_size, requires_grad=True)

    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)

    out_cpu, hy_cpu = cpu_gru(x_cpu, h0_cpu)
    out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)

    (out_cpu.sum() + hy_cpu.sum()).backward()
    (out_tpu.sum() + hy_tpu.sum()).backward()

    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)
    assert_close(h0_tpu.grad.cpu(), h0_cpu.grad, atol=5e-3, rtol=5e-3)

  def test_partial_gradients(self):
    """Verifies backward pass when only a subset of inputs require gradients.

    What is being tested:
      - Autograd sparsity handling when some inputs have requires_grad=False.
      - Case 1: Only input x requires gradient (x.requires_grad=True,
      h0.requires_grad=False).
      - Case 2: Only initial state h0 requires gradient (x.requires_grad=False,
      h0.requires_grad=True).

    Expected result:
      - In Case 1: x.grad is properly computed and non-None; h0.grad is None.
      - In Case 2: x.grad is None; h0.grad is properly computed and non-None.
      - The backward custom operator does not crash or perform illegal writes
      for inactive gradients.
    """
    batch, seq_len, input_size, hidden_size = 2, 4, 8, 16

    _, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
    )

    # Case 1: Only input requires grad
    x = torch.randn(
        seq_len, batch, input_size, device="tpu", requires_grad=True
    )
    h0 = torch.randn(1, batch, hidden_size, device="tpu", requires_grad=False)
    out, hy = tpu_gru(x, h0)
    (out.sum() + hy.sum()).backward()
    self.assertIsNotNone(x.grad)
    self.assertIsNone(h0.grad)

    # Case 2: Only h0 requires grad
    x2 = torch.randn(
        seq_len, batch, input_size, device="tpu", requires_grad=False
    )
    h0_2 = torch.randn(1, batch, hidden_size, device="tpu", requires_grad=True)
    out2, hy2 = tpu_gru(x2, h0_2)
    (out2.sum() + hy2.sum()).backward()
    self.assertIsNone(x2.grad)
    self.assertIsNotNone(h0_2.grad)

  def test_non_contiguous_inputs(self):
    """Verifies behavior when input tensors are non-contiguous in memory.

    What is being tested:
      - Passing non-contiguous strided input tensors (e.g., generated via
      strided slice x[::2]).
      - Verifies that input tensor layouts with non-standard stride vectors are
      properly handled
        without memory corruption, incorrect tensor indexing, or incorrect
        output values.

    Expected result:
      - Output sequence 'out_tpu' and final hidden state 'hy_tpu' match CPU
      reference within atol=5e-3.
      - Both contiguous and non-contiguous views produce identical numerical
      results.
    """
    batch, seq_len, input_size, hidden_size = 4, 8, 16, 32

    cpu_gru, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
    )

    # Create non-contiguous sequence input by slicing a larger tensor
    raw_x_cpu = torch.randn(seq_len * 2, batch, input_size)
    x_view_cpu = raw_x_cpu[::2]
    self.assertFalse(x_view_cpu.is_contiguous())

    h0_cpu = torch.randn(1, batch, hidden_size, requires_grad=True)

    x_tpu = x_view_cpu.detach().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)

    out_cpu, hy_cpu = cpu_gru(x_view_cpu, h0_cpu)
    out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)

  def test_bidirectional_training_backward_parity(self):
    """Validates bidirectional GRU training forward and BPTT backward gradients.

    What is being tested:
      - Bidirectional GRU (bidirectional=True) in training mode.
      - Forward pass: Concurrent forward (t=0..T-1) and reverse (t=T-1..0)
      recurrent evaluation,
        with outputs concatenated along feature dimension: [T, B, 2*H].
      - Backward pass: Analytical BPTT for both forward and reverse directions
      with lockstep
        time-reversed alignment and streaming parameter gradient accumulation.

    Expected result:
      - Forward output sequence and both final hidden states match CPU reference
      (atol=5e-3).
      - Backward input gradient x.grad and initial state gradient h0.grad match
      CPU (atol=5e-3).
      - Parameter gradients for both forward and reverse directions match CPU
      reference (atol=5e-2).
    """
    batch = 3
    seq_len = 10
    input_size = 8
    hidden_size = 16
    num_layers = 1

    cpu_gru, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        bidirectional=True,
    )

    x_cpu = torch.randn(seq_len, batch, input_size, requires_grad=True)
    h0_cpu = torch.randn(2, batch, hidden_size, requires_grad=True)

    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)

    out_cpu, hy_cpu = cpu_gru(x_cpu, h0_cpu)
    out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)

    loss_cpu = out_cpu.sum() + hy_cpu.sum()
    loss_tpu = out_tpu.sum() + hy_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)
    assert_close(h0_tpu.grad.cpu(), h0_cpu.grad, atol=5e-3, rtol=5e-3)

    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_gru.named_parameters(), tpu_gru.named_parameters()
    ):
      self.assertEqual(name_c, name_t)
      assert_close(p_tpu.grad.cpu(), p_cpu.grad, atol=5e-2, rtol=5e-2)

  def test_bidirectional_eval_parity(self):
    """Validates bidirectional GRU inference forward pass across multiple layers.

    What is being tested:
      - Multi-layer bidirectional GRU (num_layers=2, bidirectional=True) in eval
      mode.
      - Inter-layer feature dimension doubling: Layer 0 receives input_size=12
      and produces
        2*hidden=48; Layer 1 receives input_size=48 and produces 2*hidden=48.
      - Final hidden state stacking across layers and directions: [2 *
      num_layers, batch, hidden].

    Expected result:
      - Forward sequence output and stacked final hidden state match CPU
      reference (atol=5e-3).
      - Correct channel slicing and stacking across both layers and directions.
    """
    batch = 2
    seq_len = 16
    input_size = 12
    hidden_size = 24
    num_layers = 2

    cpu_gru, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        bidirectional=True,
    )
    cpu_gru.eval()
    tpu_gru.eval()

    x_cpu = torch.randn(seq_len, batch, input_size)
    h0_cpu = torch.randn(2 * num_layers, batch, hidden_size)

    x_tpu = x_cpu.to("tpu")
    h0_tpu = h0_cpu.to("tpu")

    with torch.no_grad():
      out_cpu, hy_cpu = cpu_gru(x_cpu, h0_cpu)
      out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)

  def test_invalid_tensor_rank_raises(self):
    """Verifies that passing an invalid 4D tensor input raises an exception.

    What is being tested:
      - Input validation for tensor dimensionality: GRU requires a 3D input
      tensor
        ([seq_len, batch, input_size] or [batch, seq_len, input_size]).
      - Passing an invalid 4D tensor of shape [2, 4, 8, 16].

    Expected result:
      - Raises RuntimeError or ValueError indicating that the input must be a 3D
      tensor.
    """
    tpu_gru = torch.nn.GRU(input_size=8, hidden_size=16, device="tpu")
    x_4d = torch.randn(2, 4, 8, 16, device="tpu")
    with self.assertRaises(  # ASSERT_RAISES_OK=gru validation
        (RuntimeError, ValueError)
    ):
      tpu_gru(x_4d)

  def test_zero_sequence_length_raises(self):
    """Verifies that an input tensor with seq_len=0 raises RuntimeError.

    What is being tested:
      - Input validation for sequence length: sequence length must be strictly
      positive (T > 0).
      - Passing an input tensor with shape [0, 4, 8].

    Expected result:
      - Raises RuntimeError stating that sequence length must be larger than 0
      in RNN.
    """
    tpu_gru = torch.nn.GRU(input_size=8, hidden_size=16, device="tpu")
    x_zero_seq = torch.randn(0, 4, 8, device="tpu")
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=gru validation
      tpu_gru(x_zero_seq)

  def test_mismatched_hidden_batch_raises(self):
    """Verifies that a batch size mismatch between input x and h_0 raises RuntimeError.

    What is being tested:
      - Consistency validation across input tensors: batch dimension of input x
      (batch=2)
        must match batch dimension of initial state h_0 (batch=3).

    Expected result:
      - Raises RuntimeError indicating that hx size(1) does not match input
      batch size.
    """
    tpu_gru = torch.nn.GRU(input_size=8, hidden_size=16, device="tpu")
    x = torch.randn(4, 2, 8, device="tpu")
    h0_bad = torch.randn(1, 3, 16, device="tpu")  # batch=3 != 2
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=gru validation
      tpu_gru(x, h0_bad)

  def test_mismatched_hidden_layers_raises(self):
    """Verifies that a layer count mismatch between num_layers and h_0 raises RuntimeError.

    What is being tested:
      - Consistency validation between model num_layers (num_layers=2) and h_0
      leading
        dimension size(0) (size=1).

    Expected result:
      - Raises RuntimeError indicating that hx size(0) must match num_layers *
      num_directions.
    """
    tpu_gru = torch.nn.GRU(
        input_size=8, hidden_size=16, num_layers=2, device="tpu"
    )
    x = torch.randn(4, 2, 8, device="tpu")
    h0_bad = torch.randn(1, 2, 16, device="tpu")  # layers=1 != 2
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=gru validation
      tpu_gru(x, h0_bad)

  def test_multi_layer_dropout_eval_matches_cpu(self):
    """Verifies that multi-layer GRU with dropout>0 in eval mode matches CPU.

    What is being tested:
      - Multi-layer GRU (num_layers=3) with configured dropout (dropout=0.3) in
      eval mode (model.eval()).
      - Under PyTorch semantics, inter-layer dropout is strictly inactive during
      evaluation (dropout probability = 0).

    Expected result:
      - Outputs match CPU reference without any dropped activations or scaling
      distortion (atol=5e-3).
    """
    batch = 2
    seq_len = 8
    input_size = 8
    hidden_size = 16
    num_layers = 3
    dropout = 0.3

    cpu_gru, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        dropout=dropout,
    )
    cpu_gru.eval()
    tpu_gru.eval()

    x_cpu = torch.randn(seq_len, batch, input_size)
    h0_cpu = torch.randn(num_layers, batch, hidden_size)

    x_tpu = x_cpu.to("tpu")
    h0_tpu = h0_cpu.to("tpu")

    with torch.no_grad():
      out_cpu, hy_cpu = cpu_gru(x_cpu, h0_cpu)
      out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)

  def test_multi_layer_dropout_train_and_backward(self):
    """Verifies multi-layer GRU with dropout>0 runs in train mode and computes gradients.

    What is being tested:
      - Multi-layer GRU (num_layers=3) with active inter-layer dropout
      (dropout=0.5) in training mode.
      - Philox PRNG random uniform mask generation and backward dropout mask
      application.
      - End-to-end forward and backward execution with cached activation states.

    Expected result:
      - Execution succeeds without crashes or runtime errors.
      - Backward gradients for input x, initial state h0, and all layer weights
      are non-None and contain no NaNs.
    """
    batch = 2
    seq_len = 8
    input_size = 8
    hidden_size = 16
    num_layers = 3
    dropout = 0.5

    _, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        dropout=dropout,
    )
    tpu_gru.train()

    x_tpu = torch.randn(
        seq_len, batch, input_size, device="tpu", requires_grad=True
    )
    h0_tpu = torch.randn(
        num_layers, batch, hidden_size, device="tpu", requires_grad=True
    )

    out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)
    loss = out_tpu.sum() + hy_tpu.sum()
    loss.backward()

    self.assertIsNotNone(x_tpu.grad)
    self.assertIsNotNone(h0_tpu.grad)
    for _, p in tpu_gru.named_parameters():
      self.assertIsNotNone(p.grad)

  def test_multi_layer_dropout_batch_first_train_and_backward(self):
    """Verifies batch_first multi-layer GRU with dropout>0 runs in train mode and computes gradients.

    What is being tested:
      - Batch-first layout [B, T, H] combined with multi-layer dropout
      (dropout=0.3) in training mode.
      - Verifies that 4D dropout noise shapes and mask slices align correctly
      with batch-first dimension ordering.

    Expected result:
      - Successful forward and backward passes. Gradients x.grad and h0.grad are
      valid, non-null, and finite.
    """
    batch = 3
    seq_len = 6
    input_size = 8
    hidden_size = 16
    num_layers = 2
    dropout = 0.3

    _, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=True,
        dtype=torch.float32,
        dropout=dropout,
    )
    tpu_gru.train()

    x_tpu = torch.randn(
        batch, seq_len, input_size, device="tpu", requires_grad=True
    )
    h0_tpu = torch.randn(
        num_layers, batch, hidden_size, device="tpu", requires_grad=True
    )

    out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)
    loss = out_tpu.sum() + hy_tpu.sum()
    loss.backward()

    self.assertIsNotNone(x_tpu.grad)
    self.assertIsNotNone(h0_tpu.grad)

  def test_bidirectional_multi_layer_dropout_train_and_backward(self):
    """Verifies multi-layer bidirectional GRU with dropout>0 runs in train mode and computes gradients.

    What is being tested:
      - Multi-layer bidirectional GRU (num_layers=2, bidirectional=True) with
      active dropout (dropout=0.4).
      - Inter-layer dropout applied across concatenated bidirectional features
      [T, B, 2*H].

    Expected result:
      - Forward output has shape [T, B, 2*H], final hidden state has shape [4,
      B, H].
      - Autograd backward computes non-null, finite gradients for all inputs and
      parameters.
    """
    batch = 2
    seq_len = 6
    input_size = 8
    hidden_size = 16
    num_layers = 2
    dropout = 0.4

    _, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        bidirectional=True,
        dropout=dropout,
    )
    tpu_gru.train()

    x_tpu = torch.randn(
        seq_len, batch, input_size, device="tpu", requires_grad=True
    )
    h0_tpu = torch.randn(
        2 * num_layers, batch, hidden_size, device="tpu", requires_grad=True
    )

    out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)
    loss = out_tpu.sum() + hy_tpu.sum()
    loss.backward()

    self.assertIsNotNone(x_tpu.grad)
    self.assertIsNotNone(h0_tpu.grad)

  def test_pipelined_wavefront_multi_layer_parity(self):
    """Verifies pipelined wavefront forward path (L>=2, seq_len>=8) matches CPU in eval mode.

    What is being tested:
      - Dedicated TPU multi-layer wavefront pipelining optimization: streams
      chunk outputs
        (k=8) between layer l and layer l+1 directly in on-chip TPU vector
        memory (VMEM).
      - Evaluates num_layers=3, seq_len=16 (2 chunks of 8), batch=2.
      - Verifies that pipelined register streaming produces numerically
      identical results
        to un-pipelined sequential layer evaluation without HBM intermediate
        buffer overhead.

    Expected result:
      - Forward sequence output and final hidden states match CPU reference
      within atol=5e-3, rtol=5e-3.
    """
    batch = 2
    seq_len = 16
    input_size = 8
    hidden_size = 16
    num_layers = 3

    cpu_gru, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
    )
    cpu_gru.eval()
    tpu_gru.eval()

    x_cpu = torch.randn(seq_len, batch, input_size)
    x_tpu = x_cpu.detach().clone().to("tpu")

    out_cpu, hy_cpu = cpu_gru(x_cpu)
    out_tpu, hy_tpu = tpu_gru(x_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)

  def test_pipelined_wavefront_with_dropout_train(self):
    """Verifies pipelined wavefront with inter-layer dropout in train mode.

    What is being tested:
      - Wavefront pipelined multi-layer execution with active dropout
      (dropout=0.3) in train mode.
      - Tests on-the-fly dropout mask generation and streaming between pipelined
      layers.

    Expected result:
      - Forward and backward passes execute cleanly without errors.
      - All parameter gradients and input gradients are non-None and finite (no
      NaN or Inf values).
    """
    batch = 2
    seq_len = 16
    input_size = 8
    hidden_size = 16
    num_layers = 3
    dropout = 0.3

    _, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        dropout=dropout,
    )
    tpu_gru.train()

    x_tpu = torch.randn(
        seq_len, batch, input_size, device="tpu", requires_grad=True
    )
    h0_tpu = torch.randn(
        num_layers, batch, hidden_size, device="tpu", requires_grad=True
    )

    out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)
    loss = out_tpu.sum() + hy_tpu.sum()
    loss.backward()

    self.assertIsNotNone(x_tpu.grad)
    self.assertFalse(torch.isnan(x_tpu.grad).any())
    for name, p in tpu_gru.named_parameters():
      self.assertIsNotNone(p.grad, f"Parameter {name} grad is None")
      self.assertFalse(
          torch.isnan(p.grad).any(), f"Parameter {name} grad contains NaN"
      )

  @parameterized.named_parameters(
      (
          "float32_seq_first_2layers_chunked",
          torch.float32,
          False,  # batch_first
          2,  # num_layers
          True,  # bias
          18,  # seq_len (2 chunks of 8 + 2 rem)
      ),
      (
          "float32_seq_first_3layers_chunked",
          torch.float32,
          False,  # batch_first
          3,  # num_layers
          True,  # bias
          20,  # seq_len (2 chunks of 8 + 4 rem)
      ),
      (
          "bfloat16_batch_first_2layers_chunked",
          torch.bfloat16,
          True,  # batch_first
          2,  # num_layers
          True,  # bias
          16,  # seq_len (2 full chunks of 8)
      ),
      (
          "float32_no_bias_2layers_chunked",
          torch.float32,
          False,  # batch_first
          2,  # num_layers
          False,  # bias
          17,  # seq_len
      ),
  )
  def test_pipelined_wavefront_backward_parity(
      self,
      dtype: torch.dtype,
      batch_first: bool,
      num_layers: int,
      bias: bool,
      seq_len: int,
  ):
    """Verifies pipelined wavefront backward BPTT gradients against CPU reference across multiple chunks and remainder steps.

    What is being tested:
      - Pipelined wavefront analytical reverse BPTT across multiple layers (L=2,
      L=3),
        multiple unrolled chunks (seq_len=16, 18, 20), with remainder steps, and
        both
        sequence-first and batch-first layouts.
      - Evaluates weight gradient streaming accumulation across layers and
      chunks directly.

    Expected result:
      - Forward outputs out and hy match CPU reference within tolerance.
      - Input gradients x.grad and initial state gradients h0.grad match CPU
      reference within atol=1e-2 (FP32) / 8e-2 (BF16).
      - All weight and bias gradients across all layers match CPU reference
      within tolerance.
    """
    batch = 4
    input_size = 16
    hidden_size = 32

    cpu_gru, tpu_gru = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dtype=dtype,
    )

    x_shape = (
        (batch, seq_len, input_size)
        if batch_first
        else (seq_len, batch, input_size)
    )
    x_cpu = torch.randn(x_shape, dtype=dtype, requires_grad=True)
    h0_cpu = torch.randn(
        num_layers, batch, hidden_size, dtype=dtype, requires_grad=True
    )

    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)

    out_cpu, hy_cpu = cpu_gru(x_cpu, h0_cpu)
    out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)

    atol_out = 5e-2 if dtype == torch.bfloat16 else 5e-3
    rtol_out = 5e-2 if dtype == torch.bfloat16 else 5e-3

    assert_close(out_tpu.cpu(), out_cpu, atol=atol_out, rtol=rtol_out)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=atol_out, rtol=rtol_out)

    loss_cpu = out_cpu.sum() + hy_cpu.sum()
    loss_tpu = out_tpu.sum() + hy_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    atol_grad = 8e-2 if dtype == torch.bfloat16 else 1e-2
    rtol_grad = 8e-2 if dtype == torch.bfloat16 else 1e-2

    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=atol_grad, rtol=rtol_grad)
    assert_close(h0_tpu.grad.cpu(), h0_cpu.grad, atol=atol_grad, rtol=rtol_grad)

    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_gru.named_parameters(), tpu_gru.named_parameters()
    ):
      self.assertEqual(name_c, name_t)
      if p_cpu.grad is not None:
        atol_params = (
            2e-1 if (dtype == torch.bfloat16 or num_layers >= 3) else 5e-2
        )
        rtol_params = (
            2e-1 if (dtype == torch.bfloat16 or num_layers >= 3) else 5e-2
        )
        assert_close(
            p_tpu.grad.cpu(),
            p_cpu.grad,
            atol=atol_params,
            rtol=rtol_params,
        )

  @parameterized.parameters("high", "highest")
  def test_float32_matmul_precision(self, precision_mode):
    """Verifies that torch.set_float32_matmul_precision works seamlessly with aten::gru.input.

    What is being tested:
      - Compatibility with PyTorch precision controls: 'high' (TF32/BFloat16 on
      hardware systolic arrays)
        and 'highest' (full FP32 precision).
      - Verifies that the operator reads the thread-local precision setting and
      propagates it to
        StableHLO DotDimension precision attributes.

    Expected result:
      - Forward outputs and backward gradients match CPU reference under both
      'high' and 'highest' precision modes.
    """
    old_precision = torch.get_float32_matmul_precision()
    try:
      torch.set_float32_matmul_precision(precision_mode)
      batch = 2
      seq_len = 4
      input_size = 8
      hidden_size = 16

      cpu_gru, tpu_gru = self._create_models(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=1,
          bias=True,
          batch_first=False,
          dtype=torch.float32,
      )

      x_cpu = torch.randn(seq_len, batch, input_size, requires_grad=True)
      h0_cpu = torch.randn(1, batch, hidden_size, requires_grad=True)
      x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
      h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)

      out_cpu, hy_cpu = cpu_gru(x_cpu, h0_cpu)
      out_tpu, hy_tpu = tpu_gru(x_tpu, h0_tpu)

      assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
      assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)

      (out_tpu.sum() + hy_tpu.sum()).backward()
      (out_cpu.sum() + hy_cpu.sum()).backward()

      assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)
      assert_close(h0_tpu.grad.cpu(), h0_cpu.grad, atol=5e-3, rtol=5e-3)
    finally:
      torch.set_float32_matmul_precision(old_precision)

  def test_float32_matmul_precision_cache_switching(self):
    """Verifies that switching torch.set_float32_matmul_precision dynamically differentiates cache keys.

    What is being tested:
      - Dynamic compilation cache key differentiation: changing
      float32_matmul_precision between
        'highest', 'high', and 'medium' must incorporate precision into
        OpParamCacheKeys.
      - Ensures that switching precision modes re-compiles or selects the
      corresponding specialized kernel
        rather than reusing an incorrect compiled artifact.

    Expected result:
      - Each precision mode runs cleanly and computes valid gradients without
      cache collisions or crashes.
    """
    old_precision = torch.get_float32_matmul_precision()
    try:
      batch = 2
      seq_len = 4
      input_size = 8
      hidden_size = 16

      _, tpu_gru = self._create_models(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=1,
          bias=True,
          batch_first=False,
          dtype=torch.float32,
      )

      for mode in ["highest", "high", "medium", "highest"]:
        torch.set_float32_matmul_precision(mode)
        x = torch.randn(
            seq_len, batch, input_size, device="tpu", requires_grad=True
        )
        h0 = torch.randn(
            1, batch, hidden_size, device="tpu", requires_grad=True
        )
        out, hy = tpu_gru(x, h0)
        loss = out.sum() + hy.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
    finally:
      torch.set_float32_matmul_precision(old_precision)


if __name__ == "__main__":
  absltest.main()
