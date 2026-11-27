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

"""Numerical correctness and parity tests for fused StableHLO LSTM on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._dynamo import testing as dynamo_testing
import torch_tpu  # pylint: disable=unused-import  # ruff: ignore[unused-import]
from torch_tpu._internal.utils.test_utils import assert_close
from tests import seed_test_utils


class LstmOpTest(seed_test_utils.RepeatableTest):
  """Unit and integration tests for TPU fused StableHLO LSTM (aten::lstm.input).

  Verifies numerical parity, autograd gradient calculation, corner-case input
  handling, strided memory layouts, and error validations against PyTorch's
  canonical CPU LSTM reference implementation.
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
      proj_size: int = 0,
      dropout: float = 0.0,
  ):
    """Instantiates CPU reference and TPU target models with identical weights.

    Copies initialized weights directly from the CPU module to the TPU module
    so that differences in outputs and gradients reflect only numerical and
    arithmetic variations between CPU and TPU execution.
    """
    cpu_lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dtype=dtype,
        bidirectional=bidirectional,
        proj_size=proj_size,
        dropout=dropout,
    )
    tpu_lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dtype=dtype,
        bidirectional=bidirectional,
        proj_size=proj_size,
        dropout=dropout,
        device="tpu",
    )
    with torch.no_grad():
      for cpu_p, tpu_p in zip(cpu_lstm.parameters(), tpu_lstm.parameters()):
        tpu_p.copy_(cpu_p.to("tpu"))
    return cpu_lstm, tpu_lstm

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

    Tests full round-trip execution across different data types (FP32, BF16),
    tensor layouts (sequence-first vs batch-first), multi-layer stacking, and
    bias options.
    """
    batch = 4
    seq_len = 8
    input_size = 16
    hidden_size = 32

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dtype=dtype,
    )

    # Prepare inputs matching the batch layout configuration.
    x_shape = (
        (batch, seq_len, input_size)
        if batch_first
        else (seq_len, batch, input_size)
    )
    x_cpu = torch.randn(x_shape, dtype=dtype, requires_grad=True)
    h0_cpu = torch.randn(
        num_layers, batch, hidden_size, dtype=dtype, requires_grad=True
    )
    c0_cpu = torch.randn(
        num_layers, batch, hidden_size, dtype=dtype, requires_grad=True
    )

    # Clone exact inputs to TPU with gradient tracking enabled.
    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)
    c0_tpu = c0_cpu.detach().clone().to("tpu").requires_grad_(True)

    # 1. Forward Pass Evaluation
    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu, (h0_cpu, c0_cpu))
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu, (h0_tpu, c0_tpu))

    # Numerical tolerances: FP32 uses 5e-3; BF16 has 8-bit mantissa with ~7.8e-3 precision.
    atol = 5e-3 if dtype == torch.float32 else 5e-2
    rtol = 5e-3 if dtype == torch.float32 else 5e-2

    # Verify forward sequence outputs, final hidden states, and final cell states.
    assert_close(out_tpu.cpu(), out_cpu, atol=atol, rtol=rtol, check_dtype=True)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=atol, rtol=rtol, check_dtype=True)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=atol, rtol=rtol, check_dtype=True)

    # 2. Backward Pass Evaluation (BPTT)
    # Define scalar loss combining all recurrent output tensors.
    loss_cpu = out_cpu.sum() + hy_cpu.sum() + cy_cpu.sum()
    loss_tpu = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    # Verify input and initial state gradients (dx, dh0, dc0).
    assert_close(
        x_tpu.grad.cpu(), x_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )
    assert_close(
        h0_tpu.grad.cpu(), h0_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )
    assert_close(
        c0_tpu.grad.cpu(), c0_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )

    # Weight gradients accumulate across all batch elements and sequence timesteps
    # via TPU systolic GEMMs (TF32/BF16 arithmetic in MXU), leading to higher numerical
    # variance against IEEE FP32 CPU math. We check parameters with atol=5e-2 for FP32, 1e-1 for BF16.
    atol_params = 5e-2 if dtype == torch.float32 else 1e-1
    rtol_params = 5e-2
    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters()
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
    """Verifies recurrence when initial states (h_0, c_0) are not provided (hx=None).

    PyTorch nn.LSTM initializes h_0 and c_0 to zeros when omitted. This test
    verifies that the TPU kernel creates zero states properly, achieves forward
    parity, allows gradient propagation back to input x, and updates parameter
    gradients.
    """
    batch = 2
    seq_len = 4
    input_size = 8
    hidden_size = 16

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
    )

    x_cpu = torch.randn(seq_len, batch, input_size, requires_grad=True)
    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)

    # Forward pass with omitted initial hidden/cell states
    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu)
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=2e-3, rtol=2e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=2e-3, rtol=2e-3)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=2e-3, rtol=2e-3)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_tpu = out_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    # Input gradient verification
    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=2e-3, rtol=2e-3)

    # Verify parameter gradients match when hx is omitted (None)
    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters()
    ):
      self.assertEqual(name_c, name_t)
      self.assertIsNotNone(p_cpu.grad)
      self.assertIsNotNone(p_tpu.grad)
      assert_close(
          p_tpu.grad.cpu(),
          p_cpu.grad,
          atol=5e-2,
          rtol=5e-2,
          check_dtype=True,
      )

  @parameterized.named_parameters(
      ("single_step", 1, 4, 16, 32, 1, torch.float32, True),
      ("single_batch", 8, 1, 16, 32, 1, torch.float32, True),
      ("minimal_dims", 4, 2, 1, 1, 1, torch.float32, True),
      ("contracting_dims", 4, 2, 64, 16, 1, torch.float32, True),
      ("three_layers", 4, 2, 16, 16, 3, torch.float32, True),
      ("bfloat16_no_bias", 4, 2, 16, 16, 1, torch.bfloat16, False),
  )
  def test_boundary_and_corner_dimensions(
      self,
      seq_len: int,
      batch: int,
      input_size: int,
      hidden_size: int,
      num_layers: int,
      dtype: torch.dtype,
      bias: bool,
  ):
    """Tests boundary dimensions and corner configurations.

    Covers:
    - single_step (seq_len=1): Degenerate single-step sequence with no recurrent
    chaining.
    - single_batch (batch=1): Edge dimension exercising TPU systolic array
    padding.
    - minimal_dims (input_size=1, hidden_size=1): Smallest non-trivial
    scalar-like feature sizes.
    - contracting_dims (input_size=64, hidden_size=16): Large feature reduction
    across gates.
    - three_layers (num_layers=3): Deep multi-layer recurrence and inter-layer
    backprop.
    - bfloat16_no_bias: Mixed precision execution with zero bias additions.
    """
    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=False,
        dtype=dtype,
    )

    x_cpu = torch.randn(
        seq_len, batch, input_size, dtype=dtype, requires_grad=True
    )
    h0_cpu = torch.randn(
        num_layers, batch, hidden_size, dtype=dtype, requires_grad=True
    )
    c0_cpu = torch.randn(
        num_layers, batch, hidden_size, dtype=dtype, requires_grad=True
    )

    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)
    c0_tpu = c0_cpu.detach().clone().to("tpu").requires_grad_(True)

    # Forward pass
    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu, (h0_cpu, c0_cpu))
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu, (h0_tpu, c0_tpu))

    atol = 1e-2 if dtype == torch.float32 else 5e-2
    rtol = 1e-2 if dtype == torch.float32 else 5e-2

    assert_close(out_tpu.cpu(), out_cpu, atol=atol, rtol=rtol, check_dtype=True)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=atol, rtol=rtol, check_dtype=True)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=atol, rtol=rtol, check_dtype=True)

    # Backward pass
    loss_cpu = out_cpu.sum() + hy_cpu.sum() + cy_cpu.sum()
    loss_tpu = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    # Input gradients
    assert_close(
        x_tpu.grad.cpu(), x_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )
    assert_close(
        h0_tpu.grad.cpu(), h0_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )
    assert_close(
        c0_tpu.grad.cpu(), c0_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )

    # Parameter gradients
    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters()
    ):
      self.assertEqual(name_c, name_t)
      self.assertIsNotNone(p_cpu.grad)
      self.assertIsNotNone(p_tpu.grad)
      assert_close(
          p_tpu.grad.cpu(),
          p_cpu.grad,
          atol=5e-2,
          rtol=5e-2,
          check_dtype=True,
      )

  def test_partial_gradients(self):
    """Verifies autograd pruning when only a subset of inputs require gradients.

    In practical training loops (e.g. stateful recurrence or fine-tuning),
    initial hidden states may be fixed buffers that do not require gradient.
    This test verifies that:
    1. Gradients are computed correctly for tensors with requires_grad=True.
    2. Inputs with requires_grad=False receive no gradient (remain None).
    """
    batch = 2
    seq_len = 4
    input_size = 8
    hidden_size = 16

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
    )

    # Only x requires gradient; h0 and c0 do not
    x_cpu = torch.randn(seq_len, batch, input_size, requires_grad=True)
    h0_cpu = torch.randn(1, batch, hidden_size, requires_grad=False)
    c0_cpu = torch.randn(1, batch, hidden_size, requires_grad=False)
    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu")
    c0_tpu = c0_cpu.detach().clone().to("tpu")

    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu, (h0_cpu, c0_cpu))
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu, (h0_tpu, c0_tpu))
    (out_cpu.sum() + hy_cpu.sum() + cy_cpu.sum()).backward()
    (out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()).backward()

    # Verify input gradient matches CPU reference
    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)
    # Verify initial states have no gradients allocated
    self.assertIsNone(h0_tpu.grad)
    self.assertIsNone(c0_tpu.grad)

  def test_non_contiguous_inputs(self):
    """Verifies forward and backward correctness with non-contiguous strided inputs.

    Tests that sliced strided views (e.g., raw_input[::2]) execute correctly on
    TPU without triggering CompositeOpCheck decomposition or unexpected memory
    copies.
    """
    batch = 4
    seq_len = 6
    input_size = 8
    hidden_size = 16

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
    )

    # Create non-contiguous strided input via strided slicing [::2]
    raw_cpu = torch.randn(seq_len * 2, batch, input_size)
    x_cpu = raw_cpu[::2].detach().requires_grad_(True)
    self.assertFalse(x_cpu.is_contiguous())

    raw_tpu = raw_cpu.to("tpu")
    x_tpu = raw_tpu[::2].detach().requires_grad_(True)
    self.assertFalse(x_tpu.is_contiguous())

    # Forward pass
    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu)
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=5e-3, rtol=5e-3)

    # Backward pass
    (out_cpu.sum() + hy_cpu.sum() + cy_cpu.sum()).backward()
    (out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()).backward()

    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)

  @parameterized.named_parameters(
      (
          "batch_first_1layer",
          True,  # batch_first
          1,  # num_layers
          8,  # seq_len
          4,  # batch_size
      ),
      (
          "seq_first_1layer",
          False,  # batch_first
          1,  # num_layers
          8,  # seq_len
          4,  # batch_size
      ),
      (
          "batch_first_2layers",
          True,  # batch_first
          2,  # num_layers
          8,  # seq_len
          4,  # batch_size
      ),
      (
          "batch_first_1layer_remainder",
          True,  # batch_first
          1,  # num_layers
          11,  # seq_len with remainder (11 % 8 = 3)
          4,  # batch_size
      ),
      (
          "batch_first_2layers_remainder",
          True,  # batch_first
          2,  # num_layers
          11,  # seq_len with remainder (11 % 8 = 3)
          4,  # batch_size
      ),
  )
  def test_bidirectional_training_backward_parity(
      self, batch_first: bool, num_layers: int, seq_len: int, batch_size: int
  ):
    """Verifies forward and BPTT backward numerical parity for bidirectional LSTM against CPU."""
    input_size = 16
    hidden_size = 32
    cpu_lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True,
        batch_first=batch_first,
    )
    tpu_lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True,
        batch_first=batch_first,
        device="tpu",
    )
    with torch.no_grad():
      for cpu_p, tpu_p in zip(cpu_lstm.parameters(), tpu_lstm.parameters()):
        tpu_p.copy_(cpu_p.to("tpu"))

    if batch_first:
      x_cpu = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
    else:
      x_cpu = torch.randn(seq_len, batch_size, input_size, requires_grad=True)
    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)

    h0_cpu = torch.randn(
        num_layers * 2, batch_size, hidden_size, requires_grad=True
    )
    c0_cpu = torch.randn(
        num_layers * 2, batch_size, hidden_size, requires_grad=True
    )
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)
    c0_tpu = c0_cpu.detach().clone().to("tpu").requires_grad_(True)

    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu, (h0_cpu, c0_cpu))
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu, (h0_tpu, c0_tpu))

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=5e-3, rtol=5e-3)

    loss_cpu = out_cpu.sum() + hy_cpu.sum() + cy_cpu.sum()
    loss_tpu = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=7e-3, rtol=5e-3)
    assert_close(h0_tpu.grad.cpu(), h0_cpu.grad, atol=7e-3, rtol=5e-3)
    assert_close(c0_tpu.grad.cpu(), c0_cpu.grad, atol=7e-3, rtol=5e-3)

    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters()
    ):
      self.assertEqual(name_c, name_t)
      self.assertIsNotNone(p_cpu.grad)
      self.assertIsNotNone(p_tpu.grad)
      assert_close(
          p_tpu.grad.cpu(),
          p_cpu.grad,
          atol=5e-2,
          rtol=5e-2,
          check_dtype=True,
      )

  @parameterized.named_parameters(
      (
          "batch_first_1layer_chunked",
          True,  # batch_first
          1,  # num_layers
          16,  # seq_len >= 8 (chunked)
          4,  # batch_size
      ),
      (
          "seq_first_1layer_chunked",
          False,  # batch_first
          1,  # num_layers
          16,  # seq_len >= 8
          4,  # batch_size
      ),
      (
          "batch_first_2layers_chunked",
          True,  # batch_first
          2,  # num_layers
          16,  # seq_len >= 8
          4,  # batch_size
      ),
      (
          "batch_first_1layer_short",
          True,  # batch_first
          1,  # num_layers
          4,  # seq_len < 8 (static unroll)
          4,  # batch_size
      ),
      (
          "batch_first_1layer_remainder",
          True,  # batch_first
          1,  # num_layers
          11,  # seq_len with remainder (11 % 8 = 3)
          4,  # batch_size
      ),
  )
  def test_bidirectional_eval_parity(
      self, batch_first: bool, num_layers: int, seq_len: int, batch_size: int
  ):
    """Verifies forward numerical parity for bidirectional LSTM against CPU."""
    input_size = 16
    hidden_size = 32
    cpu_lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True,
        batch_first=batch_first,
    )
    tpu_lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True,
        batch_first=batch_first,
        device="tpu",
    )
    with torch.no_grad():
      for cpu_p, tpu_p in zip(cpu_lstm.parameters(), tpu_lstm.parameters()):
        tpu_p.copy_(cpu_p.to("tpu"))

    if batch_first:
      x_cpu = torch.randn(batch_size, seq_len, input_size)
    else:
      x_cpu = torch.randn(seq_len, batch_size, input_size)
    x_tpu = x_cpu.to("tpu")

    h0_cpu = torch.randn(num_layers * 2, batch_size, hidden_size)
    c0_cpu = torch.randn(num_layers * 2, batch_size, hidden_size)
    h0_tpu = h0_cpu.to("tpu")
    c0_tpu = c0_cpu.to("tpu")

    with torch.no_grad():
      out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu, (h0_cpu, c0_cpu))
      out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu, (h0_tpu, c0_tpu))

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=5e-3, rtol=5e-3)

  @parameterized.named_parameters(
      (
          "float32_seq_first_1layer",
          torch.float32,
          False,  # batch_first
          1,  # num_layers
          True,  # bias
          False,  # bidirectional
      ),
      (
          "float32_batch_first_1layer",
          torch.float32,
          True,  # batch_first
          1,  # num_layers
          True,  # bias
          False,  # bidirectional
      ),
      (
          "float32_2layers",
          torch.float32,
          False,  # batch_first
          2,  # num_layers
          True,  # bias
          False,  # bidirectional
      ),
      (
          "float32_bidirectional_1layer",
          torch.float32,
          False,  # batch_first
          1,  # num_layers
          True,  # bias
          True,  # bidirectional
      ),
      (
          "float32_bidirectional_2layers",
          torch.float32,
          False,  # batch_first
          2,  # num_layers
          True,  # bias
          True,  # bidirectional
      ),
      (
          "float32_batch_first_2layers",
          torch.float32,
          True,  # batch_first
          2,  # num_layers
          True,  # bias
          False,  # bidirectional
      ),
      (
          "float32_batch_first_bidirectional_1layer",
          torch.float32,
          True,  # batch_first
          1,  # num_layers
          True,  # bias
          True,  # bidirectional
      ),
      (
          "float32_no_bias_1layer",
          torch.float32,
          False,  # batch_first
          1,  # num_layers
          False,  # bias
          False,  # bidirectional
      ),
      (
          "bfloat16_1layer",
          torch.bfloat16,
          False,  # batch_first
          1,  # num_layers
          True,  # bias
          False,  # bidirectional
      ),
      (
          "bfloat16_bidirectional",
          torch.bfloat16,
          False,  # batch_first
          1,  # num_layers
          True,  # bias
          True,  # bidirectional
      ),
  )
  def test_projected_lstm_parity(
      self,
      dtype: torch.dtype,
      batch_first: bool,
      num_layers: int,
      bias: bool,
      bidirectional: bool,
  ):
    """Verifies forward outputs and BPTT backward gradients for Projected LSTM (proj_size > 0)."""
    batch = 4
    seq_len = 8
    input_size = 16
    hidden_size = 32
    proj_size = 12
    num_directions = 2 if bidirectional else 1

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dtype=dtype,
        bidirectional=bidirectional,
        proj_size=proj_size,
    )

    x_shape = (
        (batch, seq_len, input_size)
        if batch_first
        else (seq_len, batch, input_size)
    )
    x_cpu = torch.randn(x_shape, dtype=dtype, requires_grad=True)
    h0_cpu = torch.randn(
        num_layers * num_directions,
        batch,
        proj_size,
        dtype=dtype,
        requires_grad=True,
    )
    c0_cpu = torch.randn(
        num_layers * num_directions,
        batch,
        hidden_size,
        dtype=dtype,
        requires_grad=True,
    )

    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)
    c0_tpu = c0_cpu.detach().clone().to("tpu").requires_grad_(True)

    # 1. Forward Pass Evaluation
    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu, (h0_cpu, c0_cpu))
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu, (h0_tpu, c0_tpu))

    atol = 5e-3 if dtype == torch.float32 else 5e-2
    rtol = 5e-3 if dtype == torch.float32 else 5e-2

    assert_close(out_tpu.cpu(), out_cpu, atol=atol, rtol=rtol, check_dtype=True)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=atol, rtol=rtol, check_dtype=True)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=atol, rtol=rtol, check_dtype=True)

    # 2. Backward Pass Evaluation (BPTT)
    loss_cpu = out_cpu.sum() + hy_cpu.sum() + cy_cpu.sum()
    loss_tpu = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    # Verify input and state gradients
    assert_close(
        x_tpu.grad.cpu(), x_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )
    assert_close(
        h0_tpu.grad.cpu(), h0_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )
    assert_close(
        c0_tpu.grad.cpu(), c0_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )

    # Verify parameter gradients (including weight_hr)
    atol_params = 5e-2 if dtype == torch.float32 else 1e-1
    rtol_params = 5e-2
    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters()
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

  def test_projected_lstm_default_initial_states(self):
    """Verifies projected recurrence when initial states are omitted (hx=None)."""
    batch = 2
    seq_len = 4
    input_size = 8
    hidden_size = 16
    proj_size = 10

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        proj_size=proj_size,
    )

    x_cpu = torch.randn(seq_len, batch, input_size, requires_grad=True)
    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)

    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu)
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=2e-3, rtol=2e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=2e-3, rtol=2e-3)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=2e-3, rtol=2e-3)

    loss_cpu = out_cpu.sum() + hy_cpu.sum() + cy_cpu.sum()
    loss_tpu = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=2e-3, rtol=2e-3)
    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters()
    ):
      self.assertEqual(name_c, name_t)
      assert_close(p_tpu.grad.cpu(), p_cpu.grad, atol=5e-2, rtol=5e-2)

  def test_invalid_tensor_rank_raises(self):
    """Verifies that passing a 4D tensor input raises an exception."""
    tpu_lstm = torch.nn.LSTM(input_size=8, hidden_size=16, device="tpu")
    # 4D input instead of 3D (or unbatched 2D)
    x_4d = torch.randn(2, 4, 8, 16, device="tpu")
    with self.assertRaises(  # ASSERT_RAISES_OK=lstm validation
        (RuntimeError, ValueError)
    ):
      tpu_lstm(x_4d)

  def test_zero_sequence_length_raises(self):
    """Verifies that an input tensor with seq_len=0 raises RuntimeError."""
    tpu_lstm = torch.nn.LSTM(input_size=8, hidden_size=16, device="tpu")
    x_zero_seq = torch.randn(0, 4, 8, device="tpu")
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=lstm validation
      tpu_lstm(x_zero_seq)

  def test_mismatched_hidden_batch_raises(self):
    """Verifies that a batch size mismatch between input x and h_0 raises RuntimeError."""
    tpu_lstm = torch.nn.LSTM(input_size=8, hidden_size=16, device="tpu")
    x = torch.randn(4, 2, 8, device="tpu")
    h0_bad = torch.randn(1, 3, 16, device="tpu")  # batch=3 != 2
    c0 = torch.randn(1, 2, 16, device="tpu")
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=lstm validation
      tpu_lstm(x, (h0_bad, c0))

  def test_mismatched_cell_shape_raises(self):
    """Verifies that a hidden dimension mismatch between h_0 and c_0 raises RuntimeError."""
    tpu_lstm = torch.nn.LSTM(input_size=8, hidden_size=16, device="tpu")
    x = torch.randn(4, 2, 8, device="tpu")
    h0 = torch.randn(1, 2, 16, device="tpu")
    c0_bad = torch.randn(1, 2, 8, device="tpu")  # hidden=8 != 16
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=lstm validation
      tpu_lstm(x, (h0, c0_bad))

  def test_multi_layer_dropout_eval_matches_cpu(self):
    """Verifies that multi-layer LSTM with dropout>0 in eval mode matches CPU."""
    batch = 2
    seq_len = 8
    input_size = 8
    hidden_size = 16
    num_layers = 3
    dropout = 0.3

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        dropout=dropout,
    )
    cpu_lstm.eval()
    tpu_lstm.eval()

    x_cpu = torch.randn(seq_len, batch, input_size)
    x_tpu = x_cpu.detach().clone().to("tpu")

    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu)
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=5e-3, rtol=5e-3)

  def test_multi_layer_dropout_train_and_backward(self):
    """Verifies multi-layer LSTM with dropout>0 runs in train mode and computes gradients."""
    batch = 2
    seq_len = 8
    input_size = 8
    hidden_size = 16
    num_layers = 2
    dropout = 0.5

    _, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        dropout=dropout,
    )
    tpu_lstm.train()

    x_tpu = torch.randn(
        seq_len, batch, input_size, device="tpu", requires_grad=True
    )
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu)

    self.assertFalse(torch.isnan(out_tpu).any())
    self.assertFalse(torch.isnan(hy_tpu).any())
    self.assertFalse(torch.isnan(cy_tpu).any())

    loss = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss.backward()

    self.assertIsNotNone(x_tpu.grad)
    self.assertFalse(torch.isnan(x_tpu.grad).any())
    for name, p in tpu_lstm.named_parameters():
      self.assertIsNotNone(p.grad, f"Parameter {name} grad is None")
      self.assertFalse(
          torch.isnan(p.grad).any(), f"Parameter {name} grad contains NaN"
      )

  def test_pipelined_wavefront_multi_layer_parity(self):
    """Verifies pipelined wavefront forward path (L>=2, seq_len>=8) matches CPU in eval mode."""
    batch = 2
    seq_len = 16
    input_size = 8
    hidden_size = 16
    num_layers = 3

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
    )
    cpu_lstm.eval()
    tpu_lstm.eval()

    x_cpu = torch.randn(seq_len, batch, input_size)
    x_tpu = x_cpu.detach().clone().to("tpu")

    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu)
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu)

    assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=5e-3, rtol=5e-3)

  def test_pipelined_wavefront_with_dropout_train(self):
    """Verifies pipelined wavefront with inter-layer dropout in train mode."""
    batch = 2
    seq_len = 16
    input_size = 8
    hidden_size = 16
    num_layers = 3
    dropout = 0.3

    _, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        dropout=dropout,
    )
    tpu_lstm.train()

    x_tpu = torch.randn(
        seq_len, batch, input_size, device="tpu", requires_grad=True
    )
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu)

    self.assertFalse(torch.isnan(out_tpu).any())
    self.assertFalse(torch.isnan(hy_tpu).any())
    self.assertFalse(torch.isnan(cy_tpu).any())

    loss = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss.backward()

    self.assertIsNotNone(x_tpu.grad)
    self.assertFalse(torch.isnan(x_tpu.grad).any())
    for name, p in tpu_lstm.named_parameters():
      self.assertIsNotNone(p.grad, f"Parameter {name} grad is None")
      self.assertFalse(
          torch.isnan(p.grad).any(), f"Parameter {name} grad contains NaN"
      )

  def test_multi_layer_dropout_batch_first_train_and_backward(self):
    """Verifies batch_first multi-layer LSTM with dropout>0 runs in train mode and computes gradients."""
    batch = 2
    seq_len = 16
    input_size = 8
    hidden_size = 16
    num_layers = 3
    dropout = 0.3

    _, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=True,
        dtype=torch.float32,
        dropout=dropout,
    )
    tpu_lstm.train()

    x_tpu = torch.randn(
        batch, seq_len, input_size, device="tpu", requires_grad=True
    )
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu)

    self.assertFalse(torch.isnan(out_tpu).any())
    self.assertFalse(torch.isnan(hy_tpu).any())
    self.assertFalse(torch.isnan(cy_tpu).any())

    loss = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss.backward()

    self.assertIsNotNone(x_tpu.grad)
    self.assertFalse(torch.isnan(x_tpu.grad).any())
    for name, p in tpu_lstm.named_parameters():
      self.assertIsNotNone(p.grad, f"Parameter {name} grad is None")
      self.assertFalse(
          torch.isnan(p.grad).any(), f"Parameter {name} grad contains NaN"
      )

  def test_bidirectional_multi_layer_dropout_train_and_backward(self):
    """Verifies multi-layer bidirectional LSTM with dropout>0 runs in train mode and computes gradients."""
    batch = 2
    seq_len = 12
    input_size = 8
    hidden_size = 16
    num_layers = 2
    dropout = 0.4

    _, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        bidirectional=True,
        dtype=torch.float32,
        dropout=dropout,
    )
    tpu_lstm.train()

    x_tpu = torch.randn(
        seq_len, batch, input_size, device="tpu", requires_grad=True
    )
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu)

    self.assertFalse(torch.isnan(out_tpu).any())
    self.assertFalse(torch.isnan(hy_tpu).any())
    self.assertFalse(torch.isnan(cy_tpu).any())

    loss = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss.backward()

    self.assertIsNotNone(x_tpu.grad)
    self.assertFalse(torch.isnan(x_tpu.grad).any())
    for name, p in tpu_lstm.named_parameters():
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
          0,  # proj_size
      ),
      (
          "float32_seq_first_3layers_chunked",
          torch.float32,
          False,  # batch_first
          3,  # num_layers
          True,  # bias
          20,  # seq_len (2 chunks of 8 + 4 rem)
          0,  # proj_size
      ),
      (
          "bfloat16_batch_first_2layers_chunked",
          torch.bfloat16,
          True,  # batch_first
          2,  # num_layers
          True,  # bias
          16,  # seq_len (2 full chunks of 8)
          0,  # proj_size
      ),
      (
          "float32_projected_2layers_chunked",
          torch.float32,
          False,  # batch_first
          2,  # num_layers
          True,  # bias
          18,  # seq_len
          12,  # proj_size
      ),
      (
          "float32_no_bias_2layers_chunked",
          torch.float32,
          False,  # batch_first
          2,  # num_layers
          False,  # bias
          17,  # seq_len
          0,  # proj_size
      ),
  )
  def test_pipelined_wavefront_backward_parity(
      self,
      dtype: torch.dtype,
      batch_first: bool,
      num_layers: int,
      bias: bool,
      seq_len: int,
      proj_size: int = 0,
  ):
    """Verifies pipelined wavefront backward BPTT gradients against CPU reference across multiple chunks and remainder steps."""
    batch = 4
    input_size = 16
    hidden_size = 32
    out_h = proj_size if proj_size > 0 else hidden_size

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dtype=dtype,
        proj_size=proj_size,
    )

    x_shape = (
        (batch, seq_len, input_size)
        if batch_first
        else (seq_len, batch, input_size)
    )
    x_cpu = torch.randn(x_shape, dtype=dtype, requires_grad=True)
    h0_cpu = torch.randn(
        num_layers, batch, out_h, dtype=dtype, requires_grad=True
    )
    c0_cpu = torch.randn(
        num_layers, batch, hidden_size, dtype=dtype, requires_grad=True
    )

    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)
    c0_tpu = c0_cpu.detach().clone().to("tpu").requires_grad_(True)

    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu, (h0_cpu, c0_cpu))
    out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu, (h0_tpu, c0_tpu))

    atol = 5e-3 if dtype == torch.float32 else 5e-2
    rtol = 5e-3 if dtype == torch.float32 else 5e-2

    assert_close(out_tpu.cpu(), out_cpu, atol=atol, rtol=rtol, check_dtype=True)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=atol, rtol=rtol, check_dtype=True)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=atol, rtol=rtol, check_dtype=True)

    loss_cpu = out_cpu.sum() + hy_cpu.sum() + cy_cpu.sum()
    loss_tpu = out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()
    loss_cpu.backward()
    loss_tpu.backward()

    assert_close(
        x_tpu.grad.cpu(), x_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )
    assert_close(
        h0_tpu.grad.cpu(), h0_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )
    assert_close(
        c0_tpu.grad.cpu(), c0_cpu.grad, atol=atol, rtol=rtol, check_dtype=True
    )

    atol_params = 5e-2 if dtype == torch.float32 else 1e-1
    rtol_params = 5e-2
    for (name_c, p_cpu), (name_t, p_tpu) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters()
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

  @parameterized.parameters("high", "highest")
  def test_float32_matmul_precision(self, precision_mode):
    """Verifies that torch.set_float32_matmul_precision ('high' and 'highest')

    functions seamlessly with aten::lstm.input forward and backward passes,
    producing correct outputs and gradients without error.
    """
    old_precision = torch.get_float32_matmul_precision()
    try:
      torch.set_float32_matmul_precision(precision_mode)
      batch = 2
      seq_len = 4
      input_size = 8
      hidden_size = 16

      cpu_lstm, tpu_lstm = self._create_models(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=1,
          bias=True,
          batch_first=False,
          dtype=torch.float32,
      )

      x_cpu = torch.randn(seq_len, batch, input_size, requires_grad=True)
      h0_cpu = torch.randn(1, batch, hidden_size, requires_grad=True)
      c0_cpu = torch.randn(1, batch, hidden_size, requires_grad=True)
      x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
      h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)
      c0_tpu = c0_cpu.detach().clone().to("tpu").requires_grad_(True)

      out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu, (h0_cpu, c0_cpu))
      out_tpu, (hy_tpu, cy_tpu) = tpu_lstm(x_tpu, (h0_tpu, c0_tpu))

      assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
      assert_close(hy_tpu.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)
      assert_close(cy_tpu.cpu(), cy_cpu, atol=5e-3, rtol=5e-3)

      (out_tpu.sum() + hy_tpu.sum() + cy_tpu.sum()).backward()
      (out_cpu.sum() + hy_cpu.sum() + cy_cpu.sum()).backward()

      assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)
      assert_close(h0_tpu.grad.cpu(), h0_cpu.grad, atol=5e-3, rtol=5e-3)
      assert_close(c0_tpu.grad.cpu(), c0_cpu.grad, atol=5e-3, rtol=5e-3)
    finally:
      torch.set_float32_matmul_precision(old_precision)

  def test_float32_matmul_precision_cache_switching(self):
    """Verifies that switching torch.set_float32_matmul_precision dynamically

    correctly invalidates/differentiates the OpParamCacheKeys cache.
    """
    old_precision = torch.get_float32_matmul_precision()
    try:
      batch = 2
      seq_len = 4
      input_size = 8
      hidden_size = 16

      _, tpu_lstm = self._create_models(
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
        c0 = torch.randn(
            1, batch, hidden_size, device="tpu", requires_grad=True
        )
        out, (hy, cy) = tpu_lstm(x, (h0, c0))
        loss = out.sum() + hy.sum() + cy.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
    finally:
      torch.set_float32_matmul_precision(old_precision)

  @parameterized.named_parameters(
      ("f32_seq_first", torch.float32, False, 1, False, 0),
      ("f32_batch_first_2layer_bidir", torch.float32, True, 2, True, 0),
      ("f32_batch_first_proj", torch.float32, True, 2, True, 6),
      ("bf16_batch_first_bidir", torch.bfloat16, True, 2, True, 0),
  )
  def test_torch_compile_forward_backward(
      self,
      dtype: torch.dtype,
      batch_first: bool,
      num_layers: int,
      bidirectional: bool,
      proj_size: int,
  ):
    """Verifies torch.compile forward and backward parity for nn.LSTM."""
    torch._dynamo.reset()
    batch, seq_len, input_size, hidden_size = 3, 5, 8, 12
    num_directions = 2 if bidirectional else 1
    out_h = proj_size if proj_size > 0 else hidden_size
    tol = 5e-2 if dtype == torch.bfloat16 else 5e-3

    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=batch_first,
        dtype=dtype,
        bidirectional=bidirectional,
        proj_size=proj_size,
    )
    compiled_lstm = torch.compile(tpu_lstm)

    in_shape = (
        (batch, seq_len, input_size)
        if batch_first
        else (seq_len, batch, input_size)
    )
    x_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
    h0_cpu = torch.randn(
        num_layers * num_directions,
        batch,
        out_h,
        dtype=dtype,
        requires_grad=True,
    )
    c0_cpu = torch.randn(
        num_layers * num_directions,
        batch,
        hidden_size,
        dtype=dtype,
        requires_grad=True,
    )
    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
    h0_tpu = h0_cpu.detach().clone().to("tpu").requires_grad_(True)
    c0_tpu = c0_cpu.detach().clone().to("tpu").requires_grad_(True)

    out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_cpu, (h0_cpu, c0_cpu))
    out_tpu, (hy_tpu, cy_tpu) = compiled_lstm(x_tpu, (h0_tpu, c0_tpu))

    assert_close(out_tpu.cpu(), out_cpu, atol=tol, rtol=tol)
    assert_close(hy_tpu.cpu(), hy_cpu, atol=tol, rtol=tol)
    assert_close(cy_tpu.cpu(), cy_cpu, atol=tol, rtol=tol)

    grad_out = torch.randn_like(out_cpu)
    grad_hy = torch.randn_like(hy_cpu)
    grad_cy = torch.randn_like(cy_cpu)
    torch.autograd.backward(
        [out_cpu, hy_cpu, cy_cpu], [grad_out, grad_hy, grad_cy]
    )
    torch.autograd.backward(
        [out_tpu, hy_tpu, cy_tpu],
        [grad_out.to("tpu"), grad_hy.to("tpu"), grad_cy.to("tpu")],
    )

    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=tol, rtol=tol)
    assert_close(h0_tpu.grad.cpu(), h0_cpu.grad, atol=tol, rtol=tol)
    assert_close(c0_tpu.grad.cpu(), c0_cpu.grad, atol=tol, rtol=tol)
    # Weight gradients accumulate over the batch and the sequence in TPU
    # matmuls, so they get the eager tests' parameter tolerances.
    atol_params = 5e-2 if dtype == torch.float32 else 1e-1
    rtol_params = 5e-2
    for (name, cpu_p), (_, tpu_p) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters()
    ):
      self.assertIsNotNone(tpu_p.grad, msg=f"Missing grad for {name}")
      assert_close(
          tpu_p.grad.cpu(), cpu_p.grad, atol=atol_params, rtol=rtol_params
      )

  def test_torch_compile_batch_first_slice_head_and_no_grad(self):
    """Verifies compiled batch_first LSTM followed by sequence slicing and Linear."""
    torch._dynamo.reset()

    class SeqClassifier(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(8, 12, batch_first=True)
        self.head = torch.nn.Linear(12, 4)

      def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

    cpu_mod = SeqClassifier()
    tpu_mod = SeqClassifier().to("tpu")
    with torch.no_grad():
      for cpu_p, tpu_p in zip(cpu_mod.parameters(), tpu_mod.parameters()):
        tpu_p.copy_(cpu_p.to("tpu"))

    compiled_mod = torch.compile(tpu_mod)
    x_cpu = torch.randn(3, 6, 8, requires_grad=True)
    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)

    y_cpu = cpu_mod(x_cpu)
    y_tpu = compiled_mod(x_tpu)
    assert_close(y_tpu.cpu(), y_cpu, atol=5e-3, rtol=5e-3)

    y_cpu.sum().backward()
    y_tpu.sum().backward()
    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)
    for (name, cpu_p), (_, tpu_p) in zip(
        cpu_mod.named_parameters(), tpu_mod.named_parameters()
    ):
      self.assertIsNotNone(tpu_p.grad, msg=f"Missing grad for {name}")
      assert_close(tpu_p.grad.cpu(), cpu_p.grad, atol=5e-3, rtol=5e-3)

    with torch.no_grad():
      x_eval = torch.randn(3, 6, 8)
      assert_close(
          compiled_mod(x_eval.to("tpu")).cpu(),
          cpu_mod(x_eval),
          atol=5e-3,
          rtol=5e-3,
      )
      # AOTAutograd's fake and functional tensors must keep the fused op; only
      # functorch, wrapper subclasses and forward AD use the composite.
      backend = dynamo_testing.AotEagerAndRecordGraphs()
      torch.compile(tpu_mod, backend=backend)(x_eval.to("tpu"))
      self.assertIn(
          "torch.ops.tpu._lstm_input.default", backend.fw_graphs[0].code
      )

  def test_torch_compile_dropout_backward(self):
    """Verifies seeded dropout parity between eager and compiled multi-layer LSTM."""
    torch._dynamo.reset()
    _, tpu_lstm = self._create_models(
        input_size=8,
        hidden_size=12,
        num_layers=2,
        bias=True,
        batch_first=True,
        dtype=torch.float32,
        dropout=0.4,
    )
    tpu_lstm.train()
    compiled_lstm = torch.compile(tpu_lstm)

    x_base = torch.randn(3, 5, 8, device="tpu")
    warmup_out, _ = compiled_lstm(x_base.clone().requires_grad_(True))
    warmup_out.sum().backward()
    tpu_lstm.zero_grad(set_to_none=True)

    torch.manual_seed(12345)
    x_eager = x_base.clone().requires_grad_(True)
    out_eager, (hy_eager, cy_eager) = tpu_lstm(x_eager)
    (out_eager.sum() + hy_eager.sum() + cy_eager.sum()).backward()
    eager_x_grad = x_eager.grad.clone()
    eager_param_grads = [
        (name, p.grad.clone()) for name, p in tpu_lstm.named_parameters()
    ]
    tpu_lstm.zero_grad(set_to_none=True)

    torch.manual_seed(12345)
    x_comp = x_base.clone().requires_grad_(True)
    out_comp, (hy_comp, cy_comp) = compiled_lstm(x_comp)
    (out_comp.sum() + hy_comp.sum() + cy_comp.sum()).backward()

    assert_close(out_comp.cpu(), out_eager.cpu(), atol=1e-5, rtol=1e-5)
    assert_close(hy_comp.cpu(), hy_eager.cpu(), atol=1e-5, rtol=1e-5)
    assert_close(cy_comp.cpu(), cy_eager.cpu(), atol=1e-5, rtol=1e-5)
    assert_close(x_comp.grad.cpu(), eager_x_grad.cpu(), atol=1e-5, rtol=1e-5)
    for (name, eager_g), (_, p) in zip(
        eager_param_grads, tpu_lstm.named_parameters()
    ):
      self.assertIsNotNone(p.grad, msg=f"Missing grad for {name}")
      assert_close(p.grad.cpu(), eager_g.cpu(), atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      # aten::lstm.input with a bf16 input and f32 hidden states and weights.
      ("mixed_dtype", torch.bfloat16, False, 1, 0.0, False, False, 0),
      # With dropout, the cached forward also returns the dropout RNG state.
      ("dropout_2layer_bidir_proj", torch.float32, True, 2, 0.4, True, True, 6),
  )
  def test_custom_ops_opcheck(
      self,
      dtype: torch.dtype,
      has_biases: bool,
      num_layers: int,
      dropout: float,
      bidirectional: bool,
      batch_first: bool,
      proj_size: int,
  ):
    """Checks the schemas and Meta kernels of the tpu::_lstm_input ops."""
    batch, seq_len, input_size, hidden_size = 2, 4, 8, 12
    num_directions = 2 if bidirectional else 1
    out_h = proj_size if proj_size > 0 else hidden_size
    _, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_biases,
        batch_first=batch_first,
        dtype=torch.float32,
        bidirectional=bidirectional,
        proj_size=proj_size,
    )
    in_shape = (
        (batch, seq_len, input_size)
        if batch_first
        else (seq_len, batch, input_size)
    )
    args = (
        torch.randn(in_shape, dtype=dtype, device="tpu"),
        [
            torch.randn(
                num_layers * num_directions, batch, out_h, device="tpu"
            ),
            torch.randn(
                num_layers * num_directions, batch, hidden_size, device="tpu"
            ),
        ],
        [p.detach() for p in tpu_lstm.parameters()],
        has_biases,
        num_layers,
        dropout,
        True,  # train
        bidirectional,
        batch_first,
    )
    # aten::lstm.input owns the autograd of these ops and the torch.compile
    # tests trace them, so this checks their schemas and Meta kernels.
    test_utils = ("test_schema", "test_faketensor")
    torch.library.opcheck(
        torch.ops.tpu._lstm_input.default, args, test_utils=test_utils
    )
    torch.library.opcheck(
        torch.ops.tpu._lstm_input_forward_cached.default,
        args,
        test_utils=test_utils,
    )
    fwd = torch.ops.tpu._lstm_input_forward_cached(*args)
    torch.library.opcheck(
        torch.ops.tpu._lstm_input_backward.default,
        (
            torch.randn_like(fwd[0]),
            torch.randn_like(fwd[1]),
            torch.randn_like(fwd[2]),
            *args,
            fwd[3:],
        ),
        test_utils=test_utils,
    )

  @parameterized.named_parameters(
      # The fused backward unrolls seq_len < kLstmUnrollFactor (8) and loops
      # over chunks of kLstmUnrollFactor otherwise, with a layer wavefront for
      # multi-layer unidirectional LSTMs.
      ("static", 1, False, 4, True, 0),
      ("chunked", 1, False, 9, True, 0),
      ("chunked_proj", 1, False, 9, True, 6),
      ("wavefront", 2, False, 9, True, 0),
      ("bidir_static", 1, True, 4, True, 0),
      ("bidir_chunked", 1, True, 9, True, 0),
      ("eager", 1, False, 4, False, 0),
  )
  def test_mixed_dtype_parity(
      self,
      num_layers: int,
      bidirectional: bool,
      seq_len: int,
      compiled: bool,
      proj_size: int,
  ):
    """Verifies aten::lstm.input with a bf16 input and f32 hx and weights."""
    torch._dynamo.reset()
    num_directions = 2 if bidirectional else 1
    out_h = proj_size if proj_size > 0 else 12
    cpu_lstm, tpu_lstm = self._create_models(
        input_size=8,
        hidden_size=12,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        bidirectional=bidirectional,
        proj_size=proj_size,
    )
    x = torch.randn(
        seq_len, 2, 8, dtype=torch.bfloat16, device="tpu", requires_grad=True
    )
    h0 = torch.randn(
        num_layers * num_directions, 2, out_h, device="tpu", requires_grad=True
    )
    c0 = torch.randn(
        num_layers * num_directions, 2, 12, device="tpu", requires_grad=True
    )
    # The f32 reference runs on the same bf16 input values.
    x_ref = x.detach().cpu().float().requires_grad_(True)
    h0_ref = h0.detach().cpu().requires_grad_(True)
    c0_ref = c0.detach().cpu().requires_grad_(True)

    def lstm(inp, hx, params):
      return torch.ops.aten.lstm.input(
          inp, hx, params, True, num_layers, 0.0, True, bidirectional, False
      )

    fn = torch.compile(lstm) if compiled else lstm
    out, hy, cy = fn(x, [h0, c0], list(tpu_lstm.parameters()))
    (out.sum() + hy.sum() + cy.sum()).backward()
    out_ref, hy_ref, cy_ref = lstm(
        x_ref, [h0_ref, c0_ref], list(cpu_lstm.parameters())
    )
    (out_ref.sum() + hy_ref.sum() + cy_ref.sum()).backward()

    # x is bf16 and TPU matmuls run in one bf16 pass at the default precision,
    # so all results get the bf16 tolerances of
    # test_torch_compile_forward_backward.
    tol = 5e-2
    assert_close(out.cpu().float(), out_ref, atol=tol, rtol=tol)
    assert_close(hy.cpu().float(), hy_ref, atol=tol, rtol=tol)
    assert_close(cy.cpu().float(), cy_ref, atol=tol, rtol=tol)
    self.assertEqual(x.grad.dtype, torch.bfloat16)
    assert_close(x.grad.cpu().float(), x_ref.grad, atol=tol, rtol=tol)
    assert_close(h0.grad.cpu(), h0_ref.grad, atol=tol, rtol=tol)
    assert_close(c0.grad.cpu(), c0_ref.grad, atol=tol, rtol=tol)
    for (name, cpu_p), (_, tpu_p) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters()
    ):
      self.assertIsNotNone(tpu_p.grad, msg=f"Missing grad for {name}")
      assert_close(tpu_p.grad.cpu(), cpu_p.grad, atol=1e-1, rtol=5e-2)
    # Gradients computed in bf16 and cast up to f32 would all be representable
    # in bf16.
    f32_grads = [("h0", h0.grad), ("c0", c0.grad)] + [
        (name, p.grad) for name, p in tpu_lstm.named_parameters()
    ]
    for name, grad in f32_grads:
      self.assertEqual(grad.dtype, torch.float32, msg=f"{name}.grad dtype")
      grad_cpu = grad.cpu()
      self.assertFalse(
          torch.equal(grad_cpu, grad_cpu.bfloat16().float()),
          msg=f"{name}.grad is truncated to bf16",
      )

  @parameterized.named_parameters(
      # The fused backward applies the inter-layer dropout layer by layer for
      # seq_len < kLstmUnrollFactor (8) and bidirectional LSTMs, and in a layer
      # wavefront over chunks of kLstmUnrollFactor plus remainder steps
      # otherwise.
      ("static", False, 4, False),
      ("wavefront", False, 9, False),
      ("bidir_static", True, 4, False),
      ("bidir_chunked", True, 9, False),
      ("compiled_wavefront", False, 9, True),
  )
  def test_full_dropout_train_parity(
      self, bidirectional: bool, seq_len: int, compiled: bool
  ):
    """Verifies aten::lstm.input with dropout=1.0 in train mode against CPU.

    Full dropout zeroes the input of every layer but the first, so the op is
    deterministic and no gradient flows through the dropped activations.
    """
    num_layers = 3
    num_directions = 2 if bidirectional else 1
    cpu_lstm, tpu_lstm = self._create_models(
        input_size=8,
        hidden_size=12,
        num_layers=num_layers,
        bias=True,
        batch_first=False,
        dtype=torch.float32,
        bidirectional=bidirectional,
    )
    state_shape = (num_layers * num_directions, 2, 12)
    # The input and the initial hidden and cell states.
    leaves_ref = [
        torch.randn(seq_len, 2, 8, requires_grad=True),
        torch.randn(state_shape, requires_grad=True),
        torch.randn(state_shape, requires_grad=True),
    ]
    leaves = [t.detach().to("tpu").requires_grad_(True) for t in leaves_ref]

    def lstm(inp, h0, c0, params):
      return torch.ops.aten.lstm.input(
          inp,
          [h0, c0],
          params,
          True,
          num_layers,
          1.0,
          True,
          bidirectional,
          False,
      )

    fn = torch.compile(lstm) if compiled else lstm
    outs = fn(*leaves, list(tpu_lstm.parameters()))
    sum(out.sum() for out in outs).backward()
    outs_ref = lstm(*leaves_ref, list(cpu_lstm.parameters()))
    sum(out.sum() for out in outs_ref).backward()

    tol = 5e-3
    for out, out_ref in zip(outs, outs_ref, strict=True):
      assert_close(out.cpu(), out_ref, atol=tol, rtol=tol)
    for leaf, leaf_ref in zip(leaves, leaves_ref, strict=True):
      assert_close(leaf.grad.cpu(), leaf_ref.grad, atol=tol, rtol=tol)
    for (name, cpu_p), (_, tpu_p) in zip(
        cpu_lstm.named_parameters(), tpu_lstm.named_parameters(), strict=True
    ):
      self.assertIsNotNone(tpu_p.grad, msg=f"Missing grad for {name}")
      assert_close(tpu_p.grad.cpu(), cpu_p.grad, atol=5e-2, rtol=5e-2)
    # Without gradients, the op runs the forward kernel that caches no
    # activations.
    with torch.no_grad():
      outs_no_grad = fn(*leaves, list(tpu_lstm.parameters()))
    for out, out_ref in zip(outs_no_grad, outs_ref, strict=True):
      assert_close(out.cpu(), out_ref, atol=tol, rtol=tol)

  def test_torch_compile_loss_root_inference_mode_and_recompiles(self):
    """Verifies compiled LSTM backward, recompiles and inference_mode."""
    torch._dynamo.reset()
    cpu_lstm, tpu_lstm = self._create_models(
        input_size=8,
        hidden_size=12,
        num_layers=1,
        bias=True,
        batch_first=True,
        dtype=torch.float32,
    )
    counter = dynamo_testing.CompileCounterWithBackend("tpu")
    # TorchTPU defaults to dynamic=False only for TPU backends, not counters.
    compiled_lstm = torch.compile(tpu_lstm, backend=counter, dynamic=False)

    compilations_per_call = []
    for seq_len in (4, 7, 4):
      tpu_lstm.zero_grad(set_to_none=True)
      cpu_lstm.zero_grad(set_to_none=True)
      x_cpu = torch.randn(3, seq_len, 8, requires_grad=True)
      x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)
      out_cpu, _ = cpu_lstm(x_cpu)
      compilations_before = counter.frame_count
      out_tpu, _ = compiled_lstm(x_tpu)
      compilations_per_call.append(counter.frame_count - compilations_before)
      out_cpu.sum().backward()
      out_tpu.sum().backward()
      assert_close(out_tpu.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
      assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)
    # A new sequence length compiles one graph, and a seen one reuses it.
    self.assertEqual(compilations_per_call, [1, 1, 0])

    torch._dynamo.reset()
    with torch.inference_mode():
      inf_lstm = torch.compile(tpu_lstm)
      x_inf = torch.randn(3, 6, 8)
      out_inf, (hy_inf, cy_inf) = inf_lstm(x_inf.to("tpu"))
      out_cpu, (hy_cpu, cy_cpu) = cpu_lstm(x_inf)
      assert_close(out_inf.cpu(), out_cpu, atol=5e-3, rtol=5e-3)
      assert_close(hy_inf.cpu(), hy_cpu, atol=5e-3, rtol=5e-3)
      assert_close(cy_inf.cpu(), cy_cpu, atol=5e-3, rtol=5e-3)

  def test_func_grad_parity(self):
    """Verifies torch.func.grad parity with CPU for aten::lstm.input."""
    batch, seq_len, input_size, hidden_size = 3, 5, 8, 12
    num_layers, num_directions = 2, 2
    cpu_lstm, tpu_lstm = self._create_models(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=True,
        dtype=torch.float32,
        bidirectional=True,
    )
    state_shape = (num_layers * num_directions, batch, hidden_size)
    x = torch.randn(batch, seq_len, input_size)
    hx = [torch.randn(state_shape), torch.randn(state_shape)]
    out_grad = torch.randn(batch, seq_len, num_directions * hidden_size)
    hy_grad = torch.randn(state_shape)
    cy_grad = torch.randn(state_shape)

    def loss(inp, states, params, grad_out, grad_hy, grad_cy):
      out, hy, cy = torch.ops.aten.lstm.input(
          inp, states, params, True, num_layers, 0.0, True, True, True
      )
      return (
          (out * grad_out).sum() + (hy * grad_hy).sum() + (cy * grad_cy).sum()
      )

    grad_fn = torch.func.grad(loss, argnums=(0, 1, 2))
    x_grad, hx_grads, param_grads = grad_fn(
        x,
        hx,
        [p.detach() for p in cpu_lstm.parameters()],
        out_grad,
        hy_grad,
        cy_grad,
    )
    tpu_x_grad, tpu_hx_grads, tpu_param_grads = grad_fn(
        x.to("tpu"),
        [h.to("tpu") for h in hx],
        [p.detach() for p in tpu_lstm.parameters()],
        out_grad.to("tpu"),
        hy_grad.to("tpu"),
        cy_grad.to("tpu"),
    )
    # Under torch.func, TPU runs PyTorch's composite LSTM step by step, with
    # each matmul in one bf16 pass at the default precision.
    assert_close(tpu_x_grad.cpu(), x_grad, atol=1e-2, rtol=1e-2)
    for tpu_grad, cpu_grad in zip(tpu_hx_grads, hx_grads):
      assert_close(tpu_grad.cpu(), cpu_grad, atol=1e-2, rtol=1e-2)
    for tpu_grad, cpu_grad in zip(tpu_param_grads, param_grads):
      assert_close(tpu_grad.cpu(), cpu_grad, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
  absltest.main()
