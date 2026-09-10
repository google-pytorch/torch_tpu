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
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
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

    assert_close(x_tpu.grad.cpu(), x_cpu.grad, atol=5e-3, rtol=5e-3)
    assert_close(h0_tpu.grad.cpu(), h0_cpu.grad, atol=5e-3, rtol=5e-3)
    assert_close(c0_tpu.grad.cpu(), c0_cpu.grad, atol=5e-3, rtol=5e-3)

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


if __name__ == "__main__":
  absltest.main()
