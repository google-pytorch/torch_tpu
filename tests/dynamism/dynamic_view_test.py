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

"""Test demonstrating dynamic execution through torch.compile() with input/output as dynamic views."""

from collections.abc import Callable

from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile.dynamic import view_decomposition
from torch_tpu._internal.utils import test_utils as utils
from tests import oss_utils
from tests import seed_test_utils


@oss_utils.skip_if_libtpu_version(
    lambda v: v < (0, 0, 47),
    "Bounded dynamism requires libtpu >= 0.0.47",
)
class DynamicViewTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def test_static_size_dynamic_stride_input(self):
    class SlidingWindowCacheUpdate(torch.nn.Module):

      def __init__(self, sliding_window=128):
        super().__init__()
        self.sliding_window = sliding_window

      def forward(self, new_k, past_k=None):
        if past_k is not None:
          full_k = torch.cat([past_k, new_k], dim=2)
        else:
          full_k = new_k

        # Only retain the last sliding_window - 1 tokens in the returned cache.
        next_past_k = full_k[:, :, -self.sliding_window + 1 :, :]
        return next_past_k

    model_cpu = SlidingWindowCacheUpdate(sliding_window=128)
    model_tpu = SlidingWindowCacheUpdate(sliding_window=128)
    compiled_model = torch.compile(
        model_tpu, backend="tpu", options={"bounded_dynamism": True}
    )

    with torch.no_grad():
      # Prefill (seq_len = 256 >= sliding_window 128):
      # Model returns past_k_1 of length sliding_window - 1 = 127.
      new_k_prefill_cpu = torch.arange(
          16 * 8 * 256 * 8, dtype=torch.int32
      ).reshape(16, 8, 256, 8)
      new_k_prefill_tpu = new_k_prefill_cpu.to(self.device)
      past_k_1_cpu = model_cpu(new_k_prefill_cpu, None)
      past_k_1_tpu = compiled_model(new_k_prefill_tpu, None)
      utils.assert_close(past_k_1_tpu.cpu(), past_k_1_cpu)

      # Decode Step 1 (past_k length 127 -> full_k length 128 -> past_k_2 length 127):
      # Compiles static decode frame for past_k size 127.
      new_k_token_1_cpu = torch.arange(
          16 * 8 * 1 * 8, dtype=torch.int32
      ).reshape(16, 8, 1, 8)
      new_k_token_1_tpu = new_k_token_1_cpu.to(self.device)
      past_k_2_cpu = model_cpu(new_k_token_1_cpu, past_k_1_cpu)
      past_k_2_tpu = compiled_model(new_k_token_1_tpu, past_k_1_tpu)
      utils.assert_close(past_k_2_tpu.cpu(), past_k_2_cpu)

      # Decode Step 2 (past_k length 127 -> full_k length 128 -> dynamic frame):
      # past_k_2 was produced by model's slice from dynamic full_k_1.
      # When past_k_2 enters compiled_model as an input argument:
      #   - Size is static: torch.Size([16, 8, 127, 8]) (because length is always 127)
      #   - Stride is dynamic: (8*s0, s0, 8, 1) (inherited from the dynamic full_k buffer)
      new_k_token_2_cpu = torch.arange(
          16 * 8 * 1 * 8, dtype=torch.int32
      ).reshape(16, 8, 1, 8)
      new_k_token_2_tpu = new_k_token_2_cpu.to(self.device)
      past_k_3_cpu = model_cpu(new_k_token_2_cpu, past_k_2_cpu)
      past_k_3_tpu = compiled_model(new_k_token_2_tpu, past_k_2_tpu)
      utils.assert_close(past_k_3_tpu.cpu(), past_k_3_cpu)

      # Decode Step 3 (past_k length 127 -> full_k length 128 -> dynamic frame):
      # past_k_2 was produced by model's slice from dynamic full_k_1.
      # When past_k_2 enters compiled_model as an input argument:
      #   - Size is static: torch.Size([16, 8, 127, 8]) (because length is always 127)
      #   - Stride is dynamic: (8*s0, s0, 8, 1) (inherited from the dynamic full_k buffer)
      new_k_token_3_cpu = torch.arange(
          16 * 8 * 1 * 8, dtype=torch.int32
      ).reshape(16, 8, 1, 8)
      new_k_token_3_tpu = new_k_token_3_cpu.to(self.device)
      past_k_4_cpu = model_cpu(new_k_token_3_cpu, past_k_3_cpu)
      past_k_4_tpu = compiled_model(new_k_token_3_tpu, past_k_3_tpu)
      utils.assert_close(past_k_4_tpu.cpu(), past_k_4_cpu)

  def test_dynamic_size_dynamic_stride_input(self):
    class DynamicViewModule(torch.nn.Module):

      def forward(self, x):
        # Operates on an input tensor that is a dynamic non-contiguous view
        return x + 1

    model_cpu = DynamicViewModule()
    model_tpu = DynamicViewModule()
    compiled_model = torch.compile(
        model_tpu, backend="tpu", options={"bounded_dynamism": True}
    )

    with torch.no_grad():
      # Step 1: Pass a dynamic non-contiguous view (transpose on dynamic seq_len)
      base_1_cpu = torch.arange(16 * 8 * 20 * 64, dtype=torch.int32).reshape(
          16, 8, 20, 64
      )
      view_1_cpu = base_1_cpu.transpose(
          1, 2
      )  # shape [16, 20, 8, 64], non-contiguous
      view_1_tpu = base_1_cpu.to(self.device).transpose(1, 2)
      out_1_cpu = model_cpu(view_1_cpu)
      out_1_tpu = compiled_model(view_1_tpu)
      utils.assert_close(out_1_tpu.cpu(), out_1_cpu)

      # Step 2: Pass a different dynamic seq_len with the same dynamic non-contiguous view
      base_2_cpu = torch.arange(16 * 8 * 30 * 64, dtype=torch.int32).reshape(
          16, 8, 30, 64
      )
      view_2_cpu = base_2_cpu.transpose(
          1, 2
      )  # shape [16, 30, 8, 64], non-contiguous
      view_2_tpu = base_2_cpu.to(self.device).transpose(1, 2)
      out_2_cpu = model_cpu(view_2_cpu)
      out_2_tpu = compiled_model(view_2_tpu)
      utils.assert_close(out_2_tpu.cpu(), out_2_cpu)

      # Step 3: Pass a different dynamic seq_len with the same dynamic non-contiguous view
      base_3_cpu = torch.arange(16 * 8 * 40 * 64, dtype=torch.int32).reshape(
          16, 8, 40, 64
      )
      view_3_cpu = base_3_cpu.transpose(
          1, 2
      )  # shape [16, 40, 8, 64], non-contiguous
      view_3_tpu = base_3_cpu.to(self.device).transpose(1, 2)
      out_3_cpu = model_cpu(view_3_cpu)
      out_3_tpu = compiled_model(view_3_tpu)
      utils.assert_close(out_3_tpu.cpu(), out_3_cpu)

  def test_unit_dim_arbitrary_stride_state_input(self):
    """Tests a state input whose size-1 batch dim has an arbitrary stride.

    torch.zeros_like() copies such strides verbatim (e.g. HF linear-attention
    recurrent states). The input is contiguous and must not be reconstructed
    as a view with a larger base than its storage.
    """

    class RecurrentStateUpdate(torch.nn.Module):

      def forward(self, x, state):
        # x: [1, seq, 64], state: [1, 8, 4, 64] (updated in place).
        new_state = state * 0.5 + x.sum(dim=1)[:, None, None, :]
        state.copy_(new_state)
        return x + new_state.mean(dim=(1, 2)).unsqueeze(1)

    model_cpu = RecurrentStateUpdate()
    compiled_model = torch.compile(
        RecurrentStateUpdate(),
        backend="tpu",
        options={"bounded_dynamism": True},
    )

    shape, stride = (1, 8, 4, 64), (4096, 256, 64, 1)
    state_cpu = torch.empty_strided(shape, stride).copy_(torch.randn(shape))
    state_tpu = torch.empty_strided(shape, stride, device=self.device)
    state_tpu.copy_(state_cpu.to(self.device))
    self.assertEqual(state_tpu.stride(), stride)
    # The reconstructed base must cover exactly the state's storage.
    self.assertEqual(
        view_decomposition.get_base_tensor(state_tpu).numel(),
        _storage_numel(state_tpu),
    )

    with torch.no_grad():
      # The second and third steps run the bounded-dynamic executable.
      for seq_len in (5, 9, 13):
        x_cpu = torch.randn(1, seq_len, 64)
        out_cpu = model_cpu(x_cpu, state_cpu)
        out_tpu = compiled_model(x_cpu.to(self.device), state_tpu)
        utils.assert_close(out_tpu.cpu(), out_cpu)
        utils.assert_close(state_tpu.cpu(), state_cpu)

  def test_unit_dim_arbitrary_stride_transposed_state_input(self):
    """Tests a transposed view of a state whose size-1 dim has a bogus stride."""

    class TransposedStateRead(torch.nn.Module):

      def forward(self, x, state_t):
        # x: [1, seq, 64], state_t: [1, 4, 8, 64] (transposed view).
        return x + state_t.amax(dim=2).sum(dim=1, keepdim=True)

    model_cpu = TransposedStateRead()
    compiled_model = torch.compile(
        TransposedStateRead(),
        backend="tpu",
        options={"bounded_dynamism": True},
    )

    shape, stride = (1, 8, 4, 64), (4096, 256, 64, 1)
    state_cpu = torch.empty_strided(shape, stride).copy_(torch.randn(shape))
    state_tpu = torch.empty_strided(shape, stride, device=self.device)
    state_tpu.copy_(state_cpu.to(self.device))
    state_t_cpu = state_cpu.transpose(1, 2)
    state_t_tpu = state_tpu.transpose(1, 2)
    # The reconstructed base must not exceed the state's storage.
    self.assertLessEqual(
        view_decomposition.get_base_tensor(state_t_tpu).numel(),
        _storage_numel(state_t_tpu),
    )

    with torch.no_grad():
      # The second and third steps run the bounded-dynamic executable.
      for seq_len in (5, 9, 13):
        x_cpu = torch.randn(1, seq_len, 64)
        out_cpu = model_cpu(x_cpu, state_t_cpu)
        out_tpu = compiled_model(x_cpu.to(self.device), state_t_tpu)
        utils.assert_close(out_tpu.cpu(), out_cpu)

  def _check_dynamic_unit_slice_input(
      self,
      base_shape: tuple[int, ...],
      seq_dim: int,
      view_fn: Callable[[torch.Tensor], torch.Tensor],
      slice_dim: int,
  ) -> None:
    """Runs a size-1 slice of a growing base, which has dynamic strides.

    The decomposition must rebuild the original base dims with a single slice
    (no permute or flattened dims).

    Args:
      base_shape: Shape of the sliced base; `seq_dim` grows across steps.
      seq_dim: Dimension of the base that grows across steps.
      view_fn: Takes the base and returns the size-1 slice passed as input.
      slice_dim: Base dimension sliced by `view_fn`.
    """

    class AddOne(torch.nn.Module):

      def forward(self, x):
        return x + 1

    torch._dynamo.reset()
    model_cpu = AddOne()
    compiled_model = torch.compile(
        AddOne(), backend="tpu", options={"bounded_dynamism": True}
    )

    with torch.no_grad():
      # The second and third steps run the bounded-dynamic executable.
      for seq_len in (20, 30, 40):
        shape = list(base_shape)
        shape[seq_dim] = seq_len
        base_cpu = torch.randn(shape)
        view_cpu = view_fn(base_cpu)
        view_tpu = view_fn(base_cpu.to(self.device))

        decomp = view_decomposition.decompose_into_view_sequence(view_tpu)
        self.assertIsNotNone(decomp)
        decomp_base_shape, view_ops = decomp
        self.assertEqual(decomp_base_shape, shape)
        self.assertLen(view_ops, 1)
        self.assertEqual(view_ops[0][0], "slice")
        self.assertEqual(view_ops[0][1][0], slice_dim)
        base_tpu = view_decomposition.get_base_tensor(view_tpu)
        self.assertEqual(list(base_tpu.shape), shape)

        out_tpu = compiled_model(view_tpu)
        utils.assert_close(out_tpu.cpu(), model_cpu(view_cpu))

  def test_dynamic_last_token_slice_input(self):
    """Tests `hidden[:, -1:, :]` for batch sizes 1 and 2."""
    for batch in (1, 2):
      with self.subTest(batch=batch):
        self._check_dynamic_unit_slice_input(
            (batch, 20, 64), 1, lambda x: x[:, -1:, :], slice_dim=1
        )

  def test_dynamic_last_input_id_slice_input(self):
    """Tests `input_ids[:, -1:]`, where every dim has size 1."""
    self._check_dynamic_unit_slice_input(
        (1, 20), 1, lambda x: x[:, -1:], slice_dim=1
    )

  def test_dynamic_middle_token_slice_input(self):
    """Tests `hidden[:, 3:4, :]` with batch size 1 (constant storage offset)."""
    self._check_dynamic_unit_slice_input(
        (1, 20, 64), 1, lambda x: x[:, 3:4, :], slice_dim=1
    )

  def test_dynamic_last_position_slice_input(self):
    """Tests `cache[:, :, -1:, :]`."""
    self._check_dynamic_unit_slice_input(
        (2, 4, 20, 8), 2, lambda x: x[:, :, -1:, :], slice_dim=2
    )

  def test_dynamic_first_position_slice_input(self):
    """Tests `cache[:, :, :1, :]` with batch size 1 (zero storage offset)."""
    self._check_dynamic_unit_slice_input(
        (1, 4, 20, 8), 2, lambda x: x[:, :, :1, :], slice_dim=2
    )

  def _check_dynamic_multi_op_view_input(
      self,
      base_shape: tuple[int, ...],
      seq_dim: int,
      view_fn: Callable[[torch.Tensor], torch.Tensor],
      expected_view_ops: list[str],
  ) -> None:
    """Runs a view of a growing base that decomposes into several view ops.

    Every inserted view op, not only the last one, must carry the metadata
    that the dynamism passes read.

    Args:
      base_shape: Shape of the viewed base; `seq_dim` grows across steps.
      seq_dim: Dimension of the base that grows across steps.
      view_fn: Takes the base and returns the view passed as input.
      expected_view_ops: Expected kinds of the decomposed view ops in order.
    """

    class AddOne(torch.nn.Module):

      def forward(self, x):
        return x + 1

    torch._dynamo.reset()
    model_cpu = AddOne()
    compiled_model = torch.compile(
        AddOne(), backend="tpu", options={"bounded_dynamism": True}
    )

    with torch.no_grad():
      # The second and third steps run the bounded-dynamic executable.
      for seq_len in (20, 30, 40):
        shape = list(base_shape)
        shape[seq_dim] = seq_len
        base_cpu = torch.randn(shape)
        view_cpu = view_fn(base_cpu)
        view_tpu = view_fn(base_cpu.to(self.device))

        decomp = view_decomposition.decompose_into_view_sequence(view_tpu)
        self.assertIsNotNone(decomp)
        _, view_ops = decomp
        self.assertEqual([op for op, _ in view_ops], expected_view_ops)

        out_tpu = compiled_model(view_tpu)
        utils.assert_close(out_tpu.cpu(), model_cpu(view_cpu))

  def test_dynamic_transposed_slice_input(self):
    """Tests `k.transpose(1, 2)[:, :4]` (slice, then permute)."""
    self._check_dynamic_multi_op_view_input(
        (2, 3, 20, 8),
        2,
        lambda x: x.transpose(1, 2)[:, :4],
        ["slice", "permute"],
    )

  def test_dynamic_two_dim_slice_input(self):
    """Tests `hidden[:, 1:3, 2:6]` (two slices)."""
    self._check_dynamic_multi_op_view_input(
        (2, 20, 8), 1, lambda x: x[:, 1:3, 2:6], ["slice", "slice"]
    )


def _storage_numel(tensor: torch.Tensor) -> int:
  return tensor.untyped_storage().nbytes() // tensor.element_size()


if __name__ == "__main__":
  absltest.main()
