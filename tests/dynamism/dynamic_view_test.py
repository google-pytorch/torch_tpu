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

from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class DynamicViewTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  @absltest.skip("Dynamic view as input not supported")
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
    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled_model = torch.compile(model_tpu, backend=tpu_backend)

    with torch.no_grad():
      # 1. Prefill (seq_len = 256 >= sliding_window 128):
      # Model returns past_k_1 of length sliding_window - 1 = 127.
      new_k_prefill_cpu = torch.arange(
          16 * 8 * 256 * 8, dtype=torch.int32
      ).reshape(16, 8, 256, 8)
      new_k_prefill_tpu = new_k_prefill_cpu.to(self.device)
      past_k_1_cpu = model_cpu(new_k_prefill_cpu, None)
      past_k_1_tpu = compiled_model(new_k_prefill_tpu, None)
      utils.assert_close(past_k_1_tpu.cpu(), past_k_1_cpu)

      # 2. Decode Step 1 (past_k length 127 -> full_k length 128 -> past_k_2 length 127):
      # Compiles static decode frame for past_k size 127.
      new_k_token_1_cpu = torch.arange(
          16 * 8 * 1 * 8, dtype=torch.int32
      ).reshape(16, 8, 1, 8)
      new_k_token_1_tpu = new_k_token_1_cpu.to(self.device)
      past_k_2_cpu = model_cpu(new_k_token_1_cpu, past_k_1_cpu)
      past_k_2_tpu = compiled_model(new_k_token_1_tpu, past_k_1_tpu)
      utils.assert_close(past_k_2_tpu.cpu(), past_k_2_cpu)

      # 3. Decode Step 2 (past_k length 127 -> full_k length 128 -> dynamic frame):
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

  @absltest.skip("Dynamic view as input not supported")
  def test_dynamic_size_dynamic_stride_input(self):
    class DynamicViewModule(torch.nn.Module):

      def forward(self, x):
        # Operates on an input tensor that is a dynamic non-contiguous view
        return x + 1

    model_cpu = DynamicViewModule()
    model_tpu = DynamicViewModule()
    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled_model = torch.compile(model_tpu, backend=tpu_backend)

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


if __name__ == "__main__":
  absltest.main()
