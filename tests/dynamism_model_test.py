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
from absl.testing import parameterized
import torch
from torch_tpu import api
from torch_tpu._internal import dynamism
from torch_tpu._internal import sync
from torch_tpu._internal.utils import utils


class DynamismModelTest(parameterized.TestCase):
  """Unit tests for bounded dynamism support on nn.Modules."""

  def setUp(self):
    super().setUp()
    self.device = api.tpu_device()
    torch.manual_seed(1234)

  def test_linear_model_batch_dim0_dynamic(self):
    """Tests a simple model with nn.Linear and dynamic batch dimension."""

    class SimpleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=True)

      def forward(self, x):
        return self.linear(x)

    # Run on CPU to get expected output
    model = SimpleModel()
    inp = torch.rand(10, 3)
    expected = model(inp)

    # Run on TPU with dynamic input
    tpu_model = model.to(self.device)
    inp_tpu = inp.to(self.device)
    dynamism.mark_dynamic(inp_tpu, 0, 2, 20)  # Dynamic batch dim [2, 20]
    act = tpu_model(inp_tpu)

    utils.assert_close(act.to("cpu"), expected, rtol=1e-2, atol=1e-2)

  def test_linear_model_dim1_dynamic(self):
    """Tests a simple model with nn.Linear and dynamic batch dimension."""

    class SimpleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=True)

      def forward(self, x):
        return self.linear(x)

    # Run on CPU to get expected output
    model = SimpleModel()
    inp = torch.rand(10, 3)
    expected = model(inp)

    # Run on TPU with dynamic input
    tpu_model = model.to(self.device)
    inp_tpu = inp.to(self.device)
    dynamism.mark_dynamic(inp_tpu, 1, 2, 20)  # Dynamic dim [2, 20]
    dynamism.mark_dynamic(
        tpu_model.linear.weight, 1, 2, 20
    )  # Dynamic dim [2, 20]
    act = tpu_model(inp_tpu)

    utils.assert_close(act.to("cpu"), expected, rtol=1e-2, atol=1e-2)


class KVCacheDynamismTest(parameterized.TestCase):
  """Unit tests for bounded dynamism on KV cache like operations."""

  def setUp(self):
    super().setUp()
    self.device = api.tpu_device()
    torch.manual_seed(42)

  def test_bd_with_reshape_from_matmul_op(self):
    """Reshape is called from the structured delegate matmul op on bounded dynamic input."""
    batch_size = 1
    num_heads = 2
    head_dim = 8
    seq_len = 4
    max_seq_len = 20

    query_cpu = torch.randn(batch_size, num_heads, 1, head_dim)
    k_cache_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim)
    new_k_cpu = torch.randn(batch_size, num_heads, 1, head_dim)

    # Run on CPU to get expected output
    k_cpu = torch.cat([k_cache_cpu, new_k_cpu], dim=2)
    expected = torch.matmul(query_cpu, k_cpu.transpose(2, 3))

    # Run on TPU
    query_tpu = query_cpu.to(self.device)
    k_cache_tpu = k_cache_cpu.to(self.device)
    new_k_tpu = new_k_cpu.to(self.device)
    dynamism.mark_dynamic(k_cache_tpu, 2, seq_len, max_seq_len)
    k_tpu = torch.cat([k_cache_tpu, new_k_tpu], dim=2)
    actual = torch.matmul(query_tpu, k_tpu.transpose(2, 3))

    utils.assert_close(actual.to("cpu"), expected, rtol=1e-2, atol=1e-2)

  def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = hidden_states.expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )

  def _repeat_kv_with_seq_len_for_n_rep(
      self, hidden_states: torch.Tensor
  ) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = hidden_states.expand(
        batch, num_key_value_heads, slen, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * slen, slen, head_dim
    )

  def test_repeat_kv_with_dynamic_seq_len(self):
    batch_size = 1
    num_kv_heads = 8
    num_q_heads = 16
    n_rep = num_q_heads // num_kv_heads
    head_dim = 128
    seq_len = 10
    max_seq_len = 20

    k_cpu = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    new_k_cpu = torch.randn(batch_size, num_kv_heads, 1, head_dim)

    # Run on CPU to get expected output
    k_cat_cpu = torch.cat([k_cpu, new_k_cpu], dim=2)
    expected = self._repeat_kv(k_cat_cpu, n_rep)

    # Run on TPU
    k_tpu = k_cpu.to(self.device)
    new_k_tpu = new_k_cpu.to(self.device)

    dynamism.mark_dynamic(k_tpu, 2, seq_len, max_seq_len)
    k_tpu = torch.cat([k_tpu, new_k_tpu], dim=2)
    k_repeated_tpu = self._repeat_kv(k_tpu, n_rep)
    actual = k_repeated_tpu.to("cpu")
    utils.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

  def test_bounded_dynamic_cache_misses_as_strided(self):
    batch_size = 1
    num_kv_heads = 8
    num_q_heads = 16
    n_rep = num_q_heads // num_kv_heads
    head_dim = 64
    seq_len = 2048
    max_seq_len = seq_len + 10

    k_cpu = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    new_k_cpu = torch.randn(batch_size, num_kv_heads, 1, head_dim)

    # Run on TPU
    k_tpu = k_cpu.to(self.device)
    new_k_tpu = new_k_cpu.to(self.device)
    sync.synchronize(k_tpu, wait=True)
    sync.synchronize(new_k_tpu, wait=True)

    for i in range(3):
      prev_cache_misses = torch.tpu._get_cache_misses()
      dynamism.mark_dynamic(k_tpu, 2, seq_len, max_seq_len)
      k_tpu = torch.cat([k_tpu, new_k_tpu], dim=2)
      k_repeated_tpu = self._repeat_kv(k_tpu, n_rep)
      k_repeated_tpu.to("cpu")
      current_cache_misses = torch.tpu._get_cache_misses()
      print(
          f"cache misses during step {i}: ",
          current_cache_misses - prev_cache_misses,
          flush=True,
      )
      # TODO(b/482024605): uncomment below check as with bounded dynamism,
      # cache misses are expected to be 0 after the first iteration.
      # if i >= 1:
      #   self.assertEqual(current_cache_misses - prev_cache_misses, 0)

  def test_bounded_dynamic_cache_misses_expected_with_as_strided(self):
    batch_size = 1
    num_kv_heads = 8
    head_dim = 16
    seq_len = 2048
    max_seq_len = seq_len + 10

    k_cpu = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    new_k_cpu = torch.randn(batch_size, num_kv_heads, 1, head_dim)

    # Run on TPU
    k_tpu = k_cpu.to(self.device)
    new_k_tpu = new_k_cpu.to(self.device)
    sync.synchronize(k_tpu, wait=True)
    sync.synchronize(new_k_tpu, wait=True)

    for i in range(3):
      prev_cache_misses = torch.tpu._get_cache_misses()
      dynamism.mark_dynamic(k_tpu, 2, seq_len, max_seq_len)
      k_tpu = torch.cat([k_tpu, new_k_tpu], dim=2)
      k_repeated_tpu = self._repeat_kv_with_seq_len_for_n_rep(k_tpu)
      k_repeated_tpu.to("cpu")
      current_cache_misses = torch.tpu._get_cache_misses()
      print(
          f"cache misses during step {i}: ",
          current_cache_misses - prev_cache_misses,
          flush=True,
      )
      self.assertGreater(current_cache_misses - prev_cache_misses, 1)

  def test_kv_cache_mark_dynamic_in_multiple_steps(self):
    """Test setting bounded dynamism on same layer in multiple decode steps."""
    batch_size = 1
    num_heads = 2
    head_dim = 8
    seq_len = 4
    decode_steps = 2
    max_seq_len = 20

    k_cache = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=self.device
    )

    new_k_list = [
        torch.randn(batch_size, num_heads, 1, head_dim, device=self.device)
        for _ in range(decode_steps)
    ]

    for step in range(decode_steps):
      print(f"decode step: {step}", flush=True)
      dynamism.mark_dynamic(k_cache, 2, seq_len, max_seq_len)
      k_cache = torch.cat([k_cache, new_k_list[step]], dim=2)
      add_k = torch.add(k_cache, k_cache)
      # TODO(b/478357255): uncomment the line below and remove the line after.
      # add_k.to("cpu")
      del add_k  # Suppress "unused variable" warning.


if __name__ == "__main__":
  absltest.main()
