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

"""Numerical correctness tests for Gemma4 SWA on TPU."""

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import utils
from examples.gemma4 import model


class Gemma4SWACorrectnessTPUTest(absltest.TestCase):
  # Build a ratio-based gradient comparison helper (semantics match torchtitan tests)
  def assert_grads_close(self, actual, expected, tol=0.05):
    diff = (actual.float() - expected.float()).abs()
    ref_max = expected.float().abs().max().item() + 1e-6
    rel_mean = diff.mean().item() / ref_max
    self.assertLess(
        rel_mean,
        tol,
        f'Gradient mismatch: rel_mean={rel_mean:.4f} (max allowed {tol})',
    )

  def test_swa_attention_parity(self):
    # Setup configs
    config = model.Gemma4Config(
        vocab_size=100,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        sliding_window=16,  # Small sliding window for testing
        layer_types=['sliding_attention'],  # SWA layer
    )

    # We test with sequence length > sliding_window
    seq_len = 128
    bsz = 2

    # Instantiate CPU layer (naive implementation)
    cpu_layer = model.Gemma4Attention(config, layer_idx=0)
    cpu_layer.eval()

    # Move to TPU and instantiate TPU layer
    device = 'tpu'
    try:
      torch.tensor([1.0]).to(device)
    except RuntimeError:
      self.skipTest('TPU device not available')

    tpu_layer = model.Gemma4Attention(config, layer_idx=0).to(device)
    tpu_layer.eval()

    # Copy weights
    tpu_layer.load_state_dict(cpu_layer.state_dict())

    # Generate inputs
    torch.manual_seed(42)
    hidden_states_cpu = torch.randn(bsz, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

    hidden_states_tpu = hidden_states_cpu.to(device)
    position_ids_tpu = position_ids.to(device)

    # Run forward
    with torch.no_grad():
      cpu_out, _ = cpu_layer(hidden_states_cpu, position_ids)
      tpu_out, _ = tpu_layer(hidden_states_tpu, position_ids_tpu)

    # Compare forward outputs
    # For bfloat16 or float32, we should use appropriate tolerances.
    # Splash attention usually runs in the precision of inputs.
    # Gemma4Attention weights and inputs are float32 here by default.
    utils.assert_close(cpu_out, tpu_out.cpu(), rtol=5e-2, atol=5e-2)

  def test_swa_attention_backward_parity(self):
    # Setup configs
    config = model.Gemma4Config(
        vocab_size=100,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        sliding_window=16,
        layer_types=['sliding_attention'],
    )

    seq_len = 128
    bsz = 2
    device = 'tpu'
    try:
      torch.tensor([1.0]).to(device)
    except RuntimeError:
      self.skipTest('TPU device not available')

    # Instantiate CPU layer (naive implementation)
    cpu_layer = model.Gemma4Attention(config, layer_idx=0)

    # Instantiate TPU layer
    tpu_layer = model.Gemma4Attention(config, layer_idx=0).to(device)

    # Copy weights
    tpu_layer.load_state_dict(cpu_layer.state_dict())

    # Inputs
    torch.manual_seed(42)
    hidden_states_cpu = torch.randn(
        bsz, seq_len, config.hidden_size, requires_grad=True
    )
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

    hidden_states_tpu = (
        hidden_states_cpu.detach().clone().to(device).requires_grad_(True)
    )
    position_ids_tpu = position_ids.to(device)

    # Run forward & backward CPU
    cpu_out, _ = cpu_layer(hidden_states_cpu, position_ids)
    cpu_loss = cpu_out.sum()
    cpu_loss.backward()

    # Run forward & backward TPU
    tpu_out, _ = tpu_layer(hidden_states_tpu, position_ids_tpu)
    tpu_loss = tpu_out.sum()
    tpu_loss.backward()

    # Compare gradients of inputs
    self.assert_grads_close(
        hidden_states_tpu.grad.cpu(), hidden_states_cpu.grad
    )

    # Compare gradients of weights
    for name, param in cpu_layer.named_parameters():
      tpu_param = dict(tpu_layer.named_parameters())[name]
      if param.grad is not None:
        self.assertIsNotNone(tpu_param.grad, f'TPU grad for {name} is None')
        self.assert_grads_close(tpu_param.grad.cpu(), param.grad)

  def test_global_attention_parity(self):
    # Test that when is_sliding is False, we also get parity (routing to Splash with window_size=None on TPU)
    config = model.Gemma4Config(
        vocab_size=100,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        layer_types=['full_attention'],  # Global attention
    )

    seq_len = 128
    bsz = 2
    device = 'tpu'
    try:
      torch.tensor([1.0]).to(device)
    except RuntimeError:
      self.skipTest('TPU device not available')

    cpu_layer = model.Gemma4Attention(config, layer_idx=0)
    cpu_layer.eval()

    tpu_layer = model.Gemma4Attention(config, layer_idx=0).to(device)
    tpu_layer.eval()

    tpu_layer.load_state_dict(cpu_layer.state_dict())

    torch.manual_seed(42)
    hidden_states_cpu = torch.randn(bsz, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

    hidden_states_tpu = hidden_states_cpu.to(device)
    position_ids_tpu = position_ids.to(device)

    with torch.no_grad():
      cpu_out, _ = cpu_layer(hidden_states_cpu, position_ids)
      tpu_out, _ = tpu_layer(hidden_states_tpu, position_ids_tpu)

    utils.assert_close(cpu_out, tpu_out.cpu(), rtol=5e-2, atol=5e-2)

  def test_skip_sliding_mask_parity(self):
    # Test that skip_sliding_mask works.
    # When skip_sliding_mask=True is passed on TPU, it should fallback to naive.
    # Or actually, if attention_mask is None and skip_sliding_mask is True, it routes to Splash with local_window_size=None (global causal).
    # If we pass skip_sliding_mask=True, it should match CPU with skip_sliding_mask=True.
    config = model.Gemma4Config(
        vocab_size=100,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        sliding_window=16,
        layer_types=['sliding_attention'],
    )

    seq_len = 128
    bsz = 2
    device = 'tpu'
    try:
      torch.tensor([1.0]).to(device)
    except RuntimeError:
      self.skipTest('TPU device not available')

    cpu_layer = model.Gemma4Attention(config, layer_idx=0)
    cpu_layer.eval()

    tpu_layer = model.Gemma4Attention(config, layer_idx=0).to(device)
    tpu_layer.eval()

    tpu_layer.load_state_dict(cpu_layer.state_dict())

    torch.manual_seed(42)
    hidden_states_cpu = torch.randn(bsz, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

    hidden_states_tpu = hidden_states_cpu.to(device)
    position_ids_tpu = position_ids.to(device)

    with torch.no_grad():
      cpu_out, _ = cpu_layer(
          hidden_states_cpu, position_ids, skip_sliding_mask=True
      )
      tpu_out, _ = tpu_layer(
          hidden_states_tpu, position_ids_tpu, skip_sliding_mask=True
      )

    # Since skip_sliding_mask=True, SWA mask is skipped.
    # CPU should match TPU (both are now global causal attention).
    utils.assert_close(cpu_out, tpu_out.cpu(), rtol=5e-2, atol=5e-2)


if __name__ == '__main__':
  absltest.main()
