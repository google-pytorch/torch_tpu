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
from torch_tpu._internal.utils import test_utils as utils
from examples.gemma4 import model
from tests import seed_test_utils

try:
  from torch_tpu.ops import splash_attention
except (ImportError, ModuleNotFoundError):
  splash_attention = None


class Gemma4SWACorrectnessTPUTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if splash_attention is None:
      self.skipTest('splash_attention module is not available')
    try:
      torch.tensor([1.0]).to('tpu')
    except RuntimeError:
      self.skipTest('TPU device not available')

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
    cpu_loss = cpu_out.mean()
    cpu_loss.backward()

    # Run forward & backward TPU
    tpu_out, _ = tpu_layer(hidden_states_tpu, position_ids_tpu)
    tpu_loss = tpu_out.mean()
    tpu_loss.backward()

    # Compare gradients of inputs
    utils.assert_close(
        hidden_states_tpu.grad.cpu(),
        hidden_states_cpu.grad,
        rtol=5.0e-2,
        atol=5.0e-2,
    )

    # Compare gradients of weights
    for name, param in cpu_layer.named_parameters():
      tpu_param = dict(tpu_layer.named_parameters())[name]
      if param.grad is not None:
        self.assertIsNotNone(tpu_param.grad, f'TPU grad for {name} is None')
        utils.assert_close(
            tpu_param.grad.cpu(),
            param.grad,
            rtol=5.0e-2,
            atol=5.0e-2,
        )

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
    # Test that skip_sliding_mask=True correctly falls back to naive global attention.
    # We test that the TPU output matches the CPU naive implementation.
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

  def test_splash_sdpa_direct_call_backward(self):
    device = torch.device('tpu')
    q = torch.randn(2, 4, 128, 32, device=device, requires_grad=True)
    k = torch.randn(2, 4, 128, 32, device=device, requires_grad=True)
    v = torch.randn(2, 4, 128, 32, device=device, requires_grad=True)

    out = splash_attention.splash_sdpa(
        q, k, v, local_window_size=16, use_fused_bwd_kernel=False
    )
    loss = out.mean()
    loss.backward()

    self.assertEqual(out.shape, (2, 4, 128, 32))
    self.assertIsNotNone(q.grad)
    self.assertIsNotNone(k.grad)
    self.assertIsNotNone(v.grad)


if __name__ == '__main__':
  absltest.main()
