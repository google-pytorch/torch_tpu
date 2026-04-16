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
import torch
import torch.accelerator


class RngTest(absltest.TestCase):

  def test_get_rng_state_metadata(self):
    device = torch.accelerator.current_accelerator()
    state = torch.tpu.get_rng_state(device)
    self.assertEqual(state.device.type, "cpu")
    self.assertEqual(state.dtype, torch.uint8)
    self.assertEqual(state.shape, (16,))

  def test_set_rng_state(self):
    device = torch.accelerator.current_accelerator()
    initial_state = torch.tpu.get_rng_state(device)

    # Change the state by generating some random numbers.
    torch.randn(10, device=device)
    new_state = torch.tpu.get_rng_state(device)
    self.assertFalse(torch.equal(initial_state, new_state))

    # Restore the initial state and verify.
    torch.tpu.set_rng_state(initial_state, device)
    restored_state = torch.tpu.get_rng_state(device)
    self.assertTrue(torch.equal(initial_state, restored_state))

  def test_manual_seed(self):
    device = torch.accelerator.current_accelerator()
    torch.tpu.manual_seed_all(42)
    state1 = torch.tpu.get_rng_state(device)

    torch.tpu.manual_seed_all(42)
    state2 = torch.tpu.get_rng_state(device)
    self.assertTrue(torch.equal(state1, state2))

    torch.tpu.manual_seed_all(123)
    state3 = torch.tpu.get_rng_state(device)
    self.assertFalse(torch.equal(state1, state3))

  def test_generator_metadata(self):
    g = torch.Generator(device=torch.accelerator.current_accelerator())
    g.manual_seed(42)
    state = g.get_state()
    self.assertEqual(state.device.type, "cpu")
    self.assertEqual(state.dtype, torch.uint8)
    self.assertEqual(state.shape, (16,))

  def test_generator_set_get_state(self):
    device = torch.accelerator.current_accelerator()
    g = torch.Generator(device=device)
    g.manual_seed(42)
    state1 = g.get_state()

    # Generating random numbers should change the state.
    torch.randn(10, generator=g, device=device)
    state2 = g.get_state()
    self.assertFalse(torch.equal(state1, state2))

    # Setting the state back should restore it.
    g.set_state(state1)
    state3 = g.get_state()
    self.assertTrue(torch.equal(state1, state3))

  def test_generator_manual_seed(self):
    device = torch.accelerator.current_accelerator()
    g = torch.Generator(device=device)

    g.manual_seed(42)
    val1 = torch.randn(10, generator=g, device=device)
    g.manual_seed(42)
    val2 = torch.randn(10, generator=g, device=device)
    self.assertTrue(torch.equal(val1, val2))

  def test_set_rng_state_error_tpu(self):
    device = torch.accelerator.current_accelerator()
    state = torch.tpu.get_rng_state(device).to(device)
    with self.assertRaisesRegex(
        RuntimeError, "expect rng state to be a torch.ByteTensor"
    ):
      torch.tpu.set_rng_state(state, device)

  def test_generator_set_state_error_tpu(self):
    device = torch.accelerator.current_accelerator()
    g = torch.Generator(device=device)
    state = g.get_state().to(device)
    with self.assertRaisesRegex(
        RuntimeError, "expect rng state to be a torch.ByteTensor"
    ):
      g.set_state(state)


if __name__ == "__main__":
  absltest.main()
