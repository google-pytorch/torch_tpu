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
from torch._subclasses import fake_tensor
from torch_tpu._internal.utils import annotations
from tests import seed_test_utils


def setUpModule():
  # Test internal torch.tpu RNG APIs that PyTorch uses, e.g. in
  # `torch.utils.checkpoint`.
  annotations._internal_apis_allowed_for_testing = True  # pylint: disable=protected-access


class RngTest(seed_test_utils.RepeatableTest):

  def test_get_rng_state_metadata(self):
    device = torch.device("tpu")
    state = torch.tpu.get_rng_state(device)
    self.assertEqual(state.device.type, "cpu")
    self.assertEqual(state.dtype, torch.uint8)
    self.assertEqual(state.shape, (16,))

  def test_set_rng_state(self):
    device = torch.device("tpu")
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
    device = torch.device("tpu")
    torch.tpu.manual_seed_all(42)
    state1 = torch.tpu.get_rng_state(device)

    torch.tpu.manual_seed_all(42)
    state2 = torch.tpu.get_rng_state(device)
    self.assertTrue(torch.equal(state1, state2))

    torch.tpu.manual_seed_all(123)
    state3 = torch.tpu.get_rng_state(device)
    self.assertFalse(torch.equal(state1, state3))

  def test_generator_metadata(self):
    g = torch.Generator(device=torch.device("tpu"))
    g.manual_seed(42)
    state = g.get_state()
    self.assertEqual(state.device.type, "cpu")
    self.assertEqual(state.dtype, torch.uint8)
    self.assertEqual(state.shape, (16,))

  def test_generator_set_get_state(self):
    device = torch.device("tpu")
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
    device = torch.device("tpu")
    g = torch.Generator(device=device)

    g.manual_seed(42)
    val1 = torch.randn(10, generator=g, device=device)
    g.manual_seed(42)
    val2 = torch.randn(10, generator=g, device=device)
    self.assertTrue(torch.equal(val1, val2))

  def test_set_rng_state_error_tpu(self):
    device = torch.device("tpu")
    state = torch.tpu.get_rng_state(device).to(device)
    with self.assertRaisesRegex(
        RuntimeError, "expect rng state to be a torch.ByteTensor"
    ):
      torch.tpu.set_rng_state(state, device)

  def test_generator_set_state_error_tpu(self):
    device = torch.device("tpu")
    g = torch.Generator(device=device)
    state = g.get_state().to(device)
    with self.assertRaisesRegex(
        RuntimeError, "expect rng state to be a torch.ByteTensor"
    ):
      g.set_state(state)

  def test_get_rng_state_in_fake_tensor_mode(self):
    device = torch.device("tpu")
    with fake_tensor.FakeTensorMode(allow_non_fake_inputs=False):
      state = torch.tpu.get_rng_state(device)
      self.assertEqual(state.dtype, torch.uint8)
      self.assertEqual(state.shape, (16,))

  def test_generator_get_state_in_fake_tensor_mode(self):
    device = torch.device("tpu")
    with fake_tensor.FakeTensorMode(allow_non_fake_inputs=False):
      g = torch.Generator(device=device)
      g.manual_seed(42)
      state = g.get_state()
      self.assertEqual(state.dtype, torch.uint8)
      self.assertEqual(state.shape, (16,))

  def test_generator_clone_in_fake_tensor_mode(self):
    device = torch.device("tpu")
    with fake_tensor.FakeTensorMode(allow_non_fake_inputs=False):
      g = torch.Generator(device=device)
      g.manual_seed(42)
      g2 = g.clone_state()
      state = g2.get_state()
      self.assertEqual(state.dtype, torch.uint8)
      self.assertEqual(state.shape, (16,))

  def test_generator_set_state_in_fake_tensor_mode(self):
    device = torch.device("tpu")
    g = torch.Generator(device=device)
    g.manual_seed(42)
    state = g.get_state()
    with fake_tensor.FakeTensorMode(allow_non_fake_inputs=False):
      g.set_state(state)
      state2 = g.get_state()
      self.assertEqual(state2.dtype, torch.uint8)
      self.assertEqual(state2.shape, (16,))

  def test_generator_graphsafe_get_set_state(self):
    device = torch.device("tpu")
    g = torch.Generator(device=device)
    g.manual_seed(42)

    g2 = g.graphsafe_get_state()

    # Generating random numbers with g should change the state of both g and g2.
    torch.randn(10, generator=g, device=device)

    # Since g2 shares the state intrusive pointer, its state matches g's new
    # state.
    self.assertTrue(torch.equal(g.get_state(), g2.get_state()))

    # Now set the state of g to another generator's state
    g3 = torch.Generator(device=device)
    g3.manual_seed(123)

    g.graphsafe_set_state(g3)

    # g's state is now changed to g3's state
    self.assertTrue(torch.equal(g.get_state(), g3.get_state()))

  def test_generator_graphsafe_get_state_in_fake_tensor_mode(self):
    device = torch.device("tpu")
    with fake_tensor.FakeTensorMode(allow_non_fake_inputs=False):
      g = torch.Generator(device=device)
      g.manual_seed(42)
      g2 = g.graphsafe_get_state()
      self.assertIsNotNone(g2)

  def test_default_generators_is_tuple(self):
    self.assertIsInstance(torch.tpu.default_generators, tuple)
    self.assertGreater(torch.accelerator.device_count(), 0)
    self.assertLen(
        torch.tpu.default_generators, torch.accelerator.device_count()
    )
    self.assertIsInstance(torch.tpu.default_generators[0], torch.Generator)

  @parameterized.named_parameters(
      dict(
          testcase_name="inference_mode",
          shape=(10,),
          p=0.5,
          train=False,
          advances=False,
      ),
      dict(
          testcase_name="p_zero",
          shape=(10,),
          p=0.0,
          train=True,
          advances=False,
      ),
      dict(
          testcase_name="p_one",
          shape=(10,),
          p=1.0,
          train=True,
          advances=False,
      ),
      dict(
          testcase_name="numel_zero",
          shape=(0, 10),
          p=0.5,
          train=True,
          advances=False,
      ),
      dict(
          testcase_name="training_mode",
          shape=(10,),
          p=0.5,
          train=True,
          advances=True,
      ),
  )
  def test_dropout_rng_advances(self, shape, p, train, advances):
    device = torch.accelerator.current_accelerator()
    for op_name in ("native_dropout", "dropout", "dropout_"):
      with self.subTest(op_name=op_name):
        x = torch.ones(shape, device=device)
        initial_state = torch.tpu.get_rng_state(device)

        if op_name == "native_dropout":
          out, _ = torch.native_dropout(x, p=p, train=train)
        elif op_name == "dropout":
          out = torch.dropout(x, p=p, train=train)
        else:
          out = torch.dropout_(x, p=p, train=train)
        out.cpu()

        new_state = torch.tpu.get_rng_state(device)
        if advances:
          self.assertFalse(torch.equal(initial_state, new_state))
        else:
          self.assertTrue(torch.equal(initial_state, new_state))

  @parameterized.named_parameters(
      dict(testcase_name="p_zero", shape=(4, 8), p=0.0, train=True),
      dict(testcase_name="eval_mode", shape=(4, 8), p=0.5, train=False),
      dict(testcase_name="numel_zero", shape=(0, 8), p=0.5, train=True),
  )
  def test_dropout_noop_data_ptr_and_autograd(self, shape, p, train):
    device = torch.accelerator.current_accelerator()
    for inplace in (False, True):
      with self.subTest(inplace=inplace):
        x = torch.randn(shape, device=device)
        initial_state = torch.tpu.get_rng_state(device)
        out = torch.nn.functional.dropout(
            x, p=p, training=train, inplace=inplace
        )
        self.assertEqual(out.data_ptr(), x.data_ptr())
        self.assertTrue(
            torch.equal(initial_state, torch.tpu.get_rng_state(device))
        )

        # Verify autograd pass-through (no extra grad_fn node or RNG advance)
        leaf = torch.randn(shape, device=device, requires_grad=True)
        y = leaf * 2.0
        initial_state = torch.tpu.get_rng_state(device)
        out_grad = torch.nn.functional.dropout(
            y, p=p, training=train, inplace=inplace
        )
        self.assertEqual(out_grad.data_ptr(), y.data_ptr())
        self.assertIs(out_grad.grad_fn, y.grad_fn)
        self.assertTrue(
            torch.equal(initial_state, torch.tpu.get_rng_state(device))
        )
        out_grad.sum().backward()
        self.assertTrue(torch.equal(leaf.grad.cpu(), torch.full(shape, 2.0)))

  @parameterized.named_parameters(
      dict(testcase_name="p_zero", shape=(4, 8), p=0.0, train=True),
      dict(testcase_name="eval_mode", shape=(4, 8), p=0.5, train=False),
      dict(testcase_name="numel_zero", shape=(0, 8), p=0.5, train=True),
  )
  def test_native_dropout_noop_mask_and_cow(self, shape, p, train):
    device = torch.accelerator.current_accelerator()
    x = torch.ones(shape, device=device)
    initial_state = torch.tpu.get_rng_state(device)

    out, mask = torch.native_dropout(x, p=p, train=train)
    self.assertTrue(torch.equal(initial_state, torch.tpu.get_rng_state(device)))
    self.assertEqual(mask.dtype, torch.bool)
    self.assertEqual(mask.shape, x.shape)
    self.assertTrue(bool(mask.all().item()))
    self.assertTrue(torch.equal(out.cpu(), x.cpu()))

    # In-place mutation of `out` must not corrupt `x` (copy-on-write safety).
    out.add_(5.0)
    self.assertTrue(torch.equal(x.cpu(), torch.ones(shape)))
    self.assertTrue(torch.equal(out.cpu(), torch.full(shape, 6.0)))


if __name__ == "__main__":
  absltest.main()
