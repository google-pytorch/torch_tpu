# Copyright 2025 Google LLC
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

import collections.abc

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_fixtures
from tests import seed_test_utils


class TestFixturesTest(seed_test_utils.RepeatableTest):
  """Tests for modules in test_fixtures.py."""

  def _assert_linear_has_int_weights(self, linear: torch.nn.Linear) -> None:
    """Asserts that a linear layer's weights and biases are all integers."""
    self.assertTrue(
        (linear.weight.data % 1 == 0).all(),
        "linear's weight not all ints",
    )
    self.assertTrue(
        (linear.bias.data % 1 == 0).all(),
        "linear's bias not all ints",
    )

  def test_int_weight_init_mixin_on_simple_module(self):
    """Tests the IntWeightInitMixin on the SimpleModule."""
    # Arrange
    model = test_fixtures.SimpleModule()
    model.initialize_weights_to_ints()

    # Assert weights and biases are integers.
    self._assert_linear_has_int_weights(model.linear1)
    self._assert_linear_has_int_weights(model.linear2)

  def test_int_weight_init_mixin_on_container_module(self):
    """Tests the IntWeightInitMixin on the ContainerModule."""
    # Arrange
    model = test_fixtures.ContainerModule()
    model.initialize_weights_to_ints()

    # Assert weights and biases are integers.
    self._assert_linear_has_int_weights(model.linear3)

  def test_int_weight_init_mixin_on_singleton_math_module(self):
    """Tests the IntWeightInitMixin on the SingletonMathModule."""
    # Arrange
    model = test_fixtures.SingletonMathModule()
    model.initialize_weights_to_ints()

    # Assert weights and biases are integers.
    self._assert_linear_has_int_weights(model.layer1)
    self._assert_linear_has_int_weights(model.layer2)
    self._assert_linear_has_int_weights(model.layer3)

  def test_int_weight_init_mixin_on_module_list_module(self):
    """Tests the IntWeightInitMixin on the ModuleListModule."""
    # Arrange
    model = test_fixtures.ModuleListModule()
    model.initialize_weights_to_ints()

    # Assert weights and biases are integers.
    linear1, _, linear2 = model.layers
    self._assert_linear_has_int_weights(linear1)  # pyrefly: ignore[bad-argument-type]
    self._assert_linear_has_int_weights(linear2)  # pyrefly: ignore[bad-argument-type]

  def test_unusual_sequence_not_sequence(self):
    """Tests that UnusualSequence is not a collections.abc.Sequence."""
    seq = test_fixtures.UnusualSequence()
    self.assertNotIsInstance(seq, collections.abc.Sequence)

  def test_unusual_sequence_is_stack(self):
    """Tests the LIFO behavior of UnusualSequence."""
    seq = test_fixtures.UnusualSequence()
    object1, object2 = object(), object()
    seq.push(object1)
    seq.push(object2)
    self.assertIs(seq.pop(), object2)
    self.assertIs(seq.pop(), object1)

  def test_unusual_sequence_pop_empty(self):
    """Tests popping from an empty UnusualSequence raises IndexError."""
    seq = test_fixtures.UnusualSequence()
    with self.assertRaisesWithLiteralMatch(IndexError, "pop from empty list"):
      seq.pop()

  def test_identity_module(self):
    """Tests the Identity module."""
    # Arrange
    identity_module = test_fixtures.Identity()
    test_tensor = torch.randn(3, 4)
    # Act
    output = identity_module(test_tensor)
    # Assert
    self.assertIs(output, test_tensor)

  def test_crash_on_forward(self):
    """Tests the CrashOnForward module."""
    # Arrange
    crash_module = test_fixtures.CrashOnForward()
    test_tensor = torch.randn(3, 4)
    msg = crash_module.crash_message

    # Assert
    with self.assertRaisesWithLiteralMatch(ValueError, msg):
      # Act
      crash_module.crash = True
      crash_module(test_tensor)

  def test_crash_on_backward(self):
    """Tests the CrashOnBackward module."""
    # Arrange
    crash_module = test_fixtures.CrashOnBackward()
    test_tensor = torch.randn(3, 4)
    test_tensor.requires_grad = True
    msg = test_fixtures.CrashBackwardFunction.crash_message

    # Assert
    with self.assertRaisesWithLiteralMatch(ValueError, msg):
      # Act
      crash_module.crash = True
      # TODO(b/505036496): Investigate autograd slowness after autoload.
      test_fixtures.CrashBackwardFunction.backward(None, None)


if __name__ == "__main__":
  absltest.main()
