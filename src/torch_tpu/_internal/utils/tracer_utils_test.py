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

from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_fixtures
from torch_tpu._internal.utils import test_utils
from torch_tpu._internal.utils import tracer_utils
from torch_tpu._internal.utils import utils
from tests import seed_test_utils


class AllTest(seed_test_utils.RepeatableTest):
  """Tests the OpTracer and ActivationTracer functions."""

  def test_pformat_op_tracer(self):
    """Tests the pformat_op_tracer function."""
    # Arrange
    x = torch.tensor(2.0)

    # Act
    with utils.OpTracer() as tracer:
      _ = x**0.5

    # Assert
    self.assertIn("pow", "".join(tracer.ops_log["aten"].keys()))

  def test_pformat_activation_tracer_zero_children(self):
    """Tests pformat_activation_tracer on a single layer model."""
    # Arrange
    model = torch.nn.Linear(10, 1)
    arg = torch.randn(3, 7, 10)

    # Act
    with utils.ActivationTracer(model) as tracer:
      model(arg)
    formatted_data = tracer_utils.pformat_activation_tracer(tracer)

    # Assert
    self.assertIn("+- Linear", formatted_data)
    self.assertIn("(3, 7, 10)", formatted_data)
    self.assertIn("|  Args[0]:", formatted_data)
    self.assertIn("|  Output:", formatted_data)
    self.assertIn("+- Linear END", formatted_data)

  def test_pformat_activation_tracer_one_child(self):
    """Tests pformat_activation_tracer with one child."""
    # Arrange
    model = test_fixtures.NestedAddModule()
    arg_left = torch.randn(1, 10)
    arg_right = torch.randn(1, 10)

    # Act
    with utils.ActivationTracer(model) as tracer:
      model(arg_left, arg_right)

    formatted_data = tracer_utils.pformat_activation_tracer(tracer)

    logging.info("Formatted Data: \n%s", formatted_data)

    self.assertIn("ActivationTracer Collected Data:", formatted_data)
    self.assertIn("+- NestedAddModule (#0)", formatted_data)
    self.assertIn(
        f"|  Args[0]:   {utils.get_tensor_summary(arg_left)}",
        formatted_data,
    )
    self.assertIn(
        f"|  Args[1]:   {utils.get_tensor_summary(arg_right)}",
        formatted_data,
    )
    self.assertIn("+- NestedAddModule END (#3)", formatted_data)
    self.assertIn("|  +- AddModule (#1)", formatted_data)
    self.assertIn(
        f"|  |  Args[0]:   {utils.get_tensor_summary(arg_left)}",
        formatted_data,
    )
    self.assertIn(
        f"|  |  Args[1]:   {utils.get_tensor_summary(arg_right)}",
        formatted_data,
    )
    self.assertIn("|  +- AddModule END (#2)", formatted_data)

  def test_pformat_activation_tracer_module_list(self):
    """Tests pformat_activation_tracer with a ModuleList."""
    # Arrange
    model = test_fixtures.ModuleListModule()
    arg = torch.randn(1, 10)

    # Act
    with utils.ActivationTracer(model) as tracer:
      model(arg)
    formatted_data = tracer_utils.pformat_activation_tracer(tracer)
    logging.info("Formatted Data: \n%s", formatted_data)

    # Assert
    self.assertIn("+- ModuleListModule", formatted_data)
    self.assertIn("|  Args[0]:   (1, 10)", formatted_data)
    self.assertIn("|  Output:    (1, 2)", formatted_data)
    self.assertIn("+- ModuleListModule END", formatted_data)
    self.assertIn("|  +- Linear", formatted_data)
    self.assertIn("|  |  Args[0]:   (1, 10)", formatted_data)
    self.assertIn("|  |  Output:    (1, 5)", formatted_data)
    self.assertIn("|  +- Linear END", formatted_data)
    self.assertIn("|  +- ReLU", formatted_data)
    self.assertIn("|  |  Args[0]:   (1, 5)", formatted_data)
    self.assertIn("|  |  Output:    (1, 5)", formatted_data)
    self.assertIn("|  +- ReLU END", formatted_data)
    self.assertIn("|  +- Linear", formatted_data)
    self.assertIn("|  |  Args[0]:   (1, 5)", formatted_data)
    self.assertIn("|  |  Output:    (1, 2)", formatted_data)
    self.assertIn("|  +- Linear END", formatted_data)

  def test_pformat_activation_tracer_nested_children(self):
    """Tests pformat_activation_tracer with a complex model."""
    # Arrange
    model = test_fixtures.ContainerModule()
    arg = torch.randn(1, 10)

    # Act
    with utils.ActivationTracer(model) as tracer:
      model(arg)
    formatted_data = tracer_utils.pformat_activation_tracer(tracer)
    logging.info("Formatted Data: \n%s", formatted_data)

    # Assert
    self.assertIn("ActivationTracer Collected Data:", formatted_data)
    self.assertIn("+- ContainerModule", formatted_data)
    self.assertIn("|  Args[0]:   (1, 10)", formatted_data)
    self.assertIn("|  Output:    (1, 1)", formatted_data)
    self.assertIn("+- ContainerModule END", formatted_data)
    self.assertIn("|  +- SimpleModule", formatted_data)
    self.assertIn("|  |  Args[0]:   (1, 10)", formatted_data)
    self.assertIn("|  |  +- Linear", formatted_data)
    self.assertIn("|  |  |  Args[0]:   (1, 10)", formatted_data)
    self.assertIn("|  |  |  Output:    (1, 5)", formatted_data)
    self.assertIn("|  |  +- Linear END", formatted_data)
    self.assertIn("|  |  +- ReLU", formatted_data)
    self.assertIn("|  |  |  Args[0]:   (1, 5)", formatted_data)
    self.assertIn("|  |  |  Output:    (1, 5)", formatted_data)
    self.assertIn("|  |  +- ReLU END", formatted_data)
    self.assertIn("|  |  +- AddModule", formatted_data)
    self.assertIn("|  |  |  Args[0]:   (1, 5)", formatted_data)
    self.assertIn("|  |  |  Args[1]:   (1, 5)", formatted_data)
    self.assertIn("|  |  |  Output:    (1, 5)", formatted_data)
    self.assertIn("|  |  +- AddModule END", formatted_data)
    self.assertIn("|  |  +- Linear", formatted_data)
    self.assertIn("|  |  |  Args[0]:   (1, 5)", formatted_data)
    self.assertIn("|  |  |  Output:    (1, 2)", formatted_data)
    self.assertIn("|  |  +- Linear END", formatted_data)
    self.assertIn("|  |  Output[0]: (1, 2)", formatted_data)
    self.assertIn("|  |  Output[1]: (1, 2)", formatted_data)
    self.assertIn("|  +- SimpleModule END", formatted_data)
    self.assertIn("|  +- Linear", formatted_data)
    self.assertIn("|  |  Args[0]:   (1, 2)", formatted_data)
    self.assertIn("|  |  Output:    (1, 1)", formatted_data)
    self.assertIn("|  +- Linear END", formatted_data)

  def test_pformat_activation_tracer_on_non_tensor(self):
    """Tests pformat_activation_tracer on a non-tensor input."""
    # Arrange
    model = test_fixtures.NonTensorModule()
    arg = "example of string input"

    # Act
    with utils.ActivationTracer(model) as tracer:
      model(arg)
    formatted_data = tracer_utils.pformat_activation_tracer(tracer)
    logging.info("Formatted Data: \n%s", formatted_data)

    # Assert
    self.assertIn("|  Args[0]:   str", formatted_data)
    self.assertIn("|  Output:    str", formatted_data)

  def test_pformat_activation_tracer_with_kwargs(self):
    """Tests pformat_activation_tracer with kwargs."""
    # Arrange
    model = torch.nn.Linear(10, 1)
    arg = torch.randn(1, 10)

    # Act
    with utils.ActivationTracer(model) as tracer:
      model(input=arg)
    formatted_data = tracer_utils.pformat_activation_tracer(tracer)

    # Assert
    logging.info("Formatted Data: \n%s", formatted_data)
    self.assertIn("|  Args[input]:", formatted_data)
    self.assertIn(utils.get_tensor_summary(arg), formatted_data)
    self.assertIn("|  Output:", formatted_data)
    self.assertIn("+- Linear END", formatted_data)

  def test_replay_on_linear_with_no_change_has_same_output(self):
    """Tests replay on a simple module with no change."""
    # Arrange
    model = torch.nn.Linear(10, 2)
    arg = torch.randn(1, 10)
    with utils.ActivationTracer(model) as tracer:
      model(arg)
    log, _ = tracer.forward_log, tracer.forward_pre_log

    with self.subTest("replay"):
      # Act
      replayed_log = tracer_utils.replay_log(log, "cpu")

      # Assert
      test_utils.assert_close(replayed_log[0]["output"], log[0]["output"])  # pyrefly: ignore[bad-index]

  def test_replay_on_linear_with_change_returns_different_value(self):
    """Tests replay on a simple module with a changed weight."""
    # Arrange
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.0)
    with utils.ActivationTracer(model) as tracer:
      model(torch.tensor([3.0]))
    log, _ = tracer.forward_log, tracer.forward_pre_log

    # Internal check:
    test_utils.assert_close(log[0]["output"], torch.tensor([6.0]))

    with self.subTest("replay"):
      # Act
      model.weight.data.fill_(5.0)
      replayed_log = tracer_utils.replay_log(log, "cpu")

      # Assert
      test_utils.assert_close(log[0]["output"], torch.tensor([6.0]))
      test_utils.assert_close(replayed_log[0]["output"], torch.tensor([15.0]))  # pyrefly: ignore[bad-index]

    with self.subTest("pformat"):
      text = tracer_utils.pformat_replay(log, _, replayed_log)
      delta_line = (
          "Delta:         Output:    (1,)  "
          "Min: 9.00  Max: 9.00  Mean: 9.00  Var: 0.00"
      )
      self.assertIn(delta_line, text)

  def test_replay_on_complex_module_without_change(self):
    """Tests replay_log on a complex module with no change."""
    # Arrange.
    model = test_fixtures.ContainerModule()
    model.initialize_weights_to_ints()
    arg = torch.randint(-5, 5, (1, 10), dtype=torch.float32)

    # Act.
    with utils.ActivationTracer(model) as tracer:
      model(arg)
    log, _ = tracer.forward_log, tracer.forward_pre_log
    replayed_log = tracer_utils.replay_log(log, "cpu")

    # Assert.
    for event, replayed_event in zip(log, replayed_log):
      test_utils.assert_close(event["output"], replayed_event["output"])  # pyrefly: ignore[bad-index]

  def test_replay_on_complex_module_with_change(self):
    """Tests replay_log on a complex module with one spot change."""
    # Arrange.
    model = test_fixtures.ContainerModule()
    model.initialize_weights_to_ints()

    arg = torch.ones((1, 10), dtype=model.linear3.weight.dtype)

    with utils.ActivationTracer(model) as tracer:
      model(arg)

    log, _ = tracer.forward_log, tracer.forward_pre_log

    # Act.
    # Mock a change to the way a new "device" does multiplication.
    weight = model.linear3.weight
    weight.data = torch.randn_like(weight.data)

    replayed_log = tracer_utils.replay_log(log, "cpu")

    # Assert.
    event, replayed_event = log[-2], replayed_log[-2]  # Linear3.
    self.assertNotEqual(event["output"].sum(), replayed_event["output"].sum())  # pyrefly: ignore[bad-index]

  def test_replay_log_includes_exception(self):
    """Tests replay_log includes an exception when one occurs during replay."""
    # Arrange
    model = test_fixtures.CrashOnForward()
    with utils.ActivationTracer(model) as tracer:
      model(object())

    # Act
    model.crash = True
    log, _ = tracer.forward_log, tracer.forward_pre_log
    replayed_log = tracer_utils.replay_log(log, "cpu")

    # Assert
    self.assertLen(replayed_log, 1)
    self.assertIsInstance(replayed_log[0], ValueError)


if __name__ == "__main__":
  absltest.main()
