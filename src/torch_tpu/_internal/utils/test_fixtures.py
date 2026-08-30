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
"""Shared nn.Module subclasses for testing."""
import abc
from typing import Any, TypeVar

import torch


class AddModule(torch.nn.Module):
  """A module that adds two tensors."""

  def forward(self, x, y):
    return x + y


class NestedAddModule(torch.nn.Module):
  """A module that adds two tensors."""

  def __init__(self):
    super().__init__()
    self.add1 = AddModule()

  def forward(self, x, y):
    return self.add1(x, y)


class IntWeightInitMixin(abc.ABC):
  """Mixin for modules to init Linear weights to integers values.

  The dtype remains a floating point format, but forcing the weights to be
  integers can be useful for eliminating rounding errors from floating point
  formats.
  """

  @abc.abstractmethod
  def initialize_weights_to_ints(self) -> None:
    """Initializes all Linear layer weights and biases to integers.

    Note that the dtype remains a floating point format, but forcing the weights
    to be integers can be useful for eliminating rounding errors from floating
    point formats.
    """

  def _init_linear(self, linear_module: torch.nn.Linear) -> None:
    """Helper to initialize a single Linear layer's weights to integers."""
    linear_module.weight.data = torch.randint(
        -5, 5, linear_module.weight.shape, dtype=torch.float32
    )
    if linear_module.bias is not None:
      linear_module.bias.data = torch.randint(
          -5, 5, linear_module.bias.shape, dtype=torch.float32
      )


class SimpleModule(torch.nn.Module, IntWeightInitMixin):
  """A simple module for testing."""

  def __init__(self):
    super().__init__()
    self.linear1 = torch.nn.Linear(10, 5)
    self.relu = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(5, 2)
    self.adder = AddModule()

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.adder(x, torch.ones_like(x))
    lin2_out = self.linear2(x)
    return lin2_out, lin2_out * 2.0

  def initialize_weights_to_ints(self) -> None:
    """See IntWeightInitMixin."""
    self._init_linear(self.linear1)
    self._init_linear(self.linear2)


class ContainerModule(torch.nn.Module, IntWeightInitMixin):
  """A nested module for testing."""

  def __init__(self):
    super().__init__()
    self.simple = SimpleModule()
    self.linear3 = torch.nn.Linear(2, 1)

  def forward(self, x):
    x, _ = self.simple(x)
    return self.linear3(x)

  def initialize_weights_to_ints(self) -> None:
    """See IntWeightInitMixin."""
    self.simple.initialize_weights_to_ints()
    self._init_linear(self.linear3)


class NonTensorModule(torch.nn.Module):
  """A module that returns a non-tensor value."""

  def forward(self, x):
    return "hello"


class SingletonMathModule(torch.nn.Module, IntWeightInitMixin):
  """A module with multiple layers processing [1, 1] tensors."""

  def __init__(self):
    super().__init__()
    self.layer1 = torch.nn.Linear(1, 1)
    self.sigmoid1 = torch.nn.Sigmoid()
    self.layer2 = torch.nn.Linear(1, 1)
    self.sigmoid2 = torch.nn.Sigmoid()
    self.layer3 = torch.nn.Linear(1, 1)
    self.sigmoid3 = torch.nn.Sigmoid()

  def forward(self, x):
    x = self.sigmoid1(self.layer1(x))
    x = self.sigmoid2(self.layer2(x))
    x = self.sigmoid3(self.layer3(x))
    return x

  def initialize_weights_to_ints(self) -> None:
    """See IntWeightInitMixin."""
    self._init_linear(self.layer1)
    self._init_linear(self.layer2)
    self._init_linear(self.layer3)


class ModuleListModule(torch.nn.Module, IntWeightInitMixin):
  """A module that uses a ModuleList."""

  def __init__(self):
    super().__init__()
    self.layers = torch.nn.ModuleList(
        [torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)]
    )

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def initialize_weights_to_ints(self) -> None:
    """See IntWeightInitMixin."""
    # Explicitly initialize Linear layers in the ModuleList
    linear1, _, linear2 = self.layers
    self._init_linear(linear1)  # pyrefly: ignore[bad-argument-type]
    self._init_linear(linear2)  # pyrefly: ignore[bad-argument-type]


class UnusualSequence:
  """A container that isn't a subclass of collections.abc.Sequence."""

  def __init__(self):
    self.items = []

  def push(self, item: Any) -> None:
    """Pushes an item to the sequence.

    Args:
      item: The item to push.
    """
    self.items.append(item)

  def pop(self) -> Any:
    """Pops an item from the sequence.

    Returns:
      The popped item.

    Raises:
      IndexError: If the sequence is empty.
    """
    return self.items.pop()


_T = TypeVar("_T")


class Identity(torch.nn.Module):
  """An identity module."""

  def forward(self, x: _T) -> _T:
    return x


class CrashOnForward(torch.nn.Module):
  """A module that crashes on forward.

  Attributes:
    crash_message: The message to raise in the forward pass.
    crash: Whether to crash in the forward pass.
  """

  def __init__(self):
    super().__init__()
    self.crash = False
    self.crash_message = "Intentional crash"

  def forward(self, x: _T) -> _T:
    if self.crash:
      raise ValueError(self.crash_message)
    return x


class CrashBackwardFunction(torch.autograd.Function):
  """A custom autograd function that crashes on backward pass.

  Attributes:
    crash_message (class attribute): The message to raise in the backward pass.
    crash (class attribute): Whether to crash in the backward pass.
  """

  crash: bool = True
  crash_message: str = "Intentional crash"

  # Cannot avoid staticmethod because torch.autograd.Function requires it.
  @staticmethod
  def forward(ctx: Any, x: _T) -> _T:
    """The forward pass is an identity function.

    Args:
      ctx: The autograd context provided by torch framework. Ignored.
      x: The input tensor.

    Returns:
      The input tensor.
    """
    del ctx
    return x

  # Cannot avoid staticmethod because torch.autograd.Function requires it.
  @staticmethod
  def backward(ctx: Any, grad_output: Any) -> None:  # pyrefly: ignore[bad-override]
    """The backward pass intentionally raises an exception.

    Args:
      ctx: The autograd context provided by torch framework. Ignored.
      grad_output: The gradient of the output tensor. Ignored.

    Raises:
      ValueError: Always raised.
    """
    del ctx, grad_output
    if CrashBackwardFunction.crash:
      raise ValueError(CrashBackwardFunction.crash_message)


class CrashOnBackward(torch.nn.Module):
  """A module that crashes on backward pass."""

  def forward(self, x: _T) -> _T:
    return CrashBackwardFunction.apply(x)
