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

from unittest import mock

from absl.testing import absltest
import torch
import torch.fx
from torch_tpu._internal.compile import split_compiler
from torch_tpu._internal.compile import tpu_torch_compile


class SplitCompilerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.all_reduce = torch.ops._c10d_functional.all_reduce.default
    self.wait_tensor = torch.ops._c10d_functional.wait_tensor.default
    self.add = torch.ops.aten.add.Tensor

  @mock.patch.object(
      tpu_torch_compile, "get_materialize_collective_tensors_env_value"
  )
  def test_should_split_module_env_var_disabled(self, mock_get_env):
    mock_get_env.return_value = False

    g = torch.fx.Graph()
    x = g.placeholder("x")
    col1 = g.call_function(self.all_reduce, args=(x, "sum", "0", [0], 0))
    wait1 = g.call_function(self.wait_tensor, args=(col1,))
    col2 = g.call_function(self.all_reduce, args=(wait1, "sum", "0", [0], 0))
    wait2 = g.call_function(self.wait_tensor, args=(col2,))
    g.output(wait2)

    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    self.assertFalse(split_compiler.should_split_module(gm))

  @mock.patch.object(
      tpu_torch_compile, "get_materialize_collective_tensors_env_value"
  )
  def test_should_split_module_no_collectives(self, mock_get_env):
    mock_get_env.return_value = True

    g = torch.fx.Graph()
    x = g.placeholder("x")
    y = g.call_function(self.add, args=(x, 1))
    g.output(y)

    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    self.assertFalse(split_compiler.should_split_module(gm))

  @mock.patch.object(
      tpu_torch_compile, "get_materialize_collective_tensors_env_value"
  )
  def test_should_split_module_one_collective_no_compute(self, mock_get_env):
    mock_get_env.return_value = True

    g = torch.fx.Graph()
    x = g.placeholder("x")
    col = g.call_function(self.all_reduce, args=(x, "sum", "0", [0], 0))
    wait = g.call_function(self.wait_tensor, args=(col,))
    g.output(wait)

    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    self.assertFalse(split_compiler.should_split_module(gm))

  @mock.patch.object(
      tpu_torch_compile, "get_materialize_collective_tensors_env_value"
  )
  def test_should_split_module_one_collective_with_compute(self, mock_get_env):
    mock_get_env.return_value = True

    g = torch.fx.Graph()
    x = g.placeholder("x")
    col = g.call_function(self.all_reduce, args=(x, "sum", "0", [0], 0))
    wait = g.call_function(self.wait_tensor, args=(col,))
    out = g.call_function(self.add, args=(wait, 1))
    g.output(out)

    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    self.assertTrue(split_compiler.should_split_module(gm))

  @mock.patch.object(
      tpu_torch_compile, "get_materialize_collective_tensors_env_value"
  )
  def test_should_split_module_multiple_collectives(self, mock_get_env):
    mock_get_env.return_value = True

    g = torch.fx.Graph()
    x = g.placeholder("x")
    col1 = g.call_function(self.all_reduce, args=(x, "sum", "0", [0], 0))
    wait1 = g.call_function(self.wait_tensor, args=(col1,))
    col2 = g.call_function(self.all_reduce, args=(wait1, "sum", "0", [0], 0))
    wait2 = g.call_function(self.wait_tensor, args=(col2,))
    g.output(wait2)

    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    self.assertTrue(split_compiler.should_split_module(gm))


if __name__ == "__main__":
  absltest.main()
