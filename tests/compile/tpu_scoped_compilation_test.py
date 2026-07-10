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

"""Tests for TpuScopedCompilation pass."""

from absl.testing import absltest
import torch
import torch._dynamo.exc as dynamo_exc
import torch.fx
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal.compile.fx_passes import pass_utils
from torch_tpu._internal.compile.fx_passes import tpu_scoped_compilation
from torch_tpu._internal.utils import utils


class PartitionTest(absltest.TestCase):

  def test_cpu_nullary(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        mask1 = torch.zeros((1, 4, 4, 1), device="cpu")
        mask1[0, 0, 0, 0] = 1.0

        mask2 = torch.ones((1, 4, 4, 1), device="cpu")
        mask2[0, 0, 0, 0] = 2.0

        mask1 = mask1.to(x.device)
        mask2 = mask2.to(x.device)
        return x + mask1.sum() + mask2.sum()

    def check_tpu(gm, _):
      nodes = list(gm.graph.nodes)

      self.assertTrue(
          all(
              pass_utils.get_node_device(node) == pass_utils.DeviceId.TPU
              for node in nodes[:-1]
          )
      )
      self.assertLen(nodes, 8)

      # Inputs mask1, mask2, x
      self.assertTrue(all(node.op == "placeholder" for node in nodes[:3]))

      self.assertLen(nodes[3].args, 1)
      self.assertCountEqual(
          (node.target for node in nodes[3:-1]),
          (
              torch.ops.aten.sum.default,
              torch.ops.aten.sum.default,
              torch.ops.aten.add.Tensor,
              torch.ops.aten.add.Tensor,
          ),
      )

      return gm

    compiled = torch.compile(
        Model(), backend=tpu_scoped_compilation.TpuScopedCompilation(check_tpu)
    )
    _ = compiled(torch.tensor([1, 2, 3], device="tpu"))

  def test_cpu_basic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        mask = torch.zeros((1, 4, 4, 1), device="cpu")
        mask[0, 0, 0, 0] = 1.0
        mask = mask.to(x.device)
        return x + mask.sum()

    def check_tpu(gm, _):
      nodes = list(gm.graph.nodes)

      self.assertTrue(
          all(
              pass_utils.get_node_device(node) == pass_utils.DeviceId.TPU
              for node in nodes[:-1]
          )
      )
      self.assertLen(nodes, 5)
      self.assertTrue(all(node.op == "placeholder" for node in nodes[:2]))

      self.assertCountEqual(
          (node.target for node in nodes[2:-1]),
          (
              torch.ops.aten.sum.default,
              torch.ops.aten.add.Tensor,
          ),
      )
      return gm

    compiled = torch.compile(
        Model(), backend=tpu_scoped_compilation.TpuScopedCompilation(check_tpu)
    )
    _ = compiled(torch.tensor([1, 2, 3], device="tpu"))

  def test_no_cpu(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        return x + y.sum()

    def check_tpu(gm, _):
      nodes = list(gm.graph.nodes)
      self.assertTrue(
          all(
              pass_utils.get_node_device(node) == pass_utils.DeviceId.TPU
              for node in nodes[:-1]
          )
      )
      self.assertLen(nodes, 5)
      return gm

    compiled = torch.compile(
        Model(), backend=tpu_scoped_compilation.TpuScopedCompilation(check_tpu)
    )
    _ = compiled(
        torch.tensor([1, 2, 3], device="tpu"),
        torch.tensor([4, 5, 6], device="tpu"),
    )

  def test_mixed_inputs(self):
    class Model(torch.nn.Module):

      def forward(self, x_tpu, y_cpu):
        y_cpu = y_cpu * 2.0
        y_tpu = y_cpu.to(x_tpu.device)
        return x_tpu + y_tpu

    def check_tpu(gm, _):
      nodes = list(gm.graph.nodes)
      self.assertTrue(
          all(
              pass_utils.get_node_device(node) == pass_utils.DeviceId.TPU
              for node in nodes[:-1]
          )
      )
      self.assertLen(nodes, 4)
      self.assertTrue(all(node.op == "placeholder" for node in nodes[:2]))
      self.assertEqual(nodes[2].target, torch.ops.aten.add.Tensor)
      return gm

    compiled = torch.compile(
        Model(), backend=tpu_scoped_compilation.TpuScopedCompilation(check_tpu)
    )
    _ = compiled(
        torch.tensor([1.0, 2.0, 3.0], device="tpu"),
        torch.tensor([4.0, 5.0, 6.0], device="cpu"),
    )


class TpuCompilationTest(absltest.TestCase):

  def test_no_cpu(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        return x + y.sum()

    model = Model()
    x = torch.tensor([1, 2, 3], device="tpu")
    y = torch.tensor([4, 5, 6], device="tpu")

    eager_out = model(x, y)

    compiled = torch.compile(
        model,
        backend=torch_tpu_compile.TpuBackend(),
    )
    compiled_out = compiled(x, y)

    self.assertEqual(compiled_out.device.type, "tpu")
    utils.assert_close(compiled_out, eager_out)

  def test_cpu_nullary(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        mask1 = torch.zeros((1, 4, 4, 1), device="cpu")
        mask1[0, 0, 0, 0] = 1.0

        mask2 = torch.ones((1, 4, 4, 1), device="cpu")
        mask2[0, 0, 0, 0] = 2.0

        mask1 = mask1.to(x.device)
        mask2 = mask2.to(x.device)
        return x + mask1.sum() + mask2.sum()

    model = Model()
    x = torch.tensor([1, 2, 3], device="tpu")

    eager_out = model(x)

    compiled = torch.compile(
        model,
        backend=torch_tpu_compile.TpuBackend(),
    )
    compiled_out = compiled(x)

    self.assertEqual(compiled_out.device.type, "tpu")
    utils.assert_close(compiled_out, eager_out)

  def test_cpu_basic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        mask = torch.zeros((1, 4, 4, 1), device="cpu")
        mask[0, 0, 0, 0] = 1.0
        mask = mask.to(x.device)
        return x + mask.sum()

    model = Model()
    x = torch.tensor([1, 2, 3], device="tpu")

    eager_out = model(x)

    compiled = torch.compile(
        model,
        backend=torch_tpu_compile.TpuBackend(),
    )
    compiled_out = compiled(x)

    self.assertEqual(compiled_out.device.type, "tpu")
    utils.assert_close(compiled_out, eager_out)

  def test_mixed_inputs(self):
    class Model(torch.nn.Module):

      def forward(self, x_tpu, y_cpu):
        y_cpu = y_cpu * 2.0
        y_tpu = y_cpu.to(x_tpu.device)
        return x_tpu + y_tpu

    model = Model()
    x_tpu = torch.tensor([1.0, 2.0, 3.0], device="tpu")
    y_cpu = torch.tensor([4.0, 5.0, 6.0], device="cpu")

    eager_out = model(x_tpu, y_cpu)

    compiled = torch.compile(
        model,
        backend=torch_tpu_compile.TpuBackend(),
    )
    compiled_out = compiled(x_tpu, y_cpu)

    self.assertEqual(compiled_out.device.type, "tpu")
    utils.assert_close(compiled_out, eager_out)

  def test_no_cpu_train(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        return x + y.sum()

    model = Model()
    x_eager = torch.tensor([1.0, 2.0, 3.0], device="tpu", requires_grad=True)
    y_eager = torch.tensor([4.0, 5.0, 6.0], device="tpu", requires_grad=True)

    def train_step(x, y):
      out = model(x, y)
      loss = out.sum()
      loss.backward()
      return loss.detach()

    eager_loss = train_step(x_eager, y_eager)
    eager_x_grad = x_eager.grad.clone()
    eager_y_grad = y_eager.grad.clone()

    x_compiled = torch.tensor([1.0, 2.0, 3.0], device="tpu", requires_grad=True)
    y_compiled = torch.tensor([4.0, 5.0, 6.0], device="tpu", requires_grad=True)

    compiled_train_step = torch.compile(
        train_step,
        backend=torch_tpu_compile.TpuBackend(),
    )
    compiled_loss = compiled_train_step(x_compiled, y_compiled)

    self.assertEqual(compiled_loss.device.type, "tpu")
    utils.assert_close(compiled_loss, eager_loss)
    utils.assert_close(x_compiled.grad, eager_x_grad)
    utils.assert_close(y_compiled.grad, eager_y_grad)

  def test_cpu_nullary_train(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        mask1 = torch.zeros((1, 4, 4, 1), device="cpu")
        mask1[0, 0, 0, 0] = 1.0

        mask2 = torch.ones((1, 4, 4, 1), device="cpu")
        mask2[0, 0, 0, 0] = 2.0

        mask1 = mask1.to(x.device)
        mask2 = mask2.to(x.device)
        return x + mask1.sum() + mask2.sum()

    model = Model()
    x_eager = torch.tensor([1.0, 2.0, 3.0], device="tpu", requires_grad=True)

    def train_step(x):
      out = model(x)
      loss = out.sum()
      loss.backward()
      return loss.detach()

    eager_loss = train_step(x_eager)
    eager_x_grad = x_eager.grad.clone()

    x_compiled = torch.tensor([1.0, 2.0, 3.0], device="tpu", requires_grad=True)

    compiled_train_step = torch.compile(
        train_step,
        backend=torch_tpu_compile.TpuBackend(),
    )
    compiled_loss = compiled_train_step(x_compiled)

    self.assertEqual(compiled_loss.device.type, "tpu")
    utils.assert_close(compiled_loss, eager_loss)
    utils.assert_close(x_compiled.grad, eager_x_grad)

  def test_cpu_basic_train(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        mask = torch.zeros((1, 4, 4, 1), device="cpu")
        mask[0, 0, 0, 0] = 1.0
        mask = mask.to(x.device)
        return x + mask.sum()

    model = Model()
    x_eager = torch.tensor([1.0, 2.0, 3.0], device="tpu", requires_grad=True)

    def train_step(x):
      out = model(x)
      loss = out.sum()
      loss.backward()
      return loss.detach()

    eager_loss = train_step(x_eager)
    eager_x_grad = x_eager.grad.clone()

    x_compiled = torch.tensor([1.0, 2.0, 3.0], device="tpu", requires_grad=True)

    compiled_train_step = torch.compile(
        train_step,
        backend=torch_tpu_compile.TpuBackend(),
    )
    compiled_loss = compiled_train_step(x_compiled)

    self.assertEqual(compiled_loss.device.type, "tpu")
    utils.assert_close(compiled_loss, eager_loss)
    utils.assert_close(x_compiled.grad, eager_x_grad)

  def test_mixed_inputs_train(self):
    class Model(torch.nn.Module):

      def forward(self, x_tpu, y_cpu):
        y_cpu = y_cpu * 2.0
        y_tpu = y_cpu.to(x_tpu.device)
        return x_tpu + y_tpu

    model = Model()
    x_eager = torch.tensor([1.0, 2.0, 3.0], device="tpu", requires_grad=True)
    y_eager = torch.tensor([4.0, 5.0, 6.0], device="cpu", requires_grad=True)

    def train_step(x_tpu, y_cpu):
      out = model(x_tpu, y_cpu)
      loss = out.sum()
      loss.backward()
      return loss.detach()

    eager_loss = train_step(x_eager, y_eager)
    eager_x_grad = x_eager.grad.clone()
    eager_y_grad = y_eager.grad.clone()

    x_compiled = torch.tensor([1.0, 2.0, 3.0], device="tpu", requires_grad=True)
    y_compiled = torch.tensor([4.0, 5.0, 6.0], device="cpu", requires_grad=True)

    compiled_train_step = torch.compile(
        train_step,
        backend=torch_tpu_compile.TpuBackend(),
    )
    compiled_loss = compiled_train_step(x_compiled, y_compiled)

    self.assertEqual(compiled_loss.device.type, "tpu")
    utils.assert_close(compiled_loss, eager_loss)
    utils.assert_close(x_compiled.grad, eager_x_grad)
    utils.assert_close(y_compiled.grad, eager_y_grad)

  def test_tpu_to_cpu_forward(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        x = x + 1.0
        x_cpu = x.to("cpu")
        return x_cpu + 2.0

    model = Model()
    x_eager = torch.tensor([1.0, 2.0, 3.0], device="tpu")
    eager_out = model(x_eager)

    x_compiled = torch.tensor([1.0, 2.0, 3.0], device="tpu")
    compiled = torch.compile(
        model,
        backend=torch_tpu_compile.TpuBackend(),
    )
    compiled_out = compiled(x_compiled)

    self.assertEqual(compiled_out.device.type, "cpu")
    utils.assert_close(compiled_out, eager_out)

  def test_multiple_tpu_partitions_error(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        x = x + 1.0
        x_cpu = x.to("cpu")
        x_cpu = x_cpu + 2.0
        x_tpu = x_cpu.to("tpu")
        return x_tpu + 3.0

    model = Model()
    x = torch.tensor([1.0, 2.0, 3.0], device="tpu")

    compiled = torch.compile(
        model,
        backend=torch_tpu_compile.TpuBackend(),
    )
    with self.assertRaisesRegex(
        dynamo_exc.BackendCompilerFailed, "cycle exists between partitions"
    ):
      compiled(x)


if __name__ == "__main__":
  absltest.main()
