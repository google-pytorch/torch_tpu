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
from torch_tpu._internal.compile.fx_passes import pass_utils


def _nodes_by_op(graph):
  def _op_key(op):
    target = op.target
    if not isinstance(target, str):
      target = target.name()
    return target

  res = {}
  for n in graph.nodes:
    key = _op_key(n)
    if key not in res:
      res[key] = []
    res[key].append(n)
  return res


class PassUtilsTest(absltest.TestCase):

  def test_device_id_from_device(self):
    self.assertEqual(
        pass_utils.DeviceId.from_device("cpu"), pass_utils.DeviceId.CPU
    )
    self.assertEqual(
        pass_utils.DeviceId.from_device("CPU"), pass_utils.DeviceId.CPU
    )
    self.assertEqual(
        pass_utils.DeviceId.from_device("tpu"), pass_utils.DeviceId.TPU
    )
    self.assertEqual(
        pass_utils.DeviceId.from_device("TPU"), pass_utils.DeviceId.TPU
    )
    self.assertEqual(
        pass_utils.DeviceId.from_device("cuda"), pass_utils.DeviceId.UNKNOWN
    )

    self.assertEqual(
        pass_utils.DeviceId.from_device(torch.device("cpu")),
        pass_utils.DeviceId.CPU,
    )

  def test_get_node_device(self):
    class Model(torch.nn.Module):

      def forward(self):
        x = torch.zeros(1, device="cpu")
        x = x + 2
        x_tpu = x.to("tpu")
        x_tpu = x_tpu + 3
        return x_tpu

    gm = torch.export.export(Model(), tuple())
    ops = _nodes_by_op(gm.graph)

    zeros_op = ops["aten::zeros"]
    self.assertLen(zeros_op, 1)
    self.assertEqual(
        pass_utils.get_node_device(zeros_op[0]), pass_utils.DeviceId.CPU
    )

    add_ops = ops["aten::add.Tensor"]
    self.assertLen(add_ops, 2)
    self.assertEqual(
        pass_utils.get_node_device(add_ops[0]), pass_utils.DeviceId.CPU
    )
    self.assertEqual(
        pass_utils.get_node_device(add_ops[1]), pass_utils.DeviceId.TPU
    )

    to_device_ops = ops["aten::to.dtype_layout"]
    self.assertLen(to_device_ops, 1)
    self.assertEqual(
        pass_utils.get_node_device(to_device_ops[0]), pass_utils.DeviceId.TPU
    )

  def test_is_dependent(self):
    class DepModel(torch.nn.Module):

      def forward(self, x):
        y = x + 1
        z = y * 2
        return z

    gm = torch.fx.symbolic_trace(DepModel())
    nodes = list(gm.graph.nodes)

    x_node = nodes[0]
    y_node = nodes[1]
    z_node = nodes[2]

    self.assertEqual(x_node.op, "placeholder")

    # y depends on x
    self.assertTrue(pass_utils.is_dependent(y_node, x_node))
    # x does not depend on y
    self.assertFalse(pass_utils.is_dependent(x_node, y_node))

    # z depends on y
    self.assertTrue(pass_utils.is_dependent(z_node, y_node))
    # z depends on x (transitive)
    self.assertTrue(pass_utils.is_dependent(z_node, x_node))

    # y does not depend on z
    self.assertFalse(pass_utils.is_dependent(y_node, z_node))


if __name__ == "__main__":
  absltest.main()
