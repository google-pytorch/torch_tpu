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
from torch_tpu._internal.compile import split_compiler


_LIBRARY = torch.library.Library("split_compiler_unit_test", "DEF")
_LIBRARY.define("opaque_collective(Tensor input) -> Tensor")
_OPAQUE_COLLECTIVE = torch.ops.split_compiler_unit_test.opaque_collective


class SplitCompilerUnitTest(absltest.TestCase):

  def test_registered_auto_functionalized_collective_is_detected(self):
    split_compiler.register_collective_op(_OPAQUE_COLLECTIVE.default)

    graph = torch.fx.Graph()
    input_node = graph.placeholder("input")
    direct_node = graph.call_function(
        _OPAQUE_COLLECTIVE.default, (input_node,)
    )
    wrapped_node = graph.call_function(
        torch.ops.higher_order.auto_functionalized_v2,
        (_OPAQUE_COLLECTIVE.default,),
        {"input": input_node},
    )
    graph.output((direct_node, wrapped_node))

    self.assertIs(
        split_compiler._get_collective_target(direct_node),
        _OPAQUE_COLLECTIVE,
    )
    self.assertIs(
        split_compiler._get_collective_target(wrapped_node),
        _OPAQUE_COLLECTIVE,
    )

  def test_concretize_fake_tensor_preserves_layout(self):
    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    tensor = torch.empty(32).as_strided((3, 4), (6, 1), 2)
    fake_tensor = fake_mode.from_tensor(tensor)

    concrete = split_compiler._concretize_fake_tensor(fake_tensor, fake_mode)

    self.assertIsInstance(concrete, torch._subclasses.FakeTensor)
    self.assertEqual(concrete.shape, (3, 4))
    self.assertEqual(concrete.stride(), (6, 1))
    self.assertEqual(concrete.storage_offset(), 2)
    self.assertEqual(concrete.dtype, tensor.dtype)
    self.assertEqual(concrete.fake_device, fake_tensor.fake_device)


if __name__ == "__main__":
  absltest.main()
