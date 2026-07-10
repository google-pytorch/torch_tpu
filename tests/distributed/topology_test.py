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
from torch_tpu import distributed as tpu_dist
from torch_tpu._internal.distributed.topology import _reorder_for_priority


class TopologyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Reset priority before each test
    tpu_dist.set_layout_priority(None)

  def test_get_set_priority(self):
    self.assertIsNone(tpu_dist.get_layout_priority())

    tpu_dist.set_layout_priority(["tp", "sp", "dp"])
    self.assertEqual(tpu_dist.get_layout_priority(), ["tp", "sp", "dp"])

    tpu_dist.set_layout_priority(None)
    self.assertIsNone(tpu_dist.get_layout_priority())

  def test_reorder_for_priority_basic(self):
    # Logical mesh: dp=2, tp=8, sp=4
    # Priority: tp > sp > dp (TP highest, DP lowest)
    # JAX order (reversed priority): dp, sp, tp
    # JAX shape: (2, 4, 8)
    # Transpose perm from JAX (dp, sp, tp) to User (dp, tp, sp):
    # dp is at 0, tp is at 2, sp is at 1 -> (0, 2, 1)

    mesh_shape = (2, 8, 4)
    mesh_dim_names = ("dp", "tp", "sp")
    priority = ("tp", "sp", "dp")

    jax_shape, jax_names, transpose_perm = _reorder_for_priority(
        mesh_shape, mesh_dim_names, priority
    )

    self.assertEqual(jax_shape, (2, 4, 8))
    self.assertEqual(jax_names, ("dp", "sp", "tp"))
    self.assertEqual(transpose_perm, (0, 2, 1))

  def test_reorder_for_priority_partial(self):
    # Logical mesh: dp=2, tp=8, sp=4
    # Priority: only specifies tp
    # Remaining dims (dp, sp) should be added to the front (lowest priority)
    # full_priority: tp, then remaining (dp, sp) added to front -> dp, sp, tp
    # Wait, "add to front" in loop:
    # for name in mesh_dim_names:
    #   if name not in full_priority:
    #     full_priority.insert(0, name)
    #
    # Trace for mesh_dim_names = ('dp', 'tp', 'sp'), priority = ('tp',):
    # 1. full_priority init to ['tp']
    # 2. name = 'dp': not in full_priority -> append('dp') -> ['tp', 'dp']
    # 3. name = 'tp': in full_priority -> skip
    # 4. name = 'sp': not in full_priority -> append('sp') -> ['tp', 'dp', 'sp']
    # So full_priority is ['tp', 'dp', 'sp'] (TP has highest priority, then DP,
    # then SP)
    # JAX order (reversed): ['sp', 'dp', 'tp'] (TP is last/innermost, so mapped
    # first)
    # jax_shape: (4, 2, 8)
    # jax_names: ('sp', 'dp', 'tp')
    # Transpose perm from JAX (sp, dp, tp) to User (dp, tp, sp):
    # dp is at JAX index 1, tp is at 2, sp is at 0 -> (1, 2, 0)

    mesh_shape = (2, 8, 4)
    mesh_dim_names = ("dp", "tp", "sp")
    priority = ("tp",)

    jax_shape, jax_names, transpose_perm = _reorder_for_priority(
        mesh_shape, mesh_dim_names, priority
    )

    self.assertEqual(jax_shape, (4, 2, 8))
    self.assertEqual(jax_names, ("sp", "dp", "tp"))
    self.assertEqual(transpose_perm, (1, 2, 0))


if __name__ == "__main__":
  absltest.main()
