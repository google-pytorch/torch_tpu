# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
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
import torch.distributed.tensor as dist_tensor
from torch.testing._internal.distributed import fake_pg
from torch_tpu._internal.compile import dtensor_sentinels


class DTensorSentinelsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._world_size = 4
    torch.distributed.init_process_group(
        backend="fake",
        store=fake_pg.FakeStore(),
        rank=0,
        world_size=self._world_size,
    )

  def tearDown(self):
    torch.distributed.destroy_process_group()
    super().tearDown()

  def test_parse_generic_sharding_str(self):
    mesh = dist_tensor.DeviceMesh(
        "cpu", torch.arange(4), mesh_dim_names=["data"]
    )

    # 1. Replicated
    placements = dtensor_sentinels.parse_generic_sharding_str(";", mesh)
    self.assertEqual(placements, (dist_tensor.Replicate(),))

    # 2. Sharded on dim 0
    placements = dtensor_sentinels.parse_generic_sharding_str("data;", mesh)
    self.assertEqual(placements, (dist_tensor.Shard(0),))

    # 3. 2D Mesh Sharded
    mesh_2d = dist_tensor.DeviceMesh(
        "cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=["x", "y"]
    )
    placements = dtensor_sentinels.parse_generic_sharding_str("x;y", mesh_2d)
    self.assertEqual(placements, (dist_tensor.Shard(0), dist_tensor.Shard(1)))

    # 4. Partial Sum
    placements = dtensor_sentinels.parse_generic_sharding_str(";|data", mesh)
    self.assertEqual(placements, (dist_tensor.Partial("sum"),))

  def test_eager_sentinel_ops_meta(self):
    # Verify that we can call the custom ops on meta device and they
    # propagate shapes correctly.
    x = torch.empty((8, 16), device="meta")

    # input sentinel -> returns global shape
    y = torch.ops.tpu.spmd_metadata(
        x, "mesh_str", "", "out_str", [8, 32], "input"
    )
    self.assertEqual(y.device.type, "meta")
    self.assertEqual(y.shape, torch.Size([8, 32]))

    # output sentinel -> returns global shape in meta tracing fallback
    z = torch.ops.tpu.spmd_metadata(
        x, "mesh_str", "in_str", "", [8, 16], "output"
    )
    self.assertEqual(z.device.type, "meta")
    self.assertEqual(z.shape, torch.Size([8, 16]))

    # intermediate sentinel -> returns global shape
    w = torch.ops.tpu.spmd_metadata(
        x, "mesh_str", "in_str", "out_str", [8, 16], "intermediate"
    )
    self.assertEqual(w.device.type, "meta")
    self.assertEqual(w.shape, torch.Size([8, 16]))


if __name__ == "__main__":
  absltest.main()
