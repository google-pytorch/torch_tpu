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
from torch.testing._internal.distributed.fake_pg import FakeStore


class DTensorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._world_size = 2
    torch.distributed.init_process_group(
        backend="fake", store=FakeStore(), rank=0, world_size=self._world_size
    )

  def tearDown(self):
    torch.distributed.destroy_process_group()
    super().tearDown()

  def test_dtensor_basic(self):
    mesh = torch.distributed.tensor.DeviceMesh(
        "tpu", torch.arange(self._world_size)
    )

    @torch.compile(fullgraph=True)
    def fn(x):
      return x * x + 2

    param = torch.randn(4, 4, requires_grad=True)
    x = torch.distributed.tensor.DTensor.from_local(
        param, mesh, [torch.distributed.tensor.Shard(0)], run_check=False
    )

    res = fn(x)
    res.to_local().sum().backward()


if __name__ == "__main__":
  absltest.main()
