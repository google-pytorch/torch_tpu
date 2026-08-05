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

"""Tests for layout hint annotations."""

from absl.testing import absltest
import torch
from torch_tpu._internal.compile import tpu_torch_compile
import torch_tpu._internal.device_utils.annotations as tpu_annotations
from tests import seed_test_utils


class AnnotationsTest(seed_test_utils.RepeatableTest):

  def test_layout_context_to_device(self):
    device = torch.device("tpu")
    layout = tpu_annotations.TpuLayout(
        minor_to_major=[1, 0],
        tiles=[[16, 128], [2, 1]],
    )
    with tpu_annotations.LayoutContext(layout):
      x = torch.tensor([[1.0] * 10] * 10).to(device)
      torch.tpu.synchronize()
      device_layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(
          x
      )
      self.assertIsNotNone(device_layout_tuple)
      device_layout = tpu_annotations.TpuLayout(*device_layout_tuple)
      self.assertEqual(device_layout, layout)

  def test_layout_context_device_arg(self):
    device = torch.device("tpu")
    layout = tpu_annotations.TpuLayout(
        minor_to_major=[1, 0],
        tiles=[[16, 128], [2, 1]],
    )
    with tpu_annotations.LayoutContext(layout):
      x = torch.tensor([[1.0] * 10] * 10, device=device)
      torch.tpu.synchronize()
      device_layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(
          x
      )
      self.assertIsNotNone(device_layout_tuple)
      device_layout = tpu_annotations.TpuLayout(*device_layout_tuple)
      self.assertEqual(device_layout, layout)

  def test_layout_context_nested(self):
    device = torch.device("tpu")
    layout1 = tpu_annotations.TpuLayout(
        minor_to_major=[1, 0],
        tiles=[[16, 128], [2, 1]],
    )
    layout2 = tpu_annotations.TpuLayout(
        minor_to_major=[0, 1],
        tiles=[[16, 128], [2, 1]],
    )
    with tpu_annotations.LayoutContext(layout1):
      x = torch.tensor([[1.0] * 10] * 10, device=device)
      with tpu_annotations.LayoutContext(layout2):
        y = torch.tensor([[1.0] * 10] * 10, device=device)
      z = torch.tensor([[1.0] * 10] * 10, device=device)

    torch.tpu.synchronize()

    x_layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(x)
    self.assertIsNotNone(x_layout_tuple)
    self.assertEqual(tpu_annotations.TpuLayout(*x_layout_tuple), layout1)

    y_layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(y)
    self.assertIsNotNone(y_layout_tuple)
    self.assertEqual(tpu_annotations.TpuLayout(*y_layout_tuple), layout2)

    z_layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(z)
    self.assertIsNotNone(z_layout_tuple)
    self.assertEqual(tpu_annotations.TpuLayout(*z_layout_tuple), layout1)

  def test_layout_context_1d_tiling(self):
    device = torch.device("tpu")
    layout = tpu_annotations.TpuLayout(
        minor_to_major=[1, 0],
        tiles=[[8]],
    )
    with tpu_annotations.LayoutContext(layout):
      x = torch.tensor([[1.0] * 128] * 16, device=device)
      torch.tpu.synchronize()
      device_layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(
          x
      )
      self.assertIsNotNone(device_layout_tuple)
      device_layout = tpu_annotations.TpuLayout(*device_layout_tuple)
      self.assertEqual(device_layout, layout)


if __name__ == "__main__":
  absltest.main()
