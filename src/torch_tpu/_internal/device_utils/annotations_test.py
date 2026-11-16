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
from torch.testing._internal.distributed import fake_pg
from torch_tpu._internal.compile import tpu_torch_compile
import torch_tpu._internal.device_utils.annotations as tpu_annotations
from torch_tpu._internal.utils import test_utils


class AnnotationsTest(
    absltest.TestCase  # ABSLTEST_OK=Deterministic layout annotation test
):

  def test_layout_context_to_device(self):
    device = torch.device("tpu")
    layout = tpu_annotations.TpuLayout(
        minor_to_major=[1, 0],
        tiles=[[16, 128], [2, 1]],
    )
    with tpu_annotations.LayoutContext(layout):
      x = torch.tensor([[1.0] * 10] * 10).to(device)
      torch.accelerator.synchronize()
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
      torch.accelerator.synchronize()
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

    torch.accelerator.synchronize()

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
      torch.accelerator.synchronize()
      device_layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(
          x
      )
      self.assertIsNotNone(device_layout_tuple)
      device_layout = tpu_annotations.TpuLayout(*device_layout_tuple)
      self.assertEqual(device_layout, layout)

  def test_compiled_program_rejects_input_in_another_layout(self):
    device = torch.device("tpu")
    # Column-major. A 128x128 tensor's default layout is row-major,
    # {1,0:T(8,128)}, which takes as many bytes.
    layout = tpu_annotations.TpuLayout(
        minor_to_major=[0, 1],
        tiles=[[8, 128]],
    )

    @torch.compile(backend="tpu", fullgraph=True)
    def add_one(x):
      return x + 1.0

    expected = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    # The first call compiles the program for the default layout.
    test_utils.assert_close(add_one(expected.to(device)).cpu(), expected + 1.0)

    # Programs are only compiled again for inputs that had non-default layouts.
    with tpu_annotations.LayoutContext(layout):
      x = expected.to(device)
    with self.assertRaisesRegex(RuntimeError, "compiled for"):
      add_one(x).cpu()

  def test_compiled_program_recompiles_for_default_layout_after_another_layout(
      self,
  ):
    device = torch.device("tpu")
    # Column-major, which takes as many bytes as the default row-major layout.
    layout = tpu_annotations.TpuLayout(
        minor_to_major=[0, 1],
        tiles=[[8, 128]],
    )
    debugs = []

    @torch.compile(
        backend="tpu",
        fullgraph=True,
        options={"debug_callback": debugs.append},
    )
    def add_one(x):
      return x + 1.0

    expected = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    with tpu_annotations.LayoutContext(layout):
      x = expected.to(device)
    # The first call compiles the program for `layout`.
    test_utils.assert_close(add_one(x).cpu(), expected + 1.0)
    self.assertLen(debugs, 1)

    # An input in the default layout gets a program compiled for it.
    test_utils.assert_close(add_one(expected.to(device)).cpu(), expected + 1.0)
    self.assertLen(debugs, 2)

    # An input in `layout` reuses the first program.
    test_utils.assert_close(add_one(x).cpu(), expected + 1.0)
    self.assertLen(debugs, 2)

  def test_copy_from_cpu_keeps_layout(self):
    device = torch.device("tpu")
    # Column-major, which takes as many bytes as the default row-major layout.
    layout = tpu_annotations.TpuLayout(minor_to_major=[0, 1], tiles=[[8, 128]])
    expected = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    with tpu_annotations.LayoutContext(layout):
      x = torch.zeros_like(expected).to(device)

    # Like load_state_dict, copy_ from the CPU keeps `x`'s layout.
    x.copy_(expected)
    test_utils.assert_close(x.cpu(), expected)
    x_layout = tpu_torch_compile.get_device_layout_if_materialized(x)
    self.assertEqual(tpu_annotations.TpuLayout(*x_layout), layout)

    # An active LayoutContext decides the layout instead.
    row_major = tpu_annotations.TpuLayout(
        minor_to_major=[1, 0], tiles=[[8, 128]]
    )
    with tpu_annotations.LayoutContext(row_major):
      x.copy_(expected)
    test_utils.assert_close(x.cpu(), expected)
    x_layout = tpu_torch_compile.get_device_layout_if_materialized(x)
    self.assertEqual(tpu_annotations.TpuLayout(*x_layout), row_major)

  def test_copy_from_cpu_keeps_default_layout(self):
    device = torch.device("tpu")
    # A 128x128 tensor's default layout is row-major, {1,0:T(8,128)}.
    default_layout = tpu_annotations.TpuLayout(
        minor_to_major=[1, 0], tiles=[[8, 128]]
    )
    expected = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    x = torch.zeros_like(expected).to(device)

    x.copy_(expected)
    test_utils.assert_close(x.cpu(), expected)
    x_layout = tpu_torch_compile.get_device_layout_if_materialized(x)
    self.assertEqual(tpu_annotations.TpuLayout(*x_layout), default_layout)

  def test_copy_from_cpu_keeps_bfloat16_layout(self):
    device = torch.device("tpu")
    # Column-major. The (2, 1) tiles pack pairs of bfloat16 values into 32-bit
    # words.
    layout = tpu_annotations.TpuLayout(
        minor_to_major=[0, 1], tiles=[[16, 128], [2, 1]]
    )
    expected = (
        torch.arange(128 * 128, dtype=torch.float32)
        .reshape(128, 128)
        .to(torch.bfloat16)
    )
    with tpu_annotations.LayoutContext(layout):
      x = torch.zeros_like(expected).to(device)

    x.copy_(expected)
    test_utils.assert_close(x.cpu(), expected)
    x_layout = tpu_torch_compile.get_device_layout_if_materialized(x)
    self.assertEqual(tpu_annotations.TpuLayout(*x_layout), layout)

  def test_copy_from_cpu_keeps_layout_when_converting_source(self):
    device = torch.device("tpu")
    # Column-major, which takes as many bytes as the default row-major layout.
    layout = tpu_annotations.TpuLayout(minor_to_major=[0, 1], tiles=[[8, 128]])
    expected = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    with tpu_annotations.LayoutContext(layout):
      x = torch.zeros_like(expected).to(device)

    # copy_ first converts a source in another dtype, or that isn't contiguous.
    for src in (expected.double(), expected.t()):
      with self.subTest(dtype=src.dtype, contiguous=src.is_contiguous()):
        x.copy_(src)
        test_utils.assert_close(x.cpu(), src.float())
        x_layout = tpu_torch_compile.get_device_layout_if_materialized(x)
        self.assertEqual(tpu_annotations.TpuLayout(*x_layout), layout)

  def test_compiled_program_recompiles_after_copy_resets_layout(self):
    device = torch.device("tpu")
    # Column-major, which takes as many bytes as the default row-major layout.
    layout = tpu_annotations.TpuLayout(
        minor_to_major=[0, 1],
        tiles=[[8, 128]],
    )
    debugs = []

    @torch.compile(
        backend="tpu",
        fullgraph=True,
        options={"debug_callback": debugs.append},
    )
    def add_one(x):
      return x + 1.0

    expected = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    with tpu_annotations.LayoutContext(layout):
      x = expected.to(device)
    # The first call compiles the program for `layout`.
    test_utils.assert_close(add_one(x).cpu(), expected + 1.0)
    self.assertLen(debugs, 1)

    # Copying from a default-layout TPU tensor gives `x` the default layout, so
    # the next call compiles a program for it.
    x.copy_(expected.to(device))
    test_utils.assert_close(add_one(x).cpu(), expected + 1.0)
    self.assertLen(debugs, 2)

  def test_compiled_program_is_compiled_for_layouts_of_dtensor_inputs(self):
    device = torch.device("tpu")
    # Column-major. A 16x128 tensor's default layout is row-major,
    # {1,0:T(8,128)}.
    layout = tpu_annotations.TpuLayout(
        minor_to_major=[0, 1],
        tiles=[[8, 128]],
    )
    torch.distributed.init_process_group(
        backend="fake", store=fake_pg.FakeStore(), rank=0, world_size=1
    )
    self.addCleanup(torch.distributed.destroy_process_group)
    mesh = torch.distributed.tensor.DeviceMesh("tpu", torch.arange(1))

    @torch.compile(backend="tpu", fullgraph=True)
    def add_one(x):
      return x.to_local() + 1.0

    expected = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128)
    with tpu_annotations.LayoutContext(layout):
      local = expected.to(device)
    x = torch.distributed.tensor.DTensor.from_local(
        local, mesh, [torch.distributed.tensor.Shard(0)], run_check=False
    )

    # The program receives the local tensor of `x`, with its layout.
    test_utils.assert_close(add_one(x).cpu(), expected + 1.0)


if __name__ == "__main__":
  absltest.main()
