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
from torch._dynamo import testing as dynamo_testing
from torch._subclasses import fake_tensor
import torch.distributed as dist
from torch.fx.experimental import proxy_tensor
from torch.fx.experimental import symbolic_shapes
from torch.testing._internal.distributed import fake_pg
from tests import seed_test_utils


def _privateuse1_device() -> str:
  backend = torch._C._get_privateuse1_backend_name()  # pylint: disable=protected-access
  return f"{backend}:0"


class ExperimentalCollectivesTest(seed_test_utils.RepeatableTest):

  def test_meta_shapes(self):
    value = torch.empty((2, 3, 4), device="meta")
    gathered = torch.ops.tpu._all_gather_dim(value, 1, 8, "unused")
    scattered = torch.ops.tpu._reduce_scatter_dim(
        gathered, "avg", -2, 8, "unused"
    )
    self.assertEqual(gathered.shape, (2, 24, 4))
    self.assertEqual(scattered.shape, value.shape)

  def test_public_ops_compile(self):
    dist.init_process_group(
        "fake", store=fake_pg.FakeStore(), rank=0, world_size=2
    )
    self.addCleanup(dist.destroy_process_group)
    value = torch.empty((2, 3, 4), device="meta")
    fn = torch.compile(
        lambda x: torch.tpu.ops.reduce_scatter_dim(
            torch.tpu.ops.all_gather_dim(x, 1), -2, dist.group.WORLD, "max"
        ),
        backend="eager",
        fullgraph=True,
    )
    self.assertEqual(fn(value).shape, value.shape)

  def test_meta_symbolic_shapes(self):
    """Tests the Meta kernels don't specialize symbolic input dims."""
    shape_env = symbolic_shapes.ShapeEnv()
    mode = fake_tensor.FakeTensorMode(shape_env=shape_env)
    value = mode.from_tensor(
        torch.empty((2, 6, 4), device="meta"),
        symbolic_context=symbolic_shapes.StatelessSymbolicContext(
            dynamic_sizes=[symbolic_shapes.DimDynamic.DYNAMIC] * 3
        ),
    )
    with mode:
      gathered = torch.ops.tpu._all_gather_dim(value, -1, 2, "unused")
      scattered = torch.ops.tpu._reduce_scatter_dim(
          value, "sum", 1, 2, "unused"
      )
    for size in (*value.shape, *gathered.shape, *scattered.shape):
      self.assertIsInstance(size, torch.SymInt)
    specialized = [
        dim
        for dim, size in enumerate(value.shape)
        if shape_env.replace(size.node.expr).is_number
    ]
    self.assertEmpty(specialized)

  def test_public_ops_compile_dynamic_shapes(self):
    """Tests compiled collectives recompile only once for dynamic shapes."""
    dist.init_process_group(
        "fake", store=fake_pg.FakeStore(), rank=0, world_size=2
    )
    self.addCleanup(dist.destroy_process_group)
    torch._dynamo.reset()
    self.addCleanup(torch._dynamo.reset)
    counter = dynamo_testing.CompileCounter()

    @torch.compile(backend=counter, fullgraph=True)
    def fn(x):
      return torch.tpu.ops.reduce_scatter_dim(
          torch.tpu.ops.all_gather_dim(x, -1), 1
      )

    for n in (2, 4, 6):
      # All dims change between calls.
      value = torch.empty((n + 1, n, n + 3), device="meta")
      self.assertEqual(fn(value).shape, (n + 1, n // 2, 2 * (n + 3)))
    # One static frame, then one dynamic frame that serves all later sizes.
    self.assertEqual(counter.frame_count, 2)

  def test_ragged_all_to_all_autograd_fx(self):
    """Tests ragged_all_to_all autograd traces offset transposes cleanly."""
    device = _privateuse1_device()
    dist.init_process_group(
        "fake", store=fake_pg.FakeStore(), rank=0, world_size=2
    )
    self.addCleanup(dist.destroy_process_group)
    group = dist.group.WORLD.group_name

    def fwd_bwd(x, dy, out, in_offs, send_sz, out_offs, recv_sz):
      y = torch.tpu.ops.ragged_all_to_all(
          x, out, in_offs, send_sz, out_offs, recv_sz, group
      )
      return y, y.grad_fn(dy)[0]

    with fake_tensor.FakeTensorMode():
      x = torch.empty(
          (4, 8), dtype=torch.bfloat16, device=device, requires_grad=True
      )
      out = torch.zeros((6, 8), dtype=torch.bfloat16, device=device)
      in_offs = torch.zeros(6, dtype=torch.int32, device=device)
      send_sz = torch.ones(6, dtype=torch.int32, device=device)
      out_offs = torch.ones(6, dtype=torch.int32, device=device)
      recv_sz = torch.ones(6, dtype=torch.int32, device=device)
      gm = proxy_tensor.make_fx(fwd_bwd)(
          x, torch.ones_like(out), out, in_offs, send_sz, out_offs, recv_sz
      )
      with torch.no_grad():
        no_grad_gm = proxy_tensor.make_fx(
            lambda *args: torch.tpu.ops.ragged_all_to_all(*args, group)
        )(x, out, in_offs, send_sz, out_offs, recv_sz)

    placeholders = gm.graph.find_nodes(op="placeholder")
    ragged = gm.graph.find_nodes(
        op="call_function", target=torch.ops.tpu._ragged_all_to_all.default
    )
    a2a = gm.graph.find_nodes(
        op="call_function",
        target=torch.ops._c10d_functional.all_to_all_single.default,
    )
    waits = gm.graph.find_nodes(
        op="call_function",
        target=torch.ops._c10d_functional.wait_tensor.default,
    )
    self.assertLen(ragged, 2)
    self.assertEqual(
        [node.args[0] for node in a2a], [placeholders[5], placeholders[3]]
    )
    self.assertEqual([node.args[1:3] for node in a2a], [([3, 3], [3, 3])] * 2)
    self.assertEqual([node.args[0] for node in waits], a2a)
    self.assertEqual(
        ragged[1].args[2:6],
        (waits[0], placeholders[6], waits[1], placeholders[4]),
    )
    self.assertEmpty(
        no_grad_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_to_all_single.default,
        )
    )

  def test_ragged_all_to_all_symbolic_shapes(self):
    """Tests ragged_all_to_all autograd and Meta ops preserve SymInt shapes."""
    device = _privateuse1_device()
    dist.init_process_group(
        "fake", store=fake_pg.FakeStore(), rank=0, world_size=2
    )
    self.addCleanup(dist.destroy_process_group)
    group = dist.group.WORLD.group_name
    shape_env = symbolic_shapes.ShapeEnv()
    mode = fake_tensor.FakeTensorMode(shape_env=shape_env)

    def sym(*shape, dtype=torch.bfloat16):
      return mode.from_tensor(
          torch.empty(shape, dtype=dtype, device="meta"),
          symbolic_context=symbolic_shapes.StatelessSymbolicContext(
              dynamic_sizes=[symbolic_shapes.DimDynamic.DYNAMIC] * len(shape)
          ),
      )

    raw_x, raw_t, raw_offs = sym(4, 8), sym(8, 6), sym(4, dtype=torch.int32)
    with mode:
      x = raw_x.to(device).requires_grad_(True)
      out_t = raw_t.to(device).t()
      out_buf = torch.empty(0, device=device)
      offs = raw_offs.to(device)
      sz = torch.ones_like(offs)
      y = torch.tpu.ops.ragged_all_to_all(x, out_t, offs, sz, offs, sz, group)
      dx = y.grad_fn(torch.ones_like(y))[0]
      torch.tpu.ops.ragged_all_to_all(
          x.detach(), out_t, offs, sz, offs, sz, group, out=out_buf
      )

    self.assertTrue(y.is_contiguous())
    self.assertEqual(
        (y.shape, out_buf.shape, dx.shape), (out_t.shape, out_t.shape, x.shape)
    )
    for size in (*y.shape, *out_buf.shape, *dx.shape):
      self.assertIsInstance(size, torch.SymInt)
    self.assertEmpty([
        size
        for size in (*raw_x.shape, *raw_t.shape, *raw_offs.shape)
        if shape_env.replace(size.node.expr).is_number
    ])

  def test_ragged_all_to_all_compile_dynamic_shapes(self):
    """Tests compiled ragged_all_to_all recompiles once for dynamic shapes."""
    device = _privateuse1_device()
    dist.init_process_group(
        "fake", store=fake_pg.FakeStore(), rank=0, world_size=2
    )
    self.addCleanup(dist.destroy_process_group)
    torch._dynamo.reset()
    self.addCleanup(torch._dynamo.reset)
    group = dist.group.WORLD.group_name
    counter = dynamo_testing.CompileCounter()

    @torch.compile(backend=counter, fullgraph=True)
    def compiled(x, out, offs, sz):
      return torch.tpu.ops.ragged_all_to_all(x, out, offs, sz, offs, sz, group)

    with fake_tensor.FakeTensorMode():
      offs = torch.zeros(4, dtype=torch.int32, device=device)
      sz = torch.ones(4, dtype=torch.int32, device=device)
      for n in (4, 6, 10):
        x = torch.empty(
            (n, 8), dtype=torch.bfloat16, device=device, requires_grad=True
        )
        out = torch.zeros((n + 2, 8), dtype=torch.bfloat16, device=device)
        y = compiled(x, out, offs, sz)
        dx = y.grad_fn(torch.ones_like(y))[0]
        self.assertEqual(
            (y.shape, dx.shape, dx.dtype), ((n + 2, 8), (n, 8), x.dtype)
        )
    self.assertEqual(counter.frame_count, 2)

  def test_meta_errors(self):
    value = torch.empty((2, 3, 4), device="meta")
    all_gather = torch.ops.tpu._all_gather_dim
    reduce_scatter = torch.ops.tpu._reduce_scatter_dim
    for op, args, error in (
        (reduce_scatter, (value, "sum", 1, 2), "divisible"),
        (all_gather, (value, 3, 2), r"in range \[-3, 2\]"),
        (reduce_scatter, (value, "sum", -4, 2), "got -4"),
        (all_gather, (value, 1, 0), "group_size to be > 0"),
        (
            all_gather,
            (torch.empty((), device="meta"), 0, 2),
            "at least 1 dimension",
        ),
        (reduce_scatter, (value, "premul_sum", 0, 2), "reduce_op to be one of"),
        (reduce_scatter, (value, "band", 0, 2), "integer"),
    ):
      with self.subTest(op=op, args=args[1:]):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=testing meta op
            (RuntimeError, IndexError), error
        ):
          op(*args, "unused")


if __name__ == "__main__":
  absltest.main()
