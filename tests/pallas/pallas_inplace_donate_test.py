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

"""Tests for inplace_donate=True Pallas custom kernels.

The workload: a device-resident state pool larger than half of HBM,
threaded through sequential Pallas calls per torch.compile'd step, across
steps, inference-only. Part 1 shows the pre-existing donation API cannot
express it (double-buffer OOM; set_ republish rejected in compile and
inside the op impl); part 2 shows inplace_donate does; part 3 pins the
individual mechanisms.
"""

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import torch
from torch._dynamo import exc as dynamo_exc

from absl.testing import absltest
from torch_tpu._internal import compile  # pylint: disable=redefined-builtin
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile.fx_passes import inplace_donate_ssa_rename
from torch_tpu._internal.utils import utils

_CHUNK_ELEMS = 1024 * 1024


def _pool_elems() -> int:
  """float32 element count for a pool at 55% of device HBM: one copy
  fits, two cannot — on any chip size."""
  _, total_bytes = torch.accelerator.get_memory_info()
  elems = int(total_bytes * 0.55) // 4
  return (elems // _CHUNK_ELEMS) * _CHUNK_ELEMS


def _accumulate_kernel_body(x_ref, acc_ref, o_ref):
  o_ref[...] = acc_ref[...] + x_ref[...]


def accumulate_jax(x: jax.Array, acc: jax.Array) -> jax.Array:
  out_shape = jax.ShapeDtypeStruct(acc.shape, acc.dtype)
  return pl.pallas_call(
      _accumulate_kernel_body,
      out_shape=out_shape,
      input_output_aliases={1: 0},
  )(x, acc)


# Cross-index donation: input 1 -> output 0.
inplace_accumulate = pallas.jax_op(
    "pallas::inplace_accumulate",
    accumulate_jax,
    donate_argnums=(1,),
    inplace_donate=True,
)
# The FX pass falls back to output_idx == input_idx unless the op carries
# the explicit input->output map (required for cross-index donation on a
# single-output kernel).
setattr(inplace_accumulate, "_pallas_inplace_donate_output_map", {1: 0})
_opoverload = getattr(inplace_accumulate, "_opoverload", None)
if _opoverload is not None:
  setattr(_opoverload, "_pallas_inplace_donate_output_map", {1: 0})


def _chunked(fn):
  """One grid step per _CHUNK_ELEMS chunk, so VMEM stays bounded on the
  multi-GiB pool."""

  def call(pool: jax.Array) -> jax.Array:
    out_shape = jax.ShapeDtypeStruct(pool.shape, pool.dtype)
    grid = (pool.shape[0] // _CHUNK_ELEMS,)
    spec = pl.BlockSpec((_CHUNK_ELEMS,), lambda i: (i,))
    return pl.pallas_call(
        fn,
        grid=grid,
        in_specs=[spec],
        out_specs=spec,
        out_shape=out_shape,
        input_output_aliases={0: 0},
    )(pool)

  return call


def _fill_one_kernel_body(pool_ref, o_ref):
  del pool_ref
  o_ref[...] = jnp.ones((_CHUNK_ELEMS,), dtype=jnp.float32)


def _increment_kernel_body(pool_ref, o_ref):
  o_ref[...] = pool_ref[...] + 1.0


# `torch.zeros`/`fill_` on a >HBM/2 tensor OOM on their own (the in-place
# fill lowering keeps input and output alive), so even *initializing* the
# pool requires a donating kernel.
inplace_fill_one = pallas.jax_op(
    "pallas::inplace_fill_one",
    _chunked(_fill_one_kernel_body),
    donate_argnums=(0,),
    inplace_donate=True,
)

inplace_increment = pallas.jax_op(
    "pallas::inplace_increment",
    _chunked(_increment_kernel_body),
    donate_argnums=(0,),
    inplace_donate=True,
)

# The pre-existing API: donation without the in-place machinery.
plain_donating_increment = pallas.jax_op(
    "pallas::plain_donating_increment",
    _chunked(_increment_kernel_body),
    donate_argnums=(0,),
)

plain_accumulate = pallas.jax_op(
    "pallas::plain_accumulate",
    accumulate_jax,
    donate_argnums=(1,),
)


class TestPallasInplaceDonate(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.device("tpu")

  # --- Part 1: the pre-existing donation API cannot express the workload.

  def test_existing_donation_api_cannot_update_half_hbm_pool(self):
    # Without in-place aliasing, one update of the pool holds the donated
    # input and the kernel output as two distinct >HBM/2 buffers.
    pool = torch.empty(
        _pool_elems(), dtype=torch.float32, device=self.device
    )
    with self.assertRaisesRegex(
        RuntimeError, "RESOURCE_EXHAUSTED|[Oo]ut of memory"
    ):
      plain_donating_increment(pool).cpu()  # .cpu() forces execution

  def test_caller_set_republish_rejected_under_compile(self):
    # The documented eager republish for plain donation (`x.set_(out)`)
    # cannot be used in the compiled serving loop: Dynamo rejects `set_`
    # on a graph input.
    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def step(x, acc):
      out = plain_accumulate(x, acc)
      acc.set_(out)
      return out

    x = torch.ones(3, dtype=torch.float32, device=self.device)
    acc = torch.zeros(3, dtype=torch.float32, device=self.device)
    with self.assertRaises(dynamo_exc.Unsupported):
      step(x, acc)

  def test_custom_op_impl_cannot_republish_via_set_(self):
    # Inside a custom_op impl, set_ makes the returned output share the
    # input's storage — rejected by torch.library's aliasing check (why
    # the republish is a DeviceBuffer-level rebind in C++). Dispatcher-
    # layer check, so CPU tensors suffice.
    @torch.library.custom_op(
        "pallas_test::set_republish", mutates_args=()
    )
    def set_republish(x: torch.Tensor) -> torch.Tensor:
      out = x + 1.0
      x.set_(out)
      return out

    with self.assertRaisesRegex(RuntimeError, "may not alias"):
      set_republish(torch.ones(3))

  # --- Part 2: inplace_donate expresses it.

  def test_inplace_donate_updates_half_hbm_pool_across_compiled_steps(self):
    # The serving-loop contract at the scale part 1 proved impossible:
    # fill in place, then run a compiled step twice on the same handle.
    pool = torch.empty(
        _pool_elems(), dtype=torch.float32, device=self.device
    )
    inplace_fill_one(pool)

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def step(pool):
      return inplace_increment(pool)

    step(pool)
    step(pool)

    expected = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
    utils.assert_close(pool[:3].to("cpu"), expected)

  # --- Part 3: the individual mechanisms.

  def test_eager_cross_index_rebind(self):
    # Donated input 1 aliases output 0: the rebind must follow the alias
    # pairs — an index-identity rebind never fires here and leaves the
    # input pointing at the donated buffer.
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
    acc = torch.tensor(
        [10.0, 20.0, 30.0], dtype=torch.float32, device=self.device
    )

    out = inplace_accumulate(x, acc)

    expected = torch.tensor([11.0, 22.0, 33.0], dtype=torch.float32)
    utils.assert_close(out.to("cpu"), expected)
    # The rebound donated input observes the same data, with no explicit
    # republish.
    utils.assert_close(acc.to("cpu"), expected)

  def test_eager_chained_donation(self):
    # Sequential donating calls thread the state forward: each call must
    # read its predecessor's output through the rebound input.
    x = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
    acc = torch.zeros(3, dtype=torch.float32, device=self.device)

    inplace_accumulate(x, acc)
    inplace_accumulate(x, acc)
    out = inplace_accumulate(x, acc)

    expected = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
    utils.assert_close(out.to("cpu"), expected)
    utils.assert_close(acc.to("cpu"), expected)

  def test_compiled_chained_donation(self):
    # Both calls take the graph input `acc`; the SSA-rename pass must
    # rewrite the second call to read the first call's output. Returning
    # both outputs keeps the first call alive.
    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def double_accumulate(x, acc):
      first = inplace_accumulate(x, acc)
      second = inplace_accumulate(x, acc)
      return first, second

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
    acc = torch.tensor(
        [10.0, 20.0, 30.0], dtype=torch.float32, device=self.device
    )

    first, second = double_accumulate(x, acc)

    expected = torch.tensor([12.0, 24.0, 36.0], dtype=torch.float32)
    utils.assert_close(second.to("cpu"), expected)
    # The first call's output buffer is donated into the second call, so the
    # intermediate value is destroyed: `first` observes the final state.
    utils.assert_close(first.to("cpu"), expected)

  def test_ssa_rename_pass_rewrites_later_uses(self):
    # Structural proof that the FX pass is what makes fused donation chains
    # correct: two calls of the donating op on the same placeholder. Before
    # the pass, both calls read `acc`; after it, the second call must read
    # the first call's donated output.
    op = inplace_accumulate._opoverload
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    acc = graph.placeholder("acc")
    first = graph.create_node("call_function", op, (x, acc), {})
    second = graph.create_node("call_function", op, (x, acc), {})
    graph.output((first, second))

    self.assertIs(second.args[1], acc)
    inplace_donate_ssa_rename.apply(graph)
    self.assertIs(second.args[1], first)

  def test_rejects_autograd(self):
    # Donated buffers are invalidated in place; autograd cannot replay them.
    x = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
        device=self.device,
        requires_grad=True,
    )
    acc = torch.zeros(3, dtype=torch.float32, device=self.device)

    with self.assertRaisesRegex(RuntimeError, "inference-only"):
      inplace_accumulate(x, acc)


if __name__ == "__main__":
  absltest.main()
