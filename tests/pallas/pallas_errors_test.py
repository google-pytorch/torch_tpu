# Copyright 2025 Google LLC
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

"""Tests for Pallas custom kernels."""

import enum
import os
import re
from typing import Sequence, TypeAlias

from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from packaging import version
import torch
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import execution_mode
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils

EagerMode: TypeAlias = execution_mode.EagerMode
_EagerMode: TypeAlias = execution_mode._EagerMode


def _is_deferred_mode():
  return execution_mode.eager_mode in (
      EagerMode.DEFER_AND_FUSE,
      _EagerMode.INTERNAL_DEFER_ALL,
      EagerMode._INTERNAL_COMPILE_FX_GRAPH,
  )


def add_vectors_kernel_body(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y


def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  dt = jax.ShapeDtypeStruct(x.shape, x.dtype)
  pallas_call = pl.pallas_call(add_vectors_kernel_body, out_shape=dt)
  return pallas_call(x, y)


@pallas.jax_op("pallas::pass_through")
def pass_through(x: jax.Array) -> jax.Array:
  return x


class TestPallasKernels(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.device("tpu")

  def _assert_donated(self, x: torch.Tensor):
    with self.assertRaisesRegex(
        RuntimeError,
        re.compile(
            r".*(Buffer has been deleted or donated|cannot start a device to"
            r" host copy from a donated or deleted buffer).*"
        ),
    ):
      x.cpu()

  def test_kernel_donating(self):

    donating_add_vectors = pallas.jax_op(
        "pallas::donating_add_vectors",
        add_vectors_jax,
        donate_argnums=(0,),
    )
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = donating_add_vectors(x, y).to("cpu")
    test_utils.assert_close(actual, expected)
    # x should be donated by the output
    self._assert_donated(x)
    # y should not be donated
    expected_y = torch.tensor(
        [0.4, 0.5, 0.6], dtype=torch.float32, device="cpu"
    )
    test_utils.assert_close(y.to("cpu"), expected_y)

  def test_kernel_donation_invalidates_deferred_op(self):

    donating_add_vectors = pallas.jax_op(
        "pallas::donating_add_vectors",
        add_vectors_jax,
        donate_argnums=(0,),
    )
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    # Run the donating operation, consuming the pre-donation value of x.
    z = donating_add_vectors(x, y)
    z.cpu()  # force execution

    self._assert_donated(x)

  def test_kernel_donating_compiled_mode(self):

    torch_donating_add_vectors = pallas.jax_op(
        "pallas::add_vectors_donated",
        add_vectors_jax,
        donate_argnums=(0,),
    )

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    @torch.compile(
        fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )
    def donated_add_vectors_sum(x, y):
      return torch_donating_add_vectors(x, y).sum()

    expected_updated_x = torch.add(
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device="cpu"),
        torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device="cpu"),
    )
    expected_sum = expected_updated_x.sum()

    actual_sum = donated_add_vectors_sum(x, y)
    test_utils.assert_close(actual_sum.to("cpu"), expected_sum)
    self._assert_donated(x)

  def test_kernel_compiled_mode_donation_invalidates_deferred_op(self):

    donating_add_vectors = pallas.jax_op(
        "pallas::donating_add_vectors",
        add_vectors_jax,
        donate_argnums=(0,),
    )

    # These tensors are intentionally 2D as it triggers the flatten -> unflatten
    # behaviour (b/479542146).
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32, device=self.device)
    y = torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32, device=self.device)

    # Create a deferred op that depends on the pre-donation value of x.
    pre_x_sum = x.sum()

    debugs: list[tpu_compile.TpuCompileDebug] = []

    # Run a donating compiled operation.
    @torch.compile(
        fullgraph=True,
        dynamic=False,
        backend=tpu_compile.TpuBackend(),
        options={"debug_callback": debugs.append},
    )
    def donated_add_vectors_sum(x, y):
      return donating_add_vectors(x, y).sum()

    _ = donated_add_vectors_sum(x, y)

    self.assertLen(debugs, 1)
    executables = debugs[0].compiled_executables
    self.assertLen(executables, 1)
    self.assertIn(
        "jax.buffer_donor",
        debugs[0].stablehlo_forward_text[0],
    )

    # The pre-donation value of x can no longer be used if we are in deferred
    # mode.
    if _is_deferred_mode():
      self._assert_donated(pre_x_sum)
    else:
      pre_x_sum.cpu()

  def test_jax_kernel_wrapper_with_donation(self):
    """Test a kernel wrapper with a donated argument."""

    @pallas.jax_op("pallas::add_fn", donate_argnums=(0,))
    def add_fn(x: jax.Array, y: jax.Array) -> jax.Array:
      return jax.numpy.add(x, y)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = add_fn(x, y)
    test_utils.assert_close(actual.cpu(), expected)
    # x should be donated by the output.
    self._assert_donated(x)

  def test_symbolic_shape_raises_error(self):
    symbolic_shape = torch.fx.experimental.symbolic_shapes
    shape_env = symbolic_shape.ShapeEnv()

    x = torch.randn(5, device=self.device)
    with torch._subclasses.FakeTensorMode(shape_env=shape_env) as mode:
      fake_x = mode.from_tensor(
          x,
          symbolic_context=symbolic_shape.StatelessSymbolicContext(
              dynamic_sizes=[symbolic_shape.DimDynamic.DYNAMIC]
          ),
      )
      with self.assertRaisesRegex(
          RuntimeError, "Symbolic dimensions are not supported"
      ):
        pass_through(fake_x)

  def test_inconsistent_donate_and_aliases_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "donate_argnums must be None or the same as the keys of"
        " input_output_aliases.",
    ):
      pallas.custom_jax_kernel(
          add_vectors_jax,
          name="add_vectors",
          input_output_aliases={0: 0},
          donate_argnums=(1,),
      )

  def test_custom_call(self):
    is_oss = os.getenv("IS_OSS", "0") == "1"
    jax_version = version.Version(jax.__version__)
    if is_oss and jax_version < version.Version("0.10.3"):
      self.skipTest("Jax version doesn't support custom_call export.")

    def custom_call_fn(x: jax.Array) -> jax.Array:
      return jax.ffi.ffi_call(
          "test_custom_call", jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x)

    custom_call_op = pallas.jax_op("test::custom_call", custom_call_fn)

    # Check that it successfully reaches XLA and raises an error.
    with self.assertRaisesRegex(
        RuntimeError,
        expected_regex="""failed to compile MLIR module -- the XLA compiler failed with: custom emitter for test_custom_call not found""",
    ):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
      y = custom_call_op(x)
      y.cpu()
    tt_testing.reset_eager_state()


class TestPallasCompat(seed_test_utils.RepeatableTest):

  def _assert_donated(self, x: torch.Tensor):
    with self.assertRaisesRegex(
        RuntimeError,
        re.compile(
            r".*(Buffer has been deleted or donated|cannot start a device to"
            r" host copy from a donated or deleted buffer).*"
        ),
    ):
      x.cpu()

  def test_fp4_non_contiguous_error(self):

    u8_t = torch.zeros((8, 256), dtype=torch.uint8)
    fp4_t = u8_t.view(torch.float4_e2m1fn_x2).to("tpu")

    # Transpose to make it non-contiguous
    fp4_t_transposed = fp4_t.t()
    self.assertFalse(fp4_t_transposed.is_contiguous())

    def fn_identity(x: jax.Array) -> jax.Array:
      return x

    op = pallas.jax_op("repro::identity_error", fn_identity)

    with self.assertRaisesRegex(
        ValueError,
        r"expected packed dimension of float4_e2m1fn_x2 tensor to be "
        r"contiguous \(stride 1\)",
    ):
      op(fp4_t_transposed)

  def test_container_enum_and_dtype_contract(self):
    """Tests that jax_op rejects invalid enum values."""

    class ExecutionMode(enum.StrEnum):
      FAST = "FAST"
      ACCURATE = "ACCURATE"

    def custom_fused_kernel(
        x: jax.Array,
        y: jax.Array,
        bias: jax.Array | None = None,
        *,
        out_dtype: jnp.dtype = jnp.float32,
        window_sizes: tuple[int, int] = (64, 64),
        strides: Sequence[int] = (128, 128, 128),
        mode: ExecutionMode | str = ExecutionMode.FAST,
        return_aux: bool = False,
    ) -> tuple[jax.Array, jax.Array | None]:
      self.assertIsInstance(window_sizes, tuple)
      self.assertIsInstance(strides, tuple)
      self.assertIsInstance(mode, ExecutionMode)
      scale = float(window_sizes[0] + strides[0])
      if mode == ExecutionMode.ACCURATE:
        scale *= 2.0
      out = (x + y) * scale
      if bias is not None:
        out = out + bias
      out = out.astype(out_dtype)
      aux = jnp.sum(out, axis=-1) if return_aux else None
      return out, aux

    op = pallas.jax_op("test::custom_fused_kernel", custom_fused_kernel)

    x = torch.ones((4, 8), dtype=torch.bfloat16, device="tpu")
    y = torch.full((4, 8), 2.0, dtype=torch.bfloat16, device="tpu")

    # Invalid enum value rejection
    with self.assertRaises(ValueError):
      op(
          x,
          y,
          mode="INVALID_MODE",
      )

  def test_donate_argnums_with_static_arguments(self):
    """Verifies donate_argnums properly maps when static arguments precede tensors."""

    def kernel_donate_with_static(
        scale: int,
        x: jax.Array,
        y: jax.Array,
    ) -> jax.Array:
      return (x + y) * scale

    op = pallas.jax_op(
        "test::kernel_donate_with_static",
        kernel_donate_with_static,
        donate_argnums=(
            2,
        ),  # donate y (argument 2 in signature, index 1 in tensors)
    )
    t1 = torch.ones((2, 2), dtype=torch.float32, device="tpu")
    t2 = torch.full((2, 2), 2.0, dtype=torch.float32, device="tpu")
    res = op(3, t1, t2)
    test_utils.assert_close(res.cpu(), torch.full((2, 2), 9.0))
    self._assert_donated(t2)

  def test_donate_argnums_with_optional_and_sequence_tensors(self):
    """Verifies donate_argnums with optional None and sequence tensors."""

    # 1. Optional tensor passed as None or Tensor preceding a donated tensor
    def kernel_donate_with_optional(
        opt: jax.Array | None,
        x: jax.Array,
    ) -> jax.Array:
      return x * 2.0 if opt is None else x + opt

    op_opt = pallas.jax_op(
        "test::kernel_donate_with_optional",
        kernel_donate_with_optional,
        donate_argnums=(1,),
    )
    t_donated_none = torch.full((2, 2), 3.0, dtype=torch.float32, device="tpu")
    res_none = op_opt(None, t_donated_none)
    test_utils.assert_close(res_none.cpu(), torch.full((2, 2), 6.0))
    self._assert_donated(t_donated_none)

    t_opt = torch.ones((2, 2), dtype=torch.float32, device="tpu")
    t_donated_some = torch.full((2, 2), 3.0, dtype=torch.float32, device="tpu")
    res_some = op_opt(t_opt, t_donated_some)
    test_utils.assert_close(res_some.cpu(), torch.full((2, 2), 4.0))
    test_utils.assert_close(t_opt.cpu(), torch.ones((2, 2)))
    self._assert_donated(t_donated_some)

    # 2. Sequence of tensors preceding a donated tensor
    def kernel_donate_after_seq(
        xs: Sequence[jax.Array],
        y: jax.Array,
    ) -> jax.Array:
      return xs[0] + xs[1] + y

    op_seq_after = pallas.jax_op(
        "test::kernel_donate_after_seq",
        kernel_donate_after_seq,
        donate_argnums=(1,),
    )
    s0 = torch.ones((2, 2), dtype=torch.float32, device="tpu")
    s1 = torch.full((2, 2), 2.0, dtype=torch.float32, device="tpu")
    y_donated = torch.full((2, 2), 4.0, dtype=torch.float32, device="tpu")
    res_seq_after = op_seq_after([s0, s1], y_donated)
    test_utils.assert_close(res_seq_after.cpu(), torch.full((2, 2), 7.0))
    test_utils.assert_close(s0.cpu(), torch.ones((2, 2)))
    test_utils.assert_close(s1.cpu(), torch.full((2, 2), 2.0))
    self._assert_donated(y_donated)

    # 3. Donating a sequence of tensors itself (requires matching output buffers for XLA aliasing)
    def kernel_donate_seq(
        xs: Sequence[jax.Array],
        y: jax.Array,
    ) -> list[jax.Array]:
      return [xs[0] + y, xs[1] * y]

    op_seq_donate = pallas.jax_op(
        "test::kernel_donate_seq",
        kernel_donate_seq,
        donate_argnums=(0,),
    )
    d0 = torch.ones((2, 2), dtype=torch.float32, device="tpu")
    d1 = torch.full((2, 2), 2.0, dtype=torch.float32, device="tpu")
    y_kept = torch.full((2, 2), 4.0, dtype=torch.float32, device="tpu")
    res_seq_donate = op_seq_donate([d0, d1], y_kept)
    test_utils.assert_close(res_seq_donate[0].cpu(), torch.full((2, 2), 5.0))
    test_utils.assert_close(res_seq_donate[1].cpu(), torch.full((2, 2), 8.0))
    test_utils.assert_close(y_kept.cpu(), torch.full((2, 2), 4.0))
    self._assert_donated(d0)
    self._assert_donated(d1)


if __name__ == "__main__":
  absltest.main()
