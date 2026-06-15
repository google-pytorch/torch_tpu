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

"""Unit tests for `_mlir_aliasing.inject_inplace_donate_aliases`.

These tests construct exported pallas-kernel MLIR via JAX (without
actually running the kernel on TPU) and assert the post-processed
module has the expected ``output_operand_aliases`` attribute on its
inner ``stablehlo.custom_call``. They are pure-Python tests with no
torch/TPU dependency.
"""

from absl.testing import absltest
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from torch_tpu._internal.pallas import _mlir_aliasing


def _export_kernel(jit_fn, *placeholders):
  """Returns the exported MLIR bytes for the given jit'd kernel."""
  with jax._src.config.export_ignore_forward_compatibility(True):
    exported = jax.export.export(jit_fn, platforms=["tpu"])
  return exported(*placeholders).mlir_module_serialized


def _to_text(serialized: bytes) -> str:
  """Deserializes VHLO bytecode and returns the textual StableHLO."""
  ctx = ir.Context()
  ctx.allow_unregistered_dialects = True
  stablehlo.register_dialect(ctx)
  return str(stablehlo.deserialize_portable_artifact(ctx, serialized))


def _placeholder(shape=(8, 8), dtype=jnp.float32):
  return jax.ShapeDtypeStruct(shape, dtype)


def _single_output_kernel():
  def kern(b, o):
    o[...] = b[...] + 1.0

  def f(bundle):
    return pl.pallas_call(
        kern, out_shape=jax.ShapeDtypeStruct(bundle.shape, bundle.dtype)
    )(bundle)

  return f


def _multi_output_kernel(num_outputs=2):
  def kern(b, x, *out_refs):
    out_refs[0][...] = b[...] + x[...]
    if len(out_refs) > 1:
      out_refs[1][...] = b[...] * 2.0
    if len(out_refs) > 2:
      out_refs[2][...] = x[...] - b[...]

  def f(bundle, x):
    out_shapes = [
        jax.ShapeDtypeStruct(bundle.shape, bundle.dtype)
        for _ in range(num_outputs)
    ]
    return pl.pallas_call(kern, out_shape=out_shapes)(bundle, x)

  return f


class InjectInplaceDonateAliasesTest(absltest.TestCase):

  def test_single_output_donate_zero(self):
    """Single-output kernel: donate 0 → output 0 (empty tuple indices)."""
    f = _single_output_kernel()
    jit_fn = jax.jit(f, donate_argnums=(0,))
    serialized = _export_kernel(jit_fn, _placeholder())
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    text = _to_text(out)
    self.assertIn("output_operand_aliases", text)
    # For a single-result custom_call, the output path is empty.
    self.assertIn("output_tuple_indices = []", text)
    self.assertIn("operand_index = 0", text)

  def test_multi_output_donate_zero(self):
    """Multi-output kernel: donate 0 → output 0 (tuple index [0])."""
    f = _multi_output_kernel(num_outputs=2)
    jit_fn = jax.jit(f, donate_argnums=(0,))
    serialized = _export_kernel(jit_fn, _placeholder(), _placeholder())
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    text = _to_text(out)
    self.assertIn("output_operand_aliases", text)
    # For a multi-result custom_call, the output path carries the index.
    self.assertIn("output_tuple_indices = [0]", text)
    self.assertIn("operand_index = 0", text)

  def test_multi_output_donate_both(self):
    """Both inputs donated 1:1 against outputs."""
    f = _multi_output_kernel(num_outputs=2)
    jit_fn = jax.jit(f, donate_argnums=(0, 1))
    serialized = _export_kernel(jit_fn, _placeholder(), _placeholder())
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0, 1)
    )
    text = _to_text(out)
    self.assertIn("operand_index = 0", text)
    self.assertIn("operand_index = 1", text)
    self.assertIn("output_tuple_indices = [0]", text)
    self.assertIn("output_tuple_indices = [1]", text)

  def test_three_outputs_donate_two(self):
    """3-output kernel donating only input 0 produces only one alias entry."""
    f = _multi_output_kernel(num_outputs=3)
    jit_fn = jax.jit(f, donate_argnums=(0,))
    serialized = _export_kernel(jit_fn, _placeholder(), _placeholder())
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    text = _to_text(out)
    # Exactly one alias entry — count occurrences of the per-entry tag.
    self.assertEqual(text.count("#stablehlo.output_operand_alias"), 1)
    self.assertIn("output_tuple_indices = [0]", text)

  def test_empty_donate_argnums_is_noop(self):
    """Caller passes empty donate_argnums — module returned unchanged."""
    f = _single_output_kernel()
    jit_fn = jax.jit(f)  # no donation
    serialized = _export_kernel(jit_fn, _placeholder())
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=()
    )
    self.assertEqual(out, serialized)

  def test_user_already_set_aliases_is_noop(self):
    """Module with pre-existing output_operand_aliases is left alone."""

    def kern(b, o):
      o[...] = b[...] + 1.0

    def f(bundle):
      return pl.pallas_call(
          kern,
          out_shape=jax.ShapeDtypeStruct(bundle.shape, bundle.dtype),
          input_output_aliases={0: 0},
      )(bundle)

    jit_fn = jax.jit(f, donate_argnums=(0,))
    serialized = _export_kernel(jit_fn, _placeholder())
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    # We respect user-set aliasing rather than overwriting it.
    self.assertEqual(out, serialized)
    # Sanity: the pre-existing aliasing is present.
    self.assertIn("output_operand_aliases", _to_text(serialized))

  def test_no_donation_recorded_is_noop(self):
    """No tf.aliasing_output on @main args → no-op even if caller asks."""
    # JIT without donate_argnums: JAX does not set tf.aliasing_output.
    f = _single_output_kernel()
    jit_fn = jax.jit(f)
    serialized = _export_kernel(jit_fn, _placeholder())
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    # No JIT-level donation was recorded, so the module is unchanged.
    self.assertEqual(out, serialized)

  def test_caller_donate_mismatch_raises(self):
    """Caller-supplied donate_argnums must match JIT-recorded set."""
    f = _multi_output_kernel(num_outputs=2)
    # JIT donates only arg 0, but caller claims arg 1.
    jit_fn = jax.jit(f, donate_argnums=(0,))
    serialized = _export_kernel(jit_fn, _placeholder(), _placeholder())
    with self.assertRaisesRegex(
        RuntimeError, "donate_argnums.*disagrees"
    ):
      _mlir_aliasing.inject_inplace_donate_aliases(
          serialized, donate_argnums=(1,)
      )

  def test_emitted_aliases_survive_round_trip(self):
    """Re-serializing then re-deserializing preserves the aliases."""
    f = _multi_output_kernel(num_outputs=2)
    jit_fn = jax.jit(f, donate_argnums=(0,))
    serialized = _export_kernel(jit_fn, _placeholder(), _placeholder())
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    # Round trip via the same deserializer the C++ side will use.
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    stablehlo.register_dialect(ctx)
    mod = stablehlo.deserialize_portable_artifact(ctx, out)
    text = str(mod)
    self.assertIn("output_operand_aliases", text)
    self.assertIn("operand_index = 0", text)


if __name__ == "__main__":
  absltest.main()
