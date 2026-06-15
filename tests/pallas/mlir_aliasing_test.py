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


def _slice_pallas_dus_kernel():
  """Returns a JAX fn that slices the bundle, runs a per-layer pallas kernel,
  then dynamic_update_slice's the result back into the bundle.

  This is the kernel shape produced by jax.lax.dynamic_update_slice_in_dim
  wrappers around a per-layer pallas kernel — the kernel pattern that
  vLLM's block-major KV cache uses in production. The pallas custom_call
  itself operates on a per-layer slice, but the donation contract is at
  the BUNDLE level (the @main arg whose tf.aliasing_output maps it to
  the full-bundle output).
  """

  def kern(b, o):
    o[...] = b[...] + 1.0

  def f(bundle, layer_idx):
    per_layer = jax.lax.dynamic_slice_in_dim(bundle, layer_idx, 1, axis=1)
    per_layer = per_layer.squeeze(axis=1)
    new_per_layer = pl.pallas_call(
        kern, out_shape=jax.ShapeDtypeStruct(per_layer.shape, per_layer.dtype)
    )(per_layer)
    new_per_layer_expanded = jnp.expand_dims(new_per_layer, axis=1)
    return jax.lax.dynamic_update_slice_in_dim(
        bundle, new_per_layer_expanded, layer_idx, axis=1
    )

  return f


def _arg_passthrough_kernel():
  """Returns a kernel where the donated arg is never consumed in-place.

  The @main of this kernel just returns the donated arg unchanged.
  inject_inplace_donate_aliases should refuse to annotate this since
  there is no pallas custom_call OR DUS to honor the donation contract.
  """

  def f(bundle):
    return bundle

  return f


class InjectInplaceDonateAliasesDusChainTest(absltest.TestCase):
  """Tests for the slice→pallas→DUS pattern.

  These exercise the chain-walking logic that lands
  ``output_operand_aliases`` on the DUS when the donated bundle is
  consumed by a parent dynamic_update_slice instead of being passed
  directly into the pallas custom_call.
  """

  def test_slice_pallas_dus_annotates_dus(self):
    """The DUS that writes back to the donated bundle gets aliased."""
    f = _slice_pallas_dus_kernel()
    jit_fn = jax.jit(f, donate_argnums=(0,))
    bundle = jax.ShapeDtypeStruct((8, 4, 8), jnp.float32)
    layer_idx = jax.ShapeDtypeStruct((), jnp.int32)
    serialized = _export_kernel(jit_fn, bundle, layer_idx)

    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    self.assertNotEqual(out, serialized, "Expected the module to be rewritten.")
    text = _to_text(out)
    # The DUS — not the (per-layer) pallas custom_call — owns the alias
    # since the bundle flows through it as operand 0.
    dus_pos = text.find("stablehlo.dynamic_update_slice")
    self.assertGreaterEqual(dus_pos, 0)
    # Find the aliasing tag, anchor it to the same line/op as the DUS.
    alias_tag = "#stablehlo.output_operand_alias"
    self.assertIn(alias_tag, text)
    self.assertIn("output_tuple_indices = []", text)
    self.assertIn("operand_index = 0", text)
    # And the per-layer custom_call is left alone (the per-layer slice
    # is a transient temporary, not the donated bundle).
    cc_pos = text.find("stablehlo.custom_call @tpu_custom_call")
    self.assertGreaterEqual(cc_pos, 0)
    # The alias tag must appear AFTER the custom_call (on the DUS), not
    # inside the custom_call's attribute dict.
    last_cc_attr = text.find("}", cc_pos)
    self.assertGreater(
        text.find(alias_tag, last_cc_attr),
        last_cc_attr,
        "Alias annotation must be on the DUS, not the per-layer custom_call.",
    )

  def test_pure_passthrough_arg_raises(self):
    """A donated arg with no in-place consumer raises (contract violated)."""
    f = _arg_passthrough_kernel()
    jit_fn = jax.jit(f, donate_argnums=(0,))
    serialized = _export_kernel(jit_fn, _placeholder())
    with self.assertRaisesRegex(
        RuntimeError, "no consumer that can honor the contract"
    ):
      _mlir_aliasing.inject_inplace_donate_aliases(
          serialized, donate_argnums=(0,)
      )

  def test_slice_pallas_dus_idempotent(self):
    """Running the inject twice is a no-op the second time."""
    f = _slice_pallas_dus_kernel()
    jit_fn = jax.jit(f, donate_argnums=(0,))
    bundle = jax.ShapeDtypeStruct((8, 4, 8), jnp.float32)
    layer_idx = jax.ShapeDtypeStruct((), jnp.int32)
    serialized = _export_kernel(jit_fn, bundle, layer_idx)

    # First inject: lands the alias on the DUS.
    once = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    # Second inject is a no-op (idempotent).
    twice = _mlir_aliasing.inject_inplace_donate_aliases(
        once, donate_argnums=(0,)
    )
    self.assertEqual(once, twice)


if __name__ == "__main__":
  absltest.main()
