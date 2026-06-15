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


class InjectInplaceDonateAliasesNestedCallTest(absltest.TestCase):
  """Tests for the JAX-export-with-nested-func.call pattern.

  When a jit'd inner function is invoked from an outer jit's body,
  ``jax.export.export`` lifts the inner function into a private callee
  and the outer @main just forwards arguments via ``func.call``. The
  donated arg's only direct consumer is then the ``func.call`` op —
  not a pallas custom_call and not a DUS. The chain walker must follow
  the call into the callee and annotate the in-place op that lives
  there. (Without this, the vLLM block-major KV kernel's exported
  shape fails the contract check at register time and the wheel
  refuses to install the kernel.)
  """

  @staticmethod
  def _bundled_jit_inside_jit():
    """Returns a wrapped(bundle, layer_idx, x) → bundle whose exported
    module has @main → func.call @inner → DUS."""

    def kern(b, x, o):
      o[...] = b[...] + x[...]

    @jax.jit
    def inner(bundle, layer_idx, x):
      per_layer = jax.lax.dynamic_slice_in_dim(bundle, layer_idx, 1, axis=1)
      per_layer = per_layer.squeeze(axis=1)
      new_per_layer = pl.pallas_call(
          kern, out_shape=jax.ShapeDtypeStruct(per_layer.shape, per_layer.dtype)
      )(per_layer, x)
      new_per_layer_expanded = jnp.expand_dims(new_per_layer, axis=1)
      return jax.lax.dynamic_update_slice_in_dim(
          bundle, new_per_layer_expanded, layer_idx, axis=1
      )

    def wrapped(bundle, layer_idx, x):
      return inner(bundle, layer_idx, x)

    return wrapped

  def _export_nested(self):
    wrapped = self._bundled_jit_inside_jit()
    jit_fn = jax.jit(wrapped, donate_argnums=(0,))
    return _export_kernel(
        jit_fn,
        jax.ShapeDtypeStruct((4, 3, 8), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((4, 8), jnp.float32),
    )

  def test_nested_call_module_has_two_functions(self):
    """Sanity: the export really does produce @main + a callee.

    If JAX ever changes its export shape to inline the inner jit, the
    other tests in this class would become accidentally trivial — this
    one fails loudly so the regression is visible.
    """
    serialized = self._export_nested()
    text = _to_text(serialized)
    self.assertIn("func.func public @main", text)
    self.assertIn("func.func private @", text)
    self.assertIn("call @", text)

  def test_nested_call_aliases_dus_in_callee(self):
    """The DUS inside the callee gets aliased; the outer @main is
    structurally unchanged.
    """
    serialized = self._export_nested()
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    self.assertNotEqual(out, serialized)
    text = _to_text(out)
    # The alias appears exactly once — on the DUS inside the callee.
    self.assertEqual(text.count("#stablehlo.output_operand_alias"), 1)
    # Anchor the alias to the DUS line, not to the func.call or @main.
    dus_line_start = text.find("stablehlo.dynamic_update_slice")
    self.assertGreaterEqual(dus_line_start, 0)
    dus_line_end = text.find("\n", dus_line_start)
    self.assertGreater(dus_line_end, dus_line_start)
    self.assertIn(
        "output_operand_aliases", text[dus_line_start:dus_line_end]
    )

  def test_nested_call_pallas_only_kernel_aliases_custom_call(self):
    """Variant: the callee body is a direct pallas custom_call with no
    DUS. The alias still lands on the custom_call across the call
    boundary.
    """

    def kern(x, y, o):
      o[...] = x[...] + y[...]

    @jax.jit
    def inner(x, y):
      return pl.pallas_call(
          kern, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x, y)

    def wrapped(x, y):
      return inner(x, y)

    jit_fn = jax.jit(wrapped, donate_argnums=(0,))
    serialized = _export_kernel(jit_fn, _placeholder(), _placeholder())
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    self.assertNotEqual(out, serialized)
    text = _to_text(out)
    cc_line_start = text.find("stablehlo.custom_call @tpu_custom_call")
    self.assertGreaterEqual(cc_line_start, 0)
    cc_line_end = text.find("\n", cc_line_start)
    self.assertIn(
        "output_operand_aliases", text[cc_line_start:cc_line_end]
    )


class InjectInplaceDonateAliasesViewOpChainTest(absltest.TestCase):
  """Tests for view-op traversal in the donation chain walker.

  ``stablehlo.reshape``, ``stablehlo.transpose``,
  ``stablehlo.broadcast_in_dim``, and
  ``stablehlo.bitcast_convert_type`` represent no-copy buffer-view
  transforms — the carried buffer-view is the same underlying
  allocation. The walker must thread the donation chain through them
  transparently without recording a chain step for the view itself.
  """

  def test_reshape_pallas_reshape_aliases_custom_call(self):
    """arg → reshape → pallas → reshape → return: alias lands on pallas."""

    def kern(b, o):
      o[...] = b[...] + 1.0

    def f(bundle):
      flat = bundle.reshape(
          bundle.shape[0] * bundle.shape[1], *bundle.shape[2:]
      )
      new_flat = pl.pallas_call(
          kern, out_shape=jax.ShapeDtypeStruct(flat.shape, flat.dtype)
      )(flat)
      return new_flat.reshape(bundle.shape)

    jit_fn = jax.jit(f, donate_argnums=(0,))
    bundle = jax.ShapeDtypeStruct((4, 3, 8), jnp.float32)
    serialized = _export_kernel(jit_fn, bundle)
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    self.assertNotEqual(out, serialized)
    text = _to_text(out)
    # Anchor the alias to the custom_call line itself, not to a reshape
    # or to @main's signature.
    cc_line_start = text.find("stablehlo.custom_call @tpu_custom_call")
    self.assertGreaterEqual(cc_line_start, 0)
    cc_line_end = text.find("\n", cc_line_start)
    self.assertIn(
        "output_operand_aliases", text[cc_line_start:cc_line_end]
    )

  def test_pallas_with_existing_alias_no_overwrite(self):
    """Reshape→pallas where the pallas already has output_operand_aliases.

    Mirrors the bundled-RPA shape: 10 inputs, 2 outputs, with operand 9
    aliased to result 1 via ``input_output_aliases``. The walker must
    NOT crash on ``op.results[9]`` and must NOT overwrite the existing
    alias.
    """

    def kern(*refs):
      out_refs = refs[-2:]
      in_refs = refs[:-2]
      out_refs[0][...] = in_refs[0][...] + 1.0
      out_refs[1][...] = in_refs[-1][...] * 2.0

    num_in = 10

    def f(bundle, *others):
      flat = bundle.reshape(
          bundle.shape[0] * bundle.shape[1], *bundle.shape[2:]
      )
      inputs = list(others) + [flat]
      out, new_flat = pl.pallas_call(
          kern,
          out_shape=[
              jax.ShapeDtypeStruct((8, 8), jnp.float32),
              jax.ShapeDtypeStruct(flat.shape, flat.dtype),
          ],
          input_output_aliases={num_in - 1: 1},
      )(*inputs)
      return out, new_flat.reshape(bundle.shape)

    jit_fn = jax.jit(f, donate_argnums=(0,))
    bundle = jax.ShapeDtypeStruct((4, 3, 8), jnp.float32)
    others = [
        jax.ShapeDtypeStruct((8, 8), jnp.float32) for _ in range(num_in - 1)
    ]
    serialized = _export_kernel(jit_fn, bundle, *others)
    # The pre-existing alias must be present in the exported module.
    self.assertIn("output_operand_aliases", _to_text(serialized))
    # The injection must succeed (no IndexError) and leave the bytes
    # unchanged since the kernel's own alias is already correct.
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    self.assertEqual(out, serialized)
    # Verify the alias entry survived and still maps operand 9 → output 1.
    text = _to_text(out)
    self.assertIn("operand_index = 9", text)
    self.assertIn("output_tuple_indices = [1]", text)

  def test_transpose_pallas_aliases_custom_call(self):
    """A transpose between arg and pallas is traversed."""

    def kern(b, o):
      o[...] = b[...] + 1.0

    def f(bundle):
      transposed = jnp.transpose(bundle, (1, 0))
      new = pl.pallas_call(
          kern,
          out_shape=jax.ShapeDtypeStruct(transposed.shape, transposed.dtype),
      )(transposed)
      return jnp.transpose(new, (1, 0))

    jit_fn = jax.jit(f, donate_argnums=(0,))
    bundle = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    serialized = _export_kernel(jit_fn, bundle)
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    self.assertNotEqual(out, serialized)
    text = _to_text(out)
    cc_line_start = text.find("stablehlo.custom_call @tpu_custom_call")
    self.assertGreaterEqual(cc_line_start, 0)
    cc_line_end = text.find("\n", cc_line_start)
    self.assertIn(
        "output_operand_aliases", text[cc_line_start:cc_line_end]
    )

  def test_pallas_operand_index_out_of_range_raises(self):
    """N-in/M-out kernel with M<N and no alias: walker returns no step.

    The donated arg flows into a high operand position whose default
    ``operand i → result i`` mapping would index past the end of the
    custom_call's result list. Without the
    ``_resolve_pallas_output_index`` bounds check this would
    ``IndexError`` deep inside ``_walk_donation_chain``; instead the
    walker treats the call as a non-step and the chain is empty, which
    surfaces as the contract-violation error.
    """

    def kern(*refs):
      out_refs = refs[-2:]
      in_refs = refs[:-2]
      out_refs[0][...] = in_refs[0][...] + 1.0
      out_refs[1][...] = in_refs[1][...] * 2.0

    num_in = 10

    def f(*inputs):
      return pl.pallas_call(
          kern,
          out_shape=[
              jax.ShapeDtypeStruct((8, 8), jnp.float32),
              jax.ShapeDtypeStruct((8, 8), jnp.float32),
          ],
      )(*inputs)

    # Donate the LAST input — at operand 9, with only 2 results, the
    # default 9 → result 9 mapping is out of range AND no
    # ``input_output_aliases`` is declared so the read-from-attribute
    # path returns nothing either.
    jit_fn = jax.jit(f, donate_argnums=(num_in - 1,))
    placeholders = [
        jax.ShapeDtypeStruct((8, 8), jnp.float32) for _ in range(num_in)
    ]
    serialized = _export_kernel(jit_fn, *placeholders)
    with self.assertRaisesRegex(
        RuntimeError, "no consumer that can honor the contract"
    ):
      _mlir_aliasing.inject_inplace_donate_aliases(
          serialized, donate_argnums=(num_in - 1,)
      )

  def test_reshape_only_passthrough_raises(self):
    """arg → reshape → return (no in-place op) raises the contract error."""

    def f(bundle):
      flat = bundle.reshape(bundle.shape[0] * bundle.shape[1], *bundle.shape[2:])
      return flat.reshape(bundle.shape)

    jit_fn = jax.jit(f, donate_argnums=(0,))
    bundle = jax.ShapeDtypeStruct((4, 3, 8), jnp.float32)
    serialized = _export_kernel(jit_fn, bundle)
    with self.assertRaisesRegex(
        RuntimeError, "no consumer that can honor the contract"
    ):
      _mlir_aliasing.inject_inplace_donate_aliases(
          serialized, donate_argnums=(0,)
      )

  def test_reshape_dus_chain_aliases_dus(self):
    """arg → reshape → DUS-back chain: alias lands on the DUS."""

    def f(bundle, layer_idx):
      flat = bundle.reshape(
          bundle.shape[0] * bundle.shape[1], *bundle.shape[2:]
      )
      replacement = jnp.zeros((1, *flat.shape[1:]), flat.dtype)
      new_flat = jax.lax.dynamic_update_slice_in_dim(
          flat, replacement, layer_idx, axis=0
      )
      return new_flat.reshape(bundle.shape)

    jit_fn = jax.jit(f, donate_argnums=(0,))
    bundle = jax.ShapeDtypeStruct((4, 3, 8), jnp.float32)
    layer_idx = jax.ShapeDtypeStruct((), jnp.int32)
    serialized = _export_kernel(jit_fn, bundle, layer_idx)
    out = _mlir_aliasing.inject_inplace_donate_aliases(
        serialized, donate_argnums=(0,)
    )
    self.assertNotEqual(out, serialized)
    text = _to_text(out)
    dus_line_start = text.find("stablehlo.dynamic_update_slice")
    self.assertGreaterEqual(dus_line_start, 0)
    dus_line_end = text.find("\n", dus_line_start)
    self.assertIn(
        "output_operand_aliases", text[dus_line_start:dus_line_end]
    )


if __name__ == "__main__":
  absltest.main()
