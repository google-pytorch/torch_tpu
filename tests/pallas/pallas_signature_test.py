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

from __future__ import annotations

import types
from typing import Optional
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing


class MyCustomType:
  pass


class PallasSignatureTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_valid_signature_with_future_annotations(self):

    def valid_fn(
        x: jax.Array,
        labels: jax.Array,
        reduction: str,
    ) -> tuple[jax.Array, jax.Array]:
      del reduction  # Unused.
      return x, labels

    # Should register without ValueError despite future annotations
    # stringifying types.
    op = pallas.jax_op("test::valid_fn", valid_fn)
    self.assertIsNotNone(op)

    # Verify evaluated types in the underlying JaxCallable signature.
    sig = op._init_fn.__signature__
    self.assertEqual(sig.parameters["x"].annotation, torch.Tensor)
    self.assertEqual(sig.parameters["labels"].annotation, torch.Tensor)
    self.assertEqual(sig.parameters["reduction"].annotation, str)
    self.assertEqual(sig.return_annotation, tuple[torch.Tensor, torch.Tensor])

  def test_output_partition_specs_drive_torch_placeholder_shape(self):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:1]), ("x",))
    spec = jax.sharding.PartitionSpec("x", None)

    def fn(x: jax.Array) -> jax.Array:
      return x

    op = pallas.jax_op(
        "test::output_partition_specs",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=(spec,),
    )
    self.assertEqual(op._init_fn.output_partition_specs, (spec,))

    aval = jax.core.ShapedArray((8, 4), jnp.float32)
    shape_mesh = types.SimpleNamespace(
        axis_names=("x",), devices=types.SimpleNamespace(shape=(2,))
    )
    with mock.patch.object(pallas.pallas.torch, "empty") as torch_empty:
      pallas.pallas.torch_placeholder(
          aval, mesh=shape_mesh, partition_spec=spec
      )

    torch_empty.assert_called_once_with((4, 4), dtype=torch.float32, device="tpu")

  def test_invalid_base_type_rejected(self):

    # dict is not supported as argument.
    def invalid_fn_dict(x: jax.Array, y: dict[str, int]) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, "Arguments at indices \\[1\\] are invalid"
    ):
      pallas.jax_op("test::invalid_dict", invalid_fn_dict)

    # Custom classes are not supported.
    def invalid_fn_custom(x: jax.Array, y: MyCustomType) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, "Arguments at indices \\[1\\] are invalid"
    ):
      pallas.jax_op("test::invalid_custom", invalid_fn_custom)

  def test_invalid_return_type_rejected(self):
    """Verifies unsupported return types are rejected."""

    # MyCustomType is not supported as return type.
    def invalid_return_fn(x: jax.Array) -> MyCustomType:
      del x  # Unused.
      return MyCustomType()

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test::invalid_return", invalid_return_fn)

  def test_jax_callable_fallback_path(self):
    """Verifies JaxCallable fallback path (jax_fn=None) works."""

    def my_func(x: jax.Array) -> jax.Array:
      return x

    jitted = jax.jit(my_func)

    # Instantiate JaxCallable directly.
    jc = pallas.pallas.JaxCallable(
        name="test_fallback",
        jit_fn=jitted,
        trace_key="test_key",
    )

    self.assertIsNotNone(jc.__signature__)
    sig = jc.__signature__
    self.assertEqual(sig.parameters["x"].annotation, torch.Tensor)
    self.assertEqual(sig.return_annotation, torch.Tensor)

  def test_valid_jitted_function_success(self):
    """Verifies valid jitted functions work with pallas.jax_op."""

    def my_fn(x: jax.Array) -> jax.Array:
      return x

    jitted = jax.jit(my_fn)
    op = pallas.jax_op("test::jitted_valid", jitted)
    self.assertIsNotNone(op)

    sig = op._init_fn.__signature__
    self.assertEqual(sig.parameters["x"].annotation, torch.Tensor)
    self.assertEqual(sig.return_annotation, torch.Tensor)

  def test_invalid_jitted_function_rejected(self):
    """Verifies invalid jitted functions are rejected by pallas.jax_op."""

    def my_fn(x: MyCustomType) -> MyCustomType:
      return x

    jitted = jax.jit(my_fn)
    with self.assertRaisesRegex(
        ValueError, "Arguments at indices \\[0\\] are invalid"
    ):
      pallas.jax_op("test::jitted_invalid", jitted)

  def test_optional_types_handled(self):
    """Verifies Optional types are correctly mapped to torch.Tensor | types.NoneType."""

    def optional_fn_1(x: jax.Array | None) -> jax.Array:
      return x

    def optional_fn_2(x: Optional[jax.Array]) -> jax.Array:
      return x

    op1 = pallas.jax_op("test::optional_1", optional_fn_1)
    self.assertIsNotNone(op1)
    sig1 = op1._init_fn.__signature__
    self.assertEqual(
        sig1.parameters["x"].annotation, torch.Tensor | types.NoneType
    )
    self.assertEqual(sig1.return_annotation, torch.Tensor)

    op2 = pallas.jax_op("test::optional_2", optional_fn_2)
    self.assertIsNotNone(op2)
    sig2 = op2._init_fn.__signature__
    self.assertEqual(
        sig2.parameters["x"].annotation, torch.Tensor | types.NoneType
    )
    self.assertEqual(sig2.return_annotation, torch.Tensor)

  def test_unannotated_arguments_rejected(self):
    """Verifies unannotated arguments are rejected."""

    def unannotated_fn(x) -> jax.Array:
      return x

    with self.assertRaisesRegex(
        ValueError, "Missing argument type annotation for JAX function"
    ):
      pallas.jax_op("test::unannotated", unannotated_fn)


if __name__ == "__main__":

  absltest.main()
