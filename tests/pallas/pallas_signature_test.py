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

from collections.abc import Sequence
import dataclasses
import enum
import types
from typing import Literal, Optional

from absl.testing import absltest
import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from tests import seed_test_utils


class MyCustomType:
  pass


@dataclasses.dataclass
class MyDataclass:
  bq: int = 16


class MyStrEnum(enum.StrEnum):
  MODE_A = "mode_a"
  MODE_B = "mode_b"


class MyEnum(enum.Enum):
  OPT_1 = 1
  OPT_2 = 2


class MyIntSubtype(int):
  pass


class PallasSignatureTest(seed_test_utils.RepeatableTest):

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

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid signature rejection
        ValueError, "Arguments at indices \\[1\\] are invalid"
    ):
      pallas.jax_op("test::invalid_custom", invalid_fn_custom)

    # Sequence of str is not supported in PyTorch custom op schemas.
    def invalid_fn_seq_str(x: jax.Array, y: Sequence[str]) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid signature rejection
        ValueError, "Arguments at indices \\[1\\] are invalid"
    ):
      pallas.jax_op("test::invalid_seq_str", invalid_fn_seq_str)

    # Sequence of enum is not supported in PyTorch custom op schemas.
    def invalid_fn_seq_enum(x: jax.Array, y: Sequence[MyStrEnum]) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid signature rejection
        ValueError, "Arguments at indices \\[1\\] are invalid"
    ):
      pallas.jax_op("test::invalid_seq_enum", invalid_fn_seq_enum)

  def test_invalid_return_type_rejected(self):
    """Verifies unsupported return types are rejected."""

    # MyCustomType is not supported as return type.
    def invalid_return_fn(x: jax.Array) -> MyCustomType:
      del x  # Unused.
      return MyCustomType()

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid return rejection
        ValueError, "The return annotation is invalid"
    ):
      pallas.jax_op("test::invalid_return", invalid_return_fn)

    # String is not supported as return type.
    def fn_return_str(x: jax.Array) -> str:
      del x  # Unused.
      return "foo"

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid return rejection
        ValueError, "The return annotation is invalid"
    ):
      pallas.jax_op("test::fn_return_str", fn_return_str)

    # Dtype is not supported as return type.
    def fn_return_dtype(x: jax.Array) -> jnp.dtype:
      del x  # Unused.
      return jnp.float32

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid return rejection
        ValueError, "The return annotation is invalid"
    ):
      pallas.jax_op("test::fn_return_dtype", fn_return_dtype)

    # Enum is not supported as return type.
    def fn_return_enum(x: jax.Array) -> MyEnum:
      del x  # Unused.
      return MyEnum.OPT_1

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid return rejection
        ValueError, "The return annotation is invalid"
    ):
      pallas.jax_op("test::fn_return_enum", fn_return_enum)

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

  def test_custom_op_sequence_and_container_types_supported(self):
    """Verifies all PyTorch custom_op container, dtype, enum, and scalar types."""

    def complex_sig_fn(
        x: jax.Array,
        tensors: Sequence[jax.Array],
        opt_tensors: list[jax.Array | None],
        dims_seq: Sequence[int],
        dims_list: list[int],
        dims_tuple: tuple[int, ...],
        window: tuple[int, int] = (4, 4),
        opt_dims: Sequence[int] | None = None,
        *,
        scales: Sequence[float] | None = None,
        flags: tuple[bool, ...] | None = None,
        num: torch.types.Number = 1.0,
        out_dtype: jnp.dtype | None = None,
        torch_dt: torch.dtype = torch.bfloat16,
        dev: torch.device | None = None,
        str_enum: MyStrEnum | str = MyStrEnum.MODE_A,
        plain_enum: MyEnum = MyEnum.OPT_1,
        lit_mode: Literal["fast", "accurate"] = "fast",
    ) -> list[jax.Array]:
      del (
          tensors,
          opt_tensors,
          dims_seq,
          dims_list,
          dims_tuple,
          window,
          opt_dims,
          scales,
          flags,
          num,
          out_dtype,
          torch_dt,
          dev,
          str_enum,
          plain_enum,
          lit_mode,
      )
      return [x]

    op = pallas.jax_op("test::complex_sig_fn", complex_sig_fn)
    self.assertIsNotNone(op)
    sig = op._init_fn.__signature__
    self.assertEqual(
        sig.parameters["tensors"].annotation, Sequence[torch.Tensor]
    )
    self.assertEqual(
        sig.parameters["opt_tensors"].annotation,
        list[torch.Tensor | types.NoneType],
    )
    self.assertEqual(sig.parameters["dims_seq"].annotation, Sequence[int])
    self.assertEqual(sig.parameters["dims_list"].annotation, list[int])
    self.assertEqual(sig.parameters["dims_tuple"].annotation, Sequence[int])
    self.assertEqual(
        sig.parameters["window"].annotation, Sequence[int] | types.NoneType
    )
    self.assertEqual(
        sig.parameters["out_dtype"].annotation, torch.dtype | types.NoneType
    )
    self.assertEqual(sig.parameters["str_enum"].annotation, str)
    self.assertEqual(sig.parameters["str_enum"].default, "mode_a")
    self.assertEqual(sig.return_annotation, list[torch.Tensor])
    # Tensor and tensor-sequence params (0, 1, 2) must not be marked static.
    self.assertNotIn(0, op._init_fn.static_argnums)
    self.assertNotIn(1, op._init_fn.static_argnums)
    self.assertNotIn(2, op._init_fn.static_argnums)
    # Non-tensor sequence and keyword-only params must be marked static.
    self.assertIn(3, op._init_fn.static_argnums)
    self.assertIn("scales", op._init_fn.static_argnames)
    self.assertIn("out_dtype", op._init_fn.static_argnames)
    self.assertIn("str_enum", op._init_fn.static_argnames)

    # Verify that the op executes successfully through fake-tensor tracing,
    # exercising dispatch, schema parsing, and argument normalization.
    with torch._subclasses.fake_tensor.FakeTensorMode():
      fake_x = torch.empty((2, 2), dtype=torch.float32, device="tpu")
      fake_tensors = [torch.empty((2, 2), dtype=torch.float32, device="tpu")]
      res = op(
          fake_x,
          fake_tensors,
          [None],
          (1, 2),
          [3, 4],
          (5, 6),
          window=(4, 4),
          opt_dims=None,
          scales=[1.0, 2.0],
          flags=(True, False),
          num=1.0,
          out_dtype=None,
          torch_dt=torch.bfloat16,
          dev=None,
          str_enum=MyStrEnum.MODE_B,
          plain_enum=MyEnum.OPT_2,
          lit_mode="accurate",
      )
      self.assertEqual(len(res), 1)
      self.assertEqual(res[0].shape, (2, 2))

  def test_dataclass_and_custom_object_rejected(self):
    """Verifies dataclasses and custom objects are rejected as unsupported."""

    def fn_with_dataclass(x: jax.Array, cfg: MyDataclass) -> jax.Array:
      del cfg
      return x

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid signature rejection
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_dataclass", fn_with_dataclass)

    def fn_with_kw_dataclass(
        x: jax.Array, *, cfg: MyDataclass = MyDataclass()
    ) -> jax.Array:
      del cfg
      return x

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid signature rejection
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_kw_dataclass", fn_with_kw_dataclass)

    def fn_with_custom_obj(x: jax.Array, obj: MyCustomType) -> jax.Array:
      del obj
      return x

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid signature rejection
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_custom_obj", fn_with_custom_obj)

    def fn_with_dict_return(x: jax.Array) -> dict[str, jax.Array]:
      return {"out": x}

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid signature rejection
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_dict_return", fn_with_dict_return)

    def fn_with_set_return(x: jax.Array) -> set[jax.Array]:
      return {x}

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid signature rejection
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_set_return", fn_with_set_return)

  def test_optional_argument_with_default_preserves_none(self):

    def fn_with_opt(
        x: jax.Array, *, out_dtype: jnp.dtype | None = jnp.float32
    ) -> jax.Array:
      del out_dtype
      return x

    op = pallas.jax_op("test::fn_with_opt", fn_with_opt)
    # Verify argument normalization preserves explicit None even when default is non-None
    _, norm_kwargs = op._init_fn._normalize_invocation_args(
        [], {"out_dtype": None}
    )
    self.assertIsNone(norm_kwargs["out_dtype"])
    _, jax_kwargs = op._init_fn._get_jax_placeholders([], norm_kwargs)
    self.assertIsNone(jax_kwargs["out_dtype"])

    with torch._subclasses.fake_tensor.FakeTensorMode():
      fake_x = torch.empty((2, 2), dtype=torch.float32, device="tpu")
      res = op(fake_x, out_dtype=None)
      self.assertEqual(res.shape, (2, 2))

  def test_scalar_union_and_int_subclass_return_types(self):
    """Verifies int subclasses and scalar Unions in return annotations."""

    def fn_return_int_subclass(x: jax.Array) -> MyIntSubtype:
      del x
      return MyIntSubtype(1)

    op_int_sub = pallas.jax_op(
        "test::fn_return_int_subclass", fn_return_int_subclass
    )
    self.assertEqual(op_int_sub._init_fn.__signature__.return_annotation, int)

    def fn_return_scalar_union(x: jax.Array) -> int | float:
      del x
      return 1.0

    op_scalar_union = pallas.jax_op(
        "test::fn_return_scalar_union", fn_return_scalar_union
    )
    self.assertEqual(
        op_scalar_union._init_fn.__signature__.return_annotation,
        torch.types.Number,
    )

    def fn_return_invalid_union(x: jax.Array) -> jax.Array | int:
      return x

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing invalid return union rejection
        ValueError, "The return annotation is invalid"
    ):
      pallas.jax_op("test::fn_return_invalid_union", fn_return_invalid_union)


if __name__ == "__main__":

  absltest.main()
