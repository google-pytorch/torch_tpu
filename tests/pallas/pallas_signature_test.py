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
import functools
import types
from typing import Any, Literal
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils

# `jax.jit` supports negative `donate_argnums` only from JAX 0.12 on; older
# versions, which OSS TorchTPU still uses, ignore them, and so does `jax_op`.
_JAX_SUPPORTS_NEGATIVE_DONATE_ARGNUMS = jax.__version_info__ >= (0, 12, 0)


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


def _donate_kernel(
    x: jax.Array, kv: jax.Array, scale: int, *, mode: str = "a"
) -> tuple[jax.Array, jax.Array]:
  del mode  # Unused.
  return x * scale, kv


def _pair_kernel(x: jax.Array, kv: jax.Array) -> tuple[jax.Array, jax.Array]:
  return x, kv


def _pos_only_kernel(
    x: jax.Array, /, kv: jax.Array
) -> tuple[jax.Array, jax.Array]:
  return x, kv


def _kwonly_tensor_kernel(
    x: jax.Array, kv: jax.Array, *, extra: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  return x, kv, extra


def _var_keyword_kernel(
    x: jax.Array, **extras: jax.Array
) -> tuple[jax.Array, ...]:
  return (x, *extras.values())


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

    def optional_fn_2(x: jax.Array | None) -> jax.Array:
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

  def test_abstract_mesh_multi_axis_partition_spec_and_negative_donate_argnums(
      self,
  ):
    abstract_mesh = jax.sharding.AbstractMesh((2, 2), ("dp", "tp"))
    pspec = jax.sharding.PartitionSpec(("dp", "tp"), None)

    def sharded_pair_fn(
        x: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
      return x + y, x - y

    op = pallas.jax_op(
        "test::sharded_pair_abstract_mesh",
        sharded_pair_fn,
        donate_argnums=(-1,),
        mesh=abstract_mesh,
        input_partition_specs=(pspec, pspec),
        output_partition_specs=pspec,
    )
    if _JAX_SUPPORTS_NEGATIVE_DONATE_ARGNUMS:
      self.assertEqual(op._init_fn._donate_argnums_set, {1})

    with torch._subclasses.fake_tensor.FakeTensorMode():
      fake_x = torch.empty((4, 8), dtype=torch.float32, device="tpu")
      fake_y = torch.empty((4, 8), dtype=torch.float32, device="tpu")
      out_sum, out_diff = op(fake_x, fake_y)
      self.assertEqual(out_sum.shape, (4, 8))
      self.assertEqual(out_diff.shape, (4, 8))

    cpu_x = torch.ones((4, 8), dtype=torch.float32)
    cpu_y = torch.ones((4, 8), dtype=torch.float32)
    with (
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas, "register_custom_kernel"
        ),
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas,
            "lookup_custom_kernel",
            return_value=True,
        ),
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas,
            "call_custom_kernel",
            side_effect=lambda *_, output_shapes, **__: list(output_shapes),
        ) as mock_call,
    ):
      op._init_fn(cpu_x, cpu_y)
      self.assertEqual(mock_call.call_args.kwargs["device_assignment"], [])
      if _JAX_SUPPORTS_NEGATIVE_DONATE_ARGNUMS:
        self.assertEqual(mock_call.call_args.kwargs["donate_argnums"], [1])

      concrete_mesh = jax.make_mesh(
          (1,),
          ("dp",),
          axis_types=(jax.sharding.AxisType.Explicit,),
          devices=jax.devices()[:1],
      )
      concrete_pspec = jax.sharding.PartitionSpec("dp", None)
      concrete_op = pallas.jax_op(
          "test::sharded_pair_concrete_mesh",
          sharded_pair_fn,
          donate_argnums=(-2,),
          mesh=concrete_mesh,
          input_partition_specs=(concrete_pspec, concrete_pspec),
          output_partition_specs=concrete_pspec,
      )
      concrete_op._init_fn(cpu_x, cpu_y)
      self.assertEqual(
          mock_call.call_args.kwargs["device_assignment"],
          [int(jax.devices()[0].id)],
      )
      if _JAX_SUPPORTS_NEGATIVE_DONATE_ARGNUMS:
        self.assertEqual(mock_call.call_args.kwargs["donate_argnums"], [0])

      dup_mesh = types.SimpleNamespace(
          size=1,
          devices=types.SimpleNamespace(
              flat=[types.SimpleNamespace(id=0), types.SimpleNamespace(id=0)]
          ),
      )
      with (
          mock.patch.object(
              torch.distributed, "is_initialized", return_value=True
          ),
          mock.patch.object(
              torch.distributed, "get_world_size", return_value=2
          ),
          self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing duplicate mesh device IDs
              ValueError, "Pallas kernel mesh device ids must be unique"
          ),
      ):
        pallas.pallas._validate_pallas_mesh_size(mesh=dup_mesh)

  @parameterized.named_parameters(
      # (name, fn, donate_argnames, positional args, keyword args, donated
      # leaves)
      (
          "keyword_only",
          _kwonly_tensor_kernel,
          ("extra",),
          ("x", "kv"),
          ("extra",),
          [2],
      ),
      (
          "var_keyword",
          _var_keyword_kernel,
          ("bonus",),
          ("x",),
          ("bonus",),
          [1],
      ),
      # `x` is donated at the position jax.jit inferred from its name.
      (
          "positional_and_keyword_only",
          _kwonly_tensor_kernel,
          ("x", "extra"),
          ("x", "kv"),
          ("extra",),
          [0, 2],
      ),
      (
          "positional_passed_by_keyword",
          _kwonly_tensor_kernel,
          ("kv",),
          ("x",),
          ("kv", "extra"),
          [1],
      ),
  )
  def test_custom_jax_kernel_donates_keyword_arguments_by_name(
      self, fn, donate_argnames, arg_names, kwarg_names, expected_donated
  ):
    """Verifies custom_jax_kernel donates keyword arguments by name.

    As in `jax.jit`, names donate arguments passed by keyword, including
    `**kwargs` entries. Only `custom_jax_kernel` can receive tensors by keyword:
    PyTorch passes `jax_op` tensors positionally.
    """
    args = [torch.ones((4, 4), dtype=torch.float32) for _ in arg_names]
    kwargs = {
        name: torch.ones((4, 4), dtype=torch.float32) for name in kwarg_names
    }
    kernel = pallas.custom_jax_kernel(
        fn, donate_argnames=donate_argnames, warn_deprecated=False
    )
    self.assertEqual(
        self._get_runtime_donor_indices(kernel, *args, **kwargs),
        expected_donated,
    )
    self.assertEqual(
        self._get_exported_donor_indices(kernel, *args, **kwargs),
        expected_donated,
    )

  def test_custom_jax_kernel_rejects_donate_argnames_with_aliases(self):
    """Verifies donate_argnames can't be combined with input_output_aliases."""
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=aliases with argnames
        ValueError, "donate_argnames cannot be combined"
    ):
      pallas.custom_jax_kernel(
          _pair_kernel,
          input_output_aliases={1: 1},
          donate_argnames="kv",
          warn_deprecated=False,
      )

  @parameterized.named_parameters(
      ("str_name", {"donate_argnames": "kv"}),
      ("tuple_name", {"donate_argnames": ("kv",)}),
      ("num", {"donate_argnums": (1,)}),
  )
  def test_jax_op_donate_argnames_matches_export(self, donate_kwargs):
    """Verifies donating `kv` by name matches donating it by number."""
    op = pallas.jax_op(
        f"test::{self._testMethodName}", _donate_kernel, **donate_kwargs
    )
    args = (
        torch.ones((4, 4), dtype=torch.float32),
        torch.full((4, 4), 2.0, dtype=torch.float32),
        2,
    )
    self.assertEqual(self._get_runtime_donor_indices(op, *args, mode="b"), [1])
    self.assertEqual(self._get_exported_donor_indices(op, *args, mode="b"), [1])

  @parameterized.named_parameters(
      ("unknown_name", _donate_kernel, "nope"),
      ("static_positional", _donate_kernel, "scale"),
      ("static_kwonly", _donate_kernel, "mode"),
      ("positional_only", _pos_only_kernel, "x"),
      # `scale` bound by keyword becomes a keyword-only static argument.
      (
          "partial_keyword_static",
          functools.partial(_donate_kernel, scale=2),
          "scale",
      ),
  )
  def test_jax_op_rejects_invalid_donate_argnames(self, fn, donate_argnames):
    """Verifies jax_op rejects invalid donate_argnames with ValueError."""
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=invalid donate_argnames
        ValueError, "donate_argnames"
    ):
      pallas.jax_op(
          f"test::{self._testMethodName}", fn, donate_argnames=donate_argnames
      )

  @parameterized.named_parameters(
      ("nums", (0,)),
      # `donate_argnums=()` counts as given, as in jax.jit.
      ("empty_nums", ()),
  )
  def test_jax_op_rejects_donate_argnums_with_donate_argnames(
      self, donate_argnums
  ):
    """Verifies jax_op rejects donate_argnums together with donate_argnames."""
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=both donation specs
        ValueError, "not both"
    ):
      pallas.jax_op(
          f"test::{self._testMethodName}",
          _pair_kernel,
          donate_argnums=donate_argnums,
          donate_argnames=("kv",),
      )

  def test_custom_jax_kernel_rejects_donate_argnums_with_donate_argnames(self):
    """Verifies custom_jax_kernel rejects both donation specs too."""
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=both donation specs
        ValueError, "not both"
    ):
      pallas.custom_jax_kernel(
          _pair_kernel,
          donate_argnums=[0],
          donate_argnames=("kv",),
          warn_deprecated=False,
      )

  @parameterized.named_parameters(
      ("optional_none", False, [0, 1]),
      ("optional_tensor", True, [1, 2]),
  )
  def test_jax_op_donates_sequence_by_name(
      self, pass_optional, expected_donated
  ):
    """Verifies jax_op donates every tensor of a sequence donated by name."""

    def seq_opt_fn(
        opt: jax.Array | None,
        xs: Sequence[jax.Array],
        kv: jax.Array,
        scale: int,
        *,
        mode: str = "a",
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      del mode  # Unused.
      base = xs[0] if opt is None else xs[0] + opt
      return base * scale, xs[1], kv

    op = pallas.jax_op(
        f"test::{self._testMethodName}", seq_opt_fn, donate_argnames=("xs",)
    )
    opt = torch.ones((4, 4), dtype=torch.float32) if pass_optional else None
    xs = [
        torch.ones((4, 4), dtype=torch.float32),
        torch.full((4, 4), 2.0, dtype=torch.float32),
    ]
    args = (opt, xs, torch.full((4, 4), 3.0, dtype=torch.float32), 2)
    self.assertEqual(
        self._get_runtime_donor_indices(op, *args), expected_donated
    )
    self.assertEqual(
        self._get_exported_donor_indices(op, *args), expected_donated
    )

  def test_jax_op_donates_by_name_through_positional_partial(self):
    """Verifies donation by name through a partial that binds `x`."""
    op = pallas.jax_op(
        f"test::{self._testMethodName}",
        functools.partial(_donate_kernel, jnp.ones((4, 4), dtype=jnp.float32)),
        donate_argnames="kv",
    )
    args = (torch.full((4, 4), 3.0, dtype=torch.float32), 2)
    self.assertEqual(self._get_runtime_donor_indices(op, *args), [0])
    self.assertEqual(self._get_exported_donor_indices(op, *args), [0])

  def test_jax_op_donates_by_name_through_keyword_partial(self):
    """Verifies donation by name through a partial that binds `scale`."""
    op = pallas.jax_op(
        f"test::{self._testMethodName}",
        functools.partial(_donate_kernel, scale=2),
        donate_argnames=("kv",),
    )
    args = (
        torch.ones((4, 4), dtype=torch.float32),
        torch.full((4, 4), 3.0, dtype=torch.float32),
    )
    self.assertEqual(self._get_runtime_donor_indices(op, *args), [1])
    self.assertEqual(self._get_exported_donor_indices(op, *args), [1])

  def _get_runtime_donor_indices(
      self, op: Any, *args: Any, **kwargs: Any
  ) -> list[int]:
    """Calls the kernel with mocked TPU calls; returns the donated inputs."""
    with (
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas, "register_custom_kernel"
        ),
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas,
            "lookup_custom_kernel",
            return_value=True,
        ),
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas,
            "call_custom_kernel",
            side_effect=lambda *_, output_shapes, **__: list(output_shapes),
        ) as mock_call,
    ):
      getattr(op, "_init_fn", op)(*args, **kwargs)
    return mock_call.call_args.kwargs["donate_argnums"]

  def _get_exported_donor_indices(
      self, op: Any, *args: Any, **kwargs: Any
  ) -> list[int]:
    """Returns the indices of the inputs the exported module donates."""
    # pylint: disable=protected-access
    jax_callable = getattr(op, "_init_fn", op)
    norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
        args, kwargs
    )
    jax_args, jax_kwargs = jax_callable._get_jax_placeholders(
        norm_args, norm_kwargs
    )
    with jax._src.config.export_ignore_forward_compatibility(True):
      exported = jax_callable.exported(*jax_args, **jax_kwargs)
    # pylint: enable=protected-access
    ir = jax.interpreters.mlir.ir
    with jax.interpreters.mlir.make_ir_context():
      module = ir.Module.parse(exported.mlir_module())
      for func in module.body.operations:
        attrs = func.operation.attributes
        if (
            func.operation.name == "func.func"
            and ir.StringAttr(attrs["sym_name"]).value == "main"
        ):
          return [
              i
              for i, arg_attrs in enumerate(func.arg_attrs or [])
              if "jax.buffer_donor" in str(arg_attrs)
              or "tf.aliasing_output" in str(arg_attrs)
          ]
    self.fail("The exported module has no main function.")

  def test_negative_donate_argnums_resolve_like_jax_jit(self):
    """Verifies negative donate_argnums donate what the exported module does.

    `jax.jit` resolves negative indices relative to the positional parameters,
    and doesn't resolve them if the function takes `*args`.
    """
    if not _JAX_SUPPORTS_NEGATIVE_DONATE_ARGNUMS:
      self.skipTest("jax.jit ignores negative donate_argnums before JAX 0.12.")

    def kwonly_fn(
        x: jax.Array, kv: jax.Array, *, mode: str = "a"
    ) -> tuple[jax.Array, jax.Array]:
      del mode  # Unused.
      return x, kv

    def three_tensors_kwonly_fn(
        x: jax.Array, y: jax.Array, z: jax.Array, *, mode: str = "a"
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      del mode  # Unused.
      return x, y, z

    def var_keyword_fn(
        x: jax.Array, kv: jax.Array, **extra: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
      del extra  # Unused.
      return x, kv

    def var_positional_fn(
        x: jax.Array, *rest: jax.Array
    ) -> tuple[jax.Array, ...]:
      return (x, *rest)

    t0 = torch.ones((4, 4), dtype=torch.float32)
    t1 = torch.full((4, 4), 2.0, dtype=torch.float32)
    t2 = torch.full((4, 4), 3.0, dtype=torch.float32)

    with (
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas, "register_custom_kernel"
        ),
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas,
            "lookup_custom_kernel",
            return_value=True,
        ),
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas,
            "call_custom_kernel",
            side_effect=lambda *_, output_shapes, **__: list(output_shapes),
        ) as mock_call,
    ):
      cases = [
          # -1 is `kv`; the keyword-only `mode` is not counted.
          (
              "keyword_only",
              pallas.jax_op(
                  "test::negative_donate_kwonly",
                  kwonly_fn,
                  donate_argnums=(-1,),
              ),
              (t0, t1),
              [1],
          ),
          # -2 is `y`, not `z`.
          (
              "keyword_only_three_tensors",
              pallas.jax_op(
                  "test::negative_donate_three_kwonly",
                  three_tensors_kwonly_fn,
                  donate_argnums=(-2,),
              ),
              (t0, t1, t2),
              [1],
          ),
          # -1 is `kv`; `**extra` is not counted.
          (
              "var_keyword",
              pallas.custom_jax_kernel(
                  var_keyword_fn, donate_argnums=[-1], warn_deprecated=False
              ),
              (t0, t1),
              [1],
          ),
          # With `*args`, `jax.jit` leaves -1 unresolved and donates nothing.
          (
              "var_positional",
              pallas.custom_jax_kernel(
                  var_positional_fn, donate_argnums=[-1], warn_deprecated=False
              ),
              (t0, t1, t2),
              [],
          ),
      ]
      for case_name, op, args, expected_donated in cases:
        with self.subTest(case_name):
          getattr(op, "_init_fn", op)(*args)
          self.assertEqual(
              mock_call.call_args.kwargs["donate_argnums"], expected_donated
          )
          self.assertEqual(
              self._get_exported_donor_indices(op, *args), expected_donated
          )

  def test_input_output_aliases_donate_aliased_inputs(self):
    """Verifies the deprecated input_output_aliases still donate their inputs.

    `jax.jit` isn't told about them, so they're donated in addition to the
    arguments `jax.jit` resolved.
    """

    def accumulate_fn(x: jax.Array, acc: jax.Array) -> jax.Array:
      return acc + x

    kernel = pallas.custom_jax_kernel(
        accumulate_fn, input_output_aliases={1: 0}, warn_deprecated=False
    )
    t_x = torch.ones((4, 4), dtype=torch.float32)
    t_acc = torch.zeros((4, 4), dtype=torch.float32)

    with (
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas, "register_custom_kernel"
        ),
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas,
            "lookup_custom_kernel",
            return_value=True,
        ),
        mock.patch.object(
            pallas.pallas.tpu_torch_pallas,
            "call_custom_kernel",
            # `t_acc` starts as zeros, so return values to validate it
            side_effect=lambda *_, output_shapes, **__: [
                torch.full(placeholder.shape, 5.0, dtype=placeholder.dtype)
                for placeholder in output_shapes
            ],
        ) as mock_call,
    ):
      kernel(t_x, t_acc)

    self.assertEqual(mock_call.call_args.kwargs["donate_argnums"], [1])
    test_utils.assert_close(t_acc, torch.full((4, 4), 5.0))


if __name__ == "__main__":

  absltest.main()
