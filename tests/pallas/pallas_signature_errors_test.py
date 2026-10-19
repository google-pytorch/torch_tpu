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

from absl.testing import absltest
import jax
import jax.numpy as jnp
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


class PallasSignatureTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_invalid_base_type_rejected(self):

    # dict is not supported as argument.
    def invalid_fn_dict(x: jax.Array, y: dict[str, int]) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[1\] are invalid"
    ):
      pallas.jax_op("test::invalid_dict", invalid_fn_dict)

    # Custom classes are not supported.
    def invalid_fn_custom(x: jax.Array, y: MyCustomType) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[1\] are invalid"
    ):
      pallas.jax_op("test::invalid_custom", invalid_fn_custom)

    # Sequence of str is not supported in PyTorch custom op schemas.
    def invalid_fn_seq_str(x: jax.Array, y: Sequence[str]) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[1\] are invalid"
    ):
      pallas.jax_op("test::invalid_seq_str", invalid_fn_seq_str)

    # Sequence of enum is not supported in PyTorch custom op schemas.
    def invalid_fn_seq_enum(x: jax.Array, y: Sequence[MyStrEnum]) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[1\] are invalid"
    ):
      pallas.jax_op("test::invalid_seq_enum", invalid_fn_seq_enum)

  def test_invalid_return_type_rejected(self):
    """Verifies unsupported return types are rejected."""

    # MyCustomType is not supported as return type.
    def invalid_return_fn(x: jax.Array) -> MyCustomType:
      del x  # Unused.
      return MyCustomType()

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test::invalid_return", invalid_return_fn)

    # String is not supported as return type.
    def fn_return_str(x: jax.Array) -> str:
      del x  # Unused.
      return "foo"

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test::fn_return_str", fn_return_str)

    # Dtype is not supported as return type.
    def fn_return_dtype(x: jax.Array) -> jnp.dtype:
      del x  # Unused.
      return jnp.float32

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test::fn_return_dtype", fn_return_dtype)

    # Enum is not supported as return type.
    def fn_return_enum(x: jax.Array) -> MyEnum:
      del x  # Unused.
      return MyEnum.OPT_1

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test::fn_return_enum", fn_return_enum)

  def test_invalid_jitted_function_rejected(self):
    """Verifies invalid jitted functions are rejected by pallas.jax_op."""

    def my_fn(x: MyCustomType) -> MyCustomType:
      return x

    jitted = jax.jit(my_fn)
    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[0\] are invalid"
    ):
      pallas.jax_op("test::jitted_invalid", jitted)

  def test_unannotated_arguments_rejected(self):
    """Verifies unannotated arguments are rejected."""

    def unannotated_fn(x) -> jax.Array:
      return x

    with self.assertRaisesRegex(
        ValueError, "Missing argument type annotation for JAX function"
    ):
      pallas.jax_op("test::unannotated", unannotated_fn)

  def test_dataclass_and_custom_object_rejected(self):
    """Verifies dataclasses and custom objects are rejected as unsupported."""

    def fn_with_dataclass(x: jax.Array, cfg: MyDataclass) -> jax.Array:
      del cfg
      return x

    with self.assertRaisesRegex(
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_dataclass", fn_with_dataclass)

    def fn_with_kw_dataclass(
        x: jax.Array, *, cfg: MyDataclass = MyDataclass()
    ) -> jax.Array:
      del cfg
      return x

    with self.assertRaisesRegex(
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_kw_dataclass", fn_with_kw_dataclass)

    def fn_with_custom_obj(x: jax.Array, obj: MyCustomType) -> jax.Array:
      del obj
      return x

    with self.assertRaisesRegex(
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_custom_obj", fn_with_custom_obj)

    def fn_with_dict_return(x: jax.Array) -> dict[str, jax.Array]:
      return {"out": x}

    with self.assertRaisesRegex(
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_dict_return", fn_with_dict_return)

    def fn_with_set_return(x: jax.Array) -> set[jax.Array]:
      return {x}

    with self.assertRaisesRegex(
        ValueError, "Invalid signature for JAX function"
    ):
      pallas.jax_op("test::fn_with_set_return", fn_with_set_return)

  def test_invalid_union_return_type_rejected(self):
    """Verifies mixed array/scalar union return annotations are rejected."""

    def fn_return_invalid_union(x: jax.Array) -> jax.Array | int:
      return x

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test::fn_return_invalid_union", fn_return_invalid_union)


if __name__ == "__main__":

  absltest.main()
