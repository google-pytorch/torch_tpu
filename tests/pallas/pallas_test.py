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

import functools
import os
import pathlib
import sys
from typing import Optional, Tuple, TypeAlias
import warnings
from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
import jax.export
from packaging import version
import torch
import torch._library.custom_ops as torch_custom_ops
from torch_tpu._internal import compile  # pylint: disable=redefined-builtin
from torch_tpu._internal import execution_mode
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.pallas import _compat
from torch_tpu._internal.utils import utils

EagerMode: TypeAlias = execution_mode.EagerMode


def _is_deferred_mode():
  return execution_mode.eager_mode in (
      EagerMode.DEFER_AND_FUSE,
      EagerMode.INTERNAL_DEFER_ALL,
  )


def add_vectors_kernel_body(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y


def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  dt = jax.ShapeDtypeStruct(x.shape, x.dtype)
  pallas_call = pl.pallas_call(add_vectors_kernel_body, out_shape=dt)
  return pallas_call(x, y)


@pallas.jax_op("pallas::add_vectors")
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return add_vectors_jax(x, y)


def add_subtract_vectors_kernel(x_ref, y_ref, oadd_ref, osub_ref):
  x, y = x_ref[...], y_ref[...]
  oadd_ref[...] = x + y
  osub_ref[...] = x - y


@pallas.jax_op("pallas::add_subtract_vectors")
def add_subtract_vectors(
    x: jax.Array, y: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  dt = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return pl.pallas_call(add_subtract_vectors_kernel, out_shape=(dt, dt))(x, y)


def subtract_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x - y


@pallas.jax_op("pallas::subtract_vectors")
def subtract_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      subtract_vectors_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)


@pallas.jax_op("pallas::pass_through")
def pass_through(x: jax.Array) -> jax.Array:
  return x


def jax_add_or_subtract_vectors_wrapper(
    mode: str, x: jax.Array, y: jax.Array
) -> jax.Array:
  """Emulate kernel lib like tokamax with thin JAX wrapper around pallas.

  TorchTPU custom kernel calls require that all input and output arguments are
  tensor (or None) types, so a function like this must be further wrapped before
  use, i.e. `functools.partial(jax_add_or_subtract_vectors_wrapper, "add")`.

  This constraint can be changed but should be revisited once use cases and
  usability requirements are more concrete.

  Args:
    mode: The mode to use, either "add" or "sub".
    x: The first input tensor.
    y: The second input tensor.

  Returns:
    The result of the addition or subtraction.
  """

  @functools.partial(jax.jit, static_argnums=(0,))
  def _add_or_subtract_vectors_kernel(mode, x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    if mode == "add":
      o_ref[...] = x + y
    else:
      o_ref[...] = x - y

  out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
  wrapped = functools.partial(_add_or_subtract_vectors_kernel, mode)
  return pl.pallas_call(wrapped, out_shape=out_shape)(x, y)


class TestPallasKernels(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.device("tpu")

  def _assert_donated(self, x: torch.Tensor):
    with self.assertRaisesRegex(
        RuntimeError, "INVALID_ARGUMENT: Buffer has been deleted or donated"
    ):
      x.cpu()

  def test_kernel_single_output_decorator(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = add_vectors(x, y).to("cpu")
    utils.assert_close(actual, expected)

  def test_kernel_multiple_outputs(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected_add = torch.add(x, y).to("cpu")
    expected_sub = torch.sub(x, y).to("cpu")
    actual_add, actual_sub = add_subtract_vectors(x, y)
    utils.assert_close(actual_add.to("cpu"), expected_add)
    utils.assert_close(actual_sub.to("cpu"), expected_sub)

  def test_kernel_called_twice_same_shapes(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    z = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32, device=self.device)
    expected = torch.add(torch.add(x, y), z).to("cpu")
    actual = add_vectors(add_vectors(x, y), z).to("cpu")
    utils.assert_close(actual, expected)

  def test_kernel_called_twice_different_shapes(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    z = torch.tensor([0.7, 0.8], dtype=torch.float32, device=self.device)
    w = torch.tensor([0.9, 1.0], dtype=torch.float32, device=self.device)
    expected = torch.cat([torch.add(x, y), torch.add(z, w)]).to("cpu")
    actual = torch.cat([add_vectors(x, y), add_vectors(z, w)]).to("cpu")
    utils.assert_close(actual, expected)

  def test_two_different_kernels(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    z = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32, device=self.device)
    expected = torch.sub(torch.add(x, y), z).to("cpu")
    actual = subtract_vectors(add_vectors(x, y), z).to("cpu")
    utils.assert_close(actual, expected)

  def test_two_kernels_same_name_different_kwargs(self):

    def add_vectors_with_metadata_jax(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(
          add_vectors_kernel_body,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          debug=True,
          metadata={"foo": "bar"},
      )(x, y)

    add_vectors_with_metadata = pallas.jax_op(
        "pallas::add_vectors_with_metadata", add_vectors_with_metadata_jax
    )

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual_no_metadata = add_vectors(x, y).to("cpu")
    actual_with_metadata = add_vectors_with_metadata(x, y).to("cpu")
    utils.assert_close(actual_no_metadata, expected)
    utils.assert_close(actual_with_metadata, expected)

  @absltest.skip(
      "This is expected to fail now that we wrap in"
      " torch.library.custom_ops (see cl/892251072)."
  )
  def test_two_kernels_same_name_different_func(self):
    """Test two different kernels registered with the same name.

    The explicit name of a kernel is used for debuggability. The function ID
    should be used to distinguish different python kernels.
    """

    def sub_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x - y

    def sub_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(
          sub_vectors_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x, y)

    add_vectors_fn = pallas.jax_op("pallas::math_kernel", add_vectors_jax)
    sub_vectors_fn = pallas.jax_op("pallas::math_kernel", sub_vectors_jax)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected_add = torch.add(x, y).to("cpu")
    expected_sub = torch.sub(x, y).to("cpu")
    actual_add = add_vectors_fn(x, y).to("cpu")
    actual_sub = sub_vectors_fn(x, y).to("cpu")
    utils.assert_close(actual_add, expected_add)
    utils.assert_close(actual_sub, expected_sub)

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
    utils.assert_close(actual, expected)
    # x should be donated by the output
    self._assert_donated(x)
    # y should not be donated
    expected_y = torch.tensor(
        [0.4, 0.5, 0.6], dtype=torch.float32, device="cpu"
    )
    utils.assert_close(y.to("cpu"), expected_y)

  def test_kernel_donation_inplace(self):

    donating_add_vectors = pallas.jax_op(
        "pallas::donating_add_vectors",
        add_vectors_jax,
        donate_argnums=(0,),
    )

    def add_vectors_inplace(x: torch.Tensor, y: torch.Tensor) -> None:
      result = donating_add_vectors(x, y)
      x.set_(result)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    add_vectors_inplace(x, y)
    utils.assert_close(x, expected)
    expected_y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32)
    utils.assert_close(y.to("cpu"), expected_y)

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

  def test_pallas_kernel_compiled_mode(self):

    def add_vectors_backward_kernel(g_ref, o_x_ref, o_y_ref):
      g = g_ref[...]
      o_x_ref[...] = g
      o_y_ref[...] = g

    @pallas.jax_op("pallas::add_vectors_backward")
    def add_vectors_backward(g: jax.Array) -> tuple[jax.Array, jax.Array]:
      dt = jax.ShapeDtypeStruct(g.shape, g.dtype)
      return pl.pallas_call(add_vectors_backward_kernel, out_shape=(dt, dt))(g)

    add_vectors_op = pallas.jax_op(
        "pallas::add_vectors_with_autograd",
        add_vectors_jax,
    )

    add_vectors_op.register_autograd(
        lambda ctx, grad: add_vectors_backward(grad)
    )

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def add_vectors_sum(x, y):
      return add_vectors_op(x, y).sum()

    x = torch.tensor(
        [0.1, 0.2, 0.3], dtype=torch.float32, device=self.device
    ).requires_grad_(True)
    y = torch.tensor(
        [0.4, 0.5, 0.6], dtype=torch.float32, device=self.device
    ).requires_grad_(True)

    # Do the forward pass.
    expected_forward = torch.add(x, y).sum()
    actual_forward = add_vectors_sum(x, y)
    utils.assert_close(
        actual_forward.detach().to("cpu"), expected_forward.detach().to("cpu")
    )

    # Do the backward pass for the non-compiled run.
    expected_forward.backward()
    x_grad_expected = x.grad.to("cpu")
    y_grad_expected = y.grad.to("cpu")
    x.grad = None
    y.grad = None

    # Do the backward pass for the compiled run.
    actual_forward.backward()
    x_grad_actual = x.grad.to("cpu")
    y_grad_actual = y.grad.to("cpu")
    utils.assert_close(x_grad_actual, x_grad_expected)
    utils.assert_close(y_grad_actual, y_grad_expected)

  def test_kernel_donating_compiled_mode(self):

    torch_donating_add_vectors = pallas.jax_op(
        "pallas::add_vectors_donated",
        add_vectors_jax,
        donate_argnums=(0,),
    )

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def donated_add_vectors_sum(x, y):
      return torch_donating_add_vectors(x, y).sum()

    expected_updated_x = torch.add(
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device="cpu"),
        torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device="cpu"),
    )
    expected_sum = expected_updated_x.sum()

    actual_sum = donated_add_vectors_sum(x, y)
    utils.assert_close(actual_sum.to("cpu"), expected_sum)
    self._assert_donated(x)

  def test_kernel_donation_inplace_compiled_mode(self):

    donating_add_vectors = pallas.jax_op(
        "pallas::donating_add_vectors",
        add_vectors_jax,
        donate_argnums=(0,),
    )

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def add_vectors_inplace(x: torch.Tensor, y: torch.Tensor) -> None:
      result = donating_add_vectors(x, y)
      x.copy_(result)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    add_vectors_inplace(x, y)
    utils.assert_close(x, expected)
    expected_y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32)
    utils.assert_close(y.to("cpu"), expected_y)

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

    tpu_backend = compile.TpuBackend(debug=True)

    # Run a donating compiled operation.
    @torch.compile(fullgraph=True, dynamic=False, backend=tpu_backend)
    def donated_add_vectors_sum(x, y):
      return donating_add_vectors(x, y).sum()

    _ = donated_add_vectors_sum(x, y)

    executables = tpu_backend._compiled_executables
    self.assertLen(executables, 1)
    self.assertIn(
        "jax.buffer_donor",
        executables[0].mlir_text,
    )

    # The pre-donation value of x can no longer be used if we are in deferred
    # mode.
    if _is_deferred_mode():
      self._assert_donated(pre_x_sum)
    else:
      pre_x_sum.cpu()

  def test_kernel_compile_both_donated_and_non_donated_ops(self):

    donating_add_vectors = pallas.jax_op(
        "pallas::add_vectors_donated",
        add_vectors_jax,
        donate_argnums=(0,),
    )

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def x_used_and_donated(x, y):
      # This op uses x but does not donate it.
      x_sum = x.sum()
      # This op donates x.
      z = donating_add_vectors(x, y)

      return x_sum, z

    # This operation executes successfully, because both uses of x are provided
    # to XLA; it inserts defensive copies as needed.
    x_sum, z = x_used_and_donated(x, y)
    utils.assert_close(x_sum.cpu(), torch.tensor(0.6, dtype=torch.float32))
    utils.assert_close(
        z.cpu(), torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32)
    )

  def test_jax_kernel_wrapper_simple(self):
    """Test a kernel wrapper that is a simple passthrough to a JAX kernel."""

    @pallas.jax_op("pallas::simple_add_fn")
    def add_fn(x: jax.Array, y: jax.Array) -> jax.Array:
      return jax.numpy.add(x, y)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = add_fn(x, y).to("cpu")
    utils.assert_close(actual, expected)

  def test_jax_kernel_wrapper_with_donation(self):
    """Test a kernel wrapper with a donated argument."""

    @pallas.jax_op("pallas::add_fn", donate_argnums=(0,))
    def add_fn(x: jax.Array, y: jax.Array) -> jax.Array:
      return jax.numpy.add(x, y)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = add_fn(x, y)
    utils.assert_close(actual.cpu(), expected)
    # x should be donated by the output.
    self._assert_donated(x)

  def test_jax_kernel_with_trace_time_conditional(self):
    """Test a kernel wrapper that has a trace time conditional.

    This is a common requirement for kernel libraries like Tokamax, where
    different kernels are selected at call time based on compile time constants.

    Today TorchTPU requires all kernels to resolve their trace time args with
    a wrapper function, i.e. `functools.partial`. This can be revisited once
    use cases and usability requirements are more concrete.
    """
    add_fn = functools.partial(jax_add_or_subtract_vectors_wrapper, "add")
    sub_fn = functools.partial(jax_add_or_subtract_vectors_wrapper, "sub")

    add_kernel = pallas.jax_op("pallas::add_fn", add_fn)
    sub_kernel = pallas.jax_op("pallas::sub_fn", sub_fn)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    add_expected = torch.add(x, y).to("cpu")
    sub_expected = torch.sub(x, y).to("cpu")
    actual_add = add_kernel(x, y).to("cpu")
    actual_sub = sub_kernel(x, y).to("cpu")
    utils.assert_close(actual_add, add_expected)
    utils.assert_close(actual_sub, sub_expected)

  def test_jax_kernel_retains_unused_operands(self):
    """jax_op retains operands the traced function does not use.

    A JAX function that does not reference one of its tensor arguments would
    normally have that argument dead-code-eliminated by jax.jit, leaving the
    exported kernel with fewer operands than the torch wrapper passes (which
    then fails to specialize). jax_op always builds the kernel with
    keep_unused so the operand is retained.
    """

    def ignore_second(x: jax.Array, y: jax.Array) -> jax.Array:  # pylint: disable=unused-argument
      # y is intentionally unused; it must not be pruned from the kernel.
      return jax.numpy.multiply(x, 2.0)

    op = pallas.jax_op("keep_unused_test::ignore_second", ignore_second)
    x = torch.ones((16, 8), dtype=torch.float32, device=self.device)
    y = torch.ones((16, 8), dtype=torch.float32, device=self.device)
    expected = (x * 2).to("cpu")
    actual = op(x, y).to("cpu")
    utils.assert_close(actual, expected)

  def test_jax_kernel_with_static_argnums(self):
    """Test custom_jax_kernel with static_argnums."""
    kernel = pallas.jax_op(
        "pallas::jax_test", jax_add_or_subtract_vectors_wrapper
    )

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    add_expected = torch.add(x, y).to("cpu")
    sub_expected = torch.sub(x, y).to("cpu")
    actual_add = kernel("add", x, y).to("cpu")
    actual_sub = kernel("sub", x, y).to("cpu")
    utils.assert_close(actual_add, add_expected)
    utils.assert_close(actual_sub, sub_expected)

  def test_kernel_wrapper_with_none_return(self):
    """Simulate kernel libraries like tokamax that wrap pallas kernels."""

    @pallas.jax_op("pallas::jax_test")
    def wrapper() -> None:
      return None

    none = wrapper()
    self.assertIsNone(none)

  def test_kernel_wrapper_with_none_input(self):
    """Simulate kernel libraries like tokamax that wrap pallas kernels."""

    @pallas.jax_op("pallas::jax_test")
    def wrapper(x: jax.Array, y: Optional[jax.Array]) -> jax.Array:
      if y is None:
        return x
      return jax.numpy.add(x, y)

    x = torch.tensor([1.0, 2.0, 3.0], device=self.device)
    z = wrapper(x, None)
    utils.assert_close(z.cpu(), x.cpu())

  def test_kernel_wrapper_with_none_input_new_syntax(self):
    """Simulate kernel libraries like tokamax that wrap pallas kernels."""

    @pallas.jax_op("pallas::jax_test")
    def wrapper(x: jax.Array, y: jax.Array | None) -> jax.Array:
      if y is None:
        return x
      return jax.numpy.add(x, y)

    x = torch.tensor([1.0, 2.0, 3.0], device=self.device)
    z = wrapper(x, None)
    utils.assert_close(z.cpu(), x.cpu())

  def test_jax_op_with_defaulted_int(self):
    """Check that we can bind torch support-types to the kernel."""

    def bias_add_kernel(x: jax.Array, bias: int = 1) -> jax.Array:
      return x + bias

    bias_add_op = pallas.jax_op("pallas::bias_add_kernel", bias_add_kernel)

    x = torch.tensor([1.0, 2.0, 3.0], device=self.device)
    x_plus_one = bias_add_op(x)
    utils.assert_close(x_plus_one.cpu(), x.cpu() + 1)

    # Check that we can override the default value with a positional argument.
    x_plus_two = bias_add_op(x, 2)
    utils.assert_close(x_plus_two.cpu(), x.cpu() + 2)

    # Check that we can override the default value with a keyword argument.
    x_plus_two = bias_add_op(x, bias=3)
    utils.assert_close(x_plus_two.cpu(), x.cpu() + 3)

  def test_jax_op_with_bound_mesh(self):
    """Simulate distributed kernels.

    Including pretty much any kernel that takes a non-torch types.
    """

    def kernel(x: jax.Array, mesh: jax.sharding.Mesh) -> jax.Array:
      del mesh
      return x + 1

    x = torch.tensor([1.0, 2.0, 3.0], device=self.device)
    mesh = jax.sharding.Mesh(jax.numpy.arange(8), "x")

    kernel_op = pallas.jax_op(
        "pallas::jax_test", functools.partial(kernel, mesh=mesh)
    )
    z = kernel_op(x)
    utils.assert_close(z.cpu(), x.cpu() + 1)

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

  def test_fake_tensor_on_tpu(self):
    x = torch.randn(5, device="tpu")
    with torch._subclasses.FakeTensorMode() as mode:
      fake_x = mode.from_tensor(x)
      y = pass_through(fake_x)
      self.assertEqual(y.device, x.device)

  # The following aliasing tests are for the deprecated input_output_aliases
  # so are not comprehensive.
  def test_kernel_input_output_aliases(self):

    alised_add_vectors = pallas.custom_jax_kernel(
        add_vectors_jax,
        input_output_aliases={0: 0},
    )
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected_x = torch.add(x, y).to("cpu")
    actual_x = alised_add_vectors(x, y).to("cpu")
    utils.assert_close(actual_x, expected_x)
    utils.assert_close(x.to("cpu"), expected_x)

  def test_kernel_input_output_aliasing_compiled_mode(self):

    aliasing_add_vectors = pallas.custom_jax_kernel(
        add_vectors_jax,
        name="add_vectors",
        input_output_aliases={0: 0},
    )

    torch_aliasing_add_vectors = torch.library.custom_op(
        "pallas::aliasing_add_vectors",
        lambda x, y: aliasing_add_vectors(x, y).clone(),
        mutates_args=("x",),
        schema="(Tensor(a!) x, Tensor y) -> Tensor",
        device_types=["tpu"],
    )
    torch_aliasing_add_vectors.register_fake(lambda x, _: torch.empty_like(x))

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def aliased_add_vectors_sum(x, y):
      return torch_aliasing_add_vectors(x, y).sum()

    expected_updated_x = torch.add(
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device="cpu"),
        torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device="cpu"),
    )
    expected_sum = expected_updated_x.sum()

    actual_sum = aliased_add_vectors_sum(x, y)
    utils.assert_close(actual_sum.to("cpu"), expected_sum)
    utils.assert_close(x.to("cpu"), expected_updated_x)

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

  def test_jax_dot_grad_for_backwards(self):
    """Test usage of jax.grad to generate a backwards function.

    The pullback returned from VJP is not fully parameterized.
    Although strictly "purely functional",
    it is a closure that bakes in the inputs and is intended to be disposable.

    We need to match the PyTorch
    torch._library.custom_ops.CustomOpDef.register_autograd()
    expectation of a fully parameterized, reusable, non-disposable
    backward function.

    The math trick to do this is to multiply the outputs by their grad
    and sum them, or f(x) * grad_output. Then use jax.grad.

    The math there is that the upstream grads are a "linear model"
    of every single op that comes after this, by definition, since
    grad_output = d_L / d_output. But it gets treated as a constant
    by jax.grad. So when jax.grad calculates g'(x) where g(x) = f(x) * grad_y,
    g'(x) == f'(x) * grad_y. The first term is the Jacobian. The second
    op is the Product. And the final term is the vector of grads. Hence,
    we have created a Vector Jacobian Product function, fully parameterized.
    """

    # Create the kernel in JAX. Do not jax.jit this, jax_op will do it later.
    def mul_and_add_scalars_jax_kernel(
        x: jax.Array, y: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
      if x.ndim != 0 or y.ndim != 0:
        raise ValueError("Only scalars are supported.")
      return jax.numpy.multiply(x, y), jax.numpy.add(x, y)

    # Employ the scalar surrogate trick to make jax.grad
    # create the backward. We don't want to write it ourselves.
    # Do not jax.jit this, jax_op will do it later.
    # This trick will recalculate forward in cases where
    # the grad function inherently contains the forward, such as
    # the canonical example of exp(x). Advanced cases can
    # write the backward in math, or, for lax ops, attempt to
    # extract them from jax._src.interpreters.ad.primitive_jvps
    def mul_and_add_scalars_backward_jax_kernel(
        x: jax.Array, y: jax.Array, d_mul: jax.Array, d_add: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:

      def scalar_surrogate(x_local: jax.Array, y_local: jax.Array):
        output_mul, output_add = mul_and_add_scalars_jax_kernel(
            x_local, y_local
        )
        return output_mul * d_mul + output_add * d_add

      dx, dy = jax.grad(fun=scalar_surrogate, argnums=(0, 1))(x, y)

      return dx, dy

    # Register the forward and backwards as PyTorch custom ops.
    # This is consistent with `torch.library.triton_op`:
    # "If you want the backward to call Triton kernels,
    # then those must be wrapped in triton_op as well"
    # https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
    mul_and_add_scalars: torch._library.custom_ops.CustomOpDef = pallas.jax_op(
        "test::mul_and_add_scalars", mul_and_add_scalars_jax_kernel
    )
    mul_and_add_scalars_backward: torch._library.custom_ops.CustomOpDef = (
        pallas.jax_op(
            "test::mul_and_add_scalars_backward",
            mul_and_add_scalars_backward_jax_kernel,
        )
    )

    # We have the ops registered with PyTorch,
    # but now we need to register the second op
    # as the autograd for the first op.
    def setup_context(ctx, inputs, output):
      del output  # Unused
      ctx.save_for_backward(*inputs)

    def backward(ctx, d_mul, d_add):
      x, y = ctx.saved_tensors
      return mul_and_add_scalars_backward(x, y, d_mul, d_add)

    mul_and_add_scalars.register_autograd(backward, setup_context=setup_context)

    # We are done! We have a custom op implemented in JAX
    # with its own backward implemented in JAX, fully
    # registered with torch.library for composability with
    # torch.compile and autograd.

    # Arrange - Create input
    x = torch.tensor(2.0, requires_grad=True, device=self.device)
    y = torch.tensor(7.0, requires_grad=True, device=self.device)

    with self.subTest("test mul, dL/dout = 11.0"):
      # Act - Forward
      output_mul, output_add = mul_and_add_scalars(x, y)

      # Assert - Forward output
      # No need for utils.assert_close - these are integer-valued outputs.
      self.assertEqual(output_mul.item(), 14.0)
      self.assertEqual(output_add.item(), 9.0)

      # Act - backward on d_mul.
      output_mul.backward(
          torch.tensor(11.0, device=self.device), retain_graph=True
      )

      # Assert - backward on d_mul
      expected_dx = 7.0 * 11.0
      expected_dy = 2.0 * 11.0
      self.assertEqual(x.grad.item(), expected_dx)
      self.assertEqual(y.grad.item(), expected_dy)

    with self.subTest("test mul, dL/dout = -13.0"):
      # Arrange - clear grads on graph
      x.grad = None
      y.grad = None

      # Act - backward on d_mul again.
      output_mul.backward(
          torch.tensor(-13.0, device=self.device), retain_graph=True
      )

      # Assert - backward on d_mul again.
      expected_dx = 7.0 * -13.0
      expected_dy = 2.0 * -13.0
      self.assertEqual(x.grad.item(), expected_dx)
      self.assertEqual(y.grad.item(), expected_dy)

    with self.subTest("test add, dL/dout = 17.0"):
      # Arrange - clear grads on graph
      x.grad = None
      y.grad = None

      # Act - backward on d_add.
      output_add.backward(torch.tensor(17.0, device=self.device))

      # Assert - backward on d_add
      self.assertEqual(x.grad.item(), 17.0)
      self.assertEqual(y.grad.item(), 17.0)

    with self.subTest("no graph break"):
      # Arrange
      # @torch.compile(
      #     backend="tpu",
      #     fullgraph=True,
      # )
      def step(a, b):
        out_mul, out_add = mul_and_add_scalars(a, b)
        x = out_mul + out_add + a.cos()
        loss = -((x - 10.0) ** 2)
        loss.backward()
        return loss, a.grad, b.grad

      # Act
      print(utils.format_model(step, x, y, shlo=True))
      del step

      # Assert
      # Machine validation of the MLIR is beyond the scope of a unit test,
      # but we can manually inspect the above print to see that
      # the jax kernels are called, and per discussions with xla experts,
      # a call is ultimately inlined and flattened.
      # pylint: disable=line-too-long
      #
      # module @tt_jit_export_L175C0_dispatch_fx_node_neg {
      #   func.func private @"test::mul_and_add_scalars_backward_0x728519e39ec5d7ce"(%arg0: tensor<f32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<f32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<f32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<f32> {mhlo.sharding = "{replicated}"}) -> (tensor<f32>, tensor<f32>) {  # noqa: E501
      #     %0 = stablehlo.multiply %arg2, %arg1 : tensor<f32>
      #     %1 = stablehlo.add %arg3, %0 : tensor<f32>
      #     %2 = stablehlo.multiply %arg0, %arg2 : tensor<f32>
      #     %3 = stablehlo.add %arg3, %2 : tensor<f32>
      #     return %1, %3 : tensor<f32>, tensor<f32>
      #   }
      #   func.func private @"test::mul_and_add_scalars_0x5affe934a9008caa"(%arg0: tensor<f32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<f32> {mhlo.sharding = "{replicated}"}) -> (tensor<f32>, tensor<f32>) {  # noqa: E501
      #     %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      #     %1 = stablehlo.add %arg0, %arg1 : tensor<f32>
      #     return %0, %1 : tensor<f32>, tensor<f32>
      #   }
      #   func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {  # noqa: E501
      #     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      #     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      #     %cst_1 = stablehlo.constant dense<1.000000e+01> : tensor<f64>
      #     %0 = stablehlo.cosine %arg0 : tensor<f32>
      #     %1:2 = call @"test::mul_and_add_scalars_0x5affe934a9008caa"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)  # noqa: E501
      #     %2 = stablehlo.add %1#0, %1#1 : tensor<f32>
      #     %3 = stablehlo.add %2, %0 : tensor<f32>
      #     %4 = stablehlo.convert %3 : (tensor<f32>) -> tensor<f64>
      #     %cst_2 = stablehlo.constant dense<-1.000000e+00> : tensor<f64>
      #     %5 = stablehlo.multiply %cst_1, %cst_2 : tensor<f64>
      #     %6 = stablehlo.add %4, %5 : tensor<f64>
      #     %7 = stablehlo.convert %6 : (tensor<f64>) -> tensor<f32>
      #     %8 = stablehlo.power %7, %cst_0 : tensor<f32>
      #     %9 = stablehlo.convert %8 : (tensor<f32>) -> tensor<f64>
      #     %10 = stablehlo.multiply %9, %cst : tensor<f64>
      #     %11 = stablehlo.convert %10 : (tensor<f64>) -> tensor<f32>
      #     %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      #     %12 = stablehlo.negate %cst_3 : tensor<f32>
      #     %13 = stablehlo.multiply %12, %11 : tensor<f32>
      #     %14:2 = call @"test::mul_and_add_scalars_backward_0x728519e39ec5d7ce"(%arg0, %arg1, %13, %13) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)  # noqa: E501
      #     %15 = stablehlo.sine %arg0 : tensor<f32>
      #     %16 = stablehlo.negate %15 : tensor<f32>
      #     %17 = stablehlo.multiply %13, %16 : tensor<f32>
      #     %18 = stablehlo.add %17, %14#0 : tensor<f32>
      #     %cst_4 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      #     %19 = stablehlo.power %7, %cst_4 : tensor<f32>
      #     %20 = stablehlo.negate %19 : tensor<f32>
      #     return %20, %18, %14#1 : tensor<f32>, tensor<f32>, tensor<f32>
      #   }
      # }
      # pylint: enable=line-too-long

    with self.subTest("composable_with_compile"):
      # Arrange - setup compiled function
      @torch.compile(backend="tpu")
      def step2(x_local, y_local):
        mul_out, add_out = mul_and_add_scalars(x_local, y_local)
        # This means grad_mul_out = 2, grad_add_out = -3.
        mul_out.backward(
            torch.tensor(2.0, device=x_local.device), retain_graph=True
        )
        add_out.backward(torch.tensor(-3.0, device=x_local.device))
        return mul_out, add_out, x_local.grad, y_local.grad

      # Step 1: Arrange
      x.grad = None
      y.grad = None
      x = torch.tensor(4.0, requires_grad=True, device=self.device)
      y = torch.tensor(-5.0, requires_grad=True, device=self.device)

      # Step 1: Act
      mul_out, add_out, grad_x, grad_y = step2(x, y)

      # Step 1: Assert
      self.assertEqual(mul_out.item(), 4.0 * -5.0)
      self.assertEqual(add_out.item(), 4.0 + -5.0)
      # grad_x = grad_mul * y + grad_add (passes straight through)
      self.assertEqual(grad_x.item(), 2.0 * -5.0 - 3.0)
      # grad_y = grad_mul * x + grad_add (passes straight through)
      self.assertEqual(grad_y.item(), 2.0 * 4.0 - 3.0)

      # Step 2: Arrange
      x.grad = None
      y.grad = None
      x = torch.tensor(-4.0, requires_grad=True, device=self.device)
      y = torch.tensor(-37.0, requires_grad=True, device=self.device)

      # Step 2: Act
      mul_out, add_out, grad_x, grad_y = step2(x, y)

      # Step 2: Assert
      self.assertEqual(mul_out.item(), -4.0 * -37.0)
      self.assertEqual(add_out.item(), -4.0 + -37.0)
      # grad_x = grad_mul * y + grad_add (passes straight through)
      self.assertEqual(grad_x.item(), 2.0 * -37.0 - 3.0)
      # grad_y = grad_mul * x + grad_add (passes straight through)
      self.assertEqual(grad_y.item(), 2.0 * -4.0 - 3.0)

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
        RuntimeError, "custom emitter for test_custom_call not found"
    ):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
      y = custom_call_op(x)
      y.cpu()

  def test_wrapper_tensor_support(self):

    class MockWrapperTensor(torch.Tensor):

      @staticmethod
      def __new__(cls, elem, env):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.shape,
            dtype=torch.float32,
            device="meta",
        )

      def __init__(self, elem, env):
        super().__init__()
        self._elem = elem
        self._env = env

      @classmethod
      def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        func_name = func.name()
        if "::" in func_name:
          qualname = func_name
        else:
          qualname = f"{func.namespace}::{func_name}"

        if "." in qualname:
          qualname = qualname.split(".")[0]

        if qualname in torch_custom_ops.OPDEFS:
          custom_op_def = torch_custom_ops.OPDEFS[qualname]
          impl_fn = custom_op_def._init_fn
          return impl_fn(*args, **kwargs)

        return NotImplemented

    x_elem = jax.device_put(
        jax.numpy.array([1.0, 2.0, 3.0], dtype=jax.numpy.float32)
    )
    y_elem = jax.device_put(
        jax.numpy.array([4.0, 5.0, 6.0], dtype=jax.numpy.float32)
    )
    x = MockWrapperTensor(x_elem, "mock_env")
    y = MockWrapperTensor(y_elem, "mock_env")

    res = add_vectors(x, y)
    self.assertIsInstance(res, MockWrapperTensor)
    self.assertEqual(res._env, "mock_env")
    self.assertTrue(
        jax.numpy.allclose(
            res._elem,
            jax.numpy.array([5.0, 7.0, 9.0], dtype=jax.numpy.float32),
        )
    )

  def test_wrapper_tensor_support_kwargs(self):

    class MockWrapperTensor(torch.Tensor):

      @staticmethod
      def __new__(cls, elem, env):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.shape,
            dtype=torch.float32,
            device="meta",
        )

      def __init__(self, elem, env):
        super().__init__()
        self._elem = elem
        self._env = env

      @classmethod
      def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        func_name = func.name()
        if "::" in func_name:
          qualname = func_name
        else:
          qualname = f"{func.namespace}::{func_name}"

        if "." in qualname:
          qualname = qualname.split(".")[0]

        if qualname in torch_custom_ops.OPDEFS:
          custom_op_def = torch_custom_ops.OPDEFS[qualname]
          impl_fn = custom_op_def._init_fn
          return impl_fn(*args, **kwargs)

        return NotImplemented

    @pallas.jax_op("pallas::wrapper_tensor_kwargs")
    def kwarg_add(x: jax.Array, y: jax.Array) -> jax.Array:
      return x + y

    x_elem = jax.device_put(
        jax.numpy.array([1.0, 2.0, 3.0], dtype=jax.numpy.float32)
    )
    y_elem = jax.device_put(
        jax.numpy.array([4.0, 5.0, 6.0], dtype=jax.numpy.float32)
    )
    x = MockWrapperTensor(x_elem, "mock_env")
    y = MockWrapperTensor(y_elem, "mock_env")

    res = kwarg_add(x, y=y)
    self.assertIsInstance(res, MockWrapperTensor)
    self.assertEqual(res._env, "mock_env")
    self.assertTrue(
        jax.numpy.allclose(
            res._elem,
            jax.numpy.array([5.0, 7.0, 9.0], dtype=jax.numpy.float32),
        )
    )

  def test_wrapper_tensor_support_pytree(self):

    class MockWrapperTensor(torch.Tensor):

      @staticmethod
      def __new__(cls, elem, env):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.shape,
            dtype=torch.float32,
            device="meta",
        )

      def __init__(self, elem, env):
        super().__init__()
        self._elem = elem
        self._env = env

      @classmethod
      def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        func_name = func.name()
        if "::" in func_name:
          qualname = func_name
        else:
          qualname = f"{func.namespace}::{func_name}"

        if "." in qualname:
          qualname = qualname.split(".")[0]

        if qualname in torch_custom_ops.OPDEFS:
          custom_op_def = torch_custom_ops.OPDEFS[qualname]
          impl_fn = custom_op_def._init_fn
          return impl_fn(*args, **kwargs)

        return NotImplemented

    @pallas.jax_op("pallas::wrapper_tensor_pytree")
    def pytree_fn(x: jax.Array, y: jax.Array) -> list[jax.Array]:
      return [x + y, x - y]

    x_elem = jax.device_put(
        jax.numpy.array([1.0, 2.0, 3.0], dtype=jax.numpy.float32)
    )
    y_elem = jax.device_put(
        jax.numpy.array([4.0, 5.0, 6.0], dtype=jax.numpy.float32)
    )
    x = MockWrapperTensor(x_elem, "mock_env")
    y = MockWrapperTensor(y_elem, "mock_env")

    res = pytree_fn(x, y)
    self.assertIsInstance(res, list)
    self.assertIsInstance(res[0], MockWrapperTensor)
    self.assertEqual(res[0]._env, "mock_env")
    self.assertTrue(
        jax.numpy.allclose(
            res[0]._elem,
            jax.numpy.array([5.0, 7.0, 9.0], dtype=jax.numpy.float32),
        )
    )
    self.assertIsInstance(res[1], MockWrapperTensor)
    self.assertEqual(res[1]._env, "mock_env")
    self.assertTrue(
        jax.numpy.allclose(
            res[1]._elem,
            jax.numpy.array([-3.0, -3.0, -3.0], dtype=jax.numpy.float32),
        )
    )


class TestPallasCompat(absltest.TestCase):

  def test_warn_deprecation_with_skip(self):
    if sys.version_info >= (3, 12):
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _compat.warn_deprecation_with_skip(
            "test message",
            pathlib.Path(_compat.__file__).parent,
        )
        self.assertLen(w, 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), "test message")
        self.assertIn("pallas_test.py", w[0].filename)
    else:

      def helper():
        _compat.warn_deprecation_with_skip(
            "test message",
            pathlib.Path(_compat.__file__).parent,
        )

      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        helper()
        self.assertLen(w, 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), "test message")
        self.assertIn("pallas_test.py", w[0].filename)


if __name__ == "__main__":
  absltest.main()
