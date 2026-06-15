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
import operator
import pathlib
import sys
from typing import Optional, Tuple, TypeAlias
import warnings

from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
import jax.export
import torch
from torch_tpu._internal import compile  # pylint: disable=redefined-builtin
from torch_tpu._internal import execution_mode
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile.fx_passes import inplace_donate_ssa_rename
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

  # --------------------------------------------------------------------------
  # inplace_donate=True — contract tests.
  #
  # See _internal/pallas/pallas.py JaxCallable.__init__ for the design
  # discussion. Behavior: each donated input tensor's storage is rebound
  # IN PLACE to the corresponding output buffer at the C++ binding layer
  # (pallas_py.cc::PyCallCustomKernel). The output returned by the op is
  # a fresh at::Tensor that wraps the same DeviceBufferRef as the rebound
  # input — different `c10::StorageImpl`s (so torch.library's
  # output-may-not-alias-input check passes), backed by the same buffer
  # (so reads of either tensor observe the kernel's writes). Callers can
  # then drop the canonical `input.copy_(output)` republish step.
  #
  # Tests below pin down the contract:
  #   * the kwarg is accepted (plumbing can't silently regress)
  #   * the math is still correct
  #   * out and x are distinct tensors that share underlying data
  #   * subsequent reads of x see the kernel's writes without a republish
  #   * a single tensor can be threaded through N sequential donating
  #     calls with no user-side republish in between
  # --------------------------------------------------------------------------

  def test_inplace_donate_kwarg_accepted(self):
    """All three plumbing layers (jax_op, custom_jax_kernel, JaxCallable)
    accept `inplace_donate=True` without raising. Guards against a
    silent regression where one of the overloads drops the kwarg.
    """
    op = pallas.jax_op(
        "pallas::inplace_donate_kwarg_accepted",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )
    self.assertIsNotNone(op)

  def test_inplace_donate_math_correct(self):
    """With `inplace_donate=True`, the kernel still produces the correct
    numerical result. The rebind is a host-side identity optimization,
    not a math change."""
    op = pallas.jax_op(
        "pallas::inplace_donate_math_correct",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )
    x = torch.tensor([-2.0, 0.0, 5.0], dtype=torch.float32, device=self.device)
    y = torch.tensor([3.0, 7.0, -1.0], dtype=torch.float32, device=self.device)
    expected = torch.tensor([1.0, 7.0, 4.0], dtype=torch.float32)
    out = op(x, y)
    utils.assert_close(out.to("cpu"), expected)

  def test_inplace_donate_output_and_input_share_data(self):
    """With `inplace_donate=True`, the donated input and the donated
    output back the same underlying buffer (mutating one is observable
    through the other), but they are distinct `at::Tensor` objects with
    distinct `c10::StorageImpl`s — otherwise torch.library's
    `_c_check_aliasing_constraint` would reject the call."""
    op = pallas.jax_op(
        "pallas::inplace_donate_share_data",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    out = op(x, y)
    # `out is x` would alias an input — rejected. They must be distinct.
    self.assertIsNot(out, x)
    # But the data must match — both read from the kernel's output buffer.
    utils.assert_close(out.to("cpu"), x.to("cpu"))

  def test_inplace_donate_reads_observe_writes_without_republish(self):
    """The motivating contract: after a donating kernel call with
    `inplace_donate=True`, subsequent reads of the donated input tensor
    observe the kernel's writes WITHOUT the caller having to
    `input.copy_(output)`. This is the per-layer republish step that the
    block-major KV-cache wrapper drops once this flag is on."""
    op = pallas.jax_op(
        "pallas::inplace_donate_reads_observe_writes",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    # Discard the return value — we read through `x` instead, which is
    # only safe when the rebind actually happened.
    _ = op(x, y)
    expected = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32)
    utils.assert_close(x.to("cpu"), expected)

  def test_inplace_donate_chained_calls_thread_same_tensor(self):
    """Threads a single tensor through several sequential donating calls
    without any republish step in user code. Models the per-layer
    attention loop where one KV-cache bundle is donated to N layers in
    sequence and each layer's read of the bundle must see the prior
    layer's writes."""
    op = pallas.jax_op(
        "pallas::inplace_donate_chained",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )
    x = torch.zeros(4, dtype=torch.float32, device=self.device)
    y = torch.ones(4, dtype=torch.float32, device=self.device)
    for _ in range(5):
      _ = op(x, y)
      # No `x.copy_(out)` — x's storage was rebound to the kernel's
      # output buffer in C++, so the next iteration reads the update.
    expected = torch.full((4,), 5.0, dtype=torch.float32)
    utils.assert_close(x.to("cpu"), expected)

  def test_inplace_donate_chained_calls_thread_same_tensor_compiled_mode(self):
    """Compile-mode counterpart of the chained donation test.

    Under ``torch.compile`` the sequential donating calls collapse into a
    single PJRT executable. The eager-mode host-side rebind in
    ``PyCallCustomKernel`` runs at most once during tracing, so without
    the ``inplace_donate_ssa_rename`` FX pass every call would donate the
    same SSA value and XLA would reject the program with
    ``"Donation requested for invalid buffer"``. The pass threads each
    call's output forward, giving every donation a distinct SSA value
    and producing the same chained-sum result as eager mode."""
    op = pallas.jax_op(
        "pallas::inplace_donate_chained_compiled",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def chained_sums(bundle, y0, y1, y2, y3, y4):
      # Each call donates `bundle` — without the SSA rewrite this is the
      # bug pattern that breaks vLLM's block-major KV cache under
      # torch.compile.
      s0 = op(bundle, y0).sum()
      s1 = op(bundle, y1).sum()
      s2 = op(bundle, y2).sum()
      s3 = op(bundle, y3).sum()
      s4 = op(bundle, y4).sum()
      return torch.stack([s0, s1, s2, s3, s4])

    bundle = torch.zeros(4, dtype=torch.float32, device=self.device)
    y0 = torch.ones(4, dtype=torch.float32, device=self.device)
    y1 = torch.ones(4, dtype=torch.float32, device=self.device)
    y2 = torch.ones(4, dtype=torch.float32, device=self.device)
    y3 = torch.ones(4, dtype=torch.float32, device=self.device)
    y4 = torch.ones(4, dtype=torch.float32, device=self.device)
    sums = chained_sums(bundle, y0, y1, y2, y3, y4)
    # Iteration i produces bundle_i = bundle + i * 1 = i+1, sum = 4*(i+1).
    expected = torch.tensor([4.0, 8.0, 12.0, 16.0, 20.0], dtype=torch.float32)
    utils.assert_close(sums.to("cpu"), expected)

  def test_declare_jax_op_raises_before_bind(self):
    """Eager invocation of a declared-but-unbound op must raise a clear
    error so that callers that fire too early are caught immediately
    rather than silently producing wrong results."""

    def fake_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      return torch.empty_like(x)

    op = pallas.declare_jax_op(
        "pallas::declared_unbound",
        fake_fn,
        donate_argnums=(0,),
        inplace_donate=True,
    )
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    with self.assertRaisesRegex(RuntimeError, "has not yet been bound"):
      op(x, y)

  def test_declare_then_bind_then_call_matches_jax_op(self):
    """Eager call through a declared-then-bound op must match the result
    of the same op registered via the one-shot `jax_op` helper."""

    def fake_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      return torch.empty_like(x)

    declared = pallas.declare_jax_op(
        "pallas::declared_then_bound",
        fake_fn,
    )
    pallas.bind_jax_kernel(declared, add_vectors_jax)

    one_shot = pallas.jax_op("pallas::one_shot_for_compare", add_vectors_jax)

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    utils.assert_close(declared(x, y).to("cpu"), one_shot(x, y).to("cpu"))

  def test_bind_jax_kernel_rejects_double_bind(self):
    """A declared op can only be bound once — re-binding would silently
    change the kernel out from under any FX graph that was traced
    against the first binding."""

    def fake_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      return torch.empty_like(x)

    declared = pallas.declare_jax_op(
        "pallas::declared_double_bind",
        fake_fn,
    )
    pallas.bind_jax_kernel(declared, add_vectors_jax)
    with self.assertRaisesRegex(ValueError, "already has a bound kernel"):
      pallas.bind_jax_kernel(declared, add_vectors_jax)

  def test_bind_jax_kernel_rejects_non_declared_op(self):
    """`bind_jax_kernel` only works on ops produced by `declare_jax_op`;
    a one-shot `jax_op` registration already has its kernel and there is
    no lazy slot to fill."""
    one_shot = pallas.jax_op("pallas::one_shot_not_bindable", add_vectors_jax)
    with self.assertRaisesRegex(ValueError, "not produced by"):
      pallas.bind_jax_kernel(one_shot, add_vectors_jax)

  def test_declare_jax_op_inplace_donate_chained_calls_compiled_mode(self):
    """End-to-end vLLM-shaped flow: declare the inplace-donate op at
    "init" time (before any compile), bind the JAX kernel later (the
    sequence vLLM's setup_bundled mimics), then chain donating calls
    inside `@torch.compile(backend=TpuBackend())`. The
    `inplace_donate_ssa_rename` FX pass must pick up the declared op via
    the `_pallas_inplace_donate_argnums` annotation set at declare time
    and thread the donation chain so XLA accepts the program."""

    def fake_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      return torch.empty_like(x)

    op = pallas.declare_jax_op(
        "pallas::declared_chained_compiled",
        fake_fn,
        donate_argnums=(0,),
        inplace_donate=True,
    )
    # Bind the actual JAX kernel here (vLLM's setup_bundled equivalent).
    pallas.bind_jax_kernel(op, add_vectors_jax)

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def chained_sums(bundle, y0, y1, y2, y3, y4):
      s0 = op(bundle, y0).sum()
      s1 = op(bundle, y1).sum()
      s2 = op(bundle, y2).sum()
      s3 = op(bundle, y3).sum()
      s4 = op(bundle, y4).sum()
      return torch.stack([s0, s1, s2, s3, s4])

    bundle = torch.zeros(4, dtype=torch.float32, device=self.device)
    y0 = torch.ones(4, dtype=torch.float32, device=self.device)
    y1 = torch.ones(4, dtype=torch.float32, device=self.device)
    y2 = torch.ones(4, dtype=torch.float32, device=self.device)
    y3 = torch.ones(4, dtype=torch.float32, device=self.device)
    y4 = torch.ones(4, dtype=torch.float32, device=self.device)
    sums = chained_sums(bundle, y0, y1, y2, y3, y4)
    expected = torch.tensor([4.0, 8.0, 12.0, 16.0, 20.0], dtype=torch.float32)
    utils.assert_close(sums.to("cpu"), expected)

  # ----------------------------------------------------------------------
  # Executable-level inplace_donate rebind tests.
  #
  # When `inplace_donate=True` chains run inside `torch.compile`, the
  # `inplace_donate_ssa_rename` FX pass threads the donation chain in SSA
  # form. The compile backend then identifies each chain's terminus (final
  # SSA value) via `find_donate_aliases`, plumbs it as an extra executable
  # output, and after each execution rebinds the donated input tensor's
  # storage to that output's device buffer. The net effect for users is
  # that they can drop the explicit `bundle.copy_(new_bundle)` republish
  # from per-layer wrappers (the pattern that otherwise drives a full
  # bundle memcpy per layer and dominates runtime for large KV caches)
  # while still observing the chain's writes on the next call.
  # ----------------------------------------------------------------------

  def test_inplace_donate_compiled_mode_no_copy_single_call_persists(self):
    """Single donating call inside a compiled function. Without any user
    side `bundle.copy_(new_bundle)`, the bundle tensor must still observe
    the kernel's write on a subsequent compiled-mode call — the
    executable-level rebind points the bundle's storage at the kernel's
    output buffer after each execution."""
    op = pallas.jax_op(
        "pallas::inplace_donate_no_copy_single",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def step(bundle, delta):
      return op(bundle, delta).sum()

    bundle = torch.zeros(4, dtype=torch.float32, device=self.device)
    delta = torch.ones(4, dtype=torch.float32, device=self.device)

    s0 = step(bundle, delta)
    utils.assert_close(
        bundle.to("cpu"), torch.ones(4, dtype=torch.float32)
    )
    utils.assert_close(s0.to("cpu"), torch.tensor(4.0, dtype=torch.float32))

    s1 = step(bundle, delta)
    utils.assert_close(
        bundle.to("cpu"), torch.full((4,), 2.0, dtype=torch.float32)
    )
    utils.assert_close(s1.to("cpu"), torch.tensor(8.0, dtype=torch.float32))

    s2 = step(bundle, delta)
    utils.assert_close(
        bundle.to("cpu"), torch.full((4,), 3.0, dtype=torch.float32)
    )
    utils.assert_close(s2.to("cpu"), torch.tensor(12.0, dtype=torch.float32))

  def test_inplace_donate_compiled_mode_no_copy_chained_persists(self):
    """Several sequential donating calls inside a single compiled function
    — the realistic vLLM-shaped chain. The SSA rename threads layer i's
    output into layer i+1's input; the executable-level rebind then
    threads the *chain's* terminus into the bundle tensor between
    compiled-mode invocations. No `bundle.copy_(...)` anywhere."""
    op = pallas.jax_op(
        "pallas::inplace_donate_no_copy_chained",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def step(bundle, ya, yb, yc):
      s0 = op(bundle, ya).sum()
      s1 = op(bundle, yb).sum()
      s2 = op(bundle, yc).sum()
      return torch.stack([s0, s1, s2])

    bundle = torch.zeros(4, dtype=torch.float32, device=self.device)
    ya = torch.ones(4, dtype=torch.float32, device=self.device)
    yb = torch.ones(4, dtype=torch.float32, device=self.device)
    yc = torch.ones(4, dtype=torch.float32, device=self.device)

    sums_run0 = step(bundle, ya, yb, yc)
    utils.assert_close(
        sums_run0.to("cpu"),
        torch.tensor([4.0, 8.0, 12.0], dtype=torch.float32),
    )
    # After one compiled-mode call: bundle += 3 → all 3s.
    utils.assert_close(
        bundle.to("cpu"), torch.full((4,), 3.0, dtype=torch.float32)
    )

    sums_run1 = step(bundle, ya, yb, yc)
    # Layer sums for run 1: prev_bundle was [3,3,3,3], ya=yb=yc=ones.
    # s0 = (3+1)*4 = 16, s1 = (4+1)*4 = 20, s2 = (5+1)*4 = 24.
    utils.assert_close(
        sums_run1.to("cpu"),
        torch.tensor([16.0, 20.0, 24.0], dtype=torch.float32),
    )
    utils.assert_close(
        bundle.to("cpu"), torch.full((4,), 6.0, dtype=torch.float32)
    )

  def test_inplace_donate_compiled_mode_no_copy_multi_output_op(self):
    """Multi-output donating kernel — the vLLM bundled-attention shape.
    The kernel returns ``(new_bundle, other_output)``; the FX pass
    threads ``getitem(call, 0)`` through the chain, and the executable
    rebinds the bundle tensor's storage to the kernel's bundle output."""

    def bundle_update_kernel(x_ref, y_ref, o_new_x_ref, o_combined_ref):
      x, y = x_ref[...], y_ref[...]
      o_new_x_ref[...] = x + y
      o_combined_ref[...] = x * 2 + y

    def bundle_update_jax(
        x: jax.Array, y: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
      dt = jax.ShapeDtypeStruct(x.shape, x.dtype)
      return pl.pallas_call(bundle_update_kernel, out_shape=(dt, dt))(x, y)

    op = pallas.jax_op(
        "pallas::inplace_donate_no_copy_multi",
        bundle_update_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def step(bundle, ya, yb):
      _, oa = op(bundle, ya)
      _, ob = op(bundle, yb)
      return oa.sum() + ob.sum()

    bundle = torch.zeros(4, dtype=torch.float32, device=self.device)
    ya = torch.ones(4, dtype=torch.float32, device=self.device)
    yb = torch.full((4,), 2.0, dtype=torch.float32, device=self.device)

    result0 = step(bundle, ya, yb)
    # Call 0: bundle=0, ya=1. new_bundle=1, combined=0+1=1 → oa.sum()=4.
    # Call 1: bundle=1, yb=2. new_bundle=3, combined=2+2=4 → ob.sum()=16.
    utils.assert_close(
        result0.to("cpu"), torch.tensor(20.0, dtype=torch.float32)
    )
    # Bundle ends at new_bundle from call 1 = 1 + 2 = 3.
    utils.assert_close(
        bundle.to("cpu"), torch.full((4,), 3.0, dtype=torch.float32)
    )

    result1 = step(bundle, ya, yb)
    # Call 0: bundle=3, ya=1. new_bundle=4, combined=6+1=7 → oa.sum()=28.
    # Call 1: bundle=4, yb=2. new_bundle=6, combined=8+2=10 → ob.sum()=40.
    utils.assert_close(
        result1.to("cpu"), torch.tensor(68.0, dtype=torch.float32)
    )
    utils.assert_close(
        bundle.to("cpu"), torch.full((4,), 6.0, dtype=torch.float32)
    )

  def test_inplace_donate_compiled_mode_no_copy_declare_bind_variant(self):
    """End-to-end vLLM-shaped sequence: declare the donating op at init
    time, bind the kernel later (the `setup_bundled` analog), then chain
    donating calls under torch.compile without a user-side
    ``bundle.copy_``. Executable-level rebind keeps the bundle live and
    threads its content across compiled-mode invocations. Each call's
    output is consumed downstream (mirroring how every attention layer
    feeds the next layer in real workloads) so aot_autograd does not
    DCE the chain before the SSA-rename pass runs."""

    def fake_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      return torch.empty_like(x)

    op = pallas.declare_jax_op(
        "pallas::inplace_donate_no_copy_declared",
        fake_fn,
        donate_argnums=(0,),
        inplace_donate=True,
    )
    pallas.bind_jax_kernel(op, add_vectors_jax)

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def step(bundle, ya, yb):
      # Sum is computed before the next donating call so each ``op(...).sum()``
      # observes the kernel's output at that point — matching the realistic
      # "per-layer attention output is consumed before the next layer runs"
      # shape. Holding the kernel's output in a Python local that is summed
      # only after the chain has run would expose the eager-style storage
      # rebind that ``inplace_donate=True`` faithfully preserves under
      # ``torch.compile`` (subsequent reads of the held tensor observe the
      # latest call's writes); that aliasing is correct but not what vLLM's
      # per-layer wrapper exhibits.
      s_a = op(bundle, ya).sum()
      s_b = op(bundle, yb).sum()
      return s_a + s_b

    bundle = torch.zeros(4, dtype=torch.float32, device=self.device)
    ya = torch.ones(4, dtype=torch.float32, device=self.device)
    yb = torch.ones(4, dtype=torch.float32, device=self.device)

    s_run0 = step(bundle, ya, yb)
    # Run 0: call 0 produces [1,1,1,1] (sum=4); call 1 reads call 0's output
    # via the SSA chain → [2,2,2,2] (sum=8). Total: 12.
    utils.assert_close(
        s_run0.to("cpu"), torch.tensor(12.0, dtype=torch.float32)
    )
    utils.assert_close(
        bundle.to("cpu"), torch.full((4,), 2.0, dtype=torch.float32)
    )

    s_run1 = step(bundle, ya, yb)
    # Run 1: bundle starts at [2,2,2,2].
    # call 0: +[1,1,1,1] → [3,3,3,3] sum=12.
    # call 1: chained input [3,3,3,3]+[1,1,1,1] → [4,4,4,4] sum=16. Total=28.
    utils.assert_close(
        s_run1.to("cpu"), torch.tensor(28.0, dtype=torch.float32)
    )
    utils.assert_close(
        bundle.to("cpu"), torch.full((4,), 4.0, dtype=torch.float32)
    )

  def test_inplace_donate_compiled_mode_no_copy_two_independent_chains(self):
    """Two distinct donated inputs in the same compiled function. Each
    chain is rebound to its own input tensor; their values must update
    independently across compiled-mode calls."""
    op_a = pallas.jax_op(
        "pallas::inplace_donate_no_copy_chain_a",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )
    op_b = pallas.jax_op(
        "pallas::inplace_donate_no_copy_chain_b",
        add_vectors_jax,
        donate_argnums=(0,),
        inplace_donate=True,
    )

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def step(bundle_a, bundle_b, ya, yb):
      return op_a(bundle_a, ya).sum() + op_b(bundle_b, yb).sum()

    bundle_a = torch.zeros(4, dtype=torch.float32, device=self.device)
    bundle_b = torch.zeros(4, dtype=torch.float32, device=self.device)
    ya = torch.ones(4, dtype=torch.float32, device=self.device)
    yb = torch.full((4,), 2.0, dtype=torch.float32, device=self.device)

    out0 = step(bundle_a, bundle_b, ya, yb)
    # bundle_a → ones, bundle_b → twos.
    utils.assert_close(out0.to("cpu"), torch.tensor(12.0, dtype=torch.float32))
    utils.assert_close(
        bundle_a.to("cpu"), torch.ones(4, dtype=torch.float32)
    )
    utils.assert_close(
        bundle_b.to("cpu"), torch.full((4,), 2.0, dtype=torch.float32)
    )

    out1 = step(bundle_a, bundle_b, ya, yb)
    # bundle_a now reads ones; +ones → twos; sum=8.
    # bundle_b now reads twos; +twos → fours; sum=16.
    utils.assert_close(out1.to("cpu"), torch.tensor(24.0, dtype=torch.float32))
    utils.assert_close(
        bundle_a.to("cpu"), torch.full((4,), 2.0, dtype=torch.float32)
    )
    utils.assert_close(
        bundle_b.to("cpu"), torch.full((4,), 4.0, dtype=torch.float32)
    )


class TestInplaceDonateFxPass(absltest.TestCase):
  """Pure-Python unit tests for the inplace_donate_ssa_rename FX pass and
  its companion ``find_donate_aliases`` helper.

  These tests build FX graphs by hand against synthetic call_function
  targets carrying the ``_pallas_inplace_donate_argnums`` attribute that
  pallas stamps in production. They do not touch the TPU runtime — only
  the FX-level graph manipulation. That keeps the chain-detection logic
  exercised independently of JAX/MLIR availability, which is the layer
  most likely to silently break as the graph shape evolves.
  """

  def _make_mock_op(
      self,
      name: str,
      donate_argnums: Tuple[int, ...] | None = None,
      *,
      num_outputs: int = 1,
  ):
    """Returns a callable that mimics a pallas op for FX graph testing.

    The callable carries the ``_pallas_inplace_donate_argnums`` attribute
    that the FX pass keys on, and has a stable ``__name__`` so error
    messages and ``node.name`` are readable. When ``num_outputs > 1`` the
    callable returns a tuple; otherwise a single tensor.
    """

    def _op(*args, **kwargs):
      if num_outputs == 1:
        return args[0].clone()
      return tuple(args[0].clone() for _ in range(num_outputs))

    _op.__name__ = name
    if donate_argnums is not None:
      setattr(
          _op,
          inplace_donate_ssa_rename._INPLACE_DONATE_ARGNUMS_ATTR,
          tuple(donate_argnums),
      )
    return _op

  def _build_single_output_chain_graph(self, num_calls: int) -> torch.fx.Graph:
    """Builds an FX graph: bundle → op → op → ... → op (single-output)."""
    op = self._make_mock_op(
        "chained_single", donate_argnums=(0,), num_outputs=1
    )
    graph = torch.fx.Graph()
    bundle = graph.placeholder("bundle")
    delta = graph.placeholder("delta")
    current = bundle
    for _ in range(num_calls):
      current = graph.call_function(op, (current, delta))
    graph.output((current,))
    return graph

  def _build_multi_output_chain_graph(
      self, num_calls: int
  ) -> torch.fx.Graph:
    """Builds an FX graph: bundle → op → op → ... → op (multi-output).

    The FX pass requires ``meta['val']`` to be a tuple/list for it to
    treat a call as multi-output; the helper attaches a sentinel pair to
    each call's ``meta['val']`` so the multi-output code path is
    exercised faithfully.
    """
    op = self._make_mock_op(
        "chained_multi", donate_argnums=(0,), num_outputs=2
    )
    graph = torch.fx.Graph()
    bundle = graph.placeholder("bundle")
    delta = graph.placeholder("delta")
    current = bundle
    output_collector = []
    for _ in range(num_calls):
      call = graph.call_function(op, (current, delta))
      # Mark as multi-output so the pass routes through `getitem` insert.
      call.meta["val"] = (object(), object())
      other = graph.call_function(operator.getitem, (call, 1))
      output_collector.append(other)
      current = call  # bundle slot is updated via the rewrite below
    # Pre-existing user: feed the last call's bundle into output so the
    # chain isn't dead at the FX level. find_donate_aliases doesn't
    # require this, but it matches the realistic shape where attn output
    # consumers keep the call alive.
    graph.output(tuple(output_collector))
    return graph

  def test_find_donate_aliases_no_donation_returns_empty(self):
    """Graphs with no donation attribute on any call node produce no
    aliases."""
    op = self._make_mock_op("plain", donate_argnums=None)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    graph.output((graph.call_function(op, (x, y)),))
    self.assertEqual(inplace_donate_ssa_rename.find_donate_aliases(graph), [])

  def test_find_donate_aliases_placeholder_with_no_users_returns_empty(self):
    """A placeholder that never feeds any call cannot produce a donation
    alias even if other donations exist in the graph."""
    op = self._make_mock_op("donating", donate_argnums=(0,), num_outputs=1)
    graph = torch.fx.Graph()
    bundle_unused = graph.placeholder("bundle_unused")
    del bundle_unused  # intentionally unused
    other = graph.placeholder("other")
    delta = graph.placeholder("delta")
    graph.output((graph.call_function(op, (other, delta)),))
    aliases = inplace_donate_ssa_rename.find_donate_aliases(graph)
    # `other` is donated and used; `bundle_unused` is not.
    self.assertEqual(len(aliases), 1)
    arg_idx, terminus = aliases[0]
    # `bundle_unused` is placeholder 0 → `other` is placeholder 1.
    self.assertEqual(arg_idx, 1)
    self.assertEqual(terminus.op, "call_function")
    self.assertIs(terminus.target, op)

  def test_find_donate_aliases_single_output_call_terminus_is_call_node(
      self,
  ):
    """For a single-output donating op, the call node itself is the
    chain terminus."""
    graph = self._build_single_output_chain_graph(num_calls=1)
    aliases = inplace_donate_ssa_rename.find_donate_aliases(graph)
    self.assertEqual(len(aliases), 1)
    arg_idx, terminus = aliases[0]
    self.assertEqual(arg_idx, 0)
    self.assertEqual(terminus.op, "call_function")

  def test_find_donate_aliases_single_output_chain_terminus_is_last_call(
      self,
  ):
    """Chained single-output donations produce the last call as the
    terminus, after :func:`apply` has rewritten the chain."""
    graph = self._build_single_output_chain_graph(num_calls=5)
    inplace_donate_ssa_rename.apply(graph)
    aliases = inplace_donate_ssa_rename.find_donate_aliases(graph)
    self.assertEqual(len(aliases), 1)
    arg_idx, terminus = aliases[0]
    self.assertEqual(arg_idx, 0)
    call_nodes = [n for n in graph.nodes if n.op == "call_function"]
    # The last call should be the terminus.
    self.assertIs(terminus, call_nodes[-1])

  def test_find_donate_aliases_multi_output_terminus_is_getitem_zero(self):
    """For a multi-output donating op, the chain terminus is the
    ``getitem(call, 0)`` node that ``apply`` inserts."""
    graph = self._build_multi_output_chain_graph(num_calls=3)
    inplace_donate_ssa_rename.apply(graph)
    aliases = inplace_donate_ssa_rename.find_donate_aliases(graph)
    self.assertEqual(len(aliases), 1)
    arg_idx, terminus = aliases[0]
    self.assertEqual(arg_idx, 0)
    self.assertEqual(terminus.op, "call_function")
    self.assertIs(terminus.target, operator.getitem)
    self.assertEqual(terminus.args[1], 0)

  def test_find_donate_aliases_dedup_skips_converging_placeholders(self):
    """Two placeholders converging on the same terminus produce only one
    alias — the helper defensively skips later duplicates to avoid
    rebinding the same buffer twice."""
    op = self._make_mock_op("donating", donate_argnums=(0,), num_outputs=1)
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    _b = graph.placeholder("b")  # placeholder that feeds nothing donating
    call = graph.call_function(op, (a, _b))
    graph.output((call,))
    aliases = inplace_donate_ssa_rename.find_donate_aliases(graph)
    # Only `a` triggers a donation; `_b` is just a regular tensor input.
    self.assertEqual(len(aliases), 1)
    self.assertEqual(aliases[0][0], 0)

  def test_find_donate_aliases_independent_chains_both_detected(self):
    """Two distinct donated placeholders that drive disjoint donation
    chains each contribute one alias."""
    op_a = self._make_mock_op("op_a", donate_argnums=(0,), num_outputs=1)
    op_b = self._make_mock_op("op_b", donate_argnums=(0,), num_outputs=1)
    graph = torch.fx.Graph()
    bundle_a = graph.placeholder("bundle_a")
    bundle_b = graph.placeholder("bundle_b")
    delta = graph.placeholder("delta")
    a1 = graph.call_function(op_a, (bundle_a, delta))
    a2 = graph.call_function(op_a, (a1, delta))
    b1 = graph.call_function(op_b, (bundle_b, delta))
    graph.output((a2, b1))
    aliases = inplace_donate_ssa_rename.find_donate_aliases(graph)
    arg_indices = sorted(arg for arg, _ in aliases)
    self.assertEqual(arg_indices, [0, 1])

  def test_find_donate_aliases_op_without_attribute_breaks_chain(self):
    """A call to an op that does *not* carry the donate-argnums attribute
    is not part of the chain, so a chain that crosses it terminates at
    the predecessor's output."""
    donating_op = self._make_mock_op(
        "donating", donate_argnums=(0,), num_outputs=1
    )
    plain_op = self._make_mock_op("plain", donate_argnums=None)
    graph = torch.fx.Graph()
    bundle = graph.placeholder("bundle")
    delta = graph.placeholder("delta")
    call0 = graph.call_function(donating_op, (bundle, delta))
    # `plain_op` does not carry the donate-argnums attribute — chain ends
    # at call0 even though plain_op uses call0's value.
    consumer = graph.call_function(plain_op, (call0, delta))
    graph.output((consumer,))
    aliases = inplace_donate_ssa_rename.find_donate_aliases(graph)
    self.assertEqual(len(aliases), 1)
    self.assertIs(aliases[0][1], call0)

  def test_find_donate_aliases_returns_in_placeholder_order(self):
    """When multiple placeholders both produce aliases, the helper
    returns them in the order their placeholders appear in the graph."""
    op_a = self._make_mock_op("a", donate_argnums=(0,), num_outputs=1)
    op_b = self._make_mock_op("b", donate_argnums=(0,), num_outputs=1)
    graph = torch.fx.Graph()
    b0 = graph.placeholder("b0")  # idx 0
    b1 = graph.placeholder("b1")  # idx 1
    b2 = graph.placeholder("b2")  # idx 2
    delta = graph.placeholder("delta")  # idx 3
    # b0 and b2 are donated, b1 is plain.
    c0 = graph.call_function(op_a, (b0, delta))
    c2 = graph.call_function(op_b, (b2, b1))
    graph.output((c0, c2))
    aliases = inplace_donate_ssa_rename.find_donate_aliases(graph)
    self.assertEqual([a for a, _ in aliases], [0, 2])


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
