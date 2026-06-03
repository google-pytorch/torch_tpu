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

"""Tests error handling on TPU vs on CPU."""

import re
from typing import Any
import unittest
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import env
from tests import error_testing as et

_TEST_MODE = et.TEST_MODE

# Regex used by: TpuVsCpuErrorTest.test_index_no_indices
# Matches an arbitrary file path.
_INDEX_INTERNAL_ASSERTION_ERROR_RE = re.compile(
    r"ntensor >= 3 INTERNAL ASSERT FAILED at .*, please report a bug to"
    r" PyTorch.*"
)


def setUpModule():
  """Called by absltest after flags are parsed and before tests are run."""
  et.set_up_module()


def _get_aminmax_outputs(
    op: Any, device: str, dtype: torch.dtype
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
  """Creates the output tensors for `op` of `dtype`, on `device`.

  Args:
    op: a torch function. Should be one of: amax, amin, or aminmax.
    device: the device of the output tensors
    dtype: the dtype of the output tensors

  Returns:
    The out object to be used as a kwarg when calling `op`.
  """

  assert op in (torch.amax, torch.amin, torch.aminmax)

  def scalar_tensor() -> torch.Tensor:
    return torch.tensor(0, device=device, dtype=dtype)

  return (
      scalar_tensor()
      if op != torch.aminmax
      else (scalar_tensor(), scalar_tensor())
  )


def _get_convolution_default_args(is_backward: bool = False) -> dict[str, Any]:
  """Returns the default arguments for the convolution ops.

  Default arguments for both ops:
    - `torch.convolution`
    - `torch.ops.aten.convolution_backward`

  IMPORTANT: keep the arguments ordered. Even though we can pass this dictionary
  as keyword arguments to `torch.convolution`, we can't do the same for
  `torch.ops.aten.convolution_backward`, since the latter only accepts
  positional arguments.

  This is used so that we don't have to keep specifying all default arguments
  when calling both ops.

  Args:
    is_backward: flags whether the returned parameter-value dictionary should
      correspond to the forward or backward op.

  Returns:
    A dictionary that associates parameters with default values for either the
    forward or the backward op.
  """

  default = {
      "bias": None,
      "bias_sizes": None,
      "stride": (1,),
      "padding": (0,),
      "dilation": (1,),
      "transposed": False,
      "output_padding": (0,),
      "groups": 1,
      "output_mask": (True, True, True),
  }

  # Remove forward/backward-only keyword arguments.
  if is_backward:
    default.pop("bias")
  else:
    default.pop("bias_sizes")
    default.pop("output_mask")

  return default


def _run_convolution(*args, **kwargs):
  """Runs torch.convolution with the given arguments.

  Convenient function for running `torch.convolution`, without having to specify
  all arguments.

  Reason: even though ops like `torch.conv2d` (and other more popular ops) has
  default arguments, `torch.convolution` doesn't.

  Args:
    *args: Positional arguments for `torch.convolution`.
    **kwargs: Keyword arguments to override

  Returns:
    The output of `torch.convolution`.
  """

  merged = _get_convolution_default_args()
  # Update the copied default args dictionary with the kwargs passed as
  # argument.
  merged.update(kwargs)
  # Concatenate the values to the given positional arguments.
  args = args + tuple(merged.values())

  return torch.convolution(*args)


def _run_convolution_backward(*args, **kwargs):
  """Runs `torch.ops.aten.convolution_backward` with the given arguments.

  Convenient function for running `torch.ops.aten.convolution_backward`, without
  having to specify all arguments.

  Reason: this function doesn't have default arguments, and only accepts
  positional arguments.

  Args:
    *args: Positional arguments for `torch.convolution`.
    **kwargs: Keyword arguments to override

  Returns:
    The output of `torch.ops.aten.convolution_backward`.
  """

  merged = _get_convolution_default_args(is_backward=True)
  # Update the copied default args dictionary with the kwargs passed as
  # argument.
  merged.update(kwargs)
  # Concatenate the values to the given positional arguments.
  args = args + tuple(merged.values())

  return torch.ops.aten.convolution_backward(*args)


def _parameterize_convolution_fwd_bwd(
    forward: dict[str, Any] | None = None,
    backward: dict[str, Any] | None = None,
):
  """Parameterizes convolution tests, running the forward and backward ops.

  Convenient test decorator, for parameterizing a convolution test into 2
  variants: forward (`torch.convolution`) and backward tests
  (`torch.ops.aten.convolution_backward`). It's equivalent to:

  ```py
  @parameterized.named_parameters(
      { "testcase_name": "forward", "convolution": ..., **forward },
      { "testcase_name": "forward", "convolution": ..., **backward },
  )
  ```

  In summary, this test decorator is a wrapper for calling
  `parameterized.named_parameters()` with 2 dictionaries (forward and backward).
  In each of them, it will automatically set the value for the following keys:

  - `testcase_name`: with values _"forward"_ and _"backward"_
  - `convolution` (the function that actually runs the forward or backward
  convolution): with values `_run_convolution` and `_backward_wrapped_with_grad`
  (see its definition below)

  Although parameterizing tests is not ideal, we do this because there are many
  checks (7) that are run on both forward and backward convolution ops.
  Therefore, this small extra complexity is justified by the following reasons:

    - Setup is identical
    - Error checks covered are the same
    - Error messages are identical (modulo name of the function)
    - Avoid duplicating the tests

  This decorator assumes that:

    - Both `forward` and `backward` parameters have the same key. It will raise
      an `AssertionError`, otherwise.
    - the grad shape will be (2, 1, 8, 8). It can be easily adapted by promoting
      this variable to be a parameter

  Args:
    forward: a dictionary that will be passed down to the test case keyword
      arguments of the forward test.
    backward: a dictionary that will be passed down to the test case keyword
      arguments of the backward test.

  Returns:
    A `parameterized.named_parameter()` decorator with forward and backward
    parameters set.
  """

  # Leave this variable here for visibility.
  grad_shape = (2, 1, 8, 8)

  # From this point onwards, both `forward` and `backward` should:
  #
  #   - Not be `None`
  #   - Have the same set of keys
  forward = forward or {}
  backward = backward or {}

  assert forward.keys() == backward.keys(), (
      "convolution parameterization for `forward` and `backward` dictionaries"
      f" must have the same keys, got {set(forward.keys())} and"
      f" {set(backward.keys())}"
  )

  # Wraps `_run_convolution_backward()` function.
  #
  # Before actually calling the convolution backward function, creates a grad
  # tensor. This allows the forward and the backward function to be called with
  # the same set of positional arguments.
  def backward_wrapped_with_grad(*args, **kwargs):
    grad = torch.zeros(grad_shape, device=et.device())
    return _run_convolution_backward(grad, *args, **kwargs)

  return parameterized.named_parameters(
      {
          "testcase_name": "forward",
          "convolution": _run_convolution,
          **forward,
      },
      {
          "testcase_name": "backward",
          "convolution": backward_wrapped_with_grad,
          **backward,
      },
  )


def _make_lu_unpack_outputs(
    p: tuple[int, ...], l: tuple[int, ...], u: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
  """Creates a 3-tuple of tensors for `lu_unpack()` op."""
  return (
      torch.empty(p, device=et.device()),
      torch.empty(l, device=et.device()),
      torch.empty(u, device=et.device()),
  )


def _run_native_layer_norm(
    inp: torch.Tensor, normalized_shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Runs `torch.native_layer_norm` with default arguments."""
  return torch.native_layer_norm(
      inp, normalized_shape, weight=None, bias=None, eps=1e-5
  )


def _run_native_layer_norm_backward(
    inp: torch.Tensor, normalized_shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Runs `torch.ops.aten.native_layer_norm_backward` with default arguments."""

  grad = torch.randn(inp.shape, device=et.device())

  # Shape of `mean` and `rstd`.
  diff = len(inp.shape) - len(normalized_shape)

  shape = [s if i < diff else 1 for i, s in enumerate(inp.shape)]
  mean = torch.randn(shape, device=et.device())
  rstd = torch.randn(shape, device=et.device())

  # torch.ops.aten.native_layer_norm_backward op does not take in kwargs.
  # Therefore, all of them must be passed as positional arguments.
  weight = None
  bias = None
  output_mask = (False, False, False)

  return torch.ops.aten.native_layer_norm_backward(
      grad, inp, normalized_shape, mean, rstd, weight, bias, output_mask
  )


class TpuVsCpuErrorTest(et.ErrorTestBase, parameterized.TestCase):
  """Tests error messages on TPU vs on CPU."""

  def test_triu_insufficient_dims(self):
    """Tests that triu with insufficient dims fails with expected error."""
    t = torch.ones(1, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""triu: input tensor must have at least 2 dimensions""",
    ):
      torch.triu(t, 1)

  def test_ctc_loss_log_probs_3d(self):
    log_probs = torch.randn(2, 3, device=et.device())
    targets = torch.randint(1, 3, (2, 3), dtype=torch.int32, device=et.device())
    input_lengths = torch.tensor([2, 2], dtype=torch.int32, device=et.device())
    target_lengths = torch.tensor([3, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Expected 3-dimensional tensor, but got 2-dimensional tensor for argument #1 'log_probs' (while checking arguments for ctc_loss_allocate_outputs)""",
        tpu="""_ctc_loss(): expected log_probs to be 3-D, got 2-D""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._ctc_loss.Tensor(
          log_probs, targets, input_lengths, target_lengths, 0, False
      )

  def test_ctc_loss_targets_1d_or_2d(self):
    log_probs = torch.randn(5, 2, 3, device=et.device())
    targets = torch.randint(
        1, 3, (2, 3, 4), dtype=torch.int32, device=et.device()
    )
    input_lengths = torch.tensor([5, 5], dtype=torch.int32, device=et.device())
    target_lengths = torch.tensor([3, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Expected 1 to 2 dimensions, but got 3-dimensional tensor for argument #2 'targets' (while checking arguments for ctc_loss_allocate_outputs)""",
        tpu="""_ctc_loss(): expected targets to be 1-D or 2-D, got 3-D""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._ctc_loss.Tensor(
          log_probs, targets, input_lengths, target_lengths, 0, False
      )

  def test_ctc_loss_input_lengths_size_match_batch_size(self):
    log_probs = torch.randn(5, 2, 3, device=et.device())
    targets = torch.randint(1, 3, (2, 3), dtype=torch.int32, device=et.device())
    input_lengths = torch.tensor(
        [5, 5, 5], dtype=torch.int32, device=et.device()
    )
    target_lengths = torch.tensor([3, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu="""input_lengths must be of size batch_size""",
        tpu="""_ctc_loss(): expected input_lengths to have batch_size (2) elements, got 3""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._ctc_loss.Tensor(
          log_probs, targets, input_lengths, target_lengths, 0, False
      )

  def test_ctc_loss_target_lengths_size_match_batch_size(self):
    log_probs = torch.randn(5, 2, 3, device=et.device())
    targets = torch.randint(1, 3, (2, 3), dtype=torch.int32, device=et.device())
    input_lengths = torch.tensor([5, 5], dtype=torch.int32, device=et.device())
    target_lengths = torch.tensor(
        [3, 3, 3], dtype=torch.int32, device=et.device()
    )
    with et.assert_raises_message(
        RuntimeError,
        cpu="""target_lengths must be of size batch_size""",
        tpu="""_ctc_loss(): expected target_lengths to have batch_size (2) elements, got 3""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._ctc_loss.Tensor(
          log_probs, targets, input_lengths, target_lengths, 0, False
      )

  def test_ctc_loss_backward_log_probs_3d(self):
    grad_out = torch.randn(2, device=et.device())
    log_probs = torch.randn(2, 3, device=et.device())
    targets = torch.randint(1, 3, (2, 3), dtype=torch.int32, device=et.device())
    input_lengths = torch.tensor([2, 2], dtype=torch.int32, device=et.device())
    target_lengths = torch.tensor([3, 3], dtype=torch.int32, device=et.device())
    neg_log_likelihood = torch.randn(2, device=et.device())
    log_alpha = torch.randn(2, 2, 7, device=et.device())

    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        tpu="""ctc_loss_backward(): expected log_probs to be 3-D, got 2-D""",
    ):
      torch.ops.aten._ctc_loss_backward.Tensor(
          grad_out,
          log_probs,
          targets,
          input_lengths,
          target_lengths,
          neg_log_likelihood,
          log_alpha,
          0,
          False,
      )

  def test_tril_insufficient_dims(self):
    """Tests that tril with insufficient dims fails with expected error."""
    t = torch.tensor(42, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""tril: input tensor must have at least 2 dimensions""",
    ):
      torch.tril(t, 0)

  def test_linalg_solve_triangular_non_sq_failure(self):
    """Tests that linalg.solve_triangular() fails with less than 2 dimensions."""
    a = torch.ones(2, device=et.device(), dtype=torch.float32)
    b = torch.ones(2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_solve_triangular(): expected the first argument to have at least 2 dimensions, got 1""",
        cpu="""linalg.solve_triangular: The input tensor A must have at least 2 dimensions.""",
    ):
      torch.linalg.solve_triangular(
          a, b, upper=True, left=True, unitriangular=False
      )

  def test_linalg_solve_triangular_dim_mismatch_failure_right(self):
    """Tests that linalg.solve_triangular() fails when lh and rh have mismatching dimensions."""
    a = torch.ones(2, 2, 2, device=et.device(), dtype=torch.float32)
    b = torch.ones(2, 2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_solve_triangular(): left == False means we are solving X * A = B; expected the two inputs to have matching last dimension, got 2 and 3""",
        cpu="""linalg.solve_triangular: Incompatible shapes of A and B for the equation XA = B (2x2 and 2x3)""",
    ):
      torch.linalg.solve_triangular(
          a, b, upper=True, left=False, unitriangular=False
      )

  def test_linalg_solve_triangular_dim_mismatch_failure_left(self):
    """Tests that linalg.solve_triangular() fails when lh and rh have mismatching dimensions."""
    a = torch.ones(2, 2, 2, device=et.device(), dtype=torch.float32)
    b = torch.ones(2, 3, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_solve_triangular(): left == True means we are solving A * X = B; expected the two inputs to have matching second to last dimension, got 2 and 3""",
        cpu="""linalg.solve_triangular: Incompatible shapes of A and B for the equation AX = B (2x2 and 3x2)""",
    ):
      torch.linalg.solve_triangular(
          a, b, upper=True, left=True, unitriangular=False
      )

  def test_linalg_solve_unsupported_dtype_failure(self):
    """Tests that linalg.solve_triangular() fails when lh and rh have mismatching dimensions."""
    a = torch.ones(2, 2, 2, device=et.device(), dtype=torch.bfloat16)
    b = torch.ones(2, 2, 2, device=et.device(), dtype=torch.bfloat16)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_solve_triangular(): triangular solve not supported for dtype bfloat16""",
        cpu=""""triangular_solve_cpu" not implemented for 'BFloat16'""",
    ):
      torch.linalg.solve_triangular(
          a, b, upper=True, left=True, unitriangular=False
      )

  def test_eq__for_scalar_with_integral_dtype(self):
    """Tests that eq_ with an integral dtype (e.g. uint8) doesn't fail."""
    t1 = torch.tensor(2, device=et.device(), dtype=torch.uint8)
    t2 = torch.tensor(3, device=et.device(), dtype=torch.uint8)
    t1.eq_(t2)  # The Boolean results should be converted to uint8 (0 or 1).
    t1.to("cpu")
    self.assertEqual(t1.dtype, torch.uint8)
    self.assertEqual(t1.item(), 0)

    t1 = torch.tensor(2, device=et.device(), dtype=torch.uint8)
    t2 = torch.tensor(2, device=et.device(), dtype=torch.uint8)
    t1.eq_(t2)  # The Boolean results should be converted to uint8 (0 or 1).
    t1.to("cpu")
    self.assertEqual(t1.dtype, torch.uint8)
    self.assertEqual(t1.item(), 1)

  def test_eq__for_tensor_with_integral_dtype(self):
    """Tests that eq_ with an integral dtype (e.g. uint8) doesn't fail."""
    # device = "cpu"
    t1 = torch.tensor([2], device=et.device(), dtype=torch.uint8)
    t2 = torch.tensor([3], device=et.device(), dtype=torch.uint8)
    t1.eq_(t2)  # The Boolean results should be converted to uint8 (0 or 1).
    t1.to("cpu")
    self.assertEqual(t1.dtype, torch.uint8)
    self.assertEqual(t1.item(), 0)

    t1 = torch.tensor([2], device=et.device(), dtype=torch.uint8)
    t2 = torch.tensor([2], device=et.device(), dtype=torch.uint8)
    t1.eq_(t2)  # The Boolean results should be converted to uint8 (0 or 1).
    t1.to("cpu")
    self.assertEqual(t1.dtype, torch.uint8)
    self.assertEqual(t1.item(), 1)

  def test_masked_select_with_type_mismatch(self):
    """Masked select function fails when mask has an invalid type."""
    t = torch.ones(2, 3, 3, device=et.device(), dtype=torch.float32)
    mask = torch.rand(2, 3, 3, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu="""masked_select: expected BoolTensor for mask""",
        tpu="""masked_select(): expected the mask to be bool, got float32""",
    ):
      t.masked_select(mask)

  def test_masked_select_with_shape_mismatch(self):
    """Masked select function fails when mask has a mismatching shape."""
    t = torch.ones(2, 3, 3, device=et.device(), dtype=torch.float32)
    mask = torch.rand(2, device=et.device(), dtype=torch.float32) > 0.5

    with et.assert_raises_message(
        RuntimeError,
        tpu="""The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2""",
    ):
      t.masked_select(mask)

  def test_index_copy_rank_mismatch(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (1), destination dimensionality (2)""",
        tpu="""index_copy(): self and source must have the same number of dimensions, got 2 and 1""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(2, device=et.device())
      )

  def test_index_copy_index_rank_not_1(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""index_copy_(): Index should have dimension 1 or 0 (got 2)""",
        tpu="""index_copy(): index must be 1D, got shape [1, 1]""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([[0]], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 2, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_dim_out_of_range(self):
    with et.assert_raises_message(
        IndexError,
        cpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        # This error is generated by PyTorch and we cannot easily replace it.
        tpu="""index_copy(): dimension out of range (expected to be in range of [-2, 1], but got 2)""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 2, device=et.device())
      torch.index_copy(
          t, 2, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_source_dim_ne_index_size(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""index_copy_(): Number of indices (1) should be equal to source.size(dim) (2)""",
        tpu="""index_copy(): source must have the same number of elements as the index along dimension 0, got 2 and 1""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, 2, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_self_source_size_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""index_copy_(): Source/destination tensor must have same slice shapes. Destination slice shape: 2 at dimension 0 and source slice shape: 3 at dimension 0.""",
        tpu="""index_copy(): self and source must have the same size along dimension 1, got 2 and 3""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 3, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_scalar_dim_out_of_range(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""index_copy(): dim must be 0 for scalar input, got 1""",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor(1, device=et.device())
      torch.index_copy(
          t, 1, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_copy_scalar_source_not_scalar(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""index_copy_(): Source/destination tensor must have same slice shapes. Destination slice shape:  at dimension 0 and source slice shape: 1 at dimension 0.""",
        tpu="""index_copy(): source shape must match self shape, excluding the specified dimension, got source shape [1, 1] and self shape []""",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor([[1]], device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_copy_scalar_index_size_ne_1(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""index_copy_(): When source is scalar, index should have one element (got 2)""",
        tpu="""index_copy(): index must be 1D of size 1 for scalar input, got shape [2]""",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0, 0], device=et.device(), dtype=torch.long)
      source = torch.tensor(1, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_fill_index_invalid_rank(self):
    tpu_err = """index_fill_(): expected index to be at most 1-D, got 2-D"""
    cpu_err = """Index has to be a vector/scalar"""
    t = torch.ones(2, 2, device=et.device())
    index = torch.tensor([[0]], device=et.device(), dtype=torch.long)

    # Scalar variant
    with et.assert_raises_message(RuntimeError, tpu=tpu_err, cpu=cpu_err):
      t.index_fill_(0, index, 5.0)

    # Tensor variant
    with et.assert_raises_message(RuntimeError, tpu=tpu_err, cpu=cpu_err):
      t.index_fill_(0, index, torch.tensor(5.0, device=et.device()))

  def test_index_fill_index_type_not_long(self):
    tpu_err = """index_fill_(): expected index dtype to be Long, got int32"""
    cpu_err = """index_fill_(): Expected dtype int64 for index."""
    t = torch.ones(2, 2, device=et.device())
    index = torch.tensor([0], device=et.device(), dtype=torch.int)
    err_type = RuntimeError if et.device().type == "tpu" else IndexError

    # Scalar variant
    with et.assert_raises_message(err_type, tpu=tpu_err, cpu=cpu_err):
      t.index_fill_(0, index, 5.0)

    # Tensor variant
    with et.assert_raises_message(err_type, tpu=tpu_err, cpu=cpu_err):
      t.index_fill_(0, index, torch.tensor(5.0, device=et.device()))

  def test_index_fill_dim_out_of_range(self):
    tpu_err = """index_fill_(): dimension out of range (expected to be in range of [-2, 1], but got 2)"""
    cpu_err = """Dimension out of range (expected to be in range of [-2, 1], but got 2)"""
    t = torch.ones(2, 2, device=et.device())
    index = torch.tensor([0], device=et.device(), dtype=torch.long)

    # Scalar variant
    with et.assert_raises_message(IndexError, tpu=tpu_err, cpu=cpu_err):
      t.index_fill_(2, index, 5.0)

    # Tensor variant
    with et.assert_raises_message(IndexError, tpu=tpu_err, cpu=cpu_err):
      t.index_fill_(2, index, torch.tensor(5.0, device=et.device()))

  def test_index_fill_value_tensor_not_0d(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""index_fill_(): expected value to be a 0-D tensor, got 1-D tensor""",
        cpu="""index_fill_ only supports a 0-dimensional value tensor, but got tensor with 1 dimension(s).""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      value = torch.tensor([5.0], device=et.device())
      t.index_fill_(0, index, value)

  def test_fill_with_incorrect_shape_0_dim(self):
    self.do_test_fill_with_incorrect_shape([1])

  def test_fill_with_incorrect_shape_1_dim_with_multiple_values(self):
    self.do_test_fill_with_incorrect_shape([2])

  def test_fill_with_incorrect_shape_2_dim(self):
    self.do_test_fill_with_incorrect_shape([1, 1])

  def do_test_fill_with_incorrect_shape(self, shape):
    """Tests that fill with an empty tensor fails with expected error."""
    t = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.float32)
    value = torch.ones(shape, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu="""fill_ only supports 0-dimension value tensor but got tensor with"""
        f""" {len(shape)} dimensions.""",
        tpu=f"fill_(): expected value to be a 0-D tensor, got {len(shape)}-D",
    ):
      torch.fill(t, value)

  def test_fmod_tensor_with_unsupported_dtype(self):
    t = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.complex64)
    other = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        cpu=""""fmod_cpu" not implemented for 'ComplexFloat'""",
        tpu="""fmod(): complex dtypes are not supported""",
    ):
      torch.fmod(t, other)

    t = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.bool)
    other = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        cpu=""""fmod_cpu" not implemented for 'Bool'""",
        tpu="""fmod(): boolean dtypes are not supported""",
    ):
      torch.fmod(t, other)

  def test_masked_select_out_with_different_scalar_types(self):
    """Masked select function fails when self and out have different scalar types."""
    t = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = torch.ones(5, device=et.device(), dtype=torch.int32)
    mask = torch.ones(5, device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        cpu="""masked_select(): self and result must have the same scalar type""",
        tpu="""masked_select(): the out tensor dtype is expected to be float32, got int32""",
    ):
      torch.masked_select(t, mask, out=out)

  def test_mm_with_matching_sizes(self):
    """Tests that mm with matching sizes doesn't fail."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    t3 = torch.mm(t1, t2)
    self.assertEqual(t3.shape, (2, 2))

  def test_mm_with_non_2d_arg1(self):
    """Tests that mm with non-2D argument 1 fails with expected error."""
    t1 = torch.ones(2, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(4, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""self must be a matrix""",
    ):
      torch.mm(t1, t2)

  def test_mm_with_non_2d_arg2(self):
    """Tests that mm with non-2D argument 2 fails with expected error."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 3, 4, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""mat2 must be a matrix""",
    ):
      torch.mm(t1, t2)

  def test_mm_with_mismatched_sizes(self):
    """Tests that mm with mismatched sizes fails with expected error."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(4, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""mat1 and mat2 shapes cannot be multiplied (2x3 and 4x2)""",
    ):
      torch.mm(t1, t2)

  def test_mm_with_mismatched_data_types(self):
    """Tests that mm with mismatched data types fails with expected error."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 2, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""expected m1 and m2 to have the same dtype, but got: float != int""",
        tpu="""mm(): expected the two arguments to have the same dtype, got float32 vs int32""",
        message_reviewed_by="wan",
    ):
      torch.mm(t1, t2)

  def test_nll_loss_unsupported_input_dtype(self):
    t = torch.ones(3, 5, device=et.device(), dtype=torch.int32)
    target = torch.tensor([1, 0, 4], device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        cpu=""""nll_loss_out_frame" not implemented for 'Int'""",
        tpu="""nll_loss_forward(): unsupported input dtype: int32""",
    ):
      torch.nn.functional.nll_loss(t, target)

  def test_nll_loss_unsupported_target_dtype(self):
    t = torch.ones(3, 5, device=et.device(), dtype=torch.float32)
    target = torch.tensor([1, 0, 4], device=et.device(), dtype=torch.int32)
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_message = (
          "expected target dtype to be Long or Byte, but got Int"
      )
    else:
      tpu_error_message = (
          "nll_loss_forward(): expected the target dtype to be either int64"
          " or uint8, got int32"
      )
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu=tpu_error_message,
    ):
      torch.nn.functional.nll_loss(t, target)

  def test_nll_loss2d_shape_mismatch(self):
    t = torch.ones(1, 3, 2, 2, device=et.device(), dtype=torch.float32)
    target = torch.ones(1, 2, 3, device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""size mismatch (got input: [1, 3, 2, 2] , target: [1, 2, 3]""",
        tpu="""nll_loss2d_forward(): expect the shapes of the input [N, C, d1, ..., dk] and the target [N, d1, ..., dk] (k >= 1) to match, got input: [1, 3, 2, 2], target: [1, 2, 3]""",
    ):
      torch.nn.functional.nll_loss(t, target)

  def test_nll_loss2d_backward_invalid_input_dim(self):
    grad_output = torch.ones(1, 2, 2, device=et.device(), dtype=torch.float32)
    t = torch.ones(1, 3, 2, device=et.device(), dtype=torch.float32)
    target = torch.ones(1, 2, 2, device=et.device(), dtype=torch.long)
    total_weight = torch.ones((), device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""only batches of spatial inputs supported (4D tensors), but got input of dimension: 3""",
        tpu="""nll_loss2d_backward(): expected input to be a 4D tensor, got 3D""",
    ):
      torch.ops.aten.nll_loss2d_backward(
          grad_output, t, target, None, 1, -100, total_weight
      )

  def test_nll_loss2d_backward_invalid_target_dim(self):
    grad_output = torch.ones(1, 2, 2, device=et.device(), dtype=torch.float32)
    t = torch.ones(1, 3, 2, 2, device=et.device(), dtype=torch.float32)
    target = torch.ones(1, 2, device=et.device(), dtype=torch.long)
    total_weight = torch.ones((), device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""only batches of spatial targets supported (3D tensors) but got targets of dimension: 2""",
        tpu="""nll_loss2d_backward(): expected target to be a 3D tensor, got 2D""",
    ):
      torch.ops.aten.nll_loss2d_backward(
          grad_output, t, target, None, 1, -100, total_weight
      )

  def test_nll_loss_invalid_reduction(self):
    t = torch.ones(1, 3, 2, 2, device=et.device(), dtype=torch.float32)
    target = torch.ones(1, 2, 3, device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        ValueError,
        tpu="""all is not a valid value for reduction""",
    ):
      torch.nn.functional.nll_loss(t, target, reduction="all")

  def test_zero_dim_size_in_ones(self):
    """Tests that torch.ones(...) with zero dimension sizes doesn't fail."""

    torch.ones(0, device=et.device(), dtype=torch.float32)
    torch.ones(3, 0, 2, device=et.device(), dtype=torch.float32)

  def test_negative_dim_size_in_ones(self):
    """Tests that torch.ones(...) with negative dimension sizes fails with expected error."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Trying to create tensor with negative dimension -1: [-1]""",
        tpu="""empty(): dimension sizes must be >= 0, got [-1], which contains -1""",
    ):
      torch.ones(-1, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu="""Trying to create tensor with negative dimension -2: [3, -2, -4]""",
        tpu="""empty(): dimension sizes must be >= 0, got [3, -2, -4], which contains -2 and -4""",
    ):
      torch.ones(3, -2, -4, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu="""Trying to create tensor with negative dimension -2: [3, -2, -4, 1, -5]""",
        tpu="""empty(): dimension sizes must be >= 0, got [3, -2, -4, 1, -5], which contains -2, -4, and -5""",
    ):
      torch.ones(3, -2, -4, 1, -5, device=et.device(), dtype=torch.float32)

  def test_dim_size_overflow_in_ones(self):
    """Tests that torch.ones() fails with expected error when the dimension size overlows."""
    with et.assert_raises_message(
        TypeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu=re.compile(r""".*Overflow when unpacking long.*""", re.DOTALL),
    ):
      # The dimension size 2**63 fits in uint64_t but not in int64_t.
      torch.ones(2**63, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        TypeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu=re.compile(r""".*Overflow when unpacking long.*""", re.DOTALL),
    ):
      # The dimension size 2**64 fits in neither uint64_t nor int64_t.
      torch.ones(1, 2**64, device=et.device(), dtype=torch.float32)

  def test_sign_unsupported_complex(self):
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch before our kernel is called;
        # we don't have control over this error message.
        tpu="""Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.""",
    ):
      torch.sign(torch.zeros(1, device=et.device(), dtype=torch.complex64))

  def test_size_product_overflow_in_ones(self):
    """Tests that torch.ones() fails with expected error when the size product is negative."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Storage size calculation overflowed with sizes=[2147483648, 4294967296]""",
        tpu="""empty(): product of dimension sizes [2147483648, 4294967296] overflows as int64""",
    ):
      # The product of the dimensions is 2 ** 63, which doesn't cause an
      # overflow in XLA. However, it doesn't fit in int64_t.
      torch.ones(2**31, 2**32, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu="""Storage size calculation overflowed with sizes=[1073741824, 1073741824, 1073741824]""",
        tpu="""empty(): product of dimension sizes [1073741824, 1073741824, 1073741824] overflows as int64""",
    ):
      # The product of the dimensions is 2 ** 90, which causes an overflow in
      # XLA.
      torch.ones(2**30, 2**30, 2**30, device=et.device(), dtype=torch.float32)

  def test_byte_size_overflow_in_ones(self):
    """Tests that torch.ones() fails with expected error when the byte size overlows."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Storage size calculation overflowed with sizes=[2147483648, 2147483648]""",
        tpu="""empty(): product of dimension sizes [2147483648, 2147483648] and size of f32 (4 bytes) overflows as int64""",
    ):
      # The product of the dimensions is 2 ** 62, which doesn't cause an
      # overflow in XLA. However, the byte size is 2 ** 64, which overflows
      # int64_t.
      torch.ones(2**31, 2**31, device=et.device(), dtype=torch.float32)

  def test_histc_bounds_overflow(self):
    """Tests that torch.histc() fails when the bounds overflow."""
    max_int32 = torch.iinfo(torch.int32).max
    t = torch.tensor(
        [max_int32, max_int32], device=et.device(), dtype=torch.int32
    )
    with et.assert_raises_message(
        RuntimeError,
        cpu="""value cannot be converted to type int without overflow""",
        tpu="""histc(): expected min and max to be within the range of their data types, but got min = 2147483646 and max = -2147483648. This happened because min and max were adjusted by one (due to min == max), which resulted in an overflow""",
    ):
      torch.histc(t)

  def test_histc_bounds_underflow(self):
    """Tests that torch.histc() fails when the bounds underflow."""
    min_int32 = torch.iinfo(torch.int32).min
    t = torch.tensor(
        [min_int32, min_int32], device=et.device(), dtype=torch.int32
    )
    with et.assert_raises_message(
        RuntimeError,
        cpu="""value cannot be converted to type int without overflow""",
        tpu="""histc(): expected min and max to be within the range of their data types, but got min = 2147483647 and max = -2147483647. This happened because min and max were adjusted by one (due to min == max), which resulted in an overflow""",
    ):
      torch.histc(t)

  def test_histc_bounds_not_nan(self):
    """Tests that torch.histc() fails when the bounds are NaN."""
    t = torch.tensor([0, float("nan")], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""torch.histc: range of [-nan, -nan] is not finite""",
        tpu="""histc(): expected min and max to be finite, got nan and nan. Either make sure that the input data is finite, or provide valid finite bounds""",
    ):
      torch.histc(t)

  def test_histc_bounds_inf(self):
    """Tests that torch.histc() fails when the bounds are infinity."""
    t = torch.tensor([0, float("inf")], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""torch.histc: range of [0, inf] is not finite""",
        tpu="""histc(): expected min and max to be finite, got 0 and inf. Either make sure that the input data is finite, or provide valid finite bounds""",
    ):
      torch.histc(t)

  def test_histc_bounds_not_in_order(self):
    """Tests that torch.histc() fails when the bounds are not in order."""
    t = torch.tensor([0, 0], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""torch.histc: max must be larger than min""",
        tpu="""histc(): expected min <= max, got 1 vs 0""",
    ):
      torch.histc(t, min=1, max=0)

  def test_invalid_index_dtype_in_take(self):
    """Tests that torch.take() fails when the index has the wrong dtype."""
    t = torch.tensor([0, 1, 2], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""take(): Expected a long tensor for index, but got Int""",
        tpu="""take(): expected index dtype to be int64, got int32""",
    ):
      torch.take(t, torch.tensor([0, 1], dtype=torch.int32, device=et.device()))

  def test_empty_tensor_in_take(self):
    """Tests that torch.take() fails when the input tensor is empty but index is not."""
    t = torch.tensor([], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        cpu="""take(): tried to take from an empty tensor""",
        tpu="""take(): input tensor must be non-empty when the index tensor is non-empty""",
    ):
      torch.take(t, torch.tensor([0], dtype=torch.int64, device=et.device()))

  def test_invalid_index_in_take(self):
    """Tests that torch.take() fails when the index is invalid."""
    t = torch.tensor([0, 1, 2], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        cpu="""out of range: tried to access index 3 on a tensor of 3 elements.""",
        tpu="""take(): expected indices to be in range [-3, 2], got 3""",
    ):
      torch.take(t, torch.tensor([0, 3], dtype=torch.int64, device=et.device()))

    with et.assert_raises_message(
        IndexError,
        cpu="""out of range: tried to access index -4 on a tensor of 3 elements.""",
        tpu="""take(): expected indices to be in range [-3, 2], got -4""",
    ):
      torch.take(
          t, torch.tensor([0, -4], dtype=torch.int64, device=et.device())
      )

  def test_invalid_size_in_zeros(self):
    """Tests that torch.zeros() fails with expected error when the size is invalid."""
    # This error message is generated by PyTorch. In newer PyTorch v2.9.0, shape
    # validation for factory functions (like zeros) was moved to the frontend,
    # meaning it catches the error earlier and produces a simpler, more generic
    # error message different from the one produced in the previous PyTorch
    # v2.8.0.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""zeros: Dimension size must be non-negative.""",
    ):
      torch.zeros(-1, device=et.device(), dtype=torch.float32)

  def test_invalid_size_in_empty_memory_format(self):
    """Tests that torch.empty() fails with expected error when the size is invalid."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Trying to create tensor with negative dimension -1: [-1]""",
        tpu="""empty(): dimension sizes must be >= 0, got [-1], which contains -1""",
    ):
      torch.empty(-1, device=et.device(), dtype=torch.float32)

  def test_invalid_broadcast_in_binary_op_add(self):
    """Tests that torch.add() fails with expected error when the sizes are mismatched."""

    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1""",
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1""",
    ):
      torch.add(t1, t2)

  def test_round_decimals_param_integer_input(self):
    """torch.round() errors when input is an integer and decimals is specified."""

    t = torch.ones(1, device=et.device(), dtype=torch.int64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""round(): expected the input dtype not to be integer when the decimals argument is specified (-1), got int64""",
        cpu=""""round_cpu" not implemented for 'Long'""",
        message_reviewed_by="wan",
    ):
      t.round(decimals=-1)

    t = torch.ones(1, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""round(): expected the input dtype not to be integer when the decimals argument is specified (2), got int32""",
        cpu=""""round_cpu" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      t.round_(decimals=2)

    t = torch.ones(1, device=et.device(), dtype=torch.int16)
    out_t = torch.zeros(1, device=et.device(), dtype=torch.int16)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""round(): expected the input dtype not to be integer when the decimals argument is specified (0), got int16""",
        cpu=""""round_vml_cpu" not implemented for 'Short'""",
        message_reviewed_by="wan",
    ):
      torch.round(t, decimals=0, out=out_t)

  def test_round_invalid_input_dtype(self):
    t = torch.tensor([True, False], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        cpu=""""round_vml_cpu" not implemented for 'Bool'""",
        tpu="""round(): dtype bool is not supported""",
        message_reviewed_by="wan",
    ):
      torch.round(t)

  def test_roll_errors(self):
    """roll() fails when input parameters are invalid."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""shifts and dimensions must align. shifts: 2, dims:1""",
        tpu="""roll(): shifts and dims must align, got shifts: 2, dims: 1""",
    ):
      torch.roll(t, shifts=(2, 3), dims=(0,))

    with et.assert_raises_message(
        RuntimeError,
        cpu="""shifts and dimensions must align. shifts: 2, dims:0""",
        tpu="""roll(): shifts and dims must align, got shifts: 2, dims: 0""",
    ):
      torch.roll(t, shifts=(2, 3))

  def test_reduction_dim_out_of_bounds_var(self):
    self.do_test_reduction_dim_out_of_bounds(torch.var)

  def test_reduction_dim_out_of_bounds_mean(self):
    self.do_test_reduction_dim_out_of_bounds(torch.mean)

  def test_reduction_dim_out_of_bounds_sum(self):
    self.do_test_reduction_dim_out_of_bounds(torch.sum)

  def do_test_reduction_dim_out_of_bounds(self, reduction_fn):
    """Reduction function fails when dimension is out of bounds."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)

    for dim in [-3, 3]:
      with et.assert_raises_message(
          IndexError,
          # This error is generated by PyTorch.
          tpu="""Dimension out of range (expected to be in range of [-2, 1], but"""
          f""" got {dim})""",
      ):
        reduction_fn(t, dim=dim)

  def test_reduction_dim_repeated_var(self):
    self.do_test_reduction_dim_repeated(torch.var)

  def test_reduction_dim_repeated_mean(self):
    self.do_test_reduction_dim_repeated(torch.mean)

  def test_reduction_dim_repeated_sum(self):
    self.do_test_reduction_dim_repeated(torch.sum)

  def do_test_reduction_dim_repeated(self, reduction_fn):
    """Reduction function fails when canonical dimension is repeated."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by PyTorch.
        tpu="""dim 1 appears multiple times in the list of dims""",
    ):
      reduction_fn(t, dim=[-1, 1])

  def test_reduction_dim_scalar_var(self):
    self.do_test_reduction_dim_scalar(torch.var)

  def test_reduction_dim_scalar_mean(self):
    self.do_test_reduction_dim_scalar(torch.mean)

  def test_reduction_dim_scalar_sum(self):
    self.do_test_reduction_dim_scalar(torch.sum)

  def do_test_reduction_dim_scalar(self, reduction_fn):
    """Reduction function fails when dimension is invalid for scalar."""
    t_rank0 = torch.tensor(1.0, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        IndexError,
        # This error is generated by PyTorch.
        tpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
    ):
      reduction_fn(t_rank0, dim=1)

  def test_reduction_unsupported_int_dtype_mean(self):
    """Mean fails for integral dtypes."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by PyTorch.
        tpu="""mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Int""",
    ):
      torch.mean(t, dim=0)

  def test_reduction_unsupported_int_dtype_var(self):
    """Var fails for integral dtypes."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""std and var only support floating point and complex dtypes""",
        tpu="""var(): expected a floating point or complex dtype, got int32""",
    ):
      torch.var(t, dim=0)

  def test_unfold_size_too_large(self):
    """Unfold fails when size is larger than dimension."""
    t = torch.ones(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu="""maximum size for tensor at dimension 0 is 5 but size is 6""",
        tpu="""unfold(): expected size <= dimension size (shape[0]: 5), got size: 6""",
    ):
      t.unfold(0, 6, 1)

  def test_unfold_dim_out_of_bounds(self):
    """Unfold fails when dimension is out of bounds."""
    t = torch.ones(2, 3, device=et.device())
    with et.assert_raises_message(
        IndexError,
        cpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        tpu="""unfold(): expected dimension to be in range of [-2, 1] for shape [2, 3], got 2""",
    ):
      t.unfold(2, 1, 1)

  def test_unfold_zero_step(self):
    """Unfold fails when step is 0."""
    t = torch.ones(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu="""step is 0 but must be > 0""",
        tpu="""unfold(): expected step > 0, got 0""",
    ):
      t.unfold(0, 2, 0)

  def test_var_negative_reduction_factor(self):
    """A warning is issued and nan returned when the variance degrees of freedom is <= 0."""
    # TODO: b/435570003 - Create a utility to compare warning messages.
    cpu_warn_msg = (
        "var(): degrees of freedom is <= 0. Correction should be strictly less"
        " than the reduction factor (input numel divided by output numel)."
    )

    t = torch.ones(1, device=et.device(), dtype=torch.float32)
    tpu_warn_msg = (
        "var(): degrees of freedom (i.e., reduction size - correction) should"
        " be positive, got reduction size = 1, correction = 1, and degrees of"
        " freedom = 0"
    )

    warn_msg = cpu_warn_msg if et.device().type == "cpu" else tpu_warn_msg

    with self.assertWarnsRegex(UserWarning, re.escape(warn_msg)):
      result = torch.var(t, correction=1)
    self.assertTrue(torch.all(torch.isnan(result)), f"Got {result.to('cpu')}")

    t = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    tpu_warn_msg = (
        "var(): degrees of freedom (i.e., reduction size - correction) should"
        " be positive, got reduction size = 3, correction = 4, and degrees of"
        " freedom = -1"
    )
    warn_msg = cpu_warn_msg if et.device().type == "cpu" else tpu_warn_msg

    with self.assertWarnsRegex(UserWarning, re.escape(warn_msg)):
      result = torch.var(t, dim=0, correction=4)
    self.assertTrue(torch.all(torch.isnan(result)), f"Got {result.to('cpu')}")

  def test_view_not_contiguity_like(self):
    t = torch.randn(2, 3).T
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch before our kernel is called;
        # we don't have control over this error message.
        tpu="""view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.""",
    ):
      t.view(6)

  def test_view_as_real_non_complex(self):
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_real(): expected the input dtype to be complex, got float32""",
        cpu="""view_as_real is only supported for complex tensors""",
        message_reviewed_by="wan",
    ):
      torch.view_as_real(t)

  def test_select_index_out_of_bounds(self):
    """Select function fails when index is out of bounds."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    dim = 1

    for index in [-40, 40]:
      with et.assert_raises_message(
          IndexError,
          # This error is generated by pytorch before our kernel is called;
          # we don't have control over this error message.
          tpu=f"""select(): index {index} out of range for tensor of size [2, 3]"""
          f""" at dimension {dim}""",
      ):
        t.select(dim, index)  # pylint: disable=unused-variable

  def test_select_dim_out_of_bounds(self):
    """Select function fails when dimension is out of bounds."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)

    for dim in [-3, 3]:
      with et.assert_raises_message(
          IndexError,
          # This error is generated by pytorch before our kernel is called;
          # we don't have control over this error message.
          tpu="""Dimension out of range (expected to be in range of [-2, 1], but"""
          f""" got {dim})""",
      ):
        t.select(dim, 1)  # pylint: disable=unused-variable

  def test_slice_on_scalar(self):
    t = torch.scalar_tensor(1.0, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        IndexError,
        # This error is generated by pytorch before redispatching to as_strided;
        # we don't have control over this error message.
        tpu="""slice() cannot be applied to a 0-dim tensor.""",
    ):
      sliced_t = t[0:1:1]  # pylint: disable=unused-variable

  def test_slice_zero_step(self):
    t = torch.ones(10, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        ValueError,
        # This error is generated by pytorch before redispatching to as_strided;
        # we don't have control over this error message.
        tpu="""slice step cannot be zero""",
    ):
      sliced_t = t[0:10:0]  # pylint: disable=unused-variable

  def test_slice_negative_step(self):
    t = torch.ones(10, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        ValueError,
        # This error is generated by pytorch before redispatching to as_strided;
        # we don't have control over this error message.
        tpu="""step must be greater than zero""",
    ):
      sliced_t = t[0:10:-1]  # pylint: disable=unused-variable

  def test_cat_empty_input(self):
    with et.assert_raises_message(
        ValueError,
        # This error is generated by pytorch before our kernel is called;
        # we don't have control over this error message.
        tpu="""torch.cat(): expected a non-empty list of Tensors""",
    ):
      torch.cat([])

  def test_cat_scalar_input(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""zero-dimensional tensor (at position 0) cannot be concatenated""",
    ):
      torch.cat([torch.tensor(1.0, device=et.device())])
    with et.assert_raises_message(
        RuntimeError,
        tpu="""zero-dimensional tensor (at position 1) cannot be concatenated""",
    ):
      torch.cat([
          torch.tensor([], dtype=torch.float32, device=et.device()),
          torch.tensor(1.0, device=et.device()),
      ])

  def test_cat_dim_out_of_range(self):
    with et.assert_raises_message(
        IndexError,
        tpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
    ):
      t0 = torch.tensor([], dtype=torch.float32, device=et.device())
      t3 = torch.tensor([1, 2, 3], device=et.device())
      torch.cat([t0, t3], dim=1)

  def test_cat_mismatched_dims(self):
    t3 = torch.tensor([1, 2, 3], device=et.device())
    t1x3 = torch.tensor([[1, 2, 3]], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Tensors must have same number of dimensions: got 1 and 2""",
    ):
      torch.cat([t3, t1x3])

  def test_cat_mismatched_dim_sizes(self):
    t2x2 = torch.tensor([[1, 2], [3, 4]], device=et.device())
    t2x3 = torch.tensor([[1, 2, 3], [4, 5, 6]], device=et.device())
    t3x2 = torch.tensor([[1, 2], [3, 4], [5, 6]], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Sizes of tensors must match except in dimension 0. Expected size 2 but got size 3 for tensor number 1 in the list.""",
    ):
      torch.cat([t2x2, t2x3])
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Sizes of tensors must match except in dimension 1. Expected size 2 but got size 3 for tensor number 1 in the list.""",
    ):
      torch.cat([t2x2, t3x2], dim=1)

  def test_addcmul_bool_error(self):
    """Tests that addcmul errors out with bool inputs."""
    self_tensor = torch.tensor([[True] * 5] * 5, dtype=torch.bool).to(
        et.device()
    )
    tensor1 = torch.tensor(True, dtype=torch.bool).to(et.device())
    tensor2 = torch.tensor(True, dtype=torch.bool).to(et.device())
    value = 1
    with et.assert_raises_message(
        RuntimeError,
        cpu=""""addcmul_cpu_out" not implemented for 'Bool'""",
        tpu="""addcmul(): bool tensors are not supported, got input: bool, tensor1: bool, tensor2: bool""",
    ):
      torch.addcmul(self_tensor, tensor1, tensor2, value=value)

  def test_index_put_too_many_indices_error(self):
    # TODO(mkkhanna): Fix exception type for TPU.
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""too many indices for tensor of dimension 1 (got 2)""",
        tpu="""index_put_(): too many indices for tensor of dimension 1, got 2""",
    ):
      torch.index_put_(
          torch.tensor([0, 1], device=et.device()),
          (
              torch.tensor([0], device=et.device()),
              torch.tensor([0], device=et.device()),
          ),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_index_dtype_error(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""tensors used as indices must be long, int, byte or bool tensors""",
        tpu="""index_put_(): tensors used as indices must be long, int, byte or bool tensors, got float32 at index 0""",
    ):
      torch.index_put_(
          torch.tensor([0, 1], device=et.device()),
          (torch.tensor([0], dtype=torch.float32, device=et.device()),),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_broadcast_indices_error(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""shape mismatch: indexing tensors could not be broadcast together with shapes [2], [3]""",
        tpu="""index_put_(): index tensors not broadcastable, got index tensor shape [3] and broadcast shape [2]: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0""",
    ):
      torch.index_put_(
          torch.tensor([[0, 1], [2, 3]], device=et.device()),
          (
              torch.tensor([0, 1], device=et.device()),
              torch.tensor([0, 1, 1], device=et.device()),
          ),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_broadcast_values_error(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""shape mismatch: value tensor of shape [2, 2] cannot be broadcast to indexing result of shape [2]""",
        tpu="""index_put_(): value tensor of shape [2, 2] cannot be broadcast to indexing result of shape [2]""",
    ):
      torch.index_put_(
          torch.tensor([[0, 1], [2, 3]], device=et.device()),
          (
              torch.tensor([0, 1], device=et.device()),
              torch.tensor([0, 1], device=et.device()),
          ),
          torch.tensor([[0, 1], [2, 3]], device=et.device()),
      )

  def test_index_put_dtype_mismatch_error(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Index put requires the source and destination dtypes match, got Int for the destination and Long for the source.""",
        tpu="""index_put_(): dtypes of values and destination must be the same, got int64 and int32""",
    ):
      torch.index_put_(
          torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=et.device()),
          (
              torch.tensor([0, 1], device=et.device()),
              torch.tensor([0, 1], device=et.device()),
          ),
          torch.tensor([0], dtype=torch.int64, device=et.device()),
      )

  def test_index_put_index_or_indices_must_be_specified_error(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""sym_strides() called on an undefined Tensor""",
        tpu="""index_put_(): indices must be specified""",
    ):
      torch.index_put_(
          torch.tensor([[0, 1], [2, 3]], device=et.device()),
          (),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_decompose_with_mask_mask_shape_mismatch(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""The shape of the mask [3, 5] at index 0 does not match the shape of the indexed tensor [2, 5] at index 0""",
        tpu="""index_put_(): the shape of the mask at index 0 must match the shape of the indexed tensor at index 0, got mask shape [3, 5] and indexed tensor shape [2, 5]""",
    ):
      tensor = torch.arange(10).view(2, 5).to(et.device())
      tensor_other = torch.arange(15).view(3, 5).to(et.device())
      boolean_mask = tensor_other % 2 != 0
      tensor[boolean_mask] = 100

  def test_index_put_decompose_with_mask_error_mask_dim_more_than_indexed_tensor_dim(
      self,
  ):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""index_put_(): the shape of the mask at index 1 must match the shape of the indexed tensor at index 1, got mask shape [2, 2] and indexed tensor shape [2]""",
    ):
      torch.index_put_(
          torch.tensor([0, 1], device=et.device()),
          (torch.tensor([[True, False], [False, True]], device=et.device()),),
          torch.tensor(0, device=et.device()),
      )

  def test_index_put_decompose_with_multiple_mask_error(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""The shape of the mask [3] at index 0 does not match the shape of the indexed tensor [2, 3, 5, 9] at index 2""",
        tpu="""index_put_(): the shape of the mask at index 0 must match the shape of the indexed tensor at index 2, got mask shape [3] and indexed tensor shape [2, 3, 5, 9]""",
    ):
      tensor = torch.arange(270).view(2, 3, 5, 9).to(et.device())
      boolean_mask_dim1 = tensor[0, :, 0, 0] % 2 != 0
      boolean_mask_dim3 = tensor[0, 0, 0, :] % 2 != 0
      tensor[:, :, boolean_mask_dim1, boolean_mask_dim3] = 100

  def test_index_select_index_must_be_1d(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""index_select(): Index is supposed to be a vector""",
        tpu="""index_select(): index must be 1D, got shape [2, 3]""",
    ):
      torch.index_select(
          torch.ones(2, 3, device=et.device()),
          1,
          torch.ones(2, 3, device=et.device(), dtype=torch.long),
      )

  def test_index_select_dim_out_of_bounds(self):
    with et.assert_raises_message(
        IndexError,
        cpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        # This error is generated by PyTorch and we cannot easily replace
        # it.
        tpu="""index_select(): dimension out of range (expected to be in range of [-1, 0], but got 1)""",
    ):
      torch.index_select(
          torch.ones(1, device=et.device()),
          1,
          torch.tensor([0], device=et.device(), dtype=torch.long),
      )

  def test_index_select_scalar_input(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""index_select(): dim must be 0 for scalar input, got 1""",
    ):
      torch.index_select(
          torch.tensor(1, device=et.device()),
          1,
          torch.tensor([0], device=et.device(), dtype=torch.long),
      )

  def test_index_select_scalar_index(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""index_select(): Index to scalar can have only 1 value, got 2 value(s)""",
        tpu="""index_select(): index must be 1D of size 1 for scalar input, got shape [2]""",
    ):
      torch.index_select(
          torch.tensor(1, device=et.device()),
          0,
          torch.tensor([0, 0], device=et.device(), dtype=torch.long),
      )

  def test_cumsum_with_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError if et.device().type == """tpu""" else RuntimeError,
        cpu="""Expected out tensor to have dtype c10::dummy_int1_7_t<1>, but got float instead""",
        tpu="""cumsum(): TorchTPU does not yet support dtype int1""",
        message_reviewed_by="wan",
    ):
      t = torch.ones(2, 2, device=et.device())
      output = torch.empty_like(t)
      torch.cumsum(t, dim=1, dtype=torch.int1, out=output)

  def test_cumsum_bool_out(self):
    with et.assert_raises_message(
        NotImplementedError,
        cpu=""""cumsum_out_cpu" not implemented for 'Bool'""",
        tpu="""cumsum(): invalid output dtype bool""",
    ):
      x = torch.tensor([True, False], dtype=torch.bool, device=et.device())
      torch.cumsum(x, dim=0, out=x)

  def test_cumprod_bool_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        cpu=""""cumprod_out_cpu" not implemented for 'Bool'""",
        tpu="""cumprod(): the dtype argument cannot be bool""",
        message_reviewed_by="wan",
    ):
      x = torch.tensor([1, 2], device=et.device())
      torch.cumprod(x, dim=0, dtype=torch.bool)

  def test_cumprod_bool_out(self):
    with et.assert_raises_message(
        NotImplementedError,
        cpu=""""cumprod_out_cpu" not implemented for 'Bool'""",
        tpu="""cumprod(): cumprod not implemented for bool""",
        message_reviewed_by="wan",
    ):
      x = torch.tensor([True, False], device=et.device())
      torch.cumprod(x, dim=0, out=x)

  def test_cumsum_dimension_out_of_range(self):
    with et.assert_raises_message(
        IndexError,
        cpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        # This error is generated by PyTorch and we cannot easily replace
        # it.
        tpu="""cumsum(): dimension out of range (expected to be in range of [-1, 0], but got 1)""",
    ):
      t = torch.ones(1, device=et.device())
      output = torch.empty_like(t)
      torch.cumsum(t, dim=1, out=output)

  def test_prod_out_with_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError if et.device().type == """tpu""" else RuntimeError,
        cpu="""Expected out tensor to have dtype c10::dummy_int1_7_t<1>, but got float instead""",
        tpu="""prod(): TorchTPU does not yet support dtype int1""",
        message_reviewed_by="wan",
    ):
      t = torch.ones(2, 2, device=et.device())
      output = torch.empty_like(t)
      torch.prod(t, dim=1, dtype=torch.int1, out=output)

  def test_index_add_rank_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""index_add_(): Number of indices (1) should be equal to source.size(dim): (2), for dim: 0""",
        tpu="""index_add(): self and source must have the same number of dimensions, got 2 and 1""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(2, device=et.device())
      )

  def test_index_add_index_rank_not_1(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""index_add_(): Index is supposed to be a vector, but got dim: 2 with type: Long and size: [1, 1]""",
        tpu="""index_add(): index must be 1D, got shape [1, 1]""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([[0]], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 2, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_add_dim_out_of_range(self):
    with et.assert_raises_message(
        IndexError,
        cpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        # This error is generated by PyTorch and we cannot easily replace
        # it.
        tpu="""index_add(): dimension out of range (expected to be in range of [-2, 1], but got 2)""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 2, device=et.device())
      torch.index_add(
          t, 2, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_add_source_dim_ne_index_size(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""index_add_(): Number of indices (1) should be equal to source.size(dim): (2), for dim: 0""",
        tpu="""index_add(): source must have the same number of elements as the index along dimension 0, got 2 and 1""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, 2, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_add_self_source_size_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [2, 2] source.shape = [1, 3]""",
        tpu="""index_add(): self and source must have the same size along dimension 1, got 2 and 3""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 3, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_add_scalar_dim_out_of_range(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""index_add(): dim must be 0 for scalar input, got 1""",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor(1, device=et.device())
      torch.index_add(
          t, 1, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_add_scalar_source_not_scalar(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [] source.shape = [1]""",
        tpu="""index_add(): source shape must match self shape, excluding the specified dimension, got source shape [1] and self shape []""",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor([1], device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_add_scalar_index_size_ne_1(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""Dimension specified as 0 but tensor has no dimensions""",
        tpu="""index_add(): index must be 1D of size 1 for scalar input, got shape [2]""",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0, 0], device=et.device(), dtype=torch.long)
      source = torch.tensor(1, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_addmm_input_rank_larger_than_matrices(self):
    # Arrange
    input_ = torch.ones(2, 2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)

    cpu_msg = (
        "expand(torch.FloatTensor{[2, 2, 2]}, size=[2, 2]): the number of sizes"
        " provided (2) must be greater or equal to the number of dimensions in"
        " the tensor (3)"
    )
    tpu_msg = (
        "addmm(): input tensor should not have more dimensions than the "
        "product of mat1 @ mat2, got 3-D input and 2-D product of mat1 @ mat2"
    )

    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_input_not_broadcastable_to_matmul_result(self):
    # Arrange
    input_ = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    cpu_msg = """The expanded size of the tensor (2) must match the existing size (3) at non-singleton dimension 1.  Target sizes: [2, 2].  Tensor sizes: [2, 3]"""
    tpu_msg = """addmm(): input tensor shape [2, 3] cannot be broadcasted to matmul result shape [2, 2]"""

    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_input_on_bool_tensor(self):
    # Arrange
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    beta = True
    alpha = True
    cpu_msg = """"addmm_impl_cpu_" not implemented for 'Bool'"""
    tpu_msg = """addmm(): boolean dtypes are not supported"""

    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2, beta=beta, alpha=alpha)

  def test_addmm_on_non_matrix_mat1(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    # This error is created by pytorch before our kernel is called.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""mat1 must be a matrix, got 1-D tensor""",
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_non_matrix_mat2(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, device=et.device(), dtype=torch.float32)
    # This error is created by pytorch before our kernel is called.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""mat2 must be a matrix, got 1-D tensor""",
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_mat1_mat2_mismatch_contracting_dimension(self):
    input_ = torch.ones(13, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(3, 13, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(11, 2, device=et.device(), dtype=torch.float32)
    # This error is created by pytorch before our kernel is called.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""mat1 and mat2 shapes cannot be multiplied (3x13 and 11x2)""",
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_mismatched_input_and_out_dtypes(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.int32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    tpu_msg = (
        "addmm(): input and out tensors expected to have the same dtype, got"
        " input dtype float32 and out dtype int32"
    )
    cpu_msg = "Expected out tensor to have dtype float, but got int instead"

    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2, out=out)

  def test_addmm_outdtype_must_match_out_dtype(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out_dtype = torch.int32
    tpu_msg = """addmm(): out dtype should match out_dtype, got out dtype float32 and out_dtype int32"""
    cpu_msg = re.compile(
        r"""Could not run 'aten::addmm.dtype_out' with arguments from the 'CPU' backend.*""",
        re.DOTALL,
    )

    # CPU raises NotImplementedError, a subclass of RuntimeError.
    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2, out=out, out_dtype=out_dtype)

  def test_addmm_out_dtype_unsupported(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out_dtype = torch.int1
    tpu_msg = """addmm(): TorchTPU does not yet support the output dtype int1"""
    cpu_msg = re.compile(
        r"""^Could not run 'aten::addmm.dtype' with arguments from the 'CPU' backend.*$""",
        re.DOTALL,
    )

    # CPU raises NotImplementedError, a subclass of RuntimeError.
    with et.assert_raises_message(
        RuntimeError, cpu=cpu_msg, tpu=tpu_msg, message_reviewed_by="wan"
    ):
      torch.addmm(input_, mat1, mat2, out_dtype=out_dtype)

  def test_empty_strided_size_stride_mismatch(self):
    """Tests that empty_strided fails with expected error when size and stride arrays have different lengths."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="""dimensionality of sizes (2) must match dimensionality of strides (1)""",
        tpu="""empty_strided(): the dimensionality of sizes must be the same as strides, got size [2] and stride [1]""",
    ):
      torch.empty_strided((2, 3), (1,), device=et.device(), dtype=torch.float32)

  def test_empty_strided_negative_size(self):
    """Tests that empty_strided fails with expected error when size is negative."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Trying to create tensor with negative dimension -1: [-1, 2]""",
        tpu="""empty_strided(): size must be nonnegative, got sizes [-1, 2]""",
    ):
      torch.empty_strided(
          (-1, 2), (2, 1), device=et.device(), dtype=torch.float32
      )

  def test_empty_strided_negative_stride(self):
    """Tests that empty_strided fails with expected error when stride is negative."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Storage size calculation overflowed with sizes=[2, 2] and strides=[2, -1]""",
        tpu="""empty_strided(): stride must be nonnegative, got strides [2, -1]""",
    ):
      torch.empty_strided(
          (2, 2), (2, -1), device=et.device(), dtype=torch.float32
      )

  def test_index_put_too_many_indices_after_expanding_boolean_tensors(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        tpu="""index_put_(): too many indices for tensor of dimension 2, got 3 index tensors after expanding boolean indices""",
    ):
      t = torch.zeros(10, 20, device=et.device())
      mask = torch.zeros(10, 20, dtype=torch.bool, device=et.device())
      mask[0, 0] = True
      int_index = torch.tensor([0], device=et.device())
      values = torch.tensor([5.0], device=et.device())
      # The boolean mask has dimension 2, so it expands to 2 index tensors.
      # Together with int_index, we have 3 index tensors in total, which is
      # greater than the dimension of t (2).
      t.index_put_((mask, int_index), values)

  def test_softmax_backward_data_out_unsupported_dtype(self):
    tensor = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    tensor.requires_grad_()
    tensor_int = tensor.to(torch.int32)
    dim = 1
    with et.assert_raises_message(
        NotImplementedError,
        cpu=""""softmax_lastdim_kernel_impl" not implemented for 'Int'""",
        tpu="""softmax(): not implemented for input type int32""",
    ):
      torch.nn.functional.softmax(tensor_int, dim).backward(
          torch.randn(2, 3, device=et.device())
      )

  def test_log_softmax_backward_data_out_unsupported_dtype(self):
    tensor = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    tensor.requires_grad_()
    tensor_int = tensor.to(torch.int32)
    dim = 1
    with et.assert_raises_message(
        NotImplementedError,
        cpu=""""log_softmax_lastdim_kernel_impl" not implemented for 'Int'""",
        tpu="""log_softmax(): not implemented for input type int32""",
    ):
      torch.nn.functional.log_softmax(tensor_int, dim).backward(
          torch.randn(2, 3, device=et.device())
      )

  def test_trunc_unsupported_boolean_dtype(self):
    t = torch.tensor([True, False], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""trunc(): does not support boolean types""",
        cpu=""""trunc_vml_cpu" not implemented for 'Bool'""",
    ):
      torch.trunc(t)

  def test_digamma_unsupported_complex(self):
    t = torch.tensor([1 + 1j], device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""digamma(): expected the input dtype not to be complex, got complex64""",
        cpu=""""digamma" not implemented for 'ComplexFloat'""",
    ):
      torch.digamma(t)

  @parameterized.named_parameters(
      ("bool", torch.bool, "bool", "Bool"),
      ("int64", torch.int64, "int64", "Long"),
  )
  def test_elu_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype_str: str, cpu_dtype_str: str
  ):
    inp = torch.ones(4, device=et.device(), dtype=dtype)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""elu(): expected the input dtype to be floating point, got"""
        f""" {tpu_dtype_str}""",
        cpu=f""""elu_cpu" not implemented for '{cpu_dtype_str}'""",
    ):
      torch.nn.functional.elu(inp)

  def test_gelu_unsupported_input_dtype(self):
    del self  # self is not used in this test
    t = torch.ones(2, device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""gelu(): unsupported input dtype: 'complex64'""",
        cpu=""""GeluKernelImpl" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.gelu(t)

  def test_gelu_unsupported_output_dtype(self):
    del self  # self is not used in this test
    t = torch.ones(2, device=et.device(), dtype=torch.float32)
    out = torch.ones(2, device=et.device(), dtype=torch.int64)
    with et.assert_raises_message(
        Exception,
        tpu="""gelu(): unsupported output dtype: 'int64'""",
        cpu="""Found dtype Long but expected Float""",
    ):
      torch.nn.functional.gelu(t, out=out)

  def test_gelu_unsupported_approximation_type(self):
    t = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gelu(): unsupported approximate argument: invalid""",
        cpu="""approximate argument must be either none or tanh.""",
    ):
      torch.nn.functional.gelu(t, approximate="invalid")

  def test_gelu_backward_grad_input_unsupported_approximation_type(self):
    t = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    grad_input = torch.empty_like(t)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gelu_backward(): unsupported approximate argument: invalid""",
        cpu="""approximate argument must be either none or tanh.""",
    ):
      torch.ops.aten.gelu_backward.grad_input(
          t, t, approximate="invalid", grad_input=grad_input
      )

  def test_glu_unsupported_input_dtype(self):
    t = torch.ones(2, 4, device=et.device(), dtype=torch.int32)
    out = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu(): expected the self dtype to be floating point, got int32""",
        cpu=""""glu_cpu" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=1, out=out)

  def test_glu_unsupported_out_dtype(self):
    t = torch.ones(2, 4, device=et.device(), dtype=torch.float32)
    out = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu(): expected the out dtype to be floating point, got int32""",
        cpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=1, out=out)

  def test_glu_invalid_rank(self):
    t = torch.tensor(0.0, device=et.device(), dtype=torch.float32)
    out = torch.tensor(0.0, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu(): expected input tensor to have at least 1 dimension, got 0 dimensions""",
        cpu="""glu does not support 0-dimensional tensors""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=0, out=out)

  def test_glu_invalid_dim(self):
    t = torch.ones(2, 4, device=et.device(), dtype=torch.float32)
    out = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        tpu="""glu(): dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        cpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=2, out=out)

  def test_glu_invalid_dim_size(self):
    t = torch.ones(2, 5, device=et.device(), dtype=torch.float32)
    out = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu(): expected the size of dimension 1 to be even, got 5""",
        cpu="""Halving dimension must be even, but dimension 1 is size 5""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=1, out=out)

  def test_glu_backward_unsupported_dtypes(self):
    float_tensor = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    int_tensor = torch.ones(2, 4, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected the self dtype to be floating point, got int32""",
        cpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(
          grad_output=float_tensor, self=int_tensor, dim=1
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected the grad_output dtype to be floating point, got int32""",
        cpu="""Found dtype Int but expected Float""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(
          grad_output=int_tensor, self=float_tensor, dim=1
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected the grad_input dtype to be floating point, got int32""",
        cpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward.grad_input(
          grad_output=float_tensor,
          self=float_tensor,
          dim=1,
          grad_input=int_tensor,
      )

  def test_glu_backward_dtype_mismatch(self):
    float32_tensor = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    float64_tensor = torch.ones(2, 4, device=et.device(), dtype=torch.float64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected self and grad_output to have the same dtype, got float32 and float64""",
        cpu="""Found dtype Double but expected Float""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(
          grad_output=float64_tensor, self=float32_tensor, dim=1
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected self and grad_input to have the same dtype, got float32 and float64""",
        cpu="""Found dtype Float but expected Double""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward.grad_input(
          grad_output=float32_tensor,
          self=float32_tensor,
          dim=1,
          grad_input=float64_tensor,
      )

  def test_glu_backward_zero_rank(self):
    grad_output = torch.tensor(0.0, device=et.device(), dtype=torch.float32)
    self_tensor = torch.tensor(0.0, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected self to have at least 1 dimension, got 0""",
        cpu="""glu does not support 0-dimensional tensors""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(grad_output, self_tensor, dim=0)

  def test_glu_backward_invalid_dim_size(self):
    grad_output = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    self_tensor = torch.ones(2, 5, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected the size of dimension 1 of self to be even, got 5""",
        cpu="""Halving dimension must be even, but dimension 1 is size 5""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(grad_output, self_tensor, dim=1)

  def test_glu_backward_grad_output_shape_mismatch(self):
    grad_output = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    self_tensor = torch.ones(2, 4, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected grad_output shape to be [2, 2], got [2, 3]""",
        cpu="""The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(grad_output, self_tensor, dim=1)

  def test_group_norm_backward_grad_out_numel_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Expected dY.numel() == N * C * HxW to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        tpu="""native_group_norm_backward(): expected grad_out to have 18 elements, got 24""",
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 4, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(6, device=et.device()),
          1,
          6,
          3,
          2,
          [True, True, True],
      )

  def test_group_norm_backward_input_numel_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Expected X.numel() == N * C * HxW to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        tpu="""native_group_norm_backward(): expected input to have 24 elements, got 18""",
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 4, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(6, device=et.device()),
          1,
          6,
          4,
          2,
          [True, True, True],
      )

  def test_group_norm_backward_mean_numel_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Expected mean.numel() == N * group to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        tpu="""native_group_norm_backward(): expected mean to have shape [1, 2], got [1, 3]""",
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(6, device=et.device()),
          1,
          6,
          3,
          2,
          [True, True, True],
      )

  def test_group_norm_backward_mean_rstd_shape_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Expected rstd.numel() == N * group to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        tpu="""native_group_norm_backward(): expected mean and rstd to have the same shape, got mean size [1, 2] and rstd size [1, 3]""",
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(1, 3, device=et.device()),
          torch.ones(6, device=et.device()),
          1,
          6,
          3,
          2,
          [True, True, True],
      )

  def test_group_norm_backward_weight_numel_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="""Expected !gamma.defined() || gamma.numel() == C to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        tpu="""native_group_norm_backward(): expected weight to have 6 elements, got 5""",
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(5, device=et.device()),
          1,
          6,
          3,
          2,
          [True, True, True],
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_lhs_shape",
          lhs_arg=(2,),
          rhs_arg=(2, 2),
          group_sizes_arg=[1, 1],
          expected_error="ragged_dot(): lhs must be 2D, got dim: 1",
      ),
      dict(
          testcase_name="invalid_rhs_shape",
          lhs_arg=(3, 2),
          rhs_arg=(2, 2),
          group_sizes_arg=[1, 1],
          expected_error="ragged_dot(): rhs must be 3D, got dim: 2",
      ),
      dict(
          testcase_name="invalid_group_sizes_shape",
          lhs_arg=(3, 2),
          rhs_arg=(2, 2, 2),
          group_sizes_arg=[[1, 1]],
          expected_error="ragged_dot(): group_sizes must be 1D, got dim: 2",
      ),
      dict(
          testcase_name="mismatched_contracting_dims",
          lhs_arg=(3, 2),
          rhs_arg=(2, 3, 2),
          group_sizes_arg=[1, 1],
          expected_error=(
              "ragged_dot(): contracting dimension should be the same, got: 2"
              " vs 3"
          ),
      ),
      dict(
          testcase_name="mismatched_group_dims",
          lhs_arg=(3, 2),
          rhs_arg=(2, 2, 2),
          group_sizes_arg=[1, 1, 1],
          expected_error=(
              "ragged_dot(): lhs and group_sizes should have the same number of"
              " groups, got: 2 vs 3"
          ),
      ),
  )
  def test_ragged_dot(self, lhs_arg, rhs_arg, group_sizes_arg, expected_error):
    """Tests that torch_tpu.ragged_dot fails with expected errors.

    This test checks various error conditions for the `torch_tpu.ragged_dot`
    operation, such as invalid input shapes, mismatched dimensions, and
    incorrect group sizes.

    Args:
      lhs_arg: The shape argument for the left-hand side tensor.
      rhs_arg: The shape argument for the right-hand side tensor.
      group_sizes_arg: The argument for the group sizes tensor.
      expected_error: The expected error message substring.
    """
    lhs = torch.ones(*lhs_arg, dtype=torch.float32, device=et.device())
    rhs = torch.ones(*rhs_arg, dtype=torch.float32, device=et.device())
    group_sizes = torch.tensor(
        group_sizes_arg, dtype=torch.int32, device=et.device()
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu=expected_error,
        cpu=expected_error,
    ):
      torch.ops.torch_tpu.ragged_dot(lhs, rhs, group_sizes)

  def test_max_pool2d_unsupported_dtypes(self):
    t_bool = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""max_pool2d(): bool dtype is not supported""",
        cpu=""""max_pool2d" not implemented for 'Bool'""",
    ):
      torch.nn.functional.max_pool2d(t_bool, kernel_size=3)

  def test_masked_scatter_invalid_mask_dtype(self):
    device = et.device()
    t = torch.randn(4, 4, device=device, dtype=torch.float32)
    source = torch.randn(16, device=device, dtype=torch.float32)
    mask_int = torch.ones(4, 4, device=device, dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        cpu="""masked_scatter_ only supports boolean masks, but got mask with dtype Int""",
        tpu="""masked_scatter_(): expected Boolean tensor for mask, got int32""",
    ):
      torch.masked_scatter(t, mask_int, source)

  def test_masked_scatter_dtype_mismatch(self):
    device = et.device()
    t = torch.randn(4, 4, device=device, dtype=torch.float32)
    mask = torch.ones(4, 4, device=device, dtype=torch.bool)
    source = torch.randint(0, 10, (16,), device=device, dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        cpu="""masked_scatter: expected self and source to have same dtypes but gotFloat and Int""",
        tpu="""masked_scatter_(): expected same dtype for self and source, got self dtype float32 and source dtype int32""",
    ):
      torch.masked_scatter(t, mask, source)

  def test_arange_zero_step(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): step must be non-zero""",
        cpu="""step must be nonzero""",
    ):
      torch.arange(1, 10, 0, device=et.device())

  def test_linspace_negative_steps(self):
    out = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linspace(): expected non-negative steps, got -1""",
        cpu="""number of steps must be non-negative""",
    ):
      torch.linspace(0, 10, -1, device=et.device(), out=out)

  def test_linspace_bool_error(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""linspace(): expected output dtype to be other than bool, got bool""",
        cpu=""""linspace_cpu" not implemented for 'Bool'""",
    ):
      out = torch.empty(5, dtype=torch.bool, device=et.device())
      torch.linspace(0, 10, 5, out=out)

  def test_arange_infinite_inputs(self):
    """Tests that arange fails on infinite inputs with expected error.

    The test for when `step` is infinite is skipped because PyTorch CPU does not
    error in this case. Such a test is in
    `TpuErrorsTest.test_arange_infinite_step`.
    """

    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected [start, end) interval to have finite bounds, got [inf, 0)""",
        cpu="""unsupported range: inf -> 0""",
    ):
      torch.arange(float("inf"), 0, -1, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected [start, end) interval to have finite bounds, got [0, inf)""",
        cpu="""unsupported range: 0 -> inf""",
    ):
      torch.arange(0, float("inf"), 1, device=et.device())

  def test_arange_invalid_inputs(self):
    """Tests that arange fails on invalid sets of inputs with expected error.

    Tests whether the given `step` is valid for the given interval [start, end].
    i.e. `step` should be negative iff `start` > `end`, but positive iff
    `start` < `end`.
    """

    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected step to be positive since start (0) < end (10), got step=-1""",
        cpu="""upper bound and lower bound inconsistent with step sign""",
    ):
      torch.arange(0, 10, -1, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected step to be negative since start (10) > end (0), got step=1""",
        cpu="""upper bound and lower bound inconsistent with step sign""",
    ):
      torch.arange(10, 0, 1, device=et.device())

  def test_arange_invalid_inputs_with_infinite_step(self):
    """Tests that arange fails on invalid sets of inputs with infinite step.

    Tests whether the given `step` (infinite) sign is valid for the given
    interval [start, end]. i.e. `step` should be negative infinity iff `start` >
    `end`, but positive infinity iff `start` < `end`.
    """

    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected step to be positive since start (0) < end (10), got step=-inf""",
        cpu="""upper bound and lower bound inconsistent with step sign""",
    ):
      torch.arange(0, 10, float("-inf"), device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected step to be negative since start (10) > end (0), got step=inf""",
        cpu="""upper bound and lower bound inconsistent with step sign""",
    ):
      torch.arange(10, 0, float("inf"), device=et.device())

  def test_max_pool3d_unsupported_dtypes(self):
    t_bool = torch.zeros((1, 1, 4, 4, 4), device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""max_pool3d_with_indices(): bool dtype is not supported""",
        cpu=""""max_pool3d" not implemented for 'Bool'""",
    ):
      torch.nn.functional.max_pool3d(t_bool, kernel_size=3)

  @parameterized.named_parameters(
      ("bfloat16", torch.bfloat16, "bfloat16", "BFloat16"),
      ("float16", torch.float16, "float16", "Half"),
  )
  def test_cdist_forward_unsupported_floating_point_dtypes(
      self, dtype: torch.dtype, tpu_dtype_str: str, cpu_dtype_str: str
  ):
    # Starting with 2 `float32` tensors, this test runs `cdist()` twice for each
    # unsupported dtype:
    #
    #   1. Casting the first argument
    #   2. Casting the second argument
    #
    # Note that the CPU error message is different for the both cases mentioned
    # above.

    x1 = torch.randn(2, 2, device=et.device())
    x2 = torch.randn(2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""cdist_forward(): expected the first argument's dtype not to be"""
        f""" bfloat16 or float16, got {tpu_dtype_str}""",
        cpu=f""""cdist" not implemented for '{cpu_dtype_str}'""",
        message_reviewed_by="wan",
    ):
      torch.cdist(x1.to(dtype), x2, p=1.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""cdist_forward(): expected the second argument's dtype not to be"""
        f""" bfloat16 or float16, got {tpu_dtype_str}""",
        cpu=f"""expected scalar type Float but found {cpu_dtype_str}""",
        message_reviewed_by="wan",
    ):
      torch.cdist(x1, x2.to(dtype), p=1.0)

  def test_cdist_forward_int32(self):
    # Starting with 2 `float32` tensors, this test runs `cdist()` twice:
    #
    #   1. Casting the first argument
    #   2. Casting the second argument

    x1 = torch.randn(2, 2, device=et.device())
    x2 = torch.randn(2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""cdist_forward(): expected the first argument's dtype to be floating point, got int32""",
        cpu="""cdist only supports floating-point dtypes, X1 got: Int""",
        message_reviewed_by="wan",
    ):
      torch.cdist(x1.to(torch.int32), x2, p=1.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""cdist_forward(): expected the second argument's dtype to be floating point, got int32""",
        cpu="""cdist only supports floating-point dtypes, X2 got: Int""",
        message_reviewed_by="wan",
    ):
      torch.cdist(x1, x2.to(torch.int32), p=1.0)

  def test_cdist_forward_unsupported_p(self):
    x1 = torch.randn(2, 2, device=et.device(), dtype=torch.float32)
    x2 = torch.randn(2, 2, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""cdist_forward(): expected the p value to be >= 0, got -1""",
        cpu="""cdist only supports non-negative p values""",
    ):
      torch.cdist(x1, x2, p=-1.0)

  @parameterized.named_parameters(
      ("bfloat16", torch.bfloat16, "bfloat16", "BFloat16"),
      ("float16", torch.float16, "float16", "Half"),
  )
  def test_cdist_backward_unsupported_floating_point_dtypes(
      self, dtype: torch.dtype, tpu_dtype_str: str, cpu_dtype_str: str
  ):
    grad = torch.randn(2, 2, device=et.device(), dtype=dtype)
    x1 = torch.randn(2, 2, device=et.device(), dtype=dtype)
    x2 = torch.randn(2, 2, device=et.device(), dtype=dtype)
    cdist = torch.randn(2, 2, device=et.device(), dtype=dtype)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""cdist_backward(): expected the first argument's dtype not to be bfloat16 or float16, got {tpu_dtype_str}""",
        cpu=f""""cdist_backward" not implemented for '{cpu_dtype_str}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._cdist_backward(grad, x1, x2, 1.0, cdist)

  def test_cdist_backward_int32(self):
    grad = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    x1 = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    x2 = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    cdist = torch.ones(2, 2, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""cdist_backward(): expected the first argument's dtype to be floating point, got int32""",
        cpu=""""cdist_backward" not implemented for 'Int'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._cdist_backward(grad, x1, x2, 1.0, cdist)

  def test_exponential_unsupported_dtypes(self):
    device = et.device()
    t_int = torch.ones((2, 2), device=device, dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""exponential_(): expected input tensor dtype to be a floating-point real type, got int32""",
        cpu="""Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified Int""",
    ):
      t_int.exponential_()

    t_complex = torch.ones((2, 2), device=device, dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""exponential_(): expected input tensor dtype to be a floating-point real type, got complex64""",
        cpu="""Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified ComplexFloat""",
    ):
      t_complex.exponential_()

  def test_bernoulli_invalid_p(self):
    device = et.device()
    t = torch.ones((2, 2), device=device, dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""bernoulli_(): expected p to be in the range [0, 1], got 1.5""",
        cpu="""bernoulli_ expects p to be in [0, 1], but got p=1.5""",
    ):
      t.bernoulli(p=1.5)

  def test_add_smaller_out_alias(self):
    """Tests that add fails when the out tensor is a smaller alias of an input."""
    a = torch.ones(4, device=et.device())
    b = torch.ones(4, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""add(): output with shape [4] doesn't match the broadcast shape of the tensor being operated on in-place, which has shape [1]""",
        cpu="""unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.""",
    ):
      torch.add(a, b, out=a[1:2])

  def test_avg_pool2d_unsupported_dtypes(self):
    t_complex = torch.zeros(
        (1, 1, 4, 4), device=et.device(), dtype=torch.complex64
    )
    t_uint8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.uint8)
    t_int8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int8)
    t_int16 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int16)
    t_int32 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool2d(): not yet implemented for uint8, int8, int16, int32, and complex64 dtypes, got complex64""",
        cpu=""""avg_pool2d" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.avg_pool2d(t_complex, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool2d(): not yet implemented for uint8, int8, int16, int32, and complex64 dtypes, got uint8""",
        cpu=""""avg_pool2d" not implemented for 'Byte'""",
    ):
      torch.nn.functional.avg_pool2d(t_uint8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool2d(): not yet implemented for uint8, int8, int16, int32, and complex64 dtypes, got int8""",
        cpu=""""avg_pool2d" not implemented for 'Char'""",
    ):
      torch.nn.functional.avg_pool2d(t_int8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool2d(): not yet implemented for uint8, int8, int16, int32, and complex64 dtypes, got int16""",
        cpu=""""avg_pool2d" not implemented for 'Short'""",
    ):
      torch.nn.functional.avg_pool2d(t_int16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool2d(): not yet implemented for uint8, int8, int16, int32, and complex64 dtypes, got int32""",
        cpu=""""avg_pool2d" not implemented for 'Int'""",
    ):
      torch.nn.functional.avg_pool2d(t_int32, kernel_size=3)

  def test_avg_pool3d_unsupported_dtypes(self):
    t_bool = torch.zeros((1, 1, 4, 4, 4), device=et.device(), dtype=torch.bool)
    t_bf16 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.bfloat16
    )
    t_f16 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.float16
    )
    t_complex = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.complex64
    )
    t_uint8 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.uint8
    )
    t_int8 = torch.zeros((1, 1, 4, 4, 4), device=et.device(), dtype=torch.int8)
    t_int16 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.int16
    )
    t_int32 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.int32
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): not yet implemented for bool, bfloat16, float16, uint8, int8, int16, int32, and complex64 dtypes, got bool""",
        cpu=""""avg_pool3d_out_frame" not implemented for 'Bool'""",
    ):
      torch.nn.functional.avg_pool3d(t_bool, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): not yet implemented for bool, bfloat16, float16, uint8, int8, int16, int32, and complex64 dtypes, got bfloat16""",
        cpu=""""avg_pool3d_out_frame" not implemented for 'BFloat16'""",
    ):
      torch.nn.functional.avg_pool3d(t_bf16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): not yet implemented for bool, bfloat16, float16, uint8, int8, int16, int32, and complex64 dtypes, got float16""",
        cpu=""""avg_pool3d_out_frame" not implemented for 'Half'""",
    ):
      torch.nn.functional.avg_pool3d(t_f16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): not yet implemented for bool, bfloat16, float16, uint8, int8, int16, int32, and complex64 dtypes, got complex64""",
        cpu=""""avg_pool3d_out_frame" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.avg_pool3d(t_complex, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): not yet implemented for bool, bfloat16, float16, uint8, int8, int16, int32, and complex64 dtypes, got uint8""",
        cpu=""""avg_pool3d_out_frame" not implemented for 'Byte'""",
    ):
      torch.nn.functional.avg_pool3d(t_uint8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): not yet implemented for bool, bfloat16, float16, uint8, int8, int16, int32, and complex64 dtypes, got int8""",
        cpu=""""avg_pool3d_out_frame" not implemented for 'Char'""",
    ):
      torch.nn.functional.avg_pool3d(t_int8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): not yet implemented for bool, bfloat16, float16, uint8, int8, int16, int32, and complex64 dtypes, got int16""",
        cpu=""""avg_pool3d_out_frame" not implemented for 'Short'""",
    ):
      torch.nn.functional.avg_pool3d(t_int16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): not yet implemented for bool, bfloat16, float16, uint8, int8, int16, int32, and complex64 dtypes, got int32""",
        cpu=""""avg_pool3d_out_frame" not implemented for 'Int'""",
    ):
      torch.nn.functional.avg_pool3d(t_int32, kernel_size=3)

  @parameterized.named_parameters(
      ("bfloat16", torch.bfloat16, "bfloat16", "BFloat16"),
      ("float16", torch.float16, "float16", "Half"),
  )
  def test_pdist_forward_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype_str: str, cpu_dtype_str: str
  ):
    inp = torch.randn(2, 2, device=et.device(), dtype=dtype)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""pdist_forward(): expected the input dtype not to be bfloat16 or"""
        f""" float16, got {tpu_dtype_str}""",
        cpu=f""""pdist" not implemented for '{cpu_dtype_str}'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.pdist(inp, p=2.0)

  @parameterized.named_parameters(
      ("int32", torch.int32, "int32", "Int"),
      ("bfloat16", torch.bfloat16, "bfloat16", "BFloat16"),
      ("float16", torch.float16, "float16", "Half"),
  )
  def test_pdist_backward_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype_str: str, cpu_dtype_str: str
  ):
    grad = torch.randn(1, device=et.device())
    self_tensor = (
        torch.ones(2, 2, device=et.device(), dtype=dtype)
        if dtype == torch.int32
        else torch.randn(2, 2, device=et.device(), dtype=dtype)
    )
    pdist = torch.randn(1, device=et.device())

    if dtype in (torch.bfloat16, torch.float16):
      expected_tpu_msg = (
          "pdist_backward(): expected the input dtype not to be bfloat16 or"
          f" float16, got {tpu_dtype_str}"
      )
    else:
      expected_tpu_msg = (
          "pdist_backward(): expected the input dtype to be floating point,"
          f" got {tpu_dtype_str}"
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu=expected_tpu_msg,
        cpu=f""""pdist_backward" not implemented for '{cpu_dtype_str}'""",
    ):
      torch.ops.aten._pdist_backward(grad, self_tensor, 2.0, pdist)

  # TODO(lwh): fix this test once G3 pytorch version is updated
  @unittest.skip("Disabled due to pytorch version mismatch")
  def test_replication_pad_backward(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Mismatch in shape: grad_output[0] has a shape of torch.Size([1]) and output[0] has a shape of torch.Size([1, 6, 4]).""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0],
          mode="replicate",
      ).backward(torch.randn(1, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Mismatch in shape: grad_output[0] has a shape of torch.Size([1, 2, 3, 4]) and output[0] has a shape of torch.Size([1, 6, 4]).""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 2, 3, 4, device=et.device()))
    # Incorrectly sized padding input passes forward pass and fails with a
    # pytorch assertion error on the backward pass.
    with et.assert_raises_message(
        AssertionError,
        tpu="",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0, 0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    # Empty padding input
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Mismatch in shape: grad_output[0] has a shape of torch.Size([1, 6, 4]) and output[0] has a shape of torch.Size([1, 6, 13]).""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[4, 5],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        tpu="""input (W: 4) is too small. Calculated output W: -2""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[-1, -5],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

  def test_replication_pad_backward_unsupported_dtypes(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad1d(): not implemented for 'Bool'""",
        cpu=""""replication_pad1d" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad2d(): not implemented for 'Bool'""",
        cpu=""""replication_pad2d" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad3d(): not implemented for 'Bool'""",
        cpu=""""replication_pad3d" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0, 0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, 4, 4, device=et.device()))

  def test_reflection_pad_backward(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Mismatch in shape: grad_output[0] has a shape of torch.Size([1]) and output[0] has a shape of torch.Size([1, 6, 4]).""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0],
          mode="reflect",
      ).backward(torch.randn(1, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Mismatch in shape: grad_output[0] has a shape of torch.Size([1, 2, 3, 4]) and output[0] has a shape of torch.Size([1, 6, 4]).""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 2, 3, 4, device=et.device()))
    # Incorrectly sized padding input passes forward pass and fails with a
    # pytorch assertion error on the backward pass.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""element 0 of tensors does not require grad and does not have a grad_fn""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0, 0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    # Empty padding input
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Padding size 0 is not supported for 3D input tensor.
Supported combinations for non-constant padding:
  - 2D or 3D input: padding size = 2 (pads last dimension)
  - 3D or 4D input: padding size = 4 (pads last 2 dimensions)
  - 4D or 5D input: padding size = 6 (pads last 3 dimensions)""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (4, 5) at dimension 2 of input [1, 6, 4]""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[4, 5],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        tpu="""input (W: 4) is too small. Calculated output W: -2""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[-1, -5],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

  def test_reflection_pad_backward_unsupported_dtypes(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""reflection_pad1d(): not implemented for bool""",
        cpu=""""reflection_pad1d" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""reflection_pad2d(): not implemented for bool""",
        cpu=""""reflection_pad2d" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""reflection_pad3d(): not implemented for bool""",
        cpu=""""reflection_pad3d" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0, 0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, 4, 4, device=et.device()))

  def test_adaptive_avg_pool2d_unsupported_dtypes(self):
    t_complex = torch.zeros(
        (1, 1, 4, 4), device=et.device(), dtype=torch.complex64
    )
    t_uint8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.uint8)
    t_int8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int8)
    t_int16 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int16)
    t_int32 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int32)
    t_int64 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): not yet implemented for uint8, int8, int16, int32, int64, and complex64 dtypes, got complex64""",
        cpu=""""adaptive_avg_pool2d" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_complex, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): not yet implemented for uint8, int8, int16, int32, int64, and complex64 dtypes, got uint8""",
        cpu=""""adaptive_avg_pool2d" not implemented for 'Byte'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_uint8, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): not yet implemented for uint8, int8, int16, int32, int64, and complex64 dtypes, got int8""",
        cpu=""""adaptive_avg_pool2d" not implemented for 'Char'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int8, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): not yet implemented for uint8, int8, int16, int32, int64, and complex64 dtypes, got int16""",
        cpu=""""adaptive_avg_pool2d" not implemented for 'Short'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int16, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): not yet implemented for uint8, int8, int16, int32, int64, and complex64 dtypes, got int32""",
        cpu=""""adaptive_avg_pool2d" not implemented for 'Int'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int32, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): not yet implemented for uint8, int8, int16, int32, int64, and complex64 dtypes, got int64""",
        cpu=""""adaptive_avg_pool2d" not implemented for 'Long'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int64, output_size=2)

  def test_adaptive_avg_pool3d_unsupported_dtypes(self):
    t_complex = torch.zeros(
        (1, 1, 4, 4), device=et.device(), dtype=torch.complex64
    )
    t_uint8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.uint8)
    t_int8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int8)
    t_int16 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int16)
    t_int32 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int32)
    t_int64 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int64)
    t_bool = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8, int16, int32, int64, and complex64 dtypes, got bool""",
        cpu=""""adaptive_avg_pool3d_cpu" not implemented for 'Bool'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_bool, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8, int16, int32, int64, and complex64 dtypes, got complex64""",
        cpu=""""adaptive_avg_pool3d_cpu" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_complex, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8, int16, int32, int64, and complex64 dtypes, got uint8""",
        cpu=""""adaptive_avg_pool3d_cpu" not implemented for 'Byte'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_uint8, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8, int16, int32, int64, and complex64 dtypes, got int8""",
        cpu=""""adaptive_avg_pool3d_cpu" not implemented for 'Char'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int8, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8, int16, int32, int64, and complex64 dtypes, got int16""",
        cpu=""""adaptive_avg_pool3d_cpu" not implemented for 'Short'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int16, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8, int16, int32, int64, and complex64 dtypes, got int32""",
        cpu=""""adaptive_avg_pool3d_cpu" not implemented for 'Int'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int32, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8, int16, int32, int64, and complex64 dtypes, got int64""",
        cpu=""""adaptive_avg_pool3d_cpu" not implemented for 'Long'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int64, output_size=3)

  def test_floor_divide_complex64(self):
    lhs = torch.arange(5, device=et.device())
    rhs = torch.arange(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""floor_divide(): expected dtype of the first argument to be neither complex nor bool, got complex64""",
        cpu=""""div_floor_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.floor_divide(lhs.to(torch.complex64), rhs)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""floor_divide(): expected dtype of the second argument to be neither complex nor bool, got complex64""",
        cpu=""""div_floor_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.floor_divide(lhs, rhs.to(torch.complex64))

  def test_atan2_complex(self):
    x = torch.tensor([1.0, 2.0], device=et.device())
    y = torch.tensor([1.0, 2.0], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""atan2(): expected the dtype of the first argument not to be complex, got complex64""",
        cpu=""""atan2_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.atan2(x.to(torch.complex64), y)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""atan2(): expected the dtype of the second argument not to be complex, got complex64""",
        cpu=""""atan2_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.atan2(x, y.to(torch.complex64))

  @parameterized.named_parameters(
      {
          "testcase_name": "bitwise_and",
          "op_name": "bitwise_and",
          "op": torch.bitwise_and,
      },
      {
          "testcase_name": "bitwise_or",
          "op_name": "bitwise_or",
          "op": torch.bitwise_or,
      },
      {
          "testcase_name": "bitwise_xor",
          "op_name": "bitwise_xor",
          "op": torch.bitwise_xor,
      },
  )
  def test_bitwise_ops_float64(self, op_name: str, op: Any):
    x = torch.ones(5, dtype=torch.int64, device=et.device())
    y = torch.ones(5, dtype=torch.int64, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the dtype of the first argument to be neither floating-point nor complex, got float64""",
        cpu=f""""{op_name}_cpu" not implemented for 'Double'""",
        message_reviewed_by="wan",
    ):
      op(x.to(torch.float64), y)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the dtype of the second argument to be neither floating-point nor complex, got float64""",
        cpu=f""""{op_name}_cpu" not implemented for 'Double'""",
        message_reviewed_by="wan",
    ):
      op(x, y.to(torch.float64))

  @parameterized.named_parameters(
      {
          "testcase_name": "bitwise_left_shift",
          "op_name_tpu": "bitwise_left_shift",
          "op_name_cpu": "lshift",
          "op": torch.bitwise_left_shift,
      },
      {
          "testcase_name": "bitwise_right_shift",
          "op_name_tpu": "bitwise_right_shift",
          "op_name_cpu": "rshift",
          "op": torch.bitwise_right_shift,
      },
  )
  def test_bitwise_shift_float64(
      self, op_name_tpu: str, op_name_cpu: str, op: Any
  ):
    x = torch.ones(5, dtype=torch.int64, device=et.device())
    y = torch.ones(5, dtype=torch.int64, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name_tpu}(): expected the dtype of the first argument to be integer, got float64""",
        cpu=f""""{op_name_cpu}_cpu" not implemented for 'Double'""",
        message_reviewed_by="wan",
    ):
      op(x.to(torch.float64), y)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name_tpu}(): expected the dtype of the second argument to be integer, got float64""",
        cpu=f""""{op_name_cpu}_cpu" not implemented for 'Double'""",
        message_reviewed_by="wan",
    ):
      op(x, y.to(torch.float64))

  def test_bitwise_left_shift_scalar_float(self):
    x = torch.ones(5, dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""__lshift__(): expected the dtype of the second argument to be integer, got float64""",
        cpu=""""lshift_cpu" not implemented for 'Float'""",
        message_reviewed_by="wan",
    ):
      x.__lshift__(1.5)

  def test_col2im_output_size_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected output_size to have 2 dimensions, got 3""",
        cpu="""It is expected output_size equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_kernel_size_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected kernel_size to have 2 dimensions, got 3""",
        cpu="""It is expected kernel_size equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_dilation_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected dilation to have 2 dimensions, got 3""",
        cpu="""It is expected dilation equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1, 1), (0, 0), (1, 1))

  def test_col2im_padding_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected padding to have 2 dimensions, got 3""",
        cpu="""It is expected padding equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0, 0), (1, 1))

  def test_col2im_stride_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected stride to have 2 dimensions, got 3""",
        cpu="""It is expected stride equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1, 1))

  def test_col2im_input_must_be_3d(self):
    img = torch.randn(1, 4, 16, 1, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected input to have 3 dimensions (batch, channels, length), got 4""",
        cpu="""Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch size and non-zero dimensions for input, but got: [1, 4, 16, 1]""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_kernel_size_must_be_positive(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected kernel size to be positive, got 0""",
        cpu="""kernel size should be greater than zero, but got kernel_height: 0 kernel_width: 2""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (0, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_channels_divisibility(self):
    img = torch.randn(1, 5, 15, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected input channels to be divisible by kernel product (4), got 5""",
        cpu="""Expected size of input's dimension 1 to be divisible by the product of kernel_size, but got input.size(1)=5 and kernel_size=(2, 2).""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_invalid_stride(self):
    img = torch.randn(1, 4, 16, device=et.device())

    output_size = (5, 5)
    kernel_size = (2, 2)
    dilation = (1, 1)
    padding = (0, 0)

    # Check 0 stride.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected all stride elements to be positive, got [0, 1]""",
        cpu="""stride should be greater than zero, but got stride_height: 0 stride_width: 1""",
        message_reviewed_by="wan",
    ):
      stride = (0, 1)
      torch.ops.aten.col2im(
          img, output_size, kernel_size, dilation, padding, stride
      )

    # Check negative stride.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected all stride elements to be positive, got [1, -1]""",
        cpu="""stride should be greater than zero, but got stride_height: 1 stride_width: -1""",
        message_reviewed_by="wan",
    ):
      stride = (1, -1)
      torch.ops.aten.col2im(
          img, output_size, kernel_size, dilation, padding, stride
      )

  def test_col2im_length_mismatch(self):
    # output=(5,5), k=(2,2), stride=(1,1), pad=(0,0)
    # -> col_h, col_w = (4, 4) -> L=16
    img = torch.randn(1, 4, 15, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected input length to be divisible by col size (4 * 4 = 16), got 15""",
        cpu="""Given output_size=(5, 5), kernel_size=(2, 2), dilation=(1, 1), padding=(0, 0), stride=(1, 1), expected size of input's dimension 2 to match the calculated number of sliding blocks 4 * 4 = 16, but got input.size(2)=15.""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  @parameterized.named_parameters(
      {"testcase_name": "ge", "op_name": "ge", "op": torch.ge},
      {"testcase_name": "gt", "op_name": "gt", "op": torch.gt},
      {"testcase_name": "le", "op_name": "le", "op": torch.le},
      {"testcase_name": "lt", "op_name": "lt", "op": torch.lt},
  )
  def test_comparison_ops_complex(self, op_name: str, op: Any):
    lhs = torch.tensor([1.0, 2.0], device=et.device())
    rhs = torch.tensor([1.0, 2.0], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the dtype of the first argument not to be complex, got complex64""",
        cpu=f""""{op_name}_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      op(lhs.to(torch.complex64), rhs)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the dtype of the second argument not to be complex, got complex64""",
        cpu=f""""{op_name}_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      op(lhs, rhs.to(torch.complex64))

    # TODO: b/478955517 dtype checks should run after dtype promotion.
    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the dtype of the second argument not to be complex, got complex128""",
        cpu=f""""{op_name}_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      op(lhs, 1j)

  def test_remainder_complex(self):
    c64 = torch.tensor([1 + 1j], device=et.device(), dtype=torch.complex64)
    f32 = torch.tensor([1.0], device=et.device())

    test_cases = [
        (1, c64),
        (1j, f32),
        (c64, f32),
    ]

    for args in test_cases:
      with et.assert_raises_message(
          RuntimeError,
          tpu="""remainder(): expected the dtype of the output (promoted inputs dtype) to be neither bool nor complex, got complex64""",
          cpu=""""remainder_cpu" not implemented for 'ComplexFloat'""",
          message_reviewed_by="wan",
      ):
        torch.remainder(*args)

  def test_foreach_abs_inplace_complex(self):
    self_list = [
        torch.tensor([1], device=et.device(), dtype=torch.int64),
        torch.tensor([1 + 1j], device=et.device(), dtype=torch.complex64),
        torch.tensor([1 + 1j], device=et.device(), dtype=torch.complex64),
        torch.tensor([1.0], device=et.device(), dtype=torch.float64),
    ]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_abs_(): expected all 4 tensors in the self list not to be complex, got 2 complex tensors: complex64 at index 1 and complex64 at index 2""",
        cpu="""In-place abs is not supported for complex tensors.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_abs_(self_list)

  def test_foreach_add_int_tensors_float_alpha(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_add(): expected alpha to be integral for integral input tensors, got float64""",
        cpu="""For integral input tensors, argument alpha must not be a floating point number.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_add(self_list, other_list, alpha=1.5)

  def test_foreach_add_int_tensors_bool_alpha(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_add(): expected input tensor dtypes to be bool when alpha dtype is bool, got int32 and int32""",
        cpu="""Boolean alpha only supported for Boolean results.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_add(self_list, other_list, alpha=True)

  def test_foreach_add_inplace_int_and_float(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_add_(): expected the scalar dtype to be castable to the tensor dtype (e.g. bool to int or int to float), got float64 and int32""",
        cpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch._foreach_add_(self_list, 1.5)

  def test_foreach_add_inplace_bool_tensors_and_int_scalars(self):
    self_list = [
        torch.tensor([True, True], dtype=torch.bool, device=et.device()),
        torch.tensor([True, True], dtype=torch.bool, device=et.device()),
    ]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_add_(): expected the scalar dtype to be castable to the tensor dtype (e.g. bool to int or int to float), got int64 and bool""",
        cpu="""result type Long can't be cast to the desired output type Bool""",
        message_reviewed_by="wan",
    ):
      torch._foreach_add_(self_list, [1, 1])

  def test_foreach_sub_bool(self):
    self_list = [
        torch.ones(5, device=et.device()),
        torch.tensor([True, True], dtype=torch.bool, device=et.device()),
    ]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_sub(): expected all 2 tensors in the self list not to be bool, got 1 bool tensor: bool at index 1""",
        cpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_sub(self_list, [1, 1])

  def test_foreach_sub_scalar_bool(self):
    self_list = [
        torch.tensor([1, 4], dtype=torch.int32, device=et.device()),
        torch.tensor([9, 16], dtype=torch.int32, device=et.device()),
    ]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_sub(): expected the scalar argument not to be bool, got true""",
        cpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_sub(self_list, True)

  def test_foreach_sub_scalar_list_bool(self):
    self_list = [
        torch.tensor([1, 4], dtype=torch.int32, device=et.device()),
        torch.tensor([9, 16], dtype=torch.int32, device=et.device()),
    ]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_sub(): expected all 2 scalars in the scalars list not to be bool, got 1 bool scalar: true at index 1""",
        cpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_sub(self_list, [1, True])

  def test_foreach_sub_int_tensors_float_alpha(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_sub(): expected alpha to be integral for integral input tensors, got float64""",
        cpu="""For integral input tensors, argument alpha must not be a floating point number.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_sub(self_list, other_list, alpha=1.5)

  def test_foreach_sub_inplace_int_tensors_float_alpha(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_sub_(): expected alpha to be integral for integral input tensors, got float64""",
        cpu="""For integral input tensors, argument alpha must not be a floating point number.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_sub_(self_list, other_list, alpha=1.5)

  def test_foreach_sqrt_inplace_integral(self):
    self_list = [
        torch.tensor([1, 4], dtype=torch.int32, device=et.device()),
        torch.tensor([1, 4], dtype=torch.float64, device=et.device()),
        torch.tensor([9, 16], dtype=torch.int32, device=et.device()),
        torch.tensor([9, 16], dtype=torch.float64, device=et.device()),
    ]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_sqrt_(): expected all 4 tensors in the self list not to be integral, got 2 integral tensors: int32 at index 0 and int32 at index 2""",
        cpu="""result type Float can't be cast to the desired output type Int""",
    ):
      torch._foreach_sqrt_(self_list)

  def test_foreach_addcdiv_integral(self):
    self_list = [
        torch.tensor([1, 2], dtype=torch.int64, device=et.device()),
        torch.tensor([1, 2], dtype=torch.int64, device=et.device()),
        torch.tensor([1, 2], dtype=torch.int64, device=et.device()),
    ]
    tensor1_list = [
        torch.tensor([3, 4], dtype=torch.int32, device=et.device()),
        torch.tensor([3, 4], dtype=torch.int32, device=et.device()),
        torch.tensor([3, 4], dtype=torch.float32, device=et.device()),
    ]
    tensor2_list = [
        torch.tensor([5, 6], dtype=torch.uint8, device=et.device()),
        torch.tensor([5, 6], dtype=torch.uint8, device=et.device()),
        torch.tensor([5, 6], dtype=torch.uint8, device=et.device()),
    ]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_addcdiv(): expected at least one non-integral tensor in each of the 3 dividend (second tensor list) and divisor (third tensor list) pairs, got 2 integral dividend-divisor tensor pairs: (int32, uint8) at index 0 and (int32, uint8) at index 1""",
        cpu="""Integer division with addcdiv is no longer supported, and in a future  release addcdiv will perform a true division of tensor1 and tensor2. The historic addcdiv behavior can be implemented as (input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) for integer inputs and as (input + value * tensor1 / tensor2) for float inputs. The future addcdiv behavior is just the latter implementation: (input + value * tensor1 / tensor2), for all dtypes.""",
    ):
      torch._foreach_addcdiv(self_list, tensor1_list, tensor2_list)

  def test_cat_out_invalid_cast(self):
    """Tests that cat fails when the out tensor has an incompatible dtype."""
    t_f32 = torch.tensor([1.0, 2.0], device=et.device(), dtype=torch.float32)
    out_int32 = torch.zeros(2, device=et.device(), dtype=torch.int32)
    err_type = RuntimeError if et.device().type == "tpu" else TypeError
    with et.assert_raises_message(
        err_type,
        tpu="""cat(): expected the input to be castable to the desired dtype int32, got float32""",
        cpu="""torch.cat(): input types can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch.cat([t_f32], out=out_int32)

  def test_sub_bool(self):
    lhs = torch.tensor([1.0, 1.0], device=et.device())
    rhs = torch.tensor([1.0, 1.0], device=et.device())
    out = torch.tensor([0.0, 0.0], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sub(): the dtype of the first argument cannot be bool""",
        cpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
        message_reviewed_by="wan",
    ):
      torch.sub(lhs.to(torch.bool), rhs, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sub(): the dtype of the second argument cannot be bool""",
        cpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
        message_reviewed_by="wan",
    ):
      torch.sub(lhs, rhs.to(torch.bool), out=out)

  def _test_aminmax_output_dtype_mismatch_impl(
      self, op_name: str, op: Any, cpu: str
  ):
    tensor = torch.ones(5, device=et.device(), dtype=torch.int64)
    out = _get_aminmax_outputs(op, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected output tensor dtype to match the dtype of the first argument (int64), got complex64""",
        cpu=cpu,
        message_reviewed_by="wan",
    ):
      op(tensor, dim=0, out=out)

  @parameterized.named_parameters(
      {"testcase_name": "amin", "op_name": "amin", "op": torch.amin},
      {"testcase_name": "amax", "op_name": "amax", "op": torch.amax},
  )
  def test_amin_amax_output_dtype_mismatch(self, op_name: str, op: Any):
    self._test_aminmax_output_dtype_mismatch_impl(
        op_name,
        op,
        cpu="""Expected the dtype for input and out to match, but got Long for input's dtype and ComplexFloat for out's dtype.""",
    )

  def test_aminmax_output_dtype_mismatch(self):
    self._test_aminmax_output_dtype_mismatch_impl(
        op_name="aminmax",
        op=torch.aminmax,
        cpu="""Expected out tensor to have dtype long, but got c10::complex<float> instead""",
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "amin",
          "op_name_cpu": "min_values",
          "op_name_tpu": "amin",
          "op": torch.amin,
      },
      {
          "testcase_name": "amax",
          "op_name_cpu": "max_values",
          "op_name_tpu": "amax",
          "op": torch.amax,
      },
      {
          "testcase_name": "aminmax",
          "op_name_cpu": "aminmax",
          "op_name_tpu": "aminmax",
          "op": torch.aminmax,
      },
  )
  def test_aminmax_complex(self, op_name_cpu: str, op_name_tpu: str, op: Any):
    tensor = torch.ones(5, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name_tpu}(): expected the dtype of the input not to be complex, got complex64""",
        cpu=f""""{op_name_cpu}_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      op(tensor, dim=0)

  def test_complex_int(self):
    real = torch.ones(5, device=et.device(), dtype=torch.float32)
    img = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = torch.empty(5, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""complex(): expected the dtype of the first argument to be float32 or float64, got int32""",
        cpu="""Expected both inputs to be Half, Float or Double tensors but got Int and Float""",
        message_reviewed_by="wan",
    ):
      torch.complex(real.to(torch.int32), img, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""complex(): expected the dtype of the second argument to be float32 or float64, got int32""",
        cpu="""Expected both inputs to be Half, Float or Double tensors but got Float and Int""",
        message_reviewed_by="wan",
    ):
      torch.complex(real, img.to(torch.int32), out=out)

  def test_polar_int(self):
    absv = torch.tensor([1.0, 2.0], device=et.device())
    angle = torch.tensor([1.0, 2.0], device=et.device())
    out = torch.empty(2, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""polar(): expected the dtype of the first argument to be float32 or float64, got int32""",
        cpu="""Expected both inputs to be Half, Float or Double tensors but got Int and Float""",
        message_reviewed_by="wan",
    ):
      torch.polar(absv.to(torch.int32), angle, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""polar(): expected the dtype of the second argument to be float32 or float64, got int32""",
        cpu="""Expected both inputs to be Half, Float or Double tensors but got Float and Int""",
        message_reviewed_by="wan",
    ):
      torch.polar(absv, angle.to(torch.int32), out=out)

  def test_addmv_bool(self):
    t = torch.ones(5, device=et.device(), dtype=torch.bool)
    mat = torch.ones(5, 5, device=et.device(), dtype=torch.bool)
    vec = torch.ones(5, device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""addmv(): the dtype of the first argument cannot be bool""",
        cpu=""""addmv_impl_cpu" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.addmv(t.to(torch.bool), mat, vec)

  def test_addmv_not_a_matrix(self):
    inp = torch.ones(5, device=et.device())
    mat = torch.ones(5, 5, 5, device=et.device())
    vec = torch.ones(5, device=et.device())
    out = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""addmv(): expected the second argument to be a matrix (2D tensor), got 3D tensor""",
        cpu="""vector + matrix @ vector expected, got 1, 3, 1""",
        message_reviewed_by="wan",
    ):
      torch.addmv(inp, mat, vec, out=out)

  def test_addmv_not_a_vector(self):
    inp = torch.ones(5, device=et.device())
    mat = torch.ones(5, 5, device=et.device())
    vec = torch.ones(5, 5, device=et.device())
    out = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""addmv(): expected the third argument to be a vector (1D tensor), got 2D tensor""",
        cpu="""vector + matrix @ vector expected, got 1, 2, 2""",
        message_reviewed_by="wan",
    ):
      torch.addmv(inp, mat, vec, out=out)

  def test_addmv_invalid_mm_dimensions(self):
    inp = torch.ones(5, device=et.device())
    mat = torch.ones(5, 5, device=et.device())
    vec = torch.ones(4, device=et.device())
    out = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""addmv(): expected the last dimension of the second argument (matrix of size [5, 5]) to match the first dimension of the third argument (vector of size [4]), got 5 vs 4""",
        cpu="""size mismatch, got input (5), mat (5x5), vec (4)""",
        message_reviewed_by="wan",
    ):
      torch.addmv(inp, mat, vec, out=out)

  def test_addmv_scalars_complex(self):
    args = (
        torch.ones(5, device=et.device()),
        torch.ones(5, 5, device=et.device()),
        torch.ones(5, device=et.device()),
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""addmv(): expected the dtype of alpha to be neither complex nor bool, got complex128""",
        cpu="""value cannot be converted to type float without overflow""",
        message_reviewed_by="wan",
    ):
      torch.addmv(*args, alpha=1j)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""addmv(): expected the dtype of beta to be neither complex nor bool, got complex128""",
        cpu="""value cannot be converted to type float without overflow""",
        message_reviewed_by="wan",
    ):
      torch.addmv(*args, beta=1j)

  def test_clamp_cant_cast(self):
    inp = torch.tensor([-1.0, 2.0], device=et.device(), dtype=torch.complex64)
    out = torch.empty(2, device=et.device(), dtype=torch.float64)

    # Scalar and Tensor inputs go through different execution path, so test that
    # both of them raise the same error.
    min_max_values = [
        (1.0, 2.0),
        (
            torch.zeros(2, device=et.device(), dtype=torch.float64),
            torch.ones(2, device=et.device(), dtype=torch.float64),
        ),
    ]

    for minv, maxv in min_max_values:
      with et.assert_raises_message(
          RuntimeError,
          tpu="""clamp(): unable to cast complex128, the promotion of the dtypes of the inputs (complex64, min: float64, max: float64), to the output dtype float64""",
          cpu="""clamp is not supported for complex types""",
          message_reviewed_by="wan",
      ):
        torch.clamp(inp, min=minv, max=maxv, out=out)

  def test_bmm_bool(self):
    a = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 3, 2, dtype=torch.float32, device=et.device())
    out = torch.ones(1, 2, 2, dtype=torch.float32, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): the dtype of the first argument cannot be bool""",
        cpu=""""bmm" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.bmm(a.to(torch.bool), b)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): the dtype of the second argument cannot be bool""",
        cpu="""Expected out tensor to have dtype bool, but got float instead""",
        message_reviewed_by="wan",
    ):
      # Call `bmm.out` op.
      #
      # Otherwise, it will trigger the output tensor dtype check first. This
      # happens because of 2 reasons:
      #   1. output dtype is checked first
      #   2. PyTorch generated code sets the output dtype to be whatever `b`
      #      dtype is (bool)
      torch.bmm(a, b.to(torch.bool), out=out)

  def test_bmm_output_bool(self):
    a = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 3, 2, dtype=torch.float32, device=et.device())
    out = torch.ones(1, 2, 2, dtype=torch.bool, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): the dtype of the output tensor cannot be bool""",
        cpu="""Expected out tensor to have dtype float, but got bool instead""",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

  def test_bmm_mismatch_dtypes(self):
    a = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 3, 2, dtype=torch.half, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): expected self and mat2 to have the same dtype, got float32 vs float16""",
        cpu="""expected scalar type Float but found Half""",
    ):
      torch.bmm(a, b)

  def test_bmm_not_batch_of_matrices(self):
    a = torch.ones(1, 2, 3, 4, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 4, 2, dtype=torch.float32, device=et.device())
    out = torch.ones(1, 2, 2, dtype=torch.float32, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): expected the first argument to be a 3D tensor (batch of matrices), got 4D""",
        cpu="""batch1 must be a 3D tensor""",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): expected the second argument to be a 3D tensor (batch of matrices), got 4D""",
        cpu="""batch2 must be a 3D tensor""",
        message_reviewed_by="wan",
    ):
      torch.bmm(b, a, out=out)

  def test_bmm_mismatch_batch_dimensions(self):
    a = torch.ones(1, 2, 3, device=et.device())
    b = torch.ones(2, 3, 2, device=et.device())
    out = torch.ones(1, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): expected the batch dimension of the first argument [1, 2, 3] to match the batch dimension of the second argument [2, 3, 2], got 1 vs 2""",
        cpu="""Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [2, 3].""",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

  def test_bmm_mismatch_mm_contracting_dimension(self):
    a = torch.ones(1, 2, 3, device=et.device())
    b = torch.ones(1, 2, 2, device=et.device())
    out = torch.ones(1, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): expected the last dimension of the first argument [1, 2, 3] to match the second dimension of the second argument [1, 2, 2], got 3 vs 2""",
        cpu="""Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [1, 2].""",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

  def test_grouped_mm_unsupported_bias(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    offs = torch.tensor([1, 2, 3], dtype=torch.int32, device=et.device())
    bias = torch.randn(8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected bias to be undefined, got defined tensor""",
        cpu="""Bias not supported yet""",
    ):
      torch._grouped_mm(a, b, offs=offs, bias=bias)

  def test_grouped_mm_invalid_self_dim(self):
    a = torch.randn(1, 2, 3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected self to be 2D or 3D, got 4D""",
        cpu="""mat_a has to be 2 or 3d""",
    ):
      torch._grouped_mm(a, b)

  def test_grouped_mm_invalid_mat2_dim(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(1, 2, 3, 4, device=et.device())
    offs = torch.tensor([1, 2, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected mat2 to be 2D or 3D, got 4D""",
        cpu="""mat_b has to be 2 or 3d""",
    ):
      torch._grouped_mm(a, b, offs=offs)

  def test_grouped_mm_missing_offs(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected offs to be provided if and only if either self or mat2 is 2D""",
        cpu="""Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d""",
    ):
      torch._grouped_mm(a, b)

  def test_grouped_mm_unexpected_offs(self):
    a = torch.randn(2, 8, 8, device=et.device())
    b = torch.randn(2, 8, 8, device=et.device())
    offs = torch.tensor([1, 2], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected offs to be provided if and only if either self or mat2 is 2D""",
        cpu="""Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d""",
    ):
      torch._grouped_mm(a, b, offs=offs)

  def test_grouped_mm_mismatch_matrix_batch_sizes_case1(self):
    a = torch.randn(10, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    offs = torch.tensor([2, 5], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected offs batch size to match mat2 batch size, got 2 and 3""",
        cpu="""matrix batch sizes have to match""",
    ):
      torch._grouped_mm(a, b, offs=offs)

  def test_grouped_mm_mismatch_matrix_batch_sizes_case2(self):
    a = torch.randn(3, 4, 8, device=et.device())
    b = torch.randn(5, 8, device=et.device())
    offs = torch.tensor([2, 5], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected offs batch size to match self batch size, got 2 and 3""",
        cpu="""contraction dimension of mat_a and mat_b must match""",
    ):
      torch._grouped_mm(a, b, offs=offs)

  def test_grouped_mm_mismatch_batched_dimension_case4(self):
    a = torch.randn(3, 4, 8, device=et.device())
    b = torch.randn(2, 8, 8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected self batch size to match mat2 batch size, got 3 and 2""",
        cpu="""batched dimension has to match""",
    ):
      torch._grouped_mm(a, b)

  def test_baddbmm_unsupported_bool(self):
    input_tensor = torch.ones(1, 2, 2, device=et.device())
    batch1 = torch.ones(1, 2, 3, device=et.device())
    batch2 = torch.ones(1, 3, 2, device=et.device())
    out = torch.ones(1, 2, 2, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""baddbmm(): expected out tensor to have dtype float32, got bool""",
        cpu="""Expected out tensor to have dtype float, but got bool instead""",
        message_reviewed_by="wan",
    ):
      torch.baddbmm(input_tensor, batch1, batch2, out=out.to(torch.bool))

  def test_baddbmm_mismatch_dtypes_batch(self):
    input_tensor = torch.ones(1, 2, 2, dtype=torch.float32, device=et.device())
    batch1 = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    batch2 = torch.ones(1, 3, 2, dtype=torch.half, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""baddbmm(): expected input dtypes to be the same, got: self=float32, batch1=float32, batch2=float16""",
        cpu="""expected scalar type Float but found Half""",
        message_reviewed_by="wan",
    ):
      torch.baddbmm(input_tensor, batch1, batch2)

  def test_baddbmm_not_batch_of_matrices(self):
    input_tensor = torch.ones(1, 1, 1, device=et.device())
    batch1 = torch.ones(1, 2, 3, 4, device=et.device())
    batch2 = torch.ones(1, 3, 2, device=et.device())

    # Explicitly call `baddbmm.out` op.
    # Otherwise, it will trigger the output tensor dtype check first.
    out = torch.ones(1, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""baddbmm(): expected batch1 to be a 3D tensor (batch of matrices), got 4D""",
        cpu="""batch1 must be a 3D tensor""",
        message_reviewed_by="wan",
    ):
      torch.baddbmm(input_tensor, batch1, batch2, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""baddbmm(): expected batch2 to be a 3D tensor (batch of matrices), got 4D""",
        cpu="""batch2 must be a 3D tensor""",
        message_reviewed_by="wan",
    ):
      torch.baddbmm(input_tensor, batch2, batch1, out=out)

  def test_baddbmm_mismatch_batch_dimensions(self):
    input_tensor = torch.ones(1, 2, 2, device=et.device())
    batch1 = torch.ones(1, 2, 3, device=et.device())
    batch2 = torch.ones(2, 3, 2, device=et.device())
    out = torch.ones(1, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""baddbmm(): expected the batch dimension of the first argument (of shape [1, 2, 3]) to match the batch dimension of the second argument (of shape [2, 3, 2]), got 1 vs 2""",
        cpu="""Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [2, 3].""",
        message_reviewed_by="wan",
    ):
      torch.baddbmm(input_tensor, batch1, batch2, out=out)

  def test_baddbmm_mismatch_mm_contracting_dimension(self):
    input_tensor = torch.ones(1, 2, 2, device=et.device())
    batch1 = torch.ones(1, 2, 3, device=et.device())
    batch2 = torch.ones(1, 4, 2, device=et.device())
    out = torch.ones(1, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""baddbmm(): expected the last dimension of the first argument (of shape [1, 2, 3]) to match the second dimension of the second argument (of shape [1, 4, 2]), got 3 vs 4""",
        cpu="""Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [1, 4].""",
        message_reviewed_by="wan",
    ):
      torch.baddbmm(input_tensor, batch1, batch2, out=out)

  @_parameterize_convolution_fwd_bwd(
      forward={
          "tpu_fn": "convolution",
          "cpu_fn": "slow_conv2d_cpu",
      },
      backward={
          "tpu_fn": "convolution_backward",
          "cpu_fn": "slow_conv2d_cpu_grad_input",
      },
  )
  def test_convolution_bool(self, convolution, tpu_fn: str, cpu_fn: str):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{tpu_fn}(): expected the dtype of the input tensor to be neither long nor bool, got bool""",
        cpu=f""""{cpu_fn}" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      convolution(inp.to(torch.bool), w)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{tpu_fn}(): expected the dtype of the weight tensor to be neither long nor bool, got bool""",
        cpu="""expected scalar type Float but found Bool""",
        message_reviewed_by="wan",
    ):
      convolution(inp, w.to(torch.bool))

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_convolution_input_invalid_rank(self, convolution, tpu_fn: str):
    inp = torch.ones(10, 10, device=et.device())

    # PyTorch CPU implementation will check the weight tensor before the input
    # tensor. So, it must have >3 dimensions.
    w = torch.ones(2, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{tpu_fn}(): expected the input to have >= 3 dimensions of shape [batch, in channels, ... spatial dimensions ...], got shape [10, 10]""",
        cpu="""Expected 3-dimensional input for 3-dimensional weight [2, 3, 3], but got 2-dimensional input of size [10, 10] instead""",
        message_reviewed_by="wan",
    ):
      convolution(inp, w)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_convolution_spatial_dimensions_mismatch(
      self, convolution, tpu_fn: str
  ):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    def test_arg(arg_name: str):
      with et.assert_raises_message(
          RuntimeError,
          tpu=f"""{tpu_fn}(): expected {arg_name} to be either an integer or a 2-element list that matches the convolution dimensions, got [1, 1, 1]""",
          cpu=f"""expected {arg_name} to be a single integer value or a list of 2 values to match the convolution dimensions, but got {arg_name}=[1, 1, 1]""",
          message_reviewed_by="wan",
      ):
        convolution(inp, w, **{arg_name: (1, 1, 1)})

    # Each of these parameters go through the same error check.
    # Since this test is already parameterized for forward and backward, we use
    # subTest() for checking each parameter.
    for arg_name in ("stride", "padding", "dilation", "output_padding"):
      with self.subTest(arg_name=arg_name):
        test_arg(arg_name)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_convolution_weight_invalid_rank(self, convolution, tpu_fn: str):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{tpu_fn}(): expected the weight tensor to have 4 dimensions of shape [out channels, in channels per group, ... 2 spatial dimensions ...], got shape [1, 3, 3, 3, 3]""",
        cpu="""Expected 5-dimensional input for 5-dimensional weight [1, 3, 3, 3, 3], but got 4-dimensional input of size [2, 3, 10, 10] instead""",
        message_reviewed_by="wan",
    ):
      convolution(inp, w)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_convolution_weight_dimension_mismatch(
      self, convolution, tpu_fn: str
  ):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{tpu_fn}(): expected the second dimension of the weight tensor of shape [1, 3, 3, 3] to be 1 (3 in channels divided by 3 groups), got 3""",
        cpu="""Given groups=3, expected weight to be at least 3 at dimension 0, but got weight of size [1, 3, 3, 3] instead""",
        message_reviewed_by="wan",
    ):
      convolution(inp, w, groups=3)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_transposed_convolution_weight_invalid_rank(
      self, convolution, tpu_fn: str
  ):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{tpu_fn}(): expected the weight tensor to have 4 dimensions of shape [in channels, out channels per group, ... 2 spatial dimensions ...], got shape [1, 3, 3, 3, 3]""",
        cpu="""Expected 5-dimensional input for 5-dimensional weight [1, 3, 3, 3, 3], but got 4-dimensional input of size [2, 3, 10, 10] instead""",
        message_reviewed_by="wan",
    ):
      convolution(inp, w, transposed=True)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_transposed_convolution_weight_dimension_mismatch(
      self, convolution, tpu_fn: str
  ):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{tpu_fn}(): expected the first dimension of the weight tensor of shape [1, 3, 3, 3] to be 3 (number of in channels), got 1""",
        cpu="""Given groups=3, expected weight to be at least 3 at dimension 0, but got weight of size [1, 3, 3, 3] instead""",
        message_reviewed_by="wan",
    ):
      convolution(inp, w, groups=3, transposed=True)

  def test_convolution_bias_invalid_dimensions(self):
    # Why isn't this test also run on backward?
    # =========================================
    #
    # PyTorch native device implementation doesn't error on invalid bias_sizes
    # values. This might actually be a bug on PyTorch upstream.

    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""convolution(): expected the bias tensor to have 1 dimension of shape [1 (out channels)], got shape [1, 1]""",
        cpu="""Given weight of size [1, 3, 3, 3], expected bias to be 1-dimensional with 1 elements, but got bias of size [1, 1] instead""",
        message_reviewed_by="wan",
    ):
      _run_convolution(inp, w, bias=torch.ones(1, 1, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""convolution(): expected the bias tensor to have 1 dimension of shape [1 (out channels)], got shape [5]""",
        cpu="""Given weight of size [1, 3, 3, 3], expected bias to be 1-dimensional with 1 elements, but got bias of size [5] instead""",
        message_reviewed_by="wan",
    ):
      _run_convolution(inp, w, bias=torch.ones(5, device=et.device()))

  def test_convolution_backward_grad_bool(self):
    # This test compliments the parameterized test `test_convolution_bool()`,
    # which does the same for `inp` and `w`. Since `grad` only exists in the
    # backward, this specifically tests it.
    grad = torch.ones(2, 1, 8, 8, device=et.device(), dtype=torch.bool)

    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""convolution_backward(): expected the dtype of the grad tensor to be neither long nor bool, got bool""",
        cpu="""expected scalar type Float but found Bool""",
        message_reviewed_by="wan",
    ):
      _run_convolution_backward(grad, inp, w)

  @parameterized.named_parameters(
      {"testcase_name": "min", "op": torch.min, "op_name": "min"},
      {"testcase_name": "max", "op": torch.max, "op_name": "max"},
  )
  def test_min_max_input_with_zero_elements(self, op, op_name: str):
    inp = torch.ones(2, 2, 0, 5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the dim argument to be specified when the input tensor has 0 elements""",
        cpu=f"""{op_name}(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.""",
        message_reviewed_by="wan",
    ):
      op(inp)

  @parameterized.named_parameters(
      {"testcase_name": "argmin", "op": torch.argmin, "op_name": "argmin"},
      {"testcase_name": "argmax", "op": torch.argmax, "op_name": "argmax"},
  )
  def test_argmin_float_output(self, op, op_name: str):
    inp = torch.ones(2, 2, device=et.device())
    out = torch.empty(2, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the output dtype to be int64, got float32""",
        cpu="""Expected out tensor to have dtype long, but got float instead""",
        message_reviewed_by="wan",
    ):
      op(inp, out=out)

  @parameterized.named_parameters(
      {"testcase_name": "argmin", "op": torch.argmin, "op_name": "argmin"},
      {"testcase_name": "argmax", "op": torch.argmax, "op_name": "argmax"},
  )
  def test_argmin_invalid_dtypes(self, op, op_name: str):
    # We need to call the out overload of `argmin` (`argmax`) op, so
    # that we don't go through the fallback. Otherwise, the meta kernel will
    # catch this error before it reaches TorchTPU implementation.
    out = torch.empty(2, device=et.device(), dtype=torch.int64)

    def test_with(dtype: torch.dtype, tpu: str, cpu: str):
      """Tests the `op` with the input tensor of the given `dtype`.

      Tests that running `argmin` (`argmax`) with the given `dtype` will result
      in the expected error.

      Args:
        dtype: The dtype of the op input tensor.
        tpu: String representation for `dtype` to be used in the error message
          of the TPU kernel.
        cpu: String representation for `dtype` to be used in the error message
          of the CPU kernel.
      """

      inp = torch.ones(2, 2, device=et.device(), dtype=dtype)

      with et.assert_raises_message(
          RuntimeError,
          tpu=f"""{op_name}(): expected the input dtype to be neither complex nor"""
          f""" bool, got {tpu}""",
          cpu=f"""{op_name}(): does not support {cpu} input""",
          message_reviewed_by="wan",
      ):
        op(inp, out=out)

    with self.subTest(dtype=torch.bool):
      test_with(torch.bool, tpu="""bool""", cpu="""bool""")
    with self.subTest(dtype=torch.complex64):
      test_with(torch.complex64, tpu="""complex64""", cpu="""complex""")

  def test_mm_output_dtype_mismatch(self):
    lhs = torch.ones(3, 4, device=et.device(), dtype=torch.float32)
    rhs = torch.ones(4, 5, device=et.device())
    out = torch.ones(3, 5, device=et.device(), dtype=torch.float64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected the output to have the same dtype as inputs, got out dtype float64 vs inputs dtype float32""",
        cpu="""Expected out tensor to have dtype float, but got double instead""",
        message_reviewed_by="wan",
    ):
      torch.mm(lhs, rhs, out=out)

  def test_mm_inputs_dtype_mismatch(self):
    lhs = torch.ones(3, 4, device=et.device(), dtype=torch.float32)
    rhs = torch.ones(4, 5, device=et.device(), dtype=torch.float64)

    # Call the out overload of `mm()` op.
    out = torch.ones(3, 6, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected the two arguments to have the same dtype, got float32 vs float64""",
        cpu="""expected m1 and m2 to have the same dtype, but got: float != double""",
        message_reviewed_by="wan",
    ):
      torch.mm(lhs, rhs, out=out)

  def test_mm_inputs_are_not_matrices(self):
    not_a_matrix_tensor = torch.ones(3, 4, 5, device=et.device())
    matrix_tensor = torch.ones(4, 4, device=et.device())

    # Call the out overload of `mm()` op.
    out = torch.ones(4, 4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected the first argument to be a 2D tensor (matrix), got 3D of shape [3, 4, 5]""",
        cpu="""self must be a matrix""",
        message_reviewed_by="wan",
    ):
      torch.mm(not_a_matrix_tensor, matrix_tensor, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected the second argument to be a 2D tensor (matrix), got 3D of shape [3, 4, 5]""",
        cpu="""mat2 must be a matrix""",
        message_reviewed_by="wan",
    ):
      torch.mm(matrix_tensor, not_a_matrix_tensor, out=out)

  def test_mm_inputs_dimension_mismatch(self):
    lhs = torch.ones(3, 4, device=et.device())
    rhs = torch.ones(5, 6, device=et.device())

    # Call the out overload of `mm()` op.
    out = torch.ones(3, 6, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected the column size of the first matrix to match the row size of the second matrix, got shape [3, 4] vs [5, 6] where 4 != 5""",
        cpu="""mat1 and mat2 shapes cannot be multiplied (3x4 and 5x6)""",
        message_reviewed_by="wan",
    ):
      torch.mm(lhs, rhs, out=out)

  def test_linalg_lu_factor_ex_no_pivoting(self):
    a = torch.ones(1, 2, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_factor_ex(): non-pivoting decomposition is not supported""",
        cpu="""linalg.lu_factor: LU without pivoting is not implemented on the CPU""",
        message_reviewed_by="wan",
    ):
      torch.linalg.lu_factor_ex(a, pivot=False)

  def test_linalg_lu_factor_ex_rank_too_low(self):
    a = torch.ones(4, device=et.device())

    # We need to call the out overload of linalg.lu_factor_ex() op, so
    # that we don't go through the fallback. Otherwise, the meta kernel will
    # catch this error before it reaches TorchTPU implementation.
    out = (
        torch.empty(4, device=et.device()),
        torch.empty(4, device=et.device()),
        torch.empty(4, device=et.device()),
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_factor_ex(): input tensor expected to have at least 2 dimensions, got 1""",
        cpu="""torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: [4] instead""",
    ):
      torch.linalg.lu_factor_ex(a, out=out)

  def test_lu_unpack_data_rank_too_low(self):
    data = torch.ones(4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)

    # Call the out overload of linalg.lu_unpack() op.
    out = _make_lu_unpack_outputs(p=(4,), l=(4,), u=(4,))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""lu_unpack(): lu_data must have at least 2 dimensions, got 1""",
        cpu="""torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: [4] instead""",
    ):
      torch.lu_unpack(data, pivots, out=out)

  def test_lu_solve_rank_too_low(self):
    lu = torch.ones(4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): lu must have at least 2 dimensions, got 1""",
        cpu="""torch.linalg.lu_solve: The input tensor A must have at least 2 dimensions.""",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_rectangular_matrix(self):
    lu = torch.ones(4, 2, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): lu must be square, got 4 and 2""",
        cpu="""torch.linalg.lu_solve: A must be batches of square matrices, but they are 4 by 2 matrices""",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_dimensions_mismatch(self):
    lu = torch.ones(4, 4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 4, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): b must have compatible dimensions with lu, got b.shape[-2:]=(3, 4) and lu.shape[-2:]=(4, 4), and left=1""",
        cpu="""linalg.lu_solve: Incompatible shapes of A and B for the equation AX = B (4x4 and 3x4)""",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_pivots_invalid_dimensions(self):
    lu = torch.ones(3, 3, device=et.device())
    pivots = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 3, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): pivots must have one less dimension than the tensor, got 2 and 2""",
        cpu="""linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, but got pivots with shape [2, 3] instead""",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_batch_dimensions_mismatch(self):
    lu = torch.ones(3, 3, 3, device=et.device())
    pivots = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 3, 3, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): pivots and tensor must have the same batch dimensions, got [3] and [2]""",
        cpu="""linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, but got pivots with shape [2, 3] instead""",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_pivots_dimension_too_high(self):
    lu = torch.ones(3, 3, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 3, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): pivots size must be less than or equal to the size of the matrix, got 4 and 3""",
        cpu="""linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix""",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_multinomial_int(self):
    inp = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""multinomial(): expected the input dtype to be floating-point, got int32""",
        cpu="""multinomial only supports floating-point dtypes for input, got: Int""",
        message_reviewed_by="wan",
    ):
      torch.multinomial(inp, num_samples=2)

  def test_multinomial_invalid_dimension(self):
    inp = torch.randn(2, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""multinomial(): expected the input to have either 1 or 2 dimensions, got 3 of shape [2, 2, 2]""",
        cpu="""prob_dist must be 1 or 2 dim""",
        message_reviewed_by="wan",
    ):
      torch.multinomial(inp, num_samples=2)

  def test_multinomial_invalid_samples(self):
    inp = torch.randn(2, device=et.device())

    # Make sure we don't allow, in general, the following cases:
    #   - negative values
    #   - zero
    for num_samples in [-1, 0]:
      with self.subTest(num_samples=num_samples), et.assert_raises_message(
          RuntimeError,
          tpu="""multinomial(): expected the number of samples to be > 0, got"""
          f""" {num_samples}""",
          cpu="""cannot sample n_sample <= 0 samples""",
          message_reviewed_by="wan",
      ):
        torch.multinomial(inp, num_samples=num_samples)

    # Make sure we check the number of samples when `replacement` is set to
    # `False`.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""multinomial(): expected the number of samples to be <= 2 (population size) when replacement is disabled, got 3""",
        cpu="""cannot sample n_sample > prob_dist.size(-1) samples without replacement""",
        message_reviewed_by="wan",
    ):
      torch.multinomial(inp, num_samples=3, replacement=False)

  def test_index_too_many_indices(self):
    t = torch.ones(2, 2, device=et.device())
    indices = [
        torch.ones(2, device=et.device()),
        torch.ones(2, device=et.device()),
        torch.ones(2, device=et.device()),
    ]

    # Call the out overload.
    out = torch.empty(1, device=et.device())

    with et.assert_raises_message(
        IndexError,
        tpu="""index(): expected the size of the indices to be <= 2 (number of input dimensions), got 3""",
        cpu="""too many indices for tensor of dimension 2 (got 3)""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.index.Tensor_out(t, indices, out=out)

  def test_index_no_indices(self):
    t = torch.ones(2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""index(): at least one index tensor must be defined""",
        cpu=_INDEX_INTERNAL_ASSERTION_ERROR_RE,
        message_reviewed_by="wan",
    ):
      torch.ops.aten.index.Tensor(t, [None])

  @parameterized.named_parameters(
      {"testcase_name": "dot", "op": torch.dot, "op_name": "dot"},
      {"testcase_name": "vdot", "op": torch.vdot, "op_name": "vdot"},
  )
  def test_dot_not_a_vector(self, op, op_name: str):
    lhs = torch.ones(2, device=et.device())
    rhs = torch.ones(2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the first argument to be a 1D tensor, got 2D of shape [1, 2]""",
        cpu="""1D tensors expected, but got 2D and 1D tensors""",
        message_reviewed_by="wan",
    ):
      op(lhs.unsqueeze(0), rhs)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the second argument to be a 1D tensor, got 2D of shape [1, 2]""",
        cpu="""1D tensors expected, but got 1D and 2D tensors""",
        message_reviewed_by="wan",
    ):
      op(lhs, rhs.unsqueeze(0))

  @parameterized.named_parameters(
      {"testcase_name": "dot", "op": torch.dot, "op_name": "dot"},
      {"testcase_name": "vdot", "op": torch.vdot, "op_name": "vdot"},
  )
  def test_dot_bool(self, op, op_name: str):
    lhs = torch.ones(2, device=et.device(), dtype=torch.bool)
    rhs = torch.ones(2, device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): the input dtypes cannot be bool""",
        cpu=""""dot" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      op(lhs, rhs)

  def test_vdot_size_mismatch(self):
    lhs = torch.ones(2, device=et.device())
    rhs = torch.ones(3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""vdot(): expected inputs to have the same shape, got [2] vs [3]""",
        cpu="""inconsistent tensor size, expected tensor [2] and src [3] to have the same number of elements, but got 2 and 3 elements respectively""",
        message_reviewed_by="wan",
    ):
      torch.vdot(lhs, rhs)

  def test_embedding_bag_invalid_dtypes(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""embedding_bag_forward_only(): expected weight dtype to be float16, bfloat16, float32, or float64, got int64""",
        cpu="""Expected tensor for argument #1 'weight' to have one of the following scalar types: Half, BFloat16, Float, Double; but got torch.LongTensor instead (while checking arguments for embedding_bag)""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.embedding_bag(
          torch.tensor([0, 1], dtype=torch.int64, device=et.device()),
          torch.tensor(
              [[0, 1, 2, 3, 4]], dtype=torch.int64, device=et.device()
          ),
          torch.tensor([0, 1], dtype=torch.int64, device=et.device()),
      )

  def test_native_layer_norm_int(self):
    inp = torch.ones(5, 5, device=et.device(), dtype=torch.int32)
    normalized_shape = (5,)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""layer_norm(): expected the input dtype to be floating point, got int32""",
        cpu=""""LayerNormKernelImpl" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm(inp, normalized_shape)

  def test_native_layer_norm_backward_int(self):
    inp = torch.ones(5, 5, device=et.device(), dtype=torch.int32)
    normalized_shape = (5,)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_layer_norm_backward(): expected the input dtype to be floating point, got int32""",
        cpu=""""LayerNormBackwardKernelImpl" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm_backward(inp, normalized_shape)

  def test_native_layer_norm_normalized_shape_empty(self):
    inp = torch.ones(5, 5, device=et.device())
    normalized_shape = []

    with et.assert_raises_message(
        RuntimeError,
        tpu="""layer_norm(): the normalized shape must have >= 1 dimensions""",
        cpu="""Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one element, but got normalized_shape = []""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm(inp, normalized_shape)

  def test_native_layer_norm_backward_normalized_shape_empty(self):
    inp = torch.ones(5, 5, device=et.device())
    normalized_shape = []

    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_layer_norm_backward(): the normalized shape must have >= 1 dimensions""",
        cpu="""Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one element, but got normalized_shape = []""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm_backward(inp, normalized_shape)

  def test_native_layer_norm_normalized_shape_too_large(self):
    inp = torch.ones(5, 5, device=et.device())
    normalized_shape = (5, 3, 3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""layer_norm(): expected the normalized shape to have <= 2 dimensions, got 3 dimensions of shape [5, 3, 3]""",
        cpu="""Given normalized_shape=[5, 3, 3], expected input with shape [*, 5, 3, 3], but got input of size[5, 5]""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm(inp, normalized_shape)

  def test_native_layer_norm_backward_normalized_shape_too_large(self):
    inp = torch.ones(5, 5, device=et.device())
    normalized_shape = (5, 3, 3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_layer_norm_backward(): expected the normalized shape to have <= 2 dimensions, got 3 dimensions of shape [5, 3, 3]""",
        cpu="""Given normalized_shape=[5, 3, 3], expected input with shape [*, 5, 3, 3], but got input of size[5, 5]""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm_backward(inp, normalized_shape)

  def test_random_invalid_range(self):
    t = torch.ones(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""random_(): expected 'from' to be < 'to', got 20 vs 10""",
        cpu="""random_ expects 'from' to be less than 'to', but got from=20 >= to=10""",
        message_reviewed_by="wan",
    ):
      t.random_(20, 10)

  def test_randn_unsupported_dtype(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""normal_(): expected the self tensor to be floating point or complex type, got int32""",
        cpu=""""normal_kernel_cpu" not implemented for 'Int'""",
    ):
      torch.randn(5, dtype=torch.int32, device=et.device())

  def test_uniform_unsupported_dtype(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""uniform_(): expected the input dtype to be floating point or complex, got int32""",
        cpu=""""check_uniform_bounds" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.zeros(5, dtype=torch.int32, device=et.device()).uniform_()

  def test_linalg_inv_ex_1d(self):
    a = torch.ones(5, device=et.device())

    # Call the out overload of linalg.inv_ex() op.
    out = (
        torch.ones(4, 4, 4, device=et.device()),
        torch.ones(4, 4, 4, device=et.device()),
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_inv_ex(): expected the input tensor to have at least 2 dimensions, got 1 dimensions of shape [5]""",
        cpu="""linalg.inv: The input tensor A must have at least 2 dimensions.""",
        message_reviewed_by="wan",
    ):
      torch.linalg.inv_ex(a, out=out)

  def test_linalg_inv_ex_non_square(self):
    a = torch.ones(3, 5, device=et.device())

    # Call the out overload of linalg.inv_ex() op.
    out = (
        torch.ones(4, 4, 4, device=et.device()),
        torch.ones(4, 4, 4, device=et.device()),
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_inv_ex(): expected the input tensor's last 2 dimensions to be equal, got [3, 5]""",
        cpu="""linalg.inv: A must be batches of square matrices, but they are 3 by 5 matrices""",
        message_reviewed_by="wan",
    ):
      torch.linalg.inv_ex(a, out=out)

  def test_rms_norm_int(self):
    inp = torch.ones(5, 5, device=et.device(), dtype=torch.int32)
    normalized_shape = (5,)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_rms_norm(): expected the input dtype to be floating point, got int32""",
        cpu=""""rms_norm" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.rms_norm(inp, normalized_shape)

  def test_hardswish_unsupported_dtype(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardswish(): expected the input dtype to be floating point, got int32""",
        cpu=""""hardswish_cpu" not implemented for 'Int'""",
    ):
      torch.nn.functional.hardswish(t)

  def test_hardswish_out_unsupported_dtype(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    out = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardswish(): expected the input dtype to be floating point, got int32""",
        cpu=""""hardswish_cpu" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.hardswish.out(t, out=out)

  def test_hardswish_inplace_unsupported_dtype(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardswish_(): expected the input dtype to be floating point, got int32""",
        cpu=""""hardswish_cpu" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.hardswish(t, inplace=True)

  def test_hardswish_backward_unsupported_dtype(self):
    grad = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    self_val = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardswish_backward(): expected the input dtype to be floating point, got int32""",
        cpu=""""hardswish_backward_cpu" not implemented for 'Int'""",
    ):
      torch.ops.aten.hardswish_backward(grad, self_val)

  def test_hardsigmoid_unsupported_dtype(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardsigmoid(): expected the input dtype to be floating point, got int32""",
        cpu=""""hardsigmoid_cpu" not implemented for 'Int'""",
    ):
      torch.nn.functional.hardsigmoid(t)

  def test_hardsigmoid_backward_unsupported_dtype(self):
    grad = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    self_val = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardsigmoid_backward(): expected the input dtype to be floating point, got int32""",
        cpu=""""hardsigmoid_backward" not implemented for 'Int'""",
    ):
      torch.ops.aten.hardsigmoid_backward(grad, self_val)

  def test_bincount_rank_too_high(self):
    t = torch.ones(2, 2, 2, device=et.device(), dtype=torch.int32)

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""bincount(): materialization failed with: Unexpected dimension of input tensor: [2, 2, 2]""",
        cpu="""bincount only supports 1-d non-negative integral inputs.""",
    ):
      torch.bincount(t).cpu()

  def test_gather_2d_input_on_scalar_index(self):
    inp = torch.ones(2, 2, device=et.device())
    dim = 0
    index = torch.tensor(0, device=et.device(), dtype=torch.int64)

    # Call the out overload.
    out = torch.empty(1, 1, device=et.device())

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gather(): materialization failed with: expected the input to be a 1D tensor with size at most 1 when index is 0D, got 2D with shape {2}""",
        cpu="""Index tensor must have the same number of dimensions as input tensor""",
    ):
      # cpu() is needed because the error is triggered inside the op builder.
      torch.gather(inp, dim, index, out=out).cpu()

  def test_gather_rank_mismatch(self):
    inp = torch.ones(2, 2, device=et.device())
    dim = 0
    index = torch.ones(1, 1, 1, device=et.device(), dtype=torch.int64)

    # Call the out overload.
    out = torch.empty(1, 1, device=et.device())

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gather(): materialization failed with: expected the input and the index tensor to have the same number of dimensions, got 2D vs 3D""",
        cpu="""Index tensor must have the same number of dimensions as input tensor""",
    ):
      # cpu() is needed because the error is triggered inside the op builder.
      torch.gather(inp, dim, index, out=out).cpu()

  def test_gather_0d_input_on_2d_index(self):
    inp = torch.tensor(1.0, device=et.device())
    dim = 0
    index = torch.ones(2, 2, device=et.device(), dtype=torch.int64)

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gather(): materialization failed with: expected the input and the index tensor to have the same number of dimensions, got 0D vs 2D""",
        cpu="""Index tensor must have the same number of dimensions as input tensor""",
    ):
      torch.gather(inp, dim, index).cpu()

  def test_gather_size_mismatch(self):
    inp = torch.ones(2, 3, device=et.device())
    dim = 0
    index = torch.ones(2, 4, device=et.device(), dtype=torch.int64)

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gather(): materialization failed with: expected the index tensor to have size less than or equal to the input tensor at dimension 1, got 4 vs 3""",
        cpu="""Size does not match at dimension 1 expected index [2, 4] to be no larger than self [2, 3] apart from dimension 0""",
    ):
      torch.gather(inp, dim, index).cpu()

  def test_lerp_int(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""lerp(): expected the first argument's dtype to be non-integral, got int32""",
        cpu=""""lerp_kernel_tensor" not implemented for 'Int'""",
    ):
      torch.lerp(t, t, t)

  def test_mse_loss_invalid_dtypes(self):
    uint8 = torch.ones(2, 2, device=et.device(), dtype=torch.uint8)
    int8 = torch.ones(2, 2, device=et.device(), dtype=torch.int8)
    int16 = torch.ones(2, 2, device=et.device(), dtype=torch.int16)
    int32 = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    int64 = torch.ones(2, 2, device=et.device(), dtype=torch.int64)
    complex64 = torch.ones(2, 2, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): uint8, int8, int16, int32, int64, and complex64 dtypes are not supported, got: uint8""",
        cpu=""""mse_cpu" not implemented for 'Byte'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(uint8, uint8, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): uint8, int8, int16, int32, int64, and complex64 dtypes are not supported, got: int8""",
        cpu=""""mse_cpu" not implemented for 'Char'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(int8, int8, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): uint8, int8, int16, int32, int64, and complex64 dtypes are not supported, got: int16""",
        cpu=""""mse_cpu" not implemented for 'Short'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(int16, int16, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): uint8, int8, int16, int32, int64, and complex64 dtypes are not supported, got: int32""",
        cpu=""""mse_cpu" not implemented for 'Int'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(int32, int32, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): uint8, int8, int16, int32, int64, and complex64 dtypes are not supported, got: int64""",
        cpu=""""mse_cpu" not implemented for 'Long'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(int64, int64, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): uint8, int8, int16, int32, int64, and complex64 dtypes are not supported, got: complex64""",
        cpu=""""mse_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(complex64, complex64, reduction="sum")

  def test_embedding_renorm_float(self):
    inp = torch.ones(3, 2, device=et.device(), dtype=torch.int64)
    indices = torch.ones(10, 2, device=et.device(), dtype=torch.int64)
    max_norm = 1
    norm_type = 2

    with et.assert_raises_message(
        RuntimeError,
        tpu="""embedding_renorm_(): expected floating point or complex, got int64""",
        cpu="""norm(): input dtype should be either floating point or complex. Got Long instead.""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.embedding_renorm_(inp, indices, max_norm, norm_type)

  def test_grid_sampler_invalid_input_dtype(self):
    t = torch.randint(
        0, 10, (2, 3, 4, 4), device=et.device(), dtype=torch.int32
    )
    g = torch.randn(2, 5, 5, 2, device=et.device(), dtype=torch.float32)
    inp2d = torch.ones(1, 1, 2, 2, device=et.device(), dtype=torch.complex64)
    grid2d = torch.zeros(1, 1, 2, 2, device=et.device())
    inp3d = torch.ones(1, 1, 2, 2, 2, device=et.device(), dtype=torch.complex64)
    grid3d = torch.zeros(1, 1, 2, 2, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_2d(): expected the input dtype to be floating point, got int32""",
        cpu=""""grid_sampler_2d_cpu_kernel_impl" not implemented for 'Int'""",
    ):
      torch.ops.aten.grid_sampler_2d(t, g, 0, 0, False)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_2d(): expected the input dtype to be floating point, got complex64""",
        cpu=""""grid_sampler_2d_cpu_kernel_impl" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.grid_sampler(inp2d, grid2d, 0, 0, False)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_3d(): expected the input dtype to be floating point, got complex64""",
        cpu=""""grid_sampler3d_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.grid_sampler(inp3d, grid3d, 0, 0, False)

  def test_native_dropout_invalid_p(self):
    inp = torch.ones(5, 3, device=et.device())

    def check(p: float) -> None:
      with et.assert_raises_message(
          RuntimeError,
          tpu="""dropout(): expected p to be in the range [0, 1], got"""
          f""" {p}""",
          cpu=f"""bernoulli_ expects p to be in [0, 1], but got p={1 - p}""",
          message_reviewed_by="wan",
      ):
        torch.native_dropout(inp, p=p, train=True)

    for p in (-1.5, 1.5):
      with self.subTest(p=p):
        check(p)

  def test_weight_norm_interface_dim(self):
    v = torch.ones(2, 3, 4, 5, device=et.device(), dtype=torch.float32)
    g = torch.ones(3, device=et.device(), dtype=torch.float32)
    dim = 1
    with et.assert_raises_message(
        RuntimeError,
        tpu="""weight_norm_interface(): expected dim to be 0 or the last dimension of v, got 1""",
        cpu=re.compile(
            r"""dim == 0 \|\| dim == v.dim\(\) - 1 INTERNAL ASSERT FAILED.*"""
        ),
    ):
      torch.ops.aten._weight_norm_interface(v, g, dim)

  def test_weight_norm_interface_unsupported_dtype(self):
    v = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    g = torch.ones(2, device=et.device(), dtype=torch.float32)
    dim = 0
    with et.assert_raises_message(
        RuntimeError,
        tpu="""weight_norm_interface(): expected the input dtype to be floating point, got Int""",
        cpu=""""weight_norm_kernel" not implemented for 'Int'""",
    ):
      torch.ops.aten._weight_norm_interface(v, g, dim)

  def test_weight_norm_interface_v_dim_error(self):
    """Tests error message for weight_norm when v.dim() == 0."""
    v = torch.randn((), device=et.device(), dtype=torch.float32)
    g = torch.randn((), device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        tpu="""weight_norm_interface(): expected v to have at least 1 dimension, got 0""",
        cpu="""Dimension specified as 0 but tensor has no dimensions""",
    ):
      torch.ops.aten._weight_norm_interface(v, g, 0)

  def test_softplus_unsupported_dtypes(self):
    t_bool = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    t_uint8 = torch.ones(2, 2, device=et.device(), dtype=torch.uint8)
    t_int8 = torch.ones(2, 2, device=et.device(), dtype=torch.int8)
    t_int16 = torch.ones(2, 2, device=et.device(), dtype=torch.int16)
    t_int32 = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    t_int64 = torch.ones(2, 2, device=et.device(), dtype=torch.int64)
    t_complex64 = torch.ones(2, 2, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got bool""",
        cpu=""""softplus_cpu" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got uint8""",
        cpu=""""softplus_cpu" not implemented for 'Byte'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_uint8)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got int8""",
        cpu=""""softplus_cpu" not implemented for 'Char'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_int8)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got int16""",
        cpu=""""softplus_cpu" not implemented for 'Short'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_int16)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got int32""",
        cpu=""""softplus_cpu" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got int64""",
        cpu=""""softplus_cpu" not implemented for 'Long'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got complex64""",
        cpu=""""softplus_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_complex64)

  def test_softplus_backward_unsupported_dtypes(self):
    t_bool = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    t_int32 = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    t_complex64 = torch.ones(2, 2, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus_backward(): expected the input dtype to be floating-point, got bool""",
        cpu=""""softplus_backward_cpu" not implemented for 'Bool'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.softplus_backward(t_bool, t_bool, 1.0, 20.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus_backward(): expected the input dtype to be floating-point, got int32""",
        cpu=""""softplus_backward_cpu" not implemented for 'Int'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.softplus_backward(t_int32, t_int32, 1.0, 20.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus_backward(): expected the input dtype to be floating-point, got complex64""",
        cpu=""""softplus_backward_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.softplus_backward(t_complex64, t_complex64, 1.0, 20.0)

  def test_softplus_backward_mismatched_dtypes(self):
    if et.TEST_MODE.value == "cpu":
      # CPU pass succeeds and does not raise an exception!
      grad_float = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
      self_bfloat = torch.ones(2, 2, device=et.device(), dtype=torch.bfloat16)
      torch.ops.aten.softplus_backward(grad_float, self_bfloat, 1.0, 20.0)
      return

    # TPU pass raises the strict consistency validation error!
    grad_float = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    self_bfloat = torch.ones(2, 2, device=et.device(), dtype=torch.bfloat16)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus_backward(): expected dtype to be the same across grad_output, self, and grad_input, got grad_output=float32, self=bfloat16, and grad_input=float32""",
    ):
      torch.ops.aten.softplus_backward(grad_float, self_bfloat, 1.0, 20.0)

  def test_hardtanh_unsupported_complex_dtype(self):
    t = torch.ones(2, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh(): expected the input dtype to be non-complex, got complex64""",
        cpu="""clamp is not supported for complex types""",
    ):
      torch.nn.functional.hardtanh(t)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh_backward(): expected the input dtype to be floating point, got complex64""",
        cpu=""""hardshrink_backward_cpu" not implemented for 'ComplexFloat'""",
    ):
      torch.ops.aten.hardtanh_backward(t, t, min_val=0, max_val=1)

  def test_hardtanh_unsupported_bool_dtype(self):
    t = torch.tensor([True, False], device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh(): expected the input dtype to be non-boolean, got bool""",
        cpu="""Bool inputs not supported for hardtanh""",
    ):
      torch.nn.functional.hardtanh(t)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh_backward(): expected the input dtype to be floating point, got bool""",
        cpu=""""hardshrink_backward_cpu" not implemented for 'Bool'""",
    ):
      torch.ops.aten.hardtanh_backward(t, t, min_val=0, max_val=1)

  def test_hardtanh_backward_unsupported_integral_dtypes(self):
    # hardtanh_backward does not support integral types.
    t_uint8 = torch.ones(2, device=et.device(), dtype=torch.uint8)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh_backward(): expected the input dtype to be floating point, got uint8""",
        cpu=""""hardshrink_backward_cpu" not implemented for 'Byte'""",
    ):
      torch.ops.aten.hardtanh_backward(t_uint8, t_uint8, min_val=0, max_val=1)

    t_int32 = torch.ones(2, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh_backward(): expected the input dtype to be floating point, got int32""",
        cpu=""""hardshrink_backward_cpu" not implemented for 'Int'""",
    ):
      torch.ops.aten.hardtanh_backward(t_int32, t_int32, min_val=0, max_val=1)

  def test_hardtanh_unsupported_unsigned_negative_limits(self):
    t = torch.ones(2, device=et.device(), dtype=torch.uint8)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh(): expected positive limit values when executing on an unsigned tensor, got min_val=-1 and max_val=1""",
        cpu="""cannot do hardtanh on an unsigned type with negative limits""",
    ):
      torch.nn.functional.hardtanh(t, min_val=-1, max_val=1)

  def test_leaky_relu_unsupported_bool_dtype(self):
    inp = torch.tensor([True, False], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""leaky_relu(): boolean dtypes are not supported, got Bool""",
        cpu=""""leaky_relu_cpu" not implemented for 'Bool'""",
    ):
      torch.nn.functional.leaky_relu(inp)

  def test_leaky_relu_unsupported_int_dtype(self):
    inp = torch.tensor([1, 2], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""leaky_relu(): integer dtypes are not supported, got Long""",
        cpu=""""leaky_relu_cpu" not implemented for 'Long'""",
    ):
      torch.nn.functional.leaky_relu(inp)

  def test_leaky_relu_unsupported_complex_dtype(self):
    inp = torch.tensor(
        [1 + 1j, 2 + 2j], device=et.device(), dtype=torch.complex64
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu="""leaky_relu(): complex dtypes are not supported, got ComplexFloat""",
        cpu=""""leaky_relu_cpu" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.leaky_relu(inp)

  def test_masked_fill_multi_element_value(self):
    inp = torch.ones(2, 2, device=et.device())
    mask = torch.ones(2, 2, dtype=torch.bool, device=et.device())
    value = torch.ones(2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_fill_(): only supports 1-element value tensors""",
        cpu="""masked_fill_ only supports a 0-dimensional value tensor, but got tensor with 1 dimension(s).""",
    ):
      torch.masked_fill(inp, mask, value)

  def test_masked_fill_type_mismatch(self):
    inp = torch.ones(2, 2, dtype=torch.float32, device=et.device())
    mask = torch.ones(2, 2, dtype=torch.bool, device=et.device())
    value = torch.ones(1, dtype=torch.int32, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_fill_(): value and input must have the same element type""",
        cpu="""masked_fill_ only supports a 0-dimensional value tensor, but got tensor with 1 dimension(s).""",
    ):
      torch.masked_fill(inp, mask, value)

  def test_normal_empty_std_tensor(self):
    device = et.device()
    out_empty = torch.empty(0, device=device)
    torch.normal(mean=0.0, std=torch.tensor([], device=device), out=out_empty)

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          cpu_msg=""""normal_kernel_cpu" not implemented for 'Int'""",
          tpu_msg=(
              "normal_(): expected the self tensor to be floating point or"
              " complex type, got int32"
          ),
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          cpu_msg=""""normal_kernel_cpu" not implemented for 'Long'""",
          tpu_msg=(
              "normal_(): expected the self tensor to be floating point or"
              " complex type, got int64"
          ),
      ),
  )
  def test_normal_errors_invalid_input_dtype(
      self, dtype: torch.dtype, *, cpu_msg: str, tpu_msg: str
  ):
    device = et.device()
    with et.assert_raises_message(
        RuntimeError,
        cpu=cpu_msg,
        tpu=tpu_msg,
    ):
      torch.tensor([1, 2], device=device, dtype=dtype).normal_()

  def test_normal_errors_negative_std_scalar(self):
    device = et.device()
    with et.assert_raises_message(
        RuntimeError,
        cpu="""normal expects std >= 0.0, but found std -1""",
        tpu="""normal_(): expected std >= 0.0, but found std -1""",
    ):
      torch.empty(2, device=device).normal_(mean=0.0, std=-1.0)

  def test_normal_errors_negative_std_tensor(self):
    device = et.device()
    out = torch.empty(2, device=device)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""normal expects all elements of std >= 0.0""",
        tpu="""normal(): expected all elements of std >= 0.0, got min element: -1""",
    ):
      torch.normal(
          mean=0.0, std=torch.tensor([-1.0, 1.0], device=device), out=out
      )

  def test_normal_errors_float_scalar_mean_int32_tensor_std(self):
    std = torch.ones(5, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""normal(): expected the std tensor to be floating point, got int32""",
        cpu=""""normal_kernel_cpu" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.normal(mean=0.0, std=std)

  def test_normal_errors_float_scalar_mean_complex_tensor_std_out(self):
    device = et.device()
    out = torch.empty(2, device=device)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""normal expects standard deviation to be non-complex""",
        tpu="""normal(): expected the std tensor to be floating point, got complex64""",
    ):
      torch.normal(
          mean=0.0,
          std=torch.tensor([1j], device=device, dtype=torch.complex64),
          out=out,
      )

  def test_normal_errors_float_scalar_mean_complex_tensor_std(self):
    device = et.device()
    with et.assert_raises_message(
        RuntimeError,
        cpu="""normal expects standard deviation to be non-complex""",
        tpu="""normal(): expected the std tensor to be floating point, got complex64""",
    ):
      torch.normal(
          mean=0.0,
          std=torch.tensor([1j], device=device, dtype=torch.complex64),
      )

  def test_normal_errors_tensor_mean_complex_tensor_std(self):
    device = et.device()
    with et.assert_raises_message(
        RuntimeError,
        cpu="""normal expects standard deviation to be non-complex""",
        tpu="""normal(): expected the std tensor to be non-complex, got complex64""",
    ):
      torch.normal(
          mean=torch.tensor([0.0], device=device),
          std=torch.tensor([1j], device=device, dtype=torch.complex64),
      )

  def test_normal_errors_mismatched_shapes_functional(self):
    device = et.device()
    with et.assert_raises_message(
        RuntimeError,
        cpu="""The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0""",
        tpu="""The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0""",
    ):
      torch.normal(
          mean=torch.zeros(2, device=device), std=torch.ones(3, device=device)
      )

  def test_normal_errors_invalid_mean_dtype(self):
    device = et.device()
    with et.assert_raises_message(
        (RuntimeError, NotImplementedError),
        cpu=""""normal_kernel_cpu" not implemented for 'Int'""",
        tpu="""normal(): expected the mean tensor to be floating point or complex type, got int32""",
    ):
      torch.normal(
          mean=torch.tensor([1], device=device, dtype=torch.int32), std=1.0
      )

  def test_histc_complex(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.complex64)
    bins = 10
    min_val = 0
    max_val = 1
    out = torch.empty(bins, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""histc(): expected the first argument not to be complex, got complex64""",
        cpu="""torch.histogram: input tensor and hist tensor should have the same dtype, but got input c10::complex<float> and hist float""",
        message_reviewed_by="wan",
    ):
      torch.histc(t, bins=bins, min=min_val, max=max_val, out=out)

  def test_linalg_solve_triangular_dim_mismatch(self):
    a = torch.ones(3, 3, device=et.device())
    b = torch.ones(3, device=et.device())
    out = torch.empty(3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_solve_triangular(): expected the two inputs to have the same number of dimensions, got 2 and 1""",
        cpu="""linalg.solve_triangular: The input tensor B must have at least 2 dimensions.""",
        message_reviewed_by="wan",
    ):
      torch.linalg.solve_triangular(a, b, upper=True, out=out)

  def test_linalg_solve_triangular_int32(self):
    a = torch.ones(3, 3, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 3, device=et.device(), dtype=torch.int32)
    out = torch.empty(3, 3, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_solve_triangular(): triangular solve not supported for dtype int32""",
        cpu=""""triangular_solve_cpu" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.linalg.solve_triangular(a, b, upper=True, out=out)

  def test_masked_select_mask_int32(self):
    self_tensor = torch.ones(5, device=et.device())
    mask = torch.ones(5, device=et.device(), dtype=torch.int32)
    out = torch.empty(0, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_select(): expected the mask to be bool, got int32""",
        cpu="""masked_select: expected BoolTensor for mask""",
        message_reviewed_by="wan",
    ):
      torch.masked_select(self_tensor, mask, out=out)

  def test_mse_loss_invalid_reduction(self):
    inp = torch.ones(2, 2, device=et.device())
    target = torch.ones(2, 2, device=et.device())
    out = torch.empty(1, device=et.device())

    # Reduction 3 is invalid (valid are 0: None, 1: Mean, 2: Sum).
    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): unrecognized reduction mode 3""",
        cpu=re.compile(
            r"""reduction == Reduction::Mean \|\| reduction == Reduction::Sum INTERNAL ASSERT FAILED at .*"""
        ),
    ):
      torch.ops.aten.mse_loss(inp, target, 3, out=out)

  def test_nll_loss_invalid_target_dtype(self):
    inp = torch.ones(2, 2, device=et.device())
    target = torch.ones(2, device=et.device(), dtype=torch.int32)
    output = torch.empty(1, device=et.device())
    total_weight = torch.empty(1, device=et.device())

    weight = None
    reduction = 1
    ignore_index = -100

    with et.assert_raises_message(
        RuntimeError,
        tpu="""nll_loss_forward(): expected the target dtype to be either int64 or uint8, got int32""",
        cpu="""expected target dtype to be Long or Byte, but got Int""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.nll_loss_forward(
          inp,
          target,
          weight,
          reduction,
          ignore_index,
          output=output,
          total_weight=total_weight,
      )

  def test_adaptive_avg_pool2d_invalid_rank(self):
    inp = torch.ones(10, 10, device=et.device())
    out = torch.empty(5, 5, device=et.device())

    # adaptive_avg_pool2d expects 3-D or 4-D input.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): input must be a 3-D or 4-D tensor, got 2-D tensor""",
        cpu="""adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got [10, 10]""",
    ):
      torch.ops.aten.adaptive_avg_pool2d.out(inp, tuple(out.shape), out=out)

  def test_adaptive_avg_pool3d_invalid_rank(self):
    inp = torch.ones(10, 10, 10, device=et.device())
    out = torch.empty(5, 5, 5, device=et.device())

    # adaptive_avg_pool3d expects 4-D or 5-D input.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): input must be a 4-D or 5-D tensor, got 3-D tensor""",
        cpu="""adaptive_avg_pool3d(): Expected 4D or 5D tensor, but got [10, 10, 10]""",
    ):
      torch.ops.aten.adaptive_avg_pool3d.out(inp, tuple(out.shape), out=out)

  def test_max_pool2d_with_indices_invalid_rank(self):
    inp = torch.ones(10, 10, device=et.device())

    out = torch.empty(1, device=et.device())
    indices = torch.empty(1, device=et.device(), dtype=torch.int64)

    kernel_size = [3, 3]
    stride = [2, 2]
    padding = [1, 1]
    dilation = [1, 1]
    ceil_mode = False

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""max_pool2d_with_indices(): materialization failed with: input must be a 3-D or 4-D tensor, got 2-D tensor""",
        cpu="""non-empty 3D or 4D (batch mode) tensor expected for input""",
    ):
      out, indices = torch.ops.aten.max_pool2d_with_indices.out(
          inp,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          out=out,
          indices=indices,
      )

      out.cpu()
      indices.cpu()

  def test_pooling_create_batch_input_invalid_rank(self):
    inp = torch.ones(10, 10, device=et.device())
    out = torch.empty(1, device=et.device())

    kernel_size = [3, 3]
    stride = [2, 2]
    padding = [1, 1]
    ceil_mode = False
    count_include_pad = True
    divisor_override = None

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        (RuntimeError, IndexError),
        tpu="""avg_pool2d(): materialization failed with: input must be a 3-D or 4-D tensor, got 2-D tensor""",
        cpu="""Dimension out of range (expected to be in range of [-2, 1], but got -3)""",
    ):
      torch.ops.aten.avg_pool2d.out(
          inp,
          kernel_size,
          stride,
          padding,
          ceil_mode,
          count_include_pad,
          divisor_override,
          out=out,
      )

      out.cpu()

  def test_reflection_pad2d_backward_shape_mismatch(self):
    inp = torch.randn(1, 1, 4, 4, device=et.device())
    grad_output = torch.randn(1, 1, 7, 7, device=et.device())
    padding = [1, 1, 1, 1]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""reflection_pad2d_backward(): expected the input shape to match the output (input grad) shape [1, 1, 5, 5] computed by removing the padding from the grad_output, got [1, 1, 4, 4]""",
        cpu="""gradOutput width unexpected. Expected: 6, Got: 7""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.reflection_pad2d_backward(grad_output, inp, padding)

  def test_replication_pad2d_backward_shape_mismatch(self):
    inp = torch.randn(1, 1, 4, 4, device=et.device())
    grad_output = torch.randn(1, 1, 7, 7, device=et.device())
    padding = [1, 1, 1, 1]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad2d_backward(): expected the input shape to match the output (input grad) shape [1, 1, 5, 5] computed by removing the padding from grad_output, got [1, 1, 4, 4]""",
        cpu="""gradOutput width unexpected. Expected: 6, Got: 7""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.replication_pad2d_backward(grad_output, inp, padding)

  def test_replication_pad3d_backward_shape_mismatch(self):
    inp = torch.randn(1, 1, 4, 4, 4, device=et.device())
    grad_output = torch.randn(1, 1, 7, 7, 7, device=et.device())
    padding = [1, 1, 1, 1, 1, 1]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad3d_backward(): expected the input shape to match the output (input grad) shape [1, 1, 5, 5, 5] computed by removing the padding from the grad_output, got [1, 1, 4, 4, 4]""",
        cpu="""gradOutput width unexpected. Expected: 6, Got: 7""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.replication_pad3d_backward(grad_output, inp, padding)

  def test_replication_pad3d_backward_invalid_padding_width(self):
    inp = torch.randn(1, 1, 4, 4, 4, device=et.device())
    grad_output = torch.randn(1, 1, 6, 6, 6, device=et.device())
    padding = [4, 4, 1, 1, 1, 1]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad3d_backward(): expected padding at indices 0 and 1 to sum to a value smaller than the grad_output width (at dimension 4) of 6, got 8 (4 + 4)""",
        cpu="""gradOutput width unexpected. Expected: 12, Got: 6""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.replication_pad3d_backward(grad_output, inp, padding)

  def test_replication_pad3d_backward_invalid_padding_height(self):
    inp = torch.randn(1, 1, 4, 4, 4, device=et.device())
    grad_output = torch.randn(1, 1, 6, 6, 6, device=et.device())
    padding = [1, 1, 4, 4, 1, 1]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad3d_backward(): expected padding at indices 2 and 3 to sum to a value smaller than the grad_output height (at dimension 3) of 6, got 8 (4 + 4)""",
        cpu="""gradOutput height unexpected. Expected: 12, Got: 6""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.replication_pad3d_backward(grad_output, inp, padding)

  def test_replication_pad3d_backward_invalid_padding_depth(self):
    inp = torch.randn(1, 1, 4, 4, 4, device=et.device())
    grad_output = torch.randn(1, 1, 6, 6, 6, device=et.device())
    padding = [1, 1, 1, 1, 4, 4]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad3d_backward(): expected padding at indices 4 and 5 to sum to a value smaller than the grad_output depth (at dimension 2) of 6, got 8 (4 + 4)""",
        cpu="""gradOutput depth unexpected. Expected: 12, Got: 6""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.replication_pad3d_backward(grad_output, inp, padding)

  def test_scaled_dot_product_attention_no_backends(self):
    query = torch.randn(1, 1, 512, 128, device=et.device())
    key = torch.randn(1, 1, 512, 128, device=et.device())
    value = torch.randn(1, 1, 512, 128, device=et.device())

    with torch.nn.attention.sdpa_kernel(backends=[]):
      with et.assert_raises_message(
          RuntimeError,
          tpu="""fused_sdp_choice(): no viable SDPBackend found: all supported backends are disabled, including the fallback MATH backend; enable at least one of FLASH, OVERRIDEABLE, or MATH for TorchTPU""",
          cpu="""No viable backend for scaled_dot_product_attention was found. This is likely due to turning off both the math kernel and the fused kernels.""",
          message_reviewed_by="wan",
      ):
        torch.nn.functional.scaled_dot_product_attention(query, key, value)

  def test_tril_indices_unsupported_dtype(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""tril_indices(): expected the dtype to be either int32 or int64, got float32""",
        cpu=""""tril_indices" not implemented for 'Float'""",
        message_reviewed_by="wan",
    ):
      torch.tril_indices(3, 3, dtype=torch.float32, device=et.device())

  def test_silu_unsupported_dtype_int(self):
    t = torch.ones(5, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""silu(): materialization failed with: expected the input dtype to be floating point, got int32""",
        cpu=""""silu_cpu" not implemented for 'Int'""",
    ):
      out = torch.nn.functional.silu(t)
      out.cpu()

  def test_acos_out_dtype_mismatch(self):
    t = torch.ones(5, device=et.device())

    # Call the out variant.
    out = torch.ones(5, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""acos(): expected the output dtype to be float32, got int32""",
        cpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch.acos(t, out=out)

  def test_sign_unsupported_dtype_complex(self):
    t = torch.tensor([1 + 1j], device=et.device())

    # Call the out variant.
    out = torch.ones(1, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sign(): expected the input dtype not to be complex, got complex64; use torch.sgn() instead if you intend to normalize a complex tensor to each complex element having magnitude 1""",
        cpu="""Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.""",
        message_reviewed_by="wan",
    ):
      torch.sign(t, out=out)

  def test_scatter_rank_src_rank_mismatch(self):
    self_t = torch.ones(5, 5, device=et.device())
    index = torch.zeros(5, 5, dtype=torch.int64, device=et.device())
    src = torch.ones(5, device=et.device())

    # Call the out overload.
    out = torch.empty(5, 5, device=et.device())

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        (RuntimeError, IndexError),
        tpu="""scatter(): materialization failed with: expected the self tensor of shape [5, 5] to have the same rank as the src tensor of shape [5], got 2 vs. 1""",
        cpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
    ):
      torch.scatter(self_t, 0, index, src, out=out).cpu()

  def test_scatter_index_rank_mismatch(self):
    self_t = torch.ones(5, 5, device=et.device())
    index = torch.zeros(5, dtype=torch.int64, device=et.device())
    src = torch.ones(5, 5, device=et.device())

    # Call the out overload.
    out = torch.empty(5, 5, device=et.device())

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""scatter(): materialization failed with: expected the self tensor of shape [5, 5] to have the same rank as the index tensor of shape [5], got 2 vs. 1""",
        cpu="""Index tensor must have the same number of dimensions as self tensor""",
    ):
      torch.scatter(self_t, 0, index, src, out=out).cpu()

  def test_scatter_rank2_self_scalar_src_tensor(self):
    self_t = torch.zeros(5, 5, device=et.device())
    index = torch.zeros(5, 5, dtype=torch.int64, device=et.device())
    src = torch.tensor(1.0, device=et.device())

    # PyTorch does NOT allow a 0D tensor as src for scatter if index is not 0D.
    # It only allows Python scalars (via a different overload).
    # Both TPU and CPU now raise the same eager error message.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Index tensor must have the same number of dimensions as src tensor""",
    ):
      torch.scatter(self_t, 0, index, src).cpu()

  def test_softmax_backward_data_shape_mismatch(self):
    grad_output = torch.ones(5, 5, device=et.device())
    output = torch.ones(5, device=et.device())

    # Call the out overload.
    grad_input = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softmax_backward_data(): materialization failed with: expected grad_output and output arguments to have the same shape, got [5, 5] vs. [5]""",
        cpu="""Expected tensor for argument #1 'grad' to have same size as tensor for argument #2 'output'; but [5, 5] does not equal [5] (while checking arguments for softmax_backward)""",
    ):
      torch.ops.aten._softmax_backward_data(
          grad_output, output, 0, torch.float32, grad_input=grad_input
      ).cpu()

  def test_as_strided_negative_offset(self):
    t = torch.empty(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""as_strided(): expected the given storage offset to be >= 0, got -1""",
        cpu="""Tensor: invalid storage offset -1""",
        message_reviewed_by="wan",
    ):
      torch.as_strided(t, (1,), (1,), storage_offset=-1)

  def test_as_strided_size_stride_mismatch(self):
    t = torch.empty(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""as_strided(): expected the given sizes [1, 2] and strides [1] to have the same length, got 2 vs. 1""",
        cpu="""mismatch in length of strides and shape""",
        message_reviewed_by="wan",
    ):
      torch.as_strided(t, (1, 2), (1,))

  def test_as_strided_undefined_tensor(self):
    # This error message is generated by PyTorch before reaching the backend
    # kernel.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Expected a proper Tensor but got None (or an undefined Tensor in C++) for argument #0 'self'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.as_strided(None, (1,), (1,))

  def test_scatter_unsupported_reduction(self):
    """Tests the scatter op with unsupported reduction op."""
    # This error message is generated by PyTorch.
    self_tensor = torch.zeros(3, 5, device=et.device())
    index = torch.tensor([[0, 1], [2, 3], [0, 4]], device=et.device())
    src = torch.ones(3, 2, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""reduce argument must be either add or multiply.""",
        message_reviewed_by="gunhyun",
    ):
      self_tensor.scatter(1, index, src, reduce="invalid")

  def test_scatter_reduce_unsupported_reduction(self):
    """Tests the scatter_reduce op with unsupported reduction op."""
    # This error message is generated by PyTorch.
    self_tensor = torch.zeros(3, 5, device=et.device())
    index = torch.tensor([[0, 1], [2, 3], [0, 4]], device=et.device())
    src = torch.ones(3, 2, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""reduce argument must be either sum, prod, mean, amax or amin, got invalid""",
        message_reviewed_by="gunhyun",
    ):
      self_tensor.scatter_reduce(1, index, src, reduce="invalid")

  def test_view_invalid_negative_dimension(self):
    t = torch.ones(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): expected the given sizes [-2] to be >= -1, got 1 invalid size: -2 at index 0""",
        cpu="""invalid shape dimension -2 at index 0 of shape [-2]""",
        message_reviewed_by="wan",
    ):
      t.view(-2)

  def test_view_multiple_neg1(self):
    t = torch.ones(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): expected the given sizes [1, -1, 5, -1] to have up to 1 element equal to -1 (inferred dimension), got 2 occurrences of -1 at indices 1 and 3""",
        cpu="""only one dimension can be inferred""",
        message_reviewed_by="wan",
    ):
      t.view(1, -1, 5, -1)

  def test_view_infer_dimension_0_numel(self):
    t = torch.ones(0, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): cannot infer the dimension for a 0-element view of shape [0, -1] because it's ambiguous, i.e. it could be of any value""",
        cpu="""cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous""",
        message_reviewed_by="wan",
    ):
      t.view(0, -1)

  def test_view_infer_dimension_not_multiple(self):
    t = torch.ones(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): expected the number of elements in the output view of shape [-1, 2] to be a multiple of the number of elements in the input of shape [5] in the presence of an inferred dimension (-1), got 2, which is not a multiple of 5""",
        cpu="""shape '[-1, 2]' is invalid for input of size 5""",
        message_reviewed_by="wan",
    ):
      t.view(-1, 2)

  def test_view_numel_mismatch(self):
    t = torch.ones(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): expected the input of shape [5] to have the same number of elements as the output of shape [2], got 5 vs. 2""",
        cpu="""shape '[2]' is invalid for input of size 5""",
        message_reviewed_by="wan",
    ):
      t.view(2)

  def test_view_not_compatible(self):
    t = torch.ones(2, 3, device=et.device()).T

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): cannot create a view of shape [6] from the input tensor of shape [3, 2] and strides [1, 3]; consider creating a new tensor using reshape() instead of taking a view""",
        cpu="""view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.""",
        message_reviewed_by="wan",
    ):
      t.view(6)

  def test_view_as_complex_unsupported_dtypes_int(self):
    t = torch.ones(2, 3, 2, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the input dtype to be float32 or float64, got int32""",
        cpu="""view_as_complex is only supported for half, float and double tensors, but got a tensor of scalar type: Int""",
    ):
      torch.view_as_complex(t)

  def test_view_as_complex_scalar(self):
    t = torch.tensor(1.0, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the input to be a tensor, got a scalar""",
        cpu="""Input tensor must have one or more dimensions""",
        message_reviewed_by="wan",
    ):
      torch.view_as_complex(t)

  def test_view_as_complex_invalid_last_dim(self):
    t = torch.ones(3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the size of the last dimension of the input tensor to be 2, got 3""",
        cpu="""Tensor must have a last dimension of size 2""",
        message_reviewed_by="wan",
    ):
      torch.view_as_complex(t)

  def test_view_as_complex_invalid_last_stride(self):
    t = torch.ones(2, 2, device=et.device()).T

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the stride of the last dimension of the input tensor to be 1, got 2""",
        cpu="""Tensor must have a last dimension with stride 1""",
        message_reviewed_by="wan",
    ):
      torch.view_as_complex(t)

  def test_view_as_complex_invalid_stride(self):
    t = torch.as_strided(torch.ones(5, device=et.device()), (2, 2), (3, 1))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the input strides [3, 1] to be even numbers (except in the last dimension), got 1 odd stride: 3 at index 0""",
        cpu="""Tensor must have a stride divisible by 2 for all but last dimension""",
        message_reviewed_by="wan",
    ):
      torch.view_as_complex(t)

  def test_where_out_dtype_mismatch(self):
    condition = torch.tensor([True, False], device=et.device())
    inp = torch.ones(2, device=et.device(), dtype=torch.float32)
    other = torch.zeros(2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""where(): expected the output dtype to be float32 (result of promoting the dtype of the input tensors -- float32 and float32), got int32""",
        cpu="""Expected out type to be Float but got Int""",
        message_reviewed_by="wan",
    ):
      torch.where(condition, inp, other, out=out)

  def test_bucketize_unsupported_complex_dtype_1(self):
    input_tensor = torch.tensor([1 + 1j], device=et.device())
    boundaries = torch.tensor([0.5], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu=""""searchsorted_out_cpu" not implemented for 'ComplexFloat'""",
        tpu="""bucketize(): self must not be complex, got 'ComplexFloat'""",
    ):
      torch.bucketize(input_tensor, boundaries)

  def test_bucketize_unsupported_complex_dtype_2(self):
    input_tensor = torch.tensor([0.5], device=et.device())
    boundaries = torch.tensor([1 + 1j], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu=""""searchsorted_out_cpu" not implemented for 'ComplexFloat'""",
        tpu="""bucketize(): boundaries must not be complex, got 'ComplexFloat'""",
    ):
      torch.bucketize(input_tensor, boundaries)

  def test_bucketize_invalid_boundaries_dim(self):
    input_tensor = torch.tensor([1.0, 2.0], device=et.device())
    boundaries = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu="""boundaries tensor must be 1 dimension, but got dim(2)""",
        tpu="""bucketize(): boundaries tensor must be 1 dimension, got dim(2)""",
    ):
      torch.bucketize(input_tensor, boundaries)

  def test_geqrf_insufficient_dims(self):
    input_tensor = torch.ones(1, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="""torch.geqrf: input must have at least 2 dimensions.""",
        tpu="""geqrf(): expected input to have at least 2 dimensions, got 1""",
    ):
      torch.geqrf(input_tensor)

  def test_xlogy_unsupported_complex_dtype(self):
    complex_val = torch.complex(torch.tensor(1.0), torch.tensor(1.0))
    complex_val = complex_val.to(et.device())

    x = complex_val.clone()
    y = complex_val.clone()
    out = torch.empty_like(x)
    with et.assert_raises_message(
        RuntimeError,
        cpu=""""xlogy_cpu" not implemented for 'ComplexFloat'""",
        tpu="""xlogy(): complex dtypes are not supported, got x dtype complex64 and y dtype complex64""",
    ):
      torch.ops.aten.xlogy.OutTensor(x, y, out=out)

  def test_linalg_qr_insufficient_dims(self):
    input_tensor = torch.ones(1, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by PyTorch. We don't have a good way to
        # replace it.
        tpu="""linalg.qr: The input tensor A must have at least 2 dimensions.""",
        message_reviewed_by="gunhyun",
    ):
      torch.linalg.qr(input_tensor)

  def test_linalg_qr_invalid_mode(self):
    input_tensor = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by PyTorch. We don't have a good way to
        # replace it.
        tpu="""qr received unrecognized mode 'invalid' but expected one of 'reduced' (default), 'r', or 'complete'""",
        message_reviewed_by="gunhyun",
    ):
      torch.linalg.qr(input_tensor, mode="invalid")

  def test_log_sigmoid_backward_invalid_bool_dtype(self):
    t = torch.ones(4, device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""log_sigmoid_backward(): expected the input dtype to be floating point, got bool""",
        cpu=""""log_sigmoid_backward_cpu" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.log_sigmoid_backward(t, t, t)

  def test_log_sigmoid_backward_invalid_complex_dtype(self):
    t = torch.ones(4, device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""log_sigmoid_backward(): expected the input dtype to be floating point, got complex64""",
        cpu=""""log_sigmoid_backward_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.log_sigmoid_backward(t, t, t)

  def test_log_sigmoid_invalid_bool_dtype(self):
    t = torch.ones(4, device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""log_sigmoid_forward(): expected the input dtype to be floating point, got bool""",
        cpu=""""log_sigmoid_cpu" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.log_sigmoid(t)

  def test_log_sigmoid_invalid_complex_dtype(self):
    t = torch.ones(4, device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""log_sigmoid_forward(): expected the input dtype to be floating point, got complex64""",
        cpu=""""log_sigmoid_cpu" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.log_sigmoid(t)

  def test_scaled_mm_non_2d_self(self):
    """Tests that scaled_mm fails if self is not 2D."""
    device = et.device()
    mat1 = torch.randn(16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.tensor([1.0], dtype=torch.float32, device=device)
    scale_b = torch.tensor([1.0], dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_mm(): expected the self argument to be a 2D tensor (matrix), got 1D of shape [16]""",
        cpu="""mat1 must be a matrix""",
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b)

  def test_scaled_mm_invalid_scale_a_size(self):
    """Tests that scaled_mm fails if scale_a has numel > 1."""
    device = et.device()
    mat1 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    scale_b = torch.tensor([1.0], dtype=torch.float32, device=device)

    err_type = NotImplementedError if et.device().type == "tpu" else ValueError
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm(): expected scale_a to have numel 1, got numel 2""",
        cpu=re.compile(r"""Invalid scaling configuration\..*""", re.DOTALL),
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b)

  def test_scaled_mm_invalid_scale_b_size(self):
    """Tests that scaled_mm fails if scale_b has numel > 1."""
    device = et.device()
    mat1 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.tensor([1.0], dtype=torch.float32, device=device)
    scale_b = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)

    err_type = NotImplementedError if et.device().type == "tpu" else ValueError
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm(): expected scale_b to have numel 1, got numel 2""",
        cpu=re.compile(r"""Invalid scaling configuration\..*""", re.DOTALL),
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b)

  def test_scaled_mm_invalid_scale_result_size(self):
    """Tests that scaled_mm fails if scale_result has numel > 1."""
    device = et.device()
    mat1 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.tensor([1.0], dtype=torch.float32, device=device)
    scale_b = torch.tensor([1.0], dtype=torch.float32, device=device)
    scale_result = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)

    err_type = (
        NotImplementedError if et.device().type == "tpu" else RuntimeError
    )
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm(): expected scale_result to have numel 1, got numel 2""",
        cpu="""scale_result must be a float scalar""",
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b, scale_result=scale_result)

  def test_scaled_mm_incompatible_shapes(self):
    """Tests that scaled_mm fails if matrix shapes are incompatible."""
    device = et.device()
    mat1 = torch.randn(16, 32, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 32, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.tensor([1.0], dtype=torch.float32, device=device)
    scale_b = torch.tensor([1.0], dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_mm(): expected column size of first matrix to match row size of second matrix, got shapes [16, 32] and [16, 32]""",
        cpu="""mat_a and mat_b shapes cannot be multiplied (16x32 and 16x32)""",
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b)

  def test_fft_c2c_non_complex_input(self):
    t = torch.ones(4, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""fft_c2c(): materialization failed with: expected complex input type, got float32""",
        cpu="""Only supports complex dtypes, but found: Float""",
    ):
      torch.ops.aten._fft_c2c(t, [0], 0, True)

  def test_fft_c2r_non_complex_input(self):
    t = torch.ones(4, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""fft_c2r(): materialization failed with: expected complex input type, got float32""",
        cpu="""expected scalar type ComplexDouble but found Float""",
    ):
      torch.ops.aten._fft_c2r(t, [0], 0, 4)

  def test_dunder_lshift_unsupported_dtype(self):
    x = torch.tensor([1.0], device=et.device())
    y = torch.tensor([1.0], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""__lshift__(): expected the dtype of the first argument to be integer, got float32""",
        cpu=""""lshift_cpu" not implemented for 'Float'""",
        message_reviewed_by="wan",
    ):
      x << y  # pylint: disable=pointless-statement

  def test_dunder_rshift_unsupported_dtype(self):
    x = torch.tensor([1.0], device=et.device())
    y = torch.tensor([1.0], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""__rshift__(): expected the dtype of the first argument to be integer, got float32""",
        cpu=""""rshift_cpu" not implemented for 'Float'""",
        message_reviewed_by="wan",
    ):
      x >> y  # pylint: disable=pointless-statement


if __name__ == "__main__":
  absltest.main()
