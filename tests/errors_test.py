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

"""Tests error handling on TPU vs on GPU."""

import re
from typing import Any
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import env  # pylint: disable=unused-import
from torch_tpu._internal.distributed import multiprocessing
from tests import error_testing as et
from tests import oss_utils  # pylint: disable=unused-import  # noqa: F401

_TEST_MODE = et.TEST_MODE

# Regex used by: TpuVsGpuErrorTest.test_index_no_indices
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


def _get_grid_sampler_inputs(dims: int) -> tuple[torch.Tensor, torch.Tensor]:
  """Creates the inputs for torch.grid_samplerXd(), where X is the number of dimensions."""
  inp_shape = [1, 1] + [2] * dims
  inp = torch.ones(inp_shape, device=et.device(), dtype=torch.complex64)

  grid_shape = [1] + [2] * dims + [dims]
  grid = torch.zeros(grid_shape, device=et.device())

  return (inp, grid)


class TpuVsGpuErrorTest(et.ErrorTestBase):
  """Tests error messages on TPU vs on GPU."""

  def test_triu_insufficient_dims(self):
    """Tests that triu with insufficient dims fails with expected error."""
    t = torch.ones(1, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""triu: input tensor must have at least 2 dimensions""",
    ):
      torch.triu(t, 1)

  def test_col2im_unsupported_integer_dtype(self):
    """Tests that col2im with integer dtype fails with expected error."""
    t = torch.ones((1, 4, 1), device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected non-integer dtype, got int32""",
        gpu=""""col2im_out_cuda" not implemented for 'Int'""",
    ):
      torch.ops.aten.col2im(t, (2, 2), (2, 2), (1, 1), (0, 0), (1, 1))

  def test_segment_reduce_indices_not_supported(self):
    """Tests that segment_reduce with indices raises expected error."""
    t = torch.ones(5, device=et.device(), dtype=torch.float32)
    indices = torch.tensor(
        [0, 0, 1, 1, 2], device=et.device(), dtype=torch.long
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu="""segment_reduce(): expected lengths or offsets for reduction, got indices""",
        gpu=re.compile(
            r".*segment_reduce\(\): indices based reduction is not supported"
            r" yet\..*"
        ),
        message_reviewed_by="gunhyun",
    ):
      torch.segment_reduce(t, "sum", indices=indices)

  def test_segment_reduce_neither_lengths_nor_offsets(self):
    """Tests that segment_reduce without lengths or offsets raises expected error."""
    t = torch.ones(5, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""segment_reduce(): expected lengths or offsets to be defined, got neither""",
        gpu=re.compile(
            r".*segment_reduce\(\): Either lengths or offsets must be"
            r" defined\..*"
        ),
        message_reviewed_by="gunhyun",
    ):
      torch.segment_reduce(t, "sum")

  def test_segment_reduce_invalid_reduce_mode(self):
    """Tests that segment_reduce with invalid reduce mode raises expected error."""
    t = torch.ones(5, device=et.device(), dtype=torch.float32)
    lengths = torch.tensor([2, 3], device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""segment_reduce(): expected reduce mode to be sum, mean, max, min, or prod, got invalid_mode""",
        gpu=re.compile(
            r".*reduce argument must be either sum, prod, mean, amax or amin,"
            r" got invalid_mode.*"
        ),
        message_reviewed_by="gunhyun",
    ):
      torch.segment_reduce(t, "invalid_mode", lengths=lengths)

  def test_segment_reduce_scalar_tensor_invalid_rank(self):
    """Tests that segment_reduce with a 0-D scalar tensor raises expected error."""
    t = torch.tensor(5.0, device=et.device(), dtype=torch.float32)
    lengths = torch.tensor([1], device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""segment_reduce(): expected input with at least 1 dim, got input with 0 dims""",
        gpu=re.compile(r".*Expected data\.dim\(\) >= lengths_value\.dim\(\).*"),
        message_reviewed_by="gunhyun",
    ):
      torch.segment_reduce(t, "sum", lengths=lengths)

  def test_segment_reduce_invalid_axis(self):
    """Tests that segment_reduce with out-of-bounds axis raises expected error."""
    t = torch.ones(5, 5, device=et.device(), dtype=torch.float32)
    lengths = torch.tensor([2, 3], device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        IndexError,
        tpu="""segment_reduce(): expected dimension to be in range [-2, 1], got 10""",
        gpu=re.compile(r".*Dimension out of range.*"),
        message_reviewed_by="gunhyun",
    ):
      torch.segment_reduce(t, "sum", lengths=lengths, axis=10)

  def test_segment_reduce_data_dim_smaller_than_lengths_dim(self):
    """Tests segment_reduce with data dim smaller than lengths dim."""
    t = torch.ones(5, device=et.device(), dtype=torch.float32)
    lengths = torch.ones(2, 3, device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""segment_reduce(): expected data dim >= lengths/offsets dim, got 1""",
        gpu=re.compile(r".*Expected data\.dim\(\) >= lengths_value\.dim\(\).*"),
        message_reviewed_by="gunhyun",
    ):
      torch.segment_reduce(t, "sum", lengths=lengths)

  def test_segment_reduce_axis_not_last_dim_of_lengths(self):
    """Tests segment_reduce where axis is not the last dimension of lengths."""
    t = torch.ones(5, 5, 5, device=et.device(), dtype=torch.float32)
    lengths = torch.ones(2, 2, device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""segment_reduce(): expected axis to be the last dimension of lengths/offsets, got 0""",
        gpu=re.compile(
            r".*segment_reduce\(\): Expected axis to be the last dimension of"
            r" lengths but got 0\..*"
        ),
        message_reviewed_by="gunhyun",
    ):
      torch.segment_reduce(t, "sum", lengths=lengths, axis=0)

  def test_segment_reduce_mismatched_outer_dimension(self):
    """Tests segment_reduce with mismatched outer dimensions between data and lengths."""
    t = torch.ones(2, 6, device=et.device(), dtype=torch.float32)
    lengths = torch.ones(3, 2, device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""segment_reduce(): expected outer dimension 0 to match between data and lengths/offsets, got 2 and 3""",
        gpu=re.compile(
            r".*segment_reduce\(\): Expected all rows of lengths along axis to"
            r" sum to data\.size\(lengths\.dim\(\)-1\) when !unsafe\..*"
        ),
        message_reviewed_by="gunhyun",
    ):
      torch.segment_reduce(t, "sum", lengths=lengths, axis=1)

  def test_upsample_bicubic2d_invalid_rank(self):
    t = torch.ones(1, 2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""It is expected input_size equals to 4, but got size 3""",
        tpu="""It is expected input_size equals to 4, but got size 3""",
    ):
      torch.ops.aten.upsample_bicubic2d(t, [10, 10], False)

  def test_upsample_bicubic2d_invalid_output_size(self):
    t = torch.ones(1, 1, 4, 4, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""It is expected output_size equals to 2, but got size 1""",
        tpu="""It is expected output_size equals to 2, but got size 1""",
    ):
      torch.ops.aten.upsample_bicubic2d(t, [10], False)

  def test_upsample_bicubic2d_dtype_mismatch(self):
    t = torch.ones(1, 1, 4, 4, device=et.device(), dtype=torch.float32)
    out = torch.empty(1, 1, 10, 10, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""upsample_bicubic2d(): expected out dtype float32, got int32""",
        gpu="""Expected out tensor to have dtype float, but got int instead""",
    ):
      torch.ops.aten.upsample_bicubic2d.out(
          t, [10, 10], False, None, None, out=out
      )

  def test_upsample_bicubic2d_backward_invalid_grad_output_rank(self):
    grad_output = torch.ones(1, 2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Expected grad_output to be a tensor of dimension 4 but got: dimension 3""",
        tpu="""Expected grad_output to be a tensor of dimension 4 but got: dimension 3""",
    ):
      torch.ops.aten.upsample_bicubic2d_backward(
          grad_output, [10, 10], [1, 1, 4, 4], False
      )

  def test_upsample_bicubic2d_backward_invalid_input_size(self):
    grad_output = torch.ones(
        1, 1, 10, 10, device=et.device(), dtype=torch.float32
    )
    with et.assert_raises_message(
        RuntimeError,
        gpu="""It is expected input_size equals to 4, but got size 3""",
        tpu="""It is expected input_size equals to 4, but got size 3""",
    ):
      torch.ops.aten.upsample_bicubic2d_backward(
          grad_output, [10, 10], [1, 1, 4], False
      )

  def test_upsample_bicubic2d_backward_dtype_mismatch(self):
    grad_output = torch.ones(
        1, 1, 10, 10, device=et.device(), dtype=torch.float32
    )
    grad_input = torch.empty(1, 1, 4, 4, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""upsample_bicubic2d_backward(): expected grad_input dtype float32, got int32""",
        gpu="""Expected out tensor to have dtype float, but got int instead""",
    ):
      torch.ops.aten.upsample_bicubic2d_backward.grad_input(
          grad_output,
          [10, 10],
          [1, 1, 4, 4],
          False,
          None,
          None,
          grad_input=grad_input,
      )

  def test_ctc_loss_log_probs_3d(self):
    log_probs = torch.randn(2, 3, device=et.device())
    targets = torch.randint(1, 3, (2, 3), dtype=torch.int32, device=et.device())
    input_lengths = torch.tensor([2, 2], dtype=torch.int32, device=et.device())
    target_lengths = torch.tensor([3, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Expected 3-dimensional tensor, but got 2-dimensional tensor for argument #1 'log_probs' (while checking arguments for ctc_loss_gpu)""",
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
        gpu="""Expected 1 to 2 dimensions, but got 3-dimensional tensor for argument #2 'targets' (while checking arguments for ctc_loss_gpu)""",
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
        gpu="""input_lengths must be of size batch_size""",
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
        gpu="""target_lengths must be of size batch_size""",
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

    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
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
        gpu="""linalg.solve_triangular: The input tensor A must have at least 2 dimensions.""",
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
        gpu="""linalg.solve_triangular: Incompatible shapes of A and B for the equation XA = B (2x2 and 2x3)""",
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
        gpu="""linalg.solve_triangular: Incompatible shapes of A and B for the equation AX = B (2x2 and 3x2)""",
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
        gpu=""""triangular_solve_cuda" not implemented for 'BFloat16'""",
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
        gpu="""masked_select: expected BoolTensor for mask""",
        tpu="""masked_select(): expected mask to be a BoolTensor, got float32""",
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
    with et.assert_raises_message(
        IndexError,
        gpu="""index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (1), destination dimensionality (2)""",
        tpu="""index_copy(): expected self and source to have the same number of dimensions, got 2 and 1""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(2, device=et.device())
      )

  def test_index_copy_index_rank_not_1(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""index_copy_(): Index should have dimension 1 or 0 (got 2)""",
        tpu="""index_copy(): expected index to be 1D, got shape [1, 1]""",
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
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        tpu="""index_copy(): expected dimension to be in range [-2, 1], got 2""",
        message_reviewed_by="wan",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 2, device=et.device())
      torch.index_copy(
          t, 2, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_source_dim_ne_index_size(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""index_copy_(): Number of indices (1) should be equal to source.size(dim) (2)""",
        tpu="""index_copy(): expected source to have the same number of elements as the index along dimension 0, got 2 and 1""",
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
        gpu="""index_copy_(): Source/destination tensor must have same slice shapes. Destination slice shape: 2 at dimension 0 and source slice shape: 3 at dimension 0.""",
        tpu="""index_copy(): expected self and source to have the same size along dimension 1, got 2 and 3""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 3, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_scalar_dim_out_of_range(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""index_copy(): expected dim to be 0 for scalar input, got 1""",
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
        gpu="""index_copy_(): Source/destination tensor must have same slice shapes. Destination slice shape:  at dimension 0 and source slice shape: 1 at dimension 0.""",
        tpu="""index_copy(): expected source shape to match self shape, excluding the specified dimension, got source shape [1, 1] and self shape []""",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor([[1]], device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_copy_scalar_index_size_ne_1(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""index_copy_(): When source is scalar, index should have one element (got 2)""",
        tpu="""index_copy(): expected index to be 1D of size 1 for scalar input, got shape [2]""",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0, 0], device=et.device(), dtype=torch.long)
      source = torch.tensor(1, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_fill_index_invalid_rank(self):
    tpu_err = """index_fill_(): expected index to be at most 1-D, got 2-D"""
    gpu_err = """Index has to be a vector/scalar"""
    t = torch.ones(2, 2, device=et.device())
    index = torch.tensor([[0]], device=et.device(), dtype=torch.long)

    # Scalar variant
    with et.assert_raises_message(RuntimeError, tpu=tpu_err, gpu=gpu_err):
      t.index_fill_(0, index, 5.0)

    # Tensor variant
    with et.assert_raises_message(RuntimeError, tpu=tpu_err, gpu=gpu_err):
      t.index_fill_(0, index, torch.tensor(5.0, device=et.device()))

  def test_index_fill_index_type_not_long(self):
    tpu_err = """index_fill_(): expected index dtype to be Long, got int32"""
    gpu_err = """index_fill_(): Expected dtype int64 for index."""
    t = torch.ones(2, 2, device=et.device())
    index = torch.tensor([0], device=et.device(), dtype=torch.int)
    err_type = RuntimeError if et.is_on_tpu() else IndexError

    # Scalar variant
    with et.assert_raises_message(err_type, tpu=tpu_err, gpu=gpu_err):
      t.index_fill_(0, index, 5.0)

    # Tensor variant
    with et.assert_raises_message(err_type, tpu=tpu_err, gpu=gpu_err):
      t.index_fill_(0, index, torch.tensor(5.0, device=et.device()))

  def test_index_fill_dim_out_of_range(self):
    t = torch.ones(2, 2, device=et.device())
    index = torch.tensor([0], device=et.device(), dtype=torch.long)

    # Scalar variant
    with et.assert_raises_message(
        IndexError,
        tpu="""index_fill_(): expected dimension to be in range [-2, 1], got 2""",
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        message_reviewed_by="wan",
    ):
      t.index_fill_(2, index, 5.0)

    # Tensor variant
    with et.assert_raises_message(
        IndexError,
        tpu="""index_fill_(): expected dimension to be in range [-2, 1], got 2""",
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        message_reviewed_by="wan",
    ):
      t.index_fill_(2, index, torch.tensor(5.0, device=et.device()))

  def test_index_fill_value_tensor_not_0d(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""index_fill_(): expected value to be a 0-D tensor, got 1-D tensor""",
        gpu="""index_fill_ only supports a 0-dimensional value tensor, but got tensor with 1 dimension(s).""",
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
        gpu="""fill_ only supports 0-dimension value tensor but got tensor with"""
        f""" {len(shape)} dimensions.""",
        tpu=f"fill_(): expected value to be a 0-D tensor, got {len(shape)}-D",
    ):
      torch.fill(t, value)

  def test_fmod_tensor_with_unsupported_dtype(self):
    t = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.complex64)
    other = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        gpu=""""fmod_cuda" not implemented for 'ComplexFloat'""",
        tpu="""fmod(): complex dtypes are not supported""",
    ):
      torch.fmod(t, other)

    t = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.bool)
    other = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        gpu=""""fmod_cuda" not implemented for 'Bool'""",
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
        gpu="""masked_select(): self and result must have the same scalar type""",
        tpu="""masked_select(): expected out tensor to have dtype float32, got int32""",
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
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_msg = "mm: self must be a matrix, got 1-D tensor"
    else:
      tpu_error_msg = "self must be a matrix"
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error_msg,
    ):
      torch.mm(t1, t2)

  def test_mm_with_non_2d_arg2(self):
    """Tests that mm with non-2D argument 2 fails with expected error."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 3, 4, device=et.device(), dtype=torch.float32)
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_msg = "mm: mat2 must be a matrix, got 3-D tensor"
    else:
      tpu_error_msg = "mat2 must be a matrix"
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error_msg,
    ):
      torch.mm(t1, t2)

  def test_mm_with_mismatched_sizes(self):
    """Tests that mm with mismatched sizes fails with expected error."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(4, 2, device=et.device(), dtype=torch.float32)
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_msg = (
          "mm: mat1 and mat2 shapes cannot be multiplied (2x3 and 4x2)"
      )
    else:
      tpu_error_msg = "mat1 and mat2 shapes cannot be multiplied (2x3 and 4x2)"
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error_msg,
    ):
      torch.mm(t1, t2)

  @parameterized.named_parameters(
      dict(
          testcase_name="bool",
          dtype=torch.bool,
          tpu_dtype="bool",
          gpu_dtype="Bool",
      ),
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
  )
  def test_mm_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    a = torch.ones(2, 3, dtype=dtype, device=et.device())
    b = torch.ones(3, 2, dtype=dtype, device=et.device())

    # `matmul` decomposes to `mm`, so both report the`mm` kernel.
    for op in (torch.mm, torch.matmul):
      with et.assert_raises_message(
          NotImplementedError,
          tpu=f"""mm(): not implemented for {tpu_dtype}""",
          gpu=f""""addmm_cuda" not implemented for '{gpu_dtype}'""",
          message_reviewed_by="wan",
      ):
        op(a, b)

  def test_mm_with_mismatched_data_types(self):
    """Tests that mm with mismatched data types fails with expected error."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 2, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""expected mat1 and mat2 to have the same dtype, but got: float != int""",
        tpu="""mm(): expected the two arguments to have the same dtype, got float32 vs int32""",
        message_reviewed_by="wan",
    ):
      torch.mm(t1, t2)

  def test_native_batch_norm_complex(self):
    """Tests native_batch_norm and native_batch_norm_backward with complex dtype."""
    input_dtype = torch.complex64
    stats_dtype = torch.float32
    n, c, h, w = 2, 4, 4, 4
    input_val = torch.randn(n, c, h, w, dtype=input_dtype, device=et.device())
    weight = torch.randn(c, dtype=stats_dtype, device=et.device())
    bias = torch.randn(c, dtype=stats_dtype, device=et.device())
    running_mean = torch.randn(c, dtype=stats_dtype, device=et.device())
    running_var = torch.rand(c, dtype=stats_dtype, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_batch_norm(): expected input to be floating point, got complex64""",
        gpu=re.compile(
            r"(native_batch_norm: expected input to be floating point, got"
            r" complex64|\".*batch_norm.*\" not implemented for"
            r" 'ComplexFloat')"
        ),
    ):
      torch.ops.aten.native_batch_norm(
          input_val,
          weight,
          bias,
          running_mean,
          running_var,
          training=True,
          momentum=0.1,
          eps=1e-5,
      )

    grad_out = torch.randn(n, c, h, w, dtype=input_dtype, device=et.device())
    save_mean = torch.randn(c, dtype=stats_dtype, device=et.device())
    save_invstd = torch.rand(c, dtype=stats_dtype, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_batch_norm_backward(): expected input to be floating point, got complex64""",
        gpu=re.compile(
            r"(native_batch_norm_backward: expected input to be floating point,"
            r" got complex64|\".*batch_norm.*\" not implemented for"
            r" 'ComplexFloat')"
        ),
    ):
      torch.ops.aten.native_batch_norm_backward(
          grad_out,
          input_val,
          weight,
          running_mean,
          running_var,
          save_mean,
          save_invstd,
          train=True,
          eps=1e-5,
          output_mask=[True, True, True],
      )

  def test_binary_cross_entropy_invalid_input_dtype(self):
    """Tests binary_cross_entropy with non-floating point input."""
    input_val = torch.randint(
        0, 2, (3, 3), dtype=torch.int32, device=et.device()
    )
    target_val = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""binary_cross_entropy(): expected floating point input, got int32""",
        gpu="""Found dtype Float but expected Int""",
    ):
      torch.ops.aten.binary_cross_entropy(input_val, target_val)

  def test_binary_cross_entropy_invalid_target_dtype(self):
    """Tests binary_cross_entropy with non-floating point target."""
    input_val = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    target_val = torch.randint(
        0, 2, (3, 3), dtype=torch.int32, device=et.device()
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu="""binary_cross_entropy(): expected floating point target, got int32""",
        gpu="""Found dtype Int but expected Float""",
    ):
      torch.ops.aten.binary_cross_entropy(input_val, target_val)

  def test_binary_cross_entropy_mismatched_shapes(self):
    """Tests binary_cross_entropy with mismatched shapes."""
    input_val = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    target_val = torch.rand(3, 4, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""binary_cross_entropy(): expected input and target shapes to match, got [3, 3] vs [3, 4]""",
        gpu="""The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1""",
    ):
      torch.ops.aten.binary_cross_entropy(input_val, target_val)

  def test_binary_cross_entropy_backward_grad_input_invalid_dtype(self):
    """Tests binary_cross_entropy_backward.grad_input with invalid grad_input dtype."""
    grad_output = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    input_val = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    target_val = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    grad_input = torch.empty(3, 3, dtype=torch.int32, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""binary_cross_entropy_backward(): expected grad_input dtype float32, got int32""",
        gpu="""Found dtype Int but expected Float""",
    ):
      torch.ops.aten.binary_cross_entropy_backward.grad_input(
          grad_output, input_val, target_val, None, 0, grad_input=grad_input
      )

  @parameterized.named_parameters(
      # (testcase_name, invalid_arg)
      ("grad_dtype", "grad_output"),
      ("input_dtype", "input"),
      ("target_dtype", "target"),
  )
  def test_binary_cross_entropy_backward_invalid_dtypes(self, invalid_arg: str):
    """Tests binary_cross_entropy_backward with non-floating point dtypes."""

    def make_tensor(is_valid):
      if is_valid:
        return torch.rand(3, 3, dtype=torch.float32, device=et.device())
      else:
        return torch.randint(
            0, 2, (3, 3), dtype=torch.int32, device=et.device()
        )

    grad_output = make_tensor(invalid_arg != "grad_output")
    input_val = make_tensor(invalid_arg != "input")
    target_val = make_tensor(invalid_arg != "target")

    tpu_error = (
        "binary_cross_entropy_backward(): expected floating point"
        f" {invalid_arg}, got int32"
    )
    gpu_error = (
        "Found dtype Float but expected Int"
        if invalid_arg == "grad_output"
        else "Found dtype Int but expected Float"
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error,
        gpu=gpu_error,
    ):
      torch.ops.aten.binary_cross_entropy_backward(
          grad_output, input_val, target_val
      )

  def test_binary_cross_entropy_backward_mismatched_shapes(self):
    """Tests binary_cross_entropy_backward with mismatched shapes."""
    grad_output = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    input_val = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    target_val = torch.rand(3, 4, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""binary_cross_entropy_backward(): expected input and target shapes to match, got [3, 3] vs [3, 4]""",
        gpu="""The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1""",
    ):
      torch.ops.aten.binary_cross_entropy_backward(
          grad_output, input_val, target_val
      )

  def test_binary_cross_entropy_backward_grad_output_non_broadcastable(self):
    """Tests binary_cross_entropy_backward with non-broadcastable grad_output."""
    grad_output = torch.rand(3, 4, dtype=torch.float32, device=et.device())
    input_val = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    target_val = torch.rand(3, 3, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""binary_cross_entropy_backward(): expected grad_output to be broadcastable to input shape, got [3, 4] vs [3, 3]""",
        gpu="""The expanded size of the tensor (3) must match the existing size (4) at non-singleton dimension 1.  Target sizes: [3, 3].  Tensor sizes: [3, 4]""",
    ):
      torch.ops.aten.binary_cross_entropy_backward(
          grad_output, input_val, target_val
      )

  def test_nll_loss_unsupported_input_dtype(self):
    t = torch.ones(3, 5, device=et.device(), dtype=torch.int32)
    target = torch.tensor([1, 0, 4], device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        gpu=""""nll_loss_forward_reduce_cuda_kernel_2d" not implemented for 'Int'""",
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
      # TODO: b/533385796 - Change this back to a string later.
      tpu_error_message = re.compile(
          r"^(expected target dtype to be Long or Byte, but got Int|"
          r"nll_loss_forward\(\): expected the target dtype to be either int64"
          r" or uint8, got int32)$"
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
        gpu="""input and target batch or spatial sizes don't match: target [1, 2, 3], input [1, 3, 2, 2]""",
        tpu="""nll_loss2d_forward(): expected the shapes of the input [N, C, d1, ..., dk] and the target [N, d1, ..., dk] (k >= 1) to match, got input: [1, 3, 2, 2], target: [1, 2, 3]""",
    ):
      torch.nn.functional.nll_loss(t, target)

  def test_nll_loss2d_backward_invalid_input_dim(self):
    grad_output = torch.ones(1, 2, 2, device=et.device(), dtype=torch.float32)
    t = torch.ones(1, 3, 2, device=et.device(), dtype=torch.float32)
    target = torch.ones(1, 2, 2, device=et.device(), dtype=torch.long)
    total_weight = torch.ones((), device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""only batches of spatial inputs supported (4D tensors), but got input of size: [1, 3, 2]""",
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
        gpu="""only batches of spatial targets supported (3D tensors) but got targets of size: : [1, 2]""",
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
        gpu="""Trying to create tensor with negative dimension -1: [-1]""",
        tpu="""empty(): expected dimension sizes to be >= 0, got [-1], which contains -1""",
    ):
      torch.ones(-1, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        gpu="""Trying to create tensor with negative dimension -2: [3, -2, -4]""",
        tpu="""empty(): expected dimension sizes to be >= 0, got [3, -2, -4], which contains -2 and -4""",
    ):
      torch.ones(3, -2, -4, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        gpu="""Trying to create tensor with negative dimension -2: [3, -2, -4, 1, -5]""",
        tpu="""empty(): expected dimension sizes to be >= 0, got [3, -2, -4, 1, -5], which contains -2, -4, and -5""",
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
        gpu="""Storage size calculation overflowed with sizes=[2147483648, 4294967296]""",
        tpu="""empty(): product of dimension sizes [2147483648, 4294967296] overflows as int64""",
    ):
      # The product of the dimensions is 2 ** 63, which doesn't cause an
      # overflow in XLA. However, it doesn't fit in int64_t.
      torch.ones(2**31, 2**32, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        gpu="""Storage size calculation overflowed with sizes=[1073741824, 1073741824, 1073741824]""",
        tpu="""empty(): product of dimension sizes [1073741824, 1073741824, 1073741824] overflows as int64""",
    ):
      # The product of the dimensions is 2 ** 90, which causes an overflow in
      # XLA.
      torch.ones(2**30, 2**30, 2**30, device=et.device(), dtype=torch.float32)

  def test_byte_size_overflow_in_ones(self):
    """Tests that torch.ones() fails with expected error when the byte size overlows."""
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Storage size calculation overflowed with sizes=[2147483648, 2147483648]""",
        tpu="""empty(): product of dimension sizes [2147483648, 2147483648] and size of float32 (4 bytes) overflows as int64""",
    ):
      # The product of the dimensions is 2 ** 62, which doesn't cause an
      # overflow in XLA. However, the byte size is 2 ** 64, which overflows
      # int64_t.
      torch.ones(2**31, 2**31, device=et.device(), dtype=torch.float32)

  def test_histc_bounds_overflow(self):
    """Tests that torch.histc() fails when the bounds overflow."""
    if et.is_on_gpu():
      # On TPU, when min == max, bounds adjustment (+/- 1) causes int32
      # overflow, synchronously raising a RuntimeError. On GPU,
      # converting float to int bounds does not overflow or raise an
      # error.
      self.skipTest("GPU behavior difference")
    max_int32 = torch.iinfo(torch.int32).max
    t = torch.tensor(
        [max_int32, max_int32], device=et.device(), dtype=torch.int32
    )
    with et.assert_raises_message(
        RuntimeError,
        gpu="""value cannot be converted to type int without overflow""",
        tpu="""histc(): expected min and max to be within the range of their data types, got min = 2147483646 and max = -2147483648. This happened because min and max were adjusted by one (due to min == max), which resulted in an overflow""",
    ):
      torch.histc(t)

  def test_histc_bounds_underflow(self):
    """Tests that torch.histc() fails when the bounds underflow."""
    if et.is_on_gpu():
      # On TPU, when min == max, bounds adjustment (+/- 1) causes int32
      # underflow, synchronously raising a RuntimeError. On GPU,
      # converting float to int bounds does not underflow or raise an
      # error.
      self.skipTest("GPU behavior difference")
    min_int32 = torch.iinfo(torch.int32).min
    t = torch.tensor(
        [min_int32, min_int32], device=et.device(), dtype=torch.int32
    )
    with et.assert_raises_message(
        RuntimeError,
        gpu="""value cannot be converted to type int without overflow""",
        tpu="""histc(): expected min and max to be within the range of their data types, got min = 2147483647 and max = -2147483647. This happened because min and max were adjusted by one (due to min == max), which resulted in an overflow""",
    ):
      torch.histc(t)

  def test_histc_bounds_not_nan(self):
    """Tests that torch.histc() fails when the bounds are NaN."""
    t = torch.tensor([0, float("nan")], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""range of [nan, nan] is not finite""",
        tpu="""histc(): expected min and max to be finite, got nan and nan. Either make sure that the input data is finite, or provide valid finite bounds""",
    ):
      torch.histc(t)

  def test_histc_bounds_inf(self):
    """Tests that torch.histc() fails when the bounds are infinity."""
    t = torch.tensor([0, float("inf")], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""range of [0, inf] is not finite""",
        tpu="""histc(): expected min and max to be finite, got 0 and inf. Either make sure that the input data is finite, or provide valid finite bounds""",
    ):
      torch.histc(t)

  def test_histc_bounds_not_in_order(self):
    """Tests that torch.histc() fails when the bounds are not in order."""
    t = torch.tensor([0, 0], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""max must be larger than min""",
        tpu="""histc(): expected min <= max, got 1 vs 0""",
    ):
      torch.histc(t, min=1, max=0)

  def test_invalid_index_dtype_in_take(self):
    """Tests that torch.take() fails when the index has the wrong dtype."""
    t = torch.tensor([0, 1, 2], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""take(): Expected a long tensor for index, but got Int""",
        tpu="""take(): expected index dtype to be int64, got int32""",
    ):
      torch.take(t, torch.tensor([0, 1], dtype=torch.int32, device=et.device()))

  def test_empty_tensor_in_take(self):
    """Tests that torch.take() fails when the input tensor is empty but index is not."""
    t = torch.tensor([], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        gpu="""take(): tried to take from an empty tensor""",
        tpu="""take(): input tensor must be non-empty when the index tensor is non-empty""",
    ):
      torch.take(t, torch.tensor([0], dtype=torch.int64, device=et.device()))

  def test_invalid_index_in_take(self):
    """Tests that torch.take() fails when the index is invalid."""
    if et.is_on_gpu():
      # On TPU, out-of-bounds indices synchronously raise an IndexError
      # with a clean range message. On GPU, out-of-bounds indexing in
      # CUDA kernels triggers an asynchronous device-side assert,
      # raising RuntimeError instead of IndexError.
      self.skipTest("GPU behavior difference")
    t = torch.tensor([0, 1, 2], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        gpu="""CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
Device-side assertion tracking was not enabled by user.""",
        tpu="""take(): expected indices to be in range [-3, 2], got 3""",
    ):
      torch.take(t, torch.tensor([0, 3], dtype=torch.int64, device=et.device()))

    with et.assert_raises_message(
        IndexError,
        gpu="""out of range: tried to access index -4 on a tensor of 3 elements.""",
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
        gpu="""Trying to create tensor with negative dimension -1: [-1]""",
        tpu="""empty(): expected dimension sizes to be >= 0, got [-1], which contains -1""",
    ):
      torch.empty(-1, device=et.device(), dtype=torch.float32)

  def test_invalid_broadcast_in_binary_op_add(self):
    """Tests that torch.add() fails with expected error when the sizes are mismatched."""

    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1""",
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
        gpu=""""round_cuda" not implemented for 'Long'""",
        message_reviewed_by="wan",
    ):
      t.round(decimals=-1)

    t = torch.ones(1, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""round(): expected the input dtype not to be integer when the decimals argument is specified (2), got int32""",
        gpu=""""round_cuda" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      t.round_(decimals=2)

    t = torch.ones(1, device=et.device(), dtype=torch.int16)
    out_t = torch.zeros(1, device=et.device(), dtype=torch.int16)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""round(): expected the input dtype not to be integer when the decimals argument is specified (0), got int16""",
        gpu=""""round_cuda" not implemented for 'Short'""",
        message_reviewed_by="wan",
    ):
      torch.round(t, decimals=0, out=out_t)

  def test_round_invalid_input_dtype(self):
    t = torch.tensor([True, False], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        gpu=""""round_cuda" not implemented for 'Bool'""",
        tpu="""round(): dtype bool is not supported""",
        message_reviewed_by="wan",
    ):
      torch.round(t)

  def test_roll_errors(self):
    """roll() fails when input parameters are invalid."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""shifts and dimensions must align. shifts: 2, dims:1""",
        tpu="""roll(): expected shifts to have the same size as dims (1), got 2""",
    ):
      torch.roll(t, shifts=(2, 3), dims=(0,))

    with et.assert_raises_message(
        RuntimeError,
        gpu="""shifts and dimensions must align. shifts: 2, dims:0""",
        tpu="""roll(): expected shifts to have the same size as dims (0), got 2""",
    ):
      torch.roll(t, shifts=(2, 3))

  def test_reduction_dim_out_of_bounds_var(self):
    self.do_test_reduction_dim_out_of_bounds(
        torch.var,
        tpu_msg_template=(
            "var(): expected dimension to be in range of [-2, 1], got {dim}"
        ),
        message_reviewed_by="gunhyun",
    )

  def test_reduction_dim_out_of_bounds_std(self):
    self.do_test_reduction_dim_out_of_bounds(
        torch.std,
        tpu_msg_template=(
            "std(): expected dimension to be in range of [-2, 1], got {dim}"
        ),
        message_reviewed_by="gunhyun",
    )

  def test_reduction_dim_out_of_bounds_mean(self):
    self.do_test_reduction_dim_out_of_bounds(torch.mean)

  def test_reduction_dim_out_of_bounds_sum(self):
    self.do_test_reduction_dim_out_of_bounds(torch.sum)

  def do_test_reduction_dim_out_of_bounds(
      self,
      reduction_fn,
      tpu_msg_template=None,
      message_reviewed_by=None,
  ):
    """Reduction function fails when dimension is out of bounds."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)

    for dim in [-3, 3]:
      gpu_msg = (
          "Dimension out of range (expected to be in range of [-2, 1], but got"
          f" {dim})"
      )
      tpu_msg = (
          tpu_msg_template.format(dim=dim) if tpu_msg_template else gpu_msg
      )
      with et.assert_raises_message(
          IndexError,
          tpu=tpu_msg,
          gpu=gpu_msg,
          message_reviewed_by=message_reviewed_by,
      ):
        reduction_fn(t, dim=dim)

  def test_reduction_dim_repeated_var(self):
    self.do_test_reduction_dim_repeated(
        torch.var,
        tpu_msg="var(): dim 1 appears multiple times in the list of dims",
        message_reviewed_by="gunhyun",
    )

  def test_reduction_dim_repeated_std(self):
    self.do_test_reduction_dim_repeated(
        torch.std,
        tpu_msg="std(): dim 1 appears multiple times in the list of dims",
        message_reviewed_by="gunhyun",
    )

  def test_reduction_dim_repeated_mean(self):
    self.do_test_reduction_dim_repeated(torch.mean)

  def test_reduction_dim_repeated_sum(self):
    self.do_test_reduction_dim_repeated(torch.sum)

  def do_test_reduction_dim_repeated(
      self, reduction_fn, tpu_msg=None, message_reviewed_by=None
  ):
    """Reduction function fails when canonical dimension is repeated."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    gpu_msg = "dim 1 appears multiple times in the list of dims"
    tpu_msg = tpu_msg or gpu_msg
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_msg,
        gpu=gpu_msg,
        message_reviewed_by=message_reviewed_by,
    ):
      reduction_fn(t, dim=[-1, 1])

  def test_reduction_dim_scalar_var(self):
    self.do_test_reduction_dim_scalar(
        torch.var,
        tpu_msg="var(): expected dimension to be in range of [-1, 0], got 1",
        message_reviewed_by="gunhyun",
    )

  def test_reduction_dim_scalar_std(self):
    self.do_test_reduction_dim_scalar(
        torch.std,
        tpu_msg="std(): expected dimension to be in range of [-1, 0], got 1",
        message_reviewed_by="gunhyun",
    )

  def test_reduction_dim_scalar_mean(self):
    self.do_test_reduction_dim_scalar(torch.mean)

  def test_reduction_dim_scalar_sum(self):
    self.do_test_reduction_dim_scalar(torch.sum)

  def do_test_reduction_dim_scalar(
      self, reduction_fn, tpu_msg=None, message_reviewed_by=None
  ):
    """Reduction function fails when dimension is invalid for scalar."""
    t_rank0 = torch.tensor(1.0, device=et.device(), dtype=torch.float32)
    gpu_msg = (
        "Dimension out of range (expected to be in range of [-1, 0], but got 1)"
    )
    tpu_msg = tpu_msg or gpu_msg
    with et.assert_raises_message(
        IndexError,
        tpu=tpu_msg,
        gpu=gpu_msg,
        message_reviewed_by=message_reviewed_by,
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
        gpu="""std and var only support floating point and complex dtypes""",
        tpu="""var(): expected a floating point or complex dtype, got int32""",
    ):
      torch.var(t, dim=0)

  def test_reduction_unsupported_int_dtype_std(self):
    """Std fails for integral dtypes."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""std(): expected a floating point or complex dtype, got int32""",
        gpu="""std and var only support floating point and complex dtypes""",
    ):
      torch.std(t, dim=0)

  def test_reduction_unsupported_int_dtype_std_out(self):
    """Std out-variant fails when input has integral dtype."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    out = torch.empty(3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""std(): expected a floating point or complex dtype, got int32""",
        gpu="""std and var only support floating point and complex dtypes""",
    ):
      torch.std(t, dim=0, out=out)

  def test_reduction_unsupported_int_out_dtype_std(self):
    """Std out-variant fails when output has non-floating dtype."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    out = torch.empty(3, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""std(): expected floating point output dtype, got int32""",
        gpu="""result type Float can't be cast to the desired output type Int""",
    ):
      torch.std(t, dim=0, out=out)

  def test_unfold_size_too_large(self):
    """Unfold fails when size is larger than dimension."""
    t = torch.ones(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        gpu="""maximum size for tensor at dimension 0 is 5 but size is 6""",
        tpu="""unfold(): expected size <= dimension size (shape[0]: 5), got size: 6""",
    ):
      t.unfold(0, 6, 1)

  def test_unfold_dim_out_of_bounds(self):
    """Unfold fails when dimension is out of bounds."""
    t = torch.ones(2, 3, device=et.device())
    with et.assert_raises_message(
        IndexError,
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        tpu="""unfold(): expected dimension to be in range of [-2, 1] for shape [2, 3], got 2""",
    ):
      t.unfold(2, 1, 1)

  def test_unfold_zero_step(self):
    """Unfold fails when step is 0."""
    t = torch.ones(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        gpu="""step is 0 but must be > 0""",
        tpu="""unfold(): expected step > 0, got 0""",
    ):
      t.unfold(0, 2, 0)

  def test_std_negative_reduction_factor(self):
    """A warning is issued and nan returned when degrees of freedom <= 0."""
    # TODO: b/435570003 - Create a utility to compare warning messages.
    tpu_warn_msg = (
        "std(): degrees of freedom (i.e., reduction size - correction) should"
        " be positive, got reduction size = 1, correction = 1, and degrees of"
        " freedom = 0"
    )
    gpu_warn_msg = (
        "std(): degrees of freedom is <= 0. Correction should be strictly less"
        " than the reduction factor (input numel divided by output numel)."
    )

    t = torch.ones(1, device=et.device(), dtype=torch.float32)
    warn_msg = tpu_warn_msg if et.is_on_tpu() else gpu_warn_msg

    with self.assertWarnsRegex(UserWarning, re.escape(warn_msg)):
      result = torch.std(t, correction=1)
    self.assertTrue(torch.all(torch.isnan(result)), f"Got {result.to('cpu')}")

    t = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    tpu_warn_msg = (
        "std(): degrees of freedom (i.e., reduction size - correction) should"
        " be positive, got reduction size = 3, correction = 4, and degrees of"
        " freedom = -1"
    )
    warn_msg = tpu_warn_msg if et.is_on_tpu() else gpu_warn_msg

    with self.assertWarnsRegex(UserWarning, re.escape(warn_msg)):
      result = torch.std(t, dim=0, correction=4)
    self.assertTrue(torch.all(torch.isnan(result)), f"Got {result.to('cpu')}")

  def test_var_negative_reduction_factor(self):
    """A warning is issued and nan returned when the variance degrees of freedom is <= 0."""
    # TODO: b/435570003 - Create a utility to compare warning messages.
    gpu_warn_msg = (
        "var(): degrees of freedom is <= 0. Correction should be strictly less"
        " than the reduction factor (input numel divided by output numel)."
    )

    t = torch.ones(1, device=et.device(), dtype=torch.float32)
    tpu_warn_msg = (
        "var(): degrees of freedom (i.e., reduction size - correction) should"
        " be positive, got reduction size = 1, correction = 1, and degrees of"
        " freedom = 0"
    )

    warn_msg = tpu_warn_msg if et.is_on_tpu() else gpu_warn_msg

    with self.assertWarnsRegex(UserWarning, re.escape(warn_msg)):
      result = torch.var(t, correction=1)
    self.assertTrue(torch.all(torch.isnan(result)), f"Got {result.to('cpu')}")

    t = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    tpu_warn_msg = (
        "var(): degrees of freedom (i.e., reduction size - correction) should"
        " be positive, got reduction size = 3, correction = 4, and degrees of"
        " freedom = -1"
    )
    warn_msg = tpu_warn_msg if et.is_on_tpu() else gpu_warn_msg

    with self.assertWarnsRegex(UserWarning, re.escape(warn_msg)):
      result = torch.var(t, dim=0, correction=4)
    self.assertTrue(torch.all(torch.isnan(result)), f"Got {result.to('cpu')}")

  def test_var_mean_invalid_dtype(self):
    t = torch.randint(0, 10, (5, 5), dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""var_mean(): expected a floating point or complex dtype, got int32""",
        gpu="""var_mean only support floating point and complex dtypes""",
    ):
      torch.var_mean(t)

  def test_var_mean_duplicate_dims(self):
    t = torch.randn(5, 5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""var_mean(): dim 0 appears multiple times in the list of dims""",
        gpu="""dim 0 appears multiple times in the list of dims""",
    ):
      torch.var_mean(t, dim=(0, 0))

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
    err_type = TypeError if et.is_on_gpu() else RuntimeError
    with et.assert_raises_message(
        err_type,
        tpu="""view_as_real(): expected the input dtype to be complex, got float32""",
        gpu="""view_as_real is only supported for complex tensors""",
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
        t.select(dim, index)

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
        t.select(dim, 1)

  def test_slice_on_scalar(self):
    t = torch.scalar_tensor(1.0, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        IndexError,
        # This error is generated by pytorch before redispatching to as_strided;
        # we don't have control over this error message.
        tpu="""slice() cannot be applied to a 0-dim tensor.""",
    ):
      _ = t[0:1:1]

  def test_slice_zero_step(self):
    t = torch.ones(10, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        ValueError,
        # This error is generated by pytorch before redispatching to as_strided;
        # we don't have control over this error message.
        tpu="""slice step cannot be zero""",
    ):
      _ = t[0:10:0]

  def test_slice_negative_step(self):
    t = torch.ones(10, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        ValueError,
        # This error is generated by pytorch before redispatching to as_strided;
        # we don't have control over this error message.
        tpu="""step must be greater than zero""",
    ):
      _ = t[0:10:-1]

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

  def test_channel_shuffle_invalid_dim(self):
    t = torch.randn(4, 4, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        gpu=re.compile(
            r".*channel_shuffle expects input with > 2 dims.*got input with"
            r" sizes \[4, 4\].*"
        ),
        tpu=re.compile(
            r".*expect(ed|s) input with > 2 dims.*got input with sizes \[4,"
            r" 4\].*"
        ),
    ):
      torch.channel_shuffle(t, 2)

  def test_channel_shuffle_invalid_groups(self):
    t = torch.randn(2, 4, 8, 8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Number of groups to divide channels in must be positive. Value of groups:0""",
        tpu="""channel_shuffle(): expected number of groups to divide channels in to be positive, got 0""",
    ):
      torch.channel_shuffle(t, 0)

  def test_channel_shuffle_channels_not_divisible(self):
    t = torch.randn(2, 5, 8, 8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        gpu=re.compile(
            r".*Number of channels must be divisible by groups.*Got 5 channels"
            r" and 2 groups.*"
        ),
        tpu=re.compile(
            r".*expected number of channels to be divisible by"
            r" groups.*(but )?got 5 channels and 2 groups.*"
        ),
    ):
      torch.channel_shuffle(t, 2)

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
        gpu=""""addcmul_cuda" not implemented for 'Bool'""",
        tpu="""addcmul(): expected no input tensor dtype to be bool, got input: bool, tensor1: bool, tensor2: bool""",
    ):
      torch.addcmul(self_tensor, tensor1, tensor2, value=value)

  def test_index_put_too_many_indices_error(self):
    # TODO(mkkhanna): Fix exception type for TPU.
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""too many indices for tensor of dimension 1 (got 2)""",
        tpu="""index_put_(): expected at most 1 indices for tensor of dimension 1, got 2""",
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
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""tensors used as indices must be long, int, byte or bool tensors""",
        tpu="""index_put_(): expected index tensors to have dtype bool, byte, int32, or int64, got float32 at index 0""",
    ):
      torch.index_put_(
          torch.tensor([0, 1], device=et.device()),
          (torch.tensor([0], dtype=torch.float32, device=et.device()),),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_broadcast_indices_error(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""shape mismatch: indexing tensors could not be broadcast together with shapes [2], [3]""",
        tpu="""index_put_(): expected index tensors to be broadcastable, got index tensor shape [3] and broadcast shape [2]: the size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0""",
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
        gpu="""shape mismatch: value tensor of shape [2, 2] cannot be broadcast to indexing result of shape [2]""",
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
        gpu="""Index put requires the source and destination dtypes match, got Int for the destination and Long for the source.""",
        tpu="""index_put_(): expected dtypes of values and destination to be the same, got int64 and int32""",
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
    if et.is_on_gpu():
      # On TPU, passing empty indices () raises a clean Python RuntimeError
      # ("indices must be specified"). On GPU, this triggers an internal C++
      # assertion in OffsetCalculator.cuh instead of a standard Python error.
      self.skipTest("GPU behavior difference")
    with et.assert_raises_message(
        RuntimeError,
        gpu=re.compile(
            r"""N <= iter\.ntensors\(\) INTERNAL ASSERT FAILED at.*OffsetCalculator\.cuh.*please report a bug to PyTorch\.\s*"""
        ),
        tpu="""index_put_(): indices must be specified""",
    ):
      torch.index_put_(
          torch.tensor([[0, 1], [2, 3]], device=et.device()),
          (),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_decompose_with_mask_mask_shape_mismatch(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""The shape of the mask [3, 5] at index 0 does not match the shape of the indexed tensor [2, 5] at index 0""",
        tpu="""index_put_(): expected the shape of the mask at index 0 to match the shape of the indexed tensor at index 0, got mask shape [3, 5] and indexed tensor shape [2, 5]""",
    ):
      tensor = torch.arange(10).view(2, 5).to(et.device())
      tensor_other = torch.arange(15).view(3, 5).to(et.device())
      boolean_mask = tensor_other % 2 != 0
      torch.index_put_(
          tensor,
          (boolean_mask,),
          torch.tensor(100, device=et.device()),
      )

  def test_index_put_decompose_with_mask_error_mask_dim_more_than_indexed_tensor_dim(
      self,
  ):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""index_put_(): expected the shape of the mask at index 1 to match the shape of the indexed tensor at index 1, got mask shape [2, 2] and indexed tensor shape [2]""",
    ):
      torch.index_put_(
          torch.tensor([0, 1], device=et.device()),
          (torch.tensor([[True, False], [False, True]], device=et.device()),),
          torch.tensor(0, device=et.device()),
      )

  def test_index_put_decompose_with_multiple_mask_error(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""The shape of the mask [4] at index 0 does not match the shape of the indexed tensor [3, 5] at index 1""",
        tpu="""index_put_(): expected the shape of the mask at index 0 to match the shape of the indexed tensor at index 1, got mask shape [4] and indexed tensor shape [3, 5]""",
    ):
      tensor = torch.arange(15).view(3, 5).to(et.device())
      mask0 = (torch.arange(3) % 2 != 0).to(et.device())
      mask1 = (torch.arange(4) % 2 != 0).to(et.device())
      torch.index_put_(
          tensor,
          (mask0, mask1),
          torch.tensor(100, device=et.device()),
      )

  def test_index_select_index_must_be_1d(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Index is supposed to be an empty tensor or a vector""",
        tpu="""index_select(): expected index to be 1D, got shape [2, 3]""",
    ):
      torch.index_select(
          torch.ones(2, 3, device=et.device()),
          1,
          torch.ones(2, 3, device=et.device(), dtype=torch.long),
      )

  def test_index_select_dim_out_of_bounds(self):
    with et.assert_raises_message(
        IndexError,
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""index_select(): expected dimension to be in range [-1, 0], got 1""",
        message_reviewed_by="wan",
    ):
      torch.index_select(
          torch.ones(1, device=et.device()),
          1,
          torch.tensor([0], device=et.device(), dtype=torch.long),
      )

  def test_index_select_scalar_input(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""index_select(): expected dim to be 0 for scalar input, got 1""",
    ):
      torch.index_select(
          torch.tensor(1, device=et.device()),
          1,
          torch.tensor([0], device=et.device(), dtype=torch.long),
      )

  def test_index_select_scalar_index(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""index_select(): Index to scalar can have only 1 value, got 2 value(s)""",
        tpu="""index_select(): expected index to be 1D of size 1 for scalar input, got shape [2]""",
    ):
      torch.index_select(
          torch.tensor(1, device=et.device()),
          0,
          torch.tensor([0, 0], device=et.device(), dtype=torch.long),
      )

  def test_cumsum_with_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError if et.is_on_tpu() else RuntimeError,
        gpu="""Expected out tensor to have dtype c10::dummy_int1_7_t<1>, but got float instead""",
        tpu="""cumsum(): TorchTPU does not yet support dtype int1""",
        message_reviewed_by="wan",
    ):
      t = torch.ones(2, 2, device=et.device())
      output = torch.empty_like(t)
      torch.cumsum(t, dim=1, dtype=torch.int1, out=output)

  def test_cumsum_bool_out(self):
    with et.assert_raises_message(
        NotImplementedError,
        gpu=""""cumsum_cuda" not implemented for 'Bool'""",
        tpu="""cumsum(): invalid output dtype bool""",
    ):
      x = torch.tensor([True, False], dtype=torch.bool, device=et.device())
      torch.cumsum(x, dim=0, out=x)

  def test_cumprod_bool_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        gpu=""""cumprod_cuda" not implemented for 'Bool'""",
        tpu="""cumprod(): the dtype argument cannot be bool""",
        message_reviewed_by="wan",
    ):
      x = torch.tensor([1, 2], device=et.device())
      torch.cumprod(x, dim=0, dtype=torch.bool)

  def test_cumprod_bool_out(self):
    with et.assert_raises_message(
        NotImplementedError,
        gpu=""""cumprod_cuda" not implemented for 'Bool'""",
        tpu="""cumprod(): cumprod not implemented for bool""",
        message_reviewed_by="wan",
    ):
      x = torch.tensor([True, False], device=et.device())
      torch.cumprod(x, dim=0, out=x)

  def test_cumsum_dimension_out_of_range(self):
    with et.assert_raises_message(
        IndexError,
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""cumsum(): expected dimension to be in range [-1, 0], got 1""",
        message_reviewed_by="wan",
    ):
      t = torch.ones(1, device=et.device())
      output = torch.empty_like(t)
      torch.cumsum(t, dim=1, out=output)

  def test_prod_out_with_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError if et.is_on_tpu() else RuntimeError,
        gpu="""Expected out tensor to have dtype c10::dummy_int1_7_t<1>, but got float instead""",
        tpu="""prod(): TorchTPU does not yet support dtype int1""",
        message_reviewed_by="wan",
    ):
      t = torch.ones(2, 2, device=et.device())
      output = torch.empty_like(t)
      torch.prod(t, dim=1, dtype=torch.int1, out=output)

  def test_index_add_rank_mismatch(self):
    with et.assert_raises_message(
        RuntimeError if et.is_on_gpu() else IndexError,
        gpu="""index_add_(): Number of indices (1) should be equal to source.size(dim): (2), for dim: 0""",
        tpu="""index_add(): expected self and source to have the same number of dimensions, got 2 and 1""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(2, device=et.device())
      )

  def test_index_add_index_rank_not_1(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""index_add_(): Index is supposed to be a vector, but got dim: 2 with type: Long and size: [1, 1]""",
        tpu="""index_add(): expected index to be 1D, got shape [1, 1]""",
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
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        tpu="""index_add(): expected dimension to be in range [-2, 1], got 2""",
        message_reviewed_by="wan",
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
        gpu="""index_add_(): Number of indices (1) should be equal to source.size(dim): (2), for dim: 0""",
        tpu="""index_add(): expected source to have the same number of elements as the index along dimension 0, got 2 and 1""",
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
        gpu="""source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [2, 2] source.shape = [1, 3]""",
        tpu="""index_add(): expected self and source to have the same size along dimension 1, got 2 and 3""",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 3, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_add_scalar_dim_out_of_range(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""index_add(): expected dim to be 0 for scalar input, got 1""",
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
        gpu="""source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [] source.shape = [1]""",
        tpu="""index_add(): expected source shape to match self shape, excluding the specified dimension, got source shape [1] and self shape []""",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor([1], device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_add_scalar_index_size_ne_1(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""Dimension specified as 0 but tensor has no dimensions""",
        tpu="""index_add(): expected index to be 1D of size 1 for scalar input, got shape [2]""",
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

    tpu_msg = """addmm(): expected input tensor to have at most as many dimensions as the product of mat1 @ mat2, got 3-D input and 2-D product of mat1 @ mat2"""

    with et.assert_raises_message(
        RuntimeError,
        gpu="""expand(torch.cuda.FloatTensor{[2, 2, 2]}, size=[2, 2]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (3)""",
        tpu=tpu_msg,
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_input_not_broadcastable_to_matmul_result(self):
    # Arrange
    input_ = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    gpu_msg = """The expanded size of the tensor (2) must match the existing size (3) at non-singleton dimension 1.  Target sizes: [2, 2].  Tensor sizes: [2, 3]"""
    tpu_msg = """addmm(): input tensor shape [2, 3] cannot be broadcasted to matmul result shape [2, 2]"""

    with et.assert_raises_message(RuntimeError, gpu=gpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_input_on_bool_tensor(self):
    # Arrange
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    beta = True
    alpha = True
    tpu_msg = """addmm(): boolean dtypes are not supported"""

    with et.assert_raises_message(
        RuntimeError,
        gpu=""""addmm_cuda" not implemented for 'Bool'""",
        tpu=tpu_msg,
    ):
      torch.addmm(input_, mat1, mat2, beta=beta, alpha=alpha)

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          tpu_dtype="int64",
          gpu_dtype="Long",
      ),
  )
  def test_addmm_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    input_ = torch.ones(2, 3, dtype=dtype, device=et.device())
    mat1 = torch.ones(2, 4, dtype=dtype, device=et.device())
    mat2 = torch.ones(4, 3, dtype=dtype, device=et.device())

    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""addmm(): not implemented for {tpu_dtype}""",
        gpu=f""""addmm_cuda" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_non_matrix_mat1(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    # This error is created by pytorch before our kernel is called.
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_msg = "addmm: self must be a matrix, got 1-D tensor"
    else:
      tpu_error_msg = "mat1 must be a matrix, got 1-D tensor"
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error_msg,
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_non_matrix_mat2(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, device=et.device(), dtype=torch.float32)
    # This error is created by pytorch before our kernel is called.
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_msg = "addmm: mat2 must be a matrix, got 1-D tensor"
    else:
      tpu_error_msg = "mat2 must be a matrix, got 1-D tensor"
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error_msg,
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_mat1_mat2_mismatch_contracting_dimension(self):
    input_ = torch.ones(13, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(3, 13, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(11, 2, device=et.device(), dtype=torch.float32)
    # This error is created by pytorch before our kernel is called.
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_msg = (
          "addmm: mat1 and mat2 shapes cannot be multiplied (3x13 and 11x2)"
      )
    else:
      tpu_error_msg = (
          "mat1 and mat2 shapes cannot be multiplied (3x13 and 11x2)"
      )
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error_msg,
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_mismatched_input_and_out_dtypes(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.int32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    tpu_msg = (
        "addmm(): expected input and out tensors to have the same dtype, got"
        " float32 vs int32"
    )
    gpu_msg = "Expected out tensor to have dtype float, but got int instead"

    with et.assert_raises_message(RuntimeError, gpu=gpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2, out=out)

  def test_addmm_outdtype_must_match_out_dtype(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out_dtype = torch.int32
    tpu_msg = """addmm(): expected out dtype to match out_dtype, got out dtype float32 and out_dtype int32"""

    # CPU raises NotImplementedError, a subclass of RuntimeError.
    with et.assert_raises_message(
        RuntimeError,
        gpu="""out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs""",
        tpu=tpu_msg,
    ):
      torch.addmm(input_, mat1, mat2, out=out, out_dtype=out_dtype)

  def test_addmm_out_dtype_unsupported(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out_dtype = torch.int1
    tpu_msg = """addmm(): TorchTPU does not yet support the output dtype int1"""

    # CPU raises NotImplementedError, a subclass of RuntimeError.
    with et.assert_raises_message(
        RuntimeError,
        gpu="""out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs""",
        tpu=tpu_msg,
        message_reviewed_by="wan",
    ):
      torch.addmm(input_, mat1, mat2, out_dtype=out_dtype)

  def test__addmm_activation_mismatched_inner_dimensions(self):
    input_ = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(3, 4, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(5, 2, device=et.device(), dtype=torch.float32)
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_msg = (
          "addmm: mat1 and mat2 shapes cannot be multiplied (3x4 and 5x2)"
      )
    else:
      tpu_error_msg = "mat1 and mat2 shapes cannot be multiplied (3x4 and 5x2)"
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error_msg,
    ):
      torch.ops.aten._addmm_activation(
          input_, mat1, mat2, beta=1.0, alpha=1.0, use_gelu=False
      )

  def test__addmm_activation_out_mismatched_inner_dimensions(self):
    input_ = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(3, 4, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(5, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(3, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""addmm: mat1 and mat2 shapes cannot be multiplied (3x4 and 5x2)""",
        tpu="""addmm_activation(): expected size 1 of mat1 to be same as size 0 of mat2, got 4 and 5 respectively""",
    ):
      torch.ops.aten._addmm_activation.out(
          input_, mat1, mat2, beta=1.0, alpha=1.0, use_gelu=False, out=out
      )

  def test__addmm_activation_non_matrix_mat1(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_msg = "addmm: self must be a matrix, got 1-D tensor"
    else:
      tpu_error_msg = "mat1 must be a matrix, got 1-D tensor"
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error_msg,
    ):
      torch.ops.aten._addmm_activation(
          input_, mat1, mat2, beta=1.0, alpha=1.0, use_gelu=False
      )

  def test__addmm_activation_out_non_matrix_mat1(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""addmm: self must be a matrix, got 1-D tensor""",
        tpu="""addmm_activation(): expected mat1 to be a matrix, got 1-D tensor""",
    ):
      torch.ops.aten._addmm_activation.out(
          input_, mat1, mat2, beta=1.0, alpha=1.0, use_gelu=False, out=out
      )

  def test__addmm_activation_non_matrix_mat2(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, 2, device=et.device(), dtype=torch.float32)
    if env.IS_INTERNAL_TORCH_TPU:
      tpu_error_msg = "addmm: mat2 must be a matrix, got 3-D tensor"
    else:
      tpu_error_msg = "mat2 must be a matrix, got 3-D tensor"
    with et.assert_raises_message(
        RuntimeError,
        tpu=tpu_error_msg,
    ):
      torch.ops.aten._addmm_activation(
          input_, mat1, mat2, beta=1.0, alpha=1.0, use_gelu=False
      )

  def test__addmm_activation_out_non_matrix_mat2(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""addmm: mat2 must be a matrix, got 3-D tensor""",
        tpu="""addmm_activation(): expected mat2 to be a matrix, got 3-D tensor""",
    ):
      torch.ops.aten._addmm_activation.out(
          input_, mat1, mat2, beta=1.0, alpha=1.0, use_gelu=False, out=out
      )

  def test__addmm_activation_out_mismatched_dtype_int32(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.int32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    tpu_msg = (
        "addmm_activation(): expected input and out tensors to have the"
        " same dtype, got float32 vs int32"
    )
    gpu_msg = "Expected out tensor to have dtype float, but got int instead"
    with et.assert_raises_message(RuntimeError, gpu=gpu_msg, tpu=tpu_msg):
      torch.ops.aten._addmm_activation.out(
          input_, mat1, mat2, beta=1.0, alpha=1.0, use_gelu=False, out=out
      )

  def test__addmm_activation_out_mismatched_dtype_float16(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.float16)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    tpu_msg = (
        "addmm_activation(): expected input and out tensors to have the"
        " same dtype, got float32 vs float16"
    )
    gpu_msg = (
        "Expected out tensor to have dtype float, but got c10::Half instead"
    )
    with et.assert_raises_message(RuntimeError, gpu=gpu_msg, tpu=tpu_msg):
      torch.ops.aten._addmm_activation.out(
          input_, mat1, mat2, beta=1.0, alpha=1.0, use_gelu=False, out=out
      )

  def test_empty_strided_size_stride_mismatch(self):
    """Tests that empty_strided fails with expected error when size and stride arrays have different lengths."""
    with et.assert_raises_message(
        RuntimeError,
        gpu="""dimensionality of sizes (2) must match dimensionality of strides (1)""",
        tpu="""empty_strided(): expected the dimensionality of sizes to be the same as strides, got size [2] and stride [1]""",
    ):
      torch.empty_strided((2, 3), (1,), device=et.device(), dtype=torch.float32)

  def test_empty_strided_negative_size(self):
    """Tests that empty_strided fails with expected error when size is negative."""
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Trying to create tensor with negative dimension -1: [-1, 2]""",
        tpu="""empty_strided(): expected sizes to be >= 0, got sizes [-1, 2]""",
    ):
      torch.empty_strided(
          (-1, 2), (2, 1), device=et.device(), dtype=torch.float32
      )

  def test_empty_strided_negative_stride(self):
    """Tests that empty_strided fails with expected error when stride is negative."""
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Storage size calculation overflowed with sizes=[2, 2] and strides=[2, -1]""",
        tpu="""empty_strided(): expected strides to be >= 0, got strides [2, -1]""",
    ):
      torch.empty_strided(
          (2, 2), (2, -1), device=et.device(), dtype=torch.float32
      )

  def test_index_put_too_many_indices_after_expanding_boolean_tensors(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        tpu="""index_put_(): expected at most 2 indices for tensor of dimension 2, got 3 index tensors after expanding boolean indices""",
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
        gpu=""""host_softmax" not implemented for 'Int'""",
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
        gpu=""""host_softmax" not implemented for 'Int'""",
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
        gpu=""""trunc_cuda" not implemented for 'Bool'""",
    ):
      torch.trunc(t)

  def test_digamma_unsupported_complex(self):
    t = torch.tensor([1 + 1j], device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""digamma(): expected the input dtype not to be complex, got complex64""",
        gpu=""""digamma_cuda" not implemented for 'ComplexFloat'""",
    ):
      torch.digamma(t)

  def test_logit_unsupported_complex(self):
    t = torch.tensor([1 + 1j], device=et.device(), dtype=torch.complex64)
    out = torch.tensor([1 + 1j], device=et.device(), dtype=torch.complex64)
    funcs = [
        (lambda: torch.logit(t), "logit()"),
        (lambda: torch.logit(t, out=out), "logit()"),
        (lambda: torch.logit_(t), "logit_()"),
        (lambda: torch.ops.aten.logit_backward(t, t), "logit_backward()"),
    ]
    for func, op_prefix in funcs:
      with et.assert_raises_message(
          RuntimeError,
          tpu=f"{op_prefix}: expected non-complex dtype, got complex64",
          gpu=""""logit_cuda" not implemented for 'ComplexFloat'""",
      ):
        func()

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
        gpu=f""""elu_cuda" not implemented for '{cpu_dtype_str}'""",
    ):
      torch.nn.functional.elu(inp)

  def test_gelu_unsupported_input_dtype(self):
    del self  # self is not used in this test
    t = torch.ones(2, device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""gelu(): unsupported input dtype: 'complex64'""",
        gpu=""""GeluCUDAKernelImpl" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.gelu(t)

  def test_gelu_unsupported_output_dtype(self):
    del self  # self is not used in this test
    t = torch.ones(2, device=et.device(), dtype=torch.float32)
    out = torch.ones(2, device=et.device(), dtype=torch.int64)
    with et.assert_raises_message(
        Exception,
        tpu="""gelu(): unsupported output dtype: 'int64'""",
        gpu="""Found dtype Long but expected Float""",
    ):
      torch.nn.functional.gelu(t, out=out)  # pylint: disable=unexpected-keyword-arg

  def test_gelu_unsupported_approximation_type(self):
    t = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gelu(): unsupported approximate argument: invalid""",
        gpu="""approximate argument must be either none or tanh.""",
    ):
      torch.nn.functional.gelu(t, approximate="invalid")

  def test_gelu_backward_grad_input_unsupported_approximation_type(self):
    t = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    grad_input = torch.empty_like(t)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gelu_backward(): unsupported approximate argument: invalid""",
        gpu="""approximate argument must be either none or tanh.""",
    ):
      torch.ops.aten.gelu_backward.grad_input(
          t, t, approximate="invalid", grad_input=grad_input
      )

  def test_glu_unsupported_input_dtype(self):
    if et.is_on_gpu():
      # On TPU, passing an integer tensor raises RuntimeError ("expected
      # the self dtype to be floating point"). On GPU, it fails at the
      # dispatcher level with NotImplementedError ("'glu_cuda' not
      # implemented for 'Int'").
      self.skipTest("GPU behavior difference")
    t = torch.ones(2, 4, device=et.device(), dtype=torch.int32)
    out = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu(): expected the self dtype to be floating point, got int32""",
        gpu=""""glu_cpu" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=1, out=out)

  def test_glu_unsupported_out_dtype(self):
    t = torch.ones(2, 4, device=et.device(), dtype=torch.float32)
    out = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu(): expected the out dtype to be floating point, got int32""",
        gpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=1, out=out)

  def test_glu_invalid_rank(self):
    t = torch.tensor(0.0, device=et.device(), dtype=torch.float32)
    out = torch.tensor(0.0, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu(): expected input tensor to have at least 1 dimension, got 0 dimensions""",
        gpu="""glu does not support 0-dimensional tensors""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=0, out=out)

  def test_glu_invalid_dim(self):
    t = torch.ones(2, 4, device=et.device(), dtype=torch.float32)
    out = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        tpu="""glu(): expected dimension to be in range [-2, 1], got 2""",
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got 2)""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=2, out=out)

  def test_glu_invalid_dim_size(self):
    t = torch.ones(2, 5, device=et.device(), dtype=torch.float32)
    out = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu(): expected the size of dimension 1 to be even, got 5""",
        gpu="""Halving dimension must be even, but dimension 1 is size 5""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu.out(t, dim=1, out=out)

  def test_glu_backward_unsupported_dtypes(self):
    float_tensor = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    int_tensor = torch.ones(2, 4, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected the self dtype to be floating point, got int32""",
        gpu="""Found dtype Float but expected Int""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(
          grad_output=float_tensor, self=int_tensor, dim=1
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected the grad_output dtype to be floating point, got int32""",
        gpu="""Expected grad_output.sizes() == IntArrayRef{iter_shape} to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(
          grad_output=int_tensor, self=float_tensor, dim=1
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected the grad_input dtype to be floating point, got int32""",
        gpu="""Expected grad_output.sizes() == IntArrayRef{iter_shape} to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
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
        gpu="""Expected grad_output.sizes() == IntArrayRef{iter_shape} to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(
          grad_output=float64_tensor, self=float32_tensor, dim=1
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected self and grad_input to have the same dtype, got float32 and float64""",
        gpu="""Expected grad_output.sizes() == IntArrayRef{iter_shape} to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
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
        gpu="""glu does not support 0-dimensional tensors""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(grad_output, self_tensor, dim=0)

  def test_glu_backward_invalid_dim_size(self):
    grad_output = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    self_tensor = torch.ones(2, 5, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected the size of dimension 1 of self to be even, got 5""",
        gpu="""Halving dimension must be even, but dimension 1 is size 5""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(grad_output, self_tensor, dim=1)

  def test_glu_backward_grad_output_shape_mismatch(self):
    grad_output = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    self_tensor = torch.ones(2, 4, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""glu_backward(): expected grad_output shape to be [2, 2], got [2, 3]""",
        gpu="""Expected grad_output.sizes() == IntArrayRef{iter_shape} to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.glu_backward(grad_output, self_tensor, dim=1)

  def test_prelu_kernel_dtype_mismatch(self):
    self_tensor = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    weight = torch.ones(3, device=et.device(), dtype=torch.float64)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Found dtype Double but expected Float""",
        tpu="""prelu_kernel(): expected self and weight to have the same dtype, got float32 and float64""",
    ):
      torch.ops.aten._prelu_kernel(self_tensor, weight)

  def test_prelu_kernel_backward_dtype_mismatch(self):
    grad_output = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    self_tensor = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    weight = torch.ones(3, device=et.device(), dtype=torch.bfloat16)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Found dtype BFloat16 but expected Float""",
        tpu="""prelu_kernel_backward(): expected grad_output, self, and weight to have the same dtype, got float32, float32, and bfloat16""",
    ):
      torch.ops.aten._prelu_kernel_backward(grad_output, self_tensor, weight)

  def test_group_norm_backward_grad_out_numel_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Expected dY.numel() == N * C * HxW to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
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
        gpu="""Expected X.numel() == N * C * HxW to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
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
        gpu="""Expected mean.numel() == N * G to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
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
        gpu="""Expected rstd.numel() == N * G to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
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
        gpu="""Expected !gamma.defined() || gamma.numel() == C to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
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
          expected_error="""ragged_dot(): expected lhs to be 2D, got dim: 1""",
      ),
      dict(
          testcase_name="invalid_rhs_shape",
          lhs_arg=(3, 2),
          rhs_arg=(2, 2),
          group_sizes_arg=[1, 1],
          expected_error="""ragged_dot(): expected rhs to be 3D, got dim: 2""",
      ),
      dict(
          testcase_name="invalid_group_sizes_shape",
          lhs_arg=(3, 2),
          rhs_arg=(2, 2, 2),
          group_sizes_arg=[[1, 1]],
          expected_error="""ragged_dot(): expected group_sizes to be 1D, got dim: 2""",
      ),
      dict(
          testcase_name="mismatched_contracting_dims",
          lhs_arg=(3, 2),
          rhs_arg=(2, 3, 2),
          group_sizes_arg=[1, 1],
          expected_error="""ragged_dot(): expected contracting dimension to be the same, got 2 vs 3""",
      ),
      dict(
          testcase_name="mismatched_group_dims",
          lhs_arg=(3, 2),
          rhs_arg=(2, 2, 2),
          group_sizes_arg=[1, 1, 1],
          expected_error="""ragged_dot(): expected lhs and group_sizes to have the same number of groups, got 2 vs 3""",
      ),
  )
  def test_ragged_dot(self, lhs_arg, rhs_arg, group_sizes_arg, expected_error):
    """Tests that tpu.ragged_dot fails with expected errors.

    This test checks various error conditions for the `tpu.ragged_dot`
    operation, such as invalid input shapes, mismatched dimensions, and
    incorrect group sizes.

    Args:
      lhs_arg: The shape argument for the left-hand side tensor.
      rhs_arg: The shape argument for the right-hand side tensor.
      group_sizes_arg: The argument for the group sizes tensor.
      expected_error: The expected error message substring.
    """
    if et.is_on_gpu():
      # torch.ops.tpu.ragged_dot is a custom TPU operator. On GPU,
      # calling it immediately fails with a backend dispatch error
      # before any op error checks (such as invalid shapes or
      # contracting dims) can run.
      self.skipTest("GPU behavior difference")
    lhs = torch.ones(*lhs_arg, dtype=torch.float32, device=et.device())
    rhs = torch.ones(*rhs_arg, dtype=torch.float32, device=et.device())
    group_sizes = torch.tensor(
        group_sizes_arg, dtype=torch.int32, device=et.device()
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu=expected_error,
        gpu="""Could not run 'tpu::ragged_dot' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'tpu::ragged_dot' is only available for these backends: [CPU, PrivateUse1, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMTIA, AutogradMAIA, AutogradPrivateUse1, AutogradMeta, Tracer, AutocastCPU, AutocastMTIA, AutocastMAIA, AutocastXPU, AutocastMPS, AutocastCUDA, AutocastPrivateUse1, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].""",
    ):
      torch.ops.tpu.ragged_dot(lhs, rhs, group_sizes)

  @parameterized.named_parameters(
      dict(
          testcase_name="bool",
          dtype=torch.bool,
          error=RuntimeError,
          tpu_message="max_pool2d(): bool dtype is not supported",
          gpu_dtype="Bool",
      ),
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          error=NotImplementedError,
          tpu_message="max_pool2d(): not implemented for int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          error=NotImplementedError,
          tpu_message="max_pool2d(): not implemented for int64",
          gpu_dtype="Long",
      ),
  )
  def test_max_pool2d_unsupported_dtypes(
      self,
      dtype: torch.dtype,
      error: type[Exception],
      tpu_message: str,
      gpu_dtype: str,
  ):
    t = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=dtype)
    with et.assert_raises_message(
        error,
        tpu=tpu_message,
        gpu=f""""max_pool2d_with_indices_out_cuda_frame" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.nn.functional.max_pool2d(t, kernel_size=3)

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          tpu_dtype="int64",
          gpu_dtype="Long",
      ),
  )
  def test_max_pool2d_with_indices_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    t = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=dtype)
    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""max_pool2d_with_indices(): not implemented for {tpu_dtype}""",
        gpu=f""""max_pool2d_with_indices_out_cuda_frame" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.nn.functional.max_pool2d(t, kernel_size=3, return_indices=True)

  def test_masked_scatter_invalid_mask_dtype(self):
    device = et.device()
    t = torch.randn(4, 4, device=device, dtype=torch.float32)
    source = torch.randn(16, device=device, dtype=torch.float32)
    mask_int = torch.ones(4, 4, device=device, dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        gpu="""masked_scatter_ only supports boolean masks, but got mask with dtype int""",
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
        gpu="""masked_scatter_: expected self and source to have same dtypes but got Float and Int""",
        tpu="""masked_scatter_(): expected same dtype for self and source, got self dtype float32 and source dtype int32""",
    ):
      torch.masked_scatter(t, mask, source)

  def test_arange_zero_step(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected step to be non-zero, got 0""",
        gpu="""step must be nonzero""",
    ):
      torch.arange(1, 10, 0, device=et.device())

  def test_linspace_negative_steps(self):
    out = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linspace(): expected non-negative steps, got -1""",
        gpu="""number of steps must be non-negative""",
    ):
      torch.linspace(0, 10, -1, device=et.device(), out=out)

  def test_linspace_bool_error(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""linspace(): expected output dtype to be other than bool, got bool""",
        gpu=""""linspace_cuda" not implemented for 'Bool'""",
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
        gpu="""unsupported range: inf -> 0""",
    ):
      torch.arange(float("inf"), 0, -1, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected [start, end) interval to have finite bounds, got [0, inf)""",
        gpu="""unsupported range: 0 -> inf""",
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
        gpu="""upper bound and lower bound inconsistent with step sign""",
    ):
      torch.arange(0, 10, -1, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected step to be negative since start (10) > end (0), got step=1""",
        gpu="""upper bound and lower bound inconsistent with step sign""",
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
        gpu="""upper bound and lower bound inconsistent with step sign""",
    ):
      torch.arange(0, 10, float("-inf"), device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""arange(): expected step to be negative since start (10) > end (0), got step=inf""",
        gpu="""upper bound and lower bound inconsistent with step sign""",
    ):
      torch.arange(10, 0, float("inf"), device=et.device())

  @parameterized.named_parameters(
      dict(
          testcase_name="bool",
          dtype=torch.bool,
          error=RuntimeError,
          tpu_message="max_pool3d_with_indices(): bool dtype is not supported",
          gpu_dtype="Bool",
      ),
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          error=NotImplementedError,
          tpu_message="max_pool3d_with_indices(): not implemented for int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          error=NotImplementedError,
          tpu_message="max_pool3d_with_indices(): not implemented for int64",
          gpu_dtype="Long",
      ),
  )
  def test_max_pool3d_unsupported_dtypes(
      self,
      dtype: torch.dtype,
      error: type[Exception],
      tpu_message: str,
      gpu_dtype: str,
  ):
    t = torch.zeros((1, 1, 4, 4, 4), device=et.device(), dtype=dtype)
    with et.assert_raises_message(
        error,
        tpu=tpu_message,
        gpu=f""""max_pool3d_with_indices_out_frame" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.nn.functional.max_pool3d(t, kernel_size=3)

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
        gpu=f""""cdist_cuda" not implemented for '{cpu_dtype_str}'""",
        message_reviewed_by="wan",
    ):
      torch.cdist(x1.to(dtype), x2, p=1.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""cdist_forward(): expected the second argument's dtype not to be"""
        f""" bfloat16 or float16, got {tpu_dtype_str}""",
        gpu=f"""expected scalar type Float but found {cpu_dtype_str}""",
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
        gpu="""cdist only supports floating-point dtypes, X1 got: Int""",
        message_reviewed_by="wan",
    ):
      torch.cdist(x1.to(torch.int32), x2, p=1.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""cdist_forward(): expected the second argument's dtype to be floating point, got int32""",
        gpu="""cdist only supports floating-point dtypes, X2 got: Int""",
        message_reviewed_by="wan",
    ):
      torch.cdist(x1, x2.to(torch.int32), p=1.0)

  def test_cdist_forward_unsupported_p(self):
    x1 = torch.randn(2, 2, device=et.device(), dtype=torch.float32)
    x2 = torch.randn(2, 2, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""cdist_forward(): expected the p value to be >= 0, got -1""",
        gpu="""cdist only supports non-negative p values""",
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
        gpu=f""""cdist_cuda_backward" not implemented for '{cpu_dtype_str}'""",
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
        gpu=""""cdist_cuda_backward" not implemented for 'Int'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._cdist_backward(grad, x1, x2, 1.0, cdist)

  def test_exponential_unsupported_dtypes(self):
    device = et.device()
    t_int = torch.ones((2, 2), device=device, dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""exponential_(): expected input tensor dtype to be a floating-point real type, got int32""",
        gpu="""Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified Int""",
    ):
      t_int.exponential_()

    t_complex = torch.ones((2, 2), device=device, dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""exponential_(): expected input tensor dtype to be a floating-point real type, got complex64""",
        gpu="""Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified ComplexFloat""",
    ):
      t_complex.exponential_()

  def test_bernoulli_invalid_p(self):
    device = et.device()
    t = torch.ones((2, 2), device=device, dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""bernoulli_(): expected p to be in the range [0, 1], got 1.5""",
        gpu="""bernoulli_ expects p to be in [0, 1], but got p=1.5""",
    ):
      t.bernoulli(p=1.5)

  def test_add_smaller_out_alias(self):
    """Tests that add fails when the out tensor is a smaller alias of an input."""
    a = torch.ones(4, device=et.device())
    b = torch.ones(4, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""add(): output with shape [4] doesn't match the broadcast shape of the tensor being operated on in-place, which has shape [1]""",
        gpu="""unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.""",
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
        tpu="""avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, complex64), got complex64""",
        gpu=""""avg_pool2d_out_cuda_frame" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.avg_pool2d(t_complex, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, complex64), got uint8""",
        gpu=""""avg_pool2d_out_cuda_frame" not implemented for 'Byte'""",
    ):
      torch.nn.functional.avg_pool2d(t_uint8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, complex64), got int8""",
        gpu=""""avg_pool2d_out_cuda_frame" not implemented for 'Char'""",
    ):
      torch.nn.functional.avg_pool2d(t_int8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, complex64), got int16""",
        gpu=""""avg_pool2d_out_cuda_frame" not implemented for 'Short'""",
    ):
      torch.nn.functional.avg_pool2d(t_int16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, complex64), got int32""",
        gpu=""""avg_pool2d_out_cuda_frame" not implemented for 'Int'""",
    ):
      torch.nn.functional.avg_pool2d(t_int32, kernel_size=3)

  @parameterized.named_parameters(
      dict(
          testcase_name="avg_pool2d",
          op=torch.nn.functional.avg_pool2d,
          shape=(1, 1, 4, 4),
          op_name="avg_pool2d",
          gpu_op_name="avg_pool2d_out_cuda_frame",
      ),
      dict(
          testcase_name="avg_pool3d",
          op=torch.nn.functional.avg_pool3d,
          shape=(1, 1, 4, 4, 4),
          op_name="avg_pool3d",
          gpu_op_name="avg_pool3d_out_cuda",
      ),
  )
  def test_avg_pool_int64_unsupported(
      self, op, shape, op_name: str, gpu_op_name: str
  ):
    t = torch.zeros(shape, device=et.device(), dtype=torch.int64)
    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""{op_name}(): not implemented for int64""",
        gpu=f""""{gpu_op_name}" not implemented for 'Long'""",
        message_reviewed_by="gunhyun",
    ):
      op(t, kernel_size=3)

  def test_avg_pool3d_unsupported_dtypes(self):
    if et.is_on_gpu():
      # On TPU, avg_pool3d raises RuntimeError for unsupported dtypes
      # (bool, bfloat16, float16). On GPU, CUDA natively supports 3D
      # average pooling for float16 and bfloat16 without error.
      self.skipTest("GPU behavior difference")
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
        tpu="""avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, complex64), got bool""",
        gpu=""""avg_pool3d_out_cuda" not implemented for 'Bool'""",
    ):
      torch.nn.functional.avg_pool3d(t_bool, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, complex64), got bfloat16""",
        gpu=""""avg_pool3d_out_cuda" not implemented for 'BFloat16'""",
    ):
      torch.nn.functional.avg_pool3d(t_bf16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, complex64), got float16""",
        gpu=""""avg_pool3d_out_cuda" not implemented for 'Half'""",
    ):
      torch.nn.functional.avg_pool3d(t_f16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, complex64), got complex64""",
        gpu=""""avg_pool3d_out_cuda" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.avg_pool3d(t_complex, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, complex64), got uint8""",
        gpu=""""avg_pool3d_out_cuda" not implemented for 'Byte'""",
    ):
      torch.nn.functional.avg_pool3d(t_uint8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, complex64), got int8""",
        gpu=""""avg_pool3d_out_cuda" not implemented for 'Char'""",
    ):
      torch.nn.functional.avg_pool3d(t_int8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, complex64), got int16""",
        gpu=""""avg_pool3d_out_cuda" not implemented for 'Short'""",
    ):
      torch.nn.functional.avg_pool3d(t_int16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, complex64), got int32""",
        gpu=""""avg_pool3d_out_cuda" not implemented for 'Int'""",
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
        gpu=f""""pdist_cuda" not implemented for '{cpu_dtype_str}'""",
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
        gpu=f""""pdist_cuda_backward" not implemented for '{cpu_dtype_str}'""",
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
        tpu="""""",
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
        gpu=""""replication_pad1d_cuda" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad2d(): not implemented for 'Bool'""",
        gpu=""""replication_pad2d_cuda" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""replication_pad3d(): not implemented for 'Bool'""",
        gpu=""""replication_pad3d_cuda" not implemented for 'Bool'""",
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
        gpu=""""reflection_pad1d_out" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""reflection_pad2d(): not implemented for bool""",
        gpu=""""reflection_pad2d_out_template" not implemented for 'Bool'""",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""reflection_pad3d(): not implemented for bool""",
        gpu=""""reflection_pad3d_out_cuda" not implemented for 'Bool'""",
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
        tpu="""adaptive_avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got complex64""",
        gpu=""""adaptive_avg_pool2d_cuda" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_complex, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got uint8""",
        gpu=""""adaptive_avg_pool2d_cuda" not implemented for 'Byte'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_uint8, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got int8""",
        gpu=""""adaptive_avg_pool2d_cuda" not implemented for 'Char'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int8, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got int16""",
        gpu=""""adaptive_avg_pool2d_cuda" not implemented for 'Short'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int16, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got int32""",
        gpu=""""adaptive_avg_pool2d_cuda" not implemented for 'Int'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int32, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool2d(): expected input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got int64""",
        gpu=""""adaptive_avg_pool2d_cuda" not implemented for 'Long'""",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int64, output_size=2)

  def test_adaptive_max_pool2d_unsupported_dtypes(self):
    t_bool = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.bool)
    t_complex = torch.zeros(
        (1, 1, 4, 4), device=et.device(), dtype=torch.complex64
    )
    t_complex128 = torch.zeros(
        (1, 1, 4, 4), device=et.device(), dtype=torch.complex128
    )
    t_uint8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.uint8)
    t_int8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int8)
    t_int16 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int16)
    t_int32 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int32)
    t_int64 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): bool dtype is not supported""",
        gpu=""""adaptive_max_pool2d_cuda" not implemented for 'Bool'""",
    ):
      torch.nn.functional.adaptive_max_pool2d(t_bool, output_size=2)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""adaptive_max_pool2d(): not implemented for complex64""",
        gpu=""""adaptive_max_pool2d_cuda" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.adaptive_max_pool2d(t_complex, output_size=2)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""adaptive_max_pool2d(): not implemented for complex128""",
        gpu=""""adaptive_max_pool2d_cuda" not implemented for 'ComplexDouble'""",
    ):
      torch.nn.functional.adaptive_max_pool2d(t_complex128, output_size=2)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""adaptive_max_pool2d(): not implemented for uint8""",
        gpu=""""adaptive_max_pool2d_cuda" not implemented for 'Byte'""",
    ):
      torch.nn.functional.adaptive_max_pool2d(t_uint8, output_size=2)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""adaptive_max_pool2d(): not implemented for int8""",
        gpu=""""adaptive_max_pool2d_cuda" not implemented for 'Char'""",
    ):
      torch.nn.functional.adaptive_max_pool2d(t_int8, output_size=2)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""adaptive_max_pool2d(): not implemented for int16""",
        gpu=""""adaptive_max_pool2d_cuda" not implemented for 'Short'""",
    ):
      torch.nn.functional.adaptive_max_pool2d(t_int16, output_size=2)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""adaptive_max_pool2d(): not implemented for int32""",
        gpu=""""adaptive_max_pool2d_cuda" not implemented for 'Int'""",
    ):
      torch.nn.functional.adaptive_max_pool2d(t_int32, output_size=2)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""adaptive_max_pool2d(): not implemented for int64""",
        gpu=""""adaptive_max_pool2d_cuda" not implemented for 'Long'""",
    ):
      torch.nn.functional.adaptive_max_pool2d(t_int64, output_size=2)

  def test_adaptive_max_pool2d_invalid_rank(self):
    inp_2d = torch.ones(10, 10, device=et.device())
    inp_5d = torch.ones(1, 1, 1, 10, 10, device=et.device())
    out = torch.empty(1, device=et.device())
    indices = torch.empty(1, device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): expected 3D or 4D tensor, got [10, 10]""",
        gpu="""adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: [10, 10]""",
    ):
      torch.ops.aten.adaptive_max_pool2d.out(
          inp_2d, [2, 2], out=out, indices=indices
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): expected 3D or 4D tensor, got [1, 1, 1, 10, 10]""",
        gpu="""adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: [1, 1, 1, 10, 10]""",
    ):
      torch.ops.aten.adaptive_max_pool2d.out(
          inp_5d, [2, 2], out=out, indices=indices
      )

  def test_adaptive_max_pool2d_empty_spatial_dim(self):
    inp_empty_h = torch.ones(1, 0, 4, device=et.device())
    inp_empty_w = torch.ones(1, 4, 0, device=et.device())
    out = torch.empty(1, device=et.device())
    indices = torch.empty(1, device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): expected input to have non-zero size for non-batch dimensions, got [1, 0, 4] with dimension 1 being empty""",
        gpu="""adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, but input has sizes [1, 0, 4] with dimension 1 being empty""",
    ):
      torch.ops.aten.adaptive_max_pool2d.out(
          inp_empty_h, [2, 2], out=out, indices=indices
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): expected input to have non-zero size for non-batch dimensions, got [1, 4, 0] with dimension 2 being empty""",
        gpu="""adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, but input has sizes [1, 4, 0] with dimension 2 being empty""",
    ):
      torch.ops.aten.adaptive_max_pool2d.out(
          inp_empty_w, [2, 2], out=out, indices=indices
      )

  def test_adaptive_max_pool2d_invalid_output_size(self):
    inp = torch.ones(1, 1, 4, 4, device=et.device())
    out = torch.empty(1, device=et.device())
    indices = torch.empty(1, device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): expected output_size to have 2 elements, got 1""",
        gpu="""adaptive_max_pool2d(): internal error: output_size.size() must be 2""",
    ):
      torch.ops.aten.adaptive_max_pool2d.out(inp, [1], out=out, indices=indices)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): expected output_size to have 2 elements, got 3""",
        gpu="""adaptive_max_pool2d(): internal error: output_size.size() must be 2""",
    ):
      torch.ops.aten.adaptive_max_pool2d.out(
          inp, [1, 2, 3], out=out, indices=indices
      )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): expected output size to be non-negative, got [-1, 2]""",
        gpu="""numel: integer multiplication overflow""",
    ):
      torch.ops.aten.adaptive_max_pool2d.out(
          inp, [-1, 2], out=out, indices=indices
      )

  def test_adaptive_max_pool2d_out_dtype_mismatch(self):
    inp = torch.ones(1, 1, 4, 4, device=et.device(), dtype=torch.float32)
    out = torch.empty(1, device=et.device(), dtype=torch.int32)
    indices = torch.empty(1, device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): expected out tensor to have dtype float32, got int32""",
        gpu="""Expected out tensor to have dtype float, but got int instead""",
    ):
      torch.ops.aten.adaptive_max_pool2d.out(
          inp, [2, 2], out=out, indices=indices
      )

  def test_adaptive_max_pool2d_indices_dtype_mismatch(self):
    inp = torch.ones(1, 1, 4, 4, device=et.device(), dtype=torch.float32)
    out = torch.empty(1, device=et.device(), dtype=torch.float32)
    indices = torch.empty(1, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_max_pool2d(): expected indices tensor to have dtype int64, got int32""",
        gpu="""Expected out tensor to have dtype long, but got int instead""",
    ):
      torch.ops.aten.adaptive_max_pool2d.out(
          inp, [2, 2], out=out, indices=indices
      )

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
        tpu="""adaptive_avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, int64, complex64), got bool""",
        gpu=""""adaptive_avg_pool3d_cuda" not implemented for 'Bool'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_bool, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, int64, complex64), got complex64""",
        gpu=""""adaptive_avg_pool3d_cuda" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_complex, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, int64, complex64), got uint8""",
        gpu=""""adaptive_avg_pool3d_cuda" not implemented for 'Byte'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_uint8, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, int64, complex64), got int8""",
        gpu=""""adaptive_avg_pool3d_cuda" not implemented for 'Char'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int8, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, int64, complex64), got int16""",
        gpu=""""adaptive_avg_pool3d_cuda" not implemented for 'Short'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int16, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, int64, complex64), got int32""",
        gpu=""""adaptive_avg_pool3d_cuda" not implemented for 'Int'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int32, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): expected input dtype to be none of (bool, uint8, int8, int16, int32, int64, complex64), got int64""",
        gpu=""""adaptive_avg_pool3d_cuda" not implemented for 'Long'""",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int64, output_size=3)

  def test_floor_divide_complex64(self):
    lhs = torch.arange(5, device=et.device())
    rhs = torch.arange(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""floor_divide(): expected dtype of the first argument to be neither complex nor bool, got complex64""",
        gpu=""""div_floor_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.floor_divide(lhs.to(torch.complex64), rhs)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""floor_divide(): expected dtype of the second argument to be neither complex nor bool, got complex64""",
        gpu=""""div_floor_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.floor_divide(lhs, rhs.to(torch.complex64))

  def test_atan2_complex(self):
    x = torch.tensor([1.0, 2.0], device=et.device())
    y = torch.tensor([1.0, 2.0], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""atan2(): expected the dtype of the first argument not to be complex, got complex64""",
        gpu=""""atan2_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.atan2(x.to(torch.complex64), y)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""atan2(): expected the dtype of the second argument not to be complex, got complex64""",
        gpu=""""atan2_cuda" not implemented for 'ComplexFloat'""",
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
        gpu=f""""{op_name}_cuda" not implemented for 'Double'""",
        message_reviewed_by="wan",
    ):
      op(x.to(torch.float64), y)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the dtype of the second argument to be neither floating-point nor complex, got float64""",
        gpu=f""""{op_name}_cuda" not implemented for 'Double'""",
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
        gpu=f""""{op_name_cpu}_cuda" not implemented for 'Double'""",
        message_reviewed_by="wan",
    ):
      op(x.to(torch.float64), y)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name_tpu}(): expected the dtype of the second argument to be integer, got float64""",
        gpu=f""""{op_name_cpu}_cuda" not implemented for 'Double'""",
        message_reviewed_by="wan",
    ):
      op(x, y.to(torch.float64))

  def test_bitwise_left_shift_scalar_float(self):
    x = torch.ones(5, dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""__lshift__(): expected the dtype of the second argument to be integer, got float64""",
        gpu=""""lshift_cuda" not implemented for 'Float'""",
        message_reviewed_by="wan",
    ):
      x.__lshift__(1.5)

  def test_col2im_output_size_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected output_size to have 2 dimensions, got 3""",
        gpu="""It is expected output_size equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_kernel_size_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected kernel_size to have 2 dimensions, got 3""",
        gpu="""It is expected kernel_size equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_dilation_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected dilation to have 2 dimensions, got 3""",
        gpu="""It is expected dilation equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1, 1), (0, 0), (1, 1))

  def test_col2im_padding_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected padding to have 2 dimensions, got 3""",
        gpu="""It is expected padding equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0, 0), (1, 1))

  def test_col2im_stride_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected stride to have 2 dimensions, got 3""",
        gpu="""It is expected stride equals to 2, but got size 3""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1, 1))

  def test_col2im_input_must_be_3d(self):
    img = torch.randn(1, 4, 16, 1, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected input to have 3 dimensions (batch, channels, length), got 4""",
        gpu="""Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch size and non-zero dimensions for input, but got: [1, 4, 16, 1]""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_kernel_size_must_be_positive(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected kernel size to be positive, got 0""",
        gpu="""kernel size should be greater than zero, but got kernel_height: 0 kernel_width: 2""",
    ):
      torch.ops.aten.col2im(img, (5, 5), (0, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_channels_divisibility(self):
    img = torch.randn(1, 5, 15, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""col2im(): expected input channels to be divisible by kernel product (4), got 5""",
        gpu="""Expected size of input's dimension 1 to be divisible by the product of kernel_size, but got input.size(1)=5 and kernel_size=(2, 2).""",
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
        gpu="""stride should be greater than zero, but got stride_height: 0 stride_width: 1""",
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
        gpu="""stride should be greater than zero, but got stride_height: 1 stride_width: -1""",
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
        gpu="""Given output_size=(5, 5), kernel_size=(2, 2), dilation=(1, 1), padding=(0, 0), stride=(1, 1), expected size of input's dimension 2 to match the calculated number of sliding blocks 4 * 4 = 16, but got input.size(2)=15.""",
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
        gpu=""""compare_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      op(lhs.to(torch.complex64), rhs)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the dtype of the second argument not to be complex, got complex64""",
        gpu=""""compare_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      op(lhs, rhs.to(torch.complex64))

    # TODO: b/478955517 dtype checks should run after dtype promotion.
    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the dtype of the second argument not to be complex, got complex128""",
        gpu=""""compare_cuda" not implemented for 'ComplexFloat'""",
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
          gpu=""""remainder_cuda" not implemented for 'ComplexFloat'""",
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
        tpu="""foreach_abs_(): expected no tensor in the self list to be complex, got 2 complex tensors: complex64 at index 1 and complex64 at index 2""",
        gpu="""In-place abs is not supported for complex tensors.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_abs_(self_list)

  def test_foreach_add_int_tensors_float_alpha(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_add(): expected alpha to be integral for integral input tensors, got float64""",
        gpu="""For integral input tensors, argument alpha must not be a floating point number.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_add(self_list, other_list, alpha=1.5)

  def test_foreach_add_int_tensors_bool_alpha(self):
    if et.is_on_gpu():
      # On TPU, boolean alpha with integer tensors raises RuntimeError
      # ("expected input tensor dtypes to be bool when alpha dtype is
      # bool"). On GPU, CUDA silently promotes bool alpha True to
      # integer 1 and completes without raising an error.
      self.skipTest("GPU behavior difference")

    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_add(): expected input tensor dtypes to be bool when alpha dtype is bool, got int32 and int32""",
        gpu="""Boolean alpha only supported for Boolean results.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_add(self_list, other_list, alpha=True)

  def test_foreach_add_inplace_int_and_float(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_add_(): expected the scalar dtype to be castable to the tensor dtype (e.g. bool to int or int to float), got float64 and int32""",
        gpu="""result type Float can't be cast to the desired output type Int""",
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
        gpu="""result type Long can't be cast to the desired output type Bool""",
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
        tpu="""foreach_sub(): expected no tensor in the self list to be bool, got 1 bool tensor: bool at index 1""",
        gpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
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
        gpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
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
        tpu="""foreach_sub(): expected no scalar in the scalars list to be bool, got 1 bool scalar: true at index 1""",
        gpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_sub(self_list, [1, True])

  def test_foreach_clamp_max_scalar_bool(self):
    self_list = [
        torch.tensor([True, False], dtype=torch.bool, device=et.device()),
    ]

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""foreach_clamp_max(): not implemented for bool""",
        gpu=""""clamp_scalar_cuda" not implemented for 'Bool'""",
    ):
      torch._foreach_clamp_max(self_list, True)

  @parameterized.named_parameters(
      dict(testcase_name="tensor_list", other_kind="tensor_list"),
      dict(testcase_name="scalar", other_kind="scalar"),
      dict(testcase_name="scalar_list", other_kind="scalar_list"),
      dict(testcase_name="tensor", other_kind="tensor"),
  )
  def test_foreach_div_inplace_integral(self, other_kind: str):
    device = et.device()
    for dtype, tpu_dtype, gpu_dtype in (
        (torch.int32, "int32", "Int"),
        (torch.int64, "int64", "Long"),
    ):
      self_list = [torch.tensor([4, 6], dtype=dtype, device=device)]
      other = {
          "tensor_list": [torch.tensor([2, 3], dtype=dtype, device=device)],
          "scalar": 2,
          "scalar_list": [2],
          "tensor": torch.tensor(2, dtype=dtype, device=device),
      }[other_kind]

      with et.assert_raises_message(
          RuntimeError,
          tpu=f"""foreach_div_(): expected no tensor in the self list to be integral, got 1 integral tensor: {tpu_dtype} at index 0""",
          gpu=f"""result type Float can't be cast to the desired output type {gpu_dtype}""",
          message_reviewed_by="gunhyun",
      ):
        torch._foreach_div_(self_list, other)

  def test_foreach_sub_int_tensors_float_alpha(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_sub(): expected alpha to be integral for integral input tensors, got float64""",
        gpu="""For integral input tensors, argument alpha must not be a floating point number.""",
        message_reviewed_by="wan",
    ):
      torch._foreach_sub(self_list, other_list, alpha=1.5)

  def test_foreach_sub_inplace_int_tensors_float_alpha(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foreach_sub_(): expected alpha to be integral for integral input tensors, got float64""",
        gpu="""For integral input tensors, argument alpha must not be a floating point number.""",
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
        tpu="""foreach_sqrt_(): expected no tensor in the self list to be integral, got 2 integral tensors: int32 at index 0 and int32 at index 2""",
        gpu="""result type Float can't be cast to the desired output type Int""",
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
        gpu="""Integer division with addcdiv is no longer supported, and in a future  release addcdiv will perform a true division of tensor1 and tensor2. The historic addcdiv behavior can be implemented as (input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) for integer inputs and as (input + value * tensor1 / tensor2) for float inputs. The future addcdiv behavior is just the latter implementation: (input + value * tensor1 / tensor2), for all dtypes.""",
    ):
      torch._foreach_addcdiv(self_list, tensor1_list, tensor2_list)

  def test_foreach_unary_ops_complex(self):
    for dtype in [torch.complex64, torch.complex128]:
      t = torch.randn(2, 2, dtype=dtype, device=et.device())
      ops = [
          torch._foreach_ceil,
          torch._foreach_erf,
          torch._foreach_erfc,
          torch._foreach_floor,
          torch._foreach_frac,
          torch._foreach_lgamma,
          torch._foreach_round,
          torch._foreach_trunc,
      ]
      for op in ops:
        with et.assert_raises_message(
            RuntimeError,
            tpu=re.compile(
                r"foreach_[a-z0-9_]+\(\): expected no tensor in the self"
                r" list to be complex, got 1 complex tensor: "
                r"complex(64|128) at index 0"
            ),
            gpu=re.compile(r".*not implemented for 'Complex(Float|Double)'"),
            message_reviewed_by="wan",
        ):
          op([t])

  def test_erf_unsupported_complex(self):
    for dtype in [torch.complex64, torch.complex128]:
      t = torch.randn(2, 2, dtype=dtype, device=et.device())
      with et.assert_raises_message(
          RuntimeError,
          tpu=re.compile(
              r"erf(\.out)?\(\): expected self not to be complex, got"
              r" complex(64|128)"
          ),
          gpu=re.compile(
              r".*not implemented for ('ComplexFloat'|'ComplexDouble')"
          ),
          message_reviewed_by="wan",
      ):
        torch.erf(t)

  def test_erfinv_unsupported_complex(self):
    for dtype in [torch.complex64, torch.complex128]:
      t = torch.randn(2, 2, dtype=dtype, device=et.device())
      with et.assert_raises_message(
          RuntimeError,
          tpu=re.compile(
              r"erfinv(\.out)?\(\): expected self not to be complex, got"
              r" complex(64|128)"
          ),
          gpu=re.compile(
              r".*not implemented for ('ComplexFloat'|'ComplexDouble')"
          ),
          message_reviewed_by="wan",
      ):
        torch.erfinv(t)

  def test_cat_out_invalid_cast(self):
    """Tests that cat fails when the out tensor has an incompatible dtype."""
    t_f32 = torch.tensor([1.0, 2.0], device=et.device(), dtype=torch.float32)
    out_int32 = torch.zeros(2, device=et.device(), dtype=torch.int32)
    err_type = RuntimeError if et.is_on_tpu() else TypeError
    with et.assert_raises_message(
        err_type,
        tpu="""cat(): expected the input to be castable to the desired dtype int32, got float32""",
        gpu="""torch.cat(): input types can't be cast to the desired output type Int""",
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
        gpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
        message_reviewed_by="wan",
    ):
      torch.sub(lhs.to(torch.bool), rhs, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sub(): the dtype of the second argument cannot be bool""",
        gpu="""Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.""",
        message_reviewed_by="wan",
    ):
      torch.sub(lhs, rhs.to(torch.bool), out=out)

  def _test_aminmax_output_dtype_mismatch_impl(
      self, op_name: str, op: Any, gpu: str
  ):
    tensor = torch.ones(5, device=et.device(), dtype=torch.int64)
    out = _get_aminmax_outputs(op, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected output tensor dtype to match the dtype of the first argument (int64), got complex64""",
        gpu=gpu,
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
        gpu="""Expected the dtype for input and out to match, but got Long for input's dtype and ComplexFloat for out's dtype.""",
    )

  def test_aminmax_output_dtype_mismatch(self):
    self._test_aminmax_output_dtype_mismatch_impl(
        op_name="aminmax",
        op=torch.aminmax,
        gpu="""Expected out tensor to have dtype long, but got c10::complex<float> instead""",
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

    if op is torch.aminmax and env.IS_INTERNAL_TORCH_TPU:
      err_type = TypeError
      tpu_msg = "aminmax not implemented for ComplexFloat"
      gpu_msg = "aminmax not implemented for ComplexFloat"
    else:
      err_type = RuntimeError
      tpu_msg = f"""{op_name_tpu}(): expected the dtype of the input not to be complex, got complex64"""
      gpu_msg = f""""{op_name_cpu}_cuda" not implemented for 'ComplexFloat'"""

    with et.assert_raises_message(
        err_type,
        tpu=tpu_msg,
        gpu=gpu_msg,
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
        gpu="""Expected both inputs to be Half, Float or Double tensors but got Int and Float""",
        message_reviewed_by="wan",
    ):
      torch.complex(real.to(torch.int32), img, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""complex(): expected the dtype of the second argument to be float32 or float64, got int32""",
        gpu="""Expected both inputs to be Half, Float or Double tensors but got Float and Int""",
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
        gpu="""Expected both inputs to be Half, Float or Double tensors but got Int and Float""",
        message_reviewed_by="wan",
    ):
      torch.polar(absv.to(torch.int32), angle, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""polar(): expected the dtype of the second argument to be float32 or float64, got int32""",
        gpu="""Expected both inputs to be Half, Float or Double tensors but got Float and Int""",
        message_reviewed_by="wan",
    ):
      torch.polar(absv, angle.to(torch.int32), out=out)

  def test_polygamma_negative_n(self):
    t = torch.tensor([1.0, 2.0], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(r"polygamma\(n, x\) does not support negative n\."),
        gpu="""polygamma(n, x) does not support negative n.""",
        message_reviewed_by="gunhyun",
    ):
      torch.polygamma(-1, t)

    out = torch.empty_like(t)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""polygamma(): expected n to be non-negative, got -1""",
        gpu="""polygamma(n, x) does not support negative n.""",
        message_reviewed_by="gunhyun",
    ):
      torch.polygamma(-1, t, out=out)

  def test_polygamma_complex(self):
    t = torch.tensor([1.0 + 1.0j], device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""polygamma(): expected the input dtype not to be complex, got complex64""",
        gpu=""""polygamma_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="gunhyun",
    ):
      torch.polygamma(2, t)

  def test_polygamma_invalid_out(self):
    t = torch.tensor([1.0, 2.0], device=et.device())
    out = torch.empty(2, device=et.device(), dtype=torch.uint8)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""polygamma(): expected the output dtype to be floating point or complex, got uint8""",
        gpu="""result type Float can't be cast to the desired output type Byte""",
        message_reviewed_by="gunhyun",
    ):
      torch.polygamma(1, t, out=out)

  def test_addmv_bool(self):
    t = torch.ones(5, device=et.device(), dtype=torch.bool)
    mat = torch.ones(5, 5, device=et.device(), dtype=torch.bool)
    vec = torch.ones(5, device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""addmv(): the dtype of the first argument cannot be bool""",
        gpu=""""addmv_impl_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.addmv(t.to(torch.bool), mat, vec)

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          tpu_dtype="int64",
          gpu_dtype="Long",
      ),
  )
  def test_addmv_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    t = torch.ones(5, device=et.device(), dtype=dtype)
    mat = torch.ones(5, 5, device=et.device(), dtype=dtype)
    vec = torch.ones(5, device=et.device(), dtype=dtype)

    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""addmv(): not implemented for {tpu_dtype}""",
        gpu=f""""addmv_impl_cuda" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.addmv(t, mat, vec)

  def test_addmv_not_a_matrix(self):
    inp = torch.ones(5, device=et.device())
    mat = torch.ones(5, 5, 5, device=et.device())
    vec = torch.ones(5, device=et.device())
    out = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""addmv(): expected the second argument to be a matrix (2D tensor), got 3D tensor""",
        gpu="""vector + matrix @ vector expected, got 1, 3, 1""",
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
        gpu="""vector + matrix @ vector expected, got 1, 2, 2""",
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
        gpu="""size mismatch, got input (5), mat (5x5), vec (4)""",
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
        gpu="""value cannot be converted to type float without overflow""",
        message_reviewed_by="wan",
    ):
      torch.addmv(*args, alpha=1j)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""addmv(): expected the dtype of beta to be neither complex nor bool, got complex128""",
        gpu="""value cannot be converted to type float without overflow""",
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
          gpu="""clamp is not supported for complex types""",
          message_reviewed_by="wan",
      ):
        torch.clamp(inp, min=minv, max=maxv, out=out)

  def test_clamp_unsupported_bool(self):
    t = torch.tensor([True, False], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""clamp(): self must not be bool""",
        gpu=""""clamp_scalar_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.clamp(t, min=False, max=True)

  def test_bmm_bool(self):
    a = torch.ones(1, 2, 3, dtype=torch.bool, device=et.device())
    b = torch.ones(1, 3, 2, dtype=torch.bool, device=et.device())
    out = torch.ones(1, 2, 2, dtype=torch.bool, device=et.device())

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""bmm(): not implemented for bool""",
        gpu=""""baddbmm_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""bmm(): not implemented for bool""",
        gpu=""""baddbmm_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          tpu_dtype="int64",
          gpu_dtype="Long",
      ),
  )
  def test_bmm_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    a = torch.ones(1, 2, 3, dtype=dtype, device=et.device())
    b = torch.ones(1, 3, 2, dtype=dtype, device=et.device())

    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""bmm(): not implemented for {tpu_dtype}""",
        gpu=f""""baddbmm_cuda" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.bmm(a, b)

  def test_bmm_output_bool(self):
    a = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 3, 2, dtype=torch.float32, device=et.device())
    out = torch.ones(1, 2, 2, dtype=torch.bool, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): expected out tensor to have dtype float32, got bool""",
        gpu="""Expected out tensor to have dtype float, but got bool instead""",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

  def test_bmm_mismatch_dtypes(self):
    a = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 3, 2, dtype=torch.half, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): expected self and mat2 to have the same dtype, got float32 vs float16""",
        gpu="""expected scalar type Float but found Half""",
    ):
      torch.bmm(a, b)

  def test_bmm_not_batch_of_matrices(self):
    a = torch.ones(1, 2, 3, 4, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 4, 2, dtype=torch.float32, device=et.device())
    out = torch.ones(1, 2, 2, dtype=torch.float32, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): expected the first argument to be a 3D tensor (batch of matrices), got 4D""",
        gpu="""batch1 must be a 3D tensor""",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""bmm(): expected the second argument to be a 3D tensor (batch of matrices), got 4D""",
        gpu="""batch2 must be a 3D tensor""",
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
        gpu="""Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [2, 3].""",
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
        gpu="""Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [1, 2].""",
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
        gpu="""Bias not supported yet""",
    ):
      torch._grouped_mm(a, b, offs=offs, bias=bias)

  def test_grouped_mm_invalid_self_dim(self):
    a = torch.randn(1, 2, 3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected self to be 2D or 3D, got 4D""",
        gpu="""mat_a has to be 2 or 3d""",
    ):
      torch._grouped_mm(a, b)

  def test_grouped_mm_invalid_mat2_dim(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(1, 2, 3, 4, device=et.device())
    offs = torch.tensor([1, 2, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected mat2 to be 2D or 3D, got 4D""",
        gpu="""mat_b has to be 2 or 3d""",
    ):
      torch._grouped_mm(a, b, offs=offs)

  def test_grouped_mm_missing_offs(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): offsets must be provided when at least one of self or mat2 is a 2D tensor""",
        gpu="""Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d""",
    ):
      torch._grouped_mm(a, b)

  def test_grouped_mm_unexpected_offs(self):
    a = torch.randn(2, 8, 8, device=et.device())
    b = torch.randn(2, 8, 8, device=et.device())
    offs = torch.tensor([1, 2], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): offsets must not be provided when both self and mat2 are 3D tensors""",
        gpu="""Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d""",
    ):
      torch._grouped_mm(a, b, offs=offs)

  def test_grouped_mm_mismatch_matrix_batch_sizes_case1(self):
    a = torch.randn(10, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    offs = torch.tensor([2, 5], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected offs batch size to match mat2 batch size, got 2 and 3""",
        gpu="""matrix batch sizes have to match""",
    ):
      torch._grouped_mm(a, b, offs=offs)

  def test_grouped_mm_mismatch_matrix_batch_sizes_case2(self):
    a = torch.randn(3, 4, 8, device=et.device())
    b = torch.randn(5, 8, device=et.device())
    offs = torch.tensor([2, 5], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected offs batch size to match self batch size, got 2 and 3""",
        gpu="""contraction dimension of mat_a and mat_b must match""",
    ):
      torch._grouped_mm(a, b, offs=offs)

  def test_grouped_mm_mismatch_batched_dimension_case4(self):
    a = torch.randn(3, 4, 8, device=et.device())
    b = torch.randn(2, 8, 8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grouped_mm(): expected self batch size to match mat2 batch size, got 3 and 2""",
        gpu="""batched dimension has to match""",
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
        gpu="""Expected out tensor to have dtype float, but got bool instead""",
        message_reviewed_by="wan",
    ):
      torch.baddbmm(input_tensor, batch1, batch2, out=out.to(torch.bool))

  @parameterized.named_parameters(
      dict(
          testcase_name="bool",
          dtype=torch.bool,
          tpu_dtype="bool",
          gpu_dtype="Bool",
      ),
      dict(
          testcase_name="uint8",
          dtype=torch.uint8,
          tpu_dtype="uint8",
          gpu_dtype="Byte",
      ),
      dict(
          testcase_name="int8",
          dtype=torch.int8,
          tpu_dtype="int8",
          gpu_dtype="Char",
      ),
      dict(
          testcase_name="int16",
          dtype=torch.int16,
          tpu_dtype="int16",
          gpu_dtype="Short",
      ),
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          tpu_dtype="int64",
          gpu_dtype="Long",
      ),
  )
  def test_baddbmm_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    input_tensor = torch.ones(1, 2, 2, device=et.device(), dtype=dtype)
    batch1 = torch.ones(1, 2, 3, device=et.device(), dtype=dtype)
    batch2 = torch.ones(1, 3, 2, device=et.device(), dtype=dtype)

    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""baddbmm(): not implemented for {tpu_dtype}""",
        gpu=f""""baddbmm_cuda" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.baddbmm(input_tensor, batch1, batch2)

  def test_baddbmm_invalid_out_dtype(self):
    input_tensor = torch.ones(1, 2, 2, dtype=torch.float32, device=et.device())
    batch1 = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    batch2 = torch.ones(1, 3, 2, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""baddbmm(): expected out_dtype to be the same as input dtype or float32 for float16/bfloat16 inputs, got int32""",
        gpu="""out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs""",
    ):
      torch.ops.aten.baddbmm.dtype(input_tensor, batch1, batch2, torch.int32)

  def test_baddbmm_mismatch_dtypes_batch(self):
    input_tensor = torch.ones(1, 2, 2, dtype=torch.float32, device=et.device())
    batch1 = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    batch2 = torch.ones(1, 3, 2, dtype=torch.half, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""baddbmm(): expected input dtypes to be the same, got: self=float32, batch1=float32, batch2=float16""",
        gpu="""expected scalar type Float but found Half""",
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
        gpu="""batch1 must be a 3D tensor""",
        message_reviewed_by="wan",
    ):
      torch.baddbmm(input_tensor, batch1, batch2, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""baddbmm(): expected batch2 to be a 3D tensor (batch of matrices), got 4D""",
        gpu="""batch2 must be a 3D tensor""",
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
        gpu="""Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [2, 3].""",
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
        gpu="""Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [1, 4].""",
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
        gpu="""Input type (CUDABoolType) and weight type (torch.cuda.FloatTensor) should be the same""",
        message_reviewed_by="wan",
    ):
      convolution(inp.to(torch.bool), w)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{tpu_fn}(): expected the dtype of the weight tensor to be neither long nor bool, got bool""",
        gpu="""Input type (torch.cuda.FloatTensor) and weight type (CUDABoolType) should be the same""",
        message_reviewed_by="wan",
    ):
      convolution(inp, w.to(torch.bool))

  def test_conv2d_int32_unsupported(self):
    # Use a depthwise configuration so that CUDA reaches the
    # "conv_depthwise2d_forward_cuda" dispatch (a NotImplementedError) instead
    # of the cuDNN engine lookup (a RuntimeError).
    t_int = torch.ones((1, 2, 5, 5), device=et.device(), dtype=torch.int32)
    weight = torch.ones((2, 1, 3, 3), device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""convolution(): not implemented for int32""",
        gpu=""""conv_depthwise2d_forward_cuda" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.conv2d(t_int, weight, groups=2)

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
        gpu="""Expected 3-dimensional input for 3-dimensional weight [2, 3, 3], but got 2-dimensional input of size [10, 10] instead""",
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
          gpu=f"""expected {arg_name} to be a single integer value or a list of 2 values to match the convolution dimensions, but got {arg_name}=[1, 1, 1]""",
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
        gpu="""Expected 5-dimensional input for 5-dimensional weight [1, 3, 3, 3, 3], but got 4-dimensional input of size [2, 3, 10, 10] instead""",
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
        gpu="""Given groups=3, expected weight to be at least 3 at dimension 0, but got weight of size [1, 3, 3, 3] instead""",
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
        gpu="""Expected 5-dimensional input for 5-dimensional weight [1, 3, 3, 3, 3], but got 4-dimensional input of size [2, 3, 10, 10] instead""",
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
        gpu="""Given groups=3, expected weight to be at least 3 at dimension 0, but got weight of size [1, 3, 3, 3] instead""",
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
        gpu="""Given weight of size [1, 3, 3, 3], expected bias to be 1-dimensional with 1 elements, but got bias of size [1, 1] instead""",
        message_reviewed_by="wan",
    ):
      _run_convolution(inp, w, bias=torch.ones(1, 1, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""convolution(): expected the bias tensor to have 1 dimension of shape [1 (out channels)], got shape [5]""",
        gpu="""Given weight of size [1, 3, 3, 3], expected bias to be 1-dimensional with 1 elements, but got bias of size [5] instead""",
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
        gpu="""Expected tensor for argument #1 'grad_output' to have the same type as tensor for argument #2 'weight'; but type CUDABoolType does not equal torch.cuda.FloatTensor (while checking arguments for cudnn_convolution_backward_input)""",
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
        tpu=f"""{op_name}(): the dim argument must be specified when the input tensor has 0 elements""",
        gpu=f"""{op_name}(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.""",
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
        gpu="""Expected out tensor to have dtype long, but got float instead""",
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

    def test_with(dtype: torch.dtype, tpu: str, gpu: str):
      """Tests the `op` with the input tensor of the given `dtype`.

      Tests that running `argmin` (`argmax`) with the given `dtype` will result
      in the expected error.

      Args:
        dtype: The dtype of the op input tensor.
        tpu: String representation for `dtype` to be used in the error message
          of the TPU kernel.
        gpu: String representation for `dtype` to be used in the error message
          of the GPU kernel.
      """

      inp = torch.ones(2, 2, device=et.device(), dtype=dtype)

      err_type = RuntimeError
      if et.is_on_gpu() and dtype == torch.complex64:
        err_type = TypeError

      with et.assert_raises_message(
          err_type,
          tpu=f"""{op_name}(): expected the input dtype to be neither complex nor"""
          f""" bool, got {tpu}""",
          gpu=f"""{op_name}(): does not support {gpu} input""",
          message_reviewed_by="wan",
      ):
        op(inp, out=out)

    with self.subTest(dtype=torch.bool):
      test_with(torch.bool, tpu="""bool""", gpu="""bool""")
    with self.subTest(dtype=torch.complex64):
      test_with(torch.complex64, tpu="""complex64""", gpu="""complex""")

  def test_mm_output_dtype_mismatch(self):
    lhs = torch.ones(3, 4, device=et.device(), dtype=torch.float32)
    rhs = torch.ones(4, 5, device=et.device())
    out = torch.ones(3, 5, device=et.device(), dtype=torch.float64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected the output to have the same dtype as inputs, got out dtype float64 vs inputs dtype float32""",
        gpu="""Expected out tensor to have dtype float, but got double instead""",
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
        gpu="""expected mat1 and mat2 to have the same dtype, but got: float != double""",
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
        gpu="""mm: self must be a matrix, got 3-D tensor""",
        message_reviewed_by="wan",
    ):
      torch.mm(not_a_matrix_tensor, matrix_tensor, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected the second argument to be a 2D tensor (matrix), got 3D of shape [3, 4, 5]""",
        gpu="""mm: mat2 must be a matrix, got 3-D tensor""",
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
        gpu="""mm: mat1 and mat2 shapes cannot be multiplied (3x4 and 5x6)""",
        message_reviewed_by="wan",
    ):
      torch.mm(lhs, rhs, out=out)

  def test_linalg_lu_factor_ex_no_pivoting(self):
    if et.is_on_gpu():
      # On TPU, non-pivoting LU decomposition (pivot=False) raises
      # RuntimeError ("non-pivoting decomposition is not supported").
      # On GPU, cuSOLVER implements non-pivoting LU without error.
      self.skipTest("GPU behavior difference")
    a = torch.ones(1, 2, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_factor_ex(): non-pivoting decomposition is not supported""",
        gpu="""linalg.lu_factor: LU without pivoting is not implemented on the CPU""",
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
        gpu="""torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: [4] instead""",
    ):
      torch.linalg.lu_factor_ex(a, out=out)

  def test_lu_unpack_data_rank_too_low(self):
    data = torch.ones(4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)

    # Call the out overload of linalg.lu_unpack() op.
    out = _make_lu_unpack_outputs(p=(4,), l=(4,), u=(4,))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""lu_unpack(): expected lu_data to have at least 2 dimensions, got 1""",
        gpu="""torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: [4] instead""",
    ):
      torch.lu_unpack(data, pivots, out=out)

  def test_lu_factor_ex_unsupported_dtypes(self):
    t = torch.ones(2, 2, dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_factor_ex(): expected the input dtype to be float32, float64, complex64, or complex128, got int32""",
        gpu=""""lu_factor_cusolver" not implemented for 'Int'""",
        message_reviewed_by="chizz",
    ):
      torch.linalg.lu_factor(t)

  def test_lu_solve_rank_too_low(self):
    lu = torch.ones(4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): expected lu to have at least 2 dimensions, got 1""",
        gpu="""torch.linalg.lu_solve: The input tensor A must have at least 2 dimensions.""",
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
        tpu="""linalg_lu_solve(): expected lu to be square, got 4 and 2""",
        gpu="""torch.linalg.lu_solve: A must be batches of square matrices, but they are 4 by 2 matrices""",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_last_dimension_mismatch(self):
    lu = torch.ones(4, 4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 4, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): expected b second-to-last dimension to match the second-to-last dimension of lu (4) when left=True, got 3""",
        gpu="""linalg.lu_solve: Incompatible shapes of A and B for the equation AX = B (4x4 and 3x4)""",
        message_reviewed_by="wan",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_second_to_last_dimension_mismatch(self):
    lu = torch.ones(4, 4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, 3, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): expected b last dimension to match the last dimension of lu (4) when left=False, got 3""",
        gpu="""linalg.lu_solve: Incompatible shapes of A and B for the equation XA = B (4x4 and 4x3)""",
        message_reviewed_by="wan",
    ):
      torch.linalg.lu_solve(lu, pivots, b, left=False, out=out)

  def test_lu_solve_pivots_invalid_dimensions(self):
    lu = torch.ones(3, 3, device=et.device())
    pivots = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 3, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): expected pivots to have one less dimension than the tensor, got 2 and 2""",
        gpu="""linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, but got pivots with shape [2, 3] instead""",
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
        tpu="""linalg_lu_solve(): expected pivots and tensor to have the same batch dimensions, got [3] and [2]""",
        gpu="""linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, but got pivots with shape [2, 3] instead""",
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
        tpu="""linalg_lu_solve(): expected pivots size to be <= 3 (the size of the matrix), got 4 and 3""",
        gpu="""linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix""",
        message_reviewed_by="wan",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_multinomial_int(self):
    inp = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""multinomial(): expected the input dtype to be floating-point, got int32""",
        gpu="""multinomial only supports floating-point dtypes for input, got: Int""",
        message_reviewed_by="wan",
    ):
      torch.multinomial(inp, num_samples=2)

  def test_multinomial_invalid_dimension(self):
    inp = torch.randn(2, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""multinomial(): expected the input to have either 1 or 2 dimensions, got 3 of shape [2, 2, 2]""",
        gpu="""prob_dist must be 1 or 2 dim""",
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
          gpu="""cannot sample n_sample <= 0 samples""",
          message_reviewed_by="wan",
      ):
        torch.multinomial(inp, num_samples=num_samples)

    # Make sure we check the number of samples when `replacement` is set to
    # `False`.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""multinomial(): expected the number of samples to be <= 2 (population size) when replacement is disabled, got 3""",
        gpu="""cannot sample n_sample > prob_dist.size(-1) samples without replacement""",
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
        gpu="""too many indices for tensor of dimension 2 (got 3)""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.index.Tensor_out(t, indices, out=out)

  def test_index_no_indices(self):
    if et.is_on_gpu():
      # On TPU, passing [None] indices raises a clean RuntimeError
      # ("at least one index tensor must be defined"). On GPU, empty
      # indices trigger an internal C++ assertion in OffsetCalculator.cuh
      # instead of a standard Python error.
      self.skipTest("GPU behavior difference")
    t = torch.ones(2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""index(): expected at least one index tensor to be defined, got 0""",
        gpu=re.compile(
            r"""N <= iter\.ntensors\(\) INTERNAL ASSERT FAILED at.*OffsetCalculator\.cuh.*please report a bug to PyTorch\.\s*"""
        ),
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
        gpu="""1D tensors expected, but got 2D and 1D tensors""",
        message_reviewed_by="wan",
    ):
      op(lhs.unsqueeze(0), rhs)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected the second argument to be a 1D tensor, got 2D of shape [1, 2]""",
        gpu="""1D tensors expected, but got 1D and 2D tensors""",
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
        gpu=""""dot" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      op(lhs, rhs)

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          tpu_dtype="int64",
          gpu_dtype="Long",
      ),
  )
  def test_dot_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    lhs = torch.ones(2, device=et.device(), dtype=dtype)
    rhs = torch.ones(2, device=et.device(), dtype=dtype)

    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""dot(): not implemented for {tpu_dtype}""",
        gpu=f""""dot" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.dot(lhs, rhs)

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          tpu_dtype="int64",
          gpu_dtype="Long",
      ),
  )
  def test_vdot_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    lhs = torch.ones(2, device=et.device(), dtype=dtype)
    rhs = torch.ones(2, device=et.device(), dtype=dtype)

    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""vdot(): not implemented for {tpu_dtype}""",
        gpu=f""""dot" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="wan",
    ):
      torch.vdot(lhs, rhs)

  def test_vdot_size_mismatch(self):
    lhs = torch.ones(2, device=et.device())
    rhs = torch.ones(3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""vdot(): expected inputs to have the same shape, got [2] vs [3]""",
        gpu="""inconsistent tensor size, expected tensor [2] and src [3] to have the same number of elements, but got 2 and 3 elements respectively""",
        message_reviewed_by="wan",
    ):
      torch.vdot(lhs, rhs)

  def test_embedding_bag_invalid_dtypes(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""embedding_bag_forward_only(): expected weight dtype to be float16, bfloat16, float32, or float64, got int64""",
        gpu=""""embedding_bag_cuda" not implemented for 'Long'""",
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
        gpu=""""LayerNormKernelImpl" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm(inp, normalized_shape)

  def test_native_layer_norm_backward_int(self):
    inp = torch.ones(5, 5, device=et.device(), dtype=torch.int32)
    normalized_shape = (5,)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_layer_norm_backward(): expected the input dtype to be floating point, got int32""",
        gpu=""""LayerNormBackwardKernelImpl" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm_backward(inp, normalized_shape)

  def test_native_layer_norm_normalized_shape_empty(self):
    inp = torch.ones(5, 5, device=et.device())
    normalized_shape = []

    with et.assert_raises_message(
        RuntimeError,
        tpu="""layer_norm(): the normalized shape must have >= 1 dimensions""",
        gpu="""Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one element, but got normalized_shape = []""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm(inp, normalized_shape)

  def test_native_layer_norm_backward_normalized_shape_empty(self):
    inp = torch.ones(5, 5, device=et.device())
    normalized_shape = []

    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_layer_norm_backward(): the normalized shape must have >= 1 dimensions""",
        gpu="""Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one element, but got normalized_shape = []""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm_backward(inp, normalized_shape)

  def test_native_layer_norm_normalized_shape_too_large(self):
    inp = torch.ones(5, 5, device=et.device())
    normalized_shape = (5, 3, 3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""layer_norm(): expected the normalized shape to have <= 2 dimensions, got 3 dimensions of shape [5, 3, 3]""",
        gpu="""Given normalized_shape=[5, 3, 3], expected input with shape [*, 5, 3, 3], but got input of size[5, 5]""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm(inp, normalized_shape)

  def test_native_layer_norm_backward_normalized_shape_too_large(self):
    inp = torch.ones(5, 5, device=et.device())
    normalized_shape = (5, 3, 3)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_layer_norm_backward(): expected the normalized shape to have <= 2 dimensions, got 3 dimensions of shape [5, 3, 3]""",
        gpu="""Given normalized_shape=[5, 3, 3], expected input with shape [*, 5, 3, 3], but got input of size[5, 5]""",
        message_reviewed_by="wan",
    ):
      _run_native_layer_norm_backward(inp, normalized_shape)

  def test_random_invalid_range(self):
    t = torch.ones(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""random_(): expected 'from' to be < 'to', got 20 vs 10""",
        gpu="""random_ expects 'from' to be less than 'to', but got from=20 >= to=10""",
        message_reviewed_by="wan",
    ):
      t.random_(20, 10)

  def test_randn_unsupported_dtype(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""normal_(): expected the self tensor to be floating point or complex type, got int32""",
        gpu=""""normal_kernel_cuda" not implemented for 'Int'""",
    ):
      torch.randn(5, dtype=torch.int32, device=et.device())

  def test_uniform_unsupported_dtype(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""uniform_(): expected the input dtype to be floating point or complex, got int32""",
        gpu=""""check_uniform_bounds" not implemented for 'Int'""",
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
        gpu="""linalg.inv: The input tensor A must have at least 2 dimensions.""",
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
        gpu="""linalg.inv: A must be batches of square matrices, but they are 3 by 5 matrices""",
        message_reviewed_by="wan",
    ):
      torch.linalg.inv_ex(a, out=out)

  def test_linalg_vector_norm_dtype_mismatch(self):
    x = torch.ones(5, device=et.device())
    out = torch.empty(5, dtype=torch.float64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_vector_norm(): expected the output dtype to be float32, got float64""",
        gpu="""Expected out tensor to have dtype float, but got double instead""",
        message_reviewed_by="gunhyun",
    ):
      torch.linalg.vector_norm(x, dtype=torch.float32, out=out)

  def test_norm_out_dtype_mismatch(self):
    x = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = torch.empty(5, dtype=torch.float64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""norm(): expected the output dtype to be float32, got float64""",
        gpu="""Expected out tensor to have dtype float, but got double instead""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.norm.out(x, 2.0, [0], False, out=out)

  def test_norm_dtype_out_dtype_mismatch(self):
    x = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = torch.empty(5, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""norm(): expected the output dtype to be float64, got float32""",
        gpu="""Expected out tensor to have dtype double, but got float instead""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.norm.dtype_out(
          x, 2.0, [0], False, dtype=torch.float64, out=out
      )

  def test_norm_dim_out_of_bounds(self):
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    for dim in [-3, 3]:
      msg = (
          "Dimension out of range (expected to be in range of [-2, 1], but got"
          f" {dim})"
      )
      with et.assert_raises_message(
          IndexError,
          tpu=msg,
          gpu=msg,
      ):
        torch.norm(t, p=2, dim=dim)

  def test_norm_dim_repeated(self):
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    msg = "dim 0 appears multiple times in the list of dims"
    with et.assert_raises_message(
        RuntimeError,
        tpu=msg,
        gpu=msg,
    ):
      torch.norm(t, p=2, dim=[0, 0])

  def test_rms_norm_int(self):
    inp = torch.ones(5, 5, device=et.device(), dtype=torch.int32)
    normalized_shape = (5,)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_rms_norm(): expected the input dtype to be floating point, got int32""",
        gpu=""""LayerNormKernelImpl" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.rms_norm(inp, normalized_shape)

  def test_rms_norm_bool(self):
    inp = torch.ones(5, 5, device=et.device(), dtype=torch.bool)
    normalized_shape = (5,)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_rms_norm(): expected the input dtype to be floating point, got bool""",
        gpu=""""LayerNormKernelImpl" not implemented for 'Bool'""",
        message_reviewed_by="chizz",
    ):
      torch.nn.functional.rms_norm(inp, normalized_shape)

  def test_hardswish_unsupported_dtype(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardswish(): expected the input dtype to be floating point, got int32""",
        gpu=""""hardswish_cuda" not implemented for 'Int'""",
    ):
      torch.nn.functional.hardswish(t)

  def test_hardswish_out_unsupported_dtype(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    out = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardswish(): expected the input dtype to be floating point, got int32""",
        gpu=""""hardswish_cuda" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.hardswish.out(t, out=out)

  def test_hardswish_inplace_unsupported_dtype(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardswish_(): expected the input dtype to be floating point, got int32""",
        gpu=""""hardswish_cuda" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.hardswish(t, inplace=True)

  def test_hardswish_backward_unsupported_dtype(self):
    grad = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    self_val = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardswish_backward(): expected the input dtype to be floating point, got int32""",
        gpu=""""hardswish_backward_cuda" not implemented for 'Int'""",
    ):
      torch.ops.aten.hardswish_backward(grad, self_val)

  def test_hardsigmoid_unsupported_dtype(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardsigmoid(): expected the input dtype to be floating point, got int32""",
        gpu=""""hardsigmoid_cuda" not implemented for 'Int'""",
    ):
      torch.nn.functional.hardsigmoid(t)

  def test_hardsigmoid_backward_unsupported_dtype(self):
    grad = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    self_val = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardsigmoid_backward(): expected the input dtype to be floating point, got int32""",
        gpu=""""hardsigmoid_backward_cuda" not implemented for 'Int'""",
    ):
      torch.ops.aten.hardsigmoid_backward(grad, self_val)

  def test_bincount_rank_too_high(self):
    t = torch.ones(2, 2, 2, device=et.device(), dtype=torch.int32)

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""bincount(): materialization failed with: Unexpected dimension of input tensor: [2, 2, 2]""",
        gpu="""bincount only supports 1-d non-negative integral inputs.""",
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
        gpu="""Index tensor must have the same number of dimensions as input tensor""",
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
        gpu="""Index tensor must have the same number of dimensions as input tensor""",
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
        gpu="""Index tensor must have the same number of dimensions as input tensor""",
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
        gpu="""Size does not match at dimension 1 expected index [2, 4] to be no larger than self [2, 3] apart from dimension 0""",
    ):
      torch.gather(inp, dim, index).cpu()

  def test_lerp_int(self):
    t = torch.tensor([1, 2], device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""lerp(): expected the first argument's dtype to be non-integral, got int32""",
        gpu=""""lerp_cuda" not implemented for 'Int'""",
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
        tpu="""mse_loss(): expected the input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got uint8""",
        gpu=""""mse_cuda" not implemented for 'Byte'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(uint8, uint8, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): expected the input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got int8""",
        gpu=""""mse_cuda" not implemented for 'Char'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(int8, int8, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): expected the input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got int16""",
        gpu=""""mse_cuda" not implemented for 'Short'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(int16, int16, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): expected the input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got int32""",
        gpu=""""mse_cuda" not implemented for 'Int'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(int32, int32, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): expected the input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got int64""",
        gpu=""""mse_cuda" not implemented for 'Long'""",
        message_reviewed_by="yilingyuan",
    ):
      torch.nn.functional.mse_loss(int64, int64, reduction="sum")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): expected the input dtype to be none of (uint8, int8, int16, int32, int64, complex64), got complex64""",
        gpu=""""mse_cuda" not implemented for 'ComplexFloat'""",
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
        gpu=""""embedding_renorm_cuda_" not implemented for 'Long'""",
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
        gpu=""""grid_sampler_2d_cuda" not implemented for 'Int'""",
    ):
      torch.ops.aten.grid_sampler_2d(t, g, 0, 0, False)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_2d(): expected the input dtype to be floating point, got complex64""",
        gpu=""""grid_sampler_2d_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.grid_sampler(inp2d, grid2d, 0, 0, False)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_3d(): expected the input dtype to be floating point, got complex64""",
        gpu=""""grid_sampler_3d_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.grid_sampler(inp3d, grid3d, 0, 0, False)

  def test_native_dropout_invalid_p(self):
    if et.is_on_gpu():
      self.skipTest("GPU behavior difference")
    inp = torch.ones(5, 3, device=et.device())

    def check(p: float) -> None:
      with et.assert_raises_message(
          RuntimeError,
          tpu="""dropout(): expected p to be in the range [0, 1], got"""
          f""" {p}""",
          gpu=f"""bernoulli_ expects p to be in [0, 1], but got p={1 - p}""",
          message_reviewed_by="wan",
      ):
        torch.native_dropout(inp, p=p, train=True)

    for p in (-1.5, 1.5):
      with self.subTest(p=p):
        check(p)

  def test_fused_dropout_invalid_p(self):
    if et.is_on_gpu():
      self.skipTest("GPU behavior difference")
    x = torch.ones(5, 3, device=et.device())

    def check(p: float) -> None:
      with et.assert_raises_message(
          RuntimeError,
          tpu=f"fused_dropout(): expected p to be in the range [0, 1], got {p}",
          message_reviewed_by="adivinpatel",
      ):
        torch.ops.aten._fused_dropout(x, p=p)

    for p in (-1.5, -0.0001, 1.0001, 1.5):
      with self.subTest(p=p):
        check(p)

  def test_fused_dropout_invalid_dtype(self):
    x = torch.ones((2, 3), device=et.device(), dtype=torch.int32)
    err_type = RuntimeError if et.is_on_tpu() else NotImplementedError
    with et.assert_raises_message(
        err_type,
        tpu="""fused_dropout(): expected input to be floating point or complex, got Int""",
        gpu=""""fused_dropout" not implemented for 'Int'""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_dropout(x, 0.5)

  def test_native_dropout_backward_invalid_mask_dtype(self):
    grad_output = torch.ones((2, 3), device=et.device(), dtype=torch.float32)
    mask = torch.ones((2, 3), device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_dropout_backward(): expected mask to be Bool scalar type, got Int""",
        gpu="""Mask should be Bool Scalar TypeInt""",
    ):
      torch.ops.aten.native_dropout_backward(grad_output, mask, 2.0)

  @parameterized.named_parameters(
      dict(
          testcase_name="bool",
          dtype=torch.bool,
          tpu_dtype="bool",
          gpu_dtype="Bool",
      ),
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          tpu_dtype="int64",
          gpu_dtype="Long",
      ),
  )
  def test_native_dropout_backward_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    grad_output = torch.ones((2, 3), device=et.device(), dtype=dtype)
    mask = torch.ones((2, 3), device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""native_dropout_backward(): not implemented for {tpu_dtype}""",
        gpu=f""""masked_scale" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.native_dropout_backward(grad_output, mask, 2.0)

  def test_weight_norm_interface_dim(self):
    if et.is_on_gpu():
      self.skipTest("GPU behavior difference")
    v = torch.ones(2, 3, 4, 5, device=et.device(), dtype=torch.float32)
    g = torch.ones(3, device=et.device(), dtype=torch.float32)
    dim = 1
    with et.assert_raises_message(
        RuntimeError,
        tpu="""weight_norm_interface(): expected dim to be 0 or the last dimension of v, got 1""",
        gpu=re.compile(
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
        tpu="""weight_norm_interface(): expected the input dtype to be floating point, got int32""",
        gpu=""""weight_norm_fwd_first_dim_kernel" not implemented for 'Int'""",
    ):
      torch.ops.aten._weight_norm_interface(v, g, dim)

  def test_weight_norm_interface_v_dim_error(self):
    """Tests error message for weight_norm when v.dim() == 0."""
    v = torch.randn((), device=et.device(), dtype=torch.float32)
    g = torch.randn((), device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        tpu="""weight_norm_interface(): expected v to have at least 1 dimension, got 0""",
        gpu="""Dimension specified as 0 but tensor has no dimensions""",
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
        gpu=""""softplus_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got uint8""",
        gpu=""""softplus_cuda" not implemented for 'Byte'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_uint8)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got int8""",
        gpu=""""softplus_cuda" not implemented for 'Char'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_int8)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got int16""",
        gpu=""""softplus_cuda" not implemented for 'Short'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_int16)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got int32""",
        gpu=""""softplus_cuda" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got int64""",
        gpu=""""softplus_cuda" not implemented for 'Long'""",
        message_reviewed_by="wan",
    ):
      torch.nn.functional.softplus(t_int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus(): expected the input dtype to be floating-point, got complex64""",
        gpu=""""softplus_cuda" not implemented for 'ComplexFloat'""",
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
        gpu=""""softplus_backward_cuda" not implemented for 'Bool'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.softplus_backward(t_bool, t_bool, 1.0, 20.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus_backward(): expected the input dtype to be floating-point, got int32""",
        gpu=""""softplus_backward_cuda" not implemented for 'Int'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.softplus_backward(t_int32, t_int32, 1.0, 20.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""softplus_backward(): expected the input dtype to be floating-point, got complex64""",
        gpu=""""softplus_backward_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.softplus_backward(t_complex64, t_complex64, 1.0, 20.0)

  def test_softplus_backward_mismatched_dtypes(self):
    if et.is_on_gpu():
      self.skipTest("GPU behavior difference")

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
        gpu="""clamp is not supported for complex types""",
    ):
      torch.nn.functional.hardtanh(t)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh_backward(): expected the input dtype to be floating point, got complex64""",
        gpu=""""hardtanh_backward_cuda" not implemented for 'ComplexFloat'""",
    ):
      torch.ops.aten.hardtanh_backward(t, t, min_val=0, max_val=1)

  def test_hardtanh_unsupported_bool_dtype(self):
    t = torch.tensor([True, False], device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh(): expected the input dtype to be non-boolean, got bool""",
        gpu="""Bool inputs not supported for hardtanh""",
    ):
      torch.nn.functional.hardtanh(t)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh_backward(): expected the input dtype to be floating point, got bool""",
        gpu=""""hardtanh_backward_cuda" not implemented for 'Bool'""",
    ):
      torch.ops.aten.hardtanh_backward(t, t, min_val=0, max_val=1)

  def test_hardtanh_backward_unsupported_integral_dtypes(self):
    # hardtanh_backward does not support integral types.
    t_uint8 = torch.ones(2, device=et.device(), dtype=torch.uint8)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh_backward(): expected the input dtype to be floating point, got uint8""",
        gpu=""""hardtanh_backward_cuda" not implemented for 'Byte'""",
    ):
      torch.ops.aten.hardtanh_backward(t_uint8, t_uint8, min_val=0, max_val=1)

    t_int32 = torch.ones(2, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh_backward(): expected the input dtype to be floating point, got int32""",
        gpu=""""hardtanh_backward_cuda" not implemented for 'Int'""",
    ):
      torch.ops.aten.hardtanh_backward(t_int32, t_int32, min_val=0, max_val=1)

  def test_hardtanh_unsupported_out_of_range_min(self):
    t = torch.ones(2, device=et.device(), dtype=torch.uint8)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh(): expected clamp min value to be representable as uint8, got 256""",
        gpu="""Clamp min value 256 is outside the representable range of Byte""",
    ):
      torch.nn.functional.hardtanh(t, min_val=256, max_val=256)

  def test_hardtanh_unsupported_out_of_range_max(self):
    t = torch.ones(2, device=et.device(), dtype=torch.uint8)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""hardtanh(): expected clamp max value to be representable as uint8, got -1""",
        gpu="""Clamp max value -1 is outside the representable range of Byte""",
    ):
      torch.nn.functional.hardtanh(t, min_val=-2, max_val=-1)

  def test_prelu_kernel_unsupported_self_dtype(self):
    self_tensor = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    weight = torch.ones(3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prelu_kernel(): expected the self dtype to be floating point, got int32""",
        gpu="""Found dtype Float but expected Int""",
    ):
      torch.ops.aten._prelu_kernel(self_tensor, weight)

  def test_prelu_kernel_unsupported_weight_dtype(self):
    self_tensor = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    weight = torch.ones(3, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prelu_kernel(): expected the weight dtype to be floating point, got int32""",
        gpu="""Found dtype Int but expected Float""",
    ):
      torch.ops.aten._prelu_kernel(self_tensor, weight)

  def test_prelu_kernel_unbroadcastable_shapes(self):
    self_tensor = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    weight = torch.ones(4, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prelu_kernel(): the size of tensor a (2) must match the size of tensor b (4) at non-singleton dimension 0""",
        gpu="""The size of tensor a (2) must match the size of tensor b (4) at non-singleton dimension 0""",
    ):
      torch.ops.aten._prelu_kernel(self_tensor, weight)

  def test_prelu_kernel_backward_unsupported_dtype(self):
    grad_output = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    self_tensor = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    weight = torch.ones(3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prelu_kernel_backward(): expected the grad_output dtype to be floating point, got int32""",
        gpu="""Found dtype Int but expected Float""",
    ):
      torch.ops.aten._prelu_kernel_backward(grad_output, self_tensor, weight)

  def test_prelu_kernel_backward_grad_output_shape_mismatch(self):
    grad_output = torch.ones(2, 4, device=et.device(), dtype=torch.float32)
    self_tensor = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    weight = torch.ones(3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prelu_kernel_backward(): expected grad_output shape to match self shape [2, 3], got [2, 4]""",
        gpu="""The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1""",
    ):
      torch.ops.aten._prelu_kernel_backward(grad_output, self_tensor, weight)

  def test_leaky_relu_unsupported_bool_dtype(self):
    inp = torch.tensor([True, False], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""leaky_relu(): expected non-bool dtype, got Bool""",
        gpu=""""leaky_relu_cuda" not implemented for 'Bool'""",
    ):
      torch.nn.functional.leaky_relu(inp)

  def test_leaky_relu_unsupported_int_dtype(self):
    inp = torch.tensor([1, 2], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""leaky_relu(): expected non-integer dtype, got Long""",
        gpu=""""leaky_relu_cuda" not implemented for 'Long'""",
    ):
      torch.nn.functional.leaky_relu(inp)

  def test_leaky_relu_unsupported_complex_dtype(self):
    inp = torch.tensor(
        [1 + 1j, 2 + 2j], device=et.device(), dtype=torch.complex64
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu="""leaky_relu(): expected non-complex dtype, got ComplexFloat""",
        gpu=""""leaky_relu_cuda" not implemented for 'ComplexFloat'""",
    ):
      torch.nn.functional.leaky_relu(inp)

  def test_masked_fill_multi_element_value(self):
    inp = torch.ones(2, 2, device=et.device())
    mask = torch.ones(2, 2, dtype=torch.bool, device=et.device())
    value = torch.ones(2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_fill_(): expected value to be a 0D tensor, got 1D tensor of shape [2]""",
        gpu="""masked_fill_ only supports a 0-dimensional value tensor, but got tensor with 1 dimension(s).""",
    ):
      torch.masked_fill(inp, mask, value)

  def test_masked_fill_value_overflow(self):
    inp = torch.ones(2, 2, dtype=torch.uint8, device=et.device())
    mask = torch.ones(2, 2, dtype=torch.bool, device=et.device())
    value = torch.tensor(2**31, dtype=torch.int64, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_fill_(): expected value to be representable as uint8, got 2147483648""",
        gpu="""value cannot be converted to type uint8_t without overflow""",
    ):
      torch.masked_fill(inp, mask, value)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_fill_(): expected value to be representable as uint8, got 2147483648""",
        gpu="""value cannot be converted to type uint8_t without overflow""",
    ):
      torch.masked_fill(inp, mask, 2**31)

  def test_masked_fill_inplace_invalid_broadcast(self):
    inp = torch.ones(2, dtype=torch.float32, device=et.device())
    mask = torch.ones(2, 2, dtype=torch.bool, device=et.device())
    value = torch.tensor(1, dtype=torch.float32, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_fill_(): expected the output shape (broadcast of self and mask) to match [2] (shape of self), got [2, 2]""",
        gpu="""expand(CUDABoolType{[2, 2]}, size=[2]): the number of sizes provided (1) must be greater or equal to the number of dimensions in the tensor (2)""",
    ):
      inp.masked_fill_(mask, value)

  def test_normal_empty_std_tensor(self):
    device = et.device()
    out_empty = torch.empty(0, device=device)
    torch.normal(mean=0.0, std=torch.tensor([], device=device), out=out_empty)

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          gpu_msg=""""normal_kernel_cuda" not implemented for 'Int'""",
          tpu_msg=(
              "normal_(): expected the self tensor to be floating point or"
              " complex type, got int32"
          ),
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          gpu_msg=""""normal_kernel_cuda" not implemented for 'Long'""",
          tpu_msg=(
              "normal_(): expected the self tensor to be floating point or"
              " complex type, got int64"
          ),
      ),
  )
  def test_normal_errors_invalid_input_dtype(
      self, dtype: torch.dtype, *, gpu_msg: str, tpu_msg: str
  ):
    device = et.device()
    with et.assert_raises_message(
        RuntimeError,
        gpu=gpu_msg,
        tpu=tpu_msg,
    ):
      torch.tensor([1, 2], device=device, dtype=dtype).normal_()

  def test_normal_errors_negative_std_scalar(self):
    device = et.device()
    with et.assert_raises_message(
        RuntimeError,
        gpu="""normal expects std >= 0.0, but found std -1""",
        tpu="""normal_(): expected std >= 0.0, got -1""",
    ):
      torch.empty(2, device=device).normal_(mean=0.0, std=-1.0)

  def test_normal_errors_negative_std_tensor(self):
    self.synchronize_tensors_in_tear_down = False  # CUDA device-side assert triggers an unrecoverable error state on GPU, preventing synchronization.
    device = et.device()
    out = torch.empty(2, device=device)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
Device-side assertion tracking was not enabled by user.""",
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
        gpu=""""normal_kernel_cuda" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.normal(mean=0.0, std=std)

  def test_normal_errors_float_scalar_mean_complex_tensor_std_out(self):
    device = et.device()
    out = torch.empty(2, device=device)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""normal expects standard deviation to be non-complex""",
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
        gpu="""normal expects standard deviation to be non-complex""",
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
        gpu="""normal expects standard deviation to be non-complex""",
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
        gpu="""The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0""",
        tpu="""normal(): the size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0""",
    ):
      torch.normal(
          mean=torch.zeros(2, device=device), std=torch.ones(3, device=device)
      )

  def test_normal_errors_invalid_mean_dtype(self):
    device = et.device()
    with et.assert_raises_message(
        RuntimeError,
        gpu=""""normal_kernel_cuda" not implemented for 'Int'""",
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
        gpu=""""histc" not implemented for 'ComplexFloat'""",
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
        gpu="""linalg.solve_triangular: The input tensor B must have at least 2 dimensions.""",
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
        gpu=""""triangular_solve_cuda" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.linalg.solve_triangular(a, b, upper=True, out=out)

  def test_masked_select_mask_int32(self):
    self_tensor = torch.ones(5, device=et.device())
    mask = torch.ones(5, device=et.device(), dtype=torch.int32)
    out = torch.empty(0, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_select(): expected mask to be a BoolTensor, got int32""",
        gpu="""masked_select: expected BoolTensor for mask""",
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
        gpu=re.compile(
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
        gpu="""expected target dtype to be Long or Byte, but got Int""",
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
        tpu="""adaptive_avg_pool2d(): expected input to be a 3-D or 4-D tensor, got 2-D tensor""",
        gpu="""adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got [10, 10]""",
    ):
      torch.ops.aten.adaptive_avg_pool2d.out(inp, tuple(out.shape), out=out)

  def test_adaptive_avg_pool3d_invalid_rank(self):
    inp = torch.ones(10, 10, 10, device=et.device())
    out = torch.empty(5, 5, 5, device=et.device())

    # adaptive_avg_pool3d expects 4-D or 5-D input.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""adaptive_avg_pool3d(): expected input to be a 4-D or 5-D tensor, got 3-D tensor""",
        gpu="""adaptive_avg_pool3d_cuda(): Expected 4D or 5D tensor, but got [10, 10, 10]""",
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

    with et.assert_raises_message(
        RuntimeError,
        tpu="""max_pool2d_with_indices(): expected non-empty 3D or 4D (batch mode) tensor for input, got 2D tensor""",
        gpu="""non-empty 3D or 4D (batch mode) tensor expected for input""",
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

    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        tpu="""avg_pool2d(): expected non-empty 3D or 4D (batch mode) tensor for input, got 2D tensor""",
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got -3)""",
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
        gpu="""grad_output width unexpected. Expected: 6, Got: 7""",
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
        gpu="""gradOutput width unexpected. Expected: 6, Got: 7""",
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
        gpu="""gradOutput width unexpected. Expected: 6, Got: 7""",
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
        gpu="""gradOutput width unexpected. Expected: 12, Got: 6""",
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
        gpu="""gradOutput height unexpected. Expected: 12, Got: 6""",
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
        gpu="""gradOutput depth unexpected. Expected: 12, Got: 6""",
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
          gpu="""No viable backend for scaled_dot_product_attention was found. This is likely due to turning off both the math kernel and the fused kernels.""",
          message_reviewed_by="wan",
      ):
        torch.nn.functional.scaled_dot_product_attention(query, key, value)

  def test_scaled_dot_product_attention_backward_unsupported_dtype(self):
    grad_output = torch.randn(
        1, 1, 512, 128, device=et.device(), dtype=torch.float64
    )
    query = torch.randn(1, 1, 512, 128, device=et.device(), dtype=torch.float64)
    key = torch.randn(1, 1, 512, 128, device=et.device(), dtype=torch.float64)
    value = torch.randn(1, 1, 512, 128, device=et.device(), dtype=torch.float64)
    out = torch.randn(1, 1, 512, 128, device=et.device(), dtype=torch.float64)
    logsumexp = torch.randn(1, 1, 512, device=et.device(), dtype=torch.float32)
    philox_seed = torch.zeros(1, device=et.device(), dtype=torch.int64)
    philox_offset = torch.zeros(1, device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        NotImplementedError if et.is_on_tpu() else RuntimeError,
        tpu="""scaled_dot_product_efficient_attention_backward(): unsupported dtype for sdpa custom kernel""",
        gpu="""Only fp32, half & bf16 supported at the moment""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._scaled_dot_product_efficient_attention_backward(
          grad_output,
          query,
          key,
          value,
          None,
          out,
          logsumexp,
          philox_seed,
          philox_offset,
          0.0,
          [True, True, True, False],
          False,
          scale=None,
      )

  def test_tril_indices_unsupported_dtype(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""tril_indices(): expected the dtype to be either int32 or int64, got float32""",
        gpu=""""tril_indices_cuda" not implemented for 'Float'""",
        message_reviewed_by="wan",
    ):
      torch.tril_indices(3, 3, dtype=torch.float32, device=et.device())

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          dtype=torch.int32,
          tpu_dtype="int32",
          gpu_dtype="Int",
      ),
      dict(
          testcase_name="int64",
          dtype=torch.int64,
          tpu_dtype="int64",
          gpu_dtype="Long",
      ),
  )
  def test_silu_unsupported_int_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    t = torch.ones(5, device=et.device(), dtype=dtype)

    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""silu(): not implemented for {tpu_dtype}""",
        gpu=f""""silu_cuda" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="gunhyun",
    ):
      torch.nn.functional.silu(t)

  def test_silu_unsupported_dtype_int16(self):
    # int16 is not rejected up front, so the error comes from the lowering
    # when the result is materialized.
    t = torch.ones(5, device=et.device(), dtype=torch.int16)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""silu(): materialization failed with: expected the input dtype to be floating point, got int16""",
        gpu=""""silu_cuda" not implemented for 'Short'""",
    ):
      out = torch.nn.functional.silu(t)
      out.cpu()

  def test_acos_out_dtype_mismatch(self):
    t = torch.ones(5, device=et.device())

    # Call the out variant.
    out = torch.ones(5, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""acos(): result type float32 can't be cast to the desired output type int32""",
        gpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch.acos(t, out=out)

  def test_abs_out_dtype_mismatch(self):
    # Real-to-real mismatched out dtype is disallowed.
    real_f32 = torch.tensor([5.0], device=et.device(), dtype=torch.float32)
    out_f64 = torch.empty(1, device=et.device(), dtype=torch.float64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""abs(): expected the output dtype to be float32, got float64""",
        gpu="""Found dtype Double but expected Float""",
        message_reviewed_by="wan",
    ):
      torch.abs(real_f32, out=out_f64)

    # Complex-to-real invalid downcast.
    x_complex = torch.tensor(
        [3.0 + 4.0j], device=et.device(), dtype=torch.complex64
    )
    out_i32 = torch.empty(1, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""abs(): result type float32 can't be cast to the desired output type int32""",
        gpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch.abs(x_complex, out=out_i32)

  def test_neg_out_dtype_mismatch(self):
    t = torch.ones(5, device=et.device(), dtype=torch.float32)

    # Call the out variant with a mismatching dtype for dtype-preserving ops.
    out = torch.ones(5, device=et.device(), dtype=torch.float64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""neg(): expected the output dtype to be float32, got float64""",
        gpu="""Found dtype Double but expected Float""",
        message_reviewed_by="wan",
    ):
      torch.neg(t, out=out)

  def test_all_out_dtype_mismatch(self):
    t = torch.tensor([True, False], device=et.device())

    # Call the out variant with an invalid dtype (not bool or uint8).
    out = torch.empty(0, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""all(): expected the output dtype to be bool or uint8, got float32""",
        gpu="""all only supports bool tensor for result, got: Float""",
        message_reviewed_by="wan",
    ):
      torch.all(t, out=out)

  def test_any_out_dtype_mismatch(self):
    t = torch.tensor([True, False], device=et.device())

    # Call the out variant with an invalid dtype (not bool or uint8).
    out = torch.empty(0, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""any(): expected the output dtype to be bool or uint8, got float32""",
        gpu="""any only supports bool tensor for result, got: Float""",
        message_reviewed_by="wan",
    ):
      torch.any(t, out=out)

  def test_angle_out_dtype_mismatch(self):
    t = torch.tensor([1.0 + 1.0j], device=et.device())
    out_int = torch.empty(1, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""angle(): expected the output dtype to be float32, got int32""",
        gpu="""result type Float can't be cast to the desired output type Int""",
    ):
      torch.angle(t, out=out_int)

    t_real = torch.tensor([1.0], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""angle(): expected the output dtype to be float32, got int32""",
        gpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="wan",
    ):
      torch.angle(t_real, out=out_int)

  def test_sign_unsupported_dtype_complex(self):
    t = torch.tensor([1 + 1j], device=et.device())

    # Call the out variant.
    out = torch.ones(1, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sign(): expected the input dtype not to be complex, got complex64; use torch.sgn() instead if you intend to normalize a complex tensor to each complex element having magnitude 1""",
        gpu="""Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.""",
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
    err_type = RuntimeError if et.is_on_tpu() else IndexError
    with et.assert_raises_message(
        err_type,
        tpu="""scatter(): materialization failed with: expected the self tensor of shape [5, 5] to have the same rank as the src tensor of shape [5], got 2 vs. 1""",
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
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
        gpu="""Index tensor must have the same number of dimensions as self tensor""",
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
        gpu="""Expected tensor for argument #1 'grad' to have same size as tensor for argument #2 'output'; but [5, 5] does not equal [5] (while checking arguments for softmax_backward)""",
    ):
      torch.ops.aten._softmax_backward_data(
          grad_output, output, 0, torch.float32, grad_input=grad_input
      ).cpu()

  def test_as_strided_negative_offset(self):
    t = torch.empty(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""as_strided(): expected the given storage offset to be >= 0, got -1""",
        gpu="""Tensor: invalid storage offset -1""",
        message_reviewed_by="wan",
    ):
      torch.as_strided(t, (1,), (1,), storage_offset=-1)

  def test_as_strided_size_stride_mismatch(self):
    t = torch.empty(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""as_strided(): expected the given sizes [1, 2] and strides [1] to have the same length, got 2 vs. 1""",
        gpu="""mismatch in length of strides and shape""",
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
        gpu="""invalid shape dimension -2 at index 0 of shape [-2]""",
        message_reviewed_by="wan",
    ):
      t.view(-2)

  def test_view_multiple_neg1(self):
    t = torch.ones(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): expected the given sizes [1, -1, 5, -1] to have up to 1 element equal to -1 (inferred dimension), got 2 occurrences of -1 at indices 1 and 3""",
        gpu="""only one dimension can be inferred""",
        message_reviewed_by="wan",
    ):
      t.view(1, -1, 5, -1)

  def test_view_infer_dimension_0_numel(self):
    t = torch.ones(0, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): cannot infer the dimension for a 0-element view of shape [0, -1] because it's ambiguous, i.e. it could be of any value""",
        gpu="""cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous""",
        message_reviewed_by="wan",
    ):
      t.view(0, -1)

  def test_view_infer_dimension_not_multiple(self):
    t = torch.ones(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): expected the number of elements in the output view of shape [-1, 2] to be a multiple of the number of elements in the input of shape [5] in the presence of an inferred dimension (-1), got 2, which is not a multiple of 5""",
        gpu="""shape '[-1, 2]' is invalid for input of size 5""",
        message_reviewed_by="wan",
    ):
      t.view(-1, 2)

  def test_view_numel_mismatch(self):
    t = torch.ones(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): expected the input of shape [5] to have the same number of elements as the output of shape [2], got 5 vs. 2""",
        gpu="""shape '[2]' is invalid for input of size 5""",
        message_reviewed_by="wan",
    ):
      t.view(2)

  def test_view_not_compatible(self):
    t = torch.ones(2, 3, device=et.device()).T

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view(): cannot create a view of shape [6] from the input tensor of shape [3, 2] and strides [1, 3]; consider creating a new tensor using reshape() instead of taking a view""",
        gpu="""view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.""",
        message_reviewed_by="wan",
    ):
      t.view(6)

  def test_view_as_complex_unsupported_dtypes_int(self):
    t = torch.ones(2, 3, 2, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the input dtype to be float32 or float64, got int32""",
        gpu="""view_as_complex is only supported for half, float and double tensors, but got a tensor of scalar type: Int""",
    ):
      torch.view_as_complex(t)

  def test_view_as_complex_scalar(self):
    t = torch.tensor(1.0, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the input to be a tensor, got a scalar""",
        gpu="""Input tensor must have one or more dimensions""",
        message_reviewed_by="wan",
    ):
      torch.view_as_complex(t)

  def test_view_as_complex_invalid_last_dim(self):
    t = torch.ones(3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the size of the last dimension of the input tensor to be 2, got 3""",
        gpu="""Tensor must have a last dimension of size 2""",
        message_reviewed_by="wan",
    ):
      torch.view_as_complex(t)

  def test_view_as_complex_invalid_last_stride(self):
    t = torch.ones(2, 2, device=et.device()).T

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the stride of the last dimension of the input tensor to be 1, got 2""",
        gpu="""Tensor must have a last dimension with stride 1""",
        message_reviewed_by="wan",
    ):
      torch.view_as_complex(t)

  def test_view_as_complex_invalid_stride(self):
    t = torch.as_strided(torch.ones(5, device=et.device()), (2, 2), (3, 1))

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): expected the input strides [3, 1] to be even numbers (except in the last dimension), got 1 odd stride: 3 at index 0""",
        gpu="""Tensor must have a stride divisible by 2 for all but last dimension""",
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
        gpu="""Expected out type to be Float but got Int""",
        message_reviewed_by="wan",
    ):
      torch.where(condition, inp, other, out=out)

  def test_bucketize_unsupported_complex_dtype_1(self):
    input_tensor = torch.tensor([1 + 1j], device=et.device())
    boundaries = torch.tensor([0.5], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        gpu=""""searchsorted_out_cuda" not implemented for 'ComplexFloat'""",
        tpu="""bucketize(): expected self to be non-complex, got 'ComplexFloat'""",
    ):
      torch.bucketize(input_tensor, boundaries)

  def test_bucketize_unsupported_complex_dtype_2(self):
    input_tensor = torch.tensor([0.5], device=et.device())
    boundaries = torch.tensor([1 + 1j], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        gpu=""""searchsorted_out_cuda" not implemented for 'ComplexFloat'""",
        tpu="""bucketize(): expected boundaries to be non-complex, got 'ComplexFloat'""",
    ):
      torch.bucketize(input_tensor, boundaries)

  def test_bucketize_invalid_boundaries_dim(self):
    input_tensor = torch.tensor([1.0, 2.0], device=et.device())
    boundaries = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        gpu="""boundaries tensor must be 1 dimension, but got dim(2)""",
        tpu="""bucketize(): expected boundaries tensor to be 1D, got dim(2)""",
    ):
      torch.bucketize(input_tensor, boundaries)

  def test_bucketize_bool_unsupported(self):
    input_tensor = torch.tensor([True, False], device=et.device())
    boundaries = torch.tensor([False, True], device=et.device())
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""bucketize(): not implemented for bool""",
        gpu=""""searchsorted_out_cuda" not implemented for 'Bool'""",
    ):
      torch.bucketize(input_tensor, boundaries)

  def test_geqrf_insufficient_dims(self):
    input_tensor = torch.ones(1, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        gpu="""torch.geqrf: input must have at least 2 dimensions.""",
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
        gpu=""""xlogy_cuda" not implemented for 'ComplexFloat'""",
        tpu="""xlogy(): expected non-complex dtypes, got x dtype complex64 and y dtype complex64""",
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
        gpu=""""log_sigmoid_backward_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.log_sigmoid_backward(t, t, t)

  def test_log_sigmoid_backward_invalid_complex_dtype(self):
    t = torch.ones(4, device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""log_sigmoid_backward(): expected the input dtype to be floating point, got complex64""",
        gpu=""""log_sigmoid_backward_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.log_sigmoid_backward(t, t, t)

  def test_log_sigmoid_invalid_bool_dtype(self):
    t = torch.ones(4, device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""log_sigmoid_forward(): expected the input dtype to be floating point, got bool""",
        gpu=""""log_sigmoid_forward_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.log_sigmoid(t)

  def test_log_sigmoid_invalid_complex_dtype(self):
    t = torch.ones(4, device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""log_sigmoid_forward(): expected the input dtype to be floating point, got complex64""",
        gpu=""""log_sigmoid_forward_cuda" not implemented for 'ComplexFloat'""",
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

    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm(): expected the self argument to be a 2D tensor (matrix), got 1D of shape [16]""",
        gpu=re.compile(
            r".*(mat1 must be a matrix|mat_a must be a matrix|self must be a"
            r" matrix|torch\._scaled_mm.*is only supported on CUDA devices).*",
            re.DOTALL,
        ),
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

    err_type = Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm(): expected scale_a to have numel 1 (tensorwise) or 16 (row-wise), got numel 2""",
        gpu=re.compile(
            r".*(Invalid scaling configuration|torch\._scaled_mm.*is only"
            r" supported on CUDA devices).*",
            re.DOTALL,
        ),
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

    err_type = Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm(): expected scale_b to have numel 1 (tensorwise) or 16 (per-channel), got numel 2""",
        gpu=re.compile(
            r".*(Invalid scaling configuration|torch\._scaled_mm.*is only"
            r" supported on CUDA devices).*",
            re.DOTALL,
        ),
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

    err_type = Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm(): expected scale_result to have numel 1, got numel 2""",
        gpu=re.compile(
            r".*(scale_result must be|Invalid scaling"
            r" configuration|torch\._scaled_mm.*is only supported on CUDA"
            r" devices).*",
            re.DOTALL,
        ),
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
        gpu=re.compile(
            r".*(mat1 and mat2 shapes cannot be multiplied|scaled_mm\(\):"
            r" expected column size of first matrix to match row size of second"
            r" matrix|torch\._scaled_mm is only supported on CUDA devices).*",
            re.DOTALL,
        ),
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b)

  def test_scaled_mm_v2_non_2d_self(self):
    """Tests that _scaled_mm_v2 fails if self is not 2D."""
    (
        _,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    dev = et.device()
    self_1d = torch.randn(16, dtype=torch.float32, device=dev).to(
        torch.float8_e4m3fn
    )
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected the self argument to be a 2D tensor (matrix), got 1D of shape [16]""",
        gpu=re.compile(
            r".*(mat1 must be a matrix|mat_a must be a matrix|self must be a 2D"
            r" matrix|torch\._scaled_mm.*is only"
            r" supported on CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_1d,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_incompatible_shapes(self):
    """Tests that _scaled_mm_v2 fails if matrix shapes are incompatible."""
    (
        _,
        _,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    dev = et.device()
    mat1 = torch.randn(16, 32, dtype=torch.float32, device=dev).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 32, dtype=torch.float32, device=dev).to(
        torch.float8_e4m3fn
    )
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected column size of first matrix to match row size of second matrix, got shapes [16, 32] and [16, 32]""",
        gpu=re.compile(
            r".*(mat_a and mat_b shapes cannot be multiplied|expected column"
            r" size of first matrix to match row size of second"
            r" matrix|torch\._scaled_mm.*is only supported on CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          mat1,
          mat2,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_invalid_scale_a_size(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected scale_a list to contain exactly 1 tensor, got 2""",
        gpu=re.compile(
            r".*(scale_a must have 1 Float element|torch\._scaled_mm is only"
            r" supported on CUDA devices|torch\._scaled_mm_v2 is only"
            r" supported on CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu + scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_invalid_recipe_a_size(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected recipe_a list to contain exactly 1 recipe, got 2""",
        gpu=re.compile(
            r".*(Invalid scaling configuration for _scaled_mm_v2: unsupported"
            r" recipe|torch\._scaled_mm is only supported on CUDA"
            r" devices|torch\._scaled_mm_v2 is only supported on CUDA"
            r" devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a + recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_invalid_swizzle_a_size(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected swizzle_a list to contain exactly 1 swizzle mode, got 2""",
        gpu=re.compile(
            r".*(Only multiplication of row-major|swizzle_a must have 1"
            r" value|torch\._scaled_mm.*is only"
            r" supported on CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a + swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_unsupported_recipe(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        _,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected scaling recipe for scale_a to be TensorWise, RowWise, or BlockWise1x32, got 2""",
        gpu=re.compile(
            r".*(Invalid scaling configuration for _scaled_mm_v2: unsupported"
            r" recipe|torch\._scaled_mm is only supported on CUDA"
            r" devices|torch\._scaled_mm_v2 is only supported on CUDA"
            r" devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          [2],
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_unsupported_swizzle(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        _,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected swizzle type for scale_a to be NO_SWIZZLE or SWIZZLE_32_4_4, got 2""",
        gpu=re.compile(
            r".*(Only multiplication of row-major|scale_a must be swizzled to"
            r" SWIZZLE_32_4_4"
            r" format|torch\._scaled_mm.*is only supported on CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          [2],
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_invalid_scale_b_size(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected scale_b list to contain exactly 1 tensor, got 2""",
        gpu=re.compile(
            r".*(scale_b must have 1 Float element|torch\._scaled_mm is only"
            r" supported on CUDA devices|torch\._scaled_mm_v2 is only"
            r" supported on CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu + scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_invalid_recipe_b_size(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected recipe_b list to contain exactly 1 recipe, got 2""",
        gpu=re.compile(
            r".*(Invalid scaling configuration for _scaled_mm_v2: unsupported"
            r" recipe|torch\._scaled_mm is only supported on CUDA"
            r" devices|torch\._scaled_mm_v2 is only supported on CUDA"
            r" devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b + recipe_b,
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_invalid_swizzle_b_size(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected swizzle_b list to contain exactly 1 swizzle mode, got 2""",
        gpu=re.compile(
            r".*(Only multiplication of row-major|swizzle_b must have 1"
            r" value|torch\._scaled_mm.*is only"
            r" supported on CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b + swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_unsupported_recipe_b(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        _,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected scaling recipe for scale_b to be TensorWise, RowWise, or BlockWise1x32, got 2""",
        gpu=re.compile(
            r".*(Invalid scaling configuration for _scaled_mm_v2: unsupported"
            r" recipe|torch\._scaled_mm is only supported on CUDA"
            r" devices|torch\._scaled_mm_v2 is only supported on CUDA"
            r" devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          [2],
          swizzle_b,
          None,
          None,
      )

  def test_scaled_mm_v2_unsupported_swizzle_b(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        _,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected swizzle type for scale_b to be NO_SWIZZLE or SWIZZLE_32_4_4, got 2""",
        gpu=re.compile(
            r".*(Only multiplication of row-major|scale_b must be swizzled to"
            r" SWIZZLE_32_4_4"
            r" format|torch\._scaled_mm.*is only supported on CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          [2],
          None,
          None,
      )

  def test_scaled_mm_v2_invalid_bias_dim(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    dev = et.device()
    bias = torch.randn(1, 1, 1, dtype=torch.float32, device=dev)
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected bias to be 1D or 2D tensor, got 3D""",
        gpu=re.compile(
            r".*(Bias must be size|bias|torch\._scaled_mm.*is only supported on"
            r" CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          bias,
          None,
      )

  def test_scaled_mm_v2_bias_dtype_mismatch(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    dev = et.device()
    bias = torch.zeros(16, dtype=torch.int32, device=dev)
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected bias dtype to be float32 or bfloat16, got int32""",
        gpu=re.compile(
            r".*(Bias must be|bias|torch\._scaled_mm.*is only supported on CUDA"
            r" devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          bias,
          None,
      )

  def test_scaled_mm_v2_invalid_out_dtype(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    err_type = RuntimeError if et.is_on_tpu() else Exception
    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected out dtype to be float32, bfloat16, or float16, got int32""",
        gpu=re.compile(
            r".*(Only multiplication of row-major|out|torch\._scaled_mm.*is"
            r" only supported on CUDA devices).*",
            re.DOTALL,
        ),
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          torch.int32,
      )

  def test_fft_c2c_non_complex_input(self):
    t = torch.ones(4, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""fft_c2c(): materialization failed with: expected complex input type, got float32""",
        gpu="""Expected self.is_complex() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
    ):
      torch.ops.aten._fft_c2c(t, [0], 0, True)

  def test_fft_c2r_non_complex_input(self):
    t = torch.ones(4, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""fft_c2r(): materialization failed with: expected complex input type, got float32""",
        gpu="""Expected self.is_complex() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
    ):
      torch.ops.aten._fft_c2r(t, [0], 0, 4)

  def test_dunder_lshift_unsupported_dtype(self):
    x = torch.tensor([1.0], device=et.device())
    y = torch.tensor([1.0], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""__lshift__(): expected the dtype of the first argument to be integer, got float32""",
        gpu=""""lshift_cuda" not implemented for 'Float'""",
        message_reviewed_by="wan",
    ):
      x << y  # pylint: disable=pointless-statement

  def test_dunder_rshift_unsupported_dtype(self):
    x = torch.tensor([1.0], device=et.device())
    y = torch.tensor([1.0], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""__rshift__(): expected the dtype of the first argument to be integer, got float32""",
        gpu=""""rshift_cuda" not implemented for 'Float'""",
        message_reviewed_by="wan",
    ):
      x >> y  # pylint: disable=pointless-statement

  def test_put_index_dtype_mismatch(self):
    self_t = torch.ones(5, device=et.device())
    index = torch.tensor([1.0], device=et.device())
    source = torch.tensor([2.0], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""put_(): expected a long tensor for index, got float32""",
        gpu="""put_(): Expected a long tensor for index, but got Float""",
        message_reviewed_by="gunhyun",
    ):
      self_t.put_(index, source)

  def test_put_dtype_mismatch(self):
    self_t = torch.ones(5, device=et.device())
    index = torch.tensor([1], device=et.device())
    source = torch.tensor([2], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""put_(): expected self and source to have the same dtype, got self dtype float32 and source dtype int32""",
        gpu="""put_(): self and source expected to have the same dtype, but got self.dtype = Float and source.dtype = Int""",
        message_reviewed_by="gunhyun",
    ):
      self_t.put_(index, source)

  def test_put_size_mismatch(self):
    self_t = torch.ones(5, device=et.device())
    index = torch.tensor([1, 2], device=et.device())
    source = torch.tensor([2.0], device=et.device())
    with et.assert_raises_message(
        IndexError,
        tpu="""put_(): expected source and index to have the same number of elements, got source numel 1 and index numel 2""",
        gpu="""put_(): Expected source and index to have the same number of elements, but got source.numel() = 1, index.numel() = 2""",
        message_reviewed_by="gunhyun",
    ):
      self_t.put_(index, source)

  def test_put_empty_destination(self):
    self_t = torch.ones(0, device=et.device())
    index = torch.tensor([0], device=et.device())
    source = torch.tensor([2.0], device=et.device())
    with et.assert_raises_message(
        IndexError,
        tpu="""put_(): expected self to be non-empty, got self numel 0""",
        gpu="""put_(): Tried to put elements into an empty tensor""",
        message_reviewed_by="gunhyun",
    ):
      self_t.put_(index, source)

  def test_ldexp_invalid_output_type(self):
    out = torch.empty((2,), dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""ldexp(): ldexp can't be cast to the desired output type int32""",
        gpu="""ldexp can't be cast to the desired output type Int""",
        message_reviewed_by="jparkerh",
    ):
      torch.ldexp(
          torch.tensor([1.0, 2.0], device=et.device()),
          torch.tensor([1, 2], device=et.device()),
          out=out,
      )

  def test_searchsorted_invalid_side(self):
    a = torch.tensor([1, 2, 3])
    v = torch.tensor([2])
    with et.assert_raises_message(
        RuntimeError,
        tpu="""searchsorted(): expected side to be 'left' or 'right', got 'middle'""",
        gpu="""torch.searchsorted(): side can only be 'left' or 'right' but got middle""",
        message_reviewed_by="adivinpatel",
    ):
      torch.searchsorted(a.to(et.device()), v.to(et.device()), side="middle")

  def test_searchsorted_dim_mismatch(self):
    a = torch.tensor([[1, 2], [3, 4]])
    v = torch.tensor([2])
    with et.assert_raises_message(
        RuntimeError,
        tpu="""searchsorted(): expected sorted_sequence to be 1-dimensional or have the same number of dimensions as values, got 2 and 1""",
        gpu="""torch.searchsorted(): boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries tensor and input value tensor must match, but we got boundaries tensor [2, 2] and input value tensor [1]""",
        message_reviewed_by="adivinpatel",
    ):
      torch.searchsorted(a.to(et.device()), v.to(et.device()))

  def test_searchsorted_shape_mismatch(self):
    a = torch.tensor([[1, 2], [3, 4]])
    v = torch.tensor([[2], [3], [4]])
    with et.assert_raises_message(
        RuntimeError,
        tpu="""searchsorted(): expected sorted_sequence to have same shape as values except for the last dimension, got [2, 2] and [3, 1]""",
        gpu="""torch.searchsorted(): boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries tensor and input value tensor must match, but we got boundaries tensor [2, 2] and input value tensor [3, 1]""",
        message_reviewed_by="adivinpatel",
    ):
      torch.searchsorted(a.to(et.device()), v.to(et.device()))

  def test_searchsorted_invalid_sorter_shape(self):
    a = torch.tensor([[1, 2], [3, 4]])
    v = torch.tensor([[2, 2], [3, 3]])
    sorter = torch.tensor([[1, 0]])
    with et.assert_raises_message(
        RuntimeError,
        tpu="""searchsorted(): expected sorter and sorted_sequence to have the same shape, got [1, 2] and [2, 2]""",
        gpu="""torch.searchsorted(): boundary and sorter must have the same size, but got boundary tensor [2, 2]and got sorter tensor [1, 2]""",
        message_reviewed_by="adivinpatel",
    ):
      torch.searchsorted(
          a.to(et.device()), v.to(et.device()), sorter=sorter.to(et.device())
      )

  def test_searchsorted_invalid_sorter_dtype(self):
    a = torch.tensor([1, 2, 3])
    v = torch.tensor([2])
    sorter = torch.tensor([0, 1, 2], dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""searchsorted(): expected sorter to have Long dtype, got Int""",
        gpu="""torch.searchsorted(): sorter must be a tensor of long dtype but got dtype Int""",
        message_reviewed_by="adivinpatel",
    ):
      torch.searchsorted(
          a.to(et.device()), v.to(et.device()), sorter=sorter.to(et.device())
      )

  def test_searchsorted_scalar_value_invalid(self):
    a = torch.tensor([[1, 2], [3, 4]])
    v = torch.tensor(2)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""searchsorted(): expected values to not be a scalar when sorted_sequence dimension is not 1, got sorted_sequence dim 2 and values dim 0""",
        gpu="""torch.searchsorted(): input value can be a scalar only when boundaries tensor dimension is 1, but we got boundaries tensor dim(2) and input value's dim(0) numel(1)""",
        message_reviewed_by="adivinpatel",
    ):
      torch.searchsorted(a.to(et.device()), v.to(et.device()))

  def test_searchsorted_scalar_sequence_invalid(self):
    a = torch.tensor(1)
    v = torch.tensor([2])
    with et.assert_raises_message(
        RuntimeError,
        tpu="""searchsorted(): expected sorted_sequence to have >0 dimension, got 0""",
        gpu="""torch.searchsorted(): boundaries tensor should have positive dimension, but got 0 dimension""",
        message_reviewed_by="adivinpatel",
    ):
      torch.searchsorted(a.to(et.device()), v.to(et.device()))

  def test_searchsorted_right_and_side_contradiction(self):
    a = torch.tensor([1, 2, 3])
    v = torch.tensor([2])
    with et.assert_raises_message(
        RuntimeError,
        tpu="""searchsorted(): expected side and right to not be opposites, got side 'left' and right True""",
        gpu="""torch.searchsorted(): side and right can't be set to opposites, got side of left while right was True""",
        message_reviewed_by="adivinpatel",
    ):
      torch.searchsorted(
          a.to(et.device()), v.to(et.device()), right=True, side="left"
      )

  def test_searchsorted_bool_unsupported(self):
    sorted_seq = torch.tensor([False, True], device=et.device())
    values = torch.tensor([True, False], device=et.device())
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""searchsorted(): not implemented for bool""",
        gpu=""""searchsorted_out_cuda" not implemented for 'Bool'""",
    ):
      torch.searchsorted(sorted_seq, values)

  def test_fused_adagrad_default_int32_dtype(self):
    device = et.device()
    p = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g = torch.tensor([1, 2], dtype=torch.int32, device=device)
    v = torch.tensor([1, 2], dtype=torch.int32, device=device)
    s = torch.tensor([1, 2], dtype=torch.int32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adagrad_(): expected the input dtype to be floating-point, got int32""",
        gpu=re.compile(
            r""".*"fused_adagrad_kernel_cuda" not implemented for 'Int'.*"""
        ),
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.default(
          [p],
          [g],
          [v],
          [s],
          lr=0.1,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  def test_fused_adagrad_default_complex64_dtype(self):
    device = et.device()
    p = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=device)
    g = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    v = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    s = torch.tensor([1.0 + 0.0j], dtype=torch.complex64, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adagrad_(): expected the input dtype to be floating-point, got complex64""",
        gpu=re.compile(
            r""".*"fused_adagrad_kernel_cuda" not implemented for 'ComplexFloat'.*"""
        ),
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.default(
          [p],
          [g],
          [v],
          [s],
          lr=0.1,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  def test_fused_adagrad_tensor_lr_int32_dtype(self):
    device = et.device()
    p = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g = torch.tensor([1, 2], dtype=torch.int32, device=device)
    v = torch.tensor([1, 2], dtype=torch.int32, device=device)
    s = torch.tensor([1, 2], dtype=torch.int32, device=device)
    lr = torch.tensor(0.1, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adagrad_(): expected the input dtype to be floating-point, got int32""",
        gpu=re.compile(
            r""".*"fused_adagrad_kernel_cuda" not implemented for 'Int'.*"""
        ),
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.tensor_lr(
          [p],
          [g],
          [v],
          [s],
          lr=lr,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  def test_fused_adagrad_tensor_lr_complex64_dtype(self):
    device = et.device()
    p = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=device)
    g = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    v = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    s = torch.tensor([1.0 + 0.0j], dtype=torch.complex64, device=device)
    lr = torch.tensor(0.1, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adagrad_(): expected the input dtype to be floating-point, got complex64""",
        gpu=re.compile(
            r""".*"fused_adagrad_kernel_cuda" not implemented for 'ComplexFloat'.*"""
        ),
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.tensor_lr(
          [p],
          [g],
          [v],
          [s],
          lr=lr,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  def test_thnn_fused_lstm_cell_invalid_gate_sizes(self):
    device = et.device()
    ig = torch.randn(2, 32, device=device)
    hg = torch.randn(4, 32, device=device)
    cx = torch.randn(4, 8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""thnn_fused_lstm_cell(): expected size of argument #1 'input_gates' to match size of argument #2 'hidden_gates' ([4, 32]), got [2, 32]""",
        gpu="""Expected tensor for argument #1 'input_gates' to have same size as tensor for argument #2 'hidden_gates'; but [2, 32] does not equal [4, 32] (while checking arguments for _thnn_fused_lstm_cell_cuda)""",
    ):
      torch.ops.aten._thnn_fused_lstm_cell(ig, hg, cx)

  def test_thnn_fused_lstm_cell_invalid_batch_size(self):
    device = et.device()
    ig = torch.randn(2, 32, device=device)
    hg = torch.randn(2, 32, device=device)
    cx = torch.randn(4, 8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""thnn_fused_lstm_cell(): expected batch size of argument #1 'input_gates' to match batch size of argument #3 'cx' (4), got 2""",
        gpu="""Expected tensor for argument #5 'prev_hidden' to have 16 elements; but it actually has 32 elements (while checking arguments for _thnn_fused_lstm_cell_cuda)""",
    ):
      torch.ops.aten._thnn_fused_lstm_cell(ig, hg, cx)

  def test_thnn_fused_lstm_cell_invalid_feature_size(self):
    device = et.device()
    ig = torch.randn(4, 10, device=device)
    hg = torch.randn(4, 10, device=device)
    cx = torch.randn(4, 8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""thnn_fused_lstm_cell(): expected feature size of argument #1 'input_gates' to match 4 * feature size of argument #3 'cx' (32), got 10""",
        gpu="""Expected tensor for argument #5 'prev_hidden' to have 10 elements; but it actually has 32 elements (while checking arguments for _thnn_fused_lstm_cell_cuda)""",
    ):
      torch.ops.aten._thnn_fused_lstm_cell(ig, hg, cx)

  def test_thnn_fused_gru_cell_invalid_gate_sizes(self):
    device = et.device()
    ig = torch.randn(2, 24, device=device)
    hg = torch.randn(4, 24, device=device)
    hx = torch.randn(4, 8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""thnn_fused_gru_cell(): expected size of argument #1 'input_gates' to match size of argument #2 'hidden_gates' ([4, 24]), got [2, 24]""",
        gpu="""Expected tensor for argument #1 'input_gates' to have same size as tensor for argument #2 'hidden_gates'; but [2, 24] does not equal [4, 24] (while checking arguments for _thnn_fused_gru_cell_cuda)""",
    ):
      torch.ops.aten._thnn_fused_gru_cell(ig, hg, hx)

  def test_thnn_fused_gru_cell_invalid_batch_size(self):
    device = et.device()
    ig = torch.randn(2, 24, device=device)
    hg = torch.randn(2, 24, device=device)
    hx = torch.randn(4, 8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""thnn_fused_gru_cell(): expected batch size of argument #1 'input_gates' to match batch size of argument #3 'hx' (4), got 2""",
        gpu="""Expected tensor for argument #5 'prev_hidden' to have 16 elements; but it actually has 32 elements (while checking arguments for _thnn_fused_gru_cell_cuda)""",
    ):
      torch.ops.aten._thnn_fused_gru_cell(ig, hg, hx)

  def test_thnn_fused_gru_cell_invalid_feature_size(self):
    device = et.device()
    ig = torch.randn(4, 10, device=device)
    hg = torch.randn(4, 10, device=device)
    hx = torch.randn(4, 8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""thnn_fused_gru_cell(): expected feature size of argument #1 'input_gates' to match 3 * feature size of argument #3 'hx' (24), got 10""",
        gpu="""Expected tensor for argument #5 'prev_hidden' to have 13 elements; but it actually has 32 elements (while checking arguments for _thnn_fused_gru_cell_cuda)""",
    ):
      torch.ops.aten._thnn_fused_gru_cell(ig, hg, hx)

  def test_fused_moving_avg_obs_fq_helper_non_floating(self):
    dev = et.device()
    self_t = torch.ones((2, 3), dtype=torch.int32, device=dev)
    observer_on = torch.tensor([1], dtype=torch.int32, device=dev)
    fake_quant_on = torch.tensor([1], dtype=torch.int32, device=dev)
    running_min = torch.empty((0,), dtype=torch.float32, device=dev)
    running_max = torch.empty((0,), dtype=torch.float32, device=dev)
    scale = torch.empty((0,), dtype=torch.float32, device=dev)
    zero_point = torch.empty((0,), dtype=torch.int32, device=dev)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_moving_avg_obs_fq_helper(): expected floating point tensor for self, got int32""",
        gpu=""""aminmax_kernel" not implemented for 'Int'""",
    ):
      torch.ops.aten._fused_moving_avg_obs_fq_helper(
          self_t,
          observer_on,
          fake_quant_on,
          running_min,
          running_max,
          scale,
          zero_point,
          0.01,
          0,
          255,
          0,
          False,
          False,
      )

  def test_fused_moving_avg_obs_fq_helper_zero_dim_per_channel(self):
    dev = et.device()
    self_t = torch.tensor(1.5, dtype=torch.float32, device=dev)
    observer_on = torch.tensor([1], dtype=torch.int32, device=dev)
    fake_quant_on = torch.tensor([1], dtype=torch.int32, device=dev)
    running_min = torch.empty((0,), dtype=torch.float32, device=dev)
    running_max = torch.empty((0,), dtype=torch.float32, device=dev)
    scale = torch.empty((0,), dtype=torch.float32, device=dev)
    zero_point = torch.empty((0,), dtype=torch.int32, device=dev)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_moving_avg_obs_fq_helper(): expected positive tensor rank when per_row_fake_quant is true, got rank 0""",
        gpu="""Error in fused_moving_avg_obs_fake_quant_cpu: ch_axis must be < self.dim()""",
    ):
      torch.ops.aten._fused_moving_avg_obs_fq_helper(
          self_t,
          observer_on,
          fake_quant_on,
          running_min,
          running_max,
          scale,
          zero_point,
          0.01,
          0,
          255,
          0,
          True,
          False,
      )

  def test_fused_moving_avg_obs_fq_helper_out_of_bounds_ch_axis(self):
    dev = et.device()
    self_t = torch.ones((2, 3), dtype=torch.float32, device=dev)
    observer_on = torch.tensor([1], dtype=torch.int32, device=dev)
    fake_quant_on = torch.tensor([1], dtype=torch.int32, device=dev)
    running_min = torch.empty((0,), dtype=torch.float32, device=dev)
    running_max = torch.empty((0,), dtype=torch.float32, device=dev)
    scale = torch.empty((0,), dtype=torch.float32, device=dev)
    zero_point = torch.empty((0,), dtype=torch.int32, device=dev)
    err_type = Exception
    with et.assert_raises_message(
        err_type,
        tpu="""fused_moving_avg_obs_fq_helper(): expected dimension to be in range [-2, 1], got 5""",
        gpu="""Error in fused_moving_avg_obs_fake_quant_cpu: ch_axis must be < self.dim()""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._fused_moving_avg_obs_fq_helper(
          self_t,
          observer_on,
          fake_quant_on,
          running_min,
          running_max,
          scale,
          zero_point,
          0.01,
          0,
          255,
          5,
          True,
          False,
      )

  def test_native_multi_head_attention_invalid_query_dim(self):
    device = et.device()
    query = torch.ones(2, 4, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected 3-D query, got 2-D tensor""",
        gpu="""expected 3-D `query`, got 2-D tensor""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_query_embed_dim(self):
    device = et.device()
    query = torch.ones(2, 4, 7, device=device)
    key = torch.ones(2, 4, 7, device=device)
    value = torch.ones(2, 4, 7, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected embed_dim to match the last dim of query (7), got 8""",
        gpu="""passed-in embed_dim 8 didn't match last dim of query 7""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_key_dim(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected 3-D key, got 2-D tensor""",
        gpu="""expected 3-D `key`, got 2-D tensor""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_value_dim(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected 3-D value, got 2-D tensor""",
        gpu="""expected 3-D `value`, got 2-D tensor""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_shapes_mismatch(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 5, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected query, key, and value shapes to match, got [2, 4, 8], [2, 5, 8], and [2, 4, 8]""",
        gpu="""expected `query`/`key`/`value` shapes to match""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_qkv_weight_dim(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected 2-D qkv_weight, got 1-D tensor""",
        gpu="""expected 2-D `qkv_weight`, got 1-D tensor""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_qkv_weight_dim0(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(20, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected qkv_weight first dim to be 3x embed_dim (24), got 20""",
        gpu="""expected `qkv_weight` first dim to be 3x embed_dim""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_qkv_weight_dim1(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 7, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected qkv_weight second dim to be embed_dim (8), got 7""",
        gpu="""expected `qkv_weight` second dim to be embed_Dim""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_qkv_bias_dim(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, 1, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected 1-D qkv_bias, got 2-D tensor""",
        gpu="""expected 1-D `qkv_bias`, got 2-D tensor""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_qkv_bias_dim0(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(20, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected qkv_bias first dim to be 3x embed_dim (24), got 20""",
        gpu="""expected `qkv_bias` first dim and first dim of query to be equal""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_embed_dim_not_divisible_by_num_head(
      self,
  ):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected embed_dim to be divisible by num_head, got 8, which is not divisible by 3""",
        gpu="""`embed_dim` must divide evenly by `num_heads`""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 3, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_proj_weight_dim(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected 2-D proj_weight, got 1-D tensor""",
        gpu="""addmm: mat2 must be a matrix, got 1-D tensor""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_proj_weight_dim0(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(7, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected proj_weight first dim to be embed_dim (8), got 7""",
        gpu="""The expanded size of the tensor (7) must match the existing size (8) at non-singleton dimension 1.  Target sizes: [8, 7].  Tensor sizes: [8]""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_proj_weight_dim1(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 7, device=device)
    proj_bias = torch.ones(8, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected proj_weight second dim to be embed_dim (8), got 7""",
        gpu="""addmm: mat1 and mat2 shapes cannot be multiplied (8x8 and 7x8)""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_proj_bias_dim0(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(7, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected proj_bias first dim to be embed_dim (8), got 7""",
        gpu="""The expanded size of the tensor (8) must match the existing size (7) at non-singleton dimension 1.  Target sizes: [8, 8].  Tensor sizes: [7]""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_invalid_mask_shape(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    mask = torch.ones(2, 2, 4, 8, dtype=torch.bool, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected 4-D mask shape to be [2, 2, 4, 4], got [2, 2, 4, 8]""",
        gpu="""Mask shape should match input. mask: [2, 2, 4, 8] input: [2, 2, 4, 4]""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query,
          key,
          value,
          8,
          2,
          qkv_weight,
          qkv_bias,
          proj_weight,
          proj_bias,
          mask,
          True,
          True,
          2,
      )

  def test_native_multi_head_attention_invalid_mask_rank(self):
    device = et.device()
    query = torch.ones(2, 4, 8, device=device)
    key = torch.ones(2, 4, 8, device=device)
    value = torch.ones(2, 4, 8, device=device)
    qkv_weight = torch.ones(24, 8, device=device)
    qkv_bias = torch.ones(24, device=device)
    proj_weight = torch.ones(8, 8, device=device)
    proj_bias = torch.ones(8, device=device)
    mask = torch.ones(2, 4, 4, dtype=torch.bool, device=device)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected mask to be a 2D or a 4D tensor, got 3D tensor of shape [2, 4, 4]""",
        gpu="""Mask shape should match input. mask: [2, 4, 4] input: [2, 2, 4, 4]""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query,
          key,
          value,
          8,
          2,
          qkv_weight,
          qkv_bias,
          proj_weight,
          proj_bias,
          mask,
          True,
          False,
          0,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="bfloat16",
          dtype=torch.bfloat16,
          tpu_dtype="bfloat16",
          gpu_dtype="BFloat16",
      ),
      dict(
          testcase_name="bool",
          dtype=torch.bool,
          tpu_dtype="bool",
          gpu_dtype="Bool",
      ),
  )
  def test_histc_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype: str, gpu_dtype: str
  ):
    x = torch.ones(2, 2, device=et.device(), dtype=dtype)
    with et.assert_raises_message(
        NotImplementedError,
        tpu=f"""histc(): not implemented for {tpu_dtype}""",
        gpu=f""""histc" not implemented for '{gpu_dtype}'""",
        message_reviewed_by="wan",
    ):
      torch.histc(x)

  def test_foreach_unary_ops_bool(self):
    t = torch.ones(2, 2, dtype=torch.bool, device=et.device())
    ops = [
        (torch._foreach_ceil, "ceil"),
        (torch._foreach_floor, "floor"),
    ]
    for op, op_name in ops:
      with et.assert_raises_message(
          NotImplementedError,
          gpu=f""""{op_name}_cuda" not implemented for 'Bool'""",
          tpu=f"""foreach_{op_name}(): not implemented for bool""",
      ):
        op([t])

  def test_unary_ops_bool(self):
    t = torch.ones(2, 2, dtype=torch.bool, device=et.device())
    ops = [
        (torch.ceil, "ceil"),
        (torch.floor, "floor"),
        (torch.nn.functional.silu, "silu"),
    ]
    for op, op_name in ops:
      with et.assert_raises_message(
          NotImplementedError,
          gpu=f""""{op_name}_cuda" not implemented for 'Bool'""",
          tpu=f"""{op_name}(): not implemented for bool""",
      ):
        op(t)

  def test_unary_out_ops_bool(self):
    t = torch.ones(2, 2, dtype=torch.bool, device=et.device())
    out = torch.empty(2, 2, dtype=torch.bool, device=et.device())
    ops = [
        (torch.ops.aten.ceil.out, "ceil"),
        (torch.ops.aten.floor.out, "floor"),
        (torch.ops.aten.silu.out, "silu"),
    ]
    for op, op_name in ops:
      with et.assert_raises_message(
          NotImplementedError,
          gpu=f""""{op_name}_cuda" not implemented for 'Bool'""",
          tpu=f"""{op_name}(): not implemented for bool""",
      ):
        op(t, out=out)

  @parameterized.named_parameters(
      dict(
          testcase_name="nearest",
          mode="nearest",
          gpu_op_name="upsample_nearest2d_out_frame",
          tpu_op_name="upsample_nearest2d",
      ),
      dict(
          testcase_name="bilinear",
          mode="bilinear",
          gpu_op_name="upsample_bilinear2d_out_frame",
          tpu_op_name="upsample_bilinear2d",
      ),
      dict(
          testcase_name="bicubic",
          mode="bicubic",
          gpu_op_name="upsample_bicubic2d_out_frame",
          tpu_op_name="upsample_bicubic2d",
      ),
  )
  def test_upsample_2d_unsupported_dtypes(
      self, mode: str, gpu_op_name: str, tpu_op_name: str
  ):
    for dtype, tpu_dtype, gpu_dtype in (
        (torch.bool, "bool", "Bool"),
        (torch.int32, "int32", "Int"),
        (torch.int64, "int64", "Long"),
    ):
      t = torch.ones(1, 1, 2, 2, device=et.device(), dtype=dtype)
      with et.assert_raises_message(
          NotImplementedError,
          tpu=f"""{tpu_op_name}(): not implemented for {tpu_dtype}""",
          gpu=f""""{gpu_op_name}" not implemented for '{gpu_dtype}'""",
          message_reviewed_by="gunhyun",
      ):
        torch.nn.functional.interpolate(t, scale_factor=2, mode=mode)

  def test_transformer_encoder_layer_fwd_invalid_src_dim(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 3-D src, got 2-D tensor""",
        gpu="""expected 3-D `query`, got 2-D tensor""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_mismatch_embed_dim(self):
    embed_dim = 32
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected last dimension of src to match embed_dim 32, got 16""",
        gpu="""Given normalized_shape=[32], expected input with shape [*, 32], but got input of size[2, 8, 16]""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_num_heads(self):
    embed_dim = 16
    num_heads = 5
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected embed_dim to divide cleanly by num_heads, got embed_dim 16 and num_heads 5""",
        gpu="""`embed_dim` must divide evenly by `num_heads`""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_weight_shape_mismatch(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(30, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected qkv_weight shape [48, 16], got [30, 16]""",
        gpu="""expected `qkv_weight` first dim to be 3x embed_dim""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_qkv_weight_dim(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3, 48, 16, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 2-D qkv_weight, got 3-D tensor""",
        gpu="""expected 2-D `qkv_weight`, got 3-D tensor""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_qkv_bias_dim(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(48, 1, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 1-D qkv_bias, got 2-D tensor""",
        gpu="""expected 1-D `qkv_bias`, got 2-D tensor""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_qkv_bias_size(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(30, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected qkv_bias size 48, got 30""",
        gpu="""expected `qkv_bias` first dim and first dim of query to be equal""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_proj_weight_dim(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(16, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 2-D proj_weight, got 1-D tensor""",
        gpu="""addmm: mat2 must be a matrix, got 1-D tensor""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_proj_weight_shape(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(16, 32, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected proj_weight shape [16, 16], got [16, 32]""",
        gpu="""addmm: mat1 and mat2 shapes cannot be multiplied (16x16 and 32x16)""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_proj_bias_size(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(32, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected proj_bias size 16, got 32""",
        gpu="""The expanded size of the tensor (16) must match the existing size (32) at non-singleton dimension 1.  Target sizes: [16, 16].  Tensor sizes: [32]""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_norm_weight_1(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(32, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 1-D norm_weight_1 with size 16, got [32]""",
        gpu="""Expected weight to be of same shape as normalized_shape, but got weight of shape [32] and normalized_shape = [16]""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_norm_bias_1(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(32, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 1-D norm_bias_1 with size 16, got [32]""",
        gpu="""Expected bias to be of same shape as normalized_shape, but got bias of shape [32] and normalized_shape = [16]""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_norm_weight_2(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(32, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 1-D norm_weight_2 with size 16, got [32]""",
        gpu="""Expected weight to be of same shape as normalized_shape, but got weight of shape [32] and normalized_shape = [16]""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_norm_bias_2(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(32, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 1-D norm_bias_2 with size 16, got [32]""",
        gpu="""Expected bias to be of same shape as normalized_shape, but got bias of shape [32] and normalized_shape = [16]""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_ffn_weight_1_dim(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(32, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 2-D ffn_weight_1, got 1-D tensor""",
        gpu="""2d weights expected""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_ffn_weight_1_size1(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(32, 24, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected ffn_weight_1 second dim to be 16, got 24""",
        gpu="""addmm: mat1 and mat2 shapes cannot be multiplied (16x16 and 24x32)""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_ffn_bias_1(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(16, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 1-D ffn_bias_1 with size 32, got [16]""",
        gpu="""The expanded size of the tensor (32) must match the existing size (16) at non-singleton dimension 1.  Target sizes: [16, 32].  Tensor sizes: [16]""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_ffn_weight_2_shape(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, 16, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected ffn_weight_2 shape [16, 32], got [16, 16]""",
        gpu="""addmm: mat1 and mat2 shapes cannot be multiplied (16x32 and 16x16)""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_ffn_bias_2(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 1-D ffn_bias_2 with size 16, got [32]""",
        gpu="""The expanded size of the tensor (16) must match the existing size (32) at non-singleton dimension 1.  Target sizes: [16, 16].  Tensor sizes: [32]""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  def test_transformer_encoder_layer_fwd_invalid_mask_dim(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)
    mask = torch.randn(16, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 2-D, 3-D, or 4-D mask, got 1-D tensor""",
        gpu="""Mask Type should be defined""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
          mask,
      )

  def test_transformer_encoder_layer_fwd_invalid_mask_dtype(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(embed_dim, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)
    mask = torch.zeros(2, 8, 16, dtype=torch.int32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected mask dtype to be bool or floating-point, got int32""",
        gpu="""Mask Type should be defined""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
          mask,
      )

  def test_ones_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""empty(): TorchTPU does not yet support dtype int1""",
        gpu=""""fill_cuda" not implemented for 'Int1'""",
        message_reviewed_by="wan",
    ):
      torch.ones(2, 3, device=et.device(), dtype=torch.int1)

  def test_add_out_device_mismatch(self):
    a = torch.ones(4, device=et.device())
    b = torch.ones(4, device=et.device())
    out = torch.empty(4, device="cpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""add(): expected output tensor to be on 'tpu' device, got 'cpu'""",
        gpu="""Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!""",
        message_reviewed_by="wan",
    ):
      torch.add(a, b, alpha=2, out=out)

  def test_add_out_cpu_tensor(self):
    # This test exercises the BinaryOp path (`BinaryOpOut`), whereas
    # `test_add_out_device_mismatch` passes `alpha` and exercises the
    # TernaryOp path (`TernaryOpOut`).
    tpu_tensor1 = torch.tensor([1, 2, 3], device=et.device())
    tpu_tensor2 = torch.tensor([4, 5, 6], device=et.device())
    cpu_tensor = torch.tensor([7, 8, 9], device="cpu")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""add(): the out tensor is expected to be on tpu, got cpu""",
        gpu="""Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!""",
        message_reviewed_by="wan",
    ):
      torch.add(tpu_tensor1, tpu_tensor2, out=cpu_tensor)

  def test_add_dev_and_cpu_tensors(self):
    dev_tensor = torch.tensor([1, 2, 3], device=et.device())
    cpu_tensor = torch.tensor([4, 5, 6], device="cpu")
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""Expected all tensors to be on the same device, but found at least two devices, tpu:0 and cpu!""",
        gpu="""Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!""",
        message_reviewed_by="wan",
    ):
      _ = dev_tensor + cpu_tensor  # pylint: disable=pointless-statement

  def test_add_cpu_and_dev_tensors(self):
    cpu_tensor = torch.tensor([4, 5, 6], device="cpu")
    tpu_tensor = torch.tensor([1, 2, 3], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""Expected all tensors to be on the same device, but found at least two devices, tpu:0 and cpu!""",
        gpu="""Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!""",
        message_reviewed_by="wan",
    ):
      _ = cpu_tensor + tpu_tensor  # pylint: disable=pointless-statement

  def test_is_nonzero_with_more_than_one_value(self):
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""Boolean value of Tensor with more than one value is ambiguous""",
        gpu="""Boolean value of Tensor with more than one value is ambiguous""",
        message_reviewed_by="wan",
    ):
      torch.is_nonzero(torch.tensor([1, 3, 5], device=et.device()))

  def test_is_nonzero_with_empty_tensor(self):
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""Boolean value of Tensor with no values is ambiguous""",
        gpu="""Boolean value of Tensor with no values is ambiguous""",
        message_reviewed_by="wan",
    ):
      torch.is_nonzero(torch.tensor([], device=et.device()))

  def test_is_nonzero_with_nested_empty_tensor(self):
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""Boolean value of Tensor with no values is ambiguous""",
        gpu="""Boolean value of Tensor with no values is ambiguous""",
        message_reviewed_by="wan",
    ):
      torch.is_nonzero(torch.tensor([[]], device=et.device()))

  def test_masked_select_cpu_out_tensor(self):
    """Masked select function fails when self and out have different devices."""
    t = torch.ones(5, device=et.device(), dtype=torch.float32)
    mask = torch.ones(5, device=et.device(), dtype=torch.bool)
    out = torch.ones(5, device="cpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_select(): expected out tensor to be on tpu, got cpu""",
        gpu="""Expected all tensors to be on the same device, but got self is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_out_masked_select_out)""",
        message_reviewed_by="wan",
    ):
      torch.masked_select(t, mask, out=out)

  def test_masked_select_cpu_tensors(self):
    """Masked select function fails when self and out have different devices."""
    t = torch.ones(5, device="cpu", dtype=torch.float32)
    mask = torch.ones(5, device="cpu", dtype=torch.bool)
    out = torch.ones(5, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_select(): tensor is expected to be on tpu, got cpu""",
        gpu="""Expected all tensors to be on the same device, but got self is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_out_masked_select_out)""",
        message_reviewed_by="wan",
    ):
      torch.masked_select(t, mask, out=out)

  def test_cumsum_bool_dtype(self):
    """Tests that cumsum with a boolean out tensor throws the correct NotImplementedError."""
    err_type = RuntimeError if et.is_on_tpu() else NotImplementedError

    with et.assert_raises_message(
        err_type,
        tpu="""cumsum(): invalid output dtype bool""",
        gpu=""""cumsum_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      t = torch.ones(2, 2, device=et.device())
      torch.cumsum(t, dim=1, dtype=torch.bool)

  def test_cumsum_with_bool_tensor(self):
    err_type = RuntimeError if et.is_on_tpu() else NotImplementedError

    t = torch.tensor([True, False], dtype=torch.bool, device=et.device())

    # Test out-of-place with out=
    with et.assert_raises_message(
        err_type,
        tpu="""cumsum(): invalid output dtype bool""",
        gpu=""""cumsum_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      torch.cumsum(t, dim=0, out=t)

    # Test in-place
    with et.assert_raises_message(
        err_type,
        tpu="""cumsum(): invalid output dtype bool""",
        gpu=""""cumsum_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      t.cumsum_(dim=0)

  def test_cummin_with_unsupported_complex_dtype(self):
    """Tests cummin fails with unsupported complex dtypes."""
    with et.assert_raises_message(
        RuntimeError,
        tpu="""cummin_helper(): expected supported element type, got ComplexFloat""",
        gpu=""""cummin_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      t = torch.ones(2, 2, device=et.device(), dtype=torch.complex64)
      torch.cummin(t, dim=1)

  def test_logcumsumexp_with_unsupported_integer_dtype(self):
    """Tests logcumsumexp rejects non-floating-point inputs.

    Only the functional path is exercised. The out= path decomposes through the
    functional _logcumsumexp under functionalization, so AtenLogcumsumexpOut's
    identical check is unreachable (marked ERROR_COV_INFEASIBLE in the kernel).
    """
    t = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""logcumsumexp(): expected the input dtype to be floating point, got int32""",
        gpu=""""logcumsumexp_cuda" not implemented for 'Int'""",
        message_reviewed_by="wan",
    ):
      torch.logcumsumexp(t, dim=1)

  def test_logcumsumexp_with_unsupported_bool_dtype(self):
    """Tests logcumsumexp rejects bool inputs with expected error."""
    t = torch.tensor([True, False], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""logcumsumexp(): expected the input dtype to be floating point, got bool""",
        gpu=""""logcumsumexp_cuda" not implemented for 'Bool'""",
        message_reviewed_by="adivinpatel",
    ):
      torch.logcumsumexp(t, dim=0)

  def test_unsafe_masked_index_error(self):
    with et.assert_raises_message(
        IndexError,
        tpu="""at least one index must be provided""",
        gpu="""at least one index must be provided""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._unsafe_masked_index(
          torch.tensor([1.0], device=et.device()),
          torch.tensor([True], device=et.device()),
          [],  # indices
          0.0,  # fill
      )

  def test_threshold_unsupported_bool_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""threshold(): expected the input dtype to be non-bool and non-complex, got bool""",
        gpu=""""threshold_cuda" not implemented for 'Bool'""",
        message_reviewed_by="wan",
    ):
      t = torch.tensor([True, False], device=et.device())
      torch.threshold(t, 0.5, 0.0)

  def test_threshold_unsupported_complex_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""threshold(): expected the input dtype to be non-bool and non-complex, got complex64""",
        gpu=""""threshold_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      t = torch.tensor([1 + 1j, 2 + 2j], device=et.device())
      torch.threshold(t, 0.5, 0.0)

  def test_aminmax_invalid_output_device(self):
    tensor = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = _get_aminmax_outputs(torch.aminmax, device="cpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""aminmax(): expected output tensor to be on tpu, got cpu""",
        gpu="""Expected out tensor to have device cuda:0, but got cpu instead""",
        message_reviewed_by="wan",
    ):
      torch.aminmax(tensor, dim=0, out=out)

  def test_amin_invalid_output_device(self):
    tensor = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = _get_aminmax_outputs(torch.amin, device="cpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""amin(): expected output tensor to be on tpu, got cpu""",
        gpu="""Expected all tensors to be on the same device, but got self is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_amin_out_out)""",
        message_reviewed_by="wan",
    ):
      torch.amin(tensor, dim=0, out=out)

  def test_amax_invalid_output_device(self):
    tensor = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = _get_aminmax_outputs(torch.amax, device="cpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""amax(): expected output tensor to be on tpu, got cpu""",
        gpu="""Expected all tensors to be on the same device, but got self is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_amax_out_out)""",
        message_reviewed_by="wan",
    ):
      torch.amax(tensor, dim=0, out=out)

  def test_argmin_invalid_output_device(self):
    tensor = torch.ones(5, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(5, 1, device="cpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""argmin(): expected output tensor to be on tpu, got cpu""",
        gpu="""Expected all tensors to be on the same device, but got self is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_argmin_out_out)""",
        message_reviewed_by="wan",
    ):
      torch.argmin(tensor, dim=0, out=out)

  def test_argmax_invalid_output_device(self):
    tensor = torch.ones(5, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(5, 1, device="cpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""argmax(): expected output tensor to be on tpu, got cpu""",
        gpu="""Expected all tensors to be on the same device, but got self is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_argmax_out_out)""",
        message_reviewed_by="wan",
    ):
      torch.argmax(tensor, dim=0, out=out)

  def test_lu_solve_pivots_rank_too_low(self):
    err_type = RuntimeError if et.is_on_tpu() else IndexError

    lu = torch.ones(4, 4, device=et.device())
    pivots = torch.tensor(0, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, 4, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        err_type,
        tpu="""linalg_lu_solve(): expected pivots to have at least 1 dimension, got 0""",
        gpu="""Dimension specified as -1 but tensor has no dimensions""",
        message_reviewed_by="wan",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_unpack_pivots_rank_too_low(self):
    err_type = RuntimeError if et.is_on_tpu() else ValueError

    lu_data = torch.ones(4, 4, device=et.device())
    lu_pivots = torch.tensor(0, device=et.device(), dtype=torch.int32)

    # Call the out overload of lu_unpack() op.
    out = _make_lu_unpack_outputs(p=(4, 4), l=(4, 4), u=(4, 4))

    with et.assert_raises_message(
        err_type,
        tpu="""lu_unpack(): expected lu_pivots to have at least 1 dimension, got 0""",
        gpu="""Expected LU_pivots to have shape [4] but got [] instead.""",
        message_reviewed_by="wan",
    ):
      torch.lu_unpack(lu_data, lu_pivots, out=out)

  def test_grid_sampler2d_invalid_padding_mode(self):
    err_type = RuntimeError if et.is_on_tpu() else NotImplementedError
    inp, grid = _get_grid_sampler_inputs(dims=2)

    with et.assert_raises_message(
        err_type,
        tpu="""grid_sampler_2d(): expected the padding mode to be 0 (zeros), 1 (border), or 2 (reflection), got 3""",
        gpu=""""grid_sampler_2d_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      padding_mode = 3
      torch.grid_sampler_2d(inp, grid, 0, padding_mode, False)

  def test_grid_sampler3d_invalid_padding_mode(self):
    err_type = RuntimeError if et.is_on_tpu() else NotImplementedError
    inp, grid = _get_grid_sampler_inputs(dims=3)

    with et.assert_raises_message(
        err_type,
        tpu="""grid_sampler_3d(): expected the padding mode to be 0 (zeros), 1 (border), or 2 (reflection), got 3""",
        gpu=""""grid_sampler_3d_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="wan",
    ):
      padding_mode = 3
      torch.grid_sampler_3d(inp, grid, 0, padding_mode, False)

  def test_grid_sampler_3d_invalid_interpolation_mode(self):
    inp = torch.ones(1, 1, 2, 2, 2, device=et.device())
    grid = torch.zeros(1, 2, 2, 2, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_3d(): expected the interpolation mode to be 0 (bilinear) or 1 (nearest), got 2""",
        gpu="""grid_sampler(): bicubic interpolation only supports 4D input""",
        message_reviewed_by="wan",
    ):
      interpolation_mode = 2
      torch.grid_sampler(inp, grid, interpolation_mode, 0, False)

  def test_grid_sampler_3d_backward_invalid_interpolation_mode(self):
    grad_output = torch.randn(
        2, 3, 5, 5, 5, device=et.device(), dtype=torch.float32
    )
    inp_backward = torch.randn(
        2, 3, 4, 4, 4, device=et.device(), dtype=torch.float32
    )
    grid_backward = torch.randn(
        2, 5, 5, 5, 3, device=et.device(), dtype=torch.float32
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_3d_backward(): expected the interpolation mode to be 0 (bilinear) or 1 (nearest), got 2""",
        gpu="""grid_sampler(): bicubic interpolation only supports 4D input""",
        message_reviewed_by="wan",
    ):
      interpolation_mode = 2
      torch.ops.aten.grid_sampler_3d_backward(
          grad_output,
          inp_backward,
          grid_backward,
          interpolation_mode,
          0,
          False,
          [True, True],
      )

  def test_threshold_backward_unsupported_dtype_complex(self):
    grad_output = torch.ones(2, device=et.device())
    self_tensor = torch.tensor([1 + 1j, 2 + 2j], device=et.device())

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""threshold_backward(): expected the input dtype to be non-bool and non-complex, got complex64""",
        gpu=""""threshold_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.threshold_backward(grad_output, self_tensor, 0.5)

  def test_acos_cpu_out_tensor(self):
    t = torch.ones(5, device=et.device())
    out = torch.ones(5, device="cpu")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""acos(): expected the output tensor to be on tpu, got cpu""",
        gpu="""Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!""",
        message_reviewed_by="wan",
    ):
      torch.acos(t, out=out)

  def test_mm_dtype_outdtype_mismatch(self):
    lhs = torch.ones(3, 4, device=et.device(), dtype=torch.int32)
    rhs = torch.ones(4, 5, device=et.device(), dtype=torch.int32)
    out = torch.ones(3, 5, device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected out_dtype to be the same as the dtype of the out tensor, got out_dtype int32 vs out tensor dtype int64""",
        gpu="""out_dtype must be the same as the dtype of the provided out tensor""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.mm.dtype_out(lhs, rhs, out_dtype=torch.int32, out=out)

  def test_prod_cpu_out_tensor(self):
    """Tests prod() with a TPU input and CPU out tensor."""
    t = torch.tensor([1.0, 2.0, 3.0], device=et.device())
    out = torch.empty(1, device="cpu")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""prod(): expected the output tensor to be on 'tpu', got 'cpu'""",
        gpu="""Expected out tensor to have device cuda:0, but got cpu instead""",
        message_reviewed_by="wan",
    ):
      torch.prod(t, dim=0, out=out)

  def test_fused_adamw_default_int32_dtype(self):
    device = et.device()
    p_int = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g_int = torch.tensor([1, 1], dtype=torch.int32, device=device)
    ea_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    eas_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adamw_(): expected the input dtype to be floating-point, got int32""",
        gpu=""""fused_adamw_kernel_cuda" not implemented for 'Int'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.default(
          [p_int],
          [g_int],
          [ea_int],
          [eas_int],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-08,
          amsgrad=False,
          maximize=False,
      )

  def test_fused_adamw_default_complex64_dtype(self):
    device = et.device()
    p_cplx = torch.tensor([1.0 + 2j], dtype=torch.complex64, device=device)
    g_cplx = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    ea_cplx = torch.tensor([0.0 + 0j], dtype=torch.complex64, device=device)
    eas_cplx = torch.tensor([0.0 + 0j], dtype=torch.complex64, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adamw_(): expected the input dtype to be floating-point, got complex64""",
        gpu=""""fused_adamw_kernel_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.default(
          [p_cplx],
          [g_cplx],
          [ea_cplx],
          [eas_cplx],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-08,
          amsgrad=False,
          maximize=False,
      )

  def test_fused_adamw_tensor_lr_int32_dtype(self):
    device = et.device()
    p_int = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g_int = torch.tensor([1, 1], dtype=torch.int32, device=device)
    ea_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    eas_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)
    lr_tensor = torch.tensor(0.001, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adamw_(): expected the input dtype to be floating-point, got int32""",
        gpu=""""fused_adamw_kernel_cuda" not implemented for 'Int'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.tensor_lr(
          [p_int],
          [g_int],
          [ea_int],
          [eas_int],
          [],
          [step],
          lr=lr_tensor,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-08,
          amsgrad=False,
          maximize=False,
      )

  def test_fused_adamw_tensor_lr_complex64_dtype(self):
    device = et.device()
    p_cplx = torch.tensor([1.0 + 2j], dtype=torch.complex64, device=device)
    g_cplx = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    ea_cplx = torch.tensor([0.0 + 0j], dtype=torch.complex64, device=device)
    eas_cplx = torch.tensor([0.0 + 0j], dtype=torch.complex64, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)
    lr_tensor = torch.tensor(0.001, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adamw_(): expected the input dtype to be floating-point, got complex64""",
        gpu=""""fused_adamw_kernel_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.tensor_lr(
          [p_cplx],
          [g_cplx],
          [ea_cplx],
          [eas_cplx],
          [],
          [step],
          lr=lr_tensor,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-08,
          amsgrad=False,
          maximize=False,
      )

  def test_fused_adam_default_int32_dtype(self):
    device = et.device()
    p_int = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g_int = torch.tensor([1, 1], dtype=torch.int32, device=device)
    ea_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    eas_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adam_(): expected the input dtype to be floating-point, got int32""",
        gpu=""""fused_adam_kernel_cuda" not implemented for 'Int'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.default(
          [p_int],
          [g_int],
          [ea_int],
          [eas_int],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-08,
          amsgrad=False,
          maximize=False,
      )

  def test_fused_adam_default_complex64_dtype(self):
    device = et.device()
    p_cplx = torch.tensor([1.0 + 2j], dtype=torch.complex64, device=device)
    g_cplx = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    ea_cplx = torch.tensor([0.0 + 0j], dtype=torch.complex64, device=device)
    eas_cplx = torch.tensor([0.0 + 0j], dtype=torch.complex64, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adam_(): expected the input dtype to be floating-point, got complex64""",
        gpu=""""fused_adam_kernel_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.default(
          [p_cplx],
          [g_cplx],
          [ea_cplx],
          [eas_cplx],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-08,
          amsgrad=False,
          maximize=False,
      )

  def test_fused_adam_tensor_lr_int32_dtype(self):
    device = et.device()
    p_int = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g_int = torch.tensor([1, 1], dtype=torch.int32, device=device)
    ea_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    eas_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)
    lr_tensor = torch.tensor(0.001, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adam_(): expected the input dtype to be floating-point, got int32""",
        gpu=""""fused_adam_kernel_cuda" not implemented for 'Int'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.tensor_lr(
          [p_int],
          [g_int],
          [ea_int],
          [eas_int],
          [],
          [step],
          lr=lr_tensor,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-08,
          amsgrad=False,
          maximize=False,
      )

  def test_fused_adam_tensor_lr_complex64_dtype(self):
    device = et.device()
    p_cplx = torch.tensor([1.0 + 2j], dtype=torch.complex64, device=device)
    g_cplx = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    ea_cplx = torch.tensor([0.0 + 0j], dtype=torch.complex64, device=device)
    eas_cplx = torch.tensor([0.0 + 0j], dtype=torch.complex64, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)
    lr_tensor = torch.tensor(0.001, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adam_(): expected the input dtype to be floating-point, got complex64""",
        gpu=""""fused_adam_kernel_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.tensor_lr(
          [p_cplx],
          [g_cplx],
          [ea_cplx],
          [eas_cplx],
          [],
          [step],
          lr=lr_tensor,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-08,
          amsgrad=False,
          maximize=False,
      )

  def test_fused_sgd_default_int32_dtype(self):
    device = et.device()
    p = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g = torch.tensor([1, 2], dtype=torch.int32, device=device)
    mb = torch.tensor([1, 2], dtype=torch.int32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_sgd_(): expected the input dtype to be floating-point, got int32""",
        gpu=""""fused_sgd_with_momentum_kernel_cuda" not implemented for 'Int'""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_sgd_.default(
          [p],
          [g],
          [mb],
          weight_decay=0.01,
          momentum=0.9,
          lr=0.1,
          dampening=0.0,
          nesterov=False,
          maximize=False,
          is_first_step=False,
      )

  def test_fused_sgd_default_complex64_dtype(self):
    device = et.device()
    p = torch.tensor([1.0 + 2j], dtype=torch.complex64, device=device)
    g = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    mb = torch.tensor([0.0 + 0j], dtype=torch.complex64, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_sgd_(): expected the input dtype to be floating-point, got complex64""",
        gpu=""""fused_sgd_with_momentum_kernel_cuda" not implemented for 'ComplexFloat'""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_sgd_.default(
          [p],
          [g],
          [mb],
          weight_decay=0.01,
          momentum=0.9,
          lr=0.1,
          dampening=0.0,
          nesterov=False,
          maximize=False,
          is_first_step=False,
      )

  def test_fused_sgd_tensor_lr_int32_dtype(self):
    device = et.device()
    p = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g = torch.tensor([1, 2], dtype=torch.int32, device=device)
    mb = torch.tensor([1, 2], dtype=torch.int32, device=device)
    lr = torch.tensor(0.1, dtype=torch.float32, device=device)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_sgd_(): expected the input dtype to be floating-point, got int32""",
        gpu=""""fused_sgd_with_momentum_kernel_cuda" not implemented for 'Int'""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_sgd_.tensor_lr(
          [p],
          [g],
          [mb],
          weight_decay=0.01,
          momentum=0.9,
          lr=lr,
          dampening=0.0,
          nesterov=False,
          maximize=False,
          is_first_step=False,
      )

  def test_scaled_mm_v2_invalid_contraction_dim_size(self):
    err_type = RuntimeError if et.is_on_tpu() else ValueError

    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()

    with et.assert_raises_message(
        err_type,
        tpu="""scaled_mm_v2(): expected contraction_dim list to contain exactly 2 elements, got 1""",
        gpu="""contraction_dim must have exactly 2 elements""",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
          contraction_dim=[1],
      )

  def test_current_stream_device_index_above_max_devices(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Device index 8 is out of bounds of the maximum device count 8""",
        gpu="""Device index value 8 is out of index range [0, 1)""",
    ):
      torch.accelerator.current_stream(f"{et.device().type}:8")

  @et.skip_if(
      lambda: et.is_on_tpu() and torch.accelerator.device_count() >= 8,
      "Error is raised only if the number of TPUs is < 8 due to TorchTPU"
      " hardcoded maximum device count: 8 TPUs.",
  )
  def test_current_stream_device_index_above_device_count(self):
    actual_device_count = torch.accelerator.device_count()

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""Device index {actual_device_count} is out of bounds of the number of addressable devices {actual_device_count}""",
        gpu=f"""Device index value {actual_device_count} is out of index range [0, {actual_device_count})""",
    ):
      torch.accelerator.current_stream(
          torch.device(et.device().type, actual_device_count)
      )

  @et.skip_if(
      lambda: et.is_on_tpu() and torch.accelerator.device_count() < 8,
      "Error is raised only if the number of TPUs is >= 8 due to TorchTPU"
      " hardcoded maximum device count: 8 TPUs.",
  )
  def test_current_stream_device_index_above_max_device_count(self):
    actual_device_count = torch.accelerator.device_count()

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""Device index {actual_device_count} is out of bounds of the maximum device count 8""",
        gpu=f"""Device index value {actual_device_count} is out of index range [0, {actual_device_count})""",
    ):
      torch.accelerator.current_stream(
          torch.device(et.device().type, actual_device_count)
      )

  def test_safe_wrap_dim_tensor_with_invalid_dim(self):
    """Tests the dimension wrapping error check for invalid dimension."""
    t = torch.ones(1, device=et.device())
    out = torch.empty_like(t)

    with et.assert_raises_message(
        IndexError,
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""cumsum(): expected dimension to be in range [-1, 0], got 1""",
    ):
      torch.cumsum(t, dim=1, out=out)

  def test_safe_wrap_dim_scalar_with_invalid_dim(self):
    """Tests the dimension wrapping error check for invalid dimension of scalars.

    Specifically, cases where we are trying to wrap a given dimension, and the
    dimensions' upper bound, usually given by `tensor.dim()` is 0 (i.e. a
    scalar).
    """
    scalar = torch.tensor(42, device=et.device())

    with et.assert_raises_message(
        IndexError,
        gpu="""Dimension specified as -2 but tensor has no dimensions""",
        tpu="""unique_consecutive(): expected dimension to be either 0 or -1 for a shape with 0 dimensions, got -2""",
    ):
      torch.unique_consecutive(scalar, dim=-2)

  @parameterized.named_parameters(
      ("u_mismatch", "u"),
      ("s_mismatch", "s"),
      ("vh_mismatch", "vh"),
  )
  def test_linalg_svd_out_dtype_mismatch(self, mismatch_arg):
    u_dtype = torch.float32 if mismatch_arg == "u" else torch.float64
    s_dtype = torch.float32 if mismatch_arg == "s" else torch.float64
    vh_dtype = torch.float32 if mismatch_arg == "vh" else torch.float64

    u = torch.empty(5, 3, device=et.device(), dtype=u_dtype)
    s = torch.empty(3, device=et.device(), dtype=s_dtype)
    vh = torch.empty(3, 3, device=et.device(), dtype=vh_dtype)

    t = torch.randn(5, 3, device=et.device(), dtype=torch.float64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(
            r".*expected out tensor to have dtype.*", re.IGNORECASE | re.DOTALL
        ),
    ):
      torch.linalg.svd(t, full_matrices=False, out=(u, s, vh))

  def test_min_unary_invalid_output_device(self):
    tensor = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = torch.tensor(0.0, device="cpu", dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""min(): expected output tensor to be on tpu, got cpu""",
        gpu="""Expected self.device() == out.device() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        message_reviewed_by="gunhyun",
    ):
      torch.min(tensor, out=out)

  def test_max_unary_invalid_output_device(self):
    tensor = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = torch.tensor(0.0, device="cpu", dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""max(): expected output tensor to be on tpu, got cpu""",
        gpu="""Expected self.device() == out.device() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)""",
        message_reviewed_by="gunhyun",
    ):
      torch.max(tensor, out=out)

  @et.skip_if(
      lambda: torch.accelerator.device_count() > 8,
      "Only works when there are up to 8 accelerator devices.",
  )
  def test_create_stream_with_out_of_bounds_device_index(self):
    Stream = torch.cuda.Stream if et.is_on_gpu() else torch.tpu.Stream

    with et.assert_raises_message(
        RuntimeError,
        tpu="""expected device index to be in range [0, 8), got 8""",
        gpu="""CUDA error: invalid device ordinal
GPU device may be out of range, do you have enough GPUs?
Device-side assertion tracking was not enabled by user.""",
        message_reviewed_by="wan",
    ):
      Stream(8)

  def test_native_multi_head_attention_dtypes_mismatch(self):
    query = torch.ones(2, 4, 8, device=et.device(), dtype=torch.float32)
    key = torch.ones(2, 4, 8, device=et.device(), dtype=torch.float16)
    value = torch.ones(2, 4, 8, device=et.device(), dtype=torch.float32)
    qkv_weight = torch.ones(24, 8, device=et.device(), dtype=torch.float32)
    qkv_bias = torch.ones(24, device=et.device(), dtype=torch.float32)
    proj_weight = torch.ones(8, 8, device=et.device(), dtype=torch.float32)
    proj_bias = torch.ones(8, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected query, key, value, qkv_weight, qkv_bias, proj_weight, and proj_bias to have matching dtypes, got float32, float16, float32, float32, float32, float32, float32""",
        gpu="""expected mat1 and mat2 to have the same dtype, but got: c10::Half != float""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_native_multi_head_attention_non_positive_embed_dim(self):
    query = torch.ones(2, 4, 8, device=et.device(), dtype=torch.float32)
    key = torch.ones(2, 4, 8, device=et.device(), dtype=torch.float32)
    value = torch.ones(2, 4, 8, device=et.device(), dtype=torch.float32)
    qkv_weight = torch.ones(24, 8, device=et.device(), dtype=torch.float32)
    qkv_bias = torch.ones(24, device=et.device(), dtype=torch.float32)
    proj_weight = torch.ones(8, 8, device=et.device(), dtype=torch.float32)
    proj_bias = torch.ones(8, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected embed_dim to be > 0, got 0""",
        gpu="""passed-in embed_dim 0 didn't match last dim of query 8""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 0, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  def test_jagged_offsets_multiple_lists(self):
    values = torch.randn(5, 4, device=et.device())
    o1 = torch.tensor([0, 2, 5], dtype=torch.int64, device=et.device())
    o2 = torch.tensor([0, 3, 5], dtype=torch.int64, device=et.device())
    err_type = NotImplementedError if et.is_on_tpu() else RuntimeError
    with et.assert_raises_message(
        err_type,
        tpu="""jagged_to_padded_dense_forward(): expected only 1 offset tensor (only 1 jagged dim is supported for now), got 2 offset tensors""",
        gpu="""max_lengths.size(), 1 != num_jagged_dim, 2""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(values, [o1, o2], [3], 0.0)

  def test_jagged_offsets_empty(self):
    values = torch.randn(5, 4, device=et.device())
    offsets = torch.tensor([], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""jagged_to_padded_dense_forward(): expected offsets tensors to have size >= 1, got 0""",
        gpu="""Trying to create tensor with negative dimension -1: [-1, 3, 4]""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [3], 0.0
      )

  def test_leaky_relu_backward_negative_slope_with_self_is_result(self):
    grad_output = torch.ones(2, device=et.device())
    self_or_result = torch.ones(2, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""In-place leakyReLu backward calculation is triggered with a negative slope which is not supported. This is caused by calling in-place forward function with a negative slope, please call out-of-place version instead. File an issue at https://github.com/pytorch/pytorch if you do require supporting in-place leakRelu backward calculation with negative slope""",
        gpu="""In-place leakyReLu backward calculation is triggered with a negative slope which is not supported. This is caused by calling in-place forward function with a negative slope, please call out-of-place version instead. File an issue at https://github.com/pytorch/pytorch if you do require supporting in-place leakRelu backward calculation with negative slope""",
    ):
      torch.ops.aten.leaky_relu_backward(
          grad_output, self_or_result, negative_slope=-1.0, self_is_result=True
      )

  def test_jagged_to_padded_multiple_max_lengths(self):
    values = torch.randn(5, 4, device=et.device())
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""jagged_to_padded_dense_forward(): expected max_lengths to have only 1 element, got 2""",
        gpu="""max_lengths.size(), 2 != num_jagged_dim, 1""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [3, 4], 0.0
      )

  def test_jagged_to_padded_negative_max_length(self):
    values = torch.randn(5, 4, device=et.device())
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""jagged_to_padded_dense_forward(): expected max_lengths[0] to be >= 0, got -1""",
        gpu="""Trying to create tensor with negative dimension -1: [2, -1, 4]""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [-1], 0.0
      )

  def test_padded_to_jagged_dense_1d(self):
    dense = torch.randn(10, device=et.device())
    offsets = torch.tensor([0, 5, 10], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""padded_dense_to_jagged_forward(): expected dense to have >= 2 dimensions, got 1""",
        gpu="""x_offsets.size(), 1 != num_jagged_dim, -1""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._padded_dense_to_jagged_forward(dense, [offsets], 10)

  def test_padded_to_jagged_batch_size_mismatch(self):
    dense = torch.randn(2, 5, 4, device=et.device())
    offsets = torch.tensor([0, 2, 3, 5], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""padded_dense_to_jagged_forward(): expected dense first dimension size to match the batch size inferred from the offsets (3), got 2""",
        gpu="""outer_dense_size, 2 != offsets[0].numel() - 1, 3""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._padded_dense_to_jagged_forward(dense, [offsets], 5)


class InputPreprocessingErrorTest(et.ErrorTestBase):

  def test_input_preprocessing_offsets_invalid_dtype(self):
    if et.is_on_gpu():
      self.skipTest("TPU only op")
    val = torch.tensor([0, 1], dtype=torch.int32)
    off = torch.tensor([0.0, 2.0], dtype=torch.float32)  # invalid dtype
    with et.assert_raises_message(
        RuntimeError,
        tpu="""offsets must be int32""",
    ):
      torch.ops.tpu.preprocess_sparse_dense_matmul_input(
          val,
          off,
          128,
          128,
          128,
          16,
          "sum",
          1,
          1,
          2,
          True,
          "table_0",
      )

  def test_input_preprocessing_invalid_dtype(self):
    if et.is_on_gpu():
      self.skipTest("TPU only op")
    val = torch.tensor([0.0, 1.0], dtype=torch.float32)  # invalid dtype
    off = torch.tensor([0, 2], dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""indices must be int32""",
    ):
      torch.ops.tpu.preprocess_sparse_dense_matmul_input(
          val,
          off,
          128,
          128,
          128,
          16,
          "sum",
          1,
          1,
          2,
          True,
          "table_0",
      )


class ComputeStatsErrorTest(et.ErrorTestBase):

  def test_compute_stats_indices_invalid_dtype(self):
    if et.is_on_gpu():
      self.skipTest("TPU only op")
    indices = torch.tensor([0.0, 1.0], dtype=torch.float32)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""indices must be int32""",
    ):
      torch.ops.tpu.compute_sparse_dense_matmul_stats(indices, offsets, 1, 4)

  def test_compute_stats_offsets_invalid_dtype(self):
    if et.is_on_gpu():
      self.skipTest("TPU only op")
    indices = torch.tensor([0, 1], dtype=torch.int32)
    offsets = torch.tensor([0.0, 2.0], dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""offsets must be int32""",
    ):
      torch.ops.tpu.compute_sparse_dense_matmul_stats(indices, offsets, 1, 4)

  def test_compute_stats_global_device_count_non_positive(self):
    if et.is_on_gpu():
      self.skipTest("TPU only op")
    indices = torch.tensor([0, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""global_device_count must be positive""",
    ):
      torch.ops.tpu.compute_sparse_dense_matmul_stats(indices, offsets, 0, 4)

  def test_compute_stats_num_sc_per_device_non_positive(self):
    if et.is_on_gpu():
      self.skipTest("TPU only op")
    indices = torch.tensor([0, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""num_sc_per_device must be positive""",
    ):
      torch.ops.tpu.compute_sparse_dense_matmul_stats(indices, offsets, 1, 0)

  def test_compute_stats_offsets_empty(self):
    if et.is_on_gpu():
      self.skipTest("TPU only op")
    indices = torch.tensor([], dtype=torch.int32)
    offsets = torch.tensor([], dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""offsets cannot be empty""",
    ):
      torch.ops.tpu.compute_sparse_dense_matmul_stats(indices, offsets, 1, 4)

  def test_compute_stats_offsets_range_invalid(self):
    if et.is_on_gpu():
      self.skipTest("TPU only op")
    indices = torch.tensor([1, 2, 3], dtype=torch.int32)
    offsets = torch.tensor([0, 10], dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""Invalid offsets range [0, 10) for indices size 3""",
    ):
      torch.ops.tpu.compute_sparse_dense_matmul_stats(indices, offsets, 1, 4)


class MaskedSoftmaxErrorTest(et.ErrorTestBase):

  def test_masked_softmax_non_float_input(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu=""""masked_softmax" not implemented for 'Int'""",
        tpu="""masked_softmax(): expected input to be a floating point tensor, got int32""",
    ):
      torch.ops.aten._masked_softmax(
          torch.tensor([1, 2], dtype=torch.int32, device=et.device()),
          torch.tensor([True, False], dtype=torch.bool, device=et.device()),
          dim=0,
          mask_type=2,
      )

  def test_masked_softmax_non_bool_mask(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Mask should be a boolean tensor""",
        tpu="""masked_softmax(): expected mask to be a boolean tensor, got float32""",
    ):
      torch.ops.aten._masked_softmax(
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor([1.0, 0.0], dtype=torch.float32, device=et.device()),
      )

  def test_masked_softmax_invalid_mask_type(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Mask Type should be 0 (src_mask), 1 (src_key_padding_mask), or 2 (default_mask)""",
        tpu="""masked_softmax(): expected mask_type to be 0, 1, or 2, got 5""",
    ):
      torch.ops.aten._masked_softmax(
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor([True, False], dtype=torch.bool, device=et.device()),
          dim=0,
          mask_type=5,
      )

  def test_masked_softmax_mask_type_0_shape_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Mask shape should match input. mask: [5, 5] input: [2, 4, 8, 16]""",
        tpu="""masked_softmax(): expected mask shape to be (8, 16) for mask_type 0, got [5, 5]""",
    ):
      torch.ops.aten._masked_softmax(
          torch.randn(2, 4, 8, 16, dtype=torch.float32, device=et.device()),
          torch.randint(0, 2, (5, 5), dtype=torch.bool, device=et.device()),
          dim=-1,
          mask_type=0,
      )

  def test_masked_softmax_mask_type_1_shape_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Mask shape should match input. mask: [5, 5] input: [2, 4, 8, 16]""",
        tpu="""masked_softmax(): expected mask shape to be (2, 16) for mask_type 1, got [5, 5]""",
    ):
      torch.ops.aten._masked_softmax(
          torch.randn(2, 4, 8, 16, dtype=torch.float32, device=et.device()),
          torch.randint(0, 2, (5, 5), dtype=torch.bool, device=et.device()),
          dim=-1,
          mask_type=1,
      )

  def test_masked_softmax_mask_shape_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Mask Type should be defined""",
        tpu="""masked_softmax(): expected mask shape to be [2], got [3]""",
    ):
      torch.ops.aten._masked_softmax(
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor(
              [True, False, True], dtype=torch.bool, device=et.device()
          ),
      )

  def test_masked_softmax_backward_non_float_grad(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu=""""masked_softmax_backward" not implemented for 'Int'""",
        tpu="""masked_softmax_backward(): expected grad_output to be a floating point tensor, got int32""",
    ):
      torch.ops.aten._masked_softmax_backward(
          torch.tensor([1, 2], dtype=torch.int32, device=et.device()),
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor([True, False], dtype=torch.bool, device=et.device()),
          dim=0,
      )

  def test_masked_softmax_backward_non_float_output(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""expected scalar type Float but found Int""",
        tpu="""masked_softmax_backward(): expected output to be a floating point tensor, got int32""",
    ):
      torch.ops.aten._masked_softmax_backward(
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor([1, 2], dtype=torch.int32, device=et.device()),
          torch.tensor([True, False], dtype=torch.bool, device=et.device()),
          dim=0,
      )

  def test_masked_softmax_backward_non_bool_mask(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Mask should be a boolean tensor""",
        tpu="""masked_softmax_backward(): expected mask to be a boolean tensor, got float32""",
    ):
      torch.ops.aten._masked_softmax_backward(
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor([1.0, 0.0], dtype=torch.float32, device=et.device()),
          dim=0,
      )

  def test_masked_softmax_backward_grad_output_shape_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Mask shape should match grad shape""",
        tpu="""masked_softmax_backward(): expected grad_output shape to be [2], got [3]""",
    ):
      torch.ops.aten._masked_softmax_backward(
          torch.tensor(
              [1.0, 2.0, 3.0], dtype=torch.float32, device=et.device()
          ),
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor([True, False], dtype=torch.bool, device=et.device()),
          dim=0,
      )

  def test_masked_softmax_backward_shape_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        gpu="""Mask shape should match grad shape""",
        tpu="""masked_softmax_backward(): expected mask shape to be [2], got [3]""",
    ):
      torch.ops.aten._masked_softmax_backward(
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=et.device()),
          torch.tensor(
              [True, False, True], dtype=torch.bool, device=et.device()
          ),
      )

  def test_masked_softmax_scalar_dim_out_of_bounds(self):
    with et.assert_raises_message(
        IndexError,
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""masked_softmax(): expected dimension to be in range [-1, 0], got 1""",
    ):
      torch.ops.aten._masked_softmax(
          torch.tensor(3.14, dtype=torch.float32, device=et.device()),
          torch.tensor(False, dtype=torch.bool, device=et.device()),
          dim=1,
          mask_type=2,
      )

  def test_masked_softmax_backward_scalar_dim_out_of_bounds(self):
    with et.assert_raises_message(
        IndexError,
        gpu="""Dimension out of range (expected to be in range of [-1, 0], but got 1)""",
        tpu="""masked_softmax_backward(): expected dimension to be in range [-1, 0], got 1""",
    ):
      torch.ops.aten._masked_softmax_backward(
          torch.tensor(1.0, dtype=torch.float32, device=et.device()),
          torch.tensor(3.14, dtype=torch.float32, device=et.device()),
          torch.tensor(False, dtype=torch.bool, device=et.device()),
          dim=1,
      )

  def test_histc_bfloat16_unsupported(self):
    x = torch.ones(2, 2, device=et.device(), dtype=torch.bfloat16)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""histc(): not implemented for bfloat16""",
        gpu=""""histc" not implemented for 'BFloat16'""",
        message_reviewed_by="gunhyun",
    ):
      torch.histc(x)

  def test_unique_consecutive_dim_out_of_bounds(self):
    with et.assert_raises_message(
        IndexError,
        tpu="""unique_consecutive(): expected dimension to be in range [-2, 1], got 5""",
        gpu="""Dimension out of range (expected to be in range of [-2, 1], but got 5)""",
        message_reviewed_by="gunhyun",
    ):
      torch.unique_consecutive(
          torch.ones(2, 3, device=et.device()),
          dim=5,
      )

  def test_unique_consecutive_dim_multiple_zero_dims(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""unique_consecutive(): expected at most 1 zero sized dimension when applying unique on a zero sized dimension, got 2""",
        gpu="""Number of zero sized dimensions is more than one, so unique cannot be applied """,
    ):
      torch.unique_consecutive(
          torch.empty(0, 0, 3, device=et.device()),
          dim=0,
      )

  def test_unique_consecutive_dim_unselected_zero_dim(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""unique_consecutive(): expected 0 unselected zero sized dimensions, got 1""",
        gpu="""There are 0 sized dimensions, and they aren't selected, so unique cannot be applied""",
    ):
      torch.unique_consecutive(
          torch.empty(0, 3, device=et.device()),
          dim=1,
      )

  def test_sparse_gather_meta_invalid_row_pointers_dim(self):
    row_pointers_2d = torch.tensor([[0, 8]], dtype=torch.int32, device="meta")
    indices = torch.tensor([0] * 8, dtype=torch.int32, device="meta")
    operand = torch.ones(10, 8, dtype=torch.float32, device="meta")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""expected row_pointers to be a 1D tensor, got a 2D tensor of shape [1, 2]""",
    ):
      torch.ops.tpu.sparse_gather(row_pointers_2d, indices, operand, 8)

  def test_sparse_gather_meta_invalid_indices_dim(self):
    row_pointers = torch.tensor([0, 8], dtype=torch.int32, device="meta")
    indices_2d = torch.tensor([[0] * 8], dtype=torch.int32, device="meta")
    operand = torch.ones(10, 8, dtype=torch.float32, device="meta")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""expected indices to be a 1D tensor, got a 2D tensor of shape [1, 8]""",
    ):
      torch.ops.tpu.sparse_gather(row_pointers, indices_2d, operand, 8)

  def test_sparse_gather_meta_invalid_operand_dim(self):
    row_pointers = torch.tensor([0, 8], dtype=torch.int32, device="meta")
    indices = torch.tensor([0] * 8, dtype=torch.int32, device="meta")
    operand_1d = torch.ones(10, dtype=torch.float32, device="meta")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""expected operand to be a 2D tensor, got a 1D tensor of shape [10]""",
    ):
      torch.ops.tpu.sparse_gather(row_pointers, indices, operand_1d, 8)

  def test_sparse_gather_meta_invalid_indices_length(self):
    row_pointers = torch.tensor([0, 8], dtype=torch.int32, device="meta")
    indices_wrong_len = torch.tensor([0] * 7, dtype=torch.int32, device="meta")
    operand = torch.ones(10, 8, dtype=torch.float32, device="meta")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""expected indices length to match the maximum number of non-zeroes, i.e. row_pointers length * maximum number of non-zeroes per row (2 * 8 = 16), got 7""",
    ):
      torch.ops.tpu.sparse_gather(row_pointers, indices_wrong_len, operand, 8)

  def test_pow_unsupported_dtypes(self):
    """Tests that pow with unsupported bool tensor inputs fails with expected error."""
    base = torch.tensor([True, False], dtype=torch.bool, device=et.device())
    exp = torch.tensor([True, False], dtype=torch.bool, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""pow(): expected non-boolean first argument, got bool""",
        gpu=""""pow_cuda" not implemented for 'Bool'""",
    ):
      torch.pow(base, exp)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
