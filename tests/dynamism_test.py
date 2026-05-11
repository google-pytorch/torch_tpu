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

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import dynamism
from torch_tpu._internal import execution_mode
from torch_tpu._internal.utils import utils
from tests import op_testing


# TODO: this is not concurrency safe, causes flaky test failures if there is
# more than one test case in this file that uses this.
# This is because the context windows for parallel-executing tests may overlap,
# causing tests to count cache events from other tests.
class CompilationCounter:
  """A context manager to count cache requests and hits."""

  def __init__(self, device):
    self.device = device
    self.closed = False
    self.compilations = 0
    self.cache_hits = 0

  def __enter__(self):
    start_cache_stats = getattr(torch, str(self.device))._get_cache_stats()
    self.start_reqs = start_cache_stats.num_cache_reqs
    self.start_hits = start_cache_stats.num_cache_hits
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    end_cache_stats = getattr(torch, str(self.device))._get_cache_stats()
    self.closed = True
    end_reqs = end_cache_stats.num_cache_reqs
    end_hits = end_cache_stats.num_cache_hits
    self.cache_hits = end_hits - self.start_hits
    self.compilation_events = end_reqs - self.start_reqs - self.cache_hits

  def num_compile_events(self):
    assert self.closed, "can only query compilations once context is closed"
    return self.compilation_events

  def num_cache_hits(self):
    assert self.closed, "can only query compilations once context is closed"
    return self.cache_hits


class DynamismApiTest(absltest.TestCase):
  """Tests for dynamism module APIs `get_dynamism_info` and `mark_dynamic`."""

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")
    # Dynamism in eager mode works only when EagerMode.DEFER_AND_FUSE is
    # selected.
    self.old_eager_mode = execution_mode.get_eager_mode()
    execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE)

  def tearDown(self):
    execution_mode.set_eager_mode(self.old_eager_mode)
    super().tearDown()

  def test_mark_get_dynamism_info(self):
    x = torch.ones(10, device=self.device)
    y = torch.ones(10, device=self.device)
    z = x + y

    # Nothing has been marked dynamic.
    for tensor in [x, y, z]:
      self.assertEmpty(dynamism.get_dynamism_info(tensor))

    dynamism.mark_dynamic(x, 0, 2, 10)

    # x has been marked dynamic, but y and z have not.
    x_dynamism_info = dynamism.get_dynamism_info(x)
    self.assertLen(x_dynamism_info, 1)
    self.assertEqual(x_dynamism_info[0].dimension, 0)
    self.assertEqual(x_dynamism_info[0].lower_bound, 2)
    self.assertEqual(x_dynamism_info[0].upper_bound, 10)
    self.assertEmpty(dynamism.get_dynamism_info(y))
    self.assertEmpty(dynamism.get_dynamism_info(z))

  def test_mark_dynamic_invalid_dim(self):
    x = torch.ones(10, device=self.device)
    with self.assertRaisesRegex(RuntimeError, "dimension -1 is out of bounds"):
      dynamism.mark_dynamic(x, -1, 2, 10)

  def test_mark_dynamic_invalid_bounds(self):
    x = torch.ones(10, device=self.device)
    with self.assertRaisesRegex(RuntimeError, "but the dimension size is 10"):
      dynamism.mark_dynamic(x, 0, 2, 9)

  def test_mark_dynamic_multiple_dims_fails(self):
    x = torch.ones(10, 10, device=self.device)
    dynamism.mark_dynamic(x, 0, 2, 10)
    with self.assertRaisesRegex(
        RuntimeError, "only one dynamic dimension is supported"
    ):
      dynamism.mark_dynamic(x, 1, 2, 10)

  def test_mark_dynamic_multiple_tensors(self):
    x = torch.ones(10, device=self.device)
    y = torch.ones(10, device=self.device)
    dynamism.mark_dynamic(x, 0, 2, 10)
    dynamism.mark_dynamic(y, 0, 5, 10)

    x_dynamism_info = dynamism.get_dynamism_info(x)
    self.assertLen(x_dynamism_info, 1)
    self.assertEqual(x_dynamism_info[0].dimension, 0)
    self.assertEqual(x_dynamism_info[0].lower_bound, 2)
    self.assertEqual(x_dynamism_info[0].upper_bound, 10)

    y_dynamism_info = dynamism.get_dynamism_info(y)
    self.assertLen(y_dynamism_info, 1)
    self.assertEqual(y_dynamism_info[0].dimension, 0)
    self.assertEqual(y_dynamism_info[0].lower_bound, 5)
    self.assertEqual(y_dynamism_info[0].upper_bound, 10)

  def test_mark_dynamic_overwrite(self):
    x = torch.ones(10, device=self.device)
    dynamism.mark_dynamic(x, 0, 2, 10)
    x_dynamism_info = dynamism.get_dynamism_info(x)
    self.assertLen(x_dynamism_info, 1)
    self.assertEqual(x_dynamism_info[0].lower_bound, 2)

    dynamism.mark_dynamic(x, 0, 5, 10)
    x_dynamism_info = dynamism.get_dynamism_info(x)
    self.assertLen(x_dynamism_info, 1)
    self.assertEqual(x_dynamism_info[0].lower_bound, 5)

  def test_mark_dynamic_view_tensor(self):
    x = torch.ones(10, device=self.device)
    y = x.view(5, 2)
    y = dynamism.mark_dynamic(y, 0, 2, 10)
    y_dynamism_info = dynamism.get_dynamism_info(y)
    self.assertLen(y_dynamism_info, 1)
    self.assertEqual(y_dynamism_info[0].lower_bound, 2)
    self.assertEqual(y_dynamism_info[0].upper_bound, 10)

  def test_mark_dynamic_view_tensor_using_reshape(self):
    x = torch.ones(1, 1024, device=self.device)
    y = x.reshape(1, 1, 1, 1024)
    y_dynamic = dynamism.mark_dynamic(y, 3, 2, 2048)
    y_dynamism_info = dynamism.get_dynamism_info(y_dynamic)
    self.assertLen(y_dynamism_info, 1)
    self.assertEqual(y_dynamism_info[0].lower_bound, 2)
    self.assertEqual(y_dynamism_info[0].upper_bound, 2048)


class DynamismTest(parameterized.TestCase):
  """Unit tests for bounded dynamism support."""

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")
    torch.tpu._clear_cache()
    # Dynamism in eager mode works only when EagerMode.DEFER_AND_FUSE is
    # selected.
    self.old_eager_mode = execution_mode.get_eager_mode()
    execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE)

  def tearDown(self):
    execution_mode.set_eager_mode(self.old_eager_mode)
    super().tearDown()

  def _run_bounded_dynamism_test(self, fn, mark_dynamic_fn, *args):
    expected = fn(*args)

    # Move args to TPU and mark dynamic
    args_tpu = [arg.to(device=self.device) for arg in args]
    mark_dynamic_fn(*args_tpu)
    act = fn(*args_tpu)

    # Compare outputs
    if isinstance(act, torch.Tensor):
      utils.assert_close(act.to("cpu"), expected, rtol=1e-5, atol=1e-5)
      return

    # Allow multiple outputs
    for act, expected in zip(act, expected):
      utils.assert_close(act.to("cpu"), expected, rtol=1e-5, atol=1e-5)

  @parameterized.product(dtype=op_testing.all_xla_supported_dtypes())
  def test_elementwise_unary_op_one_dimension_dynamic(self, dtype):
    mark_dynamic = lambda x: dynamism.mark_dynamic(x, 0, 2, 20)
    args = (torch.rand(5, 3, dtype=torch.float32).to(dtype),)
    aten_op = torch.abs if dtype != torch.bool else torch.logical_not
    self._run_bounded_dynamism_test(aten_op, mark_dynamic, *args)

  def test_elementwise_binary_op_one_dimension_dynamic(self):
    mark_dynamic = lambda x: dynamism.mark_dynamic(x, 0, 2, 20)
    args = (torch.rand(5, 3, dtype=torch.float32),)
    pow2_fn = lambda x: torch.pow(x, 2)
    self._run_bounded_dynamism_test(pow2_fn, mark_dynamic, *args)

  def test_elementwise_binary_op_simple(self):
    def mark_dynamic(x, y):
      dynamism.mark_dynamic(x, 0, 2, 10)
      dynamism.mark_dynamic(y, 0, 2, 10)

    args = (
        torch.rand(3, 1, device=self.device, dtype=torch.float32),
        torch.rand(3, 1, device=self.device, dtype=torch.float32),
    )
    self._run_bounded_dynamism_test(torch.mul, mark_dynamic, *args)

  def test_elementwise_binary_op_3d(self):
    def mark_dynamic(x, y):
      dynamism.mark_dynamic(x, 2, 2, 15)
      dynamism.mark_dynamic(y, 2, 2, 15)

    args = (
        torch.rand(5, 10, 5, device=self.device, dtype=torch.float64),
        torch.rand(5, 10, 5, device=self.device, dtype=torch.float64),
    )
    self._run_bounded_dynamism_test(torch.pow, mark_dynamic, *args)

  def test_elementwise_binary_op_with_broadcast(self):
    def mark_dynamic(x, y):
      dynamism.mark_dynamic(x, 0, 2, 10)
      dynamism.mark_dynamic(y, 1, 2, 15)

    args = (
        torch.rand(3, 1, device=self.device, dtype=torch.float32),
        torch.rand(1, 4, device=self.device, dtype=torch.float32),
    )
    self._run_bounded_dynamism_test(torch.add, mark_dynamic, *args)

  def test_concat_with_dynamic_concat_dim(self):
    def mark_dynamic(x, y):
      dynamism.mark_dynamic(x, 0, 2, 10)
      dynamism.mark_dynamic(y, 0, 2, 10)

    args = (
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )
    cat_fn = lambda x, y: torch.cat([x, y], dim=0)
    self._run_bounded_dynamism_test(cat_fn, mark_dynamic, *args)

  def test_concat_with_static_concat_dim(self):
    def mark_dynamic(x, y):
      dynamism.mark_dynamic(x, 1, 2, 10)
      dynamism.mark_dynamic(y, 1, 2, 10)

    args = (
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )
    cat_fn = lambda x, y: torch.cat([x, y], dim=0)
    self._run_bounded_dynamism_test(cat_fn, mark_dynamic, *args)

  def test_addmm_bias_1d(self):
    def mark_dynamic(biased_mat, mat1, mat2):
      del biased_mat, mat2  # Unused
      dynamism.mark_dynamic(mat1, 0, 2, 10)

    args = (
        torch.arange(4, dtype=torch.int32).reshape(4),  # self/bias
        torch.arange(15, dtype=torch.int32).reshape(3, 5),  # mat1
        torch.arange(20, dtype=torch.int32).reshape(5, 4),  # mat2
    )
    self._run_bounded_dynamism_test(torch.addmm, mark_dynamic, *args)

  def test_addmm_bias_2d(self):
    def mark_dynamic(biased_mat, mat1, mat2):
      del mat2  # Unused
      dynamism.mark_dynamic(biased_mat, 0, 2, 10)
      dynamism.mark_dynamic(mat1, 0, 2, 10)

    args = (
        torch.arange(12, dtype=torch.int32).reshape(3, 4),  # self/bias
        torch.arange(15, dtype=torch.int32).reshape(3, 5),  # mat1
        torch.arange(20, dtype=torch.int32).reshape(5, 4),  # mat2
    )
    self._run_bounded_dynamism_test(torch.addmm, mark_dynamic, *args)

  @absltest.skip("torch.sort lowering needs MakeIotaLike lowering")
  def test_sort_dynamic_sort_dim(self):
    def mark_dynamic(x):
      dynamism.mark_dynamic(x, 0, 2, 10)

    args = (torch.rand(5, 3, dtype=torch.float32),)
    self._run_bounded_dynamism_test(torch.sort, mark_dynamic, *args)

  @absltest.skip("torch.sort lowering needs MakeIotaLike lowering")
  def test_sort_static_sort_dim(self):
    def mark_dynamic(x):
      dynamism.mark_dynamic(x, 1, 2, 10)

    args = (torch.rand(5, 3, dtype=torch.float32),)
    self._run_bounded_dynamism_test(torch.sort, mark_dynamic, *args)

  @absltest.skip("CompilationCounter is flaky unless tests run sequentially.")
  def test_executable_is_reused(self):
    def dyn_pow(x):
      dynamism.mark_dynamic(x, 0, 2, 20)
      return x**2

    # Compile for x**2 where x.shape[0] < 20.
    # This should be a cache miss as we haven't compiled it yet.
    x = torch.rand(5, 3, dtype=torch.float32, device="cpu").to(self.device)
    x_res = dyn_pow(x)
    with CompilationCounter(self.device) as counter:
      print(x_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 0)

    # Call the same function with compatible bounds but different static shape.
    # This should be a cache hit.
    y = torch.rand(18, 3, dtype=torch.float32, device="cpu").to(self.device)
    y_res = dyn_pow(y)
    with CompilationCounter(self.device) as counter:
      print(y_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 1)

    # Call the same function with a compatible static shape, but not marked.
    # This should be a cache hit that uses the dynamic executable.
    z = torch.rand(11, 3, dtype=torch.float32, device="cpu").to(self.device)
    z_res = z**2
    with CompilationCounter(self.device) as counter:
      print(z_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 1)

    # Call the same function with an incompatible static shape.
    # This should be a cache miss.
    w = torch.rand(21, 3, dtype=torch.float32, device="cpu").to(self.device)
    w_res = w**2
    with CompilationCounter(self.device) as counter:
      print(w_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 0)

    # Call the same function with an already seen compatible static shape,
    # marked. This should be two cache hits, one for the padding module and one
    # for the actual computation.
    a = torch.rand(5, 3, dtype=torch.float32, device="cpu").to(self.device)
    a_res = dyn_pow(a)
    with CompilationCounter(self.device) as counter:
      print(a_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 2)

    # Call the same function with an already seen compatible static shape,
    # not marked.
    # This should be two cache hits, one for the padding module and one for the
    # actual computation.
    b = torch.rand(5, 3, dtype=torch.float32, device="cpu").to(self.device)
    b_res = b**2
    with CompilationCounter(self.device) as counter:
      print(b_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 2)

  def test_executable_is_reused_with_view_ops(self):

    def dyn_view_pow(x, do_mark=True):
      if do_mark:
        dynamism.mark_dynamic(x, 0, 2, 20)
      x = x.transpose(1, 0)
      x = x.reshape(x.shape[0] * x.shape[1])
      # Bug in bitcast lowering to fix separately:
      # x = x.view(torch.int32)  # ComplexToRealBitcast
      # x = x + 1
      # x = x.view(torch.float32)  # RealToRealBitcast
      return x**2

    # Compile for x**2 where x.shape[0] < 20.
    # This should be a cache miss as we haven't compiled it yet.
    x = torch.rand(6, 2, dtype=torch.float32, device="cpu").to(self.device)
    x_res = dyn_view_pow(x)
    with CompilationCounter(self.device) as counter:
      print(x_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 0)

    # Call the same function with compatible bounds but different static shape.
    # This should be a cache hit.
    y = torch.rand(18, 2, dtype=torch.float32, device="cpu").to(self.device)
    y_res = dyn_view_pow(y)
    with CompilationCounter(self.device) as counter:
      print(y_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 1)

    # Call the same function with a compatible static shape, but not marked.
    # This should be a cache hit that uses the dynamic executable.
    z = torch.rand(10, 2, dtype=torch.float32, device="cpu").to(self.device)
    z_res = dyn_view_pow(z, do_mark=False)
    with CompilationCounter(self.device) as counter:
      print(z_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 1)

    # Call the same function with an incompatible static shape.
    # This should be a cache miss.
    w = torch.rand(21, 2, dtype=torch.float32, device="cpu").to(self.device)
    w_res = dyn_view_pow(w, do_mark=False)
    with CompilationCounter(self.device) as counter:
      print(w_res.cpu())
    self.assertEqual(counter.num_cache_hits(), 0)

  @parameterized.product(first_dim=[6, 8], last_dim=[6, 8])
  def test_mark_dynamic_on_materialized_tensor_not_sent_to_cpu(
      self, first_dim, last_dim
  ):
    """Test for bounded dynamic value flowing between materialization points.

    Bounded dynamic outputs should be static after the bounded dynamic
    computation. Currently buffers are forwarded to the next computation as
    bounded dynamic PJRT buffers, which causes issues.

    Args:
      first_dim: The size of the first dimension of the tensor.
      last_dim: The size of the last dimension of the tensor.
    """
    x = torch.ones(first_dim, 5, last_dim, device=self.device)
    dynamism.mark_dynamic(x, 1, 2, 20)
    x = x + 1
    print(x.cpu())  # < -- OK

    # to fix, make a fresh x
    # x = x.cpu().to(self.device)

    dynamism.mark_dynamic(x, 1, 2, 20)
    x = x + 1
    print(x.cpu())  # < -- sync(x) <-- fail, x is dynamic

  def test_mean_dynamic_shapes_reduce_all(self):
    """Tests mean op with dynamic shapes reducing all dims."""

    def dyn_mean(x):
      dynamism.mark_dynamic(x, 0, 2, 10)
      return torch.mean(x)

    x = torch.randn(7, 13, device="cpu").to(self.device)
    res_x = dyn_mean(x)
    with CompilationCounter(self.device) as counter:
      res_x.cpu()
    self.assertEqual(counter.num_cache_hits(), 0)

    y = torch.randn(3, 13, device="cpu").to(self.device)
    res_y = dyn_mean(y)
    with CompilationCounter(self.device) as counter:
      res_y.cpu()
    self.assertEqual(counter.num_cache_hits(), 1)

  def test_mean_dynamic_shapes_reduce_dynamic_dim(self):
    """Tests mean op with dynamic shapes reducing the dynamic dim."""

    def dyn_mean(x):
      dynamism.mark_dynamic(x, 0, 2, 10)
      return torch.mean(x, dim=0)

    x = torch.randn(8, 17, device="cpu").to(self.device)
    res_x = dyn_mean(x)
    with CompilationCounter(self.device) as counter:
      res_x.cpu()
    self.assertEqual(counter.num_cache_hits(), 0)

    y = torch.randn(4, 17, device="cpu").to(self.device)
    res_y = dyn_mean(y)
    with CompilationCounter(self.device) as counter:
      res_y.cpu()
    self.assertEqual(counter.num_cache_hits(), 1)

  def test_mean_dynamic_shapes_reduce_static_dim(self):
    """Tests mean op with dynamic shapes reducing a static dim."""

    def dyn_mean(x):
      dynamism.mark_dynamic(x, 0, 2, 10)
      return torch.mean(x, dim=1)

    x = torch.randn(9, 19, device="cpu").to(self.device)
    res_x = dyn_mean(x)
    with CompilationCounter(self.device) as counter:
      res_x.cpu()
    self.assertEqual(counter.num_cache_hits(), 0)

    y = torch.randn(5, 19, device="cpu").to(self.device)
    res_y = dyn_mean(y)
    with CompilationCounter(self.device) as counter:
      res_y.cpu()
    self.assertEqual(counter.num_cache_hits(), 1)

  def test_same_rank_dynamic_reshape(self):
    # [1, 16, ?, 128] to [16, ?, 128, 1]
    def mark_dynamic(x: torch.Tensor) -> None:
      dynamism.mark_dynamic(x, 2, 2, 10)

    def reshape_fn(x: torch.Tensor) -> torch.Tensor:
      return x.reshape(16, x.shape[2], 128, 1)

    args = (torch.ones(1, 16, 5, 128),)
    self._run_bounded_dynamism_test(reshape_fn, mark_dynamic, *args)


if __name__ == "__main__":
  absltest.main()
