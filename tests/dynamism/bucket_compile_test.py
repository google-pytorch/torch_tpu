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

"""Test demonstrating framework-level bucketing through torch.compile()."""

from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import utils


class BucketCompileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def call_and_compare(
      self,
      func,
      test_inputs,
      buckets,
      mark_dynamic_tests_info=None,
      bucketing=True,
      dynamism=True,
  ) -> dict[str, int]:
    if mark_dynamic_tests_info is None:
      mark_dynamic_tests_info = [() for _ in test_inputs]

    # Initialize TpuBackend with requested bucketing and dynamism settings
    backend = _backend.TpuBackend(
        debug=True,
        dynamism=dynamism,
        bucketing=bucketing,
        buckets=buckets,
    )

    compiled = torch.compile(func, backend=backend)

    for inputs, per_case_mark_info in zip(test_inputs, mark_dynamic_tests_info):
      # Clone inputs to avoid side effects from in-place mutations
      eager_inputs = [
          arg.clone() if isinstance(arg, torch.Tensor) else arg
          for arg in inputs
      ]
      expected = func(*eager_inputs)

      compiled_inputs = [
          arg.clone() if isinstance(arg, torch.Tensor) else arg
          for arg in inputs
      ]
      device_inputs = [
          input.to(self.device) if isinstance(input, torch.Tensor) else input
          for input in compiled_inputs
      ]

      for input_tensor, input_mark_info in zip(
          device_inputs, per_case_mark_info
      ):
        if not isinstance(input_tensor, torch.Tensor):
          continue
        mark_dynamic, dim_infos = input_mark_info
        if mark_dynamic:
          for dim_info in dim_infos:
            dim, min_val, max_val = dim_info
            torch._dynamo.mark_dynamic(
                input_tensor, dim, min=min_val, max=max_val
            )
      actual = compiled(*device_inputs)
      # If actual is a tuple/list (e.g. from multi-output func), move all to CPU
      if isinstance(actual, (tuple, list)):
        actual_cpu = type(actual)(
            t.to("cpu") if isinstance(t, torch.Tensor) else t for t in actual
        )
      else:
        actual_cpu = (
            actual.to("cpu") if isinstance(actual, torch.Tensor) else actual
        )

      utils.assert_close(actual_cpu, expected)

    # Count compiled buckets by inspecting the compiled executables stored on
    # the backend
    compiled_buckets = 0
    for executable in backend._compiled_executables:
      if hasattr(executable, "_executables"):
        compiled_buckets += len(executable._executables)
      else:
        compiled_buckets += 1

    metrics = {"static_compile_events": compiled_buckets}
    return metrics

  def test_simple_add_bucketing(self):
    def simple(x):
      return x + 3.0

    # Bucket configuration
    buckets = [4, 8, 16]

    test_inputs = [
        (
            torch.arange(2).reshape(1, 2).float(),
        ),  # S=2 -> resolves to bucket B=4
        (
            torch.arange(3).reshape(1, 3).float(),
        ),  # S=3 -> resolves to bucket B=4 (cache HIT)
        (
            torch.arange(4).reshape(1, 4).float(),
        ),  # S=4 -> resolves to bucket B=4 (cache HIT)
        (
            torch.arange(6).reshape(1, 6).float(),
        ),  # S=6 -> resolves to bucket B=8 (cache MISS)
        (
            torch.arange(7).reshape(1, 7).float(),
        ),  # S=7 -> resolves to bucket B=8 (cache HIT)
    ]

    mark_dynamic_tests_info = [
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
    ]

    metrics = self.call_and_compare(
        simple, test_inputs, buckets, mark_dynamic_tests_info
    )

    # We expect exactly 2 static compilation calls:
    # 1. First run with S=2 (compiles for B=4)
    # 2. Fourth run with S=6 (compiles for B=8)
    # S=3, S=4, and S=7 should hit the cache!
    # Note: Since this is inference, the backward compiler is not run.
    self.assertEqual(metrics["static_compile_events"], 2)

  def test_inplace_mutation_bucketing(self):
    def inplace_add(x, state):
      state.add_(x)
      return state * 2.0

    # Bucket configuration
    buckets = [4, 8, 16]

    # Input shapes:
    # x is dynamic (1, S)
    # state is dynamic (1, S) and is mutated in-place
    test_inputs = [
        (
            torch.arange(2).reshape(1, 2).float(),
            torch.ones(1, 2).float(),
        ),  # S=2 -> B=4
        (
            torch.arange(3).reshape(1, 3).float(),
            torch.ones(1, 3).float(),
        ),  # S=3 -> B=4 (cache HIT)
        (
            torch.arange(6).reshape(1, 6).float(),
            torch.ones(1, 6).float(),
        ),  # S=6 -> B=8
    ]

    mark_dynamic_tests_info = [
        ((True, [(1, 1, 16)]), (True, [(1, 1, 16)])),
        ((True, [(1, 1, 16)]), (True, [(1, 1, 16)])),
        ((True, [(1, 1, 16)]), (True, [(1, 1, 16)])),
    ]

    metrics = self.call_and_compare(
        inplace_add, test_inputs, buckets, mark_dynamic_tests_info
    )

    # We expect 2 compilation events (B=4, B=8)
    self.assertEqual(metrics["static_compile_events"], 2)

  def test_static_cache_style_prefill(self):
    def prefill_update(x, state):
      # x: (1, seq_len) - dynamic input
      # state: (1, 16) - static cache (initialized with 9.0 to verify no
      # overwrite)
      seq_len = x.shape[1]
      updated_state = torch.ops.aten.slice_scatter(
          state, x, dim=1, start=0, end=seq_len
      )
      # Return only the active part to verify write correctness.
      # The unused slots might be overwritten with 0.0 in bucketed run, which
      # is fine.
      return updated_state[:, :seq_len]

    buckets = [4, 8, 16]

    # test inputs: x is (1, S), state is (1, 16) initialized with 9.0
    test_inputs = [
        (
            torch.arange(2).reshape(1, 2).float() + 1.0,  # [1, 2]
            torch.fill(torch.zeros(1, 16), 9.0).float(),
        ),  # S=2 -> B=4. Expected: [1, 2, 9, 9, ... 9]
        (
            torch.arange(3).reshape(1, 3).float() + 1.0,  # [1, 2, 3]
            torch.fill(torch.zeros(1, 16), 9.0).float(),
        ),  # S=3 -> B=4 (cache HIT). Expected: [1, 2, 3, 9, ... 9]
        (
            torch.arange(6).reshape(1, 6).float() + 1.0,  # [1, 2, 3, 4, 5, 6]
            torch.fill(torch.zeros(1, 16), 9.0).float(),
        ),  # S=6 -> B=8. Expected: [1, 2, 3, 4, 5, 6, 9, 9, ... 9]
    ]

    mark_dynamic_tests_info = [
        ((True, [(1, 1, 16)]), (False, [])),
        ((True, [(1, 1, 16)]), (False, [])),
        ((True, [(1, 1, 16)]), (False, [])),
    ]

    metrics = self.call_and_compare(
        prefill_update, test_inputs, buckets, mark_dynamic_tests_info
    )

    # We expect 2 compilation events (B=4, B=8)
    self.assertEqual(metrics["static_compile_events"], 2)

  def test_static_cache_style_decode(self):
    def decode_update(x, state, index):
      # x: (1, 1) - new token
      # state: (1, 16) - cache
      # index: (1,) - tensor containing S
      updated_state = torch.ops.aten.index_copy(state, 1, index, x)
      return updated_state

    buckets = [4, 8, 16]

    # All shapes are static!
    # x is (1, 1), state is (1, 16), index is (1,)
    # We pass different values for index: [2], [3], [6]
    test_inputs = [
        (
            torch.tensor([[55.0]]).float(),
            torch.fill(torch.zeros(1, 16), 9.0).float(),
            torch.tensor([2]).long(),
        ),  # Expected: state with 55.0 at index 2
        (
            torch.tensor([[66.0]]).float(),
            torch.fill(torch.zeros(1, 16), 9.0).float(),
            torch.tensor([3]).long(),
        ),  # Expected: state with 66.0 at index 3
        (
            torch.tensor([[77.0]]).float(),
            torch.fill(torch.zeros(1, 16), 9.0).float(),
            torch.tensor([6]).long(),
        ),  # Expected: state with 77.0 at index 6
    ]

    # Since there are NO dynamic shapes, we pass mark_dynamic = False for all
    # inputs
    mark_dynamic_tests_info = [
        ((False, []), (False, []), (False, [])),
        ((False, []), (False, []), (False, [])),
        ((False, []), (False, []), (False, [])),
    ]

    metrics = self.call_and_compare(
        decode_update, test_inputs, buckets, mark_dynamic_tests_info
    )

    # Since shapes are static, it compiles ONCE at the first step and hits the
    # cache.
    # Note: This executes via the standard StaticCompiler path, not
    # LazyBucketedCompiledExecutable, but it is registered in TpuBackend's
    # _compiled_executables list.
    self.assertEqual(metrics["static_compile_events"], 1)

  def test_chained_cache_update(self):
    def cache_update(x, cache):
      # x: (1, 1) - new token
      # cache: (1, S) - dynamic cache
      return torch.cat([cache, x], dim=1)

    buckets = [4, 8, 16]

    # Initialize compiler backend
    backend = _backend.TpuBackend(
        debug=True,
        bucketing=True,
        buckets=buckets,
    )
    compiled = torch.compile(cache_update, backend=backend)

    # Initial cache: shape [1, 2] (B=4)
    cache = torch.ones(1, 2).to(self.device).float()

    # Step 1
    x1 = torch.fill(torch.zeros(1, 1), 2.0).to(self.device).float()
    torch._dynamo.mark_dynamic(cache, 1, min=1, max=16)
    cache = compiled(x1, cache)
    self.assertEqual(cache.shape, (1, 3))
    utils.assert_close(cache.to("cpu"), torch.tensor([[1.0, 1.0, 2.0]]))

    # Step 2
    x2 = torch.fill(torch.zeros(1, 1), 3.0).to(self.device).float()
    torch._dynamo.mark_dynamic(cache, 1, min=1, max=16)
    cache = compiled(x2, cache)
    self.assertEqual(cache.shape, (1, 4))
    utils.assert_close(cache.to("cpu"), torch.tensor([[1.0, 1.0, 2.0, 3.0]]))

    # Step 3
    x3 = torch.fill(torch.zeros(1, 1), 4.0).to(self.device).float()
    torch._dynamo.mark_dynamic(cache, 1, min=1, max=16)
    cache = compiled(x3, cache)
    self.assertEqual(cache.shape, (1, 5))
    utils.assert_close(
        cache.to("cpu"), torch.tensor([[1.0, 1.0, 2.0, 3.0, 4.0]])
    )

    # Step 4
    x4 = torch.fill(torch.zeros(1, 1), 5.0).to(self.device).float()
    torch._dynamo.mark_dynamic(cache, 1, min=1, max=16)
    cache = compiled(x4, cache)
    self.assertEqual(cache.shape, (1, 6))
    utils.assert_close(
        cache.to("cpu"), torch.tensor([[1.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    )

    # Step 5
    x5 = torch.fill(torch.zeros(1, 1), 6.0).to(self.device).float()
    torch._dynamo.mark_dynamic(cache, 1, min=1, max=16)
    cache = compiled(x5, cache)
    self.assertEqual(cache.shape, (1, 7))
    utils.assert_close(
        cache.to("cpu"), torch.tensor([[1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    )

    compiled_buckets = 0
    for executable in backend._compiled_executables:
      if hasattr(executable, "_executables"):
        compiled_buckets += len(executable._executables)
      else:
        compiled_buckets += 1
    # We expect exactly 2 compilation events (B=4, B=8) even after 5 steps
    self.assertEqual(compiled_buckets, 2)

  def test_simple_add_autobucketing(self):
    """Verifies automatic bucketization (powers of 2) when buckets is None."""

    def simple(x):
      return x + 3.0

    # Pass buckets=None to trigger automatic power-of-2 bucketization
    buckets = None

    test_inputs = [
        (
            torch.arange(2).reshape(1, 2).float(),
        ),  # S=2 -> power of 2 is 2. Bucket B=2 (Compile 1)
        (
            torch.arange(3).reshape(1, 3).float(),
        ),  # S=3 -> power of 2 is 4. Bucket B=4 (Compile 2)
        (
            torch.arange(4).reshape(1, 4).float(),
        ),  # S=4 -> power of 2 is 4. Bucket B=4 (Cache HIT)
        (
            torch.arange(6).reshape(1, 6).float(),
        ),  # S=6 -> power of 2 is 8. Bucket B=8 (Compile 3)
        (
            torch.arange(7).reshape(1, 7).float(),
        ),  # S=7 -> power of 2 is 8. Bucket B=8 (Cache HIT)
    ]

    mark_dynamic_tests_info = [
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
    ]

    metrics = self.call_and_compare(
        simple, test_inputs, buckets, mark_dynamic_tests_info, bucketing=True
    )

    # We expect exactly 3 static compilation calls (B=2, B=4, B=8)
    self.assertEqual(metrics["static_compile_events"], 3)

  def test_simple_add_no_bucketing(self):
    """Verifies behavior when bucketing is disabled (compiles a single dynamic graph)."""

    def simple(x):
      return x + 3.0

    test_inputs = [
        (
            torch.arange(2).reshape(1, 2).float(),
        ),  # S=2 (Compile 1 - Dynamic Graph)
        (
            torch.arange(3).reshape(1, 3).float(),
        ),  # S=3 (Cache HIT - reuse dynamic graph)
        (torch.arange(4).reshape(1, 4).float(),),  # S=4 (Cache HIT)
    ]

    mark_dynamic_tests_info = [
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
        ((True, [(1, 1, 16)]),),
    ]

    metrics = self.call_and_compare(
        simple,
        test_inputs,
        buckets=None,
        mark_dynamic_tests_info=mark_dynamic_tests_info,
        bucketing=False,
        dynamism=True,
    )

    # With bucketing=False and dynamism=True, it compiles a single dynamic
    # executable
    self.assertEqual(metrics["static_compile_events"], 1)

  def test_subclass_cache_reproduce(self):
    class MockLayer:

      def __init__(self):
        self.keys = None
        self.values = None
        self.is_initialized = False

      def lazy_initialization(self, k, v):
        self.keys = torch.tensor([], dtype=k.dtype, device=k.device)
        self.values = torch.tensor([], dtype=v.dtype, device=v.device)
        self.is_initialized = True

      def update(self, k, v):
        if not self.is_initialized:
          self.lazy_initialization(k, v)
        self.keys = torch.cat([self.keys, k], dim=-2)
        self.values = torch.cat([self.values, v], dim=-2)
        return self.keys, self.values

      def get_seq_length(self):
        if not self.is_initialized or self.keys.numel() == 0:
          return 0
        return self.keys.shape[-2]

    class MockCache:

      def __init__(self):
        self.layers = [MockLayer()]

      def get_seq_length(self):
        return self.layers[0].get_seq_length()

    def cache_update_fn(x, cache):
      key_states, _ = cache.layers[0].update(x, x)
      return key_states, cache

    buckets = [4, 8, 16]
    backend = _backend.TpuBackend(
        debug=True,
        bucketing=True,
        buckets=buckets,
    )
    compiled = torch.compile(cache_update_fn, backend=backend)

    cache = MockCache()
    x1 = torch.ones(1, 1, 2, 1, device=self.device).float()
    torch._dynamo.mark_dynamic(x1, 2, min=1, max=16)

    # Run prefill (Step 1)
    _, cache = compiled(x1, cache)

    print("Cache type after step 1:", type(cache))
    print("Layer 0 type after step 1:", type(cache.layers[0]))
    print("Layer 0 keys type after step 1:", type(cache.layers[0].keys))

    seq_len = cache.get_seq_length()
    self.assertEqual(seq_len, 2)


if __name__ == "__main__":
  absltest.main()
