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

# pylint: disable=g-unsafe-pickle-load

import pickle
import random
import tempfile

from absl.testing import absltest
import torch
from torch._dynamo.backends import registry
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.aot_autograd_result import (
    deserialize_bundled_cache_entry,
)
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class BackendSerializationTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    # Set a fixed random seed to avoid flakes.
    random.seed(42)
    torch.manual_seed(42)

  def check_serialization(self, f, inputs):
    inputs_tpu = _backend.to_device(inputs, torch.device("tpu"))
    gm = dynamo_graph_capture_for_export(f)(*inputs_tpu)

    backend = _backend.TpuBackend(enable_serialization=True)
    compiled_fn = backend(gm, inputs_tpu)

    result = compiled_fn(*inputs_tpu)
    result = _backend.to_device(result, "cpu")
    utils.assert_close(result[0], inputs[0] + inputs[1])
    utils.assert_close(result[1], inputs[0] * inputs[1])

    entry = compiled_fn.serialize()
    self.assertIsNotNone(entry)
    pickled = pickle.dumps(entry)
    restored_entry = pickle.loads(pickled)

    restored_fn = deserialize_bundled_cache_entry(restored_entry)
    self.assertTrue(callable(restored_fn))

    restored_result = restored_fn(*inputs_tpu)
    restored_result = _backend.to_device(restored_result, "cpu")
    utils.assert_close(restored_result[0], inputs[0] + inputs[1])
    utils.assert_close(restored_result[1], inputs[0] * inputs[1])

  def test_simple(self):
    def simple(x, y):
      return x + y, x * y

    inputs = [torch.randn(4, 4), torch.randn(4, 4)]
    self.check_serialization(simple, inputs)

  def _compile_simple_with_registered_backend(self):
    torch._dynamo.reset()
    counters.clear()

    def simple(x, y):
      return x + y, x * y

    x = torch.randn(4, 4).to(torch.device("tpu"))
    y = torch.randn(4, 4).to(torch.device("tpu"))
    compiled_fn = torch.compile(
        simple,
        backend="tpu",
        fullgraph=True,
        dynamic=False,
    )
    result = compiled_fn(x, y)
    result = _backend.to_device(result, "cpu")
    utils.assert_close(result[0], x.cpu() + y.cpu())
    utils.assert_close(result[1], x.cpu() * y.cpu())

    return dict(counters["aot_autograd"])

  def test_registered_backend_uses_aot_autograd_cache(self):
    registered_backend = registry.lookup_backend("tpu")
    self.assertTrue(getattr(registered_backend, "_enable_serialization", False))

    with tempfile.TemporaryDirectory() as cache_dir:
      with temporary_cache_dir(cache_dir):
        first_run = self._compile_simple_with_registered_backend()
        self.assertEqual(first_run.get("autograd_cache_miss"), 1)
        self.assertEqual(first_run.get("autograd_cache_saved"), 1)

        second_run = self._compile_simple_with_registered_backend()
        self.assertEqual(second_run.get("autograd_cache_hit"), 1)

  def test_int_scalar_input(self):
    def simple(x, y):
      return x + y, x * y

    inputs = [torch.randn(4, 4), 4]
    self.check_serialization(simple, inputs)

  def test_input_mutation(self):
    def inplace_update(x, cache):
      cache.add_(x.sum())
      return (cache + 0,)

    inputs_tpu = _backend.to_device(
        [torch.randn(4, 4), torch.zeros(8, 8)], torch.device("tpu")
    )
    gm = dynamo_graph_capture_for_export(inplace_update)(*inputs_tpu)

    backend = _backend.TpuBackend(enable_serialization=True)
    compiled_fn = backend(gm, inputs_tpu)

    x = torch.randn(4, 4).to(torch.device("tpu"))
    cache = torch.zeros(8, 8).to(torch.device("tpu"))
    result = compiled_fn(x, cache)
    result_cpu = _backend.to_device(result, "cpu")
    expected = x.cpu().sum().item()
    utils.assert_close(result_cpu[0], torch.full((8, 8), expected))

    entry = compiled_fn.serialize()
    self.assertIsNotNone(entry)
    restored_fn = deserialize_bundled_cache_entry(
        pickle.loads(pickle.dumps(entry))
    )

    cache2 = torch.zeros(8, 8).to(torch.device("tpu"))
    restored_result = restored_fn(x, cache2)
    restored_result_cpu = _backend.to_device(restored_result, "cpu")
    utils.assert_close(restored_result_cpu[0], torch.full((8, 8), expected))

  def test_different_shapes_produce_different_cache_keys(self):
    def simple(x):
      return (x * 2,)

    backend = _backend.TpuBackend(enable_serialization=True)

    x_small = torch.randn(4, 4).to(torch.device("tpu"))
    gm_small = dynamo_graph_capture_for_export(simple)(x_small)
    compiled_small = backend(gm_small, [x_small])
    result_small = compiled_small(x_small)
    result_small_cpu = _backend.to_device(result_small, "cpu")
    utils.assert_close(result_small_cpu[0], x_small.cpu() * 2)

    x_large = torch.randn(8, 8).to(torch.device("tpu"))
    gm_large = dynamo_graph_capture_for_export(simple)(x_large)
    compiled_large = backend(gm_large, [x_large])
    result_large = compiled_large(x_large)
    result_large_cpu = _backend.to_device(result_large, "cpu")
    utils.assert_close(result_large_cpu[0], x_large.cpu() * 2)

    entry_small = compiled_small.serialize()
    entry_large = compiled_large.serialize()
    self.assertIsNotNone(entry_small)
    self.assertIsNotNone(entry_large)

    restored_small = deserialize_bundled_cache_entry(
        pickle.loads(pickle.dumps(entry_small))
    )
    restored_large = deserialize_bundled_cache_entry(
        pickle.loads(pickle.dumps(entry_large))
    )

    r_small = _backend.to_device(restored_small(x_small), "cpu")
    r_large = _backend.to_device(restored_large(x_large), "cpu")
    utils.assert_close(r_small[0], x_small.cpu() * 2)
    utils.assert_close(r_large[0], x_large.cpu() * 2)

  def test_different_storage_offsets_produce_different_cache_keys(self):
    def simple(x):
      return (x + x,)

    base = torch.arange(129, dtype=torch.int32, device="tpu")
    x_128 = base[128]
    x_125 = base[125]

    backend = _backend.TpuBackend(enable_serialization=True)
    gm_128 = dynamo_graph_capture_for_export(simple)(x_128)
    compiled_128 = backend(gm_128, [x_128])
    entry_128 = compiled_128.serialize()

    gm_125 = dynamo_graph_capture_for_export(simple)(x_125)
    compiled_125 = backend(gm_125, [x_125])
    entry_125 = compiled_125.serialize()

    self.assertIsNotNone(entry_128)
    self.assertIsNotNone(entry_125)

    restored_128 = deserialize_bundled_cache_entry(
        pickle.loads(pickle.dumps(entry_128))
    )
    restored_125 = deserialize_bundled_cache_entry(
        pickle.loads(pickle.dumps(entry_125))
    )
    self.assertEqual(
        _backend.to_device(restored_128(x_128), "cpu")[0].item(), 256
    )
    self.assertEqual(
        _backend.to_device(restored_125(x_125), "cpu")[0].item(), 250
    )

    with tempfile.TemporaryDirectory() as cache_dir:
      with temporary_cache_dir(cache_dir):
        # First run compiles for storage_offset=128 (miss & save)
        torch._dynamo.reset()
        counters.clear()
        compiled_fn_1 = torch.compile(
            simple,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        res_1 = compiled_fn_1(x_128)
        self.assertEqual(res_1[0].cpu().item(), 256)
        c1 = dict(counters["aot_autograd"])
        self.assertEqual(c1.get("autograd_cache_miss"), 1)
        self.assertEqual(c1.get("autograd_cache_hit", 0), 0)

        # Second run with different storage_offset=125 in a fresh Dynamo session
        # must result in a cache MISS (no key collision) and compile cleanly.
        torch._dynamo.reset()
        counters.clear()
        compiled_fn_2 = torch.compile(
            simple,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        res_2 = compiled_fn_2(x_125)
        self.assertEqual(res_2[0].cpu().item(), 250)
        c2 = dict(counters["aot_autograd"])
        self.assertEqual(c2.get("autograd_cache_miss"), 1)
        self.assertEqual(c2.get("autograd_cache_hit", 0), 0)

        # Subsequent run for storage_offset=128 hits the autograd cache
        torch._dynamo.reset()
        counters.clear()
        compiled_fn_3 = torch.compile(
            simple,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        res_3 = compiled_fn_3(x_128)
        self.assertEqual(res_3[0].cpu().item(), 256)
        c3 = dict(counters["aot_autograd"])
        self.assertEqual(c3.get("autograd_cache_hit"), 1)

  def test_multiple_serialization_roundtrips(self):
    def simple(x):
      return (x + 1,)

    x = torch.randn(4, 4).to(torch.device("tpu"))
    gm = dynamo_graph_capture_for_export(simple)(x)
    backend = _backend.TpuBackend(enable_serialization=True)
    compiled_fn = backend(gm, [x])

    entry = compiled_fn.serialize()
    self.assertIsNotNone(entry)

    for _ in range(3):
      pickled = pickle.dumps(entry)
      entry = pickle.loads(pickled)
      restored_fn = deserialize_bundled_cache_entry(entry)
      result = _backend.to_device(restored_fn(x), "cpu")
      utils.assert_close(result[0], x.cpu() + 1)

  def test_pickling_constant_attribute_tensor_on_tpu(self):
    pickled = pickle.dumps(torch.tensor(2.0, device="tpu"))
    restored = pickle.loads(pickled).cpu()
    utils.assert_close(restored, torch.tensor(2.0))

  def test_compile_inside_tracing_context_without_dynamo_source(self):
    def simple(x):
      return (x + 1,)

    x = torch.randn(4, 4, device="tpu")
    gm = dynamo_graph_capture_for_export(simple)(x)
    # Ensure placeholder nodes do not have _dynamo_source attribute (e.g. from
    # non-Dynamo or piecewise/export frontends like vLLM).
    for node in gm.graph.find_nodes(op="placeholder"):
      if hasattr(node, "_dynamo_source"):
        delattr(node, "_dynamo_source")

    backend = _backend.TpuBackend(enable_serialization=True)
    tracing_ctx = torch._guards.TracingContext(None)
    with torch._guards.tracing(tracing_ctx):
      compiled_fn = backend(gm, [x])
    res = _backend.to_device(compiled_fn(x), "cpu")
    utils.assert_close(res[0], x.cpu() + 1)


if __name__ == "__main__":
  absltest.main()
