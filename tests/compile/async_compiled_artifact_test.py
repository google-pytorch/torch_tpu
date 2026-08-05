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

from collections.abc import Callable, Sequence
import concurrent.futures
import pickle
import threading
import time
from typing import Any

from absl.testing import absltest
from torch._inductor.utils import InputType
from torch_tpu._internal.compile import torch_tpu_compiled_executable
from tests import seed_test_utils

AsyncCompiledArtifact = torch_tpu_compiled_executable.AsyncCompiledArtifact
CompiledArtifact = torch_tpu_compiled_executable.CompiledArtifact


class DummyConcreteArtifact(CompiledArtifact):

  def __init__(self, value: int = 42):
    super().__init__()
    self.value = value
    self.call_count = 0
    self.custom_property = "custom_tpu_data"

  def __call__(
      self, inputs: Sequence[InputType] | None = None, **kwargs: Any
  ) -> Any:
    self.call_count += 1
    return {"result": self.value, "inputs": inputs, "kwargs": kwargs}

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    return DummyConcreteArtifact, (self.value,)

  def custom_method(self, multiplier: int) -> int:
    return self.value * multiplier


class AsyncCompiledArtifactTest(seed_test_utils.RepeatableTest):

  def test_initial_state_unresolved(self):
    future = concurrent.futures.Future()
    artifact = AsyncCompiledArtifact(future)

    self.assertFalse(artifact.is_resolved)
    self.assertTrue(artifact._boxed_call)
    self.assertIsNone(artifact._fx_graph_cache_key)
    self.assertFalse(artifact._is_fallback)
    self.assertIsInstance(artifact, AsyncCompiledArtifact)
    self.assertIsInstance(artifact, CompiledArtifact)

  def test_explicit_resolve_updates_state(self):
    future = concurrent.futures.Future()
    real_artifact = DummyConcreteArtifact(value=100)
    future.set_result(real_artifact)

    artifact = AsyncCompiledArtifact(future)
    self.assertFalse(artifact.is_resolved)

    artifact.resolve()

    self.assertTrue(artifact.is_resolved)
    self.assertIsInstance(artifact, DummyConcreteArtifact)
    self.assertEqual(artifact.value, 100)
    self.assertEqual(artifact.custom_property, "custom_tpu_data")

  def test_resolve_is_idempotent(self):
    future = concurrent.futures.Future()
    real_artifact = DummyConcreteArtifact(value=100)
    future.set_result(real_artifact)

    artifact = AsyncCompiledArtifact(future)
    artifact.resolve()
    self.assertTrue(artifact.is_resolved)

    artifact.resolve()
    self.assertTrue(artifact.is_resolved)
    self.assertEqual(artifact.value, 100)

  def test_call_triggers_resolution_and_executes(self):
    future = concurrent.futures.Future()
    real_artifact = DummyConcreteArtifact(value=7)
    future.set_result(real_artifact)

    artifact = AsyncCompiledArtifact(future)
    self.assertFalse(artifact.is_resolved)

    out = artifact(["t1", "t2"], extra_kwarg="foo")

    self.assertTrue(artifact.is_resolved)
    self.assertIsInstance(artifact, DummyConcreteArtifact)
    self.assertEqual(
        out,
        {
            "result": 7,
            "inputs": ["t1", "t2"],
            "kwargs": {"extra_kwarg": "foo"},
        },
    )
    self.assertEqual(artifact.call_count, 1)

    out2 = artifact(["t3"])
    self.assertEqual(artifact.call_count, 2)
    self.assertEqual(out2["inputs"], ["t3"])

  def test_getattr_triggers_resolution(self):
    future = concurrent.futures.Future()
    real_artifact = DummyConcreteArtifact(value=15)
    future.set_result(real_artifact)

    artifact = AsyncCompiledArtifact(future)
    self.assertFalse(artifact.is_resolved)
    self.assertEqual(artifact.custom_property, "custom_tpu_data")
    self.assertTrue(artifact.is_resolved)
    self.assertEqual(artifact.custom_method(3), 45)

  def test_reduce_and_pickle_triggers_resolution(self):
    future = concurrent.futures.Future()
    real_artifact = DummyConcreteArtifact(value=99)
    future.set_result(real_artifact)

    artifact = AsyncCompiledArtifact(future)
    self.assertFalse(artifact.is_resolved)

    pickled_data = pickle.dumps(artifact)

    self.assertTrue(artifact.is_resolved)
    self.assertIsInstance(artifact, DummyConcreteArtifact)

    restored = pickle.loads(pickled_data)
    self.assertIsInstance(restored, DummyConcreteArtifact)
    self.assertEqual(restored.value, 99)

  def test_aot_autograd_hooks_do_not_cause_resolution(self):
    future = concurrent.futures.Future()
    artifact = AsyncCompiledArtifact(future)

    artifact.prepare_for_serialization()
    self.assertFalse(artifact.is_resolved)

    artifact.post_compile(example_inputs=[], constants=None, graph_kwargs={})
    self.assertFalse(artifact.is_resolved)

    artifact.set_triton_bundle(triton_bundle=None)
    self.assertFalse(artifact.is_resolved)

  def test_magic_attributes_do_not_resolve_and_raise_attribute_error(self):
    future = concurrent.futures.Future()
    artifact = AsyncCompiledArtifact(future)

    with self.assertRaises(AttributeError):
      _ = artifact.__non_existent_magic_attr__

    self.assertFalse(artifact.is_resolved)

    # Special framework attributes return class defaults without resolving
    self.assertIsNone(artifact._fx_graph_cache_key)
    self.assertFalse(artifact._is_fallback)
    self.assertFalse(artifact.is_resolved)

  def test_exception_propagation_on_resolve(self):
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError("Simulated XLA compilation failure"))

    artifact = AsyncCompiledArtifact(future)
    self.assertFalse(artifact.is_resolved)

    with self.assertRaisesRegex(
        RuntimeError, "Simulated XLA compilation failure"
    ):
      artifact.resolve()

    self.assertFalse(artifact.is_resolved)

  def test_exception_propagation_on_call(self):
    future = concurrent.futures.Future()
    future.set_exception(ValueError("Invalid tensor layout"))

    artifact = AsyncCompiledArtifact(future)

    with self.assertRaisesRegex(ValueError, "Invalid tensor layout"):
      artifact([])

    self.assertFalse(artifact.is_resolved)

  def test_exception_propagation_on_getattr(self):
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError("Compile error"))

    artifact = AsyncCompiledArtifact(future)

    with self.assertRaisesRegex(RuntimeError, "Compile error"):
      _ = artifact.some_missing_attr

  def test_exception_does_not_raise_on_passive_checks(self):
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError("Compile error"))

    artifact = AsyncCompiledArtifact(future)

    self.assertFalse(artifact.is_resolved)
    artifact.prepare_for_serialization()
    artifact.post_compile()

  def test_concurrent_resolution_thread_safety(self):
    future = concurrent.futures.Future()
    real_artifact = DummyConcreteArtifact(value=123)

    artifact = AsyncCompiledArtifact(future)
    num_threads = 10
    barrier = threading.Barrier(num_threads)

    def worker():
      barrier.wait()
      artifact.resolve()
      res = artifact(["input"])
      return res["result"]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
      futures = [executor.submit(worker) for _ in range(num_threads)]
      time.sleep(0.05)
      future.set_result(real_artifact)
      results = [f.result(timeout=5.0) for f in futures]

    self.assertTrue(artifact.is_resolved)
    self.assertIsInstance(artifact, DummyConcreteArtifact)
    self.assertEqual(results, [123] * num_threads)

  def test_base_compiled_artifact_default_sync_behavior(self):
    concrete = DummyConcreteArtifact(value=1)
    self.assertTrue(concrete.is_resolved)
    concrete.resolve()
    self.assertTrue(concrete.is_resolved)


if __name__ == "__main__":
  absltest.main()
