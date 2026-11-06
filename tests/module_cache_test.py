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

"""Unit tests for module_cache."""

import concurrent.futures
import threading
import time
import types
from unittest import mock

from absl.testing import absltest
from etils import epath
import torch
from tests import module_cache
from tests import seed_test_utils


class DummyConfig:

  def __init__(self, hidden_size: int = 4):
    self.hidden_size = hidden_size


class DummyModel(torch.nn.Module):

  def __init__(self, config: DummyConfig | None = None):
    super().__init__()
    self.config = config or DummyConfig()
    self.fc1 = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)
    self.fc2 = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)
    # Tie weights
    self.fc2.weight = self.fc1.weight


class ModuleCacheTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    module_cache.clear_all_module_caches()
    self.addCleanup(module_cache.clear_all_module_caches)

  def test_thread_safe_keyed_cache_basic_and_clear_locks(self):
    cache = module_cache.ThreadSafeKeyedCache[str, dict[str, int]]()
    loader = mock.Mock(side_effect=lambda: {"count": loader.call_count})

    val1 = cache.get_or_load("key1", loader)
    self.assertEqual(val1, {"count": 1})
    self.assertEqual(loader.call_count, 1)
    self.assertEqual(len(cache), 1)
    self.assertIn("key1", cache)

    # Cache hit
    val2 = cache.get_or_load("key1", loader)
    self.assertEqual(val2, {"count": 1})
    self.assertEqual(loader.call_count, 1)

    # Per-key lock existed, clear releases cache and locks
    self.assertGreater(cache.active_lock_count, 0)
    cache.clear()
    self.assertEqual(len(cache), 0)
    self.assertNotIn("key1", cache)
    self.assertEqual(cache.active_lock_count, 0)

  def test_thread_safe_keyed_cache_deepcopy_isolation(self):
    cache = module_cache.ThreadSafeKeyedCache[str, list[int]]()
    # Case: deepcopy_on_put=False, deepcopy_on_get=True
    # The first caller triggers a miss; mutating its returned object must not corrupt cache.
    v1 = cache.get_or_load(
        "numbers",
        lambda: [1, 2, 3],
        deepcopy_on_put=False,
        deepcopy_on_get=True,
    )
    v1.append(4)

    # Next get should return pristine cached value without the appended 4
    v2 = cache.get_or_load(
        "numbers",
        lambda: [9, 9, 9],
        deepcopy_on_put=False,
        deepcopy_on_get=True,
    )
    self.assertEqual(v2, [1, 2, 3])

  def test_thread_safe_keyed_cache_concurrent_access(self):
    cache = module_cache.ThreadSafeKeyedCache[str, int]()
    load_counts: dict[str, int] = {}
    load_lock = threading.Lock()

    def slow_loader(k: str) -> int:
      time.sleep(0.01)
      with load_lock:
        load_counts[k] = load_counts.get(k, 0) + 1
        return load_counts[k]

    def task(k: str) -> int:
      return cache.get_or_load(k, lambda: slow_loader(k))

    keys = ["a", "b", "c", "a", "b", "c"] * 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
      results = list(executor.map(task, keys))

    self.assertNotEmpty(results)
    # Each key should have been loaded exactly once
    self.assertEqual(load_counts["a"], 1)
    self.assertEqual(load_counts["b"], 1)
    self.assertEqual(load_counts["c"], 1)

  def test_config_cache(self):
    cache = module_cache.ConfigCache()
    loader = mock.Mock(
        side_effect=lambda: module_cache.ConfigCacheEntry(
            config=DummyConfig(hidden_size=256),
            path=epath.Path("/tmp/model"),
            subfolder=None,
        )
    )

    entry1 = cache.get("transformers", "/tmp", "model_x", loader=loader)
    self.assertEqual(entry1.config.hidden_size, 256)
    self.assertEqual(loader.call_count, 1)

    # Cache hit returns defensive copy
    entry2 = cache.get("transformers", "/tmp", "model_x", loader=loader)
    self.assertEqual(loader.call_count, 1)
    entry2.config.hidden_size = 512

    entry3 = cache.get("transformers", "/tmp", "model_x", loader=loader)
    self.assertEqual(entry3.config.hidden_size, 256)

  def test_cached_module_factory_lifecycle(self):
    factory = mock.Mock(side_effect=DummyModel)
    cached_factory = module_cache.ModuleFactoryCache(factory)
    self.assertEqual(cached_factory.load_count, 0)
    self.assertFalse(cached_factory.is_cached)

    # First call instantiates pristine CPU model
    m1 = cached_factory()
    self.assertEqual(cached_factory.load_count, 1)
    self.assertTrue(cached_factory.is_cached)
    self.assertIsInstance(m1, DummyModel)

    # Second call returns deepcopy without re-invoking factory
    m2 = cached_factory()
    self.assertEqual(cached_factory.load_count, 1)
    self.assertIsNot(m1, m2)

    # Tied weights preserved in deepcopy
    self.assertIs(m2.fc1.weight, m2.fc2.weight)

    # Mutating m2 does not affect template
    m2.config.hidden_size = 999
    m3 = cached_factory()
    self.assertEqual(m3.config.hidden_size, 4)

    # Meta-device bypasses cache
    with torch.device("meta"):
      m_meta = cached_factory()
      self.assertEqual(cached_factory.load_count, 2)
      self.assertEqual(m_meta.fc1.weight.device.type, "meta")

    # Clear releases reference
    cached_factory.clear()
    self.assertFalse(cached_factory.is_cached)

  def test_cached_module_factory_caches_a_template_for_each_dtype(self):
    def _factory(dtype: torch.dtype | None = None) -> DummyModel:
      model = DummyModel()
      return model if dtype is None else model.to(dtype)

    factory = mock.Mock(side_effect=_factory)
    cached_factory = module_cache.ModuleFactoryCache(factory)

    default_models = [cached_factory(), cached_factory()]
    bfloat16_models = [
        cached_factory(dtype=torch.bfloat16),
        cached_factory(dtype=torch.bfloat16),
    ]
    with torch.device("meta"):
      meta_model = cached_factory(dtype=torch.bfloat16)

    self.assertEqual(
        factory.call_args_list,
        [
            mock.call(),
            mock.call(dtype=torch.bfloat16),
            mock.call(dtype=torch.bfloat16),
        ],
    )
    for model in default_models:
      self.assertEqual(model.fc1.weight.dtype, torch.float32)
    for model in bfloat16_models:
      self.assertEqual(model.fc1.weight.dtype, torch.bfloat16)
    self.assertEqual(meta_model.fc1.weight.dtype, torch.bfloat16)
    self.assertEqual(meta_model.fc1.weight.device.type, "meta")

    cached_factory.clear()
    self.assertFalse(cached_factory.is_cached)

  def test_cached_module_factory_scope(self):
    spec = types.SimpleNamespace(module_factory=DummyModel)
    with module_cache.cached_module_factory_scope(spec) as wrapped_spec:
      self.assertIs(wrapped_spec, spec)
      self.assertIsInstance(
          spec.module_factory, module_cache.ModuleFactoryCache
      )
      _ = spec.module_factory()
      self.assertTrue(spec.module_factory.is_cached)

    # After scope exit, factory cache is cleared
    self.assertFalse(spec.module_factory.is_cached)

  def test_module_cache_session_task_transitions(self):
    session = module_cache.ModuleCacheSession()

    # Task 1: model A
    is_hit_1 = session.start_task("transformers", "model_a")
    self.assertFalse(is_hit_1)
    self.assertEqual(session.current_model_key, ("transformers", "model_a"))

    # Populate caches
    session.config_cache.get(
        "transformers",
        None,
        "model_a",
        loader=lambda: module_cache.ConfigCacheEntry(config=DummyConfig(64)),
    )
    self.assertNotEmpty(session.config_cache)

    dummy_factory = module_cache.ModuleFactoryCache(DummyModel)
    _ = dummy_factory()
    session.cached_factory = dummy_factory
    self.assertIsNotNone(session.cached_factory)

    # Task 2: same model A (e.g. Correctness after Smoke)
    is_hit_2 = session.start_task("transformers", "model_a")
    self.assertTrue(is_hit_2)
    self.assertNotEmpty(session.config_cache)
    self.assertIsNotNone(session.cached_factory)

    # Task 3: different model B (clears session and caches)
    is_hit_3 = session.start_task("transformers", "model_b")
    self.assertFalse(is_hit_3)
    self.assertEqual(session.current_model_key, ("transformers", "model_b"))
    self.assertEmpty(session.config_cache)
    self.assertIsNone(session.cached_factory)

    # Clear
    session.clear()
    self.assertIsNone(session.current_model_key)


if __name__ == "__main__":
  absltest.main()
