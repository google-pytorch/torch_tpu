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

"""Thread-safe caching infrastructure for model configurations, preprocessors, and factories.

Provides double-checked locking keyed caches to avoid redundant filesystem,
storage,
and network reads across test runners and multi-phase worker sessions.
"""

from collections.abc import Callable, Iterator
import contextlib
import copy
import threading
from typing import Any, Generic, NamedTuple, Protocol, TypeVar, cast

from absl import logging
from etils import epath
import torch

K = TypeVar("K")
V = TypeVar("V")


class ThreadSafeKeyedCache(Generic[K, V]):
  """Generic thread-safe cache with fine-grained per-key double-checked locking."""

  def __init__(self):
    self._global_lock = threading.Lock()
    self._key_locks: dict[K, threading.Lock] = {}
    self._cache: dict[K, V] = {}

  def __len__(self) -> int:
    with self._global_lock:
      return len(self._cache)

  def __contains__(self, key: K) -> bool:
    with self._global_lock:
      return key in self._cache

  def values(self) -> list[V]:
    """Returns a snapshot copy of all cached values."""
    with self._global_lock:
      return list(self._cache.values())

  def clear(self) -> None:
    """Clears all cached entries and per-key locks to avoid memory leaks."""
    with self._global_lock:
      self._cache.clear()
      self._key_locks.clear()

  @property
  def active_lock_count(self) -> int:
    """Returns the number of allocated per-key locks."""
    with self._global_lock:
      return len(self._key_locks)

  def _get_key_lock(self, key: K) -> threading.Lock:
    with self._global_lock:
      if key not in self._key_locks:
        self._key_locks[key] = threading.Lock()
      return self._key_locks[key]

  def get_or_load(
      self,
      key: K,
      loader: Callable[[], V],
      *,
      deepcopy_on_put: bool = False,
      deepcopy_on_get: bool = False,
  ) -> V:
    """Gets cached entry or loads and stores it atomically per key.

    Uses double-checked locking with a global lock for dict lookups and a
    per-key
    lock during loader execution so concurrent loads for different keys do not
    block each other.
    """
    # 1. Fast-path lookup
    with self._global_lock:
      if key in self._cache:
        val = self._cache[key]
        return copy.deepcopy(val) if deepcopy_on_get else val

    # 2. Acquire per-key lock
    key_lock = self._get_key_lock(key)
    with key_lock:
      # 3. Double-check cache inside per-key lock
      with self._global_lock:
        if key in self._cache:
          val = self._cache[key]
          return copy.deepcopy(val) if deepcopy_on_get else val

      # 4. Load fresh entry outside global lock
      fresh_entry = loader()

      # 5. Store in cache under global lock
      cached_entry = (
          copy.deepcopy(fresh_entry) if deepcopy_on_put else fresh_entry
      )
      with self._global_lock:
        self._cache[key] = cached_entry

      return (
          copy.deepcopy(fresh_entry)
          if (deepcopy_on_get and not deepcopy_on_put)
          else fresh_entry
      )


ConfigCacheKey = tuple[str, str | None, str, str | None, bool]


class ConfigCacheEntry(NamedTuple):
  config: Any
  path: epath.Path | None = None
  subfolder: str | None = None


class ConfigCache(ThreadSafeKeyedCache[ConfigCacheKey, ConfigCacheEntry]):
  """Thread-safe cache manager for model configurations across providers."""

  def _make_key(
      self,
      provider: str,
      base_path: epath.Path | str | None,
      name: str,
      subfolder: str | None = None,
      load_weights: bool = False,
  ) -> ConfigCacheKey:
    return (
        provider,
        str(base_path) if base_path is not None else None,
        name,
        subfolder,
        load_weights,
    )

  def get(
      self,
      provider: str,
      base_path: epath.Path | str | None,
      name: str,
      subfolder: str | None = None,
      *,
      load_weights: bool = False,
      loader: Callable[[], ConfigCacheEntry],
  ) -> ConfigCacheEntry:
    """Gets cached entry or loads and stores it atomically per key."""
    key = self._make_key(
        provider, base_path, name, subfolder, load_weights=load_weights
    )
    return self.get_or_load(
        key, loader, deepcopy_on_put=True, deepcopy_on_get=True
    )


PreprocessorCacheKey = tuple[Any, ...]


class PreprocessorCache(ThreadSafeKeyedCache[PreprocessorCacheKey, Any]):
  """Thread-safe cache manager for model preprocessors and tokenizers."""

  def get_or_load(
      self,
      key: PreprocessorCacheKey,
      loader: Callable[[], Any],
      *,
      deepcopy_on_put: bool = False,
      deepcopy_on_get: bool = False,
  ) -> Any:
    """Gets cached preprocessor or loads and stores it atomically per key."""
    return super().get_or_load(
        key,
        loader,
        deepcopy_on_put=deepcopy_on_put,
        deepcopy_on_get=deepcopy_on_get,
    )


def _detach_non_leaf_tensors(model: torch.nn.Module) -> None:
  """Detaches non-leaf tensors so copy.deepcopy does not fail on grad_fn."""
  for param in model.parameters():
    if not param.is_leaf and param.requires_grad:
      param.data = param.data.detach()
  for submod in model.modules():
    for mapping in (submod.__dict__, getattr(submod, "_buffers", None)):
      if not isinstance(mapping, dict):
        continue
      for k, v in list(mapping.items()):
        if isinstance(v, torch.Tensor) and not v.is_leaf and v.requires_grad:
          mapping[k] = v.detach()


class ModuleFactory(Protocol):
  """Constructs a model, optionally in a given dtype."""

  def __call__(self, dtype: torch.dtype | None = None) -> torch.nn.Module:
    ...


class ModuleFactoryCache:
  """Caches an untouched CPU template model to avoid redundant module_factory() loads.

  The cached template model is never mutated, moved off CPU, compiled, or
  executed. Every call to `__call__()` returns a fresh `copy.deepcopy` of the
  pristine CPU template, preserving tied weights while isolating all parameters,
  buffers, submodules, and configs. Falls back to calling the underlying
  `module_factory` if `copy.deepcopy` fails.

  Like `ModuleSpec.module_factory`, `__call__()` accepts an optional `dtype`,
  which is passed to the underlying `module_factory`. A separate template is
  cached for each `dtype` that is requested.
  """

  def __init__(
      self,
      orig_factory: Callable[[], Any] | ModuleFactory,
      seed: int = 42,
  ):
    self._orig_factory = orig_factory
    self._seed = seed
    self._pristine_cpu_models: dict[torch.dtype | None, Any] = {}
    self._load_count: int = 0

  @property
  def load_count(self) -> int:
    """Returns the number of times the underlying module_factory was invoked."""
    return self._load_count

  def _instantiate(self, dtype: torch.dtype | None) -> Any:
    torch.manual_seed(self._seed)
    with torch.no_grad():
      self._load_count += 1
      # Factories that predate the `dtype` argument take no arguments.
      if dtype is None:
        return self._orig_factory()
      return cast(ModuleFactory, self._orig_factory)(dtype=dtype)

  def preload(self, dtype: torch.dtype | None = None) -> None:
    """Instantiates and caches the pristine CPU model template if not yet loaded.

    Args:
      dtype: The dtype to pass to the underlying `module_factory`.
    """
    if dtype in self._pristine_cpu_models:
      return
    with torch.device("cpu"):
      model = self._instantiate(dtype)
    if isinstance(model, torch.nn.Module):
      _detach_non_leaf_tensors(model)
    if getattr(model, "config", None) is not None:
      with contextlib.suppress(Exception):
        model.config = copy.deepcopy(model.config)
    self._pristine_cpu_models[dtype] = model

  def __call__(self, dtype: torch.dtype | None = None) -> Any:
    # Meta-device callers need a meta model; never serve or cache it from CPU.
    if torch.get_default_device().type == "meta":
      return self._instantiate(dtype)
    self.preload(dtype)
    try:
      return copy.deepcopy(self._pristine_cpu_models[dtype])
    except Exception as copy_err:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to deepcopy cached CPU model (%s); falling back to"
          " calling original module_factory().",
          copy_err,
      )
      return self._instantiate(dtype)

  @property
  def is_cached(self) -> bool:
    """Returns True if a pristine CPU model template is currently cached."""
    return bool(self._pristine_cpu_models)

  def clear(self) -> None:
    """Releases the cached pristine CPU model references."""
    self._pristine_cpu_models.clear()


def cache_module_factory(
    spec: Any,
    *,
    preload: bool = False,
    seed: int = 42,
) -> Any:
  """Wraps `spec.module_factory` in a `ModuleFactoryCache` to avoid redundant loads.

  Args:
    spec: Module spec whose `module_factory` is wrapped in place.
    preload: If True, loads the CPU template immediately so that weight-loading
      errors surface at the call site (e.g. to fall back to random weights)
      instead of inside a later runner.
    seed: RNG seed applied before every underlying `module_factory()` call.

  Returns:
    The same `spec`, with a cached `module_factory`.
  """
  orig_factory = getattr(spec, "module_factory", None)
  if not callable(orig_factory):
    return spec
  if not isinstance(orig_factory, ModuleFactoryCache):
    spec.module_factory = ModuleFactoryCache(orig_factory, seed=seed)
  if preload:
    spec.module_factory.preload()
  return spec


def release_module_factory_cache(spec: Any) -> None:
  """Releases the cached CPU template if `spec.module_factory` is cached."""
  factory = getattr(spec, "module_factory", None)
  if isinstance(factory, ModuleFactoryCache):
    factory.clear()


@contextlib.contextmanager
def cached_module_factory_scope(
    spec: Any,
    *,
    preload: bool = False,
    seed: int = 42,
) -> Iterator[Any]:
  """Caches `spec.module_factory` within the scope and releases it on exit."""
  try:
    yield cache_module_factory(spec, preload=preload, seed=seed)
  finally:
    release_module_factory_cache(spec)


class ModuleCacheSession:
  """Manages cache lifecycle for a module across execution phases (e.g. Smoke & Correctness).

  Tracks the active (provider, model_id). When switching to a different model,
  it automatically clears the module caches. If the next task is for the same
  model,
  the cached configurations, preprocessors, and CPU templates are preserved.
  """

  def __init__(
      self,
      config_cache: ConfigCache | None = None,
      preprocessor_cache: PreprocessorCache | None = None,
  ):
    self._config_cache = (
        config_cache if config_cache is not None else CONFIG_CACHE
    )
    self._preprocessor_cache = (
        preprocessor_cache
        if preprocessor_cache is not None
        else PREPROCESSOR_CACHE
    )
    self._current_model_key: tuple[str, str] | None = None
    self._cached_factory: ModuleFactoryCache | None = None
    self._lock = threading.Lock()

  @property
  def current_model_key(self) -> tuple[str, str] | None:
    with self._lock:
      return self._current_model_key

  @property
  def config_cache(self) -> ConfigCache:
    return self._config_cache

  @property
  def preprocessor_cache(self) -> PreprocessorCache:
    return self._preprocessor_cache

  @property
  def cached_factory(self) -> ModuleFactoryCache | None:
    with self._lock:
      return self._cached_factory

  @cached_factory.setter
  def cached_factory(self, factory: ModuleFactoryCache | None) -> None:
    with self._lock:
      self._cached_factory = factory

  def get_cached_factory(self) -> ModuleFactoryCache | None:
    return self.cached_factory

  def set_cached_factory(self, factory: ModuleFactoryCache | None) -> None:
    self.cached_factory = factory

  def start_task(self, provider: str, model_id: str) -> bool:
    """Signals the start of a task for (provider, model_id).

    If the task is for the same model as the current session, cached state
    is preserved. If it is for a different model, caches are cleared.

    Returns:
      True if cache was preserved (session hit across tasks),
      False if caches were cleared for a new model.
    """
    new_key = (provider, model_id)
    with self._lock:
      if self._current_model_key == new_key:
        return True
      self._current_model_key = new_key
      self._config_cache.clear()
      self._preprocessor_cache.clear()
      if self._cached_factory is not None:
        self._cached_factory.clear()
        self._cached_factory = None
      return False

  def clear(self) -> None:
    """Clears the session and all underlying caches."""
    with self._lock:
      self._current_model_key = None
      self._config_cache.clear()
      self._preprocessor_cache.clear()
      if self._cached_factory is not None:
        self._cached_factory.clear()
        self._cached_factory = None


CONFIG_CACHE = ConfigCache()
PREPROCESSOR_CACHE = PreprocessorCache()
SESSION = ModuleCacheSession(CONFIG_CACHE, PREPROCESSOR_CACHE)


def clear_preprocessor_cache() -> None:
  """Clears the in-memory preprocessor and tokenizer cache."""
  PREPROCESSOR_CACHE.clear()


def clear_config_cache() -> None:
  """Clears the in-memory config and preprocessor caches for all providers."""
  CONFIG_CACHE.clear()
  PREPROCESSOR_CACHE.clear()


def clear_all_module_caches() -> None:
  """Clears all in-memory module caches and the active session."""
  SESSION.clear()
