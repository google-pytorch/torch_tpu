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

import contextlib
import dataclasses
import datetime
import functools
from typing import Any

from torch_tpu._internal.compiler_options import compiler_options_impl as compiler
from torch_tpu._internal.device import _device_module
from torch_tpu._internal.device import _device_ops_backend
from torch_tpu._internal.utils import annotations


@annotations.forbid_in_torch_compile(
    "A compiled graph is lowered to a single executable and has one set of "
    "XLA compiler options for all of its ops."
)
@annotations.experimental(
    "torch.tpu.compiler.custom_compiler_options is experimental and subject to"
    " change."
)
@contextlib.contextmanager
def custom_compiler_options(options: dict[str, str]):
  """Context manager for setting the XLA compiler options.

  .. warning::
      This API is experimental and subject to change in future releases.

  Args:
    options: A dictionary of XLA compiler options. The key is the option name,
      and the value is the option value.  In addition, we also support these
      special options: "xla_optimization_level": the XLA optimization level.
      Valid options are 'O0', 'O1', 'O2', 'O3'. "xla_memory_fitting_level": the
      XLA memory fitting level. Valid options are 'O0', 'O1', 'O2', 'O3'. See
      EffortLevel in
      https://github.com/openxla/xla/blob/main/xla/xla.proto for more details.

  For example:
    with custom_compiler_options({
        "xla_optimization_level": "O1",
        "xla_tpu_enable_deduplicated_calls": "ENABLED",
    }):
      result = torch.matmul(torch.ones((1, 1)), torch.ones((1, 1)))
      result.to("cpu")

  All operations *materialized* within the context will use the provided
  compiler options.

  When nested, compile options overrides from the inner context merge with the
  outer context, with the inner options taking precedence.

  For example:
    with custom_compiler_options({"xla_optimization_level": "O1"}):
      outer = torch.matmul(...)
      outer.to("cpu")  # Compiled with {"xla_optimization_level": "O1"}

      with custom_compiler_options({"xla_optimization_level": "O2"}):
        inner = torch.matmul(...)
        inner.to("cpu")  # Compiled with {"xla_optimization_level": "O2"}
  """

  # See
  # https://openxla.org/xla/flags_guidance
  # for available options.
  compiler.push_compiler_options(options)
  try:
    yield
  finally:
    # Guarantee the state is restored even if Python code throws an exception.
    compiler.pop_compiler_options()


@annotations.experimental("In-memory compilation cache entry statistics.")
@dataclasses.dataclass(frozen=True)
class CacheEntryStats:
  """Statistics of a compiled program in the compilation cache.

  Attributes:
    compilation_duration: How long it took to compile the program.
    last_read: When the program was last read from the cache.
    read_count: How many times the program was read from the cache.
  """

  compilation_duration: datetime.timedelta
  last_read: datetime.datetime
  read_count: int

  __tt_api_stages__ = {
      "compilation_duration": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="In-memory compilation cache entry statistics.",
      ),
      "last_read": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="In-memory compilation cache entry statistics.",
      ),
      "read_count": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="In-memory compilation cache entry statistics.",
      ),
  }


@annotations.experimental("In-memory compilation cache statistics.")
@dataclasses.dataclass(frozen=True)
class CacheStats:
  """Statistics of the in-memory compilation cache.

  Attributes:
    num_cache_reqs: How many times a program was looked up in the cache.
    num_cache_hits: How many of those lookups found the program in the cache.
    num_cache_misses: How many of those lookups didn't find the program, so it
      was compiled or loaded from a persistent compilation cache.
    num_compilations: How many programs were compiled.
    per_entry_stats: The statistics of each program in the cache.
  """

  num_cache_reqs: int
  num_cache_hits: int
  num_cache_misses: int
  num_compilations: int
  # The backend's snapshot. `per_entry_stats` converts its entries on first
  # access, as that takes much longer than reading the counters.
  _backend_stats: Any = dataclasses.field(repr=False, compare=False)

  @functools.cached_property
  def per_entry_stats(self) -> tuple[CacheEntryStats, ...]:
    return tuple(
        CacheEntryStats(
            compilation_duration=entry.compilation_duration,
            last_read=entry.last_read,
            read_count=entry.read_count,
        )
        for entry in self._backend_stats.per_entry_stats
    )

  __tt_api_stages__ = {
      "num_cache_reqs": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="In-memory compilation cache statistics.",
      ),
      "num_cache_hits": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="In-memory compilation cache statistics.",
      ),
      "num_cache_misses": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="In-memory compilation cache statistics.",
      ),
      "num_compilations": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="In-memory compilation cache statistics.",
      ),
      "per_entry_stats": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="In-memory compilation cache statistics.",
      ),
  }


@annotations.experimental("In-memory compilation cache statistics.")
def cache_stats() -> CacheStats:
  """Returns a snapshot of the statistics of the in-memory compilation cache."""
  stats = _device_ops_backend._get_cache_stats()  # pylint: disable=protected-access
  return CacheStats(
      num_cache_reqs=stats.num_cache_reqs,
      num_cache_hits=stats.num_cache_hits,
      num_cache_misses=stats.num_cache_reqs - stats.num_cache_hits,
      num_compilations=stats.num_compilations,
      _backend_stats=stats,
  )


@annotations.experimental("In-memory compilation cache eviction.")
def clear_cache() -> None:
  """Evicts all programs from the in-memory compilation cache.

  Waits for in-flight compilations to finish first, and also drops the compiled
  executables that `torch.compile(backend="tpu")` retains.
  """
  _device_module._DeviceModule._clear_cache()  # pylint: disable=protected-access


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "CacheEntryStats",
    "CacheStats",
    "cache_stats",
    "clear_cache",
    "custom_compiler_options",
    # go/keep-sorted end
]
