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

"""TPU specific profiler configuration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import os
from typing import TypeAlias

import torch
from torch_tpu._internal.utils import annotations


_ExperimentalValue: TypeAlias = int | bool | str

# LINT.IfChange(option_keys)
# go/keep-sorted start
_PK_CHECK_EXPERIMENTAL_OPTIONS = "check_experimental_options"
_PK_DEVICE_TRACER_LEVEL = "device_tracer_level"
_PK_HOST_TRACER_LEVEL = "host_tracer_level"
_PK_PYTHON_TRACER_LEVEL = "python_tracer_level"
_PK_RUN_DIR = "run_dir"
_PK_WORKER_RANK = "worker_rank"
# go/keep-sorted end
# LINT.ThenChange(profiler_config.py:stable_keys_set)


# LINT.IfChange(stable_keys_set)
# Profiler config option keys that are stable (not experimental).
_STABLE_KEYS = frozenset({
    # go/keep-sorted start
    _PK_CHECK_EXPERIMENTAL_OPTIONS,
    _PK_DEVICE_TRACER_LEVEL,
    _PK_HOST_TRACER_LEVEL,
    _PK_PYTHON_TRACER_LEVEL,
    _PK_RUN_DIR,
    _PK_WORKER_RANK,
    # go/keep-sorted end
})
# LINT.ThenChange(profiler_config.py:option_keys)


def _format_experimental_value(value: _ExperimentalValue) -> str:
  r"""Formats an experimental option value for the config string.

  Rules:
  - Strings: Wrapped in double quotes. Internal backslashes are escaped to
    double backslashes, and internal double quotes are escaped to
    backslash-quote.
    E.g., `some\path` -> `"some\\path"`, `a "quote"` -> `"a \"quote\""`.
  - Booleans: Formatted as "true" or "false".
  - Integers: Formatted as their string representation.

  Args:
    value: The value to format (must be str, bool, or int).

  Returns:
    The formatted string representation.
  """
  if isinstance(value, str):
    escaped_value = value.replace("\\", r"\\").replace('"', r"\"")
    return f'"{escaped_value}"'
  elif isinstance(value, bool):
    return "true" if value else "false"
  else:
    assert isinstance(value, int) and not isinstance(
        value, bool
    ), f"Expected int value, got {type(value)}"
    return str(value)


@annotations.experimental(
    "torch.tpu.profiler.TpuProfilerConfig is experimental and subject to"
    " change."
)
class TpuProfilerConfig(torch.profiler._ExperimentalConfig):  # pylint: disable=protected-access
  """Configuration for TPU-specific profiler options with standard XLA defaults."""

  def __init__(
      self,
      *,
      host_tracer_level: int = 2,
      device_tracer_level: int = 1,
      python_tracer_level: int = 0,
      run_dir: os.PathLike[str] | str | None = None,
      worker_rank: str | None = None,
      experimental_options: Mapping[str, _ExperimentalValue] | None = None,
      check_experimental_options: bool = True,
  ):
    """Initializes the TPU profiler configuration.

    The default values are chosen to align with standard XLA profiling defaults
    used in frameworks like JAX and TensorFlow. These defaults provide both host
    and TPU side tracing, where the level controls the verbosity of the data:
    - host_tracer_level=2: Captures user-instrumented TraceMe events and
      standard XLA annotations.
    - device_tracer_level=1: Enables hardware-level TPU activity tracing.
    - python_tracer_level=0: Disabled by default to minimize overhead.

    Args:
      host_tracer_level: Controls host-side tracing verbosity. 0=disabled,
        1=user-instrumented tracemes, 2=1+XLA tracemes, 3=2+low-level XLA
        tracemes.
      device_tracer_level: Controls TPU device-side tracing. 0=disabled,
        1=enabled.
      python_tracer_level: Controls Python-side tracing. 0=disabled, 1=enabled.
      run_dir: The directory path to save the profiler trace to. If None
        (default), falls back to Kineto's `activitiesLogFile` directory (which
        defaults to "/tmp" if no handler is provided). This ensures TPU and CPU
        traces are co-located in the same directory.
      worker_rank: The rank of the worker. If provided, the generated XPlane
        filename will include the rank as a suffix (e.g.,
        `<hostname>_<rank>.xplane.pb` instead of `<hostname>.xplane.pb`). Must
        be a string.
      experimental_options: Dictionary of experimental profiler options where
        supported value types are int, bool, and str. Strings will be
        automatically quoted and escaped, and keys must not contain colons or
        commas.
      check_experimental_options: If True (default), strictly validates that
        experimental keys are recognized by the underlying profiling compiler
        backend.
    """
    if worker_rank is not None:
      if not isinstance(worker_rank, str):
        raise TypeError(
            f"worker_rank must be a string, got {type(worker_rank)}"
        )
      if ":" in worker_rank or "," in worker_rank:
        raise ValueError(
            f"worker_rank string cannot contain ':' or ',': {worker_rank!r}"
        )

    run_dir_posix = os.fspath(run_dir) if run_dir is not None else None
    for key in experimental_options or {}:
      if key in _STABLE_KEYS:
        raise ValueError(
            f"Experimental option key cannot clash with stable option: {key!r}"
        )
      if ":" in key or "," in key:
        raise ValueError(
            f"Experimental option keys cannot contain ':' or ',': {key!r}"
        )

    def _get_parts() -> Iterator[str]:
      yield f"{_PK_HOST_TRACER_LEVEL}:{host_tracer_level}"
      yield f"{_PK_DEVICE_TRACER_LEVEL}:{device_tracer_level}"
      yield f"{_PK_PYTHON_TRACER_LEVEL}:{python_tracer_level}"
      if run_dir_posix is not None:
        yield f"{_PK_RUN_DIR}:{run_dir_posix}"
      if worker_rank is not None:
        yield f"{_PK_WORKER_RANK}:{_format_experimental_value(worker_rank)}"
      if experimental_options:
        for key, value in experimental_options.items():
          yield f"{key}:{_format_experimental_value(value)}"
        yield f"{_PK_CHECK_EXPERIMENTAL_OPTIONS}:{str(check_experimental_options).lower()}"

    config_parts = list(_get_parts())

    # We suppress 'wrong-keyword-args' because 'custom_profiler_config' is
    # explicitly supported by the C++ implementation of _ExperimentalConfig
    # but is missing from the static type stubs (.pyi files) in the base
    # PyTorch library. This suppression allows the build to pass while
    # ensuring the configuration string is correctly passed to the backend at
    # runtime.
    super().__init__(
        custom_profiler_config=",".join(config_parts)
    )  # pytype: disable=wrong-keyword-args
