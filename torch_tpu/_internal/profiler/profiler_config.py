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

import pathlib

import torch


class TpuProfilerConfig(torch.profiler._ExperimentalConfig):  # pylint: disable=protected-access
  """Configuration for TPU-specific profiler options with standard XLA defaults."""

  def __init__(
      self,
      *,
      host_tracer_level: int = 2,
      device_tracer_level: int = 1,
      python_tracer_level: int = 0,
      run_dir: pathlib.Path | None = None,
  ):
    """Initializes the TPU profiler configuration.

    The default values are chosen to align with standard XLA profiling defaults
    used in frameworks like JAX and TensorFlow. These defaults provide both host
    and TPU side tracing, where the level controls the verbosity of the data:
    - host_tracer_level=2: Captures user-instrumented TraceMe events and
    standard
      XLA annotations.
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
    """
    run_dir_posix = (
        pathlib.Path(run_dir).as_posix() if run_dir is not None else None
    )
    config_parts = [
        f"host_tracer_level:{host_tracer_level}",
        f"device_tracer_level:{device_tracer_level}",
        f"python_tracer_level:{python_tracer_level}",
    ] + ([f"run_dir:{run_dir_posix}"] if run_dir_posix is not None else [])

    # We suppress 'wrong-keyword-args' because 'custom_profiler_config' is
    # explicitly supported by the C++ implementation of _ExperimentalConfig
    # but is missing from the static type stubs (.pyi files) in the base
    # PyTorch library. This suppression allows the build to pass while
    # ensuring the configuration string is correctly passed to the backend at
    # runtime.
    super().__init__(
        custom_profiler_config=",".join(config_parts)
    )  # pytype: disable=wrong-keyword-args
