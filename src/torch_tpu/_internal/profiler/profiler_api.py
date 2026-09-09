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

"""torch_tpu._internal.profiler API."""

from collections.abc import Callable
import contextlib
import enum
import os
from typing import Any
import warnings

from torch_tpu._internal.profiler import _profiler_backend
import torch_tpu._internal.profiler._impl as profiler


class ProfilerActivity(enum.Enum):
  """Enum to specify the activities to profile.

  Attributes:
    CPU: Profile CPU activities.
    TPU: Profile TPU activities.
    GPU: Profile GPU activities.
  """

  CPU = 1
  TPU = 2
  GPU = 3


def xprof_trace_handler(
    dir_name: os.PathLike[str] | str,
    worker_rank: str | None = None,
) -> Callable[[Any], str]:
  """Standard handler to be used with the `on_trace_ready` argument in `profiler.profile`.

  The returned handler function simply returns the provided `dir_name`,
  indicating where the profiling trace files should be saved.

  Args:
    dir_name: The directory path to save the profiler trace to.
    worker_rank: Optional worker rank to append to the trace filename.

  Returns:
    A function that takes a profiler object (currently unused) and returns the
    directory name.

  Example:
    >>> from torch_tpu._internal import profiler
    >>> handler = profiler.xprof_trace_handler(dir_name="/tmp/profile_data")
    >>> with profiler.profile(
    ...   activities=[profiler.ProfilerActivity.TPU],
    ...   on_trace_ready=handler
    ... ):
    ...   # Your code to profile here
    ...   pass
  """

  def handler_fn(prof: Any) -> str:
    del prof  # Unused in this handler
    return str(dir_name)

  handler_fn.worker_rank = worker_rank  # pyrefly: ignore[missing-attribute]
  return handler_fn


def _get_profile_options(
    activities: list[ProfilerActivity],
) -> profiler.ProfileOptions:
  """Determines the profiler options based on the selected activities."""
  cpu_active = ProfilerActivity.CPU in activities
  tpu_active = ProfilerActivity.TPU in activities
  gpu_active = ProfilerActivity.GPU in activities

  options = profiler.ProfileOptions()

  if gpu_active:
    raise ValueError("GPU profiling is not supported.")

  # Disable tracer options by default.
  # For descriptions of the levels, see:
  # https://www.tensorflow.org/guide/profiler#profiler_options
  options.python_tracer_level = 0
  options.device_tracer_level = 0
  options.host_tracer_level = 0

  if tpu_active:
    # Enables device tracing.
    options.device_tracer_level = 1

  if cpu_active:
    # Enables host tracing for tf.data, framework ops, user defined TraceMe,
    # and Silva CPU profiler events.
    options.host_tracer_level = 2
    # Enables python tracing.
    options.python_tracer_level = 1

  return options


# TODO(b/509670300): remove it when the native PyTorch profiler API is fully
# integrated and available.
@contextlib.contextmanager
def profile(
    activities: list[ProfilerActivity] | None = None,
    on_trace_ready: Callable[[Any], str] | None = None,
):
  """Context manager to take a profiler trace, compatible with torch.profiler.

  Example usage:
  ```python
  from torch_tpu._internal import profiler

  with profiler.profile(
    activities=[profiler.ProfilerActivity.TPU],
    on_trace_ready=profiler.xprof_trace_handler(dir_name="/path/to/logdir")
  ) as prof:
      trainer.train()
  ```

  Args:
    activities: List of activities to profile. Defaults to
      [ProfilerActivity.CPU, ProfilerActivity.TPU] if not provided.
    on_trace_ready: A callable that takes the profiler object as argument and
      returns the directory to save the trace to.
  """
  if not on_trace_ready:
    raise ValueError("on_trace_ready must be provided for profiler.profile.")

  if activities is None:
    warnings.warn(
        "No activities provided to profiler.profile. Defaulting to CPU and TPU"
        " profiling."
    )
    activities = [ProfilerActivity.CPU, ProfilerActivity.TPU]

  options = _get_profile_options(activities)
  # TODO: b/458773929 - update on_trace_ready to match native torch.profiler
  # We don't have a profiler object to pass to the handler yet.
  # This design is a bit different from torch.profiler where the handler
  # is called after the trace is stopped.
  # For now, we call it without a profiler object.
  log_dir = on_trace_ready(None)
  worker_rank = getattr(on_trace_ready, "worker_rank", None)

  profiler.start_trace(
      log_dir, profiler_options=options, worker_rank=worker_rank
  )
  # pylint: disable-next=protected-access
  _profiler_backend._push_enable_profiler()

  try:
    yield None
  finally:
    # pylint: disable-next=protected-access
    _profiler_backend._pop_enable_profiler()
    profiler.stop_trace()


def register_kineto_backend() -> None:
  """Explicitly registers the Kineto backend for TPU profiling."""
  # Registration happens via static initializers when the module is imported.
  # This function serves as an explicit entry point to avoid unused imports.
  pass
