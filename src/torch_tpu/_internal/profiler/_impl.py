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

"""xprof_profiler API for internal use."""

import dataclasses
import datetime
import os
import socket
import threading

from absl import logging
from torch_tpu._internal.profiler._profiler_backend import start_profiler_server
from torch_tpu._internal.profiler._profiler_backend import start_trace as start_trace_backend
from torch_tpu._internal.profiler._profiler_backend import stop_profiler_server
from torch_tpu._internal.profiler._profiler_backend import stop_trace as stop_trace_backend


_profiler_server = None


@dataclasses.dataclass
class ProfileOptions:
  """Options for profiling."""

  host_tracer_level: int | None = None
  device_tracer_level: int | None = None
  python_tracer_level: int | None = None


def start_server(port: int):
  """Starts the profiler server on port `port`.

  Using the "TensorFlow profiler" feature in `TensorBoard
  <https://www.tensorflow.org/tensorboard>`_ 2.2 or newer, you can
  connect to the profiler server and sample execution traces that show CPU,
  GPU, and/or TPU device activity.

  Args:
    port: The port number to start the server on.

  Returns:
    Port number the server is running on.

  Raises:
    ValueError: If a profiler server is already running.
  """
  global _profiler_server
  if _profiler_server is not None:
    raise ValueError('Only one profiler server can be active at a time.')
  start_profiler_server(port)
  _profiler_server = port
  return port


def stop_server():
  """Stops the running profiler server.

  Raises:
    ValueError: If no profiler server is running.
  """
  global _profiler_server
  if _profiler_server is None:
    raise ValueError('No active profiler server.')
  _profiler_server = None
  stop_profiler_server()


class _ProfileState:
  """Internal class to hold the state of the profiler.

  Attributes:
    profile_session: The active profiler session, or None if no session is
      active.
    log_dir: The directory to save the profiler trace to.
    worker_rank: The worker rank for the trace file naming.
    lock: A lock to protect access to the profile state.
  """

  def __init__(self):
    self.profile_session = None
    self.log_dir = None
    self.worker_rank = None
    self.lock = threading.Lock()

  def reset(self):
    """Resets the profiler state."""
    self.profile_session = None
    self.log_dir = None
    self.worker_rank = None


_profile_state = _ProfileState()


def start_trace(
    log_dir: os.PathLike[str] | str,
    profiler_options: ProfileOptions | None = None,
    worker_rank: str | None = None,
) -> None:
  """Starts a profiler trace.

  The trace will capture CPU, GPU, and/or TPU activity, including Python
  functions and on-device operations. Use :func:`stop_trace` to end the
  trace
  and save the results to ``log_dir``.

  The resulting trace can be viewed with TensorBoard. Note that TensorBoard
  doesn't need to be running when collecting the trace.

  Only one trace may be collected at a time. A RuntimeError will be raised if
  :func:`start_trace` is called while another trace is running.

  Args:
    log_dir: The directory to save the profiler trace to (usually the
      TensorBoard log directory).
    profiler_options: Profiler options to configure the profiler for collection.
    worker_rank: Optional worker rank to append to the trace filename.

  Raises:
    RuntimeError: If a profile has already been started.
    ValueError: If worker_rank contains path separators.
  """
  with _profile_state.lock:
    if _profile_state.profile_session is not None:
      raise RuntimeError(
          'Profile has already been started. '
          'Only one profile may be run at a time.'
      )

    options = None
    if profiler_options:
      options = dataclasses.asdict(
          profiler_options,
          dict_factory=lambda x: {k: v for k, v in x if v is not None},
      )

    if worker_rank is not None:
      if '/' in worker_rank or '\\' in worker_rank:
        raise ValueError(
            f'worker_rank cannot contain path separators: {worker_rank}'
        )

    start_trace_backend(str(log_dir), options)
    _profile_state.profile_session = True
    _profile_state.log_dir = str(log_dir)
    _profile_state.worker_rank = worker_rank


def stop_trace() -> None:
  """Stops the currently-running profiler trace.

  The trace will be saved to the ``log_dir`` passed to the corresponding
  :func:`start_trace` call. Raises a RuntimeError if a trace hasn't been
  started.

  Raises:
    RuntimeError: If no profile has been started.
  """
  with _profile_state.lock:
    if _profile_state.profile_session is None:
      raise RuntimeError('No profile started')
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    plugin_dir = os.path.join(  # pyrefly: ignore[no-matching-overload]
        _profile_state.log_dir, 'plugins', 'profile', timestamp
    )
    hostname = socket.gethostname() or 'localhost'
    if _profile_state.worker_rank is not None:
      filename = f'{hostname}_{_profile_state.worker_rank}.xplane.pb'
    else:
      filename = f'{hostname}.xplane.pb'
    final_file = os.path.join(plugin_dir, filename)

    try:
      stop_trace_backend(final_file)
      logging.info('profiler saved in %s', final_file)
    finally:
      _profile_state.reset()
