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

from contextlib import contextmanager
import dataclasses
import datetime
import os
import socket
import threading
from typing import Optional

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
    lock: A lock to protect access to the profile state.
  """

  def __init__(self):
    self.profile_session = None
    self.log_dir = None
    self.lock = threading.Lock()

  def reset(self):
    """Resets the profiler state."""
    self.profile_session = None
    self.log_dir = None


_profile_state = _ProfileState()


def start_trace(
    log_dir: os.PathLike[str] | str,
    profiler_options: ProfileOptions | None = None,
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

  Raises:
    RuntimeError: If a profile has already been started.
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
    start_trace_backend(str(log_dir), options)
    _profile_state.profile_session = True
    _profile_state.log_dir = str(log_dir)


def stop_trace() -> str:
  """Stops the currently-running profiler trace.

  The trace will be saved to the ``log_dir`` passed to the corresponding
  :func:`start_trace` call. Raises a RuntimeError if a trace hasn't been
  started.

  Returns:
    The path to the saved profiler trace directory.

  Raises:
    RuntimeError: If no profile has been started.
  """
  with _profile_state.lock:
    if _profile_state.profile_session is None:
      raise RuntimeError('No profile started')
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    plugin_dir = os.path.join(
        _profile_state.log_dir, 'plugins', 'profile', timestamp
    )
    hostname = socket.gethostname() or 'localhost'
    final_file = os.path.join(plugin_dir, f'{hostname}.xplane.pb')

    try:
      stop_trace_backend(final_file)
      logging.info('profiler saved in %s', final_file)
    finally:
      _profile_state.reset()

    return plugin_dir


@contextmanager
def trace(
    log_dir: os.PathLike[str] | str,
    profiler_options: Optional[ProfileOptions] = None,
):
  """Context manager to take a profiler trace.

  The trace will capture CPU, GPU, and/or TPU activity, including Python
  functions and on-device operations.

  The resulting trace can be viewed with TensorBoard. Note that TensorBoard
  doesn't need to be running when collecting the trace.

  Only one trace may be collected at a time. A RuntimeError will be raised if a
  trace is started while another trace is running.

  Args:
    log_dir: The directory to save the profiler trace to (usually the
      TensorBoard log directory).
    profiler_options: Profiler options to configure the profiler for collection.

  Yields:
    None
  """
  start_trace(log_dir, profiler_options)

  try:
    yield
  finally:
    stop_trace()
