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

# pylint: skip-file
import os
import re
import signal
import time
import torch
from torch_tpu import api
from torch_tpu._internal.device import _device_module

_DeviceModule = _device_module.DeviceModule


def safe_shutdown():
  """Release the TPU runtime (best-effort).

  Resets the Python-level singleton state so the next safe_init() call
  will re-run the full initialization path. Note: the OS-level libtpu
  lockfile is only released when the process exits.
  """
  if _DeviceModule._is_initialized:
    _DeviceModule._shutdown_runtime()
  _DeviceModule._is_initialized = False
  _DeviceModule._device_count = None
  _DeviceModule._current_device = None
  api.tpu_device.cache_clear()
  print("✅ TPU runtime state reset.")


def _wait_for_tpu_device(max_wait=10.0):
  """Wait for /dev/vfio/1 to become available after killing a stale process."""
  vfio_path = "/dev/vfio/1"
  if not os.path.exists(vfio_path):
    return  # No VFIO device, skip check
  start = time.monotonic()
  while time.monotonic() - start < max_wait:
    try:
      fd = os.open(vfio_path, os.O_RDWR)
      os.close(fd)
      return  # Device is available
    except OSError:
      time.sleep(0.5)
  print(f"⚠️  {vfio_path} still busy after {max_wait}s wait.")


def _ensure_runtime_initialized():
  """Ensure the PjRt runtime is actually initialized.

  Handles the partial-init case where ``_init_device_impl`` registered
  ``torch.tpu`` but ``_init_runtime_options`` failed.  On retry,
  ``_init_device`` short-circuits via ``hasattr(torch, 'tpu')`` and
  never calls ``_init_runtime_options``, leaving PjRt uninitialized.
  """
  if not _DeviceModule.is_initialized():
    _DeviceModule._init_runtime_options(device_type="tpu")


def safe_init(accelerator_type="v6e-4"):
  """Initialize the TPU device, handling common failure modes.

  Designed for marimo where each notebook runs in its own kernel
  process. When switching notebooks, the previous kernel's TPU lock
  must be released before the new kernel can initialize.

  Handles:

  1. **Already initialized** → returns cached device instantly.
  2. **Lock held by a stale kernel** → kills it, waits for the IOMMU
     device to be released, then retries full initialization.
  3. **Lock held by our own PID** → reuses the existing runtime.
  4. **Partial init** (module registered but runtime not started) →
     calls ``_init_runtime_options`` explicitly.

  Args:
      accelerator_type: TPU topology string (default: 'v6e-4').

  Returns:
      torch.device for the TPU.
  """
  os.environ.setdefault("ACCELERATOR_TYPE", accelerator_type)

  # ── Fast path: runtime already alive in this process ──
  if _DeviceModule.is_initialized():
    return api.tpu_device()

  # ── Slow path: first init or recovery ──
  api.tpu_device.cache_clear()

  max_retries = 3
  for attempt in range(max_retries):
    try:
      device = api.tpu_device()
      # Guard against partial init: the torch.tpu module may have
      # been registered by a prior failed attempt, causing
      # _init_device to short-circuit without calling _init_runtime_options.
      _ensure_runtime_initialized()
      return device
    except RuntimeError as e:
      err_msg = str(e)

      # ── PjRt lock conflict ──
      match = re.search(r"process with pid (\d+)", err_msg)
      if match:
        pid = int(match.group(1))
        if pid == os.getpid():
          print(f"ℹ️  TPU already held by this process (PID {pid}). Reusing.")
          return torch.device("tpu")
        else:
          print(
              f"⚠️  TPU locked by PID {pid}. Killing (attempt"
              f" {attempt+1}/{max_retries})…"
          )
          try:
            os.kill(pid, signal.SIGKILL)
          except ProcessLookupError:
            pass
          # Wait for the killed process to fully release
          # the IOMMU device handles (/dev/vfio/1).
          _wait_for_tpu_device(max_wait=10.0)
          api.tpu_device.cache_clear()
          continue

      # ── IOMMU / device busy (after a recent kill) ──
      if "device or resource busy" in err_msg.lower():
        print(
            "⚠️  TPU device still busy. Waiting (attempt"
            f" {attempt+1}/{max_retries})…"
        )
        _wait_for_tpu_device(max_wait=10.0)
        api.tpu_device.cache_clear()
        continue

      # ── Missing environment ──
      if "global device count" in err_msg.lower():
        print(
            "⚠️  Device count not set. Setting"
            f" ACCELERATOR_TYPE={accelerator_type} (attempt"
            f" {attempt+1}/{max_retries})…"
        )
        os.environ["ACCELERATOR_TYPE"] = accelerator_type
        time.sleep(1.0)
        api.tpu_device.cache_clear()
        continue

      # ── Unknown error — don't swallow it ──
      raise
  raise RuntimeError("Failed to initialize TPU after all retries.")
