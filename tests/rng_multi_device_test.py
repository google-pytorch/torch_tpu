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

"""Multi-device RNG reference tests comparing TPU and GPU behaviors."""

import functools
import struct
import threading
import unittest

from absl import flags
from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal.utils import annotations
from tests import accelerator_test_utils
from tests import seed_test_utils

_BACKEND = flags.DEFINE_string(
    "backend", "tpu", "The backend to test: 'tpu' or 'gpu'."
)


def setUpModule():
  # The tests compare the internal `torch.tpu` RNG APIs with `torch.cuda`.
  annotations._internal_apis_allowed_for_testing = True  # pylint: disable=protected-access


def _fails_on_tpu(reason: str):
  """Decorator that asserts the test fails on TPU."""

  def decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
      if self.backend == "tpu":
        try:
          func(self, *args, **kwargs)
        except unittest.SkipTest:
          raise
        except Exception as e:  # pylint: disable=broad-exception-caught
          # Expected failure on TPU due to backend limitation or missing feature.
          logging.exception(
              "_fails_on_tpu swallowed expected failure (%s): %r", reason, e
          )
          return
        self.fail(
            f"Expected test to fail on TPU ({reason}), but it unexpectedly"
            " succeeded."
        )
      return func(self, *args, **kwargs)

    return wrapper

  return decorator


class _BaseRngTest(seed_test_utils.RepeatableTest):
  """Base test class providing common fixtures and helpers for device RNG tests."""

  def setUp(self):
    super().setUp()
    self.backend = _BACKEND.value
    self.backend_mod = accelerator_test_utils.get_backend_module(self.backend)
    self.device = accelerator_test_utils.get_device(self.backend)

  def _unpack_device_rng_uint64(
      self, start_byte: int, end_byte: int, device_idx: int | None = None
  ) -> int:
    """Unpacks a uint64 value from a byte slice of the device RNG state."""
    state = (
        self.backend_mod.get_rng_state()
        if device_idx is None
        else self.backend_mod.get_rng_state(device_idx)
    )
    (val,) = struct.unpack("<Q", state[start_byte:end_byte].numpy().tobytes())
    return val

  def _get_device_rng_seed(self, device_idx: int | None = None) -> int:
    """Returns the RNG seed for the specified device.

    Args:
      device_idx: The index of the device to query. Defaults to the current
        device if not specified.
    """
    return self._unpack_device_rng_uint64(0, 8, device_idx)

  def _get_device_rng_offset(self, device_idx: int | None = None) -> int:
    """Returns the RNG offset for the specified device.

    Args:
      device_idx: The index of the device to query. Defaults to the current
        device if not specified.
    """
    return self._unpack_device_rng_uint64(8, 16, device_idx)


class SingleProcessMultiDeviceTest(_BaseRngTest):
  """Tests documenting single-process multi-device RNG differences.

  Single-process multi-device execution is discouraged in PyTorch CUDA due
  to Python GIL contention (PyTorch recommends 1 process per GPU via DDP;
  see
  https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel)
  and is a current TPU runtime limitation planned to be addressed (see
  b/555281188).
  """

  def setUp(self):
    super().setUp()
    self.device_idx = torch.accelerator.current_device_index()
    self.num_devices = self._get_num_devices()
    self.assertGreater(
        self.num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " single-process multi-device RNG operations.",
    )

  def _get_num_devices(self) -> int:
    """Returns the number of devices to use for multi-device testing."""
    if self.backend == "tpu":
      return accelerator_test_utils.get_tpu_device_count()
    return torch.accelerator.device_count()

  @_fails_on_tpu(
      "TODO(b/555281188): Querying non-current device RNG state is not yet"
      " supported on TPU due to single-process multi-device runtime"
      " limitations."
  )
  def test_torch_manual_seed_sets_all_device_seeds(self):
    """Verifies torch.manual_seed seeds all devices."""
    torch.manual_seed(42)

    for i in range(self.num_devices):
      self.assertEqual(self._get_device_rng_seed(i), 42)
      self.assertEqual(self._get_device_rng_offset(i), 0)

  @_fails_on_tpu(
      "TODO(b/555281188): Querying non-current device RNG state is not yet"
      " supported on TPU due to single-process multi-device runtime"
      " limitations."
  )
  def test_backend_manual_seed_does_not_change_other_devices(self):
    """Verifies backend_mod.manual_seed does not modify other devices."""
    torch.manual_seed(10)
    other_devices_state_before = {
        i: (self._get_device_rng_seed(i), self._get_device_rng_offset(i))
        for i in range(self.num_devices)
        if i != self.device_idx
    }

    self.backend_mod.manual_seed(42)

    self.assertEqual(self._get_device_rng_seed(self.device_idx), 42)
    self.assertEqual(self._get_device_rng_offset(self.device_idx), 0)
    for i, (seed_before, offset_before) in other_devices_state_before.items():
      self.assertEqual(
          self._get_device_rng_seed(i),
          seed_before,
          msg=f"Device {i} seed mismatch",
      )
      self.assertEqual(
          self._get_device_rng_offset(i),
          offset_before,
          msg=f"Device {i} offset mismatch",
      )

  @_fails_on_tpu(
      "TODO(b/555281188): Querying non-current device RNG state is not yet"
      " supported on TPU due to single-process multi-device runtime"
      " limitations."
  )
  def test_backend_manual_seed_all_sets_all_device_seeds(self):
    """Verifies backend_mod.manual_seed_all seeds all devices."""
    self.backend_mod.manual_seed_all(42)

    for i in range(self.num_devices):
      self.assertEqual(
          self._get_device_rng_seed(i),
          42,
          msg=f"Device {i} seed mismatch",
      )
      self.assertEqual(
          self._get_device_rng_offset(i),
          0,
          msg=f"Device {i} offset mismatch",
      )

  @_fails_on_tpu(
      "TODO(b/555281188): Querying non-current device RNG state is not yet"
      " supported on TPU due to single-process multi-device runtime"
      " limitations."
  )
  def test_torch_seed_sets_all_device_seeds(self):
    """Verifies torch.seed seeds all devices."""
    for i in range(self.num_devices):
      _ = torch.rand(
          100, device=accelerator_test_utils.get_device(self.backend, i)
      )

    new_seed = torch.seed()

    for i in range(self.num_devices):
      self.assertEqual(
          self._get_device_rng_seed(i),
          new_seed,
          msg=f"Device {i} seed mismatch",
      )
      self.assertEqual(
          self._get_device_rng_offset(i),
          0,
          msg=f"Device {i} offset mismatch",
      )

  @_fails_on_tpu(
      "TODO(b/555281188): get_rng_state_all / set_rng_state_all is not yet"
      " supported on TPU due to single-process multi-device runtime"
      " limitations."
  )
  def test_backend_get_set_rng_state_all(self):
    """Verifies get/set_rng_state_all restore all device states."""
    for i in range(self.num_devices):
      self.backend_mod.manual_seed(42 + i)

    saved_states = self.backend_mod.get_rng_state_all()
    self.assertLen(saved_states, self.num_devices)

    expected_outs = [
        torch.rand(
            10, device=accelerator_test_utils.get_device(self.backend, i)
        )
        for i in range(self.num_devices)
    ]

    for i in range(self.num_devices):
      _ = torch.rand(
          50, device=accelerator_test_utils.get_device(self.backend, i)
      )

    self.backend_mod.set_rng_state_all(saved_states)

    actual_outs = [
        torch.rand(
            10, device=accelerator_test_utils.get_device(self.backend, i)
        )
        for i in range(self.num_devices)
    ]
    for exp, act in zip(expected_outs, actual_outs):
      self.assertTrue(torch.equal(exp, act))

  @_fails_on_tpu(
      "TODO(b/555281188): Single-process multi-device execution is not yet"
      " supported on TPU."
  )
  def test_multithread_eager_cross_device_isolation(self):
    """Verifies threads on distinct devices produce deterministic outputs.

    Each thread switches to its own device index, seeds that device's default
    generator and draws random numbers concurrently with the other threads.
    The per-device RNG streams must be independent of each other and of thread
    scheduling.
    """
    num_devices = self.num_devices

    def run_trial() -> list[torch.Tensor]:
      """Runs a multi-threaded cross-device trial and returns outputs.

      Returns:
        List of generated tensors per device, ordered by device index.
      """
      results = {}
      errors = {}

      def thread_worker(thread_id):
        try:
          dev = accelerator_test_utils.get_device(self.backend, thread_id)
          with torch.accelerator.device_index(thread_id):
            self.backend_mod.manual_seed(42 + thread_id)
            self.assertEqual(
                self._get_device_rng_seed(thread_id), 42 + thread_id
            )
            tensors = [torch.rand(10, device=dev) for _ in range(5)]
            results[thread_id] = torch.stack(tensors).cpu()
        except Exception as e:  # pylint: disable=broad-exception-caught
          errors[thread_id] = e

      threads = [
          threading.Thread(target=thread_worker, args=(i,))
          for i in range(num_devices)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      if errors:
        failed = ", ".join(
            f"device {i}: {e!r}" for i, e in sorted(errors.items())
        )
        raise AssertionError(
            f"{len(errors)} of {num_devices} threads failed: {failed}"
        ) from next(iter(errors.values()))

      return [results[i] for i in range(num_devices)]

    trial1 = run_trial()
    trial2 = run_trial()

    for res1, res2 in zip(trial1, trial2):
      self.assertTrue(torch.equal(res1, res2))
    self.assertFalse(torch.equal(trial1[0], trial1[1]))


if __name__ == "__main__":
  absltest.main()
