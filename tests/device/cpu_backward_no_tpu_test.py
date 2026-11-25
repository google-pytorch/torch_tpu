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

"""Tests CPU-only work on hosts without TPU device nodes."""

from collections.abc import Callable
import os
import sys
import time
import traceback
import warnings

from absl.testing import absltest
import torch
from torch._subclasses import fake_tensor
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.device import _device_ops_backend

# Initializing the TPU runtime on a host without TPUs could take ~10 minutes
# before it failed. Failing without TPU device nodes takes milliseconds.
_MAX_SECONDS = 30


def _tpu_device() -> str:
  backend = torch._C._get_privateuse1_backend_name()  # pylint: disable=protected-access
  return f"{backend}:0"


def _run_in_child(scenario: Callable[[], None]) -> int:
  """Runs `scenario` in a forked child process and returns its exit code.

  The TPU runtime caches its initialization result for the whole process, and
  this process never initializes it, so each scenario starts from scratch.

  Args:
    scenario: The function to run. It fails by raising an exception, whose
      traceback is printed to stderr.

  Returns:
    0 if `scenario` returned, 1 if it raised an exception.
  """
  pid = os.fork()
  if pid == 0:  # Child process.
    exit_code = 1
    try:
      scenario()
      exit_code = 0
    except BaseException:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
    finally:
      sys.stderr.flush()
      os._exit(exit_code)  # pylint: disable=protected-access
  _, status = os.waitpid(pid, 0)
  return os.waitstatus_to_exitcode(status)


def _get_tpu_runtime_error() -> str:
  """Returns the TPU runtime's initialization error."""
  try:
    _device_ops_backend._get_device_count()  # pylint: disable=protected-access
  except RuntimeError as e:
    return str(e)
  raise AssertionError("expected the TPU runtime to be unavailable")


class CpuBackwardNoTpuTest(
    absltest.TestCase  # ABSLTEST_OK=no randomness to seed
):

  def setUp(self):
    super().setUp()
    # This test is flaky (~5% failure rate on CPU machines): some CPU hosts
    # happen to have non-TPU device files (like VFIO or accel) in /dev. When
    # those files are present, TorchTPU assumes a TPU might be attached and
    # tries to start PjRt, so we skip the test on those hosts.
    if tt_testing.has_tpu_device_nodes():
      self.skipTest(
          "Host has VFIO/accel device files, so PjRt initialization will be "
          "attempted."
      )
    # Scenarios run in forked children, which need a fresh TPU runtime state.
    self.assertFalse(tt_testing.pjrt_init_attempted())

  def assert_scenario_passes(self, scenario: Callable[[], None]):
    self.assertEqual(
        _run_in_child(scenario), 0, "scenario failed, see its traceback above"
    )

  def _cpu_backward_scenario(self):
    # The autograd engine asks every registered device type for its device
    # count on the first backward(), even for a CPU-only graph. On a host
    # without TPU device nodes, that must not run the slow TPU runtime
    # initialization.
    start = time.monotonic()
    x = torch.ones(2, requires_grad=True)
    (x * 3).sum().backward()

    self.assertLess(time.monotonic() - start, _MAX_SECONDS)
    self.assertEqual(x.grad.tolist(), [3.0, 3.0])
    self.assertTrue(tt_testing.pjrt_init_attempted())
    self.assertIn("no TPU device found: ", _get_tpu_runtime_error())

  def test_cpu_backward_fails_tpu_init_fast(self):
    self.assert_scenario_passes(self._cpu_backward_scenario)

  def _fake_tpu_tensor_ops_scenario(self):
    # Autograd ops on fake TPU tensors (e.g. when tracing TPU code on a CPU
    # host) enter the TPU device guard, which validates the device index
    # against the TPU runtime. On a host without TPU device nodes, that must
    # not run the slow TPU runtime initialization, nor warn, since the runtime
    # is expected to be unavailable there.
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      with fake_tensor.FakeTensorMode():
        x = torch.empty((4, 8), device=_tpu_device(), requires_grad=True)
        y = x * 2

    self.assertLess(time.monotonic() - start, _MAX_SECONDS)
    self.assertEqual(y.shape, (4, 8))
    self.assertTrue(tt_testing.pjrt_init_attempted())
    self.assertIn("no TPU device found: ", _get_tpu_runtime_error())
    self.assertEmpty(
        [w.message for w in caught if "PJRT client" in str(w.message)]
    )

  def test_fake_tpu_tensor_ops_fail_tpu_init_fast_without_warning(self):
    self.assert_scenario_passes(self._fake_tpu_tensor_ops_scenario)

  def _tpu_runtime_init_scenario(self):
    start = time.monotonic()
    error = _get_tpu_runtime_error()

    self.assertLess(time.monotonic() - start, _MAX_SECONDS)
    self.assertIn("no TPU device found: no TPU device node exists", error)

  def test_tpu_runtime_init_fails_fast(self):
    self.assert_scenario_passes(self._tpu_runtime_init_scenario)

  def _assert_tpu_runtime_unavailable(self):
    self.assertIn("no TPU device found: ", _get_tpu_runtime_error())

  def _fork_scenario(self):
    self._assert_tpu_runtime_unavailable()
    # The TPU runtime never started, so a forked grandchild (e.g. a DataLoader
    # worker) must not be considered to be in a bad fork, and gets the same
    # error.
    self.assertEqual(_run_in_child(self._assert_tpu_runtime_unavailable), 0)

  def test_fork_after_failed_tpu_runtime_init(self):
    self.assert_scenario_passes(self._fork_scenario)


if __name__ == "__main__":
  absltest.main()
