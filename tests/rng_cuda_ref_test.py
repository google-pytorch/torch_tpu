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

import functools
import struct
from typing import Any
import unittest

from absl import flags
from absl.testing import absltest
import torch
from tests import seed_test_utils

_BACKEND = flags.DEFINE_string(
    "backend", "tpu", "The backend to test: 'tpu' or 'gpu'."
)


def _get_backend_module(backend_name: str) -> Any:
  if backend_name == "gpu":
    return torch.cuda
  elif backend_name == "tpu":
    return torch.tpu
  raise ValueError(
      f"Unsupported backend '{backend_name}'. Supported backends are: 'gpu',"
      " 'tpu'"
  )


def _get_device(
    backend_name: str, device_idx: int | None = None
) -> torch.device:
  """Returns a torch.device for the given backend name and optional device index.

  Maps the CLI flag backend names ('gpu', 'tpu') to their corresponding PyTorch
  device types ('cuda', 'tpu').

  Args:
    backend_name: Name of the backend ('gpu' or 'tpu').
    device_idx: Optional integer index of the device (e.g. 0 for 'cuda:0'). If
      None, returns a device without an index (e.g. 'cuda').
  """
  if backend_name == "gpu":
    device_type = "cuda"
  elif backend_name == "tpu":
    device_type = "tpu"
  else:
    raise ValueError(
        f"Unsupported backend '{backend_name}'. Supported backends are: 'gpu',"
        " 'tpu'"
    )
  if device_idx is None:
    return torch.device(device_type)
  return torch.device(f"{device_type}:{device_idx}")


def _fail_on_tpu(reason: str):
  """Decorator that asserts the test fails on TPU."""

  def decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
      if self.backend == "tpu":
        try:
          func(self, *args, **kwargs)
        except unittest.SkipTest:
          raise
        except Exception:
          # Expected failure on TPU due to backend limitation or missing feature.
          return
        self.fail(
            f"Expected test to fail on TPU ({reason}), but it unexpectedly"
            " succeeded."
        )
      return func(self, *args, **kwargs)

    return wrapper

  return decorator


class CpuRngTest(seed_test_utils.RepeatableTest):
  """Tests default CPU RNG behaviors.

  This is not a reference test comparing TPU vs GPU. However, the CPU RNG
  behaviors verified here are directly relied upon by other test cases to
  validate their test setups. It also ensures that the device backend
  implementation does not introduce unexpected side effects that alter native
  PyTorch CPU RNG behaviors.
  """

  def test_cpu_manual_seed_initializes_state(self):
    """Verifies torch.manual_seed sets the initial CPU seed."""
    torch.manual_seed(1)
    self.assertEqual(torch.initial_seed(), 1)

    torch.manual_seed(42)
    self.assertEqual(torch.initial_seed(), 42)

  def test_cpu_rand_changes_state(self):
    """Verifies generating random numbers mutates the CPU generator state."""
    torch.manual_seed(42)
    initial_cpu_state = torch.get_rng_state()

    _ = torch.rand(100, device="cpu")
    self.assertFalse(torch.equal(torch.get_rng_state(), initial_cpu_state))

  def test_torch_seed_sets_cpu_seed(self):
    """Verifies torch.seed generates a new random seed for CPU."""
    torch.manual_seed(42)
    old_cpu_seed = torch.initial_seed()

    new_seed = torch.seed()

    self.assertIsInstance(new_seed, int)
    self.assertNotEqual(new_seed, old_cpu_seed)
    self.assertEqual(torch.initial_seed(), new_seed)

  def test_cpu_get_set_rng_state_restores_stream(self):
    """Verifies get_rng_state and set_rng_state restore CPU stream."""
    torch.manual_seed(42)
    saved_state = torch.get_rng_state()

    expected_out = torch.rand(10, device="cpu")
    _ = torch.rand(50, device="cpu")

    torch.set_rng_state(saved_state)
    actual_out = torch.rand(10, device="cpu")

    self.assertTrue(torch.equal(actual_out, expected_out))

  def test_cpu_generator_manual_seed_isolates_from_global_cpu_rng(self):
    """Verifies g.manual_seed sets initial_seed and isolates from global CPU."""
    torch.manual_seed(10)
    g = torch.Generator(device="cpu")
    g.manual_seed(42)

    self.assertEqual(g.initial_seed(), 42)
    self.assertEqual(torch.initial_seed(), 10)

  def test_cpu_generator_manual_seed_reproducibility(self):
    """Verifies re-seeding g.manual_seed produces identical random outputs."""
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    t1 = torch.rand(10, generator=g, device="cpu")

    g.manual_seed(42)
    t2 = torch.rand(10, generator=g, device="cpu")
    self.assertTrue(torch.equal(t1, t2))

  def test_cpu_generator_seed_sets_fresh_seed(self):
    """Verifies g.seed generates a non-deterministic seed for CPU generator."""
    g = torch.Generator(device="cpu")
    g.manual_seed(42)

    new_seed = g.seed()
    self.assertIsInstance(new_seed, int)
    self.assertNotEqual(new_seed, 42)
    self.assertEqual(g.initial_seed(), new_seed)

  def test_cpu_generator_get_set_state_restores_stream(self):
    """Verifies g.get_state and g.set_state restore CPU generator stream."""
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    saved_state = g.get_state()

    expected_out = torch.rand(10, generator=g, device="cpu")
    _ = torch.rand(50, generator=g, device="cpu")

    g.set_state(saved_state)
    actual_out = torch.rand(10, generator=g, device="cpu")
    self.assertTrue(torch.equal(actual_out, expected_out))


class _BaseRngTest(seed_test_utils.RepeatableTest):
  """Base test class providing common fixtures and helpers for device RNG tests."""

  def setUp(self):
    super().setUp()
    self.backend = _BACKEND.value
    self.backend_mod = _get_backend_module(self.backend)
    self.device = _get_device(self.backend)

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


class RngCudaRefTest(_BaseRngTest):
  """Reference tests comparing default TPU and GPU RNG behaviors for current device."""

  def test_initial_seed_return_current_device_seed(self):
    """Verifies backend_mod.initial_seed returns current device seed."""
    torch.manual_seed(42)
    self.assertEqual(self.backend_mod.initial_seed(), 42)

  def test_manual_seed_sets_current_device_seed(self):
    """Verifies torch.manual_seed sets initial seed on device backend."""
    torch.manual_seed(1)
    self.assertEqual(self.backend_mod.initial_seed(), 1)

    torch.manual_seed(42)
    self.assertEqual(self.backend_mod.initial_seed(), 42)

  def test_torch_manual_seed_different_seeds_produce_different_tensors(self):
    """Verifies different seeds produce distinct random tensors."""
    torch.manual_seed(1)
    t1 = torch.rand(100, device=self.device)

    torch.manual_seed(2)
    t2 = torch.rand(100, device=self.device)

    self.assertFalse(torch.equal(t1, t2))

  def test_torch_manual_seed_reproducibility(self):
    """Verifies setting the same seed produces identical random tensors."""
    torch.manual_seed(42)
    t1 = torch.rand(100, device=self.device)

    torch.manual_seed(42)
    t2 = torch.rand(100, device=self.device)

    self.assertTrue(torch.equal(t1, t2))

  def test_backend_manual_seed_sets_current_device_seed(self):
    """Verifies backend_mod.manual_seed sets initial seed on current device."""
    self.backend_mod.manual_seed(1)
    self.assertEqual(self.backend_mod.initial_seed(), 1)
    self.assertEqual(self._get_device_rng_seed(), 1)
    self.assertEqual(self._get_device_rng_offset(), 0)

    self.backend_mod.manual_seed(42)
    self.assertEqual(self.backend_mod.initial_seed(), 42)
    self.assertEqual(self._get_device_rng_seed(), 42)
    self.assertEqual(self._get_device_rng_offset(), 0)

  def test_backend_manual_seed_resets_offset_after_rand(self):
    """Verifies re-seeding with backend_mod.manual_seed resets device offset back to 0."""
    self.backend_mod.manual_seed(42)
    _ = torch.rand(100, device=self.device)
    self.assertGreater(self._get_device_rng_offset(), 0)

    self.backend_mod.manual_seed(77)
    self.assertEqual(self._get_device_rng_offset(), 0)

  def test_backend_manual_seed_different_seeds_produce_different_tensors(self):
    """Verifies different seeds with backend_mod.manual_seed produce distinct tensors."""
    self.backend_mod.manual_seed(1)
    t1 = torch.rand(100, device=self.device)

    self.backend_mod.manual_seed(2)
    t2 = torch.rand(100, device=self.device)

    self.assertFalse(torch.equal(t1, t2))

  def test_backend_manual_seed_reproducibility(self):
    """Verifies setting the same seed with backend_mod.manual_seed produces identical tensors."""
    self.backend_mod.manual_seed(42)
    t1 = torch.rand(100, device=self.device)

    self.backend_mod.manual_seed(42)
    t2 = torch.rand(100, device=self.device)

    self.assertTrue(torch.equal(t1, t2))

  def test_backend_manual_seed_does_not_change_cpu_seed(self):
    """Verifies backend_mod.manual_seed does not change CPU seed or state."""
    torch.manual_seed(10)
    cpu_seed_before = torch.initial_seed()
    cpu_state_before = torch.get_rng_state()

    self.backend_mod.manual_seed(42)

    self.assertEqual(torch.initial_seed(), cpu_seed_before)
    self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state_before))

  def test_backend_manual_seed_all_does_not_change_cpu_seed(self):
    """Verifies backend_mod.manual_seed_all does not change CPU seed or state."""
    torch.manual_seed(10)
    cpu_seed_before = torch.initial_seed()
    cpu_state_before = torch.get_rng_state()

    self.backend_mod.manual_seed_all(42)

    self.assertEqual(torch.initial_seed(), cpu_seed_before)
    self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state_before))

  def test_rand_does_not_change_device_seed(self):
    """Verifies torch.rand on device does not change initial_seed."""
    torch.manual_seed(42)
    _ = torch.rand(100, device=self.device)
    self.assertEqual(self.backend_mod.initial_seed(), 42)

  def test_rand_advances_offset(self):
    """Verifies torch.rand on device advances the offset."""
    torch.manual_seed(42)
    _ = torch.rand(100, device=self.device)
    self.assertGreater(self._get_device_rng_offset(), 0)

  def test_manual_seed_resets_offset_after_rand(self):
    """Verifies re-seeding resets device offset back to 0."""
    torch.manual_seed(42)
    _ = torch.rand(100, device=self.device)
    torch.manual_seed(77)
    self.assertEqual(self._get_device_rng_offset(), 0)

  def test_backend_manual_seed_all_resets_offset_after_rand(self):
    """Verifies re-seeding with backend_mod.manual_seed_all resets device offset back to 0."""
    self.backend_mod.manual_seed_all(42)
    _ = torch.rand(100, device=self.device)
    self.assertGreater(self._get_device_rng_offset(), 0)

    self.backend_mod.manual_seed_all(77)
    self.assertEqual(self._get_device_rng_offset(), 0)

  def test_backend_manual_seed_all_different_seeds_produce_different_tensors(
      self,
  ):
    """Verifies different seeds with backend_mod.manual_seed_all produce distinct tensors."""
    self.backend_mod.manual_seed_all(1)
    t1 = torch.rand(100, device=self.device)

    self.backend_mod.manual_seed_all(2)
    t2 = torch.rand(100, device=self.device)

    self.assertFalse(torch.equal(t1, t2))

  def test_backend_manual_seed_all_reproducibility(self):
    """Verifies setting the same seed with backend_mod.manual_seed_all produces identical tensors."""
    self.backend_mod.manual_seed_all(42)
    t1 = torch.rand(100, device=self.device)

    self.backend_mod.manual_seed_all(42)
    t2 = torch.rand(100, device=self.device)

    self.assertTrue(torch.equal(t1, t2))

  def test_torch_seed_sets_device_seed_and_resets_offset(self):
    """Verifies torch.seed updates device seed and resets offset to 0."""
    torch.manual_seed(42)
    _ = torch.rand(100, device=self.device)
    self.assertGreater(self._get_device_rng_offset(), 0)

    new_seed = torch.seed()

    self.assertEqual(self._get_device_rng_seed(), new_seed)
    self.assertEqual(self._get_device_rng_offset(), 0)

  def test_torch_random_fork_rng_isolates_and_restores_device_rng_state(self):
    """Verifies torch.random.fork_rng isolates and restores device RNG state."""
    torch.manual_seed(42)
    expected_out1 = torch.rand(5, device=self.device)
    expected_out2 = torch.rand(5, device=self.device)

    torch.manual_seed(42)
    out1 = torch.rand(5, device=self.device)
    self.assertTrue(torch.equal(out1, expected_out1))

    with torch.random.fork_rng(
        devices=[self.device], device_type=self.device.type
    ):
      _ = torch.rand(10, device=self.device)

    out2 = torch.rand(5, device=self.device)
    self.assertTrue(torch.equal(out2, expected_out2))

  def test_torch_random_fork_rng_produces_identical_tensors_inside_and_outside(
      self,
  ):
    """Verifies RNG stream in fork_rng reproduces identically outside."""
    torch.manual_seed(42)
    pre_offset = self._get_device_rng_offset()

    with torch.random.fork_rng(
        devices=[self.device], device_type=self.device.type
    ):
      out_inside = torch.rand(10, device=self.device)
      inside_offset = self._get_device_rng_offset()
      self.assertGreater(inside_offset, pre_offset)

    out_outside = torch.rand(10, device=self.device)
    outside_offset = self._get_device_rng_offset()

    self.assertTrue(torch.equal(out_inside, out_outside))
    self.assertEqual(inside_offset, outside_offset)

  def test_backend_get_set_rng_state_restores_stream(self):
    """Verifies get_rng_state and set_rng_state save and restore RNG stream."""
    self.backend_mod.manual_seed(42)
    saved_state = self.backend_mod.get_rng_state()

    expected_out = torch.rand(10, device=self.device)
    _ = torch.rand(50, device=self.device)

    self.backend_mod.set_rng_state(saved_state)
    actual_out = torch.rand(10, device=self.device)

    self.assertTrue(torch.equal(actual_out, expected_out))

  def test_backend_get_set_rng_state_preserves_offset(self):
    """Verifies get_rng_state captures offset and set_rng_state restores it."""
    self.backend_mod.manual_seed(42)
    _ = torch.rand(20, device=self.device)
    saved_offset = self._get_device_rng_offset()
    saved_state = self.backend_mod.get_rng_state()

    _ = torch.rand(100, device=self.device)

    self.backend_mod.set_rng_state(saved_state)
    self.assertEqual(self._get_device_rng_offset(), saved_offset)

  def test_generator_manual_seed_isolates_from_device_rng(self):
    """Verifies g.manual_seed isolates from default device RNG."""
    torch.manual_seed(10)
    g = torch.Generator(device=self.device)
    g.manual_seed(42)

    self.assertEqual(g.initial_seed(), 42)
    self.assertEqual(g.get_offset(), 0)
    self.assertEqual(self._get_device_rng_seed(), 10)

  def test_generator_manual_seed_reproducibility(self):
    """Verifies re-seeding g.manual_seed produces identical random outputs."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    t1 = torch.rand(10, generator=g, device=self.device)

    g.manual_seed(42)
    t2 = torch.rand(10, generator=g, device=self.device)
    self.assertTrue(torch.equal(t1, t2))

  def test_generator_seed_sets_fresh_seed_and_resets_offset(self):
    """Verifies g.seed sets fresh random seed for custom generator."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)

    new_seed = g.seed()
    self.assertIsInstance(new_seed, int)
    self.assertNotEqual(new_seed, 42)
    self.assertEqual(g.get_offset(), 0)

  def test_generator_get_set_offset_restores_stream(self):
    """Verifies g.get_offset and g.set_offset query and modify offset."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    self.assertEqual(g.get_offset(), 0)

    _ = torch.rand(20, generator=g, device=self.device)
    saved_offset = g.get_offset()
    self.assertGreater(saved_offset, 0)

    expected_out = torch.rand(10, generator=g, device=self.device)
    g.set_offset(saved_offset)
    actual_out = torch.rand(10, generator=g, device=self.device)

    self.assertTrue(torch.equal(actual_out, expected_out))

  def test_generator_get_set_state_restores_stream(self):
    """Verifies g.get_state and g.set_state snapshot and restore state."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    saved_state = g.get_state()

    expected_out = torch.rand(10, generator=g, device=self.device)
    _ = torch.rand(50, generator=g, device=self.device)

    g.set_state(saved_state)
    self.assertEqual(g.get_offset(), 0)

    actual_out = torch.rand(10, generator=g, device=self.device)
    self.assertTrue(torch.equal(actual_out, expected_out))

  def test_generator_clone_state_creates_isolated_copy(self):
    """Verifies g.clone_state creates an isolated generator copy."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    cloned_g = g.clone_state()

    self.assertEqual(cloned_g.initial_seed(), 42)
    self.assertEqual(cloned_g.get_offset(), 0)

    expected_out = torch.rand(10, generator=g, device=self.device)
    actual_out = torch.rand(10, generator=cloned_g, device=self.device)

    self.assertTrue(torch.equal(actual_out, expected_out))

  def test_generator_device_property_returns_bound_device(self):
    """Verifies g.device queries bound device context."""
    g = torch.Generator(device=self.device)
    self.assertEqual(g.device.type, self.device.type)


class CompiledRngTest(_BaseRngTest):
  """Reference tests comparing compiled RNG behaviors for current device."""

  def setUp(self):
    super().setUp()
    # Force single-threaded synchronous compilation in Inductor to prevent
    # spawning subprocesses, which fails in hermetic test runners.
    torch._inductor.config.compile_threads = 1

  def test_compiled_manual_seed_reproducibility(self):
    """Verifies re-seeding produces identical outputs in compiled function."""

    def fn(x):
      return torch.rand_like(x) + torch.rand_like(x)

    compiled_fn = torch.compile(fn, fullgraph=True)
    x = torch.zeros(10, device=self.device)

    self.backend_mod.manual_seed(42)
    out1 = compiled_fn(x)

    self.backend_mod.manual_seed(42)
    out2 = compiled_fn(x)
    self.assertTrue(torch.equal(out1, out2))

  def test_compiled_get_set_rng_state_restores_stream(self):
    """Verifies get/set_rng_state restores compiled function random output."""

    def fn(x):
      return torch.rand_like(x) + torch.rand_like(x)

    compiled_fn = torch.compile(fn, fullgraph=True)
    x = torch.zeros(10, device=self.device)

    self.backend_mod.manual_seed(42)
    saved_state = self.backend_mod.get_rng_state()
    out1 = compiled_fn(x)

    self.backend_mod.set_rng_state(saved_state)
    out2 = compiled_fn(x)
    self.assertTrue(torch.equal(out1, out2))

  def test_compiled_consecutive_calls_without_seed_reset_differ(self):
    """Verifies consecutive calls produce distinct outputs."""

    def fn(x):
      return torch.rand_like(x) + torch.rand_like(x)

    compiled_fn = torch.compile(fn, fullgraph=True)
    x = torch.zeros(10, device=self.device)

    self.backend_mod.manual_seed(42)
    out1 = compiled_fn(x)
    out2 = compiled_fn(x)
    self.assertFalse(torch.equal(out1, out2))

  def test_compiled_different_seeds_produce_different_outputs(self):
    """Verifies different seeds produce distinct outputs."""

    def fn(x):
      return torch.rand_like(x) + torch.rand_like(x)

    compiled_fn = torch.compile(fn, fullgraph=True)
    x = torch.zeros(10, device=self.device)

    self.backend_mod.manual_seed(42)
    out1 = compiled_fn(x)

    self.backend_mod.manual_seed(99)
    out2 = compiled_fn(x)
    self.assertFalse(torch.equal(out1, out2))


class SingleProcessMultiDeviceTest(_BaseRngTest):
  """Tests documenting single-process multi-device RNG differences.

  Single-process multi-device execution is discouraged in PyTorch CUDA due
  to Python GIL contention (PyTorch recommends 1 process per GPU via DDP;
  see
  https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel)
  and is not planned to be supported on TPU due to the PJRT 1-process-per-chip
  runtime architecture.
  """

  def setUp(self):
    super().setUp()
    self.device_idx = self.backend_mod.current_device()

  @_fail_on_tpu(
      "TPU backend does not support querying non-current device RNG state."
  )
  def test_torch_manual_seed_sets_all_device_seeds(self):
    """Verifies torch.manual_seed seeds all devices."""
    num_devices = self.backend_mod.device_count()
    self.assertGreater(
        num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " seeding all devices.",
    )
    torch.manual_seed(42)

    for i in range(num_devices):
      self.assertEqual(self._get_device_rng_seed(i), 42)
      self.assertEqual(self._get_device_rng_offset(i), 0)

  @_fail_on_tpu(
      "TPU backend does not support querying non-current device RNG state."
  )
  def test_backend_manual_seed_does_not_change_other_devices(self):
    """Verifies backend_mod.manual_seed does not modify other devices."""
    num_devices = self.backend_mod.device_count()
    self.assertGreater(
        num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " seeding behavior across devices.",
    )
    torch.manual_seed(10)
    other_devices_state_before = {
        i: (self._get_device_rng_seed(i), self._get_device_rng_offset(i))
        for i in range(num_devices)
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

  @_fail_on_tpu(
      "TPU backend does not support querying non-current device RNG state."
  )
  def test_backend_manual_seed_all_sets_all_device_seeds(self):
    """Verifies backend_mod.manual_seed_all seeds all devices."""
    num_devices = self.backend_mod.device_count()
    self.assertGreater(
        num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " seeding all devices.",
    )
    self.backend_mod.manual_seed_all(42)

    for i in range(num_devices):
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

  @_fail_on_tpu(
      "TPU backend does not support querying non-current device RNG state."
  )
  def test_torch_seed_sets_all_device_seeds(self):
    """Verifies torch.seed seeds all devices."""
    num_devices = self.backend_mod.device_count()
    self.assertGreater(
        num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " seeding all devices.",
    )
    for i in range(num_devices):
      _ = torch.rand(100, device=_get_device(self.backend, i))

    new_seed = torch.seed()

    for i in range(num_devices):
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

  @_fail_on_tpu(
      "TPU backend does not support get_rng_state_all / set_rng_state_all."
  )
  def test_backend_get_set_rng_state_all(self):
    """Verifies get/set_rng_state_all restore all device states."""
    num_devices = self.backend_mod.device_count()
    self.assertGreater(
        num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " multi-device state snapshot.",
    )
    for i in range(num_devices):
      self.backend_mod.manual_seed(42 + i)

    saved_states = self.backend_mod.get_rng_state_all()
    self.assertLen(saved_states, num_devices)

    expected_outs = [
        torch.rand(10, device=_get_device(self.backend, i))
        for i in range(num_devices)
    ]

    for i in range(num_devices):
      _ = torch.rand(50, device=_get_device(self.backend, i))

    self.backend_mod.set_rng_state_all(saved_states)

    actual_outs = [
        torch.rand(10, device=_get_device(self.backend, i))
        for i in range(num_devices)
    ]
    for exp, act in zip(expected_outs, actual_outs):
      self.assertTrue(torch.equal(exp, act))


if __name__ == "__main__":
  absltest.main()
