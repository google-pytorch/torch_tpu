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
import os
import struct
import threading
from typing import Any, Callable
import unittest
from unittest import mock

from absl import flags
from absl.testing import absltest
import torch
from tests import accelerator_test_utils
from tests import seed_test_utils

_BACKEND = flags.DEFINE_string(
    "backend", "tpu", "The backend to test: 'tpu' or 'gpu'."
)


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

  def test_generator_graphsafe_get_state_shares_state(self):
    """Verifies graphsafe_get_state returns generator sharing state."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    g_shared = g.graphsafe_get_state()
    self.assertIsInstance(g_shared, torch.Generator)

    # Generating random numbers on g advances both g and g_shared because they
    # share the underlying intrusive state pointer.
    _ = torch.rand(10, generator=g, device=self.device)
    self.assertTrue(torch.equal(g.get_state(), g_shared.get_state()))

  def test_generator_graphsafe_set_state_restores_stream(self):
    """Verifies graphsafe_set_state restores PRNG state and reproducibility."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    saved_state = g.clone_state()

    expected_out = torch.rand(10, generator=g, device=self.device)

    g.graphsafe_set_state(saved_state)
    self.assertEqual(g.initial_seed(), saved_state.initial_seed())
    self.assertEqual(g.get_offset(), saved_state.get_offset())

    actual_out = torch.rand(10, generator=g, device=self.device)
    self.assertTrue(torch.equal(actual_out, expected_out))


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

  def test_compiled_generator_arg_triggers_graph_break(self):
    """Verifies passing custom generator triggers graph break."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    x = torch.zeros(10, device=self.device)

    def fn_pass_gen(x, gen):
      return torch.rand_like(x, generator=gen)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_pass_gen, fullgraph=True)(x, g)

  def test_compiled_manual_seed_inside_fn_triggers_graph_break(self):
    """Verifies seeding inside compiled function triggers graph break."""
    x = torch.zeros(10, device=self.device)

    def fn_manual_seed(x):
      torch.manual_seed(42)
      return torch.rand_like(x)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_manual_seed, fullgraph=True)(x)

  def test_compiled_generator_instantiation_inside_fn_triggers_graph_break(
      self,
  ):
    """Verifies using generator in compiled function triggers graph break."""
    x = torch.zeros(10, device=self.device)

    def fn_create_gen(x):
      g = torch.Generator(device=self.device)
      return torch.rand_like(x, generator=g)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_create_gen, fullgraph=True)(x)

  def test_compiled_manual_seed_supported_with_graph_breaks(self):
    """Verifies seeding works under standard compile."""
    x = torch.zeros(10, device=self.device)

    def fn(x):
      torch.manual_seed(42)
      return torch.rand_like(x)

    compiled_fn = torch.compile(fn)
    out = compiled_fn(x)
    self.assertEqual(out.shape, x.shape)

  def test_compiled_graphsafe_get_state_execution(self):
    """Verifies function with graphsafe_get_state executes under compile."""
    x = torch.zeros(10, device=self.device)
    g = torch.Generator(device=self.device)
    g.manual_seed(42)

    def fn_get_state(x):
      gen_state = g.graphsafe_get_state()
      out = torch.rand_like(x)
      return out, gen_state

    compiled_fn = torch.compile(fn_get_state)
    out, gen_state = compiled_fn(x)
    self.assertEqual(out.shape, x.shape)
    self.assertIsInstance(gen_state, torch.Generator)
    self.assertTrue(torch.equal(gen_state.get_state(), g.get_state()))

  def test_compiled_graphsafe_set_state_restores_stream(self):
    """Verifies function with graphsafe_set_state restores compiled stream."""
    x = torch.zeros(10, device=self.device)
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    saved_state = g.clone_state()

    expected_out = torch.rand(10, generator=g, device=self.device)

    def fn_restore(x):
      g.graphsafe_set_state(saved_state)
      return torch.rand(10, generator=g, device=x.device)

    compiled_fn = torch.compile(fn_restore)
    actual_out = compiled_fn(x)
    self.assertTrue(torch.equal(actual_out, expected_out))

  def test_compiled_graphsafe_get_state_triggers_graph_break(self):
    """Verifies graphsafe_get_state triggers a graph break in torch.compile.

    Graphsafe refers to CUDA graphs rather than FX graphs during Dynamo
    tracing, so graphsafe_get_state should have the same behavior as get_state
    and trigger a graph break.
    """
    x = torch.zeros(10, device=self.device)
    g = torch.Generator(device=self.device)

    def fn_get_state(x):
      _ = g.graphsafe_get_state()
      return torch.rand_like(x)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_get_state, fullgraph=True)(x)

  def test_compiled_graphsafe_set_state_triggers_graph_break(self):
    """Verifies graphsafe_set_state triggers a graph break in torch.compile.

    Graphsafe refers to CUDA graphs rather than FX graphs during Dynamo
    tracing, so graphsafe_set_state should have the same behavior as set_state
    and trigger a graph break.
    """
    x = torch.zeros(10, device=self.device)
    g = torch.Generator(device=self.device)
    saved_state = g.clone_state()

    def fn_set_state(x):
      g.graphsafe_set_state(saved_state)
      return torch.rand_like(x)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_set_state, fullgraph=True)(x)

  def test_compiled_generator_supported_with_graph_breaks(self):
    """Verifies generators work under standard compile."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    x = torch.zeros(10, device=self.device)

    def fn(x, gen):
      g_local = torch.Generator(device=self.device)
      g_local.manual_seed(42)
      out1 = torch.rand_like(x, generator=gen)
      out2 = torch.rand_like(x, generator=g_local)
      return out1 + out2

    compiled_fn = torch.compile(fn)
    out = compiled_fn(x, g)
    self.assertEqual(out.shape, x.shape)

  def test_eager_dropout_backward_mask_equivalence(self):
    """Verifies eager dropout backward mask matches forward."""
    model = torch.nn.Dropout(p=0.5).to(self.device)

    torch.manual_seed(42)
    # The input tensor `x` is initialized to all ones, and the upstream gradient
    # from `out.sum().backward()` is also implicitly a tensor of all ones.
    x = torch.ones(100, device=self.device, requires_grad=True)

    # When dropout is applied in the forward pass (`out = model(x)`), it zeros
    # out approximately half the elements and scales the remaining elements by
    # `1 / (1 - p)` (which is 2.0 since `p=0.5`). So `out` consists of 0s and
    # 2s, making `mask_fwd = out > 0` a boolean representation of the forward
    # dropout mask.
    out = model(x)
    mask_fwd = out > 0

    # During the backward pass, dropout propagates the upstream gradient (all
    # 1s) only through the elements that were kept, applying the same scaling.
    # Therefore, `x.grad` will also consist of 0s (where dropped) and 2.0s
    # (where kept). Checking `x.grad > 0` reconstructs the backward dropout mask,
    # allowing for a direct `torch.equal` comparison to verify that the same mask
    # was used in both directions.
    out.sum().backward()
    mask_bwd = x.grad > 0
    self.assertTrue(torch.equal(mask_fwd, mask_bwd))

  def test_compiled_dropout_backward_mask_equivalence(self):
    """Verifies compiled dropout backward mask matches forward."""
    # See test_eager_dropout_backward_mask_equivalence for mask computation details.
    model = torch.nn.Dropout(p=0.5).to(self.device)
    compiled_model = torch.compile(model)

    torch.manual_seed(42)
    x = torch.ones(100, device=self.device, requires_grad=True)
    out = compiled_model(x)
    mask_fwd = out > 0
    out.sum().backward()
    mask_bwd = x.grad > 0
    self.assertTrue(torch.equal(mask_fwd, mask_bwd))

  @_fail_on_tpu(
      "b/496168350: Compiled activation checkpointing RNG preservation requires"
      " graphsafe RNG."
  )
  def test_compiled_activation_checkpointing_dropout(self):
    """Verifies non-reentrant activation checkpointing reproduces the mask."""

    class CheckpointModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(p=0.5)

      def forward(self, x):
        def inner_fn(x):
          return self.drop(x)

        return torch.utils.checkpoint.checkpoint(
            inner_fn, x, use_reentrant=False
        )

    model = CheckpointModel().to(self.device)
    compiled_model = torch.compile(model, fullgraph=True)

    torch.manual_seed(42)
    x = torch.ones(100, device=self.device, requires_grad=True)
    out = compiled_model(x)
    mask_fwd = out > 0
    out.sum().backward()
    mask_bwd = x.grad > 0

    self.assertTrue(torch.equal(mask_fwd, mask_bwd))


class ErrorHandlingRngTest(_BaseRngTest):
  """Tests verifying RNG contracts under error and exception conditions."""

  def test_compile_tracing_failure_restores_eager_rng(self):
    """Verifies compilation tracing failure restores eager generator state."""
    self.backend_mod.manual_seed(42)
    expected = torch.rand(10, device=self.device)

    self.backend_mod.manual_seed(42)

    def failing_fn(x):
      _ = torch.rand_like(x)
      raise RuntimeError("Simulated tracing failure")

    with self.assertRaises(  # ASSERT_RAISES_OK=Tests RNG behavior on crash, not error handling.
        RuntimeError
    ):
      torch.compile(failing_fn, fullgraph=True)(
          torch.ones(10, device=self.device)
      )

    actual = torch.rand(10, device=self.device)
    self.assertTrue(torch.equal(expected, actual))

  def test_eager_exception_after_rng_op_does_not_rollback(self):
    """Verifies exception in Python does not roll back advanced RNG state."""
    self.backend_mod.manual_seed(42)
    _ = torch.rand(10, device=self.device)
    expected_next = torch.rand(10, device=self.device)

    self.backend_mod.manual_seed(42)
    try:
      _ = torch.rand(10, device=self.device)
      raise RuntimeError("Simulated eager error")
    except RuntimeError:
      pass

    actual_next = torch.rand(10, device=self.device)
    self.assertTrue(torch.equal(expected_next, actual_next))

  def test_compiled_runtime_error_without_catch(self):
    """Verifies runtime error in compiled fn does not rollback RNG state."""
    self.backend_mod.manual_seed(42)
    _ = torch.rand(10, device=self.device)
    expected_next = torch.rand(10, device=self.device)

    self.backend_mod.manual_seed(42)

    @torch.compile(fullgraph=False)
    def fn(x):
      r = torch.rand_like(x)
      # Evaluating a tensor condition triggers a graph break; the exception
      # is raised eagerly in Python after the compiled RNG op executes.
      if x.sum() > 0:
        raise RuntimeError("Runtime error after compiled execution")
      return r

    with self.assertRaises(  # ASSERT_RAISES_OK=Tests RNG behavior on crash, not error handling.
        RuntimeError
    ):
      fn(torch.ones(10, device=self.device))

    actual_next = torch.rand(10, device=self.device)
    self.assertTrue(torch.equal(expected_next, actual_next))

  def test_compiled_runtime_error_with_catch(self):
    """Verifies caught error inside compiled fn preserves RNG progression."""
    self.backend_mod.manual_seed(42)
    _ = torch.rand(10, device=self.device)
    expected_next = torch.rand(10, device=self.device)

    self.backend_mod.manual_seed(42)

    @torch.compile(fullgraph=False)
    def fn_with_try_catch(x):
      r = torch.rand_like(x)
      # Evaluating a tensor condition triggers a graph break; the exception
      # and catch block execute eagerly in Python after the compiled RNG op.
      try:
        if x.sum() > 0:
          raise RuntimeError("Caught error inside function")
      except RuntimeError:
        pass
      return r

    _ = fn_with_try_catch(torch.ones(10, device=self.device))

    actual_next = torch.rand(10, device=self.device)
    self.assertTrue(torch.equal(expected_next, actual_next))


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
      _ = torch.rand(
          100, device=accelerator_test_utils.get_device(self.backend, i)
      )

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
        torch.rand(
            10, device=accelerator_test_utils.get_device(self.backend, i)
        )
        for i in range(num_devices)
    ]

    for i in range(num_devices):
      _ = torch.rand(
          50, device=accelerator_test_utils.get_device(self.backend, i)
      )

    self.backend_mod.set_rng_state_all(saved_states)

    actual_outs = [
        torch.rand(
            10, device=accelerator_test_utils.get_device(self.backend, i)
        )
        for i in range(num_devices)
    ]
    for exp, act in zip(expected_outs, actual_outs):
      self.assertTrue(torch.equal(exp, act))

  @_fail_on_tpu(
      "TPU backend does not support querying or executing on non-current"
      " device in a single process."
  )
  def test_multithread_eager_cross_device_isolation(self):
    """Verifies threads on distinct devices produce deterministic outputs."""
    num_devices = self.backend_mod.device_count()
    self.assertGreater(
        num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " multi-threaded cross-device execution.",
    )

    def run_trial() -> list[torch.Tensor]:
      """Runs a multi-threaded cross-device trial and returns outputs.

      Returns:
        List of generated tensors per device, ordered by device index.
      """
      results = {}
      errors = []

      def thread_worker(thread_id):
        try:
          dev = accelerator_test_utils.get_device(self.backend, thread_id)
          with self.backend_mod.device(thread_id):
            self.backend_mod.manual_seed(42 + thread_id)
            tensors = [torch.rand(10, device=dev) for _ in range(5)]
            results[thread_id] = torch.stack(tensors).cpu()
        except Exception as e:  # pylint: disable=broad-exception-caught
          errors.append(e)

      threads = [
          threading.Thread(target=thread_worker, args=(i,))
          for i in range(num_devices)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      if errors:
        raise errors[0]

      return [results[i] for i in range(num_devices)]

    trial1 = run_trial()
    trial2 = run_trial()

    for res1, res2 in zip(trial1, trial2):
      self.assertTrue(torch.equal(res1, res2))
    self.assertFalse(torch.equal(trial1[0], trial1[1]))


class MultiThreadRngTest(_BaseRngTest):
  """Tests multithreaded RNG operations on a single device."""

  def _run_concurrent_threads(
      self,
      seed_fn: Callable[[int], None],
      fn: Callable[[int], Any],
      num_threads: int,
  ) -> list[Any]:
    """Executes fn across concurrent threads with seed and function locks.

    Args:
      seed_fn: Callable taking thread index to configure RNG seed.
      fn: Callable taking thread index to generate outputs or read seeds.
      num_threads: Number of concurrent threads to spawn.

    Returns:
      List of results produced by each thread, ordered by thread index.
    """
    results = [None] * num_threads
    errors = []
    seed_lock = threading.Lock()
    fn_lock = threading.Lock()
    barrier = threading.Barrier(num_threads)

    def worker(idx: int):
      try:
        with seed_lock:
          seed_fn(idx)
        barrier.wait(timeout=30)
        with fn_lock:
          out = fn(idx)
          if isinstance(out, torch.Tensor):
            out = out.cpu()
        results[idx] = out
      except Exception as e:  # pylint: disable=broad-exception-caught
        errors.append(e)
        barrier.abort()

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    if errors:
      raise errors[0]

    return results

  def test_multithread_share_same_seed(self):
    """Verifies threads on single device share global and device seeds."""
    outs = self._run_concurrent_threads(
        seed_fn=lambda idx: torch.manual_seed(42 + idx),
        fn=lambda idx: (torch.initial_seed(), self.backend_mod.initial_seed()),
        num_threads=2,
    )
    self.assertEqual(outs[0], outs[1])
    self.assertEqual(outs[0][0], outs[0][1])

  def test_multithread_share_same_state_eager(self):
    """Verifies threads on single device share RNG state in eager mode."""
    outs = self._run_concurrent_threads(
        seed_fn=lambda idx: self.backend_mod.manual_seed(42),
        fn=lambda idx: torch.rand(10, device=self.device),
        num_threads=2,
    )
    self.assertFalse(torch.equal(outs[0], outs[1]))

  def test_multithread_share_same_state_compiled(self):
    """Verifies threads share RNG state in compiled mode."""
    # Limit compilation threads to avoid contention during Dynamo tracing.
    torch._inductor.config.compile_threads = 1

    def fn_to_compile(x):
      return torch.rand_like(x) + torch.rand_like(x)

    compiled_fn = torch.compile(fn_to_compile, fullgraph=True)
    x = torch.zeros(10, device=self.device)
    # Pre-warm compiled_fn so compilation finishes before execution.
    _ = compiled_fn(x)

    outs = self._run_concurrent_threads(
        seed_fn=lambda idx: self.backend_mod.manual_seed(42),
        fn=lambda idx: compiled_fn(x),
        num_threads=2,
    )
    self.assertFalse(torch.equal(outs[0], outs[1]))

  def test_multithread_shared_generator_shares_state(self):
    """Verifies threads sharing a generator consume distinct chunks."""
    shared_g = torch.Generator(device=self.device)
    outs = self._run_concurrent_threads(
        seed_fn=lambda idx: shared_g.manual_seed(42),
        fn=lambda idx: torch.rand(10, generator=shared_g, device=self.device),
        num_threads=2,
    )
    self.assertFalse(torch.equal(outs[0], outs[1]))

  def test_multithread_separate_generators_isolate_state(self):
    """Verifies separate generators isolate RNG state across threads."""
    generators = [
        torch.Generator(device=self.device),
        torch.Generator(device=self.device),
    ]
    outs = self._run_concurrent_threads(
        seed_fn=lambda idx: generators[idx].manual_seed(42),
        fn=lambda idx: torch.rand(
            10, generator=generators[idx], device=self.device
        ),
        num_threads=2,
    )
    self.assertTrue(torch.equal(outs[0], outs[1]))
    self.assertTrue(
        torch.equal(generators[0].get_state(), generators[1].get_state())
    )

  def test_multithread_separate_generators_different_ops(self):
    """Verifies distinct operations lead to different generator states."""
    generators = [
        torch.Generator(device=self.device),
        torch.Generator(device=self.device),
    ]

    def run_ops(idx: int) -> torch.Tensor:
      out = None
      for _ in range(idx + 1):
        out = torch.rand(10, generator=generators[idx], device=self.device)
      return out

    outs = self._run_concurrent_threads(
        seed_fn=lambda idx: generators[idx].manual_seed(42),
        fn=run_ops,
        num_threads=2,
    )
    self.assertFalse(torch.equal(outs[0], outs[1]))
    self.assertFalse(
        torch.equal(generators[0].get_state(), generators[1].get_state())
    )

  @mock.patch.dict(os.environ, {"TORCHINDUCTOR_COMPILE_THREADS": "0"})
  @torch._inductor.config.patch(compile_threads=1)
  def test_concurrent_compile_tracing_and_eager_rng(self):
    """Verifies concurrent eager RNG operations during torch.compile tracing."""
    if self.backend == "tpu":
      # b/556256460: The race condition between concurrent eager RNG operations
      # and compilation tracing causes a fatal C++ CHECK failure (SIGABRT) in
      # the runtime, terminating the test process. Because _fail_on_tpu only
      # catches Python exceptions, this test must be skipped on TPU until a fix
      # lands.
      self.skipTest(
          "b/556256460: Concurrent eager RNG execution during compilation"
          " tracing hits unmaterialized placeholder state."
      )
    # Force single-threaded compilation: the env var prevents subprocess
    # spawning in hermetic test runners, while compile_threads=1 enforces
    # single-threaded compilation in this thread's ContextVar scope.
    stop_event = threading.Event()
    errors = []

    def eager_worker():
      try:
        # Repeatedly execute eager RNG operations to generate contention on the
        # default generator until the compile worker signals completion.
        while not stop_event.is_set():
          _ = torch.rand(10, device=self.device)
      except Exception as e:  # pylint: disable=broad-exception-caught
        errors.append(e)
        stop_event.set()

    def fn_to_compile(x):
      res = x
      for _ in range(30):
        res = res + torch.rand_like(res)
      return res

    # Force Dynamo to re-compile on each iteration by passing a different shape
    # with dynamic=False, triggering a fresh compilation cycle without caching.
    compiled = torch.compile(fn_to_compile, dynamic=False)

    def compile_worker():
      # Inductor configs are thread-scoped via ContextVar; patch compile_threads
      # in the worker thread to ensure synchronous compilation during tracing.
      with torch._inductor.config.patch(compile_threads=1):
        try:
          for i in range(10):
            if stop_event.is_set():
              break
            x = torch.zeros(10 + i, device=self.device)
            _ = compiled(x)
        except Exception as e:  # pylint: disable=broad-exception-caught
          errors.append(e)
        finally:
          stop_event.set()

    eager_threads = [threading.Thread(target=eager_worker) for _ in range(8)]
    compile_thread = threading.Thread(target=compile_worker)

    for t in eager_threads:
      t.start()
    compile_thread.start()

    compile_thread.join()
    for t in eager_threads:
      t.join()

    if errors:
      raise errors[0]


if __name__ == "__main__":
  absltest.main()
