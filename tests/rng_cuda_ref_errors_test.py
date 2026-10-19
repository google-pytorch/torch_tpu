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

"""Reference tests comparing compiled RNG error and graph break behaviors."""

from absl import flags
from absl.testing import absltest
import torch
from tests import accelerator_test_utils
from tests import seed_test_utils

_BACKEND = flags.DEFINE_string(
    "backend", "tpu", "The backend to test: 'tpu' or 'gpu'."
)


class _BaseRngTest(seed_test_utils.RepeatableTest):
  """Base test class providing common fixtures and helpers for device RNG tests."""

  def setUp(self):
    super().setUp()
    self.backend = _BACKEND.value
    self.backend_mod = accelerator_test_utils.get_backend_module(self.backend)
    self.device = accelerator_test_utils.get_device(self.backend)


class CompiledRngTest(_BaseRngTest):
  """Reference tests comparing compiled RNG behaviors for current device."""

  def setUp(self):
    super().setUp()
    # Force single-threaded synchronous compilation in Inductor to prevent
    # spawning subprocesses, which fails in hermetic test runners.
    torch._inductor.config.compile_threads = 1

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

    with self.assertRaises(RuntimeError):
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

    with self.assertRaises(RuntimeError):
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


if __name__ == "__main__":
  absltest.main()
