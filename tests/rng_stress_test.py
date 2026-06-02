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

"""Stress test for RNG state and random operations."""

from collections.abc import Sequence
import concurrent.futures
import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import torch

RNG_OPS = (
    ("rand", lambda device: torch.rand(100, device=device)),
    ("randn", lambda device: torch.randn(100, device=device)),
    (
        "bernoulli",
        lambda device: torch.full((100,), 0.5, device=device).bernoulli_(),
    ),
    (
        "exponential",
        lambda device: torch.ones(100, device=device).exponential_(),
    ),
    ("random", lambda device: torch.zeros(100, device=device).random_()),
    ("uniform", lambda device: torch.zeros(100, device=device).uniform_()),
    (
        "dropout",
        lambda device: torch.nn.functional.dropout(
            torch.ones(100, device=device), p=0.5
        ),
        50,  # offset increment: 100 (size) * 64 (bits) // 128 (philox step)
    ),
    (
        "randperm",
        lambda device: torch.randperm(100, device=device),
        50,  # offset increment: 100 (size) * 64 (bits) // 128 (philox step)
    ),
)


def get_current_generator_offset() -> int:
  """Returns the offset of the current default generator's state tensor."""
  return torch.tpu.get_rng_state().view(torch.int64).cpu().tolist()[-1]


class RngStressTest(parameterized.TestCase):
  """Stress test for RNG state and random operations.

  This test verifies that concurrent calls to random operations from multiple
  threads do not cause RNG state collisions (e.g., due to lack of proper
  locking).

  Testing Approach and Assumptions:
  - The test runs random operations concurrently in multiple threads and asserts
    that all generated tensors are unique.F
  - For F32 tensors of size 100, the probability of two different threads
    generating the exact same tensor by chance is extremely low (approx
    (1/2^24)^100 for uniform distribution). Thus, collisions are strong evidence
    of state sharing.
  - The test also asserts that the global RNG state offset increments by the
    expected amount (num_threads * offset_increments).

  Limitations:
  - The "all-unique" check is probabilistic. It cannot strictly prove that no
    collision occurred, nor that collisions will never occur.
  - Checking offset increments depends on internal RNG algorithm details (e.g.,
    Philox step size) and is not purely testing user-facing behavior.

  Why an Unsound Approach was Chosen:
  - An ideal test would read the RNG state directly after each operation to
    ensure uniqueness. However, in a multi-threaded environment, `op_fn()` and
    `get_state()` calls cannot be made atomic without adding extra locking
    that would defeat the purpose of the test.
  - OS thread switching between `op_fn()` and `get_state()` leads to threads
    reading the same advanced state, causing false positives (failed tests) even
    when locking is correct.
  - Peeking into internal kernel state during execution is not feasible with
    current infrastructure. Therefore, we settle for this E2E stress test with
    numeric checks and offset verification as a practical compromise.
  """

  def assert_all_unique(self, results: Sequence[torch.Tensor]) -> None:
    """Asserts that all tensors in the results are unique."""
    for (i, tensor1), (j, tensor2) in itertools.combinations(
        enumerate(results), 2
    ):
      self.assertFalse(
          torch.allclose(tensor1, tensor2),
          f"Collision detected between tensor {i} and {j}",
      )

  @parameterized.named_parameters(*RNG_OPS)
  def test_no_rng_state_collision(self, op_fn, offset_increments=1):
    device = torch.accelerator.current_accelerator()
    op_fn = functools.partial(op_fn, device=device)
    num_threads = 20
    num_iterations = 50

    def run_op():
      return op_fn().cpu()

    for iteration_id in range(num_iterations):
      torch.manual_seed(iteration_id)
      begin_rng_offset = get_current_generator_offset()
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=num_threads
      ) as executor:
        futures = [executor.submit(run_op) for _ in range(num_threads)]
        results = [f.result() for f in futures]
      end_rng_offset = get_current_generator_offset()

      self.assert_all_unique(results)
      self.assertEqual(
          begin_rng_offset + num_threads * offset_increments, end_rng_offset
      )

  @parameterized.named_parameters(*RNG_OPS)
  def test_compile_no_rng_state_collision(self, op_fn, offset_increments=1):
    device = torch.accelerator.current_accelerator()
    op_fn = functools.partial(op_fn, device=device)
    num_threads = 20
    num_iterations = 50

    compiled_op_fn = torch.compile(op_fn, backend="tpu")

    # Warm up.
    compiled_op_fn()

    def run_op():
      return compiled_op_fn().cpu()

    for iteration_id in range(num_iterations):
      torch.manual_seed(iteration_id)

      begin_rng_offset = get_current_generator_offset()
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=num_threads
      ) as executor:
        futures = [executor.submit(run_op) for _ in range(num_threads)]
        results = [f.result() for f in futures]
      end_rng_offset = get_current_generator_offset()

      self.assert_all_unique(results)
      self.assertEqual(
          begin_rng_offset + num_threads * offset_increments, end_rng_offset
      )

  @parameterized.named_parameters(*RNG_OPS)
  def test_compile_and_eager_no_rng_state_collision(
      self, op_fn, offset_increments=1
  ):
    device = torch.accelerator.current_accelerator()
    op_fn = functools.partial(op_fn, device=device)
    num_threads = 20
    num_iterations = 50

    compiled_op_fn = torch.compile(op_fn, backend="tpu")

    # Warm up.
    compiled_op_fn()

    def run_compiled():
      return compiled_op_fn().cpu()

    def run_eager():
      return op_fn().cpu()

    for iteration_id in range(num_iterations):
      torch.manual_seed(iteration_id)

      begin_rng_offset = get_current_generator_offset()
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=num_threads
      ) as executor:
        futures = []
        for i in range(num_threads):
          if i % 2 == 0:
            futures.append(executor.submit(run_compiled))
          else:
            futures.append(executor.submit(run_eager))

        results = [f.result() for f in futures]
      end_rng_offset = get_current_generator_offset()

      self.assert_all_unique(results)
      self.assertEqual(
          begin_rng_offset + num_threads * offset_increments, end_rng_offset
      )


if __name__ == "__main__":
  absltest.main()
