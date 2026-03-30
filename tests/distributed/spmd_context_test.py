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

from absl import logging
from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_tpu import api
from torch_tpu._internal import sync
from torch_tpu._internal.distributed import spmd_context
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

SPMDContext = spmd_context.SPMDContext

WORLD_SIZE = 8


class CompilationCounter:
  """Context manager that counts compilation events."""

  def __init__(self, device):
    self.device = device
    self.initial_stats = None

  def __enter__(self):
    self.initial_stats = torch.tpu._get_cache_stats()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type:
      return False
    final_stats = torch.tpu._get_cache_stats()
    self.reqs = final_stats.num_cache_reqs - self.initial_stats.num_cache_reqs
    self.hits = final_stats.num_cache_hits - self.initial_stats.num_cache_hits
    self.compiles = self.reqs - self.hits
    return False

  def print_stats(self, test_name):
    logging.info(
        "Test %s: reqs=%d, hits=%d, compiles=%d",
        test_name,
        self.reqs,
        self.hits,
        self.compiles,
    )


def initialize_worker():
  device = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")

  return device


def spmd_test(expected_compiles):
  """Decorator for SPMD tests to reduce boilerplate.

  The decorated function MUST be a generator that yields exactly once.
  The code before the yield is considered 'setup' and its compilation
  events are not counted.
  The code after the yield is the actual test logic, and its compilation events
  are counted and validated against expected_compiles.

  Args:
    expected_compiles: The number of expected compilations for the code after
      the yield.

  Returns:
    A decorator that wraps the test function.
  """

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      device = initialize_worker()
      gen = func(device, *args, **kwargs)
      try:
        # Code before yield is setup (not counted)
        next(gen)
      except StopIteration as exc:
        raise RuntimeError(
            f"Test {func.__name__} must yield exactly once."
        ) from exc

      with CompilationCounter(device) as counter:
        try:
          # Code after yield is the actual test (counted)
          next(gen)
        except StopIteration:
          pass
        else:
          # This should not happen if the test yields exactly once.
          raise RuntimeError(f"Test {func.__name__} must yield exactly once.")

      counter.print_stats(func.__name__)
      if counter.compiles != expected_compiles:
        raise RuntimeError(
            f"Test {func.__name__}: Expected {expected_compiles} compilations,"
            f" got {counter.compiles}"
        )

    return wrapper

  return decorator


@spmd_test(expected_compiles=3)
def run_baseline_no_fusion_outside_context(device):
  x = torch.randn(10, device=device)
  sync.synchronize(x, wait=True)

  yield

  y = x + 1
  dist.all_reduce(y)
  z = y * 2
  # Trigger execution
  sync.synchronize(z, wait=True)


@spmd_test(expected_compiles=1)
def run_fusion_inside_spmd_context(device):
  x = torch.randn(10, device=device)
  sync.synchronize(x, wait=True)

  yield

  with SPMDContext():
    y = x + 1
    dist.all_reduce(y)
    z = x * 2
  sync.synchronize(z, wait=True)


@spmd_test(expected_compiles=3)
def run_traversal_before_after_spmd_context(device):
  x = torch.randn(10, device=device)
  sync.synchronize(x, wait=True)

  yield

  y = x + 1  # outside
  with SPMDContext():
    a = y + 1  # inside -> SplitBefore
    dist.all_reduce(a)
    b = a + 1  # inside
  z = b * 2  # outside -> SplitBefore
  sync.synchronize(z, wait=True)


@spmd_test(expected_compiles=1)
def run_nested_spmd_context(device):
  x = torch.randn(10, device=device)
  y = torch.randn(10, device=device)
  z = torch.randn(10, device=device)
  sync.synchronize([x, y, z], wait=True)

  yield

  with SPMDContext():
    w = x + y
    with SPMDContext():
      u = w * z
      dist.all_reduce(u)
  sync.synchronize(u, wait=True)


@spmd_test(expected_compiles=1)
def run_sequential_interacting_spmd_contexts(device):
  x = torch.randn(10, device=device)
  y = torch.randn(10, device=device)
  z = torch.randn(10, device=device)
  sync.synchronize([x, y, z], wait=True)

  yield

  with SPMDContext():
    w = x + y
  # Intervening period where IsInSpmdRegion() is false, but no ops are
  # emitted.
  with SPMDContext():
    u = z + w
    dist.all_reduce(u)
  sync.synchronize(u, wait=True)


@spmd_test(expected_compiles=2)
def run_collective_input_before_and_inside_spmd_context(device):
  x = torch.randn(10, device=device)
  sync.synchronize(x, wait=True)

  yield

  # Op outside
  outside_op = x + 1
  with SPMDContext():
    # Op inside
    inside_op_1 = x + 2
    inside_op_2 = outside_op + inside_op_1
  sync.synchronize(inside_op_2, wait=True)


class SPMDContextTest(absltest.TestCase):

  def test_baseline_no_fusion_outside_context(self):
    distributed_utils.dist_run(
        nproc_per_node=WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_baseline_no_fusion_outside_context, world_size=WORLD_SIZE
        ),
    )

  def test_fusion_inside_spmd_context(self):
    distributed_utils.dist_run(
        nproc_per_node=WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_fusion_inside_spmd_context, world_size=WORLD_SIZE
        ),
    )

  def test_traversal_before_after_spmd_context(self):
    distributed_utils.dist_run(
        nproc_per_node=WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_traversal_before_after_spmd_context, world_size=WORLD_SIZE
        ),
    )

  def test_nested_spmd_context(self):
    distributed_utils.dist_run(
        nproc_per_node=WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_nested_spmd_context, world_size=WORLD_SIZE
        ),
    )

  def test_sequential_interacting_spmd_contexts(self):
    distributed_utils.dist_run(
        nproc_per_node=WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_sequential_interacting_spmd_contexts, world_size=WORLD_SIZE
        ),
    )

  def test_collective_input_before_and_inside_spmd_context(self):
    distributed_utils.dist_run(
        nproc_per_node=WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_collective_input_before_and_inside_spmd_context,
            world_size=WORLD_SIZE,
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
