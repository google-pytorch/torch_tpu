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

"""Provides base classes and utilities for controlling RNG seeds in tests."""

import random
import time
from typing import Final

from absl.testing import absltest
import torch
from torch_tpu._internal.device import _device_ops_backend

DEFAULT_RANDOM_SEED: Final[int] = 1234


def seed_rngs(seed: int) -> None:
  """Seeds Python and PyTorch RNGs with the given seed."""
  # TODO: b/542689633 - Seed NumPy RNG here as well.
  random.seed(seed)
  torch.manual_seed(seed)


class SeededTest(absltest.TestCase):  # ABSLTEST_OK=Base test class
  """Abstract base class that fixes RNG seeds to make tests reproducible.

  This class picks a random seed in setUpClass() and sets it in setUp().
  A subclass must define choose_seed() to determine the seed.
  """

  test_random_seed: int = DEFAULT_RANDOM_SEED
  seed_in_setup: bool = True

  @classmethod
  def setUpClass(cls) -> None:
    """Picks the RNG seed for the test class and remembers it."""

    super().setUpClass()
    # A subclass must define choose_seed(), which must return an int.
    cls.test_random_seed = cls.choose_seed()  # pytype: disable=attribute-error
    print(f"Repro with --test_random_seed={cls.test_random_seed}", flush=True)

  def setUp(self) -> None:
    super().setUp()
    if self.seed_in_setup:
      # Set the random seed for Python and Torch.
      seed_rngs(self.test_random_seed)


class RepeatableTest(SeededTest):
  """Base class that fixes RNG seeds so tests are reproducible.

  This base class uses a constant RNG seed or from the --test_random_seed absl
  flag if provided. It resets the same RNG seed before each test method for
  reproducibility.

  When combining RepeatableTest with another TestCase class:
  e.g., class MyTest(RepeatableTest, parameterized.TestCase):
  where:
    class RepeatableTest(seed_test_utils.RepeatableTest)
    class parameterized.TestCase(seed_test_utils.RepeatableTest)

  Through Python MRO, super().setUp() propagates through all base classes:
  MyTest -> RepeatableTest -> parameterized.TestCase -> absltest.TestCase
  # ABSLTEST_OK=Reference to absltest.TestCase
  """

  @classmethod
  def choose_seed(cls) -> int:
    """Chooses the RNG seed for the test class.

    A subclass may override this method to provide a different seed.
    """

    if absltest.FLAGS["test_random_seed"].present:
      # The user explicitly passed --test_random_seed=N, so we use that value.
      return absltest.FLAGS.test_random_seed
    return DEFAULT_RANDOM_SEED


class MultiProcessRepeatableTest(RepeatableTest):
  """Base class for multi-process TPU tests that launch processes via mp.spawn.

  Disables seeding in setUp() in the main process to prevent torch.manual_seed
  from initializing and locking the TPU device before child processes are
  spawned.
  """

  seed_in_setup = False


class VaryingSeedInPostsubmitTest(SeededTest):
  """Base class for tests that support dynamic postsubmit seeds.

  This base class uses the same RNG seed as RepeatableTest in presubmit. In
  postsubmit, it chooses a different seed in every run. The seed resets before
  every test method.

  A subclass can override varies_seed_in_postsubmit() to choose whether to
  vary the seed in postsubmit.
  """

  @classmethod
  def choose_seed(cls) -> int:
    if (
        cls.varies_seed_in_postsubmit()
        and not absltest.FLAGS["test_random_seed"].present  #
        and _device_ops_backend._is_optimized_build()  # pylint: disable=protected-access
    ):
      # --test_random_seed is not set and we are in postsubmit (opt build).
      # We set the seed based on the time, so that we get more test coverage
      # over time.
      return time.time_ns() % 100000

    return RepeatableTest.choose_seed()

  @classmethod
  def varies_seed_in_postsubmit(cls) -> bool:
    """Returns whether the test class should vary the RNG seed in postsubmit."""
    return True
