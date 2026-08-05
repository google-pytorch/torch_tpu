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

"""Tests for harness/target.py"""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from examples.benchmarks.e2e.harness import target as target_lib
from tests import seed_test_utils


class PlatformFromEnvTest(parameterized.TestCase):

  @parameterized.parameters(
      ("cpu", target_lib.Platform.CPU),
      ("b200_1", target_lib.Platform.B200_1),
      ("b200_4", target_lib.Platform.B200_4),
      ("b200_8", target_lib.Platform.B200_8),
      ("v5e_1x1", target_lib.Platform.V5E_1X1),
      ("v6e_1x1", target_lib.Platform.V6E_1X1),
      ("v5p_1x1x1", target_lib.Platform.V5P_1X1X1),
      ("v7_1x1x1", target_lib.Platform.V7_1X1X1),
      ("v7_2x2x1", target_lib.Platform.V7_2X2X1),
  )
  def test_reads_all_platforms_from_env(self, env_val, expected_platform):
    p = target_lib.platform_from_env(env={target_lib.PLATFORM_ENV_VAR: env_val})
    self.assertIs(p, expected_platform)

  def test_default_when_unset(self):
    self.assertIs(target_lib.platform_from_env(env={}), target_lib.Platform.CPU)

  def test_bad_value_raises_naming_the_string(self):
    """A typo'd platform must fail loud at read time, not silently fall back."""
    with self.assertRaises(ValueError) as cm:
      target_lib.platform_from_env(env={target_lib.PLATFORM_ENV_VAR: "b200_9"})
    self.assertIn("b200_9", str(cm.exception))

  def test_case_insensitive_value(self):
    # Enum values are lowercase; "B200_8" is not a member valu  e but should be accepted.
    self.assertIs(
        target_lib.platform_from_env(
            env={target_lib.PLATFORM_ENV_VAR: "B200_8"}
        ),
        target_lib.Platform.B200_8,
    )


class MakeTargetTest(parameterized.TestCase):

  def test_target_dtype_defaults(self):
    self.assertIs(
        target_lib.make_target(target_lib.Platform.CPU).dtype,
        target_lib.DType.BF16,
    )
    self.assertIs(
        target_lib.make_target(target_lib.Platform.V6E_1X1).dtype,
        target_lib.DType.BF16,
    )

  @parameterized.product(
      dtype=(target_lib.DType.BF16, target_lib.DType.FP32),
      config=[
          (target_lib.Platform.CPU, target_lib.DeviceKind.CPU, "cpu", 1, 1),
          (
              target_lib.Platform.B200_1,
              target_lib.DeviceKind.CUDA,
              "b200",
              1,
              1,
          ),
          (
              target_lib.Platform.B200_4,
              target_lib.DeviceKind.CUDA,
              "b200",
              1,
              4,
          ),
          (
              target_lib.Platform.B200_8,
              target_lib.DeviceKind.CUDA,
              "b200",
              1,
              8,
          ),
          (target_lib.Platform.V5E_1X1, target_lib.DeviceKind.TPU, "v5e", 1, 1),
          (target_lib.Platform.V6E_1X1, target_lib.DeviceKind.TPU, "v6e", 1, 1),
          (
              target_lib.Platform.V5P_1X1X1,
              target_lib.DeviceKind.TPU,
              "v5p",
              1,
              1,
          ),
          (target_lib.Platform.V7_1X1X1, target_lib.DeviceKind.TPU, "v7", 1, 1),
          (target_lib.Platform.V7_2X2X1, target_lib.DeviceKind.TPU, "v7", 1, 8),
      ],
  )
  def test_make_target_all_configurations(self, dtype, config):
    platform, kind, chip, nnodes, nprocs = config
    target = target_lib.make_target(platform=platform, dtype=dtype)
    self.assertEqual(target.platform, platform)
    self.assertEqual(target.dtype, dtype)
    self.assertEqual(target.device_kind, kind)
    self.assertEqual(target.platform_spec.kind, kind)
    self.assertEqual(target.platform_spec.topology.nnodes, nnodes)
    self.assertEqual(target.platform_spec.topology.nprocs_per_node, nprocs)
    self.assertEqual(target.platform_spec.topology.chip, chip)

  def test_target_dataclasses_are_frozen(self):
    t = target_lib.make_target(target_lib.Platform.CPU)
    with self.assertRaises(dataclasses.FrozenInstanceError):
      t.dtype = target_lib.DType.FP32
    with self.assertRaises(dataclasses.FrozenInstanceError):
      t.platform_spec.kind = target_lib.DeviceKind.TPU
    with self.assertRaises(dataclasses.FrozenInstanceError):
      t.platform_spec.topology.nnodes = 8


class TargetIsFrameworkAgnosticTest(seed_test_utils.RepeatableTest):

  def test_no_torch_or_jax_types(self):
    """Every field must be a plain/enum type -- no torch.device, no torch.dtype.

    This is what lets harness/target.py import neither framework.
    """
    t = target_lib.make_target(target_lib.Platform.B200_1)
    self.assertEqual(t.device_kind, target_lib.DeviceKind.CUDA)
    self.assertIsInstance(t.platform, target_lib.Platform)
    self.assertIsInstance(t.dtype, target_lib.DType)
    self.assertFalse(type(t.dtype).__module__.startswith("torch."))

  def test_module_imports_no_framework(self):
    """Importing harness.target must not bind torch or jax into its namespace."""
    self.assertFalse(hasattr(target_lib, "torch"))
    self.assertFalse(hasattr(target_lib, "jax"))


if __name__ == "__main__":
  absltest.main()
