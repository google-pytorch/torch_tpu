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

"""Tests for harness/mode.py."""

from absl.testing import absltest
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e.harness import mode as mode_lib
from examples.benchmarks.e2e.harness import target as target_lib
from tests import seed_test_utils


class ModeEnumTest(seed_test_utils.RepeatableTest):

  def test_run_mode_values(self):
    self.assertEqual(common.RunMode.EAGER_DEFAULT.value, "eager_default")
    self.assertEqual(common.RunMode.EAGER_OPTIMIZED.value, "eager_optimized")
    self.assertEqual(
        common.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING.value,
        "eager_defer_never_and_launch_blocking",
    )
    self.assertEqual(common.RunMode.COMPILED.value, "compiled")

  def test_bad_mode_string_raises(self):
    with self.assertRaises(ValueError):
      common.RunMode("compiled_max_autotune")

  def test_torch_tpu_modes(self):
    self.assertEqual(
        mode_lib.modes_for(mode_lib.Framework.TORCH, target_lib.DeviceKind.TPU),
        [
            common.RunMode.EAGER_DEFAULT,
            common.RunMode.EAGER_OPTIMIZED,
            common.RunMode.COMPILED,
        ],
    )

  def test_torch_cuda_modes(self):
    self.assertEqual(
        mode_lib.modes_for(
            mode_lib.Framework.TORCH, target_lib.DeviceKind.CUDA
        ),
        [common.RunMode.EAGER_DEFAULT, common.RunMode.COMPILED],
    )

  def test_torch_cpu_modes(self):
    self.assertEqual(
        mode_lib.modes_for(mode_lib.Framework.TORCH, target_lib.DeviceKind.CPU),
        [common.RunMode.EAGER_DEFAULT],
    )

  def test_torchax_tpu_modes(self):
    self.assertEqual(
        mode_lib.modes_for(
            mode_lib.Framework.TORCHAX, target_lib.DeviceKind.TPU
        ),
        [common.RunMode.COMPILED],
    )

  def test_torchax_non_tpu_raises(self):
    for kind in target_lib.DeviceKind:
      if kind is not target_lib.DeviceKind.TPU:
        with self.assertRaises(ValueError):
          mode_lib.modes_for(mode_lib.Framework.TORCHAX, kind)

  def test_eager_optimized_is_torch_tpu_only(self):
    """Verifies eager_optimized is supported only for torch on TPU."""
    for fw in mode_lib.Framework:
      for kind in target_lib.DeviceKind:
        if (
            fw is mode_lib.Framework.TORCHAX
            and kind is not target_lib.DeviceKind.TPU
        ):
          continue
        has_eo = common.RunMode.EAGER_OPTIMIZED in mode_lib.modes_for(fw, kind)
        expected = (
            fw is mode_lib.Framework.TORCH and kind is target_lib.DeviceKind.TPU
        )
        self.assertEqual(has_eo, expected, f"{fw.value} x {kind.value}")

  def test_modes_for_accepts_strings(self):
    self.assertEqual(
        mode_lib.modes_for("torch", "cpu"),
        [common.RunMode.EAGER_DEFAULT],
    )
    self.assertEqual(
        mode_lib.modes_for("torchax", "tpu"), [common.RunMode.COMPILED]
    )

  def test_bad_framework_raises(self):
    with self.assertRaises(ValueError):
      mode_lib.modes_for("framework", target_lib.DeviceKind.CPU)

  def test_bad_device_kind_raises(self):
    with self.assertRaises(ValueError):
      mode_lib.modes_for(mode_lib.Framework.TORCH, "invalid_device")

  def test_framework_values(self):
    self.assertEqual(mode_lib.Framework.TORCH.value, "torch")
    self.assertEqual(mode_lib.Framework.TORCHAX.value, "torchax")


class RunModeContextTest(seed_test_utils.RepeatableTest):

  def test_non_tpu_target_is_noop(self):
    target = target_lib.make_target(target_lib.Platform.CPU)
    with mode_lib.run_mode_context(common.RunMode.EAGER_DEFAULT, target):
      pass

  def test_tpu_target_sets_eager_mode(self):
    from torch_tpu._internal import execution_mode  # pylint: disable=g-import-not-at-top

    target = target_lib.make_target(target_lib.Platform.V5E_1X1)
    with mode_lib.run_mode_context(common.RunMode.EAGER_DEFAULT, target):
      self.assertEqual(
          execution_mode.eager_mode, execution_mode.EagerMode.DEFER_NEVER
      )
    with mode_lib.run_mode_context(common.RunMode.EAGER_OPTIMIZED, target):
      self.assertEqual(
          execution_mode.eager_mode, execution_mode.EagerMode.DEFER_AND_FUSE
      )
    with mode_lib.run_mode_context(
        common.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING, target
    ):
      self.assertEqual(
          execution_mode.eager_mode,
          execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING,
      )

  def test_invalid_run_mode_raises(self):
    target = target_lib.make_target(target_lib.Platform.V5E_1X1)
    with self.assertRaises(ValueError):
      with mode_lib.run_mode_context("invalid_mode", target):
        pass


if __name__ == "__main__":
  absltest.main()
