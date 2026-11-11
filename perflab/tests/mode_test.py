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

from unittest import mock

from absl.testing import absltest
import torch
from torch._subclasses import fake_tensor
from perflab import common
from perflab import mode as mode_lib
from perflab import target as target_lib
from tests import seed_test_utils

_MAX_CONSTANT_NUMEL = mode_lib._MAX_CONSTANT_NUMEL  # pylint: disable=protected-access


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
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
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
        with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
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

  def test_smoke_skipped_run_modes(self):
    for skipped, fake_tensor_mode, expected in (
        # Outside the smoke test, the skipped run modes are unchanged.
        ({"eager_default"}, False, {"eager_default"}),
        # A benchmark that runs in an accelerator-only mode runs in the smoke.
        ({"eager_default"}, True, set()),
        ({"eager_default", "compiled"}, True, {"compiled"}),
        # A benchmark that runs in no accelerator-only mode stays skipped.
        (
            {"eager_default", "eager_optimized", "compiled"},
            True,
            {"eager_default", "eager_optimized", "compiled"},
        ),
        ({"compiled"}, True, {"compiled"}),
    ):
      with self.subTest(skipped=skipped, fake_tensor_mode=fake_tensor_mode):
        self.assertEqual(
            mode_lib.smoke_skipped_run_modes(
                skipped, fake_tensor_mode=fake_tensor_mode
            ),
            frozenset(expected),
        )

  def test_bad_framework_raises(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      mode_lib.modes_for("framework", target_lib.DeviceKind.CPU)

  def test_bad_device_kind_raises(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
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
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      with mode_lib.run_mode_context("invalid_mode", target):
        pass

  def test_cuda_target_sets_float32_matmul_precision(self):
    from unittest import mock  # pylint: disable=g-import-not-at-top

    target = target_lib.make_target(target_lib.Platform.B200_1)
    with mock.patch("torch.set_float32_matmul_precision") as mock_prec:
      with mode_lib.run_mode_context(common.RunMode.COMPILED, target):
        pass
    mock_prec.assert_called_once_with("high")


class FakeTensorContextTest(seed_test_utils.RepeatableTest):

  def test_enters_fake_tensor_mode(self):
    with mode_lib.fake_tensor_context():
      layer = torch.nn.Linear(8, 4)
      layer = layer.to(dtype=torch.bfloat16)
      x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
      y = layer(x).sum()
      y.backward()
      self.assertIsInstance(x, fake_tensor.FakeTensor)
      self.assertIsInstance(x.grad, fake_tensor.FakeTensor)
      self.assertEqual(x.grad.dtype, torch.bfloat16)
      self.assertEqual(layer.weight.dtype, torch.bfloat16)

  def test_converts_constant_parameters(self):
    with mode_lib.fake_tensor_context():
      layer = torch.nn.Linear(8, 4)
      # A constant like CLIP's logit scale, converted with its own storage.
      layer.logit_scale = torch.nn.Parameter(torch.tensor(2.6592))
      layer = layer.to(dtype=torch.bfloat16)
      self.assertIsInstance(layer.logit_scale, fake_tensor.FakeTensor)
      self.assertEqual(layer.logit_scale.dtype, torch.bfloat16)

  def test_restores_real_tensors_on_exit(self):
    with mode_lib.fake_tensor_context():
      pass
    self.assertNotIsInstance(torch.zeros(4, 4), fake_tensor.FakeTensor)

  def test_trunc_normal_is_a_no_op_inside(self):
    with mode_lib.fake_tensor_context():
      layer = torch.nn.Linear(8, 4).to(dtype=torch.bfloat16)
      # As in HuggingFace ViT's position embedding init. Without the patch, this
      # would raise DataDependentOutputException.
      torch.nn.init.trunc_normal_(layer.weight)
      self.assertIsInstance(layer.weight, fake_tensor.FakeTensor)

  def test_restores_trunc_normal_on_exit(self):
    with mode_lib.fake_tensor_context():
      pass
    x = torch.zeros(4, 4)
    torch.nn.init.trunc_normal_(x, a=1.0, b=2.0)
    self.assertTrue(bool((x >= 1.0).all()))

  def test_drop_path_rates_are_readable_inside(self):
    with mode_lib.fake_tensor_context():
      rates = torch.linspace(0, 0.3, 4)
      self.assertIsInstance(rates, fake_tensor.FakeTensor)
      # As in HuggingFace Swin2SR and timm. Without the constants, `.item()`
      # would raise DataDependentOutputException.
      layer_rates = [x.item() for x in rates]
      # As in HuggingFace ConvNeXt V2.
      stage_rates = [x.tolist() for x in rates.split([1, 3])]
      # As in HuggingFace BiT, which builds them with NumPy.
      tensor_stage_rates = [
          x.tolist() for x in torch.tensor([0.0, 0.1, 0.2, 0.3]).split([1, 3])
      ]
    self.assertSequenceAlmostEqual(layer_rates, [0.0, 0.1, 0.2, 0.3], places=6)
    self.assertLen(stage_rates, 2)
    self.assertSequenceAlmostEqual(stage_rates[1], [0.1, 0.2, 0.3], places=6)
    self.assertSequenceAlmostEqual(
        tensor_stage_rates[1], [0.1, 0.2, 0.3], places=6
    )

  def test_large_linspace_stays_fake(self):
    with mode_lib.fake_tensor_context():
      x = torch.linspace(0, 1, 4096)
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        x[0].item()

  def test_restores_constant_numel_limit_on_exit(self):
    limit = fake_tensor.CONSTANT_NUMEL_LIMIT
    with mode_lib.fake_tensor_context():
      pass
    self.assertEqual(fake_tensor.CONSTANT_NUMEL_LIMIT, limit)

  def test_layer_drop_draw_is_readable_inside(self):
    with mode_lib.fake_tensor_context():
      draw = torch.rand([])
      self.assertIsInstance(draw, fake_tensor.FakeTensor)
      # As in the LayerDrop of HuggingFace BART in training. Without the
      # constants, the comparison would raise DataDependentOutputException.
      self.assertTrue(bool(draw < 1.0))
      self.assertFalse(bool(draw < 0.0))

  def test_large_rand_stays_fake(self):
    with mode_lib.fake_tensor_context():
      x = torch.rand(4096)
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        x[0].item()

  def test_input_values_are_readable_inside(self):
    with mode_lib.fake_tensor_context():
      input_ids = torch.randint(3, 100, (2, 128))
      input_ids[:, -1] = 2
      hidden_states = torch.randn(2, 128, 16)
      self.assertIsInstance(input_ids, fake_tensor.FakeTensor)
      # As in the HuggingFace BART classification head. Without the constants,
      # these would raise DynamicOutputShapeException.
      eos_mask = input_ids.eq(2)
      eos_counts = torch.unique_consecutive(eos_mask.sum(1)).tolist()
      eos_states = hidden_states[eos_mask, :]
    self.assertEqual(eos_counts, [1])
    self.assertEqual(eos_states.shape, (2, 16))

  def test_integer_tensor_factory_outputs_are_readable_inside(self):
    with mode_lib.fake_tensor_context():
      input_ids = torch.randint(0, 10, (2, 8))
      hidden_states = torch.randn(2, 4096)
      # As VITS's `torch.ones_like(input_ids)` padding mask. These read only
      # the shapes and dtypes of the fake tensors they are given.
      sums = [
          torch.ones_like(input_ids).sum().item(),
          torch.zeros_like(input_ids, dtype=torch.int).sum().item(),
          torch.full_like(input_ids, 3).sum().item(),
          hidden_states.new_ones((2,), dtype=torch.long).sum().item(),
          hidden_states.new_full((3,), 2, dtype=torch.long).sum().item(),
      ]
      # Floating-point outputs, mostly parameter initializations, stay fake.
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        torch.ones_like(hidden_states).sum().item()
    self.assertEqual(sums, [16, 0, 48, 2, 6])

  def test_large_integer_tensor_stays_fake(self):
    with mode_lib.fake_tensor_context():
      x = torch.zeros(_MAX_CONSTANT_NUMEL + 1, dtype=torch.long)
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        x[0].item()

  def test_floating_point_ones_stay_fake(self):
    with mode_lib.fake_tensor_context():
      # E.g. LayerNorm weights, whose values models do not read.
      x = torch.ones(4)
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        x[0].item()

  def test_factory_outputs_of_floating_point_dtypes_stay_fake(self):
    with mode_lib.fake_tensor_context():
      ints = torch.full((4,), 3)
      outputs = [
          torch.zeros(4, dtype=torch.bfloat16),
          torch.ones(4, dtype=torch.complex64),
          torch.full((4,), 0.5),
      ]
      for x in outputs:
        with self.assertRaises(fake_tensor.DataDependentOutputException):
          x[0].item()
      self.assertEqual(ints.sum().item(), 12)

  def test_factory_outputs_off_the_cpu_stay_fake(self):
    with mode_lib.fake_tensor_context():
      x = torch.zeros(4, dtype=torch.long, device="meta")
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        x[0].item()

  def test_real_tensors_are_converted_inside(self):
    # E.g. a tensor that a module makes when it is imported.
    real = torch.ones(4)
    with mode_lib.fake_tensor_context():
      self.assertIsInstance(real + torch.randn(4), fake_tensor.FakeTensor)

  def test_large_outputs_of_constants_are_not_computed(self):
    numels = []
    may_turn_const = fake_tensor.FakeTensorMode.may_turn_const

    def spy_may_turn_const(fake_mode, t):
      numels.append(t.numel())
      return may_turn_const(fake_mode, t)

    with mock.patch.object(
        fake_tensor.FakeTensorMode, "may_turn_const", spy_may_turn_const
    ):
      with mode_lib.fake_tensor_context():
        positions = torch.arange(1024)
        # As in the expansion of an attention mask to `[batch, 1, seq_len,
        # seq_len]`, which FakeTensorMode would compute before it checks the
        # output size.
        rows = _MAX_CONSTANT_NUMEL // 1024 + 1
        repeated = positions[None, :].repeat(rows, 1)
        last_position = positions[-1].item()
    self.assertEqual(repeated.shape, (rows, 1024))
    self.assertEqual(last_position, 1023)
    self.assertNotEmpty(numels)
    self.assertLessEqual(max(numels), _MAX_CONSTANT_NUMEL)

  def test_writing_fake_values_clears_the_constant(self):
    with mode_lib.fake_tensor_context():
      mask = torch.zeros(4, dtype=torch.long)
      mask.copy_(torch.randn(4096)[:4])
      # With its stale constant, this would return 0.
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        mask.sum().item()

  def test_indexing_a_constant_with_fake_indices_stays_fake(self):
    with mode_lib.fake_tensor_context():
      patch_index = torch.arange(8).view(4, 2)
      choice = torch.multinomial(torch.ones(4), 4)
      # As in ViLT's patch selection. The meta function of `index.Tensor` makes
      # its output with `new_empty`, whose uninitialized memory FakeTensorMode
      # would otherwise keep as the values of the output.
      selected = patch_index[choice]
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        selected[0, 0].item()

  def test_tensor_reads_constants_in_its_data_inside(self):
    with mode_lib.fake_tensor_context():
      positions = torch.arange(4)
      # As in Qwen2.5-VL's `torch.tensor(mrope_position_deltas)` of 0-d tensors.
      deltas = torch.tensor([positions.max() + 1, positions.min()]).tolist()
      # Unlike tensors in a sequence, a tensor passed as the data itself
      # dispatches, so FakeTensorMode computes its copy from its constant, and
      # `as_tensor` returns it as is.
      copied = torch.tensor(positions).tolist()
      self.assertIs(torch.as_tensor(positions), positions)
    self.assertEqual(deltas, [4, 0])
    self.assertEqual(copied, [0, 1, 2, 3])

  def test_scripts_functions_that_call_tensor_inside(self):
    with mode_lib.fake_tensor_context():
      # As the `@torch.jit.script` helpers of DeBERTa v2, which transformers
      # imports, and so scripts, while it builds the model.
      @torch.jit.script
      def scaled_size_sqrt(x: torch.Tensor, scale: int) -> torch.Tensor:
        return torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float) * scale)

    self.assertIsInstance(scaled_size_sqrt, torch.jit.ScriptFunction)

  def test_restores_tensor_on_exit(self):
    tensor, as_tensor = torch.tensor, torch.as_tensor
    with mode_lib.fake_tensor_context():
      self.assertIsNot(torch.tensor, tensor)
    self.assertIs(torch.tensor, tensor)
    self.assertIs(torch.as_tensor, as_tensor)

  def test_distribution_arguments_are_not_validated_inside(self):
    validate_args = torch.distributions.Distribution._validate_args  # pylint: disable=protected-access
    with mode_lib.fake_tensor_context():
      # Too large for a constant, as the logits of the TAPAS cell selection.
      logits = torch.randn(2, 2048)
      distribution = torch.distributions.Categorical(logits=logits)
      self.assertIsInstance(distribution.logits, fake_tensor.FakeTensor)
    self.assertEqual(
        torch.distributions.Distribution._validate_args,  # pylint: disable=protected-access
        validate_args,
    )

  def test_data_dependent_ops_fail_by_default(self):
    with mode_lib.fake_tensor_context():
      scores = torch.randn(2, 4096)
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        bool(torch.any(scores > 0))
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        scores[0, 0].item()
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        scores.argmax().item()
      with self.assertRaises(fake_tensor.DynamicOutputShapeException):
        torch.nonzero(scores[0] > 0)
      with self.assertRaises(fake_tensor.DynamicOutputShapeException):
        _ = scores[0][scores[0] > 0]

  def test_allow_data_dependent_fake_outputs_handles_scalars_nonzero_and_masks(
      self,
  ):
    # Outside `fake_tensor_context`, enabling placeholders is a no-op.
    mode_lib.allow_data_dependent_fake_outputs()
    with mode_lib.fake_tensor_context():
      mode_lib.allow_data_dependent_fake_outputs()
      scores = torch.randn(2, 4096)
      complex_scores = torch.zeros(2, dtype=torch.complex64)

      # Non-constant scalar reads return dtype-appropriate placeholders:
      # False for bool (LightGlue early stopping), 0.0 for floats/complex, and
      # 1 for integers (ZoeDepth head index and VITS predicted length).
      self.assertIs(bool(torch.any(scores > 0)), False)
      self.assertEqual(scores[0, 0].item(), 0.0)
      self.assertIsInstance(scores[0, 0].item(), float)
      self.assertEqual(complex_scores[0].item(), 0.0)
      self.assertEqual(scores.argmax().item(), 1)
      self.assertIsInstance(scores.argmax().item(), int)
      # Constant scalars still return their real values.
      self.assertIs(bool(torch.tensor(True)), True)
      self.assertEqual(torch.arange(4)[-1].item(), 3)

      # Non-constant `nonzero` returns a 1-row coordinate tensor, while
      # constant `nonzero` still computes the exact indices.
      keypoints = torch.nonzero(scores > 0)
      self.assertEqual(keypoints.shape, (1, 2))
      self.assertEqual(keypoints.dtype, torch.int64)
      self.assertIsNone(keypoints.constant)
      self.assertEqual(
          torch.nonzero(torch.tensor([1, 0, 2])).tolist(), [[0], [2]]
      )

      # Non-constant boolean masks keep all elements, while constant boolean
      # masks and integer indices are not replaced.
      pruned = scores[:, scores[0] > 0]
      self.assertEqual(pruned.shape, (2, 4096))
      const_masked = scores[torch.tensor([True, False]), scores[0] > 0]
      self.assertEqual(const_masked.shape, (4096,))
      self.assertIsNone(
          torch.arange(8)
          .view(4, 2)[scores.argmax(dim=-1), scores[:2, 0] > 0]
          .constant
      )


if __name__ == "__main__":
  absltest.main()
