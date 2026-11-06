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

"""Tests for perflab.layer_utils.registration."""

import dataclasses
from unittest import mock

from absl.testing import absltest
import torch
from torch.nn import attention
from perflab import common
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab.layer_utils import builders
from perflab.layer_utils import configs
from perflab.layer_utils import registration
from tests import seed_test_utils

_RunMode = common.RunMode
_DeviceKind = target_lib.DeviceKind
_Step = registration.Step

_CONFIGS = (
    configs.LinearConfig(
        batch_size=2, seq_len=4, in_features=8, out_features=16
    ),
    configs.LinearConfig(
        batch_size=1, seq_len=3, in_features=4, out_features=8
    ),
)
_NAME_0 = "batch_size_2_seq_len_4_in_features_8_out_features_16"
_NAME_1 = "batch_size_1_seq_len_3_in_features_4_out_features_8"


def _linear_args(config: configs.LinearConfig) -> builders.LayerArgs:
  return builders.LayerArgs(
      model_name="nn.Linear",
      batch_size=config.batch_size,
      sequence_length=config.seq_len,
      custom_kwargs={
          "in_features": config.in_features,
          "out_features": config.out_features,
      },
  )


def _ctx(platform: target_lib.Platform) -> context_lib.Context:
  return context_lib.Context(
      target=target_lib.make_target(platform),
      run_scope=context_lib.RunScope.FULL,
  )


@dataclasses.dataclass
class _NamedConfig:
  name: str


class _MathSdpEnabledRecorder(torch.nn.Module):
  """Records whether the math SDPA backend is enabled in each forward."""

  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(2, 2)
    self.math_sdp_enabled = []

  def forward(self, x):
    self.math_sdp_enabled.append(torch.backends.cuda.math_sdp_enabled())
    return self.linear(x)


class AddLayerSweepTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.dict(registry_lib.REGISTRY, clear=True))
    self.collection_flag_values = self.enter_context(
        mock.patch.object(
            flags_lib,
            "collection_flag_values",
            autospec=True,
            return_value=(context_lib.RunScope.FULL, False),
        )
    )

  def _sweep(self, **kwargs):
    registration.add_layer_sweep("linear", _CONFIGS, _linear_args, **kwargs)

  def test_spec_names_and_microbenchmark_names(self):
    self._sweep()

    self.assertCountEqual(
        registry_lib.REGISTRY,
        [
            f"layer_linear_train_{_NAME_0}",
            f"layer_linear_eval_{_NAME_0}",
            f"layer_linear_train_{_NAME_1}",
            f"layer_linear_eval_{_NAME_1}",
        ],
    )
    self.assertEqual(
        registry_lib.REGISTRY[
            f"layer_linear_eval_{_NAME_1}"
        ].microbenchmark_name,
        _NAME_1,
    )
    self.assertEqual(
        registry_lib.REGISTRY[
            f"layer_linear_train_{_NAME_0}"
        ].microbenchmark_name,
        _NAME_0,
    )

  def test_steps_select_train_or_eval(self):
    self._sweep(steps=frozenset({_Step.EVAL}))

    self.assertCountEqual(
        registry_lib.REGISTRY,
        [f"layer_linear_eval_{_NAME_0}", f"layer_linear_eval_{_NAME_1}"],
    )

  def test_name_is_sanitized_but_microbenchmark_name_is_not(self):
    registration.add_layer_sweep(
        "sdpa",
        [_NamedConfig(name="SDPBackend.FLASH_ATTENTION x-1")],
        lambda c: builders.LayerArgs(model_name=c.name, batch_size=1),
        steps=frozenset({_Step.EVAL}),
    )

    spec = registry_lib.REGISTRY[
        "layer_sdpa_eval_name_sdpbackend_flash_attention_x_1"
    ]
    self.assertEqual(
        spec.microbenchmark_name, "name_SDPBackend.FLASH_ATTENTION x-1"
    )

  def test_spec_name(self):
    self.assertEqual(
        registration.spec_name(
            "sdpa", _Step.TRAIN, "dtype_bfloat16_backend_SDPBackend.MATH"
        ),
        "layer_sdpa_train_dtype_bfloat16_backend_sdpbackend_math",
    )
    self.assertEqual(
        registration.spec_name("relu", _Step.EVAL, "a..b--c"),
        "layer_relu_eval_a_b_c",
    )
    self.assertEqual(
        registration.spec_name("pool", _Step.EVAL, "size_(2, 2)"),
        "layer_pool_eval_size__2_2",
    )
    self.assertEqual(
        registration.spec_name("flatten", _Step.EVAL, "shape_16x2048x1x1"),
        "layer_flatten_eval_shape_16x2048x1x1",
    )

  def test_train_and_eval_steppers_and_compile_configs(self):
    self._sweep()

    train = registry_lib.REGISTRY[f"layer_linear_train_{_NAME_0}"]
    evaluation = registry_lib.REGISTRY[f"layer_linear_eval_{_NAME_0}"]
    self.assertEqual(train.stepper, step_lib.StepperType.TRAINING)
    self.assertIsNone(train.compile_config)
    self.assertEqual(evaluation.stepper, step_lib.StepperType.FORWARD)
    self.assertEqual(
        evaluation.compile_config,
        compile_lib.CompileConfig(scope=compile_lib.Scope.MODEL),
    )
    self.assertEqual(train.dtype, target_lib.DType.BF16)
    self.assertEqual(evaluation.dtype, target_lib.DType.BF16)

  def test_all_run_modes_skip_nothing(self):
    self._sweep()

    for spec in registry_lib.REGISTRY.values():
      self.assertEqual(spec.skipped_run_modes, frozenset())

  def test_skipped_run_modes_are_the_complement_of_run_modes(self):
    self._sweep(run_modes=frozenset({_RunMode.COMPILED}))

    for spec in registry_lib.REGISTRY.values():
      self.assertEqual(
          spec.skipped_run_modes,
          frozenset({"eager_default", "eager_optimized"}),
      )

  def test_fake_tensor_mode_runs_eager_default_for_accelerator_modes(self):
    self.collection_flag_values.return_value = (
        context_lib.RunScope.FULL,
        True,
    )

    self._sweep(run_modes=frozenset({_RunMode.COMPILED}))

    for spec in registry_lib.REGISTRY.values():
      self.assertEqual(spec.skipped_run_modes, frozenset({"eager_optimized"}))

  def test_fake_tensor_mode_eager_default_only_layer_unchanged(self):
    self.collection_flag_values.return_value = (
        context_lib.RunScope.FULL,
        True,
    )

    self._sweep(run_modes=frozenset({_RunMode.EAGER_DEFAULT}))

    for spec in registry_lib.REGISTRY.values():
      self.assertEqual(
          spec.skipped_run_modes,
          frozenset({"eager_optimized", "compiled"}),
      )

  def test_rejects_unknown_or_empty_run_modes(self):
    with self.assertRaisesRegex(ValueError, "run_modes"):
      self._sweep(run_modes=frozenset())
    with self.assertRaisesRegex(ValueError, "run_modes"):
      self._sweep(
          run_modes=frozenset({_RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING})
      )
    self.assertEmpty(registry_lib.REGISTRY)

  def test_rejects_empty_steps(self):
    with self.assertRaisesRegex(ValueError, "steps"):
      self._sweep(steps=frozenset())
    self.assertEmpty(registry_lib.REGISTRY)

  def test_rejects_cpu_or_empty_device_kinds(self):
    with self.assertRaisesRegex(ValueError, "device_kinds"):
      self._sweep(device_kinds=frozenset())
    with self.assertRaisesRegex(ValueError, "device_kinds"):
      self._sweep(device_kinds=frozenset({_DeviceKind.CPU}))
    self.assertEmpty(registry_lib.REGISTRY)

  def test_every_spec_is_cpu_smoke_only(self):
    self._sweep()

    for spec in registry_lib.REGISTRY.values():
      self.assertEqual(
          spec.presubmit, registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY
      )
      self.assertTrue(spec.presubmit_reason)

  def test_duplicate_registration_raises(self):
    self._sweep()

    with self.assertRaisesRegex(ValueError, "duplicate benchmark name"):
      self._sweep()


class FactoryTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.dict(registry_lib.REGISTRY, clear=True))
    self.enter_context(
        mock.patch.object(
            flags_lib,
            "collection_flag_values",
            autospec=True,
            return_value=(context_lib.RunScope.FULL, False),
        )
    )

  def _factory(self, training: bool, **kwargs):
    step = _Step.TRAIN if training else _Step.EVAL
    registration.add_layer_sweep(
        "linear", _CONFIGS[:1], _linear_args, steps=frozenset({step}), **kwargs
    )
    mode = "train" if training else "eval"
    return registry_lib.REGISTRY[f"layer_linear_{mode}_{_NAME_0}"].factory

  def test_train_factory_on_cpu(self):
    model, args, kwargs, opt = self._factory(training=True)(
        _ctx(target_lib.Platform.CPU)
    )

    self.assertIsInstance(model, torch.nn.Module)
    self.assertTrue(model.training)
    self.assertLen(args, 1)
    self.assertEqual(args[0].shape, (2, 4, 8))
    self.assertEqual(args[0].dtype, torch.bfloat16)
    self.assertEqual(kwargs, {})
    self.assertIsNone(opt)
    self.assertEqual(model(*args).shape, (2, 4, 16))

  def test_eval_factory_on_cpu(self):
    outputs = self._factory(training=False)(_ctx(target_lib.Platform.CPU))

    self.assertLen(outputs, 3)
    model, args, kwargs = outputs
    self.assertFalse(model.training)
    self.assertLen(args, 1)
    self.assertEqual(args[0].shape, (2, 4, 8))
    self.assertEqual(kwargs, {})

  def test_tuple_inputs_become_args(self):
    inputs = (torch.ones(2), torch.zeros(2))
    with mock.patch.object(
        builders,
        "ml_layer_model_builder",
        autospec=True,
        return_value=builders.ModelAndInput(
            model=torch.nn.Identity(), example_inputs=inputs
        ),
    ):
      _, args, _ = self._factory(training=False)(_ctx(target_lib.Platform.CPU))

    self.assertEqual(args, inputs)

  def test_cpu_is_always_allowed(self):
    factory = self._factory(
        training=False, device_kinds=frozenset({_DeviceKind.TPU})
    )

    model, _, _ = factory(_ctx(target_lib.Platform.CPU))

    self.assertIsInstance(model, torch.nn.Module)

  def test_excluded_accelerator_raises_unsupported_benchmark(self):
    factory = self._factory(
        training=True, device_kinds=frozenset({_DeviceKind.TPU})
    )

    with mock.patch.object(
        builders, "ml_layer_model_builder", autospec=True
    ) as builder:
      with self.assertRaises(target_lib.UnsupportedBenchmark):
        factory(_ctx(target_lib.Platform.B200_1))
    builder.assert_not_called()

  def test_data_dependent_allows_data_dependent_fake_outputs(self):
    with mock.patch.object(
        mode_lib, "allow_data_dependent_fake_outputs", autospec=True
    ) as allow:
      self._factory(training=False, data_dependent=True)(
          _ctx(target_lib.Platform.CPU)
      )
    allow.assert_called_once_with()

  def test_not_data_dependent_by_default(self):
    with mock.patch.object(
        mode_lib, "allow_data_dependent_fake_outputs", autospec=True
    ) as allow:
      self._factory(training=False)(_ctx(target_lib.Platform.CPU))
    allow.assert_not_called()

  def test_cpu_builds_with_layer_args_by_default(self):
    factory = self._factory(training=False)

    _, builder = self._run_with_inner(
        factory, target_lib.Platform.CPU, torch.nn.Identity()
    )

    builder.assert_called_once_with(
        _linear_args(_CONFIGS[0]), "cpu", torch.bfloat16, False
    )

  def test_cpu_layer_args_replace_layer_args_only_on_cpu(self):
    cpu_args = builders.LayerArgs(model_name="cpu stand-in", batch_size=1)
    to_cpu_args = mock.Mock(return_value=cpu_args)
    factory = self._factory(training=True, cpu_layer_args=to_cpu_args)

    to_cpu_args.assert_called_once_with(_CONFIGS[0])
    for platform, device, expected_args in (
        (target_lib.Platform.CPU, "cpu", cpu_args),
        (target_lib.Platform.V7_1X1X1, "tpu", _linear_args(_CONFIGS[0])),
        (target_lib.Platform.B200_1, "cuda", _linear_args(_CONFIGS[0])),
    ):
      with self.subTest(platform=platform):
        _, builder = self._run_with_inner(
            factory, platform, torch.nn.Identity()
        )

        builder.assert_called_once_with(
            expected_args, device, torch.bfloat16, True
        )

  def test_cpu_layer_args_keep_names(self):
    registration.add_layer_sweep(
        "linear",
        _CONFIGS[:1],
        _linear_args,
        steps=frozenset({_Step.EVAL}),
        cpu_layer_args=lambda _: builders.LayerArgs(
            model_name="cpu stand-in", batch_size=1
        ),
    )

    self.assertCountEqual(
        registry_lib.REGISTRY, [f"layer_linear_eval_{_NAME_0}"]
    )
    self.assertEqual(
        registry_lib.REGISTRY[
            f"layer_linear_eval_{_NAME_0}"
        ].microbenchmark_name,
        _NAME_0,
    )

  def _run_with_inner(
      self,
      factory: registry_lib.Factory,
      platform: target_lib.Platform,
      inner: torch.nn.Module,
  ):
    """Runs `factory` with a builder that returns `inner` and a 1D input."""
    with mock.patch.object(
        builders,
        "ml_layer_model_builder",
        autospec=True,
        return_value=builders.ModelAndInput(
            model=inner, example_inputs=torch.ones(2)
        ),
    ) as builder:
      outputs = factory(_ctx(platform))
    return outputs, builder

  def test_no_sdpa_wrapper_off_tpu(self):
    factory = self._factory(training=False)
    for platform in (target_lib.Platform.CPU, target_lib.Platform.B200_1):
      with self.subTest(platform=platform):
        inner = _MathSdpEnabledRecorder()

        (model, _, _), _ = self._run_with_inner(factory, platform, inner)

        self.assertIs(model, inner)

  def test_sdpa_wrapper_on_tpu(self):
    inner = _MathSdpEnabledRecorder()

    (model, args, _, opt), builder = self._run_with_inner(
        self._factory(training=True), target_lib.Platform.V7_1X1X1, inner
    )

    builder.assert_called_once_with(
        _linear_args(_CONFIGS[0]),
        "tpu",
        torch.bfloat16,
        True,
    )
    self.assertIsNone(opt)
    self.assertIsNot(model, inner)
    self.assertTrue(model.training)
    self.assertTrue(inner.training)
    model(*args)
    # The math backend is disabled inside the wrapper's forward only.
    self.assertEqual(inner.math_sdp_enabled, [False])
    self.assertTrue(torch.backends.cuda.math_sdp_enabled())

  def test_sdpa_wrapper_eval_mode_and_inputs(self):

    class _Echo(torch.nn.Module):

      def forward(self, *args, **kwargs):
        return args, kwargs

    inner = _Echo()

    (model, _, _), _ = self._run_with_inner(
        self._factory(training=False), target_lib.Platform.V7_1X1X1, inner
    )

    self.assertFalse(model.training)
    self.assertFalse(inner.training)
    self.assertEqual(model(1, 2, k=3), ((1, 2), {"k": 3}))

  def test_inner_sdpa_backend_context_wins(self):

    class _InnerMath(torch.nn.Module):

      def forward(self, x):
        with attention.sdpa_kernel([attention.SDPBackend.MATH]):
          return x, torch.backends.cuda.math_sdp_enabled()

    (model, args, _), _ = self._run_with_inner(
        self._factory(training=False),
        target_lib.Platform.V7_1X1X1,
        _InnerMath(),
    )

    _, math_enabled = model(*args)
    self.assertTrue(math_enabled)


class SdpaKernelWrapperCompileTest(seed_test_utils.RepeatableTest):

  def test_compiles_without_graph_breaks(self):
    self.addCleanup(torch._dynamo.reset)  # pylint: disable=protected-access
    # The CPU has no overrideable SDPA kernel, so the wrapped layer does not
    # call SDPA. The test checks that dynamo traces the `sdpa_kernel` context.
    x = torch.randn(2, 4)
    wrapper = registration._SdpaKernelWrapper(torch.nn.Linear(4, 3))  # pylint: disable=protected-access

    out = torch.compile(wrapper, backend="eager", fullgraph=True)(x)

    self.assertEqual(out.shape, (2, 3))


if __name__ == "__main__":
  absltest.main()
