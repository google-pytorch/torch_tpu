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

"""Checks the names and policies of the layer benchmarks in `perflab.layers`.

The layer benchmarks run from their own binary
(`torch_layer_benchmark_test`), and the model benchmarks from
`torch_benchmark_test`. Both export results under the same benchmark namespace,
so these checks keep the two sets of names disjoint, and keep the layer
benchmark names parseable by the dashboards. They read the registry, and call
factories only to check device gating with a stub layer; the FakeTensorMode CPU
smoke test (`torch_layer_benchmark_cpu_smoke_test`) builds and steps the layers.
"""

import enum
import re
import types
from unittest import mock

from absl.testing import absltest
import torch
from torch.nn import attention
from perflab import context as context_lib
from perflab import discovery as discovery_lib
from perflab import layers
from perflab import mode as mode_lib
from perflab import models
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab.layer_utils import builders
from perflab.layer_utils import configs
from perflab.layer_utils import registration
from tests import seed_test_utils

# The cap on the exported test name of a layer benchmark case,
# `torch_tpu/<platform>/<benchmark>/<run mode>/<framework>`. Results are keyed
# and filtered by this name. The cap is a policy headroom well below
# MLCompass's 1,500-byte limit on the indexed Datastore string properties that
# hold values derived from these names. With the longest platform, layer run
# mode and framework, the longest sdpa benchmark has a 253-character test name.
_MAX_TEST_NAME_LENGTH = 512

# How the dashboards split a layer benchmark name into its benchmark and step.
_DASHBOARD_NAME_PATTERN = re.compile(r"^layer_(.+?)_(train|eval)_")

_LINEAR_SMOKE_MICROBENCHMARK = (
    "batch_size_1_seq_len_128_in_features_128_out_features_128"
)

_LAYERNORM_SMOKE_MICROBENCHMARK = (
    "batch_size_1_seq_len_128_num_features_128_num_normalized_dims_1"
)

# The smoke shape of the sdpa layer, without its backend.
_SDPA_SMOKE_MICROBENCHMARK = (
    "batch_size_1_embed_dim_4096_q_seq_len_128_q_num_heads_8_kv_num_heads_8"
    "_qk_head_dim_64_v_head_dim_64_is_causal_True_enable_gqa_True"
    "_dtype_bfloat16"
)

# An sdpa shape with fewer key/value heads than query heads, without its
# backend.
_SDPA_UNEQUAL_HEADS_MICROBENCHMARK = (
    "batch_size_4_embed_dim_2048_q_seq_len_2048_q_num_heads_16_kv_num_heads_2"
    "_qk_head_dim_128_v_head_dim_128_is_causal_True_enable_gqa_True"
    "_dtype_bfloat16"
)

_SDPA_TPU_BACKENDS = (attention.SDPBackend.FLASH_ATTENTION,)
_SDPA_CUDA_BACKENDS = (
    attention.SDPBackend.CUDNN_ATTENTION,
    attention.SDPBackend.EFFICIENT_ATTENTION,
)


def _import_benchmarks(
    package: types.ModuleType,
) -> tuple[list[discovery_lib.ImportFailure], frozenset[str]]:
  """Imports the benchmark modules of `package`, as the binaries do.

  Args:
    package: The package of benchmark modules to import.

  Returns:
    The import failures, and the names of the benchmarks that the import
    registered.
  """
  before = set(registry_lib.REGISTRY)
  failures = discovery_lib.import_submodules(package)
  return failures, frozenset(set(registry_lib.REGISTRY) - before)


_LAYER_IMPORT_FAILURES, _LAYER_NAMES = _import_benchmarks(layers)
# Model import failures are ignored: some model modules may not import here,
# as this target has no HuggingFace model config data, so the check that
# models register no layer benchmark only covers the models that import.
_, _MODEL_NAMES = _import_benchmarks(models)


def _layer_specs() -> list[registry_lib.BenchmarkSpec]:
  return [registry_lib.REGISTRY[name] for name in sorted(_LAYER_NAMES)]


def _step(spec: registry_lib.BenchmarkSpec) -> registration.Step:
  if spec.stepper is step_lib.StepperType.TRAINING:
    return registration.Step.TRAIN
  return registration.Step.EVAL


def _benchmark(spec: registry_lib.BenchmarkSpec) -> str:
  """Returns the layer benchmark of `spec`, e.g. `"linear"`, from its name."""
  # The name of an empty benchmark is `layer__<step>_<microbenchmark>`.
  suffix = registration.spec_name(
      "", _step(spec), spec.microbenchmark_name
  ).removeprefix("layer_")
  return spec.name.removeprefix("layer_").removesuffix(suffix)


def _longest_value(enum_type: type[enum.Enum]) -> str:
  return max((member.value for member in enum_type), key=len)


class LayersContractTest(seed_test_utils.RepeatableTest):

  def test_layers_import_without_failures(self):
    self.assertEmpty([str(failure) for failure in _LAYER_IMPORT_FAILURES])

  def test_layers_register_benchmarks(self):
    self.assertNotEmpty(_LAYER_NAMES)

  def test_layer_benchmark_names_start_with_layer(self):
    self.assertEmpty(
        sorted(name for name in _LAYER_NAMES if not name.startswith("layer_"))
    )

  def test_models_register_no_layer_benchmarks(self):
    self.assertNotEmpty(_MODEL_NAMES)
    self.assertEmpty(
        sorted(name for name in _MODEL_NAMES if name.startswith("layer_"))
    )

  def test_layer_benchmarks_set_microbenchmark_name(self):
    self.assertEmpty(
        [spec.name for spec in _layer_specs() if not spec.microbenchmark_name]
    )

  def test_layer_benchmark_names_derive_from_benchmark_step_and_micro(self):
    for spec in _layer_specs():
      with self.subTest(spec.name):
        self.assertIn(
            spec.stepper,
            (step_lib.StepperType.TRAINING, step_lib.StepperType.FORWARD),
        )
        benchmark = _benchmark(spec)
        self.assertNotEmpty(benchmark)
        self.assertEqual(
            registration.spec_name(
                benchmark, _step(spec), spec.microbenchmark_name
            ),
            spec.name,
        )

  def test_dashboards_parse_benchmark_from_layer_benchmark_names(self):
    for spec in _layer_specs():
      with self.subTest(spec.name):
        benchmark = _benchmark(spec)
        self.assertNotIn("_train_", f"_{benchmark}_")
        self.assertNotIn("_eval_", f"_{benchmark}_")
        match = _DASHBOARD_NAME_PATTERN.match(spec.name)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), benchmark)
        self.assertEqual(match.group(2), _step(spec).value)

  def test_layer_benchmarks_run_in_presubmit_scope_only_in_cpu_smoke(self):
    for spec in _layer_specs():
      with self.subTest(spec.name):
        self.assertIs(
            spec.presubmit, registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY
        )
        self.assertTrue(spec.presubmit_reason)

  def test_exported_test_names_fit_cap(self):
    # The longest platform, layer run mode and framework, whether or not a
    # layer benchmark runs on or in them today. The debug run mode
    # `eager_defer_never_and_launch_blocking` is not a layer run mode: it only
    # runs when requested explicitly.
    platform = _longest_value(target_lib.Platform)
    run_mode = max((mode.value for mode in registration.ALL_RUN_MODES), key=len)
    framework = _longest_value(mode_lib.Framework)
    too_long = []
    for name in _LAYER_NAMES:
      test_name = f"torch_tpu/{platform}/{name}/{run_mode}/{framework}"
      if len(test_name) > _MAX_TEST_NAME_LENGTH:
        too_long.append(f"{test_name} ({len(test_name)} characters)")
    self.assertEmpty(sorted(too_long))

  # TODO(jihonggan): Check that every benchmark prefix of the layer known
  # failures matches a registered layer benchmark and no model benchmark, once
  # the layer known failures are available here.


class LinearLayersTest(seed_test_utils.RepeatableTest):

  def test_registers_train_and_eval_of_every_config(self):
    linear_names = {
        name for name in _LAYER_NAMES if name.startswith("layer_linear_")
    }
    self.assertLen(linear_names, 2 * len(configs.LINEAR_CONFIGS))
    self.assertContainsSubset(
        {
            f"layer_linear_train_{_LINEAR_SMOKE_MICROBENCHMARK}",
            f"layer_linear_eval_{_LINEAR_SMOKE_MICROBENCHMARK}",
        },
        linear_names,
    )

  def test_smoke_config_runs_in_every_run_mode(self):
    for step in ("train", "eval"):
      with self.subTest(step):
        spec = registry_lib.REGISTRY[
            f"layer_linear_{step}_{_LINEAR_SMOKE_MICROBENCHMARK}"
        ]
        self.assertEqual(spec.microbenchmark_name, _LINEAR_SMOKE_MICROBENCHMARK)
        self.assertEmpty(spec.skipped_run_modes)


def _names_with_prefix(prefix: str) -> set[str]:
  return {name for name in _LAYER_NAMES if name.startswith(prefix)}


class LayerNormLayersTest(seed_test_utils.RepeatableTest):

  def test_registers_train_and_eval_of_every_config(self):
    layernorm_names = _names_with_prefix("layer_layernorm_")
    self.assertLen(layernorm_names, 2 * len(configs.LAYER_NORM_CONFIGS))
    self.assertLen(layernorm_names, 8)
    self.assertContainsSubset(
        {
            f"layer_layernorm_train_{_LAYERNORM_SMOKE_MICROBENCHMARK}",
            f"layer_layernorm_eval_{_LAYERNORM_SMOKE_MICROBENCHMARK}",
        },
        layernorm_names,
    )

  def test_smoke_config_runs_in_every_run_mode(self):
    for step in ("train", "eval"):
      with self.subTest(step):
        spec = registry_lib.REGISTRY[
            f"layer_layernorm_{step}_{_LAYERNORM_SMOKE_MICROBENCHMARK}"
        ]
        self.assertEqual(
            spec.microbenchmark_name, _LAYERNORM_SMOKE_MICROBENCHMARK
        )
        self.assertEmpty(spec.skipped_run_modes)


def _sdpa_name(step: str, shape: str, backend: attention.SDPBackend) -> str:
  """Returns the spec name of the sdpa `shape` with `backend`."""
  return registration.spec_name(
      "sdpa", registration.Step(step), f"{shape}_backend_{backend}"
  )


def _sdpa_names(backends: tuple[attention.SDPBackend, ...]) -> set[str]:
  """Returns the registered sdpa spec names with one of `backends`."""
  return {
      name
      for name in _names_with_prefix("layer_sdpa_")
      if any(
          registry_lib.REGISTRY[name].microbenchmark_name.endswith(
              f"_backend_{backend}"
          )
          for backend in backends
      )
  }


def _ctx(platform: target_lib.Platform) -> context_lib.Context:
  return context_lib.Context(
      target=target_lib.make_target(platform),
      run_scope=context_lib.RunScope.FULL,
  )


def _stub_builder():
  """Returns a context manager that stubs the layer builder with a no-op layer.

  The context manager yields the stub builder.
  """
  return mock.patch.object(
      builders,
      "ml_layer_model_builder",
      autospec=True,
      return_value=builders.ModelAndInput(
          model=torch.nn.Identity(), example_inputs=torch.ones(2)
      ),
  )


class SdpaLayersTest(seed_test_utils.RepeatableTest):

  def test_registers_only_tpu_and_cuda_backends(self):
    sdpa_names = _names_with_prefix("layer_sdpa_")
    self.assertLen(sdpa_names, 46)
    self.assertEqual(
        sdpa_names,
        _sdpa_names(_SDPA_TPU_BACKENDS) | _sdpa_names(_SDPA_CUDA_BACKENDS),
    )

  def test_tpu_backend_registers_train_and_eval_of_every_config(self):
    tpu_names = _sdpa_names(_SDPA_TPU_BACKENDS)
    self.assertLen(tpu_names, 2 * len(configs.SdpaConfig.get_base_configs()))
    self.assertLen(tpu_names, 16)
    self.assertContainsSubset(
        {
            _sdpa_name(step, shape, attention.SDPBackend.FLASH_ATTENTION)
            for step in ("train", "eval")
            for shape in (
                _SDPA_SMOKE_MICROBENCHMARK,
                _SDPA_UNEQUAL_HEADS_MICROBENCHMARK,
            )
        },
        tpu_names,
    )

  def test_cuda_backends_register_train_and_eval_of_supported_configs(self):
    cuda_names = _sdpa_names(_SDPA_CUDA_BACKENDS)
    # Efficient attention skips the one config with unequal head counts.
    self.assertLen(cuda_names, 30)
    self.assertContainsSubset(
        {
            _sdpa_name(step, _SDPA_SMOKE_MICROBENCHMARK, backend)
            for step in ("train", "eval")
            for backend in _SDPA_CUDA_BACKENDS
        },
        cuda_names,
    )

  def test_efficient_attention_skips_unequal_head_counts(self):
    sdpa_names = _names_with_prefix("layer_sdpa_")
    for step in ("train", "eval"):
      with self.subTest(step):
        self.assertIn(
            _sdpa_name(
                step,
                _SDPA_UNEQUAL_HEADS_MICROBENCHMARK,
                attention.SDPBackend.CUDNN_ATTENTION,
            ),
            sdpa_names,
        )
        self.assertNotIn(
            _sdpa_name(
                step,
                _SDPA_UNEQUAL_HEADS_MICROBENCHMARK,
                attention.SDPBackend.EFFICIENT_ATTENTION,
            ),
            sdpa_names,
        )
    for name in _sdpa_names((attention.SDPBackend.EFFICIENT_ATTENTION,)):
      with self.subTest(name):
        self.assertRegex(name, r"_q_num_heads_(\d+)_kv_num_heads_\1_")

  def test_smoke_config_runs_in_every_run_mode(self):
    for backend in _SDPA_TPU_BACKENDS + _SDPA_CUDA_BACKENDS:
      for step in ("train", "eval"):
        with self.subTest(backend=backend, step=step):
          spec = registry_lib.REGISTRY[
              _sdpa_name(step, _SDPA_SMOKE_MICROBENCHMARK, backend)
          ]
          self.assertEqual(
              spec.microbenchmark_name,
              f"{_SDPA_SMOKE_MICROBENCHMARK}_backend_{backend}",
          )
          self.assertEmpty(spec.skipped_run_modes)

  def _assert_device_gating(
      self,
      backend: attention.SDPBackend,
      supported: target_lib.Platform,
      unsupported: target_lib.Platform,
  ):
    for step in ("train", "eval"):
      factory = registry_lib.REGISTRY[
          _sdpa_name(step, _SDPA_SMOKE_MICROBENCHMARK, backend)
      ].factory
      for platform in (target_lib.Platform.CPU, supported, unsupported):
        with self.subTest(step=step, platform=platform):
          with _stub_builder() as builder:
            if platform is unsupported:
              with self.assertRaises(target_lib.UnsupportedBenchmark):
                factory(_ctx(platform))
              builder.assert_not_called()
            else:
              factory(_ctx(platform))
              builder.assert_called_once()

  def test_tpu_backend_runs_only_on_tpu_and_cpu(self):
    self._assert_device_gating(
        attention.SDPBackend.FLASH_ATTENTION,
        supported=target_lib.Platform.V7_1X1X1,
        unsupported=target_lib.Platform.B200_1,
    )

  def test_cuda_backends_run_only_on_cuda_and_cpu(self):
    for backend in _SDPA_CUDA_BACKENDS:
      with self.subTest(backend=backend):
        self._assert_device_gating(
            backend,
            supported=target_lib.Platform.B200_1,
            unsupported=target_lib.Platform.V7_1X1X1,
        )

  def _built_backend(
      self, backend: attention.SDPBackend, platform: target_lib.Platform
  ) -> attention.SDPBackend:
    """Returns the SDPA backend that the smoke spec of `backend` builds with."""
    factory = registry_lib.REGISTRY[
        _sdpa_name("train", _SDPA_SMOKE_MICROBENCHMARK, backend)
    ].factory
    with _stub_builder() as builder:
      factory(_ctx(platform))
    builder.assert_called_once()
    return builder.call_args.args[0].custom_kwargs["backend"]

  def test_accelerators_build_with_the_spec_backend(self):
    for backend, platform in (
        (attention.SDPBackend.FLASH_ATTENTION, target_lib.Platform.V7_1X1X1),
        (attention.SDPBackend.CUDNN_ATTENTION, target_lib.Platform.B200_1),
        (attention.SDPBackend.EFFICIENT_ATTENTION, target_lib.Platform.B200_1),
    ):
      with self.subTest(backend=backend):
        self.assertEqual(self._built_backend(backend, platform), backend)

  def test_cpu_builds_with_a_cpu_backend(self):
    # The CPU has flash attention, but no cuDNN or memory-efficient attention.
    for backend, cpu_backend in (
        (
            attention.SDPBackend.FLASH_ATTENTION,
            attention.SDPBackend.FLASH_ATTENTION,
        ),
        (attention.SDPBackend.CUDNN_ATTENTION, attention.SDPBackend.MATH),
        (attention.SDPBackend.EFFICIENT_ATTENTION, attention.SDPBackend.MATH),
    ):
      with self.subTest(backend=backend):
        self.assertEqual(
            self._built_backend(backend, target_lib.Platform.CPU), cpu_backend
        )


if __name__ == "__main__":
  absltest.main()
