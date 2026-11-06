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
benchmark names parseable by the dashboards. They only read
the registry; the FakeTensorMode CPU smoke test
(`torch_layer_benchmark_cpu_smoke_test`) builds and steps the layers.
"""

import enum
import re
import types

from absl.testing import absltest
from perflab import common
from perflab import discovery as discovery_lib
from perflab import layers
from perflab import mode as mode_lib
from perflab import models
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab.layer_utils import configs
from perflab.layer_utils import registration
from tests import seed_test_utils

# The cap on the exported test name of a layer benchmark case,
# `torch_tpu/<platform>/<benchmark>/<run mode>/<framework>`. Results are keyed,
# filtered and stored by this name, so it must stay usable as a key and as a
# file name, which most file systems limit to 255 bytes. With the longest
# platform, run mode and framework, the longest linear benchmark has a
# 150-character test name, which leaves layers with longer microbenchmark
# names about 100 characters of headroom.
_MAX_TEST_NAME_LENGTH = 255

# How the dashboards split a layer benchmark name into its benchmark and step.
_DASHBOARD_NAME_PATTERN = re.compile(r"^layer_(.+?)_(train|eval)_")

_LINEAR_SMOKE_MICROBENCHMARK = (
    "batch_size_1_seq_len_128_in_features_128_out_features_128"
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
    # The longest platform, run mode and framework, whether or not a layer
    # benchmark runs on or in them today.
    platform = _longest_value(target_lib.Platform)
    run_mode = _longest_value(common.RunMode)
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


if __name__ == "__main__":
  absltest.main()
