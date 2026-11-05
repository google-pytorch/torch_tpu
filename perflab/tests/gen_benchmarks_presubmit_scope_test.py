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

"""Tests the registration and presubmit policies of the generated benchmarks.

Every generated benchmark is registered regardless of the run scope, and their
`PresubmitPolicy` keeps them out of presubmit scope except in the FakeTensorMode
CPU smoke (`--fake_tensor_mode`). Under `--fake_tensor_mode`, only the ones
whose model config is in-tree are registered, so that the smoke test needs no
network access.
"""

import copy
import os
import sys
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import torch
from perflab import cases
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import target as target_lib
from perflab.models import gen_benchmarks
from tests import seed_test_utils

# Generated benchmarks that broke in the nightly runs after cl/999488021
# changed the shared model loading, while presubmit stayed green.
_MODERNBERT_BENCHMARKS = (
    "colmodernvbert_merged_st_forward_gen",
    "granite_embedding_small_english_r2_forward_gen",
    "modernbert_base_forward_gen",
    "modernbert_base_train_gen",
    "time_anchor_modernbert_32m_forward_gen",
)


def _gen_names() -> set[str]:
  """Returns the names of all generated benchmarks."""
  return {
      spec.benchmark_name(is_training=is_training, suffix="gen")
      for spec, is_training, _ in gen_benchmarks._iter_gen_benchmark_specs()  # pylint: disable=protected-access
  }


def _fake_tensor_smoke_names() -> set[str]:
  """Returns the generated benchmarks expected in the FakeTensorMode smoke."""
  names = set()
  for spec, is_training, _ in gen_benchmarks._iter_gen_benchmark_specs():  # pylint: disable=protected-access
    name = spec.benchmark_name(is_training=is_training, suffix="gen")
    if (
        name not in gen_benchmarks._FAKE_TENSOR_MODE_UNSUPPORTED  # pylint: disable=protected-access
        and gen_benchmarks.has_in_tree_config(spec.model_id)
    ):
      names.add(name)
  return names


class GenBenchmarksPresubmitScopeTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.dict(registry_lib.REGISTRY, clear=True))
    # `--fake_tensor_mode` is only valid on CPU.
    self.enter_context(
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"})
    )

  def _register(
      self, run_scope: context_lib.RunScope, fake_tensor_mode: bool
  ) -> dict[str, registry_lib.BenchmarkSpec]:
    with flagsaver.flagsaver(
        run_scope=run_scope, fake_tensor_mode=fake_tensor_mode
    ):
      gen_benchmarks._register_gen_benchmarks()  # pylint: disable=protected-access
    return dict(registry_lib.REGISTRY)

  def _registered_names_at_import_time(self, argv: list[str]) -> set[str]:
    # Unparsed flags, as when production imports gen_benchmarks.
    import_time_flags = flags.FlagValues()
    for flag in (context_lib.RUN_SCOPE, flags_lib.FAKE_TENSOR_MODE):
      import_time_flags[flag.name] = copy.deepcopy(flags.FLAGS[flag.name])
    with (
        mock.patch.object(flags, "FLAGS", import_time_flags),
        mock.patch.object(sys, "argv", argv),
    ):
      gen_benchmarks._register_gen_benchmarks()  # pylint: disable=protected-access
    return set(registry_lib.REGISTRY)

  def _case_benchmarks(
      self, run_scope: context_lib.RunScope, fake_tensor_mode: bool
  ) -> set[str]:
    with flagsaver.flagsaver(
        run_scope=run_scope, fake_tensor_mode=fake_tensor_mode
    ):
      gen_benchmarks._register_gen_benchmarks()  # pylint: disable=protected-access
      return {
          spec.name
          for _, spec, _ in cases.get_cases(
              target_lib.Platform.CPU, mode_lib.Framework.TORCH
          )
      }

  def test_fake_tensor_smoke_covers_every_in_tree_model(self):
    names = _fake_tensor_smoke_names()
    # Most generated benchmarks have in-tree configs. A much smaller set means
    # that the smoke test lost its configs (e.g. a missing data dependency).
    self.assertGreater(len(names), 300)
    self.assertContainsSubset(_MODERNBERT_BENCHMARKS, names)

  @parameterized.parameters(
      context_lib.RunScope.FULL, context_lib.RunScope.PRESUBMIT
  )
  def test_registers_every_generated_benchmark_without_fake_tensor_mode(
      self, run_scope: context_lib.RunScope
  ):
    self.assertEqual(set(self._register(run_scope, False)), _gen_names())

  @parameterized.parameters(
      context_lib.RunScope.FULL, context_lib.RunScope.PRESUBMIT
  )
  def test_fake_tensor_mode_registers_in_tree_models(
      self, run_scope: context_lib.RunScope
  ):
    self.assertEqual(
        set(self._register(run_scope, True)), _fake_tensor_smoke_names()
    )

  def test_import_time_fake_tensor_mode_registers_in_tree_models(self):
    self.assertEqual(
        self._registered_names_at_import_time(
            ["prog", "--run_scope=presubmit", "--fake_tensor_mode"]
        ),
        _fake_tensor_smoke_names(),
    )

  def test_presubmit_policies(self):
    for name, spec in self._register(context_lib.RunScope.FULL, False).items():
      with self.subTest(name=name):
        self.assertIs(
            spec.presubmit, registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY
        )
        self.assertTrue(spec.presubmit_reason)

  def test_fake_tensor_smoke_steps_benchmarks_that_run_on_accelerators(self):
    skipped_run_modes = {
        name: spec.skipped_run_modes
        for name, spec in self._register(
            context_lib.RunScope.PRESUBMIT, True
        ).items()
    }
    # Skips eager_default, the only run mode of CPU targets, but runs compiled
    # on TPUs and GPUs.
    self.assertNotIn("eager_default", skipped_run_modes["t5_small_train_gen"])
    # Runs in no run mode.
    self.assertIn(
        "eager_default", skipped_run_modes["all_minilm_l6_v2_train_gen"]
    )

  def test_full_scope_collects_every_generated_benchmark(self):
    self.assertEqual(
        self._case_benchmarks(context_lib.RunScope.FULL, False), _gen_names()
    )

  @parameterized.parameters(
      context_lib.RunScope.FULL, context_lib.RunScope.PRESUBMIT
  )
  def test_cpu_smoke_collects_in_tree_models(
      self, run_scope: context_lib.RunScope
  ):
    self.assertEqual(
        self._case_benchmarks(run_scope, True), _fake_tensor_smoke_names()
    )

  def test_presubmit_without_fake_tensor_mode_collects_nothing(self):
    self.assertEmpty(
        self._case_benchmarks(context_lib.RunScope.PRESUBMIT, False)
    )

  @parameterized.parameters(
      ("answerdotai/ModernBERT-base", True),
      ("ibm-granite/granite-embedding-small-english-r2", True),
      ("Qwen/Qwen3.5-397B-A17B", True),
      ("Alibaba-NLP/gte-modernbert-base", False),
      ("not-an-org/not-a-model", False),
  )
  def test_has_in_tree_config(self, model_id: str, expected: bool):
    self.assertEqual(gen_benchmarks.has_in_tree_config(model_id), expected)

  def test_skipped_benchmarks_have_in_tree_configs_and_reasons(self):
    model_ids = {
        spec.benchmark_name(
            is_training=is_training, suffix="gen"
        ): spec.model_id
        for spec, is_training, _ in gen_benchmarks._iter_gen_benchmark_specs()  # pylint: disable=protected-access
    }
    for name, reason in gen_benchmarks._FAKE_TENSOR_MODE_UNSUPPORTED.items():  # pylint: disable=protected-access
      with self.subTest(name):
        # Entries for benchmarks that are no longer generated, or whose model
        # config is not in-tree, skip nothing, so they must be removed.
        self.assertIn(name, model_ids)
        self.assertTrue(gen_benchmarks.has_in_tree_config(model_ids[name]))
        self.assertNotEmpty(reason)

  def test_data_dependent_benchmarks_have_in_tree_configs_and_opt_in(self):
    model_ids = {
        spec.benchmark_name(
            is_training=is_training, suffix="gen"
        ): spec.model_id
        for spec, is_training, _ in gen_benchmarks._iter_gen_benchmark_specs()  # pylint: disable=protected-access
    }
    for name in gen_benchmarks._FAKE_TENSOR_MODE_DATA_DEPENDENT:  # pylint: disable=protected-access
      with self.subTest(name):
        self.assertIn(name, model_ids)
        self.assertTrue(gen_benchmarks.has_in_tree_config(model_ids[name]))

    fake_load_fn = lambda ctx, model_id, is_training: (
        torch.nn.Linear(2, 2),
        {},
    )
    ctx = context_lib.Context(
        target=target_lib.make_target(target_lib.Platform.CPU),
        run_scope=context_lib.RunScope.PRESUBMIT,
    )
    default_factory = gen_benchmarks._make_factory(  # pylint: disable=protected-access
        "fake/model", False, fake_load_fn
    )
    with mode_lib.fake_tensor_context():
      default_factory(ctx)
      self.assertFalse(
          torch._guards.detect_fake_mode().allow_data_dependent_outputs  # pylint: disable=protected-access
      )
    opt_in_factory = gen_benchmarks._make_factory(  # pylint: disable=protected-access
        "fake/model",
        False,
        fake_load_fn,
        allow_data_dependent_outputs=True,
    )
    with mode_lib.fake_tensor_context():
      opt_in_factory(ctx)
      self.assertTrue(
          torch._guards.detect_fake_mode().allow_data_dependent_outputs  # pylint: disable=protected-access
      )

  def test_skip_reason_of_a_model_without_an_in_tree_config(self):
    self.assertEqual(
        gen_benchmarks.fake_tensor_smoke_skip_reason(
            "Alibaba-NLP/gte-modernbert-base", "gte_modernbert_base_forward_gen"
        ),
        "the config of Alibaba-NLP/gte-modernbert-base is not in-tree",
    )


if __name__ == "__main__":
  absltest.main()
