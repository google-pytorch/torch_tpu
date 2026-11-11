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

"""Tests for perflab/cases.py."""

from collections.abc import Mapping, Sequence
import itertools
import os
import random
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from perflab import cases
from perflab import common
from perflab import context as context_lib
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import workload as workload_lib
from tests import seed_test_utils


def _factory() -> tuple[Any, ...]:
  return ("model", (), {}, None)


_RUN = registry_lib.PresubmitPolicy.RUN


def _make_spec(
    name: str,
    skipped_run_modes: frozenset[str] = frozenset(),
    stepper: step_lib.StepperType = step_lib.StepperType.FORWARD,
    presubmit: registry_lib.PresubmitPolicy = _RUN,
    presubmit_reason: str | None = None,
) -> registry_lib.BenchmarkSpec:
  return registry_lib.BenchmarkSpec(
      name=name,
      factory=_factory,
      stepper=stepper,
      skipped_run_modes=skipped_run_modes,
      presubmit=presubmit,
      presubmit_reason=presubmit_reason,
  )


# One spec per presubmit policy.
_PRESUBMIT_REGISTRY = {
    "run": _make_spec("run", presubmit=registry_lib.PresubmitPolicy.RUN),
    "smoke_only": _make_spec(
        "smoke_only",
        presubmit=registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY,
        presubmit_reason="too big",
    ),
    "skipped": _make_spec(
        "skipped",
        presubmit=registry_lib.PresubmitPolicy.SKIP,
        presubmit_reason="never in presubmit",
    ),
}


class PresubmitPolicyTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # `--fake_tensor_mode` is only valid on CPU.
    self.enter_context(
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"})
    )

  def _case_benchmarks(
      self, run_scope: context_lib.RunScope, fake_tensor_mode: bool
  ) -> set[str]:
    with (
        mock.patch.dict(registry_lib.REGISTRY, _PRESUBMIT_REGISTRY, clear=True),
        flagsaver.flagsaver(
            run_scope=run_scope, fake_tensor_mode=fake_tensor_mode
        ),
    ):
      return {
          spec.name
          for _, spec, _ in cases.get_cases(
              target_lib.Platform.CPU, mode_lib.Framework.TORCH
          )
      }

  def test_full_scope_collects_every_benchmark(self):
    for fake_tensor_mode in (False, True):
      with self.subTest(fake_tensor_mode=fake_tensor_mode):
        self.assertEqual(
            self._case_benchmarks(context_lib.RunScope.FULL, fake_tensor_mode),
            set(_PRESUBMIT_REGISTRY),
        )

  def test_presubmit_scope_collects_runnable_benchmarks(self):
    self.assertEqual(
        self._case_benchmarks(context_lib.RunScope.PRESUBMIT, False), {"run"}
    )

  def test_presubmit_cpu_smoke_also_collects_smoke_only_benchmarks(self):
    self.assertEqual(
        self._case_benchmarks(context_lib.RunScope.PRESUBMIT, True),
        {"run", "smoke_only"},
    )

  def test_get_cases_logs_left_out_benchmarks(self):
    with self.assertLogs(level="INFO") as logs:
      self._case_benchmarks(context_lib.RunScope.PRESUBMIT, False)
    output = "\n".join(logs.output)
    self.assertIn("Left 2 benchmarks out", output)
    self.assertIn("CPU_SMOKE_ONLY (too big): smoke_only", output)
    self.assertIn("SKIP (never in presubmit): skipped", output)

  def test_get_cases_logs_nothing_in_full_scope(self):
    with self.assertNoLogs(level="INFO"):
      self._case_benchmarks(context_lib.RunScope.FULL, False)

  def test_presubmit_skip_reason(self):
    for name, fake_tensor_mode, expected in (
        ("run", False, None),
        ("run", True, None),
        ("smoke_only", True, None),
        (
            "smoke_only",
            False,
            (
                "Benchmark smoke_only runs in presubmit scope only in the"
                " FakeTensorMode CPU smoke (--fake_tensor_mode): too big"
            ),
        ),
        (
            "skipped",
            True,
            (
                "Benchmark skipped does not run in presubmit scope: never in"
                " presubmit"
            ),
        ),
    ):
      with self.subTest(name=name, fake_tensor_mode=fake_tensor_mode):
        self.assertEqual(
            cases.get_presubmit_skip_reason(
                _PRESUBMIT_REGISTRY[name], fake_tensor_mode=fake_tensor_mode
            ),
            expected,
        )

  def test_get_skip_reason_applies_presubmit_policy_in_presubmit_scope(self):
    spec = _PRESUBMIT_REGISTRY["skipped"]
    for run_scope, expected in (
        (context_lib.RunScope.FULL, None),
        (
            context_lib.RunScope.PRESUBMIT,
            (
                "Benchmark skipped does not run in presubmit scope: never in"
                " presubmit"
            ),
        ),
    ):
      with self.subTest(run_scope=run_scope):
        with flagsaver.flagsaver(run_scope=run_scope):
          self.assertEqual(
              cases.get_skip_reason(
                  spec,
                  common.RunMode.COMPILED,
                  framework=mode_lib.Framework.TORCH,
                  device_kind=target_lib.DeviceKind.TPU,
              ),
              expected,
          )


class GetSkipReasonTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            known_failures,
            "KNOWN_FAILURES",
            (
                known_failures.KnownFailure(
                    benchmark="known_bad",
                    framework=mode_lib.Framework.TORCH,
                    run_modes=frozenset({common.RunMode.COMPILED}),
                    reason="compiled failure",
                ),
            ),
        )
    )

  def test_returns_none_for_case_to_run(self):
    self.assertIsNone(
        cases.get_skip_reason(
            _make_spec("known_bad"),
            common.RunMode.EAGER_DEFAULT,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
        )
    )

  def test_returns_known_failure_reason(self):
    self.assertEqual(
        cases.get_skip_reason(
            _make_spec("known_bad"),
            common.RunMode.COMPILED,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
        ),
        "Benchmark known_bad is known to fail: compiled failure",
    )

  def test_returns_skipped_run_mode_reason(self):
    self.assertEqual(
        cases.get_skip_reason(
            _make_spec("healthy", skipped_run_modes=frozenset({"compiled"})),
            common.RunMode.COMPILED,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
        ),
        "Benchmark healthy explicitly skips run mode compiled",
    )

  def test_get_cases_deterministic_across_shards_and_permuted_by_seed(self):
    fake_registry = {
        name: _make_spec(name)
        for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    }
    with mock.patch.dict(registry_lib.REGISTRY, fake_registry, clear=True):
      seed_a_first = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1001",
          )
      ]
      seed_a_second = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1001",
          )
      ]
      seed_b = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1002",
          )
      ]
      self.assertEqual(seed_a_first, seed_a_second)
      self.assertCountEqual(seed_a_first, seed_b)
      self.assertNotEqual(seed_a_first, seed_b)

  def test_get_cases_uses_env_seed_when_order_seed_omitted(self):
    fake_registry = {
        name: _make_spec(name)
        for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    }
    with (
        mock.patch.dict(registry_lib.REGISTRY, fake_registry, clear=True),
        mock.patch.dict(
            os.environ,  # PY_ENVIRON_OK=test
            {"TORCH_TPU_BENCHMARK_ORDER_SEED": "cl_1001"},
            clear=False,
        ),
    ):
      from_env = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
          )
      ]
      explicit = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1001",
          )
      ]
      self.assertEqual(from_env, explicit)

  def test_seed_to_int_and_normalize_randomize_ordering_seed(self):
    self.assertEqual(cases.seed_to_int(""), 0)
    self.assertEqual(cases.seed_to_int("0"), 0)
    self.assertEqual(cases.seed_to_int("996893289"), 996893289)
    self.assertGreaterEqual(cases.seed_to_int("²"), 1)
    self.assertGreaterEqual(cases.seed_to_int("38215948283"), 1)
    self.assertLessEqual(cases.seed_to_int("38215948283"), 4294967295)

    hex_sha = "adac8863c4f1d05b3f53a53aab84c8e77490766f"
    sha_seed = cases.seed_to_int(hex_sha)
    self.assertGreaterEqual(sha_seed, 1)
    self.assertLessEqual(sha_seed, 4294967295)
    self.assertEqual(cases.seed_to_int(hex_sha), sha_seed)

    with mock.patch.dict(
        os.environ,  # PY_ENVIRON_OK=test
        {"TEST_RANDOMIZE_ORDERING_SEED": hex_sha},
        clear=False,
    ):
      loader = cases.BenchmarkTestLoader()
      self.assertEqual(
          os.environ["TEST_RANDOMIZE_ORDERING_SEED"],  # PY_ENVIRON_OK=test
          str(sha_seed),
      )
      self.assertEqual(loader._randomize_ordering_seed, sha_seed)

    flags.FLAGS.test_randomize_ordering_seed = hex_sha
    flags.FLAGS["test_randomize_ordering_seed"].present = True
    try:
      with mock.patch.dict(
          os.environ,  # PY_ENVIRON_OK=test
          {
              "TEST_RANDOMIZE_ORDERING_SEED": "",
              "TORCH_TPU_BENCHMARK_ORDER_SEED": "",
          },
          clear=False,
      ):
        loader = cases.BenchmarkTestLoader()
        self.assertEqual(loader._randomize_ordering_seed, sha_seed)
        self.assertEqual(
            os.environ["TEST_RANDOMIZE_ORDERING_SEED"],  # PY_ENVIRON_OK=test
            str(sha_seed),
        )
        self.assertEqual(
            flags.FLAGS.test_randomize_ordering_seed, str(sha_seed)
        )
    finally:
      flags.FLAGS.test_randomize_ordering_seed = ""
      flags.FLAGS["test_randomize_ordering_seed"].present = False

  def _shard_assignment(
      self,
      case_names: Sequence[str],
      total_shards: int,
      env: Mapping[str, str],
  ) -> dict[str, int]:
    """Returns the shard each case of a fake benchmark class runs in."""
    params = cases.case_params(
        (name, _make_spec(name), common.RunMode.COMPILED) for name in case_names
    )
    test_cls = type(
        "FakeBenchmarkTest",
        (seed_test_utils.RepeatableTest,),
        {
            "framework": mode_lib.Framework.TORCH,
            "platform": target_lib.Platform.V7_1X1X1,
            "case_params": params,
            **{name: lambda self: None for name in params},
        },
    )
    assignment = {}
    with mock.patch.dict(
        os.environ,  # PY_ENVIRON_OK=test
        env,
        clear=False,
    ):
      for shard_index in range(total_shards):
        loader = cases.BenchmarkTestLoader()
        names = loader.getTestCaseNames(test_cls)
        for name in loader.shardTestCaseNames(
            itertools.cycle(range(total_shards)), names, shard_index
        ):
          self.assertNotIn(name, assignment)
          assignment[name] = shard_index
    self.assertCountEqual(assignment, params)
    return assignment

  def test_shard_assignment_does_not_depend_on_ordering_seeds(self):
    # CI seeds the ordering with the commit SHA, and auto-bisect reruns one
    # shard at each candidate commit, so a case must stay in the same shard
    # whatever the seeds are.
    case_names = [f"model_{i}_compiled" for i in range(40)]
    first = self._shard_assignment(
        case_names,
        total_shards=4,
        env={
            "TEST_RANDOMIZE_ORDERING_SEED": "12345",
            "TORCH_TPU_BENCHMARK_ORDER_SEED": (
                "adac8863c4f1d05b3f53a53aab84c8e77490766f"
            ),
        },
    )
    second = self._shard_assignment(
        case_names,
        total_shards=4,
        env={
            "TEST_RANDOMIZE_ORDERING_SEED": "67890",
            "TORCH_TPU_BENCHMARK_ORDER_SEED": (
                "0123abcd4567ef890123abcd4567ef8901234567"
            ),
        },
    )
    self.assertEqual(first, second)
    self.assertLen(set(first.values()), 4)

  def test_adding_a_case_does_not_move_other_cases(self):
    env = {
        "TEST_RANDOMIZE_ORDERING_SEED": "0",
        "TORCH_TPU_BENCHMARK_ORDER_SEED": "0123abcd",
    }
    case_names = [f"model_{i}_compiled" for i in range(40)]
    before = self._shard_assignment(case_names, total_shards=4, env=env)
    after = self._shard_assignment(
        case_names + ["new_model_compiled"], total_shards=4, env=env
    )
    del after["test_benchmark_new_model_compiled"]
    self.assertEqual(before, after)

  def test_shard_assignment_does_not_depend_on_iterator_position(self):
    # absltest shares one cyclic shard iterator across all test classes, so a
    # later class can see it mid-cycle.
    names = [f"test_benchmark_model_{i}_compiled" for i in range(20)]
    loader = cases.BenchmarkTestLoader()
    total_shards = 4
    for shard_index in range(total_shards):
      expected = loader.shardTestCaseNames(
          itertools.cycle(range(total_shards)), names, shard_index
      )
      for offset in range(1, total_shards):
        iterator = itertools.cycle(range(total_shards))
        for _ in range(offset):
          next(iterator)
        self.assertEqual(
            loader.shardTestCaseNames(iterator, names, shard_index),
            expected,
            msg=f"shard {shard_index}, iterator offset {offset}",
        )

  def test_each_case_runs_in_exactly_one_shard_whatever_the_name_order(self):
    # Each shard is a separate process with its own loader, which can see the
    # test names in a different order (e.g. absltest shuffles them by
    # --test_randomize_ordering_seed). The shards must still partition the
    # cases.
    case_triples = [
        (f"model_{i}_{mode.value}", _make_spec(f"model_{i}"), mode)
        for i in range(7)
        for mode in (common.RunMode.EAGER_DEFAULT, common.RunMode.COMPILED)
    ]
    params = cases.case_params(case_triples)
    test_cls = type(
        "FakeBenchmarkTest",
        (seed_test_utils.RepeatableTest,),
        {
            "framework": mode_lib.Framework.TORCH,
            "platform": target_lib.Platform.V7_1X1X1,
            "case_params": params,
            **{name: lambda self: None for name in params},
        },
    )
    total_shards = 3

    assigned = []
    with mock.patch.dict(
        os.environ,  # PY_ENVIRON_OK=test
        {
            "TEST_RANDOMIZE_ORDERING_SEED": "0",
            "TORCH_TPU_BENCHMARK_ORDER_SEED": "0123abcd",
        },
        clear=False,
    ):
      for shard_index in range(total_shards):
        loader = cases.BenchmarkTestLoader()
        names = loader.getTestCaseNames(test_cls)
        random.Random(shard_index).shuffle(names)
        assigned.extend(
            loader.shardTestCaseNames(
                itertools.cycle(range(total_shards)), names, shard_index
            )
        )

    self.assertCountEqual(assigned, params)


# The default workload of the swept test benchmarks: a batch size that the
# sweeps below do not list, so that no sweep batch size is the default.
_SWEEP_DEFAULT_WORKLOAD = registry_lib.scoped_workload(
    full=workload_lib.Workload(16)
)


def _make_sweep_spec(
    name: str,
    stepper: step_lib.StepperType = step_lib.StepperType.FORWARD,
    batch_sweep: registry_lib.BatchSweep | None = None,
    workload: registry_lib.WorkloadFn | None = _SWEEP_DEFAULT_WORKLOAD,
) -> registry_lib.BenchmarkSpec:
  return registry_lib.BenchmarkSpec(
      name=name,
      factory=_factory,
      stepper=stepper,
      presubmit=_RUN,
      workload=workload,
      batch_sweep=batch_sweep or registry_lib.BatchSweep(),
  )


_SWEEP_REGISTRY = {
    "alpha": _make_sweep_spec("alpha"),
    "bravo": _make_sweep_spec(
        "bravo", batch_sweep=registry_lib.BatchSweep(batch_sizes=(8, 2))
    ),
    "charlie": _make_sweep_spec(
        "charlie", batch_sweep=registry_lib.BatchSweep.disabled("packed")
    ),
    "delta": _make_sweep_spec(
        "delta", stepper=step_lib.StepperType.DECODER_ONLY_DECODE
    ),
    "echo": _make_sweep_spec("echo", workload=None),
}


class BatchSweepCasesTest(seed_test_utils.RepeatableTest):

  def _sweep_cases(
      self,
      platform: target_lib.Platform = target_lib.Platform.B200_1,
      order_seed: str = "",
  ) -> list[tuple[str, registry_lib.BenchmarkSpec, common.RunMode, int]]:
    with mock.patch.dict(registry_lib.REGISTRY, _SWEEP_REGISTRY, clear=True):
      return list(
          cases.get_batch_sweep_cases(
              platform,
              mode_lib.Framework.TORCH,
              batch_sizes=(4, 1),
              order_seed=order_seed,
          )
      )

  def test_one_case_per_swept_benchmark_run_mode_and_batch_size(self):
    eager = common.RunMode.EAGER_DEFAULT
    compiled = common.RunMode.COMPILED
    self.assertCountEqual(
        [
            (name, spec.name, mode, b)
            for name, spec, mode, b in self._sweep_cases()
        ],
        [
            ("alpha_bs1_eager_default", "alpha", eager, 1),
            ("alpha_bs4_eager_default", "alpha", eager, 4),
            ("alpha_bs1_compiled", "alpha", compiled, 1),
            ("alpha_bs4_compiled", "alpha", compiled, 4),
            ("bravo_bs2_eager_default", "bravo", eager, 2),
            ("bravo_bs8_eager_default", "bravo", eager, 8),
            ("bravo_bs2_compiled", "bravo", compiled, 2),
            ("bravo_bs8_compiled", "bravo", compiled, 8),
        ],
    )

  def test_sweeps_the_run_modes_of_the_platform(self):
    for platform in (
        target_lib.Platform.CPU,
        target_lib.Platform.B200_1,
        target_lib.Platform.V7_1X1X1,
    ):
      with self.subTest(platform=platform):
        kind = target_lib.make_target(platform).device_kind
        self.assertCountEqual(
            {mode for _, _, mode, _ in self._sweep_cases(platform)},
            mode_lib.modes_for(mode_lib.Framework.TORCH, kind),
        )

  def test_cases_of_a_benchmark_and_run_mode_are_consecutive_and_ascending(
      self,
  ):
    for seed in ("", "cl_1001", "cl_1002"):
      with self.subTest(seed=seed):
        sweep_cases = self._sweep_cases(order_seed=seed)
        groups = [(spec.name, mode) for _, spec, mode, _ in sweep_cases]
        batch_sizes = [b for *_, b in sweep_cases]
        for i in range(0, len(sweep_cases), 2):
          self.assertEqual(groups[i], groups[i + 1])
          self.assertLess(batch_sizes[i], batch_sizes[i + 1])
        for name, spec, mode, batch_size in sweep_cases:
          self.assertEqual(
              name, cases.batch_sweep_case_name(spec.name, batch_size, mode)
          )

  def test_case_names_end_in_the_run_mode(self):
    self.assertEqual(
        cases.batch_sweep_case_name(
            "gemma_2_2b_forward", 16, common.RunMode.EAGER_OPTIMIZED
        ),
        "gemma_2_2b_forward_bs16_eager_optimized",
    )

  def test_default_batch_size_is_not_a_sweep_case(self):
    registry = {
        "alpha": _make_sweep_spec(
            "alpha",
            workload=registry_lib.scoped_workload(
                full=workload_lib.Workload(4, 128),
                presubmit=workload_lib.Workload(1, 128),
            ),
        )
    }
    with mock.patch.dict(registry_lib.REGISTRY, registry, clear=True):
      sweep_cases = list(
          cases.get_batch_sweep_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              batch_sizes=(1, 2, 4, 8),
          )
      )
    # The default case (batch size 4) is one of `get_cases`.
    self.assertEqual([b for *_, b in sweep_cases], [1, 2, 8])

  def test_batch_sweep_skip_reason(self):
    self.assertIsNone(
        cases.get_batch_sweep_skip_reason(_SWEEP_REGISTRY["alpha"])
    )
    self.assertEqual(
        cases.get_batch_sweep_skip_reason(_SWEEP_REGISTRY["echo"]),
        "Benchmark echo is not batch swept: it declares no workload",
    )
    self.assertEqual(
        cases.get_batch_sweep_skip_reason(_SWEEP_REGISTRY["charlie"]),
        "Benchmark charlie is not batch swept: packed",
    )
    self.assertIn(
        "decode is not swept yet",
        cases.get_batch_sweep_skip_reason(_SWEEP_REGISTRY["delta"]),
    )

  def test_case_params_keep_the_batch_size(self):
    params = cases.case_params([
        ("alpha_compiled", _SWEEP_REGISTRY["alpha"], common.RunMode.COMPILED),
        (
            "alpha_bs4_compiled",
            _SWEEP_REGISTRY["alpha"],
            common.RunMode.COMPILED,
            4,
        ),
    ])
    self.assertEqual(
        cases.unpack_case_params(params["test_benchmark_alpha_compiled"]),
        (_SWEEP_REGISTRY["alpha"], common.RunMode.COMPILED, None),
    )
    self.assertEqual(
        cases.unpack_case_params(params["test_benchmark_alpha_bs4_compiled"]),
        (_SWEEP_REGISTRY["alpha"], common.RunMode.COMPILED, 4),
    )

  def test_order_batch_sweep_cases_keeps_positions(self):
    a, b = _SWEEP_REGISTRY["alpha"], _SWEEP_REGISTRY["bravo"]
    compiled, eager = common.RunMode.COMPILED, common.RunMode.EAGER_DEFAULT
    params_by_name = {
        "b16c": (b, compiled, 16),
        "x": (a, compiled),
        "a8c": (a, compiled, 8),
        "b2c": (b, compiled, 2),
        "a1e": (a, eager, 1),
        "a1c": (a, compiled, 1),
        "a4e": (a, eager, 4),
    }
    self.assertEqual(
        cases.order_batch_sweep_cases(
            ["b16c", "x", "a4e", "a8c", "b2c", "a1e", "a1c", "unknown"],
            params_by_name,
        ),
        ["b2c", "x", "a1e", "a1c", "b16c", "a4e", "a8c", "unknown"],
    )

  def test_batch_sweep_cases_of_a_benchmark_and_run_mode_share_a_shard(self):
    specs = {name: _make_sweep_spec(name) for name in "abcdefgh"}
    modes = (common.RunMode.EAGER_DEFAULT, common.RunMode.COMPILED)

    class _SweepTest:  # pylint: disable=too-few-public-methods
      case_params = {
          f"test_benchmark_{name}_bs{b}_{mode.value}": (spec, mode, b)
          for name, spec in specs.items()
          for mode in modes
          for b in (1, 2, 4, 8, 16, 32)
      }

    loader = cases.BenchmarkTestLoader()
    names = list(_SweepTest.case_params)
    with mock.patch.object(
        absltest.TestLoader, "getTestCaseNames", return_value=names
    ):
      ordered = loader.getTestCaseNames(_SweepTest)
    total_shards = 4
    shard_of = {}
    for shard_index in range(total_shards):
      for name in loader.shardTestCaseNames(
          itertools.cycle(range(total_shards)), ordered, shard_index
      ):
        shard_of[name] = shard_index
    self.assertCountEqual(shard_of, names)
    for name, spec in specs.items():
      for mode in modes:
        with self.subTest(spec=name, mode=mode):
          group = [
              n
              for n, (s, m, _) in _SweepTest.case_params.items()
              if s is spec and m is mode
          ]
          self.assertLen({shard_of[n] for n in group}, 1)

  def test_default_cases_keep_their_shards(self):
    # Default cases hash their own name even if it looks like a batch sweep
    # name, so batch sweeps never move an existing case to another shard.
    names = [f"test_benchmark_model_bs{b}_compiled" for b in (1, 2, 4, 8)]

    class _DefaultTest:  # pylint: disable=too-few-public-methods
      case_params = {
          name: (_SWEEP_REGISTRY["alpha"], common.RunMode.COMPILED)
          for name in names
      }

    loader = cases.BenchmarkTestLoader()
    with mock.patch.object(
        absltest.TestLoader, "getTestCaseNames", return_value=names
    ):
      ordered = loader.getTestCaseNames(_DefaultTest)
    for shard_index in range(4):
      self.assertEqual(
          list(
              loader.shardTestCaseNames(
                  itertools.cycle(range(4)), ordered, shard_index
              )
          ),
          [
              name
              for name in ordered
              if cases._stable_shard_hash(name) % 4 == shard_index  # pylint: disable=protected-access
          ],
      )


if __name__ == "__main__":
  absltest.main()
