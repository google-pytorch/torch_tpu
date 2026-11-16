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

"""Tests for perflab/flags.py."""

from collections.abc import Sequence
import copy
import json
import os
import sys
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from perflab import common
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import target as target_lib
from tests import seed_test_utils


class FakeTensorModeValidatorTest(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      context_lib.RunScope.PRESUBMIT, context_lib.RunScope.FULL
  )
  def test_accepted_on_cpu(self, run_scope: context_lib.RunScope):
    with (
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"}),
        flagsaver.flagsaver(
            (context_lib.RUN_SCOPE, run_scope),
            (flags_lib.FAKE_TENSOR_MODE, True),
        ),
    ):
      self.assertTrue(flags_lib.FAKE_TENSOR_MODE.value)

  @parameterized.parameters("v5e_1x1", "b200_1", "v7_1x1x1")
  def test_rejected_outside_cpu(self, platform: str):
    with mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: platform}):
      with self.assertRaisesRegex(
          flags.IllegalFlagValueError, "fake_tensor_mode is only valid"
      ):
        with flagsaver.flagsaver((flags_lib.FAKE_TENSOR_MODE, True)):
          pass


class CollectionFlagValuesTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # Collecting cases records the command line in the environment.
    self.enter_context(mock.patch.dict(os.environ))
    os.environ.pop(flags_lib.COLLECTION_ARGV_ENV_VAR, None)

  def _values_before_parsing(
      self, argv: Sequence[str]
  ) -> tuple[context_lib.RunScope, bool]:
    # Unparsed flags, as when a benchmark binary collects its cases at import.
    import_time_flags = flags.FlagValues()
    for flag in (context_lib.RUN_SCOPE, flags_lib.FAKE_TENSOR_MODE):
      import_time_flags[flag.name] = copy.deepcopy(flags.FLAGS[flag.name])
    with (
        mock.patch.object(flags, "FLAGS", import_time_flags),
        mock.patch.object(sys, "argv", ["prog", *argv]),
    ):
      values = flags_lib.collection_flag_values()
      self.assertFalse(import_time_flags.is_parsed())
      return values

  def test_reads_parsed_flags(self):
    with flagsaver.flagsaver(
        (context_lib.RUN_SCOPE, context_lib.RunScope.PRESUBMIT),
        (flags_lib.FAKE_TENSOR_MODE, False),
    ):
      self.assertEqual(
          flags_lib.collection_flag_values(),
          (context_lib.RunScope.PRESUBMIT, False),
      )

  def test_defaults_before_parsing(self):
    self.assertEqual(
        self._values_before_parsing([]), (context_lib.RunScope.FULL, False)
    )

  def test_reads_argv_before_parsing(self):
    self.assertEqual(
        self._values_before_parsing(
            ["--run_scope=presubmit", "--fake_tensor_mode"]
        ),
        (context_lib.RunScope.PRESUBMIT, True),
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "after_double_dash",
          "argv": ["--run_scope=presubmit", "--", "--fake_tensor_mode"],
      },
      {
          "testcase_name": "negated",
          "argv": [
              "--run_scope",
              "presubmit",
              "--fake_tensor_mode",
              "--nofake_tensor_mode",
          ],
      },
      {
          "testcase_name": "false",
          "argv": ["--run_scope=presubmit", "--fake_tensor_mode=false"],
      },
  )
  def test_parses_argv_like_absl_before_parsing(self, argv: Sequence[str]):
    self.assertEqual(
        self._values_before_parsing(argv),
        (context_lib.RunScope.PRESUBMIT, False),
    )


class BatchSweepFlagsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # Collecting cases records the command line in the environment.
    self.enter_context(mock.patch.dict(os.environ))
    os.environ.pop(flags_lib.COLLECTION_ARGV_ENV_VAR, None)

  def _values_before_parsing(
      self, argv: list[str]
  ) -> flags_lib.BatchSweepFlagValues:
    import_time_flags = flags.FlagValues()
    for flag in (
        flags_lib.BATCH_SWEEP,
        flags_lib.BATCH_SWEEP_SIZES,
        flags_lib.BATCH_SWEEP_RUN_MODES,
        flags_lib.BATCH_SWEEP_ONLY,
    ):
      import_time_flags[flag.name] = copy.deepcopy(flags.FLAGS[flag.name])
    with (
        mock.patch.object(flags, "FLAGS", import_time_flags),
        mock.patch.object(sys, "argv", ["prog", *argv]),
    ):
      return flags_lib.batch_sweep_flag_values()

  def _enabled_and_sizes(self, argv: list[str]) -> tuple[bool, tuple[int, ...]]:
    values = self._values_before_parsing(argv)
    return values.enabled, values.batch_sizes

  def _record_argv(self, pid: int, argv: list[str]) -> None:
    os.environ[flags_lib.COLLECTION_ARGV_ENV_VAR] = json.dumps(
        {"pid": pid, "argv": ["prog", *argv]}
    )

  def test_records_argv_for_child_processes(self):
    self._values_before_parsing(["--batch_sweep"])
    self.assertEqual(
        json.loads(os.environ[flags_lib.COLLECTION_ARGV_ENV_VAR]),
        {"pid": os.getpid(), "argv": ["prog", "--batch_sweep"]},
    )

  def test_child_process_reads_recorded_argv(self):
    # As in the forkserver of a benchmark binary run with --batch_sweep, whose
    # own command line lacks the flag.
    self._record_argv(
        os.getppid(),
        ["--batch_sweep", "--batch_sweep_sizes=8", "--batch_sweep_only"],
    )
    values = self._values_before_parsing([])
    self.assertEqual((values.enabled, values.batch_sizes), (True, (8,)))
    self.assertTrue(values.only)

  def test_ignores_argv_recorded_by_another_process(self):
    self._record_argv(os.getpid(), ["--batch_sweep"])
    self.assertEqual(
        self._enabled_and_sizes(["--batch_sweep_sizes=2"]), (False, (2,))
    )

  def test_defaults_before_parsing(self):
    self.assertEqual(
        self._values_before_parsing([]),
        flags_lib.BatchSweepFlagValues(
            enabled=False,
            batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
            run_modes=frozenset({common.RunMode.COMPILED}),
            only=False,
        ),
    )

  def test_reads_argv_before_parsing(self):
    self.assertEqual(
        self._values_before_parsing([
            "--batch_sweep",
            "--batch_sweep_sizes=16,4,4",
            "--batch_sweep_run_modes=eager_default,compiled",
            "--batch_sweep_only",
        ]),
        flags_lib.BatchSweepFlagValues(
            enabled=True,
            batch_sizes=(4, 16),
            run_modes=frozenset(
                {common.RunMode.EAGER_DEFAULT, common.RunMode.COMPILED}
            ),
            only=True,
        ),
    )

  def test_reads_parsed_flags(self):
    with flagsaver.flagsaver(
        (flags_lib.BATCH_SWEEP, True),
        (flags_lib.BATCH_SWEEP_SIZES, ["2"]),
        (flags_lib.BATCH_SWEEP_RUN_MODES, ["eager_optimized"]),
        (flags_lib.BATCH_SWEEP_ONLY, True),
    ):
      self.assertEqual(
          flags_lib.batch_sweep_flag_values(),
          flags_lib.BatchSweepFlagValues(
              enabled=True,
              batch_sizes=(2,),
              run_modes=frozenset({common.RunMode.EAGER_OPTIMIZED}),
              only=True,
          ),
      )

  def test_batch_sweep_requires_full_scope(self):
    with flagsaver.flagsaver(
        (context_lib.RUN_SCOPE, context_lib.RunScope.FULL),
        (flags_lib.BATCH_SWEEP, True),
    ):
      self.assertTrue(flags_lib.BATCH_SWEEP.value)
    with self.assertRaisesRegex(
        flags.IllegalFlagValueError, "batch_sweep is only valid"
    ):
      with flagsaver.flagsaver(
          (context_lib.RUN_SCOPE, context_lib.RunScope.PRESUBMIT),
          (flags_lib.BATCH_SWEEP, True),
      ):
        pass

  @parameterized.parameters(["0"], ["-4", "8"], ["x"])
  def test_batch_sweep_sizes_must_be_positive_integers(self, *values: str):
    with self.assertRaises(flags.IllegalFlagValueError):
      with flagsaver.flagsaver((flags_lib.BATCH_SWEEP_SIZES, list(values))):
        pass

  @parameterized.parameters([], ["compiled", "eager"], ["COMPILED"])
  def test_batch_sweep_run_modes_must_be_run_modes(self, *values: str):
    with self.assertRaisesRegex(
        flags.IllegalFlagValueError, "batch_sweep_run_modes"
    ):
      with flagsaver.flagsaver((flags_lib.BATCH_SWEEP_RUN_MODES, list(values))):
        pass

  def test_batch_sweep_only_requires_batch_sweep(self):
    with flagsaver.flagsaver(
        (flags_lib.BATCH_SWEEP, True), (flags_lib.BATCH_SWEEP_ONLY, True)
    ):
      self.assertTrue(flags_lib.BATCH_SWEEP_ONLY.value)
    with self.assertRaisesRegex(
        flags.IllegalFlagValueError, "batch_sweep_only is only valid with"
    ):
      with flagsaver.flagsaver(
          (flags_lib.BATCH_SWEEP, False), (flags_lib.BATCH_SWEEP_ONLY, True)
      ):
        pass


if __name__ == "__main__":
  absltest.main()
