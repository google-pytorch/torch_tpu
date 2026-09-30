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

"""Tests that TORCH_WARN and TORCH_WARN_ONCE from C++ appear as TorchTpuWarning."""

import warnings

from absl.testing import absltest
from torch_tpu._internal import exceptions
from torch_tpu._internal import testing as tt_testing
from tests import seed_test_utils


class CppWarningsTest(seed_test_utils.RepeatableTest):

  def test_emit_tt_warn_generic(self) -> None:
    """Verifies TT_WARN with kTorchTpuWarning becomes TorchTpuWarning."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      tt_testing.emit_tt_warn(
          tt_testing.WarningCategory.TorchTpuWarning,
          "A test C++ warning message from TorchTPU",
      )

    self.assertNotEmpty(recorded)
    matching = [
        w
        for w in recorded
        if "A test C++ warning message from TorchTPU" in str(w.message)
    ]
    self.assertLen(matching, 1)
    w = matching[0]
    self.assertTrue(
        issubclass(w.category, exceptions.TorchTpuWarning),
        f"Expected category to subclass TorchTpuWarning, got {w.category}",
    )
    self.assertIsInstance(w.message, exceptions.TorchTpuWarning)
    self.assertNotIn("\x1e", str(w.message))
    self.assertNotIn("TT_WARN", str(w.message))

  def test_token_must_be_prefix(self) -> None:
    """Verifies that the token is only recognized as a prefix, not when embedded mid-message."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      # Emit a plain TORCH_WARN with the token placed in the middle of text
      tt_testing.emit_torch_warn(
          "Some prefix \x1e[TT_WARN:ExperimentalApiWarning]\x1e mid message"
      )

    matching = [
        w
        for w in recorded
        if "Some prefix \x1e[TT_WARN:ExperimentalApiWarning]\x1e"
        in str(w.message)
    ]
    self.assertLen(matching, 1)
    w = matching[0]
    # Since the token is not at the start (regex match fails), it remains a standard UserWarning
    self.assertEqual(w.category, UserWarning)
    self.assertFalse(issubclass(w.category, exceptions.TorchTpuWarning))

  def test_plain_torch_warn_not_converted(self) -> None:
    """Verifies that plain untagged TORCH_WARN from C++ is not converted without token."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      tt_testing.emit_torch_warn("Untagged C++ warning")

    matching = [w for w in recorded if "Untagged C++ warning" in str(w.message)]
    self.assertLen(matching, 1)
    w = matching[0]
    self.assertEqual(w.category, UserWarning)
    self.assertFalse(issubclass(w.category, exceptions.TorchTpuWarning))

  def test_filterwarnings_ignore_torch_tpu_warning(self) -> None:
    """Verifies warnings.filterwarnings can silence C++ TorchTpuWarnings."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.filterwarnings("ignore", category=exceptions.TorchTpuWarning)
      tt_testing.emit_tt_warn(
          tt_testing.WarningCategory.TorchTpuWarning,
          "Should be ignored by TorchTpuWarning filter",
      )

    matching = [
        w
        for w in recorded
        if "Should be ignored by TorchTpuWarning filter" in str(w.message)
    ]
    self.assertEmpty(matching)

  def test_filterwarnings_ignore_subclass(self) -> None:
    """Verifies filtering by specific subclass works."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.filterwarnings(
          "ignore", category=exceptions.ExperimentalApiWarning
      )
      tt_testing.emit_tt_warn(
          tt_testing.WarningCategory.ExperimentalApiWarning,
          "Feature xyz is experimental and will change",
      )

    matching = [
        w for w in recorded if "Feature xyz is experimental" in str(w.message)
    ]
    self.assertEmpty(matching)

  def test_filterwarnings_once(self) -> None:
    """Verifies that 'once' warning action deduplicates C++ warnings."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.filterwarnings("once", category=exceptions.TorchTpuWarning)
      # Emit the same warning twice
      tt_testing.emit_tt_warn(
          tt_testing.WarningCategory.TorchTpuWarning,
          "Deduplicated warning message",
      )
      tt_testing.emit_tt_warn(
          tt_testing.WarningCategory.TorchTpuWarning,
          "Deduplicated warning message",
      )

    matching = [
        w for w in recorded if "Deduplicated warning message" in str(w.message)
    ]
    self.assertLen(matching, 1)

  def test_non_torch_tpu_warnings_unaffected(self) -> None:
    """Verifies that non-TorchTPU warnings remain standard UserWarning."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      warnings.warn("A regular python warning", UserWarning)

    matching = [
        w for w in recorded if "A regular python warning" in str(w.message)
    ]
    self.assertLen(matching, 1)
    w = matching[0]
    self.assertFalse(issubclass(w.category, exceptions.TorchTpuWarning))
    self.assertEqual(w.category, UserWarning)

  def test_emit_tt_warn_experimental(self) -> None:
    """Verifies TT_WARN with ExperimentalApiWarning produces ExperimentalApiWarning."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      tt_testing.emit_tt_warn(
          tt_testing.WarningCategory.ExperimentalApiWarning,
          "Typed experimental warning",
      )

    matching = [
        w for w in recorded if "Typed experimental warning" in str(w.message)
    ]
    self.assertLen(matching, 1)
    w = matching[0]
    self.assertTrue(issubclass(w.category, exceptions.ExperimentalApiWarning))
    self.assertIsInstance(w.message, exceptions.ExperimentalApiWarning)
    # Verify token is stripped from the message
    self.assertNotIn("\x1e", str(w.message))
    self.assertNotIn("TT_WARN", str(w.message))

  def test_emit_tt_warn_once_experimental(self) -> None:
    """Verifies TT_WARN_ONCE fires once and produces ExperimentalApiWarning."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      tt_testing.emit_tt_warn_once(
          tt_testing.WarningCategory.ExperimentalApiWarning,
          "Typed once experimental warning",
      )
      tt_testing.emit_tt_warn_once(
          tt_testing.WarningCategory.ExperimentalApiWarning,
          "Typed once experimental warning",
      )

    matching = [
        w
        for w in recorded
        if "Typed once experimental warning" in str(w.message)
    ]
    self.assertLen(matching, 1)
    w = matching[0]
    self.assertTrue(issubclass(w.category, exceptions.ExperimentalApiWarning))
    self.assertIsInstance(w.message, exceptions.ExperimentalApiWarning)
    self.assertNotIn("\x1e", str(w.message))

  def test_emit_tt_warn_deprecated(self) -> None:
    """Verifies TT_WARN with DeprecatedApiWarning produces DeprecatedApiWarning."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      tt_testing.emit_tt_warn(
          tt_testing.WarningCategory.DeprecatedApiWarning,
          "Typed deprecated warning",
      )

    matching = [
        w for w in recorded if "Typed deprecated warning" in str(w.message)
    ]
    self.assertLen(matching, 1)
    w = matching[0]
    self.assertTrue(issubclass(w.category, exceptions.DeprecatedApiWarning))
    self.assertTrue(issubclass(w.category, DeprecationWarning))
    self.assertIsInstance(w.message, exceptions.DeprecatedApiWarning)
    self.assertNotIn("\x1e", str(w.message))

  def test_emit_tt_warn_performance(self) -> None:
    """Verifies TT_WARN with PerformanceWarning produces PerformanceWarning."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      tt_testing.emit_tt_warn(
          tt_testing.WarningCategory.PerformanceWarning,
          "Typed performance warning",
      )

    matching = [
        w for w in recorded if "Typed performance warning" in str(w.message)
    ]
    self.assertLen(matching, 1)
    w = matching[0]
    self.assertTrue(issubclass(w.category, exceptions.PerformanceWarning))
    self.assertIsInstance(w.message, exceptions.PerformanceWarning)
    self.assertNotIn("\x1e", str(w.message))

  def test_emit_tt_warn_correctness(self) -> None:
    """Verifies TT_WARN with CorrectnessWarning produces CorrectnessWarning."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      tt_testing.emit_tt_warn(
          tt_testing.WarningCategory.CorrectnessWarning,
          "Typed correctness warning",
      )

    matching = [
        w for w in recorded if "Typed correctness warning" in str(w.message)
    ]
    self.assertLen(matching, 1)
    w = matching[0]
    self.assertTrue(issubclass(w.category, exceptions.CorrectnessWarning))
    self.assertIsInstance(w.message, exceptions.CorrectnessWarning)
    self.assertNotIn("\x1e", str(w.message))


if __name__ == "__main__":
  absltest.main()
