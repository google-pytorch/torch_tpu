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

"""Tests for the findings data model and errors raised by `torch.tpu.pallas.verifier`."""

from __future__ import annotations

import pickle

from absl.testing import absltest
import torch
from torch.tpu.pallas import verifier
from tests import seed_test_utils


class PallasVerifyErrorsTest(seed_test_utils.RepeatableTest):

  def test_finding_format_includes_severity_category_op_and_location(self):
    finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=verifier.Severity.ERROR,
        phase=None,
        message="unsupported construct",
        op_name="my_kernel",
        source_location="kernel.py:12",
    )

    formatted = finding.format()
    self.assertIn("[ERROR] verification_incomplete in my_kernel", formatted)
    self.assertIn("kernel.py:12", formatted)
    self.assertIn("unsupported construct", formatted)

  def test_report_with_only_warnings_and_infos_passes(self):
    warn_finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=verifier.Severity.WARNING,
        phase=None,
        message="bounded loop unrolling",
        op_name="my_kernel",
    )
    info_finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=verifier.Severity.INFO,
        phase=None,
        message="skipped phase",
        op_name="my_kernel",
    )
    report = verifier.VerificationReport(findings=(warn_finding, info_finding))

    self.assertTrue(report.passed)
    self.assertEmpty(report.findings_by_severity(verifier.Severity.ERROR))
    self.assertEqual(
        report.findings_by_severity(verifier.Severity.WARNING),
        (warn_finding,),
    )
    self.assertEqual(
        report.findings_by_severity(verifier.Severity.INFO),
        (info_finding,),
    )
    self.assertIn("1 warning(s), 1 info(s)", report.summary())
    report.raise_for_errors()

  def test_report_with_error_fails_and_filters_by_category(self):
    err_finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=verifier.Severity.ERROR,
        phase=None,
        message="unsupported construct",
        op_name="my_kernel",
    )
    warn_finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=verifier.Severity.WARNING,
        phase=None,
        message="bounded loop unrolling",
        op_name="my_kernel",
    )
    report = verifier.VerificationReport(findings=(warn_finding, err_finding))

    self.assertFalse(report.passed)
    self.assertEqual(
        report.findings_by_severity(verifier.Severity.ERROR),
        (err_finding,),
    )
    self.assertEqual(report.checked_phases, ())
    self.assertEqual(
        report.findings_by_category(
            verifier.FindingCategory.VERIFICATION_INCOMPLETE
        ),
        (warn_finding, err_finding),
    )
    self.assertIn("FAILED", str(report))

  def test_raise_for_errors_attaches_report_and_pickles(self):
    err_finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=verifier.Severity.ERROR,
        phase=None,
        message="unsupported construct",
        op_name="my_kernel",
    )
    report = verifier.VerificationReport(findings=(err_finding,))

    with self.assertRaises(verifier.VerificationError) as ctx:
      report.raise_for_errors()
    self.assertIs(ctx.exception.report, report)
    roundtripped = pickle.loads(pickle.dumps(ctx.exception))
    self.assertEqual(roundtripped.report, report)
    self.assertEqual(str(roundtripped), str(ctx.exception))

  def test_verification_options_defaults_and_validation(self):
    opts = verifier.VerificationOptions()
    self.assertIsInstance(opts, verifier.VerificationOptions)
    with self.assertRaisesRegex(TypeError, "mode must be"):
      verifier.verify(lambda x: x, (torch.ones(2),), mode="static")  # type: ignore[arg-type]
    with self.assertRaisesRegex(TypeError, "options must be"):
      verifier.verify(lambda x: x, (torch.ones(2),), options={})  # type: ignore[arg-type]

  def test_verify_accepts_callable_and_returns_passing_report(self):
    report = verifier.verify(lambda x: x + 1, torch.ones((4, 4)))
    self.assertTrue(report.passed)

  def test_verify_rejects_non_callable_target(self):
    with self.assertRaisesRegex(TypeError, "target must be callable"):
      verifier.verify(42, (torch.ones(2),))  # type: ignore[arg-type]

  def test_verify_rejects_non_sequence_args(self):
    with self.assertRaisesRegex(
        TypeError, "args must be a sequence of example arguments"
    ):
      verifier.verify(lambda x: x, 123)  # type: ignore[arg-type]
    with self.assertRaisesRegex(
        TypeError, "args must be a sequence of example arguments"
    ):
      verifier.verify(lambda x: x, "not_a_tensor_tuple")  # type: ignore[arg-type]

  def test_verify_raises_in_torch_compile_region(self):
    torch._dynamo.reset()

    @torch.compile(backend="eager")
    def call_verify_in_compiled_region(x: torch.Tensor) -> torch.Tensor:
      verifier.verify(
          lambda t: t + 1.0,
          (x,),
          mode=verifier.VerificationMode.STATIC,
      )
      return x + 1.0

    with self.assertRaisesRegex(
        AssertionError, r"Attempt to trace forbidden callable"
    ):
      call_verify_in_compiled_region(torch.ones((8, 128)))


if __name__ == "__main__":
  absltest.main()
