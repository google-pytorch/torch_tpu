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
import jax
from jax.experimental import pallas as pl
import torch
from torch.tpu import pallas
from torch.tpu.pallas import verifier
from torch_tpu._internal.pallas.verify import findings
from tests import seed_test_utils

jax_op = pallas.jax_op


def _pallas_add_one(x: jax.Array) -> jax.Array:
  return pl.pallas_call(
      lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] + 1.0),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x)


class PallasVerifyErrorsTest(seed_test_utils.RepeatableTest):

  def test_finding_format_includes_severity_category_op_and_location(self):
    step = verifier.CounterexampleStep(
        timestamp=0,
        device_id=0,
        core_id=0,
        actor="VPU",
        operation="store",
        detail="out[0:8]",
        source_location="kernel.py:12",
    )
    finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.DATA_RACE,
        severity=verifier.Severity.ERROR,
        phase=verifier.Phase.MODEL_CHECK,
        message="WAW race on out[0:8]",
        op_name="my_kernel",
        source_location="kernel.py:12",
        counterexample=(step,),
    )

    formatted = finding.format()
    self.assertIn("[ERROR] (model_check) data_race in my_kernel", formatted)
    self.assertIn("Counterexample schedule:", formatted)
    self.assertIn("kernel.py:12", formatted)

  def test_report_with_only_warnings_and_infos_passes(self):
    warn_finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.LOW_PRECISION_ACCUMULATION,
        severity=verifier.Severity.WARNING,
        phase=verifier.Phase.MODEL_CHECK,
        message="bfloat16 matmul without float32 accumulation",
        op_name="my_kernel",
    )
    info_finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=verifier.Severity.INFO,
        phase=verifier.Phase.MODEL_CHECK,
        message="bounded loop unrolling",
        op_name="my_kernel",
    )
    report = findings.report_of(
        verifier.Phase.MODEL_CHECK, (warn_finding, info_finding)
    )

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
        category=verifier.FindingCategory.DATA_RACE,
        severity=verifier.Severity.ERROR,
        phase=verifier.Phase.MODEL_CHECK,
        message="WAW race on out[0:8]",
        op_name="my_kernel",
    )
    warn_finding = verifier.VerificationFinding(
        category=verifier.FindingCategory.LOW_PRECISION_ACCUMULATION,
        severity=verifier.Severity.WARNING,
        phase=verifier.Phase.CONTRACTS,
        message="bfloat16 matmul without float32 accumulation",
        op_name="my_kernel",
    )
    warn_report = findings.report_of(verifier.Phase.CONTRACTS, (warn_finding,))
    report = findings.merge_reports(
        warn_report,
        findings.report_of(verifier.Phase.MODEL_CHECK, (err_finding,)),
        findings.report_of(verifier.Phase.CONTRACTS, ()),
    )

    self.assertFalse(report.passed)
    self.assertEqual(
        report.findings_by_severity(verifier.Severity.ERROR),
        (err_finding,),
    )
    self.assertEqual(
        report.checked_phases,
        (verifier.Phase.CONTRACTS, verifier.Phase.MODEL_CHECK),
    )
    self.assertEqual(
        report.findings_by_category(verifier.FindingCategory.DATA_RACE),
        (err_finding,),
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
    self.assertTrue(opts.check_autograd)
    self.assertEqual(opts.max_unrolled_iterations, 32)
    self.assertEqual(opts.solver_timeout_seconds, 5.0)
    self.assertEqual(opts.tpu_device_kind, "TPU v7")
    self.assertEqual(
        verifier.VerificationOptions(
            tpu_device_kind="TPU v5 lite"
        ).tpu_device_kind,
        "TPU v5e",
    )
    self.assertEqual(
        verifier.VerificationOptions(tpu_device_kind="TPU7x").tpu_device_kind,
        "TPU v7",
    )
    self.assertEqual(
        verifier.VerificationOptions(tpu_device_kind="TPU v7x").tpu_device_kind,
        "TPU v7",
    )
    self.assertEqual(
        verifier.VerificationOptions(tpu_device_kind="TPU v6").tpu_device_kind,
        "TPU v6e",
    )
    self.assertEqual(
        verifier.VerificationOptions(
            tpu_device_kind="Trillium"
        ).tpu_device_kind,
        "TPU v6e",
    )
    self.assertEqual(
        findings.normalize_tpu_device_kind(" custom_tpu "), "custom_tpu"
    )
    self.assertEqual(findings.jax_tpu_device_kind(" custom_tpu "), "custom_tpu")
    with self.assertRaisesRegex(TypeError, "check_autograd must be a bool"):
      verifier.VerificationOptions(check_autograd="yes")  # type: ignore[arg-type]
    with self.assertRaisesRegex(
        TypeError, "max_unrolled_iterations must be an int"
    ):
      verifier.VerificationOptions(max_unrolled_iterations=True)
    with self.assertRaisesRegex(
        TypeError, "max_unrolled_iterations must be an int"
    ):
      verifier.VerificationOptions(max_unrolled_iterations=1.5)  # type: ignore[arg-type]
    with self.assertRaisesRegex(ValueError, "max_unrolled_iterations"):
      verifier.VerificationOptions(max_unrolled_iterations=0)
    with self.assertRaisesRegex(ValueError, "max_unrolled_iterations"):
      verifier.VerificationOptions(max_unrolled_iterations=-5)
    with self.assertRaisesRegex(
        TypeError, "solver_timeout_seconds must be a float or int"
    ):
      verifier.VerificationOptions(solver_timeout_seconds="5.0")  # type: ignore[arg-type]
    with self.assertRaisesRegex(
        TypeError, "solver_timeout_seconds must be a float or int"
    ):
      verifier.VerificationOptions(solver_timeout_seconds=True)
    with self.assertRaisesRegex(ValueError, "solver_timeout_seconds"):
      verifier.VerificationOptions(solver_timeout_seconds=-1.0)
    with self.assertRaisesRegex(ValueError, "solver_timeout_seconds"):
      verifier.VerificationOptions(solver_timeout_seconds=float("nan"))
    with self.assertRaisesRegex(ValueError, "solver_timeout_seconds"):
      verifier.VerificationOptions(solver_timeout_seconds=float("inf"))
    with self.assertRaisesRegex(TypeError, "tpu_device_kind must be a str"):
      verifier.VerificationOptions(tpu_device_kind=123)  # type: ignore[arg-type]
    with self.assertRaisesRegex(ValueError, "tpu_device_kind"):
      verifier.VerificationOptions(tpu_device_kind="")
    with self.assertRaisesRegex(ValueError, "tpu_device_kind"):
      verifier.VerificationOptions(tpu_device_kind="   ")
    with self.assertRaisesRegex(TypeError, "mode must be"):
      verifier.verify(lambda x: x, (torch.ones(2),), mode="static")  # type: ignore[arg-type]
    with self.assertRaisesRegex(TypeError, "options must be"):
      verifier.verify(lambda x: x, (torch.ones(2),), options={})  # type: ignore[arg-type]

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
        Exception,
        r"forbidden callable|verification traces and runs kernels",
    ):
      call_verify_in_compiled_region(torch.ones((8, 128)))

  def test_raise_for_errors_raises_with_the_report(self):
    donating_op = jax_op(
        "pallas_verify_errors::donate_add_one",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def buggy_program(x: torch.Tensor) -> torch.Tensor:
      return donating_op(x) + x

    param = torch.nn.Parameter(torch.ones((8, 128)), requires_grad=True)
    report = verifier.verify(
        buggy_program,
        param,
        mode=verifier.VerificationMode.STATIC,
    )
    with self.assertRaises(verifier.VerificationError) as ctx:
      report.raise_for_errors()
    self.assertIs(ctx.exception.report, report)
    self.assertIn("use_after_donate", str(ctx.exception))

    for target in (
        donating_op,
        donating_op._init_fn,
        torch.ops.pallas_verify_errors.donate_add_one.default,
        torch.ops.pallas_verify_errors.donate_add_one,
    ):
      op_report = verifier.verify(
          target,
          torch.ones((8, 128)),
          mode=verifier.VerificationMode.STATIC,
      )
      self.assertTrue(op_report.passed)

  def test_verify_untraceable_torch_program_raises_value_error(self):
    def untraceable(x: torch.Tensor) -> torch.Tensor:
      if int(x.sum().item()) > 0:
        return x + 1.0
      return x

    with self.assertRaisesRegex(ValueError, "could not trace untraceable"):
      verifier.verify(
          untraceable,
          torch.ones((4, 4)),
          mode=verifier.VerificationMode.STATIC,
      )


if __name__ == "__main__":
  absltest.main()
