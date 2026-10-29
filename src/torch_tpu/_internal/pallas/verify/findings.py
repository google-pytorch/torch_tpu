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

"""Findings, counterexample traces, and reports produced by verification."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import enum


class Severity(enum.Enum):
  """Severity of a verification finding."""

  ERROR = "error"
  WARNING = "warning"
  INFO = "info"


class Phase(enum.Enum):
  """Verification engine that produced a finding."""

  # `@jax_op` registration contracts and eager argument checks.
  CONTRACTS = "contracts"
  # Pallas TPU (Mosaic) lowering, run on the host without a TPU.
  LOWERING = "lowering"
  # Buffer donation and aliasing in the FX graph of a PyTorch program.
  FX_GRAPH = "fx_graph"
  # Registered custom backward compared against the forward kernel.
  AUTOGRAD = "autograd"
  # Static model checking of concurrency, semaphores, and memory accesses.
  MODEL_CHECK = "model_check"
  # Dynamic sweeps over CPU TPU-interpret schedules.
  INTERPRET = "interpret"


class FindingCategory(enum.Enum):
  """Kind of safety violation or warning."""

  # PyTorch program and `@jax_op` contract safety.
  USE_AFTER_DONATE = "use_after_donate"
  ALIASED_DONOR_INPUTS = "aliased_donor_inputs"
  DONATED_PARAMETER = "donated_parameter"
  ZERO_OUTPUT_SILENT_SKIP = "zero_output_silent_skip"
  PARTITION_SPEC_MISMATCH = "partition_spec_mismatch"
  AUTOGRAD_PARITY_MISMATCH = "autograd_parity_mismatch"
  LOWERING_FAILURE = "lowering_failure"

  # Kernel concurrency, memory, semaphore, and hardware safety.
  DATA_RACE = "data_race"
  CONCURRENT_DMA_SEM_SHARING = "concurrent_dma_sem_sharing"
  UNTHROTTLED_REMOTE_OVERWRITE = "unthrottled_remote_overwrite"
  PARALLEL_GRID_CONFLICT = "parallel_grid_conflict"
  SEMAPHORE_DEADLOCK = "semaphore_deadlock"
  SEMAPHORE_NONZERO_EXIT = "semaphore_nonzero_exit"
  UNWRITTEN_OUTPUT = "unwritten_output"
  OUT_OF_BOUNDS_ACCESS = "out_of_bounds_access"
  INVALID_DEVICE_ID = "invalid_device_id"
  NAN_OR_UNINIT_READ = "nan_or_uninit_read"
  SCHEDULE_DIVERGENCE = "schedule_divergence"
  HARDWARE_CONSTRAINT_VIOLATION = "hardware_constraint_violation"
  LOW_PRECISION_ACCUMULATION = "low_precision_accumulation"

  # Parts of the program that could not be verified (bounded unrolling,
  # solver timeouts, unsupported constructs, skipped phases).
  VERIFICATION_INCOMPLETE = "verification_incomplete"


class DmaScheduleMode(enum.Enum):
  """DMA execution schedule policy for CPU TPU interpret simulation."""

  EAGER = "eager"
  ON_WAIT = "on_wait"


class UninitMemoryMode(enum.Enum):
  """Scratch buffer initialization policy for CPU TPU interpret simulation."""

  NAN = "nan"
  ZERO = "zero"


@dataclasses.dataclass(frozen=True)
class CounterexampleStep:
  """One ordered event in a concrete counterexample execution schedule."""

  timestamp: int
  device_id: int
  core_id: int
  actor: str
  operation: str
  detail: str
  source_location: str | None = None


@dataclasses.dataclass(frozen=True)
class VerificationFinding:
  """A single safety violation or warning."""

  category: FindingCategory
  severity: Severity
  phase: Phase
  message: str
  op_name: str | None = None
  source_location: str | None = None
  counterexample: tuple[CounterexampleStep, ...] = ()

  def format(self) -> str:
    """Formats this finding and any counterexample trace into readable text."""
    header = (
        f"[{self.severity.value.upper()}] ({self.phase.value})"
        f" {self.category.value}"
    )
    if self.op_name:
      header += f" in {self.op_name}"
    if self.source_location:
      header += f" @ {self.source_location}"
    lines = [f"{header}: {self.message}"]
    if self.counterexample:
      lines.append("  Counterexample schedule:")
      for step in self.counterexample:
        loc = f" ({step.source_location})" if step.source_location else ""
        lines.append(
            f"    t={step.timestamp:02d} [dev={step.device_id},"
            f" core={step.core_id}, {step.actor}] {step.operation}:"
            f" {step.detail}{loc}"
        )
    return "\n".join(lines)


class VerificationError(AssertionError):
  """Raised by `VerificationReport.raise_for_errors` when errors were found.

  Attributes:
    report: The failing report.
  """

  def __init__(self, report: VerificationReport) -> None:
    super().__init__(report.summary())
    self.report = report


@dataclasses.dataclass(frozen=True)
class VerificationReport:
  """Aggregated result of one or more verification phases.

  A report is truthy if and only if it contains no errors, so
  `assert verify(...)` works; `raise_for_errors()` additionally includes the
  formatted findings in the exception message.
  """

  findings: tuple[VerificationFinding, ...] = ()
  checked_phases: tuple[Phase, ...] = ()

  @classmethod
  def of(
      cls, phase: Phase, detected: Iterable[VerificationFinding] = ()
  ) -> VerificationReport:
    """Returns a single-phase report holding `detected`."""
    return cls(findings=tuple(detected), checked_phases=(phase,))

  @classmethod
  def merge(cls, *reports: VerificationReport) -> VerificationReport:
    """Combines several reports, keeping the first-seen phase order."""
    findings: list[VerificationFinding] = []
    phases: list[Phase] = []
    for rep in reports:
      phases.extend(p for p in rep.checked_phases if p not in phases)
      findings.extend(rep.findings)
    return cls(findings=tuple(findings), checked_phases=tuple(phases))

  @property
  def passed(self) -> bool:
    """Whether no error-severity finding was produced."""
    return not self.errors

  @property
  def errors(self) -> tuple[VerificationFinding, ...]:
    """Findings with `Severity.ERROR`."""
    return tuple(f for f in self.findings if f.severity is Severity.ERROR)

  @property
  def warnings(self) -> tuple[VerificationFinding, ...]:
    """Findings with `Severity.WARNING`."""
    return tuple(f for f in self.findings if f.severity is Severity.WARNING)

  def findings_by_category(
      self, category: FindingCategory
  ) -> tuple[VerificationFinding, ...]:
    """Returns the findings of `category`."""
    return tuple(f for f in self.findings if f.category is category)

  def has_category(self, category: FindingCategory) -> bool:
    """Returns whether any finding has `category`."""
    return any(f.category is category for f in self.findings)

  def summary(self) -> str:
    """Returns a human-readable summary of all findings."""
    status = "PASSED" if self.passed else "FAILED"
    phases = ", ".join(p.value for p in self.checked_phases) or "none"
    lines = [
        f"Verification {status} (phases: {phases};"
        f" {len(self.errors)} error(s), {len(self.warnings)} warning(s))"
    ]
    lines.extend(finding.format() for finding in self.findings)
    return "\n".join(lines)

  def raise_for_errors(self) -> None:
    """Raises `VerificationError` if the report contains any error."""
    if not self.passed:
      raise VerificationError(self)

  def __bool__(self) -> bool:
    return self.passed

  def __str__(self) -> str:
    return self.summary()
