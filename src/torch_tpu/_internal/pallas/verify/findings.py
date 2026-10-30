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

"""Findings, counterexample traces, and reports produced by verification.

Every verification pass in `torch.tpu.pallas.verifier` emits structured
`VerificationFinding` records grouped into a `VerificationReport`. Reports track
both the findings discovered and the ordered sequence of verification `Phase`s
that ran, and `report.passed` evaluates to `True` if and only if no
`Severity.ERROR` findings are present.
"""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import enum

from torch_tpu._internal.utils import annotations

_PUBLIC_PREFIX = "torch.tpu.pallas.verifier"


class _DefaultTpuDeviceKind(str):
  """Sentinel `str` subclass marking an un-overridden default TPU device kind."""


# Canonical TorchTPU device names (matching `TpuSpec::name` in `tpu_hardware.cc`)
# and their common aliases.
_CANONICAL_TPU_DEVICE_KINDS: dict[str, str] = {
    "tpu v5e": "TPU v5e",
    "tpu v5 lite": "TPU v5e",
    "v5e": "TPU v5e",
    "v5 lite": "TPU v5e",
    "tpu v5p": "TPU v5p",
    "tpu v5": "TPU v5p",
    "v5p": "TPU v5p",
    "v5": "TPU v5p",
    "tpu v6e": "TPU v6e",
    "tpu v6 lite": "TPU v6e",
    "tpu v6": "TPU v6e",
    "v6e": "TPU v6e",
    "v6 lite": "TPU v6e",
    "v6": "TPU v6e",
    "trillium": "TPU v6e",
    "tpu v7": "TPU v7",
    "tpu v7x": "TPU v7",
    "tpu7x": "TPU v7",
    "tpu7": "TPU v7",
    "v7": "TPU v7",
    "v7x": "TPU v7",
    "tpu v8i": "TPU v8i",
    "tpu8i": "TPU v8i",
    "v8i": "TPU v8i",
    "tpu v8t": "TPU v8t",
    "tpu8t": "TPU v8t",
    "v8t": "TPU v8t",
}

_TORCH_TO_JAX_TPU_DEVICE_KIND: dict[str, str] = {
    "TPU v5e": "TPU v5e",
    "TPU v5p": "TPU v5p",
    "TPU v6e": "TPU v6e",
    "TPU v7": "TPU7x",
    "TPU v8i": "TPU8i",
    "TPU v8t": "TPU8t",
}

# TPU generations that support DMA done-count mode on
# `pltpu.SemaphoreType.REGULAR` semaphores for non-SMEM transfers (TPU v5p and
# newer; TPU v5e requires `pltpu.SemaphoreType.DMA` for non-SMEM transfers).
_DMA_DONE_COUNT_TPU_KINDS = frozenset({
    "TPU v5p",
    "TPU v6e",
    "TPU v7",
    "TPU v8i",
    "TPU v8t",
})

# Default TorchTPU device kind that ops without a mesh or inferred target are
# lowered for.
DEFAULT_TPU_DEVICE_KIND: str = _DefaultTpuDeviceKind("TPU v7")


def is_default_tpu_device_kind(device_kind: str) -> bool:
  """Returns whether `device_kind` is the un-overridden default sentinel."""
  return isinstance(device_kind, _DefaultTpuDeviceKind)


def normalize_tpu_device_kind(device_kind: str) -> str:
  """Normalizes TPU device aliases to TorchTPU's canonical `TPU v*` names."""
  key = " ".join(
      device_kind.strip().lower().replace("_", " ").replace("-", " ").split()
  )
  canonical = _CANONICAL_TPU_DEVICE_KINDS.get(key)
  if canonical is not None:
    return canonical
  return device_kind.strip()


def jax_tpu_device_kind(device_kind: str) -> str:
  """Maps a TorchTPU or JAX TPU device name to JAX's `AbstractDevice` string."""
  canonical = normalize_tpu_device_kind(device_kind)
  return _TORCH_TO_JAX_TPU_DEVICE_KIND.get(canonical, device_kind.strip())


def supports_dma_done_count_mode(device_kind: str) -> bool:
  """Returns whether `device_kind` supports DMA done-count regular semaphores."""
  return normalize_tpu_device_kind(device_kind) in _DMA_DONE_COUNT_TPU_KINDS


def _exp_reason(qualname: str) -> str:
  return f"{_PUBLIC_PREFIX}.{qualname} is experimental and subject to change."


def _field_stages(
    cls_name: str, field_names: Iterable[str]
) -> dict[str, annotations.ApiStageInfo]:
  return {
      name: annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason=_exp_reason(f"{cls_name}.{name}"),
      )
      for name in field_names
  }


@annotations.experimental(_exp_reason("Severity"))
class Severity(enum.Enum):
  """Severity of a verification finding.

  Only `ERROR` findings cause `VerificationReport.passed` to evaluate to `False`
  or cause `VerificationReport.raise_for_errors()` to raise `VerificationError`.
  `WARNING` and `INFO` findings are surfaced in summaries without failing
  verification.
  """

  ERROR = "error"
  WARNING = "warning"
  INFO = "info"


@annotations.experimental(_exp_reason("Phase"))
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
  # Static model checking of Pallas kernel Jaxprs, semaphores, and memory
  # accesses.
  MODEL_CHECK = "model_check"


@annotations.experimental(_exp_reason("FindingCategory"))
class FindingCategory(enum.Enum):
  """Kind of safety violation or warning."""

  # PyTorch program and `@jax_op` contract safety.
  USE_AFTER_DONATE = "use_after_donate"
  ALIASED_DONOR_INPUTS = "aliased_donor_inputs"
  DONATED_PARAMETER = "donated_parameter"
  ZERO_OUTPUT_SILENT_SKIP = "zero_output_silent_skip"
  PARTITION_SPEC_MISMATCH = "partition_spec_mismatch"
  FAKE_KERNEL_MISMATCH = "fake_kernel_mismatch"
  AUTOGRAD_PARITY_MISMATCH = "autograd_parity_mismatch"
  LOWERING_FAILURE = "lowering_failure"

  # Kernel concurrency, memory, semaphore, and hardware safety.
  UNWRITTEN_OUTPUT = "unwritten_output"
  OUT_OF_BOUNDS_ACCESS = "out_of_bounds_access"
  INVALID_DEVICE_ID = "invalid_device_id"
  HARDWARE_CONSTRAINT_VIOLATION = "hardware_constraint_violation"
  LOW_PRECISION_ACCUMULATION = "low_precision_accumulation"
  DATA_RACE = "data_race"
  CONCURRENT_DMA_SEM_SHARING = "concurrent_dma_sem_sharing"
  UNTHROTTLED_REMOTE_OVERWRITE = "unthrottled_remote_overwrite"
  PARALLEL_GRID_CONFLICT = "parallel_grid_conflict"
  SEMAPHORE_DEADLOCK = "semaphore_deadlock"
  SEMAPHORE_NONZERO_EXIT = "semaphore_nonzero_exit"

  # Parts of the program that could not be verified (bounded unrolling,
  # solver timeouts, unsupported constructs, skipped phases).
  VERIFICATION_INCOMPLETE = "verification_incomplete"


@annotations.experimental(_exp_reason("CounterexampleStep"))
@dataclasses.dataclass(frozen=True)
class CounterexampleStep:
  """One ordered event in a concrete counterexample execution schedule.

  Attributes:
    timestamp: Compacted integer logical timestamp (`0, 1, ...`).
    device_id: Mesh device index (`0` for single-device kernels).
    core_id: Core or sub-mesh index on the device.
    actor: Short identifier of the executing core, grid iteration, or DMA
      engine.
    operation: Kind of operation (`vpu_load`, `vpu_store`, `dma_start`,
      `dma_write_done`, `sem_signal`, `sem_wait`, etc.).
    detail: Human-readable description of the buffer region or semaphore action.
    source_location: Optional `file:line` in user code where the operation
      originated.
  """

  timestamp: int
  device_id: int
  core_id: int
  actor: str
  operation: str
  detail: str
  source_location: str | None = None

  __tt_api_stages__ = _field_stages(
      "CounterexampleStep",
      (
          "timestamp",
          "device_id",
          "core_id",
          "actor",
          "operation",
          "detail",
          "source_location",
      ),
  )


@annotations.experimental(_exp_reason("VerificationFinding"))
@dataclasses.dataclass(frozen=True)
class VerificationFinding:
  """A single safety violation or warning."""

  category: FindingCategory
  severity: Severity
  phase: Phase | None
  message: str
  op_name: str | None = None
  source_location: str | None = None
  counterexample: tuple[CounterexampleStep, ...] = ()

  __tt_api_stages__ = _field_stages(
      "VerificationFinding",
      (
          "category",
          "severity",
          "phase",
          "message",
          "op_name",
          "source_location",
          "counterexample",
      ),
  )

  @annotations.experimental(_exp_reason("VerificationFinding.format"))
  def format(self) -> str:
    """Formats this finding and any counterexample trace into readable text."""
    phase_part = f" ({self.phase.value})" if self.phase is not None else ""
    header = (
        f"[{self.severity.value.upper()}]{phase_part} {self.category.value}"
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


@annotations.experimental(_exp_reason("VerificationError"))
class VerificationError(AssertionError):
  """Raised by `VerificationReport.raise_for_errors` when errors were found."""

  def __init__(self, report: VerificationReport) -> None:
    super().__init__(report.summary())
    self._report = report

  def __reduce__(
      self,
  ) -> tuple[type[VerificationError], tuple[VerificationReport]]:
    return (type(self), (self._report,))

  @property
  @annotations.experimental(_exp_reason("VerificationError.report"))
  def report(self) -> VerificationReport:
    """The failing report."""
    return self._report


@annotations.experimental(_exp_reason("VerificationReport"))
@dataclasses.dataclass(frozen=True)
class VerificationReport:
  """Aggregated result of one or more verification phases.

  `report.passed` is `True` if and only if the report contains no errors;
  `raise_for_errors()` raises `VerificationError` with the formatted findings in
  the exception message.
  """

  findings: tuple[VerificationFinding, ...] = ()
  checked_phases: tuple[Phase, ...] = ()

  __tt_api_stages__ = _field_stages(
      "VerificationReport", ("findings", "checked_phases")
  )

  @property
  @annotations.experimental(_exp_reason("VerificationReport.passed"))
  def passed(self) -> bool:
    """Whether no error-severity finding was produced."""
    return not any(f.severity is Severity.ERROR for f in self.findings)

  @annotations.experimental(
      _exp_reason("VerificationReport.findings_by_severity")
  )
  def findings_by_severity(
      self, severity: Severity
  ) -> tuple[VerificationFinding, ...]:
    """Returns the findings of `severity`."""
    return tuple(f for f in self.findings if f.severity is severity)

  @annotations.experimental(
      _exp_reason("VerificationReport.findings_by_category")
  )
  def findings_by_category(
      self, category: FindingCategory
  ) -> tuple[VerificationFinding, ...]:
    """Returns the findings of `category`."""
    return tuple(f for f in self.findings if f.category is category)

  @annotations.experimental(_exp_reason("VerificationReport.summary"))
  def summary(self) -> str:
    """Returns a human-readable summary of all findings."""
    status = "PASSED" if self.passed else "FAILED"
    phases = ", ".join(p.value for p in self.checked_phases) or "none"
    num_errors = len(self.findings_by_severity(Severity.ERROR))
    num_warnings = len(self.findings_by_severity(Severity.WARNING))
    num_infos = len(self.findings_by_severity(Severity.INFO))
    info_suffix = f", {num_infos} info(s)" if num_infos else ""
    lines = [
        f"Verification {status} (phases: {phases};"
        f" {num_errors} error(s), {num_warnings}"
        f" warning(s){info_suffix})"
    ]
    lines.extend(finding.format() for finding in self.findings)
    return "\n".join(lines)

  @annotations.experimental(_exp_reason("VerificationReport.raise_for_errors"))
  def raise_for_errors(self) -> None:
    """Raises `VerificationError` if the report contains any error."""
    if not self.passed:
      raise VerificationError(self)

  def __str__(self) -> str:
    return self.summary()


def report_of(
    phase: Phase | None, detected: Iterable[VerificationFinding] = ()
) -> VerificationReport:
  """Returns a single-phase report holding `detected`."""
  phases = (phase,) if phase is not None else ()
  return VerificationReport(findings=tuple(detected), checked_phases=phases)


def merge_reports(*reports: VerificationReport) -> VerificationReport:
  """Combines several reports, keeping the first-seen phase order."""
  merged_findings: list[VerificationFinding] = []
  phases: list[Phase] = []
  for rep in reports:
    phases.extend(p for p in rep.checked_phases if p not in phases)
    merged_findings.extend(rep.findings)
  return VerificationReport(
      findings=tuple(merged_findings), checked_phases=tuple(phases)
  )
