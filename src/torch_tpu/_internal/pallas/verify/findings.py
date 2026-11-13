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
import math
from typing import TypeAlias

from torch_tpu._internal.utils import annotations
from torch_tpu._internal.utils import hardware

_PUBLIC_PREFIX = "torch.tpu.pallas.verifier"


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


TpuVersion: TypeAlias = hardware.TpuVersion


# Canonical TorchTPU device names (matching `TpuSpec::name()` in
# `tpu_hardware.h`) and their common aliases.
_TPU_VERSION_ALIASES: dict[str, TpuVersion] = {
    "tpu v5e": TpuVersion.V5E,
    "tpu v5 lite": TpuVersion.V5E,
    "v5e": TpuVersion.V5E,
    "v5 lite": TpuVersion.V5E,
    "tpu v5p": TpuVersion.V5P,
    "tpu v5": TpuVersion.V5P,
    "v5p": TpuVersion.V5P,
    "v5": TpuVersion.V5P,
    "tpu v6e": TpuVersion.V6E,
    "tpu v6 lite": TpuVersion.V6E,
    "tpu v6": TpuVersion.V6E,
    "v6e": TpuVersion.V6E,
    "v6 lite": TpuVersion.V6E,
    "v6": TpuVersion.V6E,
    "trillium": TpuVersion.V6E,
    "tpu v7": TpuVersion.V7,
    "tpu v7x": TpuVersion.V7,
    "tpu 7x": TpuVersion.V7,
    "tpu 7": TpuVersion.V7,
    "tpu7x": TpuVersion.V7,
    "tpu7": TpuVersion.V7,
    "v7": TpuVersion.V7,
    "v7x": TpuVersion.V7,
    "tpu v8i": TpuVersion.V8I,
    "tpu 8i": TpuVersion.V8I,
    "tpu8i": TpuVersion.V8I,
    "v8i": TpuVersion.V8I,
    "tpu v8t": TpuVersion.V8T,
    "tpu 8t": TpuVersion.V8T,
    "tpu8t": TpuVersion.V8T,
    "v8t": TpuVersion.V8T,
}

_TORCH_TO_JAX_TPU_DEVICE_KIND: dict[TpuVersion, str] = {
    TpuVersion.V5E: "TPU v5e",
    TpuVersion.V5P: "TPU v5p",
    TpuVersion.V6E: "TPU v6e",
    TpuVersion.V7: "TPU7x",
    TpuVersion.V8I: "TPU8i",
    TpuVersion.V8T: "TPU8t",
}

# TPU generations that do NOT support DMA done-count mode on
# `pltpu.SemaphoreType.REGULAR` semaphores for non-SMEM transfers (TPU v5e
# requires `pltpu.SemaphoreType.DMA`; TPU v5p and newer support done-count mode).
_NO_DMA_DONE_COUNT_TPU_VERSIONS: frozenset[TpuVersion] = frozenset({
    TpuVersion.V5E,
})

# Default TorchTPU generation that ops without a mesh or inferred target are
# lowered for.
DEFAULT_TPU_VERSION: TpuVersion = TpuVersion.V7


def parse_tpu_version(
    tpu_version: TpuVersion | str,
) -> TpuVersion | None:
  """Parses a `TpuVersion` or TorchTPU/JAX TPU alias string."""
  if isinstance(tpu_version, TpuVersion):
    return tpu_version
  key = " ".join(
      tpu_version.strip().lower().replace("_", " ").replace("-", " ").split()
  )
  return _TPU_VERSION_ALIASES.get(key)


def normalize_tpu_version(tpu_version: TpuVersion | str) -> str:
  """Normalizes a `TpuVersion` or TPU alias to TorchTPU's `TPU v*` name."""
  if isinstance(tpu_version, TpuVersion):
    return tpu_version.value
  parsed = parse_tpu_version(tpu_version)
  if parsed is not None:
    return parsed.value
  raise ValueError(f"unrecognized TPU version: {tpu_version!r}")


def jax_tpu_device_kind(tpu_version: TpuVersion | str) -> str:
  """Maps a `TpuVersion` or TPU device name to JAX's `AbstractDevice` string."""
  if isinstance(tpu_version, TpuVersion):
    return _TORCH_TO_JAX_TPU_DEVICE_KIND[tpu_version]
  parsed = parse_tpu_version(tpu_version)
  if parsed is not None:
    return _TORCH_TO_JAX_TPU_DEVICE_KIND[parsed]
  raise ValueError(f"unrecognized TPU version: {tpu_version!r}")


def supports_dma_done_count_mode(tpu_version: TpuVersion | str) -> bool:
  """Returns whether `tpu_version` supports done-count regular DMA sems."""
  if isinstance(tpu_version, TpuVersion):
    return tpu_version not in _NO_DMA_DONE_COUNT_TPU_VERSIONS
  parsed = parse_tpu_version(tpu_version)
  if parsed is not None:
    return parsed not in _NO_DMA_DONE_COUNT_TPU_VERSIONS
  raise ValueError(f"unrecognized TPU version: {tpu_version!r}")


@annotations.experimental(_exp_reason("Severity"))
class Severity(enum.Enum):
  """Severity of a verification finding.

  `ERROR` findings cause `VerificationReport.passed` to evaluate to `False` and
  `VerificationReport.raise_for_errors()` to raise `VerificationError`.
  `WARNING` and `INFO` findings are surfaced in summaries without failing
  verification by default (`VERIFICATION_INCOMPLETE` findings also raise when
  `raise_for_errors(fail_on_incomplete=True)` is passed).
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
  # Dynamic sweeps over CPU TPU-interpret schedules.
  INTERPRET = "interpret"


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
  NAN_OR_UNINIT_READ = "nan_or_uninit_read"
  SCHEDULE_DIVERGENCE = "schedule_divergence"

  # Parts of the program that could not be verified (bounded unrolling,
  # solver timeouts, unsupported constructs, skipped phases).
  VERIFICATION_INCOMPLETE = "verification_incomplete"


@annotations.experimental(_exp_reason("DmaScheduleMode"))
class DmaScheduleMode(enum.Enum):
  """DMA execution schedule policy for CPU TPU interpret simulation."""

  # DMAs complete as soon as `dma_start` issues.
  EAGER = "eager"
  # Pending DMAs remain in flight and execute when a semaphore wait runs.
  ON_WAIT = "on_wait"


@annotations.experimental(_exp_reason("UninitMemoryMode"))
class UninitMemoryMode(enum.Enum):
  """Buffer initialization policy for CPU TPU interpret simulation."""

  # Scratch and output buffers are poisoned (quiet NaNs for floating-point
  # dtypes, `iinfo.max` for integers, `True` for booleans) so reads of
  # unwritten regions propagate into the kernel's outputs.
  NAN = "nan"
  # Scratch and output buffers are zero-filled before kernel execution.
  ZERO = "zero"


@annotations.experimental(_exp_reason("InterpretSweepConfig"))
@dataclasses.dataclass(frozen=True)
class InterpretSweepConfig:
  """Configuration for CPU TPU interpret schedule sweeping.

  Attributes:
    schedule_modes: DMA completion timing policies (`EAGER`, `ON_WAIT`) to test.
    seeds: PRNG seeds passed as `random_seed` to TPU interpret mode to randomize
      `parallel`-dimension grid execution order.
    uninitialized_memory: Scratch and output buffer initialization policy (`NAN`
      or `ZERO`). When set to `NAN`, the verifier poisons buffers (quiet NaNs
      for floats, `iinfo.max` for integers, `True` for booleans) and compares
      against a `ZERO`-filled control run to distinguish genuine uninitialized
      reads or unwritten output regions from intentional kernel-computed values.
    rtol: Relative tolerance used when comparing floating-point/inexact outputs
      across schedules (integer and boolean outputs require exact equality).
    atol: Absolute tolerance used when comparing floating-point/inexact outputs
      across schedules (integer and boolean outputs require exact equality).
  """

  schedule_modes: tuple[DmaScheduleMode, ...] = (
      DmaScheduleMode.EAGER,
      DmaScheduleMode.ON_WAIT,
  )
  seeds: tuple[int, ...] = (0, 17)
  uninitialized_memory: UninitMemoryMode = UninitMemoryMode.NAN
  rtol: float = 1e-4
  atol: float = 1e-4

  __tt_api_stages__ = _field_stages(
      "InterpretSweepConfig",
      (
          "schedule_modes",
          "seeds",
          "uninitialized_memory",
          "rtol",
          "atol",
      ),
  )

  def __post_init__(self) -> None:
    if isinstance(self.schedule_modes, DmaScheduleMode):
      modes = (self.schedule_modes,)
    elif isinstance(self.schedule_modes, Iterable) and not isinstance(
        self.schedule_modes, (str, bytes)
    ):
      modes = tuple(self.schedule_modes)
    else:
      raise TypeError(
          "schedule_modes must be a sequence of DmaScheduleMode, got"
          f" {type(self.schedule_modes).__name__}"
      )
    if not all(isinstance(mode, DmaScheduleMode) for mode in modes):
      raise TypeError(
          "schedule_modes elements must be DmaScheduleMode instances"
      )
    if not modes:
      raise ValueError(
          "schedule_modes must be a non-empty sequence of DmaScheduleMode"
      )
    object.__setattr__(self, "schedule_modes", tuple(dict.fromkeys(modes)))

    if isinstance(self.seeds, Iterable) and not isinstance(
        self.seeds, (str, bytes)
    ):
      seeds = tuple(self.seeds)
    else:
      raise TypeError(
          f"seeds must be a sequence of ints, got {type(self.seeds).__name__}"
      )
    if not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
    ):
      raise TypeError("seeds elements must be ints")
    if not seeds or not all(0 <= seed < (1 << 31) for seed in seeds):
      raise ValueError(
          "seeds must be a non-empty sequence of ints in [0, 2**31)"
      )
    object.__setattr__(self, "seeds", tuple(dict.fromkeys(seeds)))

    if not isinstance(self.uninitialized_memory, UninitMemoryMode):
      raise TypeError(
          "uninitialized_memory must be an UninitMemoryMode, got"
          f" {type(self.uninitialized_memory).__name__}"
      )
    if not isinstance(self.rtol, (int, float)) or isinstance(self.rtol, bool):
      raise TypeError(
          f"rtol must be a float or int, got {type(self.rtol).__name__}"
      )
    if not math.isfinite(self.rtol) or self.rtol < 0:
      raise ValueError(
          f"rtol must be a finite non-negative number, got {self.rtol!r}"
      )
    if not isinstance(self.atol, (int, float)) or isinstance(self.atol, bool):
      raise TypeError(
          f"atol must be a float or int, got {type(self.atol).__name__}"
      )
    if not math.isfinite(self.atol) or self.atol < 0:
      raise ValueError(
          f"atol must be a finite non-negative number, got {self.atol!r}"
      )


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
  """Raised by `VerificationReport.raise_for_errors` on failure."""

  def __init__(self, report: VerificationReport) -> None:
    msg = report.summary()
    if report.passed and msg.startswith("Verification PASSED "):
      msg = "Verification INCOMPLETE " + msg[len("Verification PASSED ") :]
    super().__init__(msg)
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
  def raise_for_errors(self, *, fail_on_incomplete: bool = False) -> None:
    """Raises `VerificationError` if the report contains any error.

    Args:
      fail_on_incomplete: If `True`, also raises `VerificationError` when any
        `FindingCategory.VERIFICATION_INCOMPLETE` finding is present.
    """
    if not self.passed or (
        fail_on_incomplete
        and bool(
            self.findings_by_category(FindingCategory.VERIFICATION_INCOMPLETE)
        )
    ):
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
