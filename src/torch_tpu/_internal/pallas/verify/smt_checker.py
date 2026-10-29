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

"""Model checking of Pallas kernel schedules with difference logic.

Every extracted event gets an integer timestamp. Program order, DMA causality
(start < read done <= write done), and counting-semaphore semantics (a wait
completes only after enough signals) become difference and pseudo-Boolean
constraints over those timestamps. The checker then asks `idl_solver`:

* whether any complete schedule exists at all; if not, some wait can never be
  satisfied and the kernel deadlocks, and
* for each pair of conflicting accesses to overlapping memory, whether some
  schedule lets them overlap in time; if so, it reports a data race with a
  concrete counterexample schedule.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import itertools
from typing import Any

from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import idl_solver
from torch_tpu._internal.pallas.verify import jaxpr_model

_PHASE = findings.Phase.MODEL_CHECK
_Category = findings.FindingCategory
_SolverStatus = idl_solver.SolverStatus
# Memory spaces shared by all cores of a chip; VMEM and SMEM are per core.
_SHARED_MEMORY_SPACES = frozenset({"hbm", "vmem_shared"})


@dataclasses.dataclass(frozen=True)
class ModelCheckConfig:
  """Bounds for kernel model extraction and solving.

  Attributes:
    max_unrolled_iterations: Bound on unrolled grid points, loop iterations, and
      scan steps per kernel.
    solver_timeout_seconds: Time budget for each solver query.
  """

  max_unrolled_iterations: int = 32
  solver_timeout_seconds: float = 5.0


@dataclasses.dataclass(frozen=True)
class _TimePoint:
  """A schedule timestamp: an actor event or a DMA completion."""

  key: str
  actor: jaxpr_model.ActorKey | None = None
  position: int = -1
  # For DMA completions: the actor and program position of the DMA start.
  started_by: tuple[jaxpr_model.ActorKey, int] | None = None


def _static_order(a: _TimePoint, b: _TimePoint) -> int:
  """Returns -1 if `a` always precedes `b`, 1 if it always follows, else 0."""
  if a.key == b.key:
    return 0
  if a.actor is not None and a.actor == b.actor:
    return -1 if a.position < b.position else 1
  if a.started_by is not None and b.actor is not None:
    actor, start = a.started_by
    if actor == b.actor and b.position <= start:
      return 1  # b <= DMA start < completion a.
  if b.started_by is not None and a.actor is not None:
    actor, start = b.started_by
    if actor == a.actor and a.position <= start:
      return -1
  return 0


@dataclasses.dataclass(frozen=True)
class _Signal:
  point: _TimePoint
  sem: jaxpr_model.SemaphoreKey
  amount: int
  source_location: str | None


@dataclasses.dataclass(frozen=True)
class _Wait:
  point: _TimePoint
  sem: jaxpr_model.SemaphoreKey
  amount: int
  decrement: bool
  source_location: str | None


@dataclasses.dataclass(frozen=True)
class _Access:
  """A memory access that is in flight during `[start, end]`."""

  start: _TimePoint
  end: _TimePoint
  actor: jaxpr_model.ActorKey
  buffer: jaxpr_model.BufferKey
  region: jaxpr_model.HyperRectSlice
  is_write: bool
  dma: jaxpr_model.DmaStartEvent | None
  detail: str
  source_location: str | None


@dataclasses.dataclass(frozen=True)
class _TraceEntry:
  point: _TimePoint
  device_id: int
  core_id: int
  actor: str
  operation: str
  detail: str
  source_location: str | None


@dataclasses.dataclass
class _Schedule:
  """Synchronization structure of one kernel model."""

  signals: dict[jaxpr_model.SemaphoreKey, list[_Signal]]
  waits: dict[jaxpr_model.SemaphoreKey, list[_Wait]]
  accesses: list[_Access]
  trace: list[_TraceEntry]

  def add_trace(
      self,
      point: _TimePoint,
      actor: jaxpr_model.ActorKey,
      operation: str,
      detail: str,
      source_location: str | None,
  ) -> None:
    self.trace.append(
        _TraceEntry(
            point,
            actor.device_id,
            actor.core_id,
            actor.short_str(),
            operation,
            detail,
            source_location,
        )
    )


def _buffer_str(buf: jaxpr_model.BufferKey) -> str:
  return f"{buf.memory_space}:{buf.buffer_id}"


def _collect_schedule(model: jaxpr_model.KernelVerificationModel) -> _Schedule:
  """Turns the extracted events into time points, waits, and accesses."""
  positions = {
      event_id: (actor, pos)
      for actor, event_ids in model.actor_event_orders.items()
      for pos, event_id in enumerate(event_ids)
  }

  def point(event_id: int) -> _TimePoint:
    actor, pos = positions[event_id]
    return _TimePoint(f"ev_{event_id}", actor, pos)

  sched = _Schedule({}, {}, [], [])
  for mem in model.memory_events:
    p = point(mem.event_id)
    if mem.is_implicit_pipeline:
      op = "pipeline_store" if mem.is_write else "pipeline_load"
    else:
      op = "vpu_store" if mem.is_write else "vpu_load"
    detail = (
        f"{'write' if mem.is_write else 'read'} {_buffer_str(mem.buffer)}"
        f"{mem.region}"
    )
    sched.accesses.append(
        _Access(
            p,
            p,
            mem.actor,
            mem.buffer,
            mem.region,
            mem.is_write,
            None,
            detail,
            mem.source_location,
        )
    )
    sched.add_trace(p, mem.actor, op, detail, mem.source_location)

  for dma in model.dma_starts:
    _collect_dma(dma, point(dma.event_id), sched)

  for dwait in model.dma_waits:
    p = point(dwait.event_id)
    sched.waits.setdefault(dwait.sem, []).append(
        _Wait(p, dwait.sem, dwait.byte_count, True, dwait.source_location)
    )
    op = "dma_wait_send" if dwait.is_wait_send else "dma_wait"
    detail = f"wait {dwait.sem.short_str()} (-{dwait.byte_count}B)"
    sched.add_trace(p, dwait.actor, op, detail, dwait.source_location)

  for sig in model.sem_signals:
    p = point(sig.event_id)
    sched.signals.setdefault(sig.target_sem, []).append(
        _Signal(p, sig.target_sem, sig.amount, sig.source_location)
    )
    detail = f"signal {sig.target_sem.short_str()} (+{sig.amount})"
    sched.add_trace(p, sig.actor, "sem_signal", detail, sig.source_location)

  for swait in model.sem_waits:
    p = point(swait.event_id)
    sched.waits.setdefault(swait.sem, []).append(
        _Wait(
            p, swait.sem, swait.amount, swait.decrement, swait.source_location
        )
    )
    sign = "-" if swait.decrement else ">="
    detail = f"wait {swait.sem.short_str()} ({sign}{swait.amount})"
    sched.add_trace(p, swait.actor, "sem_wait", detail, swait.source_location)
  return sched


def _collect_dma(
    dma: jaxpr_model.DmaStartEvent, start: _TimePoint, sched: _Schedule
) -> None:
  """Adds the time points, accesses, and signals of one DMA."""
  started_by = (dma.actor, start.position)
  read_done = _TimePoint(f"dma_r_{dma.dma_id}", started_by=started_by)
  write_done = _TimePoint(f"dma_w_{dma.dma_id}", started_by=started_by)
  kind = "rdma" if dma.is_remote else "dma"
  engine = f"{kind}#{dma.dma_id}"
  src, dst = dma.src_buffer, dma.dst_buffer
  sched.add_trace(
      start,
      dma.actor,
      f"{kind}_start",
      f"launch {engine} dev{src.device_id}:{src.buffer_id}{dma.src_region}"
      f" -> dev{dst.device_id}:{dst.buffer_id}{dma.dst_region}"
      f" ({dma.byte_count}B)",
      dma.source_location,
  )
  done = f"complete write dev{dst.device_id}:{dst.buffer_id}{dma.dst_region}"
  if dma.dst_sem is not None:
    done += f" & signal {dma.dst_sem.short_str()} (+{dma.byte_count}B)"
    sched.signals.setdefault(dma.dst_sem, []).append(
        _Signal(write_done, dma.dst_sem, dma.byte_count, dma.source_location)
    )
  sched.trace.append(
      _TraceEntry(
          write_done,
          dst.device_id,
          dst.core_id,
          engine,
          f"{kind}_write_done",
          done,
          dma.source_location,
      )
  )
  if dma.src_sem is not None:
    sched.trace.append(
        _TraceEntry(
            read_done,
            src.device_id,
            src.core_id,
            engine,
            f"{kind}_read_done",
            f"finish reading & signal {dma.src_sem.short_str()}"
            f" (+{dma.byte_count}B)",
            dma.source_location,
        )
    )
    sched.signals.setdefault(dma.src_sem, []).append(
        _Signal(read_done, dma.src_sem, dma.byte_count, dma.source_location)
    )
  sched.accesses.append(
      _Access(
          start,
          read_done,
          dma.actor,
          src,
          dma.src_region,
          False,
          dma,
          f"{engine} reading {src.buffer_id}{dma.src_region}",
          dma.source_location,
      )
  )
  sched.accesses.append(
      _Access(
          start,
          write_done,
          dma.actor,
          dst,
          dma.dst_region,
          True,
          dma,
          f"{engine} (from dev{dma.actor.device_id}) writing"
          f" {dst.buffer_id}{dma.dst_region}",
          dma.source_location,
      )
  )


def _finding(
    category: findings.FindingCategory,
    message: str,
    op_name: str,
    *,
    severity: findings.Severity = findings.Severity.ERROR,
    source_location: str | None = None,
    counterexample: tuple[findings.CounterexampleStep, ...] = (),
) -> findings.VerificationFinding:
  return findings.VerificationFinding(
      category=category,
      severity=severity,
      phase=_PHASE,
      message=message,
      op_name=op_name,
      source_location=source_location,
      counterexample=counterexample,
  )


def _check_conservation(
    sched: _Schedule,
    op_name: str,
    skipped: Collection[jaxpr_model.SemaphoreKey] = (),
) -> tuple[list[findings.VerificationFinding], bool]:
  """Compares total signals and waits per semaphore.

  Args:
    sched: The schedule.
    op_name: Name of the op in findings.
    skipped: Semaphores whose operations are only partly modeled.

  Returns:
    The findings, and whether a deadlock was proven.
  """
  detected: list[findings.VerificationFinding] = []
  reported: set[tuple[Any, ...]] = set()
  deadlock = False
  for sem in sorted(
      (set(sched.signals) | set(sched.waits)) - set(skipped),
      key=lambda k: (k.device_id, k.core_id, k.sem_id, k.sem_index),
  ):
    sigs = sched.signals.get(sem, [])
    waits = sched.waits.get(sem, [])
    total_sig = sum(s.amount for s in sigs)
    total_dec = sum(w.amount for w in waits if w.decrement)
    needed = max([total_dec] + [w.amount for w in waits])
    if total_sig < needed:
      deadlock = True
      category = _Category.SEMAPHORE_DEADLOCK
      message = (
          f"semaphore {sem.short_str()} deadlocks: waits require {needed}"
          f" signal(s)/bytes in total, but only {total_sig} are ever issued"
      )
      loc = waits[0].source_location if waits else None
    elif total_sig > total_dec:
      category = _Category.SEMAPHORE_NONZERO_EXIT
      message = (
          f"semaphore {sem.short_str()} exits the kernel with value"
          f" +{total_sig - total_dec} (signaled {total_sig}, waited"
          f" {total_dec}); TPU semaphores must return to 0 before the kernel"
          " exits"
      )
      loc = sigs[0].source_location if sigs else None
    else:
      continue
    key = (category, sem.sem_type, sem.sem_id, sem.sem_index)
    if key not in reported:
      reported.add(key)
      detected.append(_finding(category, message, op_name, source_location=loc))
  return detected, deadlock


class _Encoder:
  """Builds the difference-logic model of a schedule."""

  def __init__(self) -> None:
    self.model = idl_solver.IdlPbModel()
    self.vars: dict[str, int] = {}

  def var(self, point: _TimePoint) -> int:
    var = self.vars.get(point.key)
    if var is None:
      var = self.model.new_int_var(0)
      self.vars[point.key] = var
    return var

  def before(
      self,
      a: _TimePoint,
      b: _TimePoint,
      guard: idl_solver.Literal | None = None,
  ) -> None:
    """Adds `t[a] + 1 <= t[b]`."""
    self.model.add_diff(self.var(a), self.var(b), -1, guard)

  def not_after(
      self,
      a: _TimePoint,
      b: _TimePoint,
      guard: idl_solver.Literal | None = None,
  ) -> None:
    """Adds `t[a] <= t[b]`."""
    self.model.add_diff(self.var(a), self.var(b), 0, guard)


def _encode_semaphore(
    enc: _Encoder, signals: Sequence[_Signal], waits: Sequence[_Wait]
) -> None:
  """Encodes counting-semaphore semantics for the waits on one semaphore."""
  # order[(i, j)] states whether wait i completes before wait j (i < j).
  order: dict[tuple[int, int], idl_solver.Literal | bool] = {}
  for i, j in itertools.combinations(range(len(waits)), 2):
    rel = _static_order(waits[i].point, waits[j].point)
    if rel:
      order[(i, j)] = rel < 0
      continue
    lit = enc.model.new_bool_var()
    enc.before(waits[i].point, waits[j].point, lit)
    enc.before(waits[j].point, waits[i].point, lit.negated())
    order[(i, j)] = lit

  def wait_precedes(i: int, j: int) -> idl_solver.Literal | bool:
    if i < j:
      return order[(i, j)]
    rel = order[(j, i)]
    return (not rel) if isinstance(rel, bool) else rel.negated()

  for j, wait in enumerate(waits):
    # A wait completes once the signals before it cover its own amount plus
    # what earlier decrementing waits consumed.
    terms: list[tuple[int, idl_solver.Literal]] = []
    rhs = wait.amount
    for sig in signals:
      rel = _static_order(sig.point, wait.point)
      if rel < 0:
        rhs -= sig.amount
      elif rel == 0:
        lit = enc.model.new_bool_var()
        enc.before(sig.point, wait.point, lit)
        enc.not_after(wait.point, sig.point, lit.negated())
        terms.append((sig.amount, lit))
    for i, other in enumerate(waits):
      if i == j or not other.decrement:
        continue
      rel = wait_precedes(i, j)
      if rel is True:
        rhs += other.amount
      elif rel is not False:
        terms.append((-other.amount, rel))
    enc.model.add_pb_ge(terms, rhs)


def _encode_schedule(
    model: jaxpr_model.KernelVerificationModel, sched: _Schedule
) -> _Encoder:
  enc = _Encoder()
  for entry in sched.trace:
    enc.var(entry.point)
  for event_ids in model.actor_event_orders.values():
    for a, b in itertools.pairwise(event_ids):
      enc.before(_TimePoint(f"ev_{a}"), _TimePoint(f"ev_{b}"))
  for dma in model.dma_starts:
    read_done = _TimePoint(f"dma_r_{dma.dma_id}")
    enc.before(_TimePoint(f"ev_{dma.event_id}"), read_done)
    enc.not_after(read_done, _TimePoint(f"dma_w_{dma.dma_id}"))
  for sem, waits in sched.waits.items():
    _encode_semaphore(enc, sched.signals.get(sem, []), waits)
  return enc


def _physical_buffer(buf: jaxpr_model.BufferKey) -> tuple[int, int, str, str]:
  # HBM and VMEM_SHARED are shared by the cores of a chip; VMEM and SMEM are
  # private to a core.
  shared = buf.memory_space in _SHARED_MEMORY_SPACES
  return (
      buf.device_id,
      -1 if shared else buf.core_id,
      buf.memory_space,
      buf.buffer_id,
  )


def _race_candidates(sched: _Schedule) -> list[tuple[_Access, _Access]]:
  """Returns access pairs that conflict unless synchronization orders them."""
  by_buffer: dict[tuple[int, int, str, str], list[_Access]] = {}
  for access in sched.accesses:
    by_buffer.setdefault(_physical_buffer(access.buffer), []).append(access)
  pairs = []
  for group in by_buffer.values():
    for a, b in itertools.combinations(group, 2):
      if not (a.is_write or b.is_write):
        continue
      if a.dma is None and b.dma is None and a.actor == b.actor:
        continue  # Program order serializes synchronous accesses.
      if a.dma is not None and b.dma is not None and a.dma is b.dma:
        continue
      if (
          a.actor.device_id == b.actor.device_id
          and a.actor.parallel_coords != b.actor.parallel_coords
          and a.buffer.memory_space not in _SHARED_MEMORY_SPACES
      ):
        # Parallel grid iterations do not interleave on a core, and each core
        # has private VMEM/SMEM, so they only conflict on shared memory.
        continue
      if not a.region.intersects(b.region):
        continue
      if _static_order(a.end, b.start) < 0 or _static_order(b.end, a.start) < 0:
        continue
      pairs.append((a, b))
  return pairs


def _classify_race(
    a: _Access, b: _Access, model: jaxpr_model.KernelVerificationModel
) -> findings.FindingCategory:
  if (a.dma is not None and a.dma.is_remote) or (
      b.dma is not None and b.dma.is_remote
  ):
    return _Category.UNTHROTTLED_REMOTE_OVERWRITE
  if (
      a.actor.parallel_coords != b.actor.parallel_coords
      and a.actor.device_id == b.actor.device_id
  ):
    return _Category.PARALLEL_GRID_CONFLICT
  involved = {d.dma_id for d in (a.dma, b.dma) if d is not None}
  for dma in (a.dma, b.dma):
    if dma is None or dma.dst_sem is None:
      continue
    if any(
        other.dst_sem == dma.dst_sem and other.dma_id not in involved
        for other in model.dma_starts
    ):
      return _Category.CONCURRENT_DMA_SEM_SHARING
  return _Category.DATA_RACE


def _counterexample(
    sched: _Schedule, times: Mapping[str, int], horizon: int
) -> tuple[findings.CounterexampleStep, ...]:
  """Returns the schedule prefix up to `horizon` with compacted timestamps."""
  entries = sorted(
      (
          (times[e.point.key], e.device_id, e.actor, idx)
          for idx, e in enumerate(sched.trace)
          if times[e.point.key] <= horizon
      ),
  )
  ranks = {t: r for r, t in enumerate(sorted({t for t, *_ in entries}))}
  steps = []
  for t, _, _, idx in entries:
    entry = sched.trace[idx]
    steps.append(
        findings.CounterexampleStep(
            timestamp=ranks[t],
            device_id=entry.device_id,
            core_id=entry.core_id,
            actor=entry.actor,
            operation=entry.operation,
            detail=entry.detail,
            source_location=entry.source_location,
        )
    )
  return tuple(steps)


def _check_races(
    model: jaxpr_model.KernelVerificationModel,
    sched: _Schedule,
    enc: _Encoder,
    config: ModelCheckConfig,
) -> list[findings.VerificationFinding]:
  """Checks feasibility, then every race candidate under an assumption."""
  op_name = model.kernel_name
  queries = []
  for a, b in _race_candidates(sched):
    lit = enc.model.new_bool_var(phase=False)
    enc.not_after(a.start, b.end, lit)
    enc.not_after(b.start, a.end, lit)
    queries.append((a, b, lit))

  solver = idl_solver.Solver(enc.model)
  timeout = config.solver_timeout_seconds
  base = solver.solve(timeout_seconds=timeout)
  if base.status is _SolverStatus.INFEASIBLE and model.sync_truncated:
    return [
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            "no schedule of the checked iterations lets every wait complete,"
            " but the missing signals may come from unchecked iterations, so"
            " deadlocks and races were not checked (raise"
            " max_unrolled_iterations)",
            op_name,
            severity=findings.Severity.WARNING,
        )
    ]
  if base.status is _SolverStatus.INFEASIBLE:
    return [
        _finding(
            _Category.SEMAPHORE_DEADLOCK,
            "no deadlock-free schedule exists: the semaphore and remote-DMA"
            " wait dependencies form a cycle",
            op_name,
        )
    ]
  if base.status is _SolverStatus.UNKNOWN:
    return [
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            f"model checking timed out after {timeout:g}s before finding a"
            " schedule; deadlocks and races were not checked (increase"
            " solver_timeout_seconds)",
            op_name,
            severity=findings.Severity.WARNING,
        )
    ]

  detected: list[findings.VerificationFinding] = []
  reported: set[tuple[Any, ...]] = set()
  timeouts = 0
  for a, b, lit in queries:
    key = (
        a.buffer.memory_space,
        a.buffer.buffer_id,
        (a.dma is not None and a.dma.is_remote)
        or (b.dma is not None and b.dma.is_remote),
        bool(a.actor.parallel_coords or b.actor.parallel_coords),
    )
    if key in reported:
      continue
    result = solver.solve([lit], timeout_seconds=timeout)
    if result.status is _SolverStatus.UNKNOWN:
      timeouts += 1
      continue
    if result.status is _SolverStatus.INFEASIBLE:
      continue
    reported.add(key)
    times = {k: result.int_values[v] for k, v in enc.vars.items()}
    horizon = max(times[a.end.key], times[b.end.key])
    hazard = "WAW" if a.is_write and b.is_write else "RAW/WAR"
    detected.append(
        _finding(
            _classify_race(a, b, model),
            f"{hazard} race on {_buffer_str(a.buffer)} (dev"
            f" {a.buffer.device_id}) between `{a.detail}` and `{b.detail}`",
            op_name,
            source_location=a.source_location or b.source_location,
            counterexample=_counterexample(sched, times, horizon),
        )
    )
  if timeouts:
    detected.append(
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            f"{timeouts} race queries timed out after {timeout:g}s each and"
            " were not checked (increase solver_timeout_seconds)",
            op_name,
            severity=findings.Severity.WARNING,
        )
    )
  return detected


def _verify_model(
    model: jaxpr_model.KernelVerificationModel, config: ModelCheckConfig
) -> list[findings.VerificationFinding]:
  """Checks one extracted kernel model."""
  op_name = model.kernel_name
  detected = [
      _finding(
          issue.category,
          issue.message,
          op_name,
          severity=issue.severity,
          source_location=issue.source_location,
      )
      for issue in model.static_issues
  ]
  if model.incomplete_reasons:
    detected.append(
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            "the model is bounded or approximate, so findings may be missed: "
            + "; ".join(model.incomplete_reasons),
            op_name,
            severity=findings.Severity.INFO,
        )
    )
  sched = _collect_schedule(model)
  if not model.sync_reliable:
    detected.append(
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            "deadlock and race checks were skipped because some semaphore"
            " operations could not be resolved statically or only run"
            " depending on runtime data",
            op_name,
            severity=findings.Severity.WARNING,
        )
    )
    return detected
  # Bounded unrolling drops the signals and waits of later iterations, so the
  # values of their semaphores at kernel exit are unknown.
  conservation, deadlock = _check_conservation(
      sched, op_name, model.truncated_sem_keys
  )
  detected.extend(conservation)
  if deadlock or not (sched.waits or sched.accesses):
    return detected
  enc = _encode_schedule(model, sched)
  detected.extend(_check_races(model, sched, enc, config))
  return detected


def verify_kernel_models(
    models: Sequence[jaxpr_model.KernelVerificationModel],
    *,
    config: ModelCheckConfig = ModelCheckConfig(),
) -> findings.VerificationReport:
  """Model-checks extracted Pallas kernel models."""
  detected: list[findings.VerificationFinding] = []
  for model in models:
    detected.extend(_verify_model(model, config))
  return findings.VerificationReport.of(_PHASE, detected)


def check_kernel_concurrency(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    config: ModelCheckConfig = ModelCheckConfig(),
) -> findings.VerificationReport:
  """Extracts the Pallas kernels called by `op_or_fn` and model-checks them."""
  report, _ = check_kernel_concurrency_with_models(
      op_or_fn, example_args, example_kwargs, config=config
  )
  return report


def check_kernel_concurrency_with_models(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    config: ModelCheckConfig = ModelCheckConfig(),
) -> tuple[
    findings.VerificationReport,
    tuple[jaxpr_model.KernelVerificationModel, ...] | None,
]:
  """Like `check_kernel_concurrency`, but also returns the extracted models.

  Args:
    op_or_fn: The op or JAX function.
    example_args: Example positional arguments.
    example_kwargs: Example keyword arguments.
    config: Model checking configuration.

  Returns:
    The report, and the models (None if the kernel could not be traced).
  """
  try:
    models = jaxpr_model.extract_kernel_models(
        op_or_fn,
        example_args,
        example_kwargs,
        max_unrolled_iterations=config.max_unrolled_iterations,
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    report = findings.VerificationReport.of(
        _PHASE,
        [
            _finding(
                _Category.VERIFICATION_INCOMPLETE,
                "could not trace the kernel for model checking"
                f" ({type(exc).__name__}: {exc})",
                jaxpr_model.display_name(op_or_fn),
                severity=findings.Severity.WARNING,
            )
        ],
    )
    return report, None
  if not models:
    report = findings.VerificationReport.of(
        _PHASE,
        [
            _finding(
                _Category.VERIFICATION_INCOMPLETE,
                "no pallas_call was found, so there was nothing to model check",
                jaxpr_model.display_name(op_or_fn),
                severity=findings.Severity.INFO,
            )
        ],
    )
    return report, ()
  return verify_kernel_models(models, config=config), models
