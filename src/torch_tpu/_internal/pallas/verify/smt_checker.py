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
constraints over those timestamps. The checker then asks the Google OR-Tools
CP-SAT solver (`cp_model`):

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
from typing import Any, TYPE_CHECKING

from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import jaxpr_model

if TYPE_CHECKING:
  from ortools.sat.python import cp_model
else:
  try:
    from ortools.sat.python import cp_model
  except ImportError:
    cp_model = None

_PHASE = findings.Phase.MODEL_CHECK
_Category = findings.FindingCategory
# Memory spaces shared by all cores of a chip; VMEM and SMEM are per core.
_SHARED_MEMORY_SPACES = frozenset({"hbm", "vmem_shared"})
# Cap on conflicting access pairs checked by the solver per kernel model.
_MAX_RACE_CANDIDATES = 2048
# Default time budget per CP-SAT solver query.
DEFAULT_SOLVER_TIMEOUT_SECONDS = 5.0


@dataclasses.dataclass(frozen=True)
class ModelCheckConfig:
  """Bounds for kernel model extraction and solving.

  Attributes:
    max_unrolled_iterations: Bound on unrolled grid points, loop iterations, and
      scan steps per kernel.
    solver_timeout_seconds: Time budget for each solver query.
    tpu_device_kind: TPU device kind used when synthesizing a compile-only
      abstract mesh for kernels that require a TPU context.
  """

  max_unrolled_iterations: int = 32
  solver_timeout_seconds: float = DEFAULT_SOLVER_TIMEOUT_SECONDS
  tpu_device_kind: str = findings.DEFAULT_TPU_DEVICE_KIND


@dataclasses.dataclass(frozen=True)
class _TimePoint:
  """A schedule timestamp: an actor event or a DMA completion."""

  key: str
  actor: jaxpr_model.ActorKey | None = None
  position: int = -1
  # For DMA completions: the actor and program position of the DMA start.
  started_by: tuple[jaxpr_model.ActorKey, int] | None = None
  # For DMA completions: earliest (actor, position) wait that unconditionally
  # awaits this completion.
  completed_by: tuple[jaxpr_model.ActorKey, int] | None = None


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
  if a.completed_by is not None:
    actor, end = a.completed_by
    if b.actor == actor and end < b.position:
      return -1  # a < wait < b.
    if (
        b.started_by is not None
        and b.started_by[0] == actor
        and end < b.started_by[1]
    ):
      return -1  # a < wait < b_start < b.
  if b.completed_by is not None:
    actor, end = b.completed_by
    if a.actor == actor and end < a.position:
      return 1  # b < wait < a.
    if (
        a.started_by is not None
        and a.started_by[0] == actor
        and end < a.started_by[1]
    ):
      return 1  # b < wait < a_start < a.
  return 0


@dataclasses.dataclass(frozen=True)
class _Signal:
  point: _TimePoint
  sem: jaxpr_model.SemaphoreKey
  amount: int
  source_location: str | None
  dma_start_point: _TimePoint | None = None


@dataclasses.dataclass(frozen=True)
class _Wait:
  point: _TimePoint
  sem: jaxpr_model.SemaphoreKey
  amount: int
  _: dataclasses.KW_ONLY
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
  _: dataclasses.KW_ONLY
  is_write: bool
  dma: jaxpr_model.DmaStartEvent | None
  detail: str
  source_location: str | None
  speculative: bool = False


@dataclasses.dataclass(frozen=True)
class _RaceCandidate:
  """A pair of potentially conflicting memory accesses encoded in CP-SAT."""

  first: _Access
  second: _Access
  exact_overlap: bool
  buf_name: str
  category: findings.FindingCategory
  bucket_key: tuple[Any, ...]
  lit: Any


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
            is_write=mem.is_write,
            dma=None,
            detail=detail,
            source_location=mem.source_location,
            speculative=mem.speculative,
        )
    )
    sched.add_trace(p, mem.actor, op, detail, mem.source_location)

  for dma in model.dma_starts:
    _collect_dma(dma, point(dma.event_id), sched)

  for dwait in model.dma_waits:
    p = point(dwait.event_id)
    sched.waits.setdefault(dwait.sem, []).append(
        _Wait(
            p,
            dwait.sem,
            dwait.byte_count,
            decrement=True,
            source_location=dwait.source_location,
        )
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
            p,
            swait.sem,
            swait.amount,
            decrement=swait.decrement,
            source_location=swait.source_location,
        )
    )
    sign = "-" if swait.decrement else ">="
    detail = f"wait {swait.sem.short_str()} ({sign}{swait.amount})"
    sched.add_trace(p, swait.actor, "sem_wait", detail, swait.source_location)
  _propagate_forced_completions(sched)
  return sched


def _propagate_forced_completions(sched: _Schedule) -> None:
  """Attaches `completed_by` to DMA completion points with forced waits."""
  completed_by: dict[str, tuple[jaxpr_model.ActorKey, int]] = {}
  for sem, waits in sched.waits.items():
    if any(not w.decrement or w.point.actor is None for w in waits):
      continue
    if any(
        _static_order(w1.point, w2.point) == 0
        for w1, w2 in itertools.combinations(waits, 2)
    ):
      continue
    signals = sched.signals.get(sem, [])
    ordered_waits = sorted(waits, key=lambda w: w.point.position)
    rhs = 0
    for wait in ordered_waits:
      rhs += wait.amount
      possible = [
          sig for sig in signals if _static_order(sig.point, wait.point) <= 0
      ]
      if not all(
          _static_order(sig.point, wait.point) < 0
          or (
              sig.dma_start_point is not None
              and _static_order(sig.dma_start_point, wait.point) < 0
          )
          for sig in possible
      ):
        break
      total_possible = sum(sig.amount for sig in possible)
      if total_possible < rhs:
        break
      if total_possible == rhs and wait.point.actor is not None:
        bound = (wait.point.actor, wait.point.position)
        for sig in possible:
          if sig.point.actor is None:
            prev = completed_by.get(sig.point.key)
            if prev is None or bound[1] < prev[1]:
              completed_by[sig.point.key] = bound
            if sig.point.key.startswith("dma_w_"):
              r_key = "dma_r_" + sig.point.key.removeprefix("dma_w_")
              r_prev = completed_by.get(r_key)
              if r_prev is None or bound[1] < r_prev[1]:
                completed_by[r_key] = bound
  if not completed_by:
    return

  def _with_bound(pt: _TimePoint) -> _TimePoint:
    bound = completed_by.get(pt.key)
    return dataclasses.replace(pt, completed_by=bound) if bound else pt

  for s_list in sched.signals.values():
    for idx, sig in enumerate(s_list):
      if sig.point.key in completed_by:
        s_list[idx] = dataclasses.replace(sig, point=_with_bound(sig.point))
  for idx, acc in enumerate(sched.accesses):
    if acc.end.key in completed_by:
      sched.accesses[idx] = dataclasses.replace(acc, end=_with_bound(acc.end))
  for idx, entry in enumerate(sched.trace):
    if entry.point.key in completed_by:
      sched.trace[idx] = dataclasses.replace(
          entry, point=_with_bound(entry.point)
      )


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
    if dma.dst_sem.sem_type == "regular":
      done += f" & signal {dma.dst_sem.short_str()} (+1)"
      sig_amount = 1
    else:
      done += f" & signal {dma.dst_sem.short_str()} (+{dma.byte_count}B)"
      sig_amount = dma.byte_count
    sched.signals.setdefault(dma.dst_sem, []).append(
        _Signal(
            write_done,
            dma.dst_sem,
            sig_amount,
            dma.source_location,
            dma_start_point=start,
        )
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
    if dma.src_sem.sem_type == "regular":
      src_sig_str = "+1"
      src_sig_amount = 1
    else:
      src_sig_str = f"+{dma.byte_count}B"
      src_sig_amount = dma.byte_count
    sched.trace.append(
        _TraceEntry(
            read_done,
            src.device_id,
            src.core_id,
            engine,
            f"{kind}_read_done",
            f"finish reading & signal {dma.src_sem.short_str()}"
            f" ({src_sig_str})",
            dma.source_location,
        )
    )
    sched.signals.setdefault(dma.src_sem, []).append(
        _Signal(
            read_done,
            dma.src_sem,
            src_sig_amount,
            dma.source_location,
            dma_start_point=start,
        )
    )
  sched.accesses.append(
      _Access(
          start,
          read_done,
          dma.actor,
          src,
          dma.src_region,
          is_write=False,
          dma=dma,
          detail=f"{engine} reading {src.buffer_id}{dma.src_region}",
          source_location=dma.source_location,
      )
  )
  sched.accesses.append(
      _Access(
          start,
          write_done,
          dma.actor,
          dst,
          dma.dst_region,
          is_write=True,
          dma=dma,
          detail=(
              f"{engine} (from dev{dma.actor.device_id}) writing"
              f" {dst.buffer_id}{dma.dst_region}"
          ),
          source_location=dma.source_location,
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
      key=lambda k: (
          k.device_id,
          k.core_id,
          k.sem_id,
          k.sem_index,
          k.parallel_coords,
      ),
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

  def __init__(self, max_time: int) -> None:
    self.model = cp_model.CpModel()
    self._max_time = max(max_time, 1)
    self._bool_count = 0
    self.vars: dict[str, cp_model.IntVar] = {}

  def new_bool_var(self) -> cp_model.IntVar:
    idx = self._bool_count
    self._bool_count += 1
    return self.model.new_bool_var(f"b_{idx}")

  def var(self, point: _TimePoint) -> cp_model.IntVar:
    var = self.vars.get(point.key)
    if var is None:
      var = self.model.new_int_var(0, self._max_time, point.key)
      self.vars[point.key] = var
    return var

  def before(
      self,
      a: _TimePoint,
      b: _TimePoint,
      guard: cp_model.BoolVarT | None = None,
  ) -> None:
    """Adds `t[a] + 1 <= t[b]`."""
    ct = self.model.add(self.var(a) + 1 <= self.var(b))
    if guard is not None:
      ct.only_enforce_if(guard)

  def not_after(
      self,
      a: _TimePoint,
      b: _TimePoint,
      guard: cp_model.BoolVarT | None = None,
  ) -> None:
    """Adds `t[a] <= t[b]`."""
    ct = self.model.add(self.var(a) <= self.var(b))
    if guard is not None:
      ct.only_enforce_if(guard)

  def add_pb_ge(
      self,
      terms: Sequence[tuple[int, cp_model.BoolVarT]],
      rhs: int,
  ) -> None:
    """Adds `sum(coeff * lit for coeff, lit in terms) >= rhs`."""
    if not terms:
      if rhs > 0:
        self.model.add_bool_or([])
      return
    lits = [lit for _, lit in terms]
    coeffs = [coeff for coeff, _ in terms]
    self.model.add(cp_model.LinearExpr.weighted_sum(lits, coeffs) >= rhs)


def _encode_semaphore(
    enc: _Encoder, signals: Sequence[_Signal], waits: Sequence[_Wait]
) -> None:
  """Encodes counting-semaphore semantics for the waits on one semaphore."""
  # order[(i, j)] states whether wait i completes before wait j (i < j).
  order: dict[tuple[int, int], cp_model.BoolVarT | bool] = {}
  for i, j in itertools.combinations(range(len(waits)), 2):
    rel = _static_order(waits[i].point, waits[j].point)
    if rel:
      order[(i, j)] = rel < 0
      continue
    lit = enc.new_bool_var()
    enc.before(waits[i].point, waits[j].point, lit)
    enc.before(waits[j].point, waits[i].point, lit.negated())
    order[(i, j)] = lit

  def wait_precedes(i: int, j: int) -> cp_model.BoolVarT | bool:
    if i < j:
      return order[(i, j)]
    rel = order[(j, i)]
    return (not rel) if isinstance(rel, bool) else rel.negated()

  amounts = {s.amount for s in signals} | {w.amount for w in waits}
  shared_dma_sem = (
      len(signals) > 1
      and len(amounts) > 1
      and any(
          s.sem.sem_type == "dma"
          and s.dma_start_point is not None
          and s.amount > 1
          for s in signals
      )
  )
  for j, wait in enumerate(waits):
    # A wait completes once the signals before it cover its own amount plus
    # what earlier decrementing waits consumed.
    terms: list[tuple[int, cp_model.BoolVarT]] = []
    rhs = wait.amount
    unordered_wait_terms: list[tuple[int, cp_model.BoolVarT]] = []
    for i, other in enumerate(waits):
      if i == j or not other.decrement:
        continue
      rel = wait_precedes(i, j)
      if rel is True:
        rhs += other.amount
      elif rel is not False:
        unordered_wait_terms.append((-other.amount, rel))

    ambiguous_signals: list[_Signal] = []
    for sig in signals:
      rel = _static_order(sig.point, wait.point)
      if rel < 0:
        rhs -= sig.amount
      elif rel == 0:
        ambiguous_signals.append(sig)

    if (
        not unordered_wait_terms
        and rhs > 0
        and sum(sig.amount for sig in ambiguous_signals) == rhs
    ):
      for sig in ambiguous_signals:
        enc.before(sig.point, wait.point)
      continue

    for sig in ambiguous_signals:
      if (
          shared_dma_sem
          and sig.sem.sem_type == "dma"
          and sig.dma_start_point is not None
          and sig.amount > 1
      ):
        # A DMA semaphore increments progressively as bytes arrive: once a DMA
        # has started (`dma_start_point < wait.point`), up to `sig.amount - 1`
        # bytes may arrive before the final byte completes at `sig.point`.
        start_rel = _static_order(sig.dma_start_point, wait.point)
        if start_rel < 0:
          rhs -= sig.amount - 1
        elif start_rel == 0:
          start_lit = enc.new_bool_var()
          enc.before(sig.dma_start_point, wait.point, start_lit)
          enc.not_after(wait.point, sig.dma_start_point, start_lit.negated())
          terms.append((sig.amount - 1, start_lit))
        done_lit = enc.new_bool_var()
        enc.before(sig.point, wait.point, done_lit)
        enc.not_after(wait.point, sig.point, done_lit.negated())
        terms.append((1, done_lit))
      else:
        lit = enc.new_bool_var()
        enc.before(sig.point, wait.point, lit)
        enc.not_after(wait.point, sig.point, lit.negated())
        terms.append((sig.amount, lit))
    terms.extend(unordered_wait_terms)
    enc.add_pb_ge(terms, rhs)


def _encode_schedule(
    model: jaxpr_model.KernelVerificationModel,
    sched: _Schedule,
    *,
    skipped_waits: Collection[jaxpr_model.SemaphoreKey] = (),
) -> _Encoder:
  max_time = max(len(sched.trace) * 2 + 16, 64)
  enc = _Encoder(max_time)
  for entry in sched.trace:
    enc.var(entry.point)
  for event_ids in model.actor_event_orders.values():
    for a, b in itertools.pairwise(event_ids):
      enc.before(_TimePoint(f"ev_{a}"), _TimePoint(f"ev_{b}"))
  for dma in model.dma_starts:
    read_done = _TimePoint(f"dma_r_{dma.dma_id}")
    enc.before(_TimePoint(f"ev_{dma.event_id}"), read_done)
    enc.not_after(read_done, _TimePoint(f"dma_w_{dma.dma_id}"))
  skipped_set = set(skipped_waits)
  for sem, waits in sched.waits.items():
    if sem in skipped_set:
      continue
    _encode_semaphore(enc, sched.signals.get(sem, []), waits)
  return enc


def _physical_buffer(
    buf: jaxpr_model.BufferKey,
) -> tuple[int, int, str, str, tuple[int, ...]]:
  # HBM, VMEM_SHARED, and buffers with empty `owner_coords` (`collective_axes`)
  # are shared across the cores of a chip; per-core VMEM and SMEM buffers are
  # keyed by their owning core coordinates (`owner_coords`).
  shared = buf.memory_space in _SHARED_MEMORY_SPACES or not buf.owner_coords
  return (
      buf.device_id,
      -1 if shared else buf.core_id,
      buf.memory_space,
      buf.buffer_id,
      () if shared else buf.owner_coords,
  )


def _race_candidates(sched: _Schedule) -> list[tuple[_Access, _Access]]:
  """Returns access pairs that conflict unless synchronization orders them."""
  by_buffer: dict[tuple[int, int, str, str, tuple[int, ...]], list[_Access]] = (
      {}
  )
  for access in sched.accesses:
    by_buffer.setdefault(_physical_buffer(access.buffer), []).append(access)
  point_order = {entry.point.key: idx for idx, entry in enumerate(sched.trace)}
  pairs: list[tuple[_Access, _Access]] = []
  seen_signatures: set[tuple[Any, ...]] = set()
  for group in by_buffer.values():
    for a, b in itertools.combinations(group, 2):
      if not (a.is_write or b.is_write):
        continue
      if a.dma is None and b.dma is None and a.actor == b.actor:
        continue  # Program order serializes synchronous accesses.
      if a.dma is not None and b.dma is not None and a.dma is b.dma:
        continue
      if not a.region.intersects(b.region):
        continue
      if _static_order(a.end, b.start) < 0 or _static_order(b.end, a.start) < 0:
        continue
      sig = (
          a.start.key,
          a.end.key,
          b.start.key,
          b.end.key,
          a.buffer,
          b.buffer,
          a.region,
          b.region,
          a.is_write,
          b.is_write,
      )
      if sig in seen_signatures:
        continue
      seen_signatures.add(sig)
      pairs.append((a, b))
  # Prioritize exact-overlap pairs over widened/approximate pairs, and order by
  # earliest schedule position and proximity so bounded candidate slicing checks
  # early/adjacent step pairs across all access kinds (including DMA-DMA pairs).
  pairs.sort(
      key=lambda pair: (
          not pair[0].region.definitely_intersects(pair[1].region),
          max(
              point_order.get(pair[0].start.key, 0),
              point_order.get(pair[1].start.key, 0),
          ),
          abs(
              point_order.get(pair[0].start.key, 0)
              - point_order.get(pair[1].start.key, 0)
          ),
      )
  )
  return pairs


def _initial_race_category(
    a: _Access, b: _Access, model: jaxpr_model.KernelVerificationModel
) -> findings.FindingCategory:
  if (a.dma is not None and a.dma.is_remote and a.is_write) or (
      b.dma is not None and b.dma.is_remote and b.is_write
  ):
    return _Category.UNTHROTTLED_REMOTE_OVERWRITE
  if (
      not model.core_axis_names
      and a.actor.parallel_coords != b.actor.parallel_coords
      and a.actor.device_id == b.actor.device_id
  ):
    return _Category.PARALLEL_GRID_CONFLICT
  return _Category.DATA_RACE


def _refine_race_category(
    a: _Access,
    b: _Access,
    initial_category: findings.FindingCategory,
    model: jaxpr_model.KernelVerificationModel,
    times: Mapping[str, int],
) -> findings.FindingCategory:
  """Refines `DATA_RACE` to `CONCURRENT_DMA_SEM_SHARING` when a shared-semaphore DMA overlaps in `times`."""
  if initial_category != _Category.DATA_RACE:
    return initial_category
  involved = {d.dma_id for d in (a.dma, b.dma) if d is not None}
  for acc in (a, b):
    dma = acc.dma
    if dma is None:
      continue
    acc_start = times.get(acc.start.key)
    acc_end = times.get(acc.end.key)
    if acc_start is None or acc_end is None:
      continue
    sems = [s for s in (dma.dst_sem, dma.src_sem) if s is not None]
    for other in model.dma_starts:
      if other.dma_id in involved:
        continue
      if other.dst_sem in sems:
        other_end_key = f"dma_w_{other.dma_id}"
      elif other.src_sem in sems:
        other_end_key = f"dma_r_{other.dma_id}"
      else:
        continue
      other_start = times.get(f"ev_{other.event_id}")
      other_end = times.get(other_end_key)
      if (
          other_start is not None
          and other_end is not None
          and other_start <= acc_end
          and acc_start <= other_end
      ):
        return _Category.CONCURRENT_DMA_SEM_SHARING
  return initial_category


def _check_parallel_grid_conflicts(
    model: jaxpr_model.KernelVerificationModel,
    sched: _Schedule,
) -> list[findings.VerificationFinding]:
  """Detects parallel-grid write conflicts in `pallas_call` without requiring CP-SAT."""
  if model.core_axis_names or "parallel" not in model.dimension_semantics:
    return []
  detected: list[findings.VerificationFinding] = []
  seen_buckets: set[tuple[str, str]] = set()
  by_buffer: dict[tuple[int, int, str, str, tuple[int, ...]], list[_Access]] = (
      {}
  )
  for access in sched.accesses:
    by_buffer.setdefault(_physical_buffer(access.buffer), []).append(access)
  for group in by_buffer.values():
    for a, b in itertools.combinations(group, 2):
      if not (a.is_write or b.is_write):
        continue
      if (
          a.actor.device_id != b.actor.device_id
          or a.actor.parallel_coords == b.actor.parallel_coords
      ):
        continue
      if not a.region.definitely_intersects(b.region):
        continue
      bucket = (a.buffer.memory_space, a.buffer.buffer_id)
      if bucket in seen_buckets:
        continue
      seen_buckets.add(bucket)
      buf_name = _buffer_str(a.buffer)
      hazard = "WAW" if a.is_write and b.is_write else "RAW/WAR"
      severity = (
          findings.Severity.WARNING
          if (a.speculative or b.speculative)
          else findings.Severity.ERROR
      )
      detected.append(
          _finding(
              _Category.PARALLEL_GRID_CONFLICT,
              f"{hazard} race on {buf_name} (dev"
              f" {a.buffer.device_id}) between `{a.detail}` and `{b.detail}`",
              model.kernel_name,
              severity=severity,
              source_location=a.source_location or b.source_location,
          )
      )
  return detected


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


def _solve(
    solver: Any,
    model: Any,
    timeout_seconds: float,
) -> Any:
  if timeout_seconds <= 0.0:
    return cp_model.UNKNOWN
  solver.parameters.max_time_in_seconds = float(timeout_seconds)
  if hasattr(solver.parameters, "num_workers"):
    solver.parameters.num_workers = 1
  else:
    solver.parameters.num_search_workers = 1
  return solver.solve(model)


def _deadlock_finding(
    sched: _Schedule, op_name: str
) -> findings.VerificationFinding:
  """Formats a `SEMAPHORE_DEADLOCK` finding with waited semaphore details."""
  waited_names: list[str] = []
  first_loc: str | None = None
  for sem, waits in sched.waits.items():
    if not waits:
      continue
    name = sem.short_str()
    if name not in waited_names:
      waited_names.append(name)
    if first_loc is None:
      for w in waits:
        if w.source_location is not None:
          first_loc = w.source_location
          break
  sem_detail = (
      f" (waited semaphores: {', '.join(waited_names[:4])})"
      if waited_names
      else ""
  )
  return _finding(
      _Category.SEMAPHORE_DEADLOCK,
      "no deadlock-free schedule exists: some semaphore or DMA waits"
      f" cannot all be satisfied{sem_detail}",
      op_name,
      source_location=first_loc,
  )


def _under_signaled_truncated_sems(
    model: jaxpr_model.KernelVerificationModel,
    sched: _Schedule,
) -> set[jaxpr_model.SemaphoreKey]:
  """Returns semaphores whose waits may depend on signals in unrolled tails."""
  candidates = (
      set(sched.waits)
      if model.unwalked_sync
      else (set(sched.waits) & set(model.truncated_sem_keys))
  )
  if len(model.actor_event_orders) > 1 or model.uses_remote_operations:
    return set(candidates)
  under_signaled: set[jaxpr_model.SemaphoreKey] = set()
  for sem in candidates:
    sigs = sched.signals.get(sem, [])
    waits = sched.waits.get(sem, [])
    total_sig = sum(s.amount for s in sigs)
    total_dec = sum(w.amount for w in waits if w.decrement)
    needed = max([total_dec] + [w.amount for w in waits])
    if total_sig < needed:
      under_signaled.add(sem)
  return under_signaled


def _check_races(
    model: jaxpr_model.KernelVerificationModel,
    sched: _Schedule,
    enc: _Encoder,
    config: ModelCheckConfig,
) -> list[findings.VerificationFinding]:
  """Checks feasibility, then every race candidate under an assumption."""
  op_name = model.kernel_name
  solver = cp_model.CpSolver()
  timeout = config.solver_timeout_seconds
  detected: list[findings.VerificationFinding] = []
  base_status = _solve(solver, enc.model, timeout)
  if base_status == cp_model.MODEL_INVALID:
    return [
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            "CP-SAT reported an invalid schedule model; deadlocks and races"
            " were not checked",
            op_name,
            severity=findings.Severity.WARNING,
        )
    ]
  if base_status == cp_model.INFEASIBLE and model.sync_truncated:
    relax_sems = _under_signaled_truncated_sems(model, sched)
    if not relax_sems:
      return [_deadlock_finding(sched, op_name)]
    relaxed_enc = _encode_schedule(model, sched, skipped_waits=relax_sems)
    relaxed_status = _solve(solver, relaxed_enc.model, timeout)
    if relaxed_status == cp_model.INFEASIBLE:
      return [_deadlock_finding(sched, op_name)]
    detected.append(
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            "no schedule of the checked iterations lets every wait complete,"
            " but the missing signals may come from unchecked iterations, so"
            " full deadlock checking was skipped (raise"
            " max_unrolled_iterations)",
            op_name,
            severity=findings.Severity.WARNING,
        )
    )
    if relaxed_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
      return detected
    enc = relaxed_enc
  elif base_status == cp_model.INFEASIBLE:
    return [_deadlock_finding(sched, op_name)]
  elif base_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
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

  resolved_buckets: set[tuple[Any, ...]] = set()
  approx_buffers: list[str] = []
  timeouts = 0
  unchecked_candidates = 0
  unchecked_buffers = 0
  all_candidates = _race_candidates(sched)
  total_candidates = len(all_candidates)
  candidates = all_candidates[:_MAX_RACE_CANDIDATES]
  cand_entries: list[_RaceCandidate] = []
  bucket_lits: dict[tuple[Any, ...], list[cp_model.IntVar]] = {}
  for a, b in candidates:
    exact_overlap = a.region.definitely_intersects(b.region)
    buf_name = _buffer_str(a.buffer)
    category = _initial_race_category(a, b, model)
    bucket_key: tuple[Any, ...] = (
        ("exact", category, a.buffer.memory_space, a.buffer.buffer_id)
        if exact_overlap
        else ("approx", buf_name)
    )
    lit = enc.new_bool_var()
    enc.not_after(a.start, b.end, lit)
    enc.not_after(b.start, a.end, lit)
    cand_entries.append(
        _RaceCandidate(
            first=a,
            second=b,
            exact_overlap=exact_overlap,
            buf_name=buf_name,
            category=category,
            bucket_key=bucket_key,
            lit=lit,
        )
    )
    bucket_lits.setdefault(bucket_key, []).append(lit)

  while True:
    active = [
        entry
        for entry in cand_entries
        if entry.bucket_key not in resolved_buckets
    ]
    if not active:
      break
    any_race = enc.new_bool_var()
    enc.model.add_bool_or(
        [entry.lit for entry in active] + [any_race.negated()]
    )
    enc.model.clear_assumptions()
    enc.model.add_assumption(any_race)
    status = _solve(solver, enc.model, timeout)
    if status == cp_model.INFEASIBLE:
      break
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
      timeouts += 1
      unchecked_candidates = len(active)
      unchecked_buffers = len({entry.bucket_key for entry in active})
      break
    times = {k: solver.value(v) for k, v in enc.vars.items()}
    for entry in active:
      if entry.bucket_key in resolved_buckets or not solver.value(entry.lit):
        continue
      resolved_buckets.add(entry.bucket_key)
      for other_lit in bucket_lits[entry.bucket_key]:
        enc.model.add(other_lit == 0)
      if not entry.exact_overlap:
        approx_buffers.append(entry.buf_name)
        continue
      a, b = entry.first, entry.second
      final_category = _refine_race_category(a, b, entry.category, model, times)
      horizon = max(times[a.end.key], times[b.end.key])
      hazard = "WAW" if a.is_write and b.is_write else "RAW/WAR"
      severity = (
          findings.Severity.WARNING
          if final_category == _Category.PARALLEL_GRID_CONFLICT
          and (a.speculative or b.speculative)
          else findings.Severity.ERROR
      )
      detected.append(
          _finding(
              final_category,
              f"{hazard} race on {entry.buf_name} (dev"
              f" {a.buffer.device_id}) between `{a.detail}` and `{b.detail}`",
              op_name,
              severity=severity,
              source_location=a.source_location or b.source_location,
              counterexample=_counterexample(sched, times, horizon),
          )
      )
  if total_candidates > _MAX_RACE_CANDIDATES:
    detected.append(
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            f"only the first {_MAX_RACE_CANDIDATES} of {total_candidates}"
            " conflicting access pairs were checked for races",
            op_name,
            severity=findings.Severity.WARNING,
        )
    )
  if approx_buffers:
    detected.append(
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            "concurrent accesses use runtime-dependent slice indices on "
            f"{', '.join(approx_buffers)}, so spatial disjointness could not"
            " be proven statically",
            op_name,
            severity=findings.Severity.INFO,
        )
    )
  if timeouts:
    detected.append(
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            f"{timeouts} race query timed out after {timeout:g}s leaving"
            f" {unchecked_candidates} candidate pair(s) across"
            f" {unchecked_buffers} buffer bucket(s) unchecked (increase"
            " solver_timeout_seconds)",
            op_name,
            severity=findings.Severity.WARNING,
        )
    )
  return detected


def _verify_model(
    model: jaxpr_model.KernelVerificationModel,
    config: ModelCheckConfig,
    *,
    shared_cross_kernel_sems: Collection[str] = (),
) -> list[findings.VerificationFinding]:
  """Checks one extracted kernel model for static, conservation, deadlock, and race bugs."""
  op_name = model.kernel_name
  detected = jaxpr_model.static_model_findings(model)
  sched = _collect_schedule(model)
  if not model.sync_reliable:
    detected.extend(_check_parallel_grid_conflicts(model, sched))
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
  # values of their semaphores at kernel exit are unknown. When synchronization
  # equations exist in unwalked loop/grid tails (`unwalked_sync`), skip global
  # conservation altogether so unwalked cleanup steps never cause false errors.
  # Semaphores that are waited on but never signaled at any walked grid point
  # (`not in model.signaled_sem_keys`) are still checked for zero-signal
  # deadlock when `not model.unwalked_sync`.
  deadlock = False
  if not model.unwalked_sync:
    skipped_sems = {
        sem
        for sem in model.truncated_sem_keys
        if sem in model.signaled_sem_keys
    } | {
        sem
        for sem in set(sched.signals) | set(sched.waits)
        if sem.sem_id in shared_cross_kernel_sems
    }
    conservation, deadlock = _check_conservation(sched, op_name, skipped_sems)
    detected.extend(conservation)
  if deadlock or not (sched.waits or sched.accesses):
    if deadlock:
      detected.extend(_check_parallel_grid_conflicts(model, sched))
    return detected
  if cp_model is None:
    detected.extend(_check_parallel_grid_conflicts(model, sched))
    detected.append(
        _finding(
            _Category.VERIFICATION_INCOMPLETE,
            "OR-Tools CP-SAT solver is not installed (install with `pip install"
            ' "ortools>=9.15"`), so schedule-feasibility and data-race checks'
            " were skipped",
            op_name,
            severity=findings.Severity.WARNING,
        )
    )
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
  barrier_model_counts: dict[str, int] = {}
  if len(models) > 1:
    for model in models:
      seen_in_model = {
          sig.target_sem.sem_id
          for sig in model.sem_signals
          if sig.target_sem.sem_type == "barrier"
      } | {
          wait.sem.sem_id
          for wait in model.sem_waits
          if wait.sem.sem_type == "barrier"
      }
      for sem_id in seen_in_model:
        barrier_model_counts[sem_id] = barrier_model_counts.get(sem_id, 0) + 1
  shared_cross_kernel_sems = {
      sem_id for sem_id, count in barrier_model_counts.items() if count > 1
  }
  detected: list[findings.VerificationFinding] = []
  for model in models:
    detected.extend(
        _verify_model(
            model,
            config,
            shared_cross_kernel_sems=shared_cross_kernel_sems,
        )
    )
  return findings.report_of(_PHASE, detected)


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
  """Extracts the Pallas kernels called by `op_or_fn`, model-checks them, and returns the models.

  Args:
    op_or_fn: The op or JAX function.
    example_args: Example positional arguments.
    example_kwargs: Example keyword arguments.
    config: Model checking configuration.

  Returns:
    The report, and the models (None if the kernel could not be traced).
  """
  static_report, models = jaxpr_model.check_kernel_static_models(
      op_or_fn,
      example_args,
      example_kwargs,
      max_unrolled_iterations=config.max_unrolled_iterations,
      tpu_device_kind=config.tpu_device_kind,
  )
  if not models:
    return static_report, models
  return verify_kernel_models(models, config=config), models
