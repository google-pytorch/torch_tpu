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

"""Pure-Python CDCL(T) solver for difference logic with pseudo-Boolean terms.

The Pallas model checker reduces "is there a deadlock-free schedule?" and "can
these two memory accesses overlap?" to satisfiability of

* integer difference constraints `t[u] - t[v] <= c`, each optionally guarded by
  a Boolean literal (the constraint only holds when the literal is true),
* Boolean clauses, and
* linear pseudo-Boolean constraints `sum(w_i * lit_i) >= rhs`.

The solver combines conflict-driven clause learning (two watched literals,
first-UIP learning, activity-based branching, phase saving, and Luby restarts)
with counter-based pseudo-Boolean propagation and an incremental negative-cycle
check over the active difference constraints. Learned clauses are kept across
`Solver.solve` calls, so many queries that differ only in their assumptions are
cheap. It only depends on the Python standard library.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import heapq
import time

_UNASSIGNED = -1
_ASSUMPTION_FAILED = -2
_RESTART_BASE = 64
_ACTIVITY_DECAY = 0.95
_ACTIVITY_RESCALE_LIMIT = 1e100


class SolverStatus(enum.Enum):
  """Outcome of a solve call."""

  FEASIBLE = "feasible"
  INFEASIBLE = "infeasible"
  UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class Literal:
  """A Boolean variable (`positive=True`) or its negation."""

  var_id: int
  positive: bool = True

  def negated(self) -> Literal:
    """Returns the logical negation of this literal."""
    return Literal(self.var_id, not self.positive)


@dataclasses.dataclass(frozen=True)
class DiffEdge:
  """Difference constraint `t[src] - t[dst] <= weight`, optionally guarded."""

  src: int
  dst: int
  weight: int
  guard: Literal | None = None


@dataclasses.dataclass(frozen=True)
class PbConstraint:
  """Normalized constraint `sum(weight * lit) >= rhs` with positive weights."""

  terms: tuple[tuple[int, Literal], ...]
  rhs: int


@dataclasses.dataclass(frozen=True)
class SolveResult:
  """Outcome of a solve call; the values are only set when feasible."""

  status: SolverStatus
  int_values: Mapping[int, int] = dataclasses.field(default_factory=dict)
  bool_values: Mapping[int, bool] = dataclasses.field(default_factory=dict)


class IdlPbModel:
  """Builder for difference-logic, clause, and pseudo-Boolean constraints.

  Integer variable 0 is the origin with the fixed value 0; `new_int_var`
  returns the ids of all other integer variables.
  """

  def __init__(self) -> None:
    self._num_int_vars = 1
    self._phases: list[bool] = []
    self._edges: list[DiffEdge] = []
    self._clauses: list[tuple[Literal, ...]] = []
    self._pb_constraints: list[PbConstraint] = []

  @property
  def num_int_vars(self) -> int:
    """Number of integer variables, including the origin."""
    return self._num_int_vars

  @property
  def num_bool_vars(self) -> int:
    """Number of Boolean variables."""
    return len(self._phases)

  @property
  def default_phases(self) -> Sequence[bool]:
    """Value tried first when branching on each Boolean variable."""
    return self._phases

  @property
  def edges(self) -> Sequence[DiffEdge]:
    """All difference constraints, including variable bounds."""
    return self._edges

  @property
  def clauses(self) -> Sequence[tuple[Literal, ...]]:
    """All clauses."""
    return self._clauses

  @property
  def pb_constraints(self) -> Sequence[PbConstraint]:
    """All pseudo-Boolean constraints, normalized to positive weights."""
    return self._pb_constraints

  def clone(self) -> IdlPbModel:
    """Returns an independent copy of this model."""
    copied = IdlPbModel()
    copied._num_int_vars = self._num_int_vars  # pylint: disable=protected-access
    copied._phases = list(self._phases)  # pylint: disable=protected-access
    copied._edges = list(self._edges)  # pylint: disable=protected-access
    copied._clauses = list(self._clauses)  # pylint: disable=protected-access
    copied._pb_constraints = list(self._pb_constraints)  # pylint: disable=protected-access
    return copied

  def new_int_var(self, lb: int, ub: int | None = None) -> int:
    """Adds an integer variable with `lb <= t <= ub` and returns its id."""
    var = self._num_int_vars
    self._num_int_vars += 1
    self._edges.append(DiffEdge(0, var, -lb))
    if ub is not None:
      self._edges.append(DiffEdge(var, 0, ub))
    return var

  def new_bool_var(self, *, phase: bool = True) -> Literal:
    """Adds a Boolean variable and returns its positive literal.

    Args:
      phase: The value the solver tries first for this variable. Use `False` for
        variables that only add constraints when true.
    """
    self._phases.append(phase)
    return Literal(len(self._phases) - 1)

  def add_diff(
      self, u: int, v: int, rhs: int, guard: Literal | None = None
  ) -> None:
    """Adds `t[u] - t[v] <= rhs`, enforced only while `guard` holds."""
    self._check_int_var(u)
    self._check_int_var(v)
    if guard is not None:
      self._check_literal(guard)
    self._edges.append(DiffEdge(u, v, rhs, guard))

  def add_clause(self, literals: Sequence[Literal]) -> None:
    """Adds the clause `OR(literals)`; an empty clause is unsatisfiable."""
    for lit in literals:
      self._check_literal(lit)
    self._clauses.append(tuple(literals))

  def add_pb_ge(self, terms: Sequence[tuple[int, Literal]], rhs: int) -> None:
    """Adds `sum(weight * lit) >= rhs` for arbitrary integer weights."""
    coefficients: dict[int, int] = {}
    bound = rhs
    for weight, lit in terms:
      self._check_literal(lit)
      if lit.positive:
        coefficients[lit.var_id] = coefficients.get(lit.var_id, 0) + weight
      else:
        # weight * (1 - x) == weight - weight * x
        coefficients[lit.var_id] = coefficients.get(lit.var_id, 0) - weight
        bound -= weight
    normalized: list[tuple[int, Literal]] = []
    for var, coefficient in sorted(coefficients.items()):
      if coefficient > 0:
        normalized.append((coefficient, Literal(var)))
      elif coefficient < 0:
        # coefficient * x == coefficient + |coefficient| * (1 - x)
        normalized.append((-coefficient, Literal(var, False)))
        bound -= coefficient
    self._pb_constraints.append(PbConstraint(tuple(normalized), bound))

  def _check_int_var(self, var: int) -> None:
    if not 0 <= var < self._num_int_vars:
      raise ValueError(f"unknown integer variable {var}")

  def _check_literal(self, lit: Literal) -> None:
    if not 0 <= lit.var_id < len(self._phases):
      raise ValueError(f"unknown boolean variable {lit.var_id}")


def _encode(lit: Literal) -> int:
  return 2 * lit.var_id + (0 if lit.positive else 1)


def _luby(index: int) -> int:
  """Returns the `index`-th (0-based) element of the Luby sequence."""
  size, exponent = 1, 0
  while size < index + 1:
    exponent += 1
    size = 2 * size + 1
  while size - 1 != index:
    size = (size - 1) >> 1
    exponent -= 1
    index %= size
  return 2**exponent


class Solver:
  """Incremental CDCL(T) solver over a snapshot of an `IdlPbModel`.

  Internally a literal is the integer `2 * var + (1 if negated else 0)`, so
  `lit ^ 1` is its negation.
  """

  def __init__(self, model: IdlPbModel) -> None:
    num_vars = model.num_bool_vars
    self._num_vars = num_vars
    self._values = [_UNASSIGNED] * num_vars
    self._levels = [0] * num_vars
    self._reasons: list[list[int] | None] = [None] * num_vars
    self._phases = list(model.default_phases)
    self._activity = [0.0] * num_vars
    self._activity_inc = 1.0
    self._heap: list[tuple[float, int]] = [(0.0, v) for v in range(num_vars)]
    self._trail: list[int] = []
    self._trail_lim: list[int] = []
    self._qhead = 0
    self._watches: list[list[list[int]]] = [[] for _ in range(2 * num_vars)]
    self._unsat = False
    self._init_theory(model)
    if not self._unsat:
      self._init_pseudo_booleans(model)
    if not self._unsat:
      self._init_clauses(model)
    if not self._unsat:
      self._unsat = any(
          self._check_pb(idx) is not None for idx in range(len(self._pb_terms))
      )

  # Construction.

  def _init_theory(self, model: IdlPbModel) -> None:
    edges = model.edges
    self._num_nodes = model.num_int_vars
    self._dist = [0] * self._num_nodes
    self._edge_src = [e.src for e in edges]
    self._edge_dst = [e.dst for e in edges]
    self._edge_weight = [e.weight for e in edges]
    self._edge_guard = [
        -1 if e.guard is None else _encode(e.guard) for e in edges
    ]
    self._edge_active = [False] * len(edges)
    self._out_edges: list[list[int]] = [[] for _ in range(self._num_nodes)]
    self._edges_by_guard: list[list[int]] = [
        [] for _ in range(2 * self._num_vars)
    ]
    for idx, edge in enumerate(edges):
      self._out_edges[edge.src].append(idx)
      if edge.guard is not None:
        self._edges_by_guard[self._edge_guard[idx]].append(idx)
    for idx, edge in enumerate(edges):
      if edge.guard is None and self._activate_edge(idx) is not None:
        self._unsat = True
        return

  def _init_pseudo_booleans(self, model: IdlPbModel) -> None:
    self._pb_terms: list[list[tuple[int, int]]] = []
    self._pb_slack: list[int] = []
    self._pb_occurrences: list[list[tuple[int, int]]] = [
        [] for _ in range(2 * self._num_vars)
    ]
    for pb in model.pb_constraints:
      if pb.rhs <= 0:
        continue
      terms = sorted(
          ((w, _encode(lit)) for w, lit in pb.terms if w > 0), reverse=True
      )
      slack = sum(w for w, _ in terms) - pb.rhs
      if slack < 0:
        self._unsat = True
        return
      idx = len(self._pb_terms)
      self._pb_terms.append(terms)
      self._pb_slack.append(slack)
      for weight, lit in terms:
        self._pb_occurrences[lit].append((idx, weight))

  def _init_clauses(self, model: IdlPbModel) -> None:
    for clause in model.clauses:
      lits = list(dict.fromkeys(_encode(lit) for lit in clause))
      lit_set = set(lits)
      if any(lit ^ 1 in lit_set for lit in lits):
        continue
      if not lits:
        self._unsat = True
        return
      if len(lits) == 1:
        value = self._lit_value(lits[0])
        if value == 0:
          self._unsat = True
          return
        if value == _UNASSIGNED:
          self._enqueue(lits[0], [lits[0]])
        continue
      self._watches[lits[0]].append(lits)
      self._watches[lits[1]].append(lits)

  # Assignment bookkeeping.

  def _lit_value(self, lit: int) -> int:
    value = self._values[lit >> 1]
    if value == _UNASSIGNED:
      return _UNASSIGNED
    return value ^ (lit & 1)

  def _enqueue(self, lit: int, reason: list[int] | None) -> None:
    var = lit >> 1
    self._values[var] = (lit & 1) ^ 1
    self._levels[var] = len(self._trail_lim)
    self._reasons[var] = reason
    self._trail.append(lit)
    for pb_idx, weight in self._pb_occurrences[lit ^ 1]:
      self._pb_slack[pb_idx] -= weight

  def _backtrack(self, level: int) -> None:
    if len(self._trail_lim) <= level:
      return
    cut = self._trail_lim[level]
    for lit in reversed(self._trail[cut:]):
      var = lit >> 1
      for pb_idx, weight in self._pb_occurrences[lit ^ 1]:
        self._pb_slack[pb_idx] += weight
      for edge_idx in self._edges_by_guard[lit]:
        self._edge_active[edge_idx] = False
      self._phases[var] = (lit & 1) == 0
      self._values[var] = _UNASSIGNED
      self._reasons[var] = None
      heapq.heappush(self._heap, (-self._activity[var], var))
    del self._trail[cut:]
    del self._trail_lim[level:]
    self._qhead = len(self._trail)

  # Propagation.

  def _propagate(self) -> list[int] | None:
    """Propagates the trail; returns a falsified clause on conflict."""
    while self._qhead < len(self._trail):
      lit = self._trail[self._qhead]
      self._qhead += 1
      conflict = self._propagate_theory(lit)
      if conflict is None:
        conflict = self._propagate_watches(lit)
      if conflict is None:
        conflict = self._propagate_pb(lit)
      if conflict is not None:
        return conflict
    return None

  def _propagate_watches(self, lit: int) -> list[int] | None:
    false_lit = lit ^ 1
    watchers = self._watches[false_lit]
    kept: list[list[int]] = []
    conflict = None
    for pos, clause in enumerate(watchers):
      if clause[0] == false_lit:
        clause[0], clause[1] = clause[1], clause[0]
      if self._lit_value(clause[0]) == 1:
        kept.append(clause)
        continue
      for k in range(2, len(clause)):
        if self._lit_value(clause[k]) != 0:
          clause[1], clause[k] = clause[k], clause[1]
          self._watches[clause[1]].append(clause)
          break
      else:
        kept.append(clause)
        if self._lit_value(clause[0]) == 0:
          conflict = clause
          kept.extend(watchers[pos + 1 :])
          break
        self._enqueue(clause[0], clause)
    self._watches[false_lit] = kept
    return conflict

  def _check_pb(self, pb_idx: int) -> list[int] | None:
    """Detects a conflict or propagates the forced literals of a constraint."""
    terms = self._pb_terms[pb_idx]
    slack = self._pb_slack[pb_idx]
    if slack < 0:
      return [lit for _, lit in terms if self._lit_value(lit) == 0]
    for weight, lit in terms:
      if weight <= slack:
        break
      if self._lit_value(lit) == _UNASSIGNED:
        false_lits = [m for _, m in terms if self._lit_value(m) == 0]
        self._enqueue(lit, [lit] + false_lits)
    return None

  def _propagate_pb(self, lit: int) -> list[int] | None:
    for pb_idx, _ in self._pb_occurrences[lit ^ 1]:
      conflict = self._check_pb(pb_idx)
      if conflict is not None:
        return conflict
    return None

  def _propagate_theory(self, lit: int) -> list[int] | None:
    for edge_idx in self._edges_by_guard[lit]:
      if self._edge_active[edge_idx]:
        continue
      cycle = self._activate_edge(edge_idx)
      if cycle is not None:
        guards = dict.fromkeys(
            self._edge_guard[e] ^ 1 for e in cycle if self._edge_guard[e] >= 0
        )
        return list(guards)
    return None

  def _activate_edge(self, edge_idx: int) -> list[int] | None:
    """Activates an edge; returns the edges of a negative cycle if any.

    `self._dist` is a potential with `dist[v] <= dist[u] + w` for every active
    edge `u -> v` of weight `w`, so `t = dist[0] - dist` satisfies all active
    constraints. A new edge can only close negative cycles through itself, so
    it suffices to relax distances from its head and stop if its tail would
    improve. Deactivating edges never invalidates the potential.
    """
    self._edge_active[edge_idx] = True
    u = self._edge_src[edge_idx]
    v = self._edge_dst[edge_idx]
    w = self._edge_weight[edge_idx]
    dist = self._dist
    if u == v:
      return [edge_idx] if w < 0 else None
    if dist[u] + w >= dist[v]:
      return None
    new_dist = {v: dist[u] + w}
    parent = {v: edge_idx}
    queue = collections.deque([v])
    queued = {v}
    while queue:
      node = queue.popleft()
      queued.discard(node)
      node_dist = new_dist[node]
      for out_idx in self._out_edges[node]:
        if not self._edge_active[out_idx]:
          continue
        head = self._edge_dst[out_idx]
        candidate = node_dist + self._edge_weight[out_idx]
        if candidate >= new_dist.get(head, dist[head]):
          continue
        if head == u:
          return self._collect_cycle(parent, out_idx, u)
        new_dist[head] = candidate
        parent[head] = out_idx
        if head not in queued:
          queued.add(head)
          queue.append(head)
    for node, node_dist in new_dist.items():
      dist[node] = node_dist
    return None

  def _collect_cycle(
      self, parent: Mapping[int, int], closing_edge: int, tail: int
  ) -> list[int]:
    cycle = [closing_edge]
    node = self._edge_src[closing_edge]
    for _ in range(len(parent) + 1):
      if node == tail:
        return cycle
      edge_idx = parent[node]
      cycle.append(edge_idx)
      node = self._edge_src[edge_idx]
    raise RuntimeError("negative cycle extraction did not terminate")

  # Conflict analysis.

  def _bump(self, var: int) -> None:
    self._activity[var] += self._activity_inc
    if self._activity[var] > _ACTIVITY_RESCALE_LIMIT:
      self._activity = [a / _ACTIVITY_RESCALE_LIMIT for a in self._activity]
      self._activity_inc /= _ACTIVITY_RESCALE_LIMIT
      self._heap = [
          (-self._activity[v], v)
          for v in range(self._num_vars)
          if self._values[v] == _UNASSIGNED
      ]
      heapq.heapify(self._heap)
    elif self._values[var] == _UNASSIGNED:
      heapq.heappush(self._heap, (-self._activity[var], var))

  def _analyze(self, conflict: list[int]) -> tuple[list[int], int]:
    """Derives a first-UIP clause; returns `(clause, backjump_level)`."""
    level = max(self._levels[lit >> 1] for lit in conflict)
    self._backtrack(level)
    seen: set[int] = set()
    learned = [0]
    pending = 0
    index = len(self._trail) - 1
    clause = conflict
    resolved = -1
    while True:
      for lit in clause:
        var = lit >> 1
        if lit == resolved or var in seen or self._levels[var] == 0:
          continue
        seen.add(var)
        self._bump(var)
        if self._levels[var] == level:
          pending += 1
        else:
          learned.append(lit)
      while (self._trail[index] >> 1) not in seen:
        index -= 1
      resolved = self._trail[index]
      index -= 1
      pending -= 1
      if pending == 0:
        break
      reason = self._reasons[resolved >> 1]
      if reason is None:
        raise RuntimeError("conflict analysis reached a decision literal")
      clause = reason
    learned[0] = resolved ^ 1
    if len(learned) == 1:
      return learned, 0
    best = max(
        range(1, len(learned)), key=lambda i: self._levels[learned[i] >> 1]
    )
    learned[1], learned[best] = learned[best], learned[1]
    return learned, self._levels[learned[1] >> 1]

  def _learn(self, learned: list[int]) -> None:
    if len(learned) > 1:
      self._watches[learned[0]].append(learned)
      self._watches[learned[1]].append(learned)
    self._enqueue(learned[0], learned)

  # Search.

  def _pick_branch_var(self) -> int:
    while self._heap:
      neg_activity, var = heapq.heappop(self._heap)
      if (
          self._values[var] == _UNASSIGNED
          and -neg_activity == self._activity[var]
      ):
        return var
    for var in range(self._num_vars):
      if self._values[var] == _UNASSIGNED:
        return var
    return -1

  def _next_decision(self, assumed: Sequence[int]) -> int | None:
    """Returns the next decision literal, or None if all are assigned."""
    while len(self._trail_lim) < len(assumed):
      lit = assumed[len(self._trail_lim)]
      value = self._lit_value(lit)
      if value == 0:
        return _ASSUMPTION_FAILED
      if value == _UNASSIGNED:
        return lit
      # Already true: open an empty level so levels and assumptions align.
      self._trail_lim.append(len(self._trail))
    var = self._pick_branch_var()
    if var < 0:
      return None
    return 2 * var + (0 if self._phases[var] else 1)

  def _model(self) -> SolveResult:
    origin = self._dist[0]
    return SolveResult(
        SolverStatus.FEASIBLE,
        int_values={
            node: origin - self._dist[node]
            for node in range(1, self._num_nodes)
        },
        bool_values={
            var: self._values[var] == 1 for var in range(self._num_vars)
        },
    )

  def solve(
      self,
      assumptions: Sequence[Literal] = (),
      *,
      timeout_seconds: float = 5.0,
  ) -> SolveResult:
    """Decides the model under `assumptions`.

    Args:
      assumptions: Literals that must hold for this call only.
      timeout_seconds: Wall-clock budget; `UNKNOWN` is returned when exceeded.

    Returns:
      The status, plus integer and Boolean values when feasible.
    """
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    if self._unsat:
      return SolveResult(SolverStatus.INFEASIBLE)
    self._backtrack(0)
    assumed = [_encode(lit) for lit in assumptions]
    conflicts = 0
    restarts = 0
    while True:
      conflict = self._propagate()
      if conflict is not None:
        if not self._trail_lim or not conflict:
          self._unsat = True
          return SolveResult(SolverStatus.INFEASIBLE)
        learned, level = self._analyze(conflict)
        self._backtrack(level)
        self._learn(learned)
        self._activity_inc /= _ACTIVITY_DECAY
        conflicts += 1
        continue
      if time.monotonic() >= deadline:
        self._backtrack(0)
        return SolveResult(SolverStatus.UNKNOWN)
      if conflicts >= _RESTART_BASE * _luby(restarts):
        conflicts = 0
        restarts += 1
        self._backtrack(0)
        continue
      decision = self._next_decision(assumed)
      if decision is None:
        result = self._model()
        self._backtrack(0)
        return result
      if decision == _ASSUMPTION_FAILED:
        self._backtrack(0)
        return SolveResult(SolverStatus.INFEASIBLE)
      self._trail_lim.append(len(self._trail))
      self._enqueue(decision, None)


def solve(model: IdlPbModel, *, timeout_seconds: float = 5.0) -> SolveResult:
  """Decides `model` from scratch; see `Solver.solve`."""
  return Solver(model).solve(timeout_seconds=timeout_seconds)
