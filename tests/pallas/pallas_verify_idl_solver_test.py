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

"""Tests for the pure-Python difference-logic + pseudo-Boolean solver."""

from __future__ import annotations

from collections.abc import Sequence
import itertools
import random

from absl.testing import absltest
from torch_tpu._internal.pallas.verify import idl_solver
from tests import seed_test_utils

Literal = idl_solver.Literal
SolverStatus = idl_solver.SolverStatus


def _lit_true(lit: Literal, bools: Sequence[bool]) -> bool:
  return bools[lit.var_id] == lit.positive


def _difference_constraints_feasible(
    num_nodes: int, edges: Sequence[tuple[int, int, int]]
) -> bool:
  """Reference Bellman-Ford: True iff all `t[u] - t[v] <= w` can hold."""
  dist = [0] * num_nodes
  for _ in range(num_nodes + 1):
    changed = False
    for u, v, w in edges:
      if dist[u] + w < dist[v]:
        dist[v] = dist[u] + w
        changed = True
    if not changed:
      return True
  return False


def _satisfies_booleans(
    model: idl_solver.IdlPbModel, bools: Sequence[bool]
) -> bool:
  if not all(any(_lit_true(l, bools) for l in c) for c in model.clauses):
    return False
  return all(
      sum(w for w, l in pb.terms if _lit_true(l, bools)) >= pb.rhs
      for pb in model.pb_constraints
  )


def _active_edges(
    model: idl_solver.IdlPbModel, bools: Sequence[bool]
) -> list[tuple[int, int, int]]:
  return [
      (e.src, e.dst, e.weight)
      for e in model.edges
      if e.guard is None or _lit_true(e.guard, bools)
  ]


def _brute_force_feasible(
    model: idl_solver.IdlPbModel, assumptions: Sequence[Literal] = ()
) -> bool:
  for bools in itertools.product((False, True), repeat=model.num_bool_vars):
    if not all(_lit_true(a, bools) for a in assumptions):
      continue
    if not _satisfies_booleans(model, bools):
      continue
    if _difference_constraints_feasible(
        model.num_int_vars, _active_edges(model, bools)
    ):
      return True
  return False


def _solution_violations(
    model: idl_solver.IdlPbModel,
    result: idl_solver.SolveResult,
    assumptions: Sequence[Literal] = (),
) -> list[str]:
  bools = [result.bool_values[v] for v in range(model.num_bool_vars)]
  times = {0: 0, **result.int_values}
  violations = [
      f"assumption {a}" for a in assumptions if not _lit_true(a, bools)
  ]
  if not _satisfies_booleans(model, bools):
    violations.append("clause or pseudo-Boolean constraint")
  violations.extend(
      f"t[{u}] - t[{v}] <= {w}"
      for u, v, w in _active_edges(model, bools)
      if times[u] - times[v] > w
  )
  return violations


def _random_model(
    rng: random.Random,
) -> tuple[idl_solver.IdlPbModel, list[Literal]]:
  model = idl_solver.IdlPbModel()
  bools = [model.new_bool_var() for _ in range(rng.randint(0, 6))]
  nodes = [0] + [
      model.new_int_var(0, rng.choice((None, 2, 5)))
      for _ in range(rng.randint(1, 4))
  ]

  def random_literal() -> Literal:
    lit = rng.choice(bools)
    return lit if rng.random() < 0.5 else lit.negated()

  for _ in range(rng.randint(0, 9)):
    guard = random_literal() if bools and rng.random() < 0.7 else None
    model.add_diff(
        rng.choice(nodes), rng.choice(nodes), rng.randint(-2, 2), guard
    )
  if bools:
    for _ in range(rng.randint(0, 4)):
      model.add_clause([random_literal() for _ in range(rng.randint(1, 3))])
    for _ in range(rng.randint(0, 3)):
      terms = [
          (rng.randint(-3, 3), random_literal())
          for _ in range(rng.randint(1, 4))
      ]
      model.add_pb_ge(terms, rng.randint(-2, 5))
  return model, bools


class IdlSolverTest(seed_test_utils.RepeatableTest):

  def test_chain_is_feasible_and_ordered(self):
    model = idl_solver.IdlPbModel()
    a, b, c = (model.new_int_var(0) for _ in range(3))
    model.add_diff(a, b, -1)  # a < b
    model.add_diff(b, c, -1)  # b < c
    result = idl_solver.solve(model)
    self.assertEqual(result.status, SolverStatus.FEASIBLE)
    self.assertLess(result.int_values[a], result.int_values[b])
    self.assertLess(result.int_values[b], result.int_values[c])
    self.assertGreaterEqual(result.int_values[a], 0)

  def test_negative_cycle_is_infeasible(self):
    model = idl_solver.IdlPbModel()
    a, b = model.new_int_var(0), model.new_int_var(0)
    model.add_diff(a, b, -1)
    model.add_diff(b, a, -1)
    self.assertEqual(idl_solver.solve(model).status, SolverStatus.INFEASIBLE)

  def test_upper_bounds_are_enforced(self):
    model = idl_solver.IdlPbModel()
    a, b = model.new_int_var(0, 1), model.new_int_var(0, 1)
    c = model.new_int_var(0, 1)
    model.add_diff(a, b, -1)
    model.add_diff(b, c, -1)  # Needs three distinct values in [0, 1].
    self.assertEqual(idl_solver.solve(model).status, SolverStatus.INFEASIBLE)

  def test_guarded_disjunction_picks_the_feasible_order(self):
    model = idl_solver.IdlPbModel()
    a, b = model.new_int_var(0), model.new_int_var(0)
    a_first = model.new_bool_var()
    model.add_diff(a, b, -1, a_first)
    model.add_diff(b, a, -1, a_first.negated())
    model.add_diff(b, a, -2)  # Forces b < a.
    result = idl_solver.solve(model)
    self.assertEqual(result.status, SolverStatus.FEASIBLE)
    self.assertFalse(result.bool_values[a_first.var_id])
    self.assertLess(result.int_values[b], result.int_values[a])

  def test_pseudo_boolean_and_clauses_conflict(self):
    model = idl_solver.IdlPbModel()
    x, y, z = (model.new_bool_var() for _ in range(3))
    model.add_pb_ge([(1, x), (1, y), (1, z)], 2)  # At least two true.
    for p, q in itertools.combinations((x, y, z), 2):
      model.add_clause([p.negated(), q.negated()])  # At most one true.
    self.assertEqual(idl_solver.solve(model).status, SolverStatus.INFEASIBLE)

  def test_pseudo_boolean_negative_weights_are_normalized(self):
    model = idl_solver.IdlPbModel()
    x, y = model.new_bool_var(), model.new_bool_var()
    model.add_pb_ge([(3, x), (-2, y)], 1)
    model.add_clause([y])
    result = idl_solver.solve(model)
    self.assertEqual(result.status, SolverStatus.FEASIBLE)
    self.assertTrue(result.bool_values[x.var_id])
    model.add_pb_ge([(-1, x)], 0)  # x must be false.
    self.assertEqual(idl_solver.solve(model).status, SolverStatus.INFEASIBLE)

  def test_assumptions_reuse_learned_state(self):
    model = idl_solver.IdlPbModel()
    a, b = model.new_int_var(0), model.new_int_var(0)
    a_first = model.new_bool_var()
    model.add_diff(a, b, -1, a_first)
    model.add_diff(b, a, -1, a_first.negated())
    solver = idl_solver.Solver(model)
    self.assertEqual(solver.solve([a_first]).status, SolverStatus.FEASIBLE)
    self.assertEqual(
        solver.solve([a_first.negated()]).status, SolverStatus.FEASIBLE
    )
    self.assertEqual(
        solver.solve([a_first, a_first.negated()]).status,
        SolverStatus.INFEASIBLE,
    )
    self.assertEqual(solver.solve().status, SolverStatus.FEASIBLE)

  def test_zero_timeout_returns_unknown(self):
    model = idl_solver.IdlPbModel()
    a, b = model.new_int_var(0), model.new_int_var(0)
    order = model.new_bool_var()
    model.add_diff(a, b, -1, order)
    self.assertEqual(
        idl_solver.solve(model, timeout_seconds=0.0).status,
        SolverStatus.UNKNOWN,
    )

  def test_random_models_match_brute_force(self):
    rng = random.Random(20260)
    for case in range(400):
      model, bools = _random_model(rng)
      expected = _brute_force_feasible(model)
      result = idl_solver.solve(model, timeout_seconds=30.0)
      self.assertEqual(
          result.status,
          SolverStatus.FEASIBLE if expected else SolverStatus.INFEASIBLE,
          msg=f"case {case}",
      )
      if expected:
        self.assertEmpty(
            _solution_violations(model, result), msg=f"case {case}"
        )
      if not bools:
        continue
      # Repeated solves on one solver must stay sound under assumptions.
      solver = idl_solver.Solver(model)
      for assumption in (bools[0], bools[0].negated(), None):
        assumed = [] if assumption is None else [assumption]
        expected = _brute_force_feasible(model, assumed)
        result = solver.solve(assumed, timeout_seconds=30.0)
        self.assertEqual(
            result.status,
            SolverStatus.FEASIBLE if expected else SolverStatus.INFEASIBLE,
            msg=f"case {case} assuming {assumed}",
        )
        if expected:
          self.assertEmpty(
              _solution_violations(model, result, assumed),
              msg=f"case {case} assuming {assumed}",
          )


if __name__ == "__main__":
  absltest.main()
