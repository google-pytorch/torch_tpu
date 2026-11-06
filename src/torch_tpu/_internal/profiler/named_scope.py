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

"""Name scopes that label TPU ops in eager, compiled, and exported execution."""

from collections.abc import Callable, Mapping, Sequence
import contextlib
import functools
from typing import Any, ParamSpec, TypeVar, cast

import torch
from torch_tpu._internal.compile import tpu_torch_compile

# Prefix of the `node.meta["custom"]` keys that carry `named_scope` names
# through torch.compile and torch.export. Each name gets its own key so that
# nested `torch.fx.traceback.annotate` regions accumulate in insertion order
# (outer first) instead of overwriting each other.
METADATA_KEY_PREFIX = "torch_tpu::name_scope::"

# The kinds of FX nodes that issue ops.
_OP_NODE_KINDS = frozenset(("call_function", "call_method", "call_module"))
# The FX node metadata keys that identify traced user ops or carry module scopes
# or `named_scope`s (excluding synthetic backward/pass nodes without metadata).
_SCOPE_METADATA_KEYS = (
    "nn_module_stack",
    "fwd_nn_module_stack",
    "source_fn_stack",
    "fwd_source_fn_stack",
    "custom",
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


class _NameScope(contextlib.AbstractContextManager[None]):
  """A `named_scope` that is created outside of Dynamo tracing.

  Subclasses implement `__enter__`/`__exit__`; this base class makes the scope
  usable as a decorator in every mode.
  """

  def __init__(self, name: str):
    self._name = name

  def __call__(self, func: Callable[_P, _R]) -> Callable[_P, _R]:
    # Functions are usually decorated outside of compiled code, so the scope is
    # created again on each call, in the frame that runs the function.
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
      with named_scope(self._name):
        return func(*args, **kwargs)

    return wrapper


class _EagerNameScope(_NameScope):
  """A `named_scope` entered in eager mode."""

  def __enter__(self) -> None:
    tpu_torch_compile.append_name_scope(self._name)
    return None

  def __exit__(self, *exc_info: object) -> None:
    tpu_torch_compile.pop_name_scopes()
    return None


class _ExportNameScope(_NameScope):
  """A `named_scope` entered during non-strict `torch.export` tracing.

  Non-strict export runs the model as plain Python under proxy tracing, which
  copies `torch.fx.traceback` annotations onto the traced nodes only while
  `preserve_node_meta()` is active. Strict export and `torch.compile` trace
  under Dynamo, whose handling of `annotate` enters `preserve_node_meta()`
  itself.
  """

  def __init__(self, name: str):
    super().__init__(name)
    self._stacks: list[contextlib.ExitStack] = []

  def __enter__(self) -> None:
    with contextlib.ExitStack() as stack:
      # `annotate` must be entered before `preserve_node_meta`: on PyTorch
      # releases where `annotate` updates the shared `custom` dict in place,
      # `preserve_node_meta` restores a shallow copy on exit, so the reverse
      # order would leak an inner scope into the outer scope's later nodes.
      stack.enter_context(
          torch.fx.traceback.annotate(
              {METADATA_KEY_PREFIX + self._name: self._name}
          )
      )
      stack.enter_context(torch.fx.traceback.preserve_node_meta())
      self._stacks.append(stack.pop_all())
    return None

  def __exit__(self, *exc_info: object) -> None:
    self._stacks.pop().close()
    return None


def _validate_name(name: str) -> None:
  """Raises `ValueError` if `name` is not a valid scope name."""
  if not name:
    raise ValueError("named_scope expects a non-empty string, got ''.")
  if "@" in name:
    # XLA truncates an op's HLO metadata name at the first "@", which would
    # drop the rest of the hierarchy and the op name itself.
    raise ValueError(f"named_scope name must not contain '@', got {name!r}.")
  if ":" in name:
    # XProf splits `<op_name>:<op_type>` at the first ":", which would corrupt
    # the framework name scope hierarchy.
    raise ValueError(f"named_scope name must not contain ':', got {name!r}.")
  if any(not segment for segment in name.split("/")):
    raise ValueError(
        "named_scope name must not have empty '/'-separated segments, got"
        f" {name!r}."
    )


def named_scope(name: str) -> _NameScope:
  """Names the TPU ops issued inside the context or the decorated function.

  In eager mode, `name` is pushed onto the active `ContextManagerState` stack
  and included in the MLIR/HLO op metadata (which XProf shows as a level of the
  "Framework Name Scope" hierarchy). Like source locations, name scopes are
  excluded from the compilation cache key so identical subgraphs across scopes
  share cached executables and keep the scope from when they were first
  compiled. Because `ContextManagerState` is stored in
  `c10::ThreadLocalDebugInfo`, scopes active when `backward()` is called also
  propagate to C++ autograd worker threads.

  Under `torch.compile` (in both the forward and the backward graphs) and
  `torch.export` (strict and non-strict), the scope is recorded in the FX node
  metadata via `torch.fx.traceback.annotate`, and each run of consecutive ops in
  the scope gets it inside the innermost `nn.Module` call that contains the
  whole run, but outside the module calls that the run spans entirely. For
  example, `@named_scope("expert")` on `Expert.forward` labels its ops
  `moe[MoE]/expert/experts.0[Expert]/...`. A name that is already in the active
  hierarchy is not added again. A graph break inside the scope splits the graph
  as usual, and the ops on both sides of the break are labeled on PyTorch
  builds that can resume `torch.fx.traceback.annotate` after a graph break
  (`torch.fx.traceback._dynamo_annotate` exists, nightly 2.16.0a0 and later);
  on earlier releases Dynamo falls back to running the frame eagerly.

  Example:

    with named_scope.named_scope("route"):
      weights = router(x)

    @named_scope.named_scope("permute")
    def permute(x, indices):
      ...

  Create a scope where it is entered: entering a scope inside `torch.compile`
  or `torch.export` that was constructed outside of it causes a graph break or
  leaves ops unlabeled.

  Args:
    name: The name of the scope. A `/` in the name creates multiple levels.
      Names must not contain `@` or `:`, or have empty `/`-separated segments.

  Returns:
    A context manager, which can also decorate a function.

  Raises:
    ValueError: If `name` is empty, contains `@` or `:`, or has an empty
      `/`-separated segment.
  """
  _validate_name(name)
  if torch.compiler.is_dynamo_compiling():
    # Dynamo can resume a graph inside `annotate` after a graph break, but not
    # inside a context manager that wraps it.
    return cast(
        _NameScope,
        torch.fx.traceback.annotate({METADATA_KEY_PREFIX + name: name}),
    )
  if torch.compiler.is_exporting():
    return _ExportNameScope(name)
  return _EagerNameScope(name)


def _get_name_scopes(meta: Mapping[str, Any]) -> list[str]:
  """Returns the `named_scope` names in FX node metadata, outer first."""
  custom = meta.get("custom")
  if not isinstance(custom, Mapping):
    return []
  return [
      value
      for key, value in custom.items()
      if isinstance(key, str)
      and key.startswith(METADATA_KEY_PREFIX)
      and isinstance(value, str)
  ]


def _common_prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
  """Returns the length of the longest common prefix of `a` and `b`."""
  length = 0
  for x, y in zip(a, b):
    if x != y:
      break
    length += 1
  return length


def compute_graph_name_scopes(
    graph: torch.fx.Graph,
    module_scopes: Callable[[torch.fx.Node], Sequence[tuple[str, str]]],
) -> dict[torch.fx.Node, tuple[str, ...]]:
  """Computes the hierarchical name scopes of the op nodes of a graph.

  The scopes of a node are its module scopes interleaved with the
  `named_scope`s in its metadata. As the metadata does not say in which module
  a `named_scope` was entered, each maximal run of consecutive nodes in a
  `named_scope` gets it inside the innermost module that contains the whole
  run, but outside of the modules whose calls the run spans entirely, i.e. that
  the nodes right before and after the run are not in.

  Args:
    graph: The FX graph.
    module_scopes: Returns a `(key, scope)` pair per module that encloses a
      node, from outer to inner, where `key` identifies the module call and
      `scope` is the name scope of the module.

  Returns:
    The name scopes of each op node with scope metadata, from outer to inner.
  """
  nodes = [
      node
      for node in graph.nodes
      if node.op in _OP_NODE_KINDS
      and any(node.meta.get(key) is not None for key in _SCOPE_METADATA_KEYS)
  ]
  module_entries = [module_scopes(node) for node in nodes]
  module_keys = [tuple(key for key, _ in entries) for entries in module_entries]
  user_scopes = [_get_name_scopes(node.meta) for node in nodes]

  # The number of module scopes outside of each `named_scope` of each node.
  depths: list[dict[str, int]] = [{} for _ in nodes]

  def place_run(scope: str, start: int, end: int) -> None:
    common = module_keys[start]
    for keys in module_keys[start + 1 : end]:
      common = common[: _common_prefix_len(common, keys)]
    depth = 0
    if start > 0:
      depth = _common_prefix_len(common, module_keys[start - 1])
    if end < len(nodes):
      depth = max(depth, _common_prefix_len(common, module_keys[end]))
    for i in range(start, end):
      depths[i][scope] = depth

  run_starts: dict[str, int] = {}
  for i, scopes in enumerate(user_scopes):
    for scope in [s for s in run_starts if s not in scopes]:
      place_run(scope, run_starts.pop(scope), i)
    for scope in scopes:
      run_starts.setdefault(scope, i)
  for scope, start in run_starts.items():
    place_run(scope, start, len(nodes))

  graph_scopes = {}
  for node, entries, scopes, scope_depths in zip(
      nodes, module_entries, user_scopes, depths, strict=True
  ):
    merged: list[str] = []
    num_modules = 0
    for scope in scopes:
      # Inner `named_scope`s stay inside outer ones.
      depth = max(num_modules, min(scope_depths[scope], len(entries)))
      merged.extend(
          module_scope for _, module_scope in entries[num_modules:depth]
      )
      merged.append(scope)
      num_modules = depth
    merged.extend(module_scope for _, module_scope in entries[num_modules:])
    if merged:
      graph_scopes[node] = tuple(merged)
  return graph_scopes
