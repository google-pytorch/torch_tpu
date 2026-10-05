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

"""TPU implementation for torch.ops.higher_order.scan and associative_scan."""

import functools
import importlib
import types

import torch
from torch._higher_order_ops import associative_scan as _orig_associative_scan
from torch._higher_order_ops import utils as _hop_utils
from torch._inductor import utils as _inductor_utils
import torch.export
import torch.utils._pytree as pytree
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.export import export

# `torch._higher_order_ops.associative_scan` resolves to the re-exported
# function rather than its module, so look the module up by name.
_assoc_scan_module = importlib.import_module(
    "torch._higher_order_ops.associative_scan"
)


def _to_placeholder(t, shape=None):
  """Returns `t` as a like-shaped placeholder, or `t` itself if not a tensor.

  Args:
    t: A body operand.
    shape: Placeholder shape, defaulting to `t`'s own.
  """
  if not isinstance(t, torch.Tensor):
    return t
  if shape is None:
    shape = list(t.shape)
  return tpu_torch_compile.placeholder(
      shape,
      t.dtype,
      t.requires_grad,
      is_pinned_host=compiler.is_pinned_host_tensor(t),
  )


def _pin_offloaded_body_args(combine_fn, body_args):
  """Returns `body_args` with the offloaded operands marked as host memory.

  The scan body is compiled as a standalone program, so every operand is
  allocated HBM. That includes activations already offloaded to host, which can
  be large enough to exhaust HBM on their own. Marking them as host memory keeps
  them out of the HBM total.

  An operand counts as offloaded if `ao.reload` copies it back from the host, or
  if the offload pass tagged its placeholder `saved_for_offloading`.

  `_backend._mark_pinned_host_inputs` does the same for the top-level graph, but
  does not see this compile.

  Args:
    combine_fn: The body function. Only a `GraphModule` can be inspected; any
      other callable is returned unchanged.
    body_args: The body operands, in the same order as the body placeholders.
  """
  if not isinstance(combine_fn, torch.fx.GraphModule):
    return body_args

  # pylint: disable=protected-access
  reloaded = set()
  for node in combine_fn.graph.nodes:
    if _backend._is_ao_op(node, "reload"):
      source = _backend._get_reload_source_placeholder(
          _backend._extract_tensor_node(node)
      )
      if source is not None:
        reloaded.add(source)
  # pylint: enable=protected-access

  placeholders = [n for n in combine_fn.graph.nodes if n.op == "placeholder"]
  pinned = list(body_args)
  for i, (node, arg) in enumerate(zip(placeholders, body_args, strict=True)):
    if (
        isinstance(arg, torch.Tensor)
        and not compiler.is_pinned_host_tensor(arg)
        and (node in reloaded or node.meta.get("saved_for_offloading", False))
    ):
      pinned[i] = tpu_torch_compile.placeholder_like(arg, is_pinned_host=True)
  return pinned


# `torch.fx.traceback.annotate` key that asks for a reverse scan.
#
# Torch's `scan` HOP has no direction argument: it reverses by flipping its
# inputs and outputs, and on TPU each flip is a full copy. A caller that traces
# a body under `annotate({SCAN_REVERSE_ANNOTATION: True})` gets a reverse loop
# instead: step i reads and writes row `len - 1 - i`, so neither flip is needed.
# The key rides on the body's own nodes because the body is re-traced into a
# fresh GraphModule on its way here, which drops any attribute set on it.
SCAN_REVERSE_ANNOTATION = "torch_tpu_scan_reverse"


def _body_asks_for_reverse(combine_fn) -> bool:
  if not isinstance(combine_fn, torch.fx.GraphModule):
    return False
  return any(
      node.meta.get("custom", {}).get(SCAN_REVERSE_ANNOTATION, False)
      for node in combine_fn.graph.nodes
  )


def _handle_scan_impl(combine_fn, init, xs, additional_inputs, reverse=False):
  """TPU implementation for scan higher-order op.

  This implementation traces the combine_fn and lowers the scan to a deferred
  C++ op, which builds a `stablehlo.while` loop.

  Args:
    combine_fn: The combine function.
    init: The initial carry state.
    xs: The input sequence.
    additional_inputs: Additional inputs to the combine function.
    reverse: Whether to scan in reverse.

  Returns:
    A tuple of flat tensors representing the final carry and the stacked
    outputs of the scan operation.
  """
  # Normalize inputs to lists.
  init_list, _ = pytree.tree_flatten(init)
  xs_list, _ = pytree.tree_flatten(xs)
  extra_list, _ = pytree.tree_flatten(additional_inputs)

  # `fx_to_mlir`'s args become the body module's block arguments, which have to
  # be graph leaves, so every body operand has to be a placeholder.
  init_args = [_to_placeholder(t) for t in init_list]
  # The body sees one slice of each `xs` along the scan dimension.
  x_list = [_to_placeholder(t, list(t.shape)[1:]) for t in xs_list]
  extra_args = [_to_placeholder(t) for t in extra_list]

  # Resolve the actual callable body function.
  actual_combine_fn = combine_fn
  if isinstance(combine_fn, torch.export.ExportedProgram):
    actual_combine_fn = combine_fn.module
  reverse = reverse or _body_asks_for_reverse(actual_combine_fn)

  # Perform a meta-run to infer the output shapes and dtypes.
  # pylint: disable=protected-access
  with torch._subclasses.fake_tensor.FakeTensorMode(allow_non_fake_inputs=True):

    def to_fake(tensors):
      return [
          tpu_torch_compile.placeholder_like(t)
          if isinstance(t, torch.Tensor)
          else t
          for t in tensors
      ]

    fake_inits = to_fake(init_args)
    fake_xs_heads = to_fake(x_list)
    fake_extras = to_fake(extra_args)

    fake_result = actual_combine_fn(*fake_inits, *fake_xs_heads, *fake_extras)
    fake_result_flat, _ = pytree.tree_flatten(fake_result)

    dummy_ys = []
    seq_len = xs_list[0].shape[0]
    num_carries = len(init_list)
    for y in fake_result_flat[num_carries:]:
      ys_shape = [seq_len] + list(y.shape)
      dummy_ys.append(torch.zeros(ys_shape, dtype=y.dtype, device="meta"))

  # Compile the body sub-graph to MLIR.
  body_args = init_args + x_list + extra_args
  body_args = _pin_offloaded_body_args(actual_combine_fn, body_args)
  body_mlir = export.fx_to_mlir(actual_combine_fn, args=body_args)

  # Call the C++ binding to create a deferred scan operation.
  scan_direction = (
      tpu_torch_compile.ScanDirection.kReverse
      if reverse
      else tpu_torch_compile.ScanDirection.kForward
  )
  num_scan_inputs = len(xs_list)
  inputs = xs_list + extra_list
  return tpu_torch_compile.create_scan_op(
      inits=init_list,
      inputs=inputs,
      body_module=body_mlir.module,
      scan_direction=scan_direction,
      dummy_ys=dummy_ys,
      num_scan_inputs=num_scan_inputs,
  )


def _slice_placeholders(xs_list, dim):
  """Returns placeholders for one slice of each `xs` along `dim`."""
  return [
      _to_placeholder(t, list(t.shape[:dim]) + list(t.shape[dim + 1 :]))
      for t in xs_list
  ]


def _build_body_mlir(
    combine_fn, body_args
) -> tpu_torch_compile.ContextedModule:
  """Traces `combine_fn` on `body_args` into an MLIR module with an `@main`.

  Unlike `export.fx_to_mlir`, this does not also compile the body into a
  standalone executable: `create_associative_scan_op` only needs the module,
  which it inlines into the scan. It also accepts plain Python callables, which
  the eager path passes.

  Args:
    combine_fn: The combine function or `GraphModule`.
    body_args: Placeholders for `(*left_elems, *right_elems, *extra_args)`.
  """
  with execution_mode.set_eager_mode(
      execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
  ):
    # Clone so in-place ops in `combine_fn` cannot overwrite the placeholders.
    cloned_args = [
        a.clone() if isinstance(a, torch.Tensor) else a for a in body_args
    ]
    if isinstance(combine_fn, torch.fx.GraphModule):
      raw_out = export.EagerLikeFxInterpreter(combine_fn).run(*cloned_args)
    else:
      raw_out = combine_fn(*cloned_args)
    result_tensors = []
    seen_ptrs = set()
    for t in pytree.tree_leaves(raw_out):
      if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"expected Tensor output from combine_fn, got {type(t)}"
        )
      # Each `@main` result needs its own node, so clone repeated outputs such
      # as `return x, x`.
      if t.data_ptr() in seen_ptrs:
        t = t.clone()
      seen_ptrs.add(t.data_ptr())
      result_tensors.append(t)
    return tpu_torch_compile.build_mlir(
        result_tensors=result_tensors,
        argument_tensors=[a for a in body_args if isinstance(a, torch.Tensor)],
        use_stablehlo_bounds=False,
    )


def _handle_associative_scan_impl(combine_fn, xs, additional_inputs):
  """TPU implementation for `torch.ops.higher_order.associative_scan`.

  This runs when `torch.compile` traces the HOP, and on the eager fallback path
  (e.g. when inputs require grad). The HOP always scans along dim 0 in the
  forward direction: the upstream frontend wraps it in `movedim` and `flip`.
  `create_associative_scan_op` recognizes the `movedim` views and scans the
  base buffers in place, so no transposes are emitted.

  Args:
    combine_fn: The combine function, a `GraphModule` when traced by Dynamo.
    xs: The scanned inputs.
    additional_inputs: Extra non-scanned inputs to `combine_fn`.

  Returns:
    The flat list of scanned outputs, one per leaf of `xs`.
  """
  xs_list = pytree.tree_leaves(xs)
  extra_list = pytree.tree_leaves(additional_inputs)

  actual_combine_fn = combine_fn
  if isinstance(combine_fn, torch.export.ExportedProgram):
    actual_combine_fn = combine_fn.module

  # Left and right operands need distinct placeholders.
  body_args = (
      _slice_placeholders(xs_list, 0)
      + _slice_placeholders(xs_list, 0)
      + [_to_placeholder(t) for t in extra_list]
  )
  body_args = _pin_offloaded_body_args(actual_combine_fn, body_args)
  return tpu_torch_compile.create_associative_scan_op(
      inputs=xs_list + extra_list,
      body_module=_build_body_mlir(actual_combine_fn, body_args),
      scan_direction=tpu_torch_compile.ScanDirection.kForward,
      dim=0,
      num_scan_inputs=len(xs_list),
  )


# Traced combine bodies for the eager fast path, keyed by
# `(_callable_cache_key(combine_fn), xs tree spec, per-leaf slice signature)`.
_MAX_BODY_MLIR_CACHE_SIZE = 256
_ASSOCIATIVE_BODY_MLIR_CACHE: dict[
    tuple[object, ...], tuple[tpu_torch_compile.ContextedModule, bool]
] = {}

_IMMUTABLE_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    type(None),
    torch.dtype,
)


def _referenced_names(code: types.CodeType) -> set[str]:
  """Returns the global/attribute names used by `code` and its nested code."""
  names = set(code.co_names)
  for const in code.co_consts:
    if isinstance(const, types.CodeType):
      names |= _referenced_names(const)
  return names


def _value_cache_key(value, seen):
  """Returns a hashable key that pins down `value`, or None if it cannot.

  Only immutable values that cannot hold tensor state get a key. Everything
  else, including tensors, `nn.Module`s and mutable containers, returns None.
  """
  if isinstance(value, _IMMUTABLE_SCALAR_TYPES):
    # Include the type so that, e.g., `1`, `1.0`, and `True` (which hash and
    # compare equal) do not share a body traced with a different dtype.
    return (type(value), value)
  if isinstance(value, (types.ModuleType, type)):
    return value
  if isinstance(value, tuple):
    keys = tuple(_value_cache_key(v, seen) for v in value)
    return None if any(k is None for k in keys) else keys
  if callable(value):
    return _callable_cache_key(value, seen)
  return None


def _callable_cache_key(fn, seen=frozenset()):
  """Returns a hashable key that determines the body traced from `fn`, or None.

  The traced body depends only on `fn`'s code and on the values it reads from
  its closure, defaults, and globals. If all of those are immutable non-tensor
  values (or functions that satisfy the same rule), they make up the key, and
  the traced MLIR can be reused across calls. Otherwise `fn` may read tensors
  or mutable state that the cached body would bake in, and this returns None.

  Args:
    fn: The candidate combine callable.
    seen: Code objects already being keyed, to stop on recursive functions.
  """
  # pylint: disable=protected-access
  if isinstance(fn, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
    return fn
  if isinstance(fn, types.BuiltinFunctionType):
    # A bound method such as `tensor.add` carries a tensor in `__self__`.
    if isinstance(getattr(fn, "__self__", None), torch.Tensor):
      return None
    return fn
  if isinstance(fn, functools.partial):
    keys = (
        _callable_cache_key(fn.func, seen),
        _value_cache_key(fn.args, seen),
        _value_cache_key(tuple(sorted(fn.keywords.items())), seen),
    )
    return None if any(k is None for k in keys) else ("partial", *keys)
  if not isinstance(fn, types.FunctionType) or fn.__code__ in seen:
    return None
  seen = seen | {fn.__code__}
  try:
    closure = tuple(cell.cell_contents for cell in fn.__closure__ or ())
  except ValueError:  # An unassigned closure cell.
    return None
  global_names = sorted(_referenced_names(fn.__code__) & fn.__globals__.keys())
  values_key = _value_cache_key(
      (
          fn.__defaults__ or (),
          tuple(sorted((fn.__kwdefaults__ or {}).items())),
          closure,
          tuple((name, fn.__globals__[name]) for name in global_names),
      ),
      seen,
  )
  return None if values_key is None else (fn.__code__, values_key)


def _eager_tpu_associative_scan(
    combine_fn,
    leaves_xs: list[torch.Tensor],
    spec_xs: pytree.TreeSpec,
    dim: int,
    reverse: bool,
    combine_mode: str,
    fn_key: object,
):
  """Runs `associative_scan` on TPU in eager mode as a single deferred op.

  Upstream eager `associative_scan` goes through `torch.compile` on every call,
  re-tracing `combine_fn`, and wraps the scan in `movedim` and `flip`. Instead,
  this traces and validates `combine_fn` once per cache key, then passes `dim`
  and `reverse` straight to `create_associative_scan_op`.

  Args:
    combine_fn: The user combine callable.
    leaves_xs: Flattened TPU input tensors, all of the same shape.
    spec_xs: PyTree structure of `xs`.
    dim: Non-negative scan dimension.
    reverse: Whether to scan in reverse order.
    combine_mode: Either `'pointwise'` or `'generic'`.
    fn_key: `_callable_cache_key(combine_fn)`.

  Returns:
    PyTree of scanned TPU tensors matching `spec_xs`.
  """
  if leaves_xs[0].shape[dim] == 0:
    return pytree.tree_unflatten([t.clone() for t in leaves_xs], spec_xs)

  slice_shapes = [
      list(t.shape[:dim]) + list(t.shape[dim + 1 :]) for t in leaves_xs
  ]
  # The body only depends on the slice shapes, not on the sequence length.
  sig_key = tuple(
      (tuple(s), t.dtype, compiler.is_pinned_host_tensor(t))
      for s, t in zip(slice_shapes, leaves_xs, strict=True)
  )
  cache_key = (fn_key, spec_xs, sig_key)
  cached = _ASSOCIATIVE_BODY_MLIR_CACHE.get(cache_key)
  if cached is None:
    num_leaves = len(leaves_xs)

    def flat_combine_fn(*flat_args):
      x = pytree.tree_unflatten(flat_args[:num_leaves], spec_xs)
      y = pytree.tree_unflatten(flat_args[num_leaves:], spec_xs)
      return pytree.tree_leaves(combine_fn(x, y))

    # Run the same output checks upstream does, once per cache entry.
    # pylint: disable=protected-access
    with torch._subclasses.fake_tensor.FakeTensorMode():
      fake_x = [
          torch.empty(s, dtype=t.dtype, device="tpu")
          for s, t in zip(slice_shapes, leaves_xs, strict=True)
      ]
      fake_y = [torch.empty_like(t) for t in fake_x]
      fake_out_leaves, out_spec = pytree.tree_flatten(
          combine_fn(
              pytree.tree_unflatten(fake_x, spec_xs),
              pytree.tree_unflatten(fake_y, spec_xs),
          )
      )
      if out_spec != spec_xs:
        raise RuntimeError(
            "the tree structure of the xs and the outs of the combine_fn are"
            f" expected to be identical, but got xs: {spec_xs} vs output:"
            f" {out_spec}"
        )
      for inp, out in zip(fake_x, fake_out_leaves, strict=True):
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != inp.shape
            or out.dtype != inp.dtype
            or out.device != inp.device
        ):
          raise RuntimeError(
              "the metadata of the combine_fn output is expected to match the"
              f" input slice metadata, got {inp} vs {out}"
          )
      fx_gm = _hop_utils._maybe_fake_tracing(
          flat_combine_fn, fake_x + fake_y, pre_dispatch=False
      )
      is_pointwise = all(
          _inductor_utils.is_pointwise_use(use) or use.op == "output"
          for node in fx_gm.graph.nodes
          for use in node.users
      )

    body_args = _slice_placeholders(leaves_xs, dim) + _slice_placeholders(
        leaves_xs, dim
    )
    cached = (_build_body_mlir(flat_combine_fn, body_args), is_pointwise)
    if len(_ASSOCIATIVE_BODY_MLIR_CACHE) >= _MAX_BODY_MLIR_CACHE_SIZE:
      _ASSOCIATIVE_BODY_MLIR_CACHE.clear()
    _ASSOCIATIVE_BODY_MLIR_CACHE[cache_key] = cached

  body_module, is_pointwise = cached
  if combine_mode == "pointwise" and not is_pointwise:
    raise RuntimeError(
        "for combine_mode='pointwise', the combine_fn needs to be pointwise"
    )
  out_leaves = tpu_torch_compile.create_associative_scan_op(
      inputs=leaves_xs,
      body_module=body_module,
      scan_direction=(
          tpu_torch_compile.ScanDirection.kReverse
          if reverse
          else tpu_torch_compile.ScanDirection.kForward
      ),
      dim=dim,
      num_scan_inputs=len(leaves_xs),
  )
  return pytree.tree_unflatten(out_leaves, spec_xs)


# An unpatched copy of upstream `associative_scan`, used as the fallback once
# `_install_associative_scan_patch` has replaced the original's `__code__`.
_SAVED_ORIG_ASSOCIATIVE_SCAN = types.FunctionType(
    _orig_associative_scan.__code__,
    _orig_associative_scan.__globals__,
    _orig_associative_scan.__name__,
    _orig_associative_scan.__defaults__,
    _orig_associative_scan.__closure__,
)


def _tpu_scan_dim(leaves_xs, dim, reverse) -> int | None:
  """Returns the non-negative scan dim if the TPU paths apply, else None.

  Returns None for non-TPU inputs and for arguments that upstream rejects, so
  that upstream's validation and error messages apply.
  """
  if (
      not leaves_xs
      or not all(
          isinstance(t, torch.Tensor) and t.device.type == "tpu"
          for t in leaves_xs
      )
      or not isinstance(dim, int)
      or not isinstance(reverse, bool)
  ):
    return None
  ndim = leaves_xs[0].ndim
  canonical_dim = dim + ndim if dim < 0 else dim
  return canonical_dim if 0 <= canonical_dim < ndim else None


def _can_use_eager_fast_path(combine_fn, leaves_xs, combine_mode) -> bool:
  """Returns whether `_eager_tpu_associative_scan` supports this call."""
  # pylint: disable=protected-access
  return (
      callable(combine_fn)
      and combine_mode in ("pointwise", "generic")
      and all(
          t.shape == leaves_xs[0].shape and t.device == leaves_xs[0].device
          for t in leaves_xs
      )
      # Autograd, `torch.jit` tracing, and functorch transforms need the HOP.
      and not any(t.requires_grad for t in leaves_xs)
      and not torch._C._get_tracing_state()
      and torch._C._functorch.peek_interpreter_stack() is None
  )


def _tpu_associative_scan_hop(
    combine_fn, leaves_xs, spec_xs, dim: int, reverse: bool
):
  """Calls the `associative_scan` HOP for TPU inputs with `'pointwise'` mode.

  This mirrors upstream's frontend, which cannot be used on TPU because some
  PyTorch releases (e.g. 2.11) reject `combine_mode='pointwise'` for any device
  other than CUDA and XPU.

  It also flips along `dim` before `movedim` rather than after, as upstream
  does. `flip` materializes a contiguous copy, so flipping first keeps the HOP
  input a `movedim` view of a contiguous tensor, which
  `create_associative_scan_op`
  scans without transposing.

  Args:
    combine_fn: The user combine callable.
    leaves_xs: Flattened TPU input tensors.
    spec_xs: PyTree structure of `xs`.
    dim: Non-negative scan dimension.
    reverse: Whether to scan in reverse order.

  Returns:
    PyTree of scanned TPU tensors matching `spec_xs`.
  """
  # pylint: disable=protected-access
  if reverse:
    leaves_xs = [torch.flip(t, [dim]) for t in leaves_xs]
  leaves_xs = [torch.movedim(t, dim, 0) for t in leaves_xs]
  flat_combine_fn = functools.partial(
      _assoc_scan_module.wrap_combine_fn_flat,
      combine_fn=combine_fn,
      spec=spec_xs,
      num_leaves=len(leaves_xs),
  )

  def run_flattened_associative_scan(combine_fn, leaves_xs):
    return _assoc_scan_module.associative_scan_op(
        combine_fn, leaves_xs, additional_inputs=()
    )

  # Outside of Dynamo this compiles the HOP call, as upstream does. The result
  # already has the structure of `xs`.
  out = _hop_utils._maybe_compile_and_run_fn(
      run_flattened_associative_scan, flat_combine_fn, leaves_xs
  )
  out = pytree.tree_map(lambda t: torch.movedim(t, 0, dim), out)
  if reverse:
    out = pytree.tree_map(lambda t: torch.flip(t, [dim]), out)
  return out


def _tpu_associative_scan(
    combine_fn,
    xs,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
):
  """TPU-aware replacement for `torch._higher_order_ops.associative_scan`."""
  leaves_xs, spec_xs = pytree.tree_flatten(xs)
  canonical_dim = _tpu_scan_dim(leaves_xs, dim, reverse)
  if canonical_dim is not None:
    if not torch.compiler.is_dynamo_compiling() and _can_use_eager_fast_path(
        combine_fn, leaves_xs, combine_mode
    ):
      fn_key = _callable_cache_key(combine_fn)
      if fn_key is not None:
        return _eager_tpu_associative_scan(
            combine_fn,
            leaves_xs,
            spec_xs,
            canonical_dim,
            reverse,
            combine_mode,
            fn_key,
        )
    # Under Dynamo, and in eager mode when the fast path does not apply (e.g.
    # autograd), go through the HOP. `'generic'` mode is device-agnostic, so it
    # keeps using upstream's frontend below.
    if combine_mode == "pointwise" and callable(combine_fn):
      return _tpu_associative_scan_hop(
          combine_fn, leaves_xs, spec_xs, canonical_dim, reverse
      )
  return _SAVED_ORIG_ASSOCIATIVE_SCAN(
      combine_fn, xs, dim, reverse=reverse, combine_mode=combine_mode
  )


def _delegating_entrypoint(
    combine_fn,
    xs,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
):
  # This code runs with upstream's module globals, where
  # `_install_associative_scan_patch` defines `_tpu_associative_scan`.
  return _tpu_associative_scan(
      combine_fn, xs, dim, reverse=reverse, combine_mode=combine_mode
  )


def _install_associative_scan_patch():
  """Routes `torch._higher_order_ops.associative_scan` to the TPU fast path.

  This swaps the function's `__code__` in place rather than rebinding module
  attributes. That way references imported before `torch_tpu` also take the
  TPU path, and the function object's identity is unchanged.
  """
  orig_fn = _orig_associative_scan
  if getattr(orig_fn, "_tpu_patched", False):
    return
  orig_fn.__globals__["_tpu_associative_scan"] = _tpu_associative_scan
  orig_fn.__code__ = _delegating_entrypoint.__code__
  orig_fn.__defaults__ = _delegating_entrypoint.__defaults__
  setattr(orig_fn, "_tpu_patched", True)


try:
  # pylint: disable=g-import-not-at-top
  # pylint: disable=protected-access
  import torch._C

  for key in (
      torch._C.DispatchKey.XLA,
      torch._C.DispatchKey.PrivateUse1,
      torch._C.DispatchKey.Meta,
  ):
    torch.ops.higher_order.scan.py_impl(key)(_handle_scan_impl)

  for key in (
      torch._C.DispatchKey.XLA,
      torch._C.DispatchKey.PrivateUse1,
  ):
    torch.ops.higher_order.associative_scan.py_impl(key)(
        _handle_associative_scan_impl
    )

  _install_associative_scan_patch()
except (AttributeError, RuntimeError):
  # AttributeError: Safe to ignore if the environment lacks XLA/TPU support.
  # RuntimeError: Safe to ignore if the implementation is already registered.
  pass
