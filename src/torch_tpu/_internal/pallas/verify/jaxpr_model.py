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

"""Static structural and hardware model extraction from Pallas TPU jaxprs.

`check_kernel_static_models` traces a TorchTPU `@jax_op` or a raw JAX function
to a closed Jaxpr without initializing TPU backends, finds every `pallas_call`
and `mpmd_map` (`pl.core_map`, including inside `shard_map` for multi-chip
kernels), and unrolls it per mesh device and grid point into a
`KernelVerificationModel`:

* Implicit pipeline `BlockSpec` input copies (`HBM -> VMEM`) and output copies
  (`VMEM -> HBM`) at the boundaries of each sequential grid step, with symbolic
  evaluation of `index_map` Jaxprs to check out-of-bounds block slices and
  compute full-grid output write coverage (`unwritten_output`).
* Explicit kernel body operations (`get`, `swap`, `addupdate`, `masked_load`,
  `masked_swap`, `dma_start`, `dma_wait`, `semaphore_signal`, `semaphore_wait`,
  `get_barrier_semaphore`, `run_scoped`, `empty_ref`) across nested `cond`,
  `while`, and `scan` control-flow primitives up to `max_unrolled_iterations`,
  tracking scalar expressions to resolve hyper-rectangular `HyperRectSlice`
  regions and remote device/core IDs.
* Static structural and hardware checks (`out_of_bounds_access`,
  `invalid_device_id`, `hardware_constraint_violation`, `unwritten_output`,
  `low_precision_accumulation` including flow-insensitive `dot_general` lints,
  and `verification_incomplete`), exposed via `static_model_findings` and
  `check_kernel_static_models`.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set as AbstractSet,
)
import contextlib
import dataclasses
import enum
import itertools
import math
import traceback
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal.pallas import pallas as pallas_lib
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import fx_verifier

_Category = findings.FindingCategory
_Severity = findings.Severity

# Default bound on unrolled loop iterations and scan steps.
DEFAULT_MAX_UNROLLED_ITERATIONS = 32
# Maximum grid points walked for bounds, hardware, and coverage checks.
_MAX_COVERAGE_GRID_POINTS = 4096
# Bound on the cells examined when computing uncovered output regions.
_MAX_COVERAGE_CELLS = 1 << 20
# Hard cap on total equation steps walked per kernel model.
_MAX_TOTAL_EQUATION_STEPS = 200_000
# Message prefixes emitted when `check_kernel_static_models` fails while
# tracing `op_or_fn` to a ClosedJaxpr.
TRACE_INCOMPLETE_PREFIX = "could not trace the kernel for model checking"
TRACE_VALIDATION_ERROR_PREFIX = (
    "kernel failed Pallas static validation during tracing"
)
# Primitives whose sub-jaxprs are walked like inlined function calls.
_CALL_PRIMITIVES = frozenset({
    "checkpoint",
    "closed_call",
    "core_call",
    "custom_jvp_call",
    "custom_vjp_call",
    "custom_vjp_call_jaxpr",
    "eval_jaxpr",
    "jit",
    "pjit",
    "remat",
    "remat2",
})
# Primitives that synchronize execution across actors or DMAs.
_SYNC_PRIMITIVES = frozenset({
    "barrier",
    "dma_start",
    "dma_wait",
    "semaphore_signal",
    "semaphore_wait",
})


class _SliceAccessKind(enum.Enum):
  """Access kind used when validating strided `pl.ds` ref slices."""

  READ = "read"
  WRITE = "write"
  DMA = "dma"


class _CoverageOverflow:
  """Sentinel returned when cell decomposition exceeds `_MAX_COVERAGE_CELLS`."""


_COVERAGE_OVERFLOW = _CoverageOverflow()


@dataclasses.dataclass(frozen=True)
class SliceDim:
  """Half-open interval `[start, start + size)` along one buffer axis."""

  start: int
  size: int
  exact: bool = True

  @property
  def stop(self) -> int:
    """Exclusive upper bound `start + size`."""
    return self.start + self.size

  def __str__(self) -> str:
    if self.size == 1:
      return f"{self.start}"
    return f"{self.start}:{self.stop}"


@dataclasses.dataclass(frozen=True)
class HyperRectSlice:
  """Axis-aligned hyper-rectangular region inside a root buffer."""

  dims: tuple[SliceDim, ...]
  opaque: bool = False
  covers_full_buffer: bool = dataclasses.field(default=False, compare=False)

  def __str__(self) -> str:
    if not self.dims:
      return "[...]"
    return "[" + ", ".join(str(d) for d in self.dims) + "]"


@dataclasses.dataclass(frozen=True)
class _ActorKey:
  """Per-(device, core, parallel_grid_coords) stream key for tile state."""

  device_id: int
  core_id: int
  parallel_coords: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class StaticIssue:
  """A static problem found while symbolically unrolling a Pallas kernel."""

  category: findings.FindingCategory
  severity: findings.Severity
  message: str
  source_location: str | None = None


@dataclasses.dataclass(frozen=True)
class KernelVerificationModel:
  """Extracted structural and memory-region model of one `pallas_call`.

  Attributes:
    kernel_name: Name of the `pallas_call` (or of the op that calls it).
    num_devices: Number of devices the kernel runs on.
    grid: The kernel grid.
    dimension_semantics: "parallel" or "arbitrary" per grid axis.
    collective_id: `collective_id` from the compiler params, if any.
    static_issues: Problems found during extraction.
    incomplete_reasons: Why the model is bounded or approximate, if it is.
    uses_remote_operations: Whether the kernel issues DMAs or semaphore signals
      to other devices.
    uses_done_count_regular_dma: Whether the kernel issues DMAs using REGULAR
      semaphores in done-count mode (TPU v5p+).
    has_revisit_any_output: Whether any output operand opts into non-consecutive
      revisits via `RevisitMode.ANY`.
    has_verified_full_overwrite_revisit: Whether the kernel has a
      non-consecutive output block revisit where every revisiting run was
      statically proven to overwrite the full tile without reading stale values.
    core_axis_names: Core-mesh axis names when lowered via `core_map` /
      `mpmd_map`, or `()` for `pallas_call`.
  """

  kernel_name: str
  num_devices: int
  grid: tuple[int, ...]
  dimension_semantics: tuple[str, ...]
  collective_id: int | None
  static_issues: tuple[StaticIssue, ...] = ()
  incomplete_reasons: tuple[str, ...] = ()
  uses_remote_operations: bool = False
  uses_done_count_regular_dma: bool = False
  has_revisit_any_output: bool = False
  has_verified_full_overwrite_revisit: bool = False
  core_axis_names: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class RefOperand:
  """A ref operand of a DMA primitive and the transforms applied to it."""

  ref: Any
  transforms: tuple[Any, ...] = ()


@dataclasses.dataclass(frozen=True)
class DmaOperands:
  """Decoded operands of a `dma_start` or `dma_wait` equation."""

  src: RefOperand
  dst: RefOperand
  dst_sem: RefOperand | None
  src_sem: RefOperand | None
  device_id: Any


def _unwrap_ref_and_transforms(ref_or_view: Any) -> tuple[Any, tuple[Any, ...]]:
  """Unwraps nested `TransformedRef`s into `(root, transforms)`."""
  transforms: list[Any] = []
  cur = ref_or_view
  while hasattr(cur, "ref") and hasattr(cur, "transforms"):
    transforms = list(cur.transforms) + transforms
    cur = cur.ref
  return cur, tuple(transforms)


def _ref_operand(ref: Any, transforms: Any = ()) -> RefOperand | None:
  if ref is None:
    return None
  root, inner = _unwrap_ref_and_transforms(ref)
  return RefOperand(root, inner + tuple(transforms or ()))


def decode_dma_operands(operands: Sequence[Any]) -> DmaOperands:
  """Decodes unflattened DMA operands across JAX versions.

  JAX up to 0.9 flattens `(src, src_transforms, dst, dst_transforms, dst_sem,
  dst_sem_transforms, src_sem, src_sem_transforms, device_id)`; newer releases
  flatten `(src, dst, dst_sem, src_sem, device_id)` of `TransformedRef`s.

  Args:
    operands: The result of `eqn.params["tree"].unflatten(eqn.invars)`.

  Returns:
    The operands with their transforms.

  Raises:
    ValueError: If the operand layout is not recognized.
  """
  if len(operands) == 9:
    src, src_tr, dst, dst_tr, dst_sem, dst_sem_tr, src_sem, src_sem_tr, dev = (
        operands
    )
  elif len(operands) == 5:
    src, dst, dst_sem, src_sem, dev = operands
    src_tr = dst_tr = dst_sem_tr = src_sem_tr = ()
  else:
    raise ValueError(f"unsupported dma operand layout of {len(operands)} items")
  src_op = _ref_operand(src, src_tr)
  dst_op = _ref_operand(dst, dst_tr)
  if src_op is None or dst_op is None:
    raise ValueError("dma operands are missing a source or destination ref")
  return DmaOperands(
      src=src_op,
      dst=dst_op,
      dst_sem=_ref_operand(dst_sem, dst_sem_tr),
      src_sem=_ref_operand(src_sem, src_sem_tr),
      device_id=dev,
  )


@dataclasses.dataclass(frozen=True)
class _RefView:
  """A `Ref` or view of it, tracked back to its root buffer or semaphore."""

  buffer_id: str
  memory_space: str
  root_shape: tuple[int, ...]
  dtype: Any
  dims: tuple[SliceDim, ...]
  active_axes: tuple[int, ...]
  logical_shape: tuple[int, ...] | None = None
  # Per-root-axis step stride (or None if a slice origin along that axis is
  # runtime-dependent, in which case `dims[ax]` holds a sound bounding range).
  axis_strides: tuple[int | None, ...] = ()
  # None for data buffers, else "dma", "regular", or "barrier".
  sem_type: str | None = None
  # Set once a transform that changes the shape (reshape, bitcast) was applied;
  # later indexing is then not interpreted.
  opaque: bool = False
  covers_full_buffer: bool = True
  shared: bool = False
  # Core-mesh ownership metadata from `CoreMemorySpace` in multi-mesh
  # `mpmd_map`.
  target_core_axes: tuple[str, ...] = ()
  target_core_sizes: tuple[int | None, ...] = ()
  target_core_semantics: tuple[str, ...] = ()
  owner_core_id: int | None = None

  @property
  def view_shape(self) -> tuple[int, ...]:
    if self.logical_shape is not None:
      return self.logical_shape
    return tuple(self.dims[ax].size for ax in self.active_axes)

  @property
  def region(self) -> HyperRectSlice:
    return HyperRectSlice(
        self.dims,
        opaque=self.opaque,
        covers_full_buffer=self.covers_full_buffer,
    )

  @property
  def sem_index(self) -> tuple[int, ...] | None:
    """Element index of a 0D semaphore view, or None if unresolved."""
    if self.opaque or self.view_shape != ():
      return None
    if any(d.size != 1 or not d.exact for d in self.dims):
      return None
    return tuple(d.start for d in self.dims)

  @property
  def byte_count(self) -> int | None:
    numel = int(math.prod(self.view_shape))
    try:
      bits = int(jax.dtypes.itemsize_bits(self.dtype))
      return max(1, (numel * bits + 7) // 8)
    except (TypeError, ValueError):
      pass
    try:
      itemsize = int(np.dtype(self.dtype).itemsize)
      return max(1, numel * itemsize)
    except TypeError:
      return None


def _full_view(
    buffer_id: str,
    memory_space: str,
    shape: Sequence[int],
    dtype: Any,
    sem_type: str | None = None,
    *,
    shared: bool = False,
    target_core_axes: Sequence[str] = (),
    target_core_sizes: Sequence[int | None] = (),
    target_core_semantics: Sequence[str] = (),
    owner_core_id: int | None = None,
) -> _RefView:
  shape = tuple(int(s) for s in shape)
  return _RefView(
      buffer_id=buffer_id,
      memory_space=memory_space,
      root_shape=shape,
      dtype=np.int32 if sem_type is not None else dtype,
      dims=tuple(SliceDim(0, s) for s in shape),
      active_axes=tuple(range(len(shape))),
      logical_shape=shape,
      axis_strides=(1,) * len(shape),
      sem_type=sem_type,
      covers_full_buffer=True,
      shared=shared or memory_space in ("hbm", "vmem_shared"),
      target_core_axes=tuple(target_core_axes),
      target_core_sizes=tuple(target_core_sizes),
      target_core_semantics=tuple(target_core_semantics),
      owner_core_id=owner_core_id,
  )


@dataclasses.dataclass(frozen=True)
class _MeshContext:
  axis_names: tuple[str, ...] = ()
  mesh_shape: tuple[int, ...] = ()

  @property
  def num_devices(self) -> int:
    return int(math.prod(self.mesh_shape)) if self.mesh_shape else 1

  def iter_devices(self) -> list[tuple[int, tuple[int, ...]]]:
    if not self.axis_names:
      return [(0, ())]
    coords = itertools.product(*(range(s) for s in self.mesh_shape))
    return list(enumerate(coords))

  def coord_dict(self, coords: Sequence[int]) -> dict[Any, int]:
    return dict(zip(self.axis_names, coords))


def _is_jax_var(x: Any) -> bool:
  return type(x).__name__ == "Var"


def _is_drop_var(x: Any) -> bool:
  return type(x).__name__ == "DropVar"


def _is_abstract_ref(aval: Any) -> bool:
  return aval is not None and any(
      cls.__name__ == "AbstractRef" for cls in type(aval).__mro__
  )


def _unwrap_closed_jaxpr(sub: Any) -> Any:
  return sub.jaxpr if hasattr(sub, "jaxpr") else sub


def _summarize_source_info(eqn: Any) -> str | None:
  source_info = getattr(eqn, "source_info", None)
  tb = getattr(source_info, "traceback", None)
  if tb is None or not hasattr(tb, "as_python_traceback"):
    return None
  py_tb = tb.as_python_traceback()
  if py_tb is None:
    return None
  for frame in reversed(traceback.extract_tb(py_tb)):
    if (
        "/jax/_src/" not in frame.filename
        and "/jax/experimental/" not in frame.filename
    ):
      return f"{frame.filename}:{frame.lineno} ({frame.name})"
  return None


def _summarize_exception_traceback(exc: BaseException) -> str | None:
  tb = exc.__traceback__
  if tb is None:
    return None
  for frame in reversed(traceback.extract_tb(tb)):
    if (
        "/jax/_src/" not in frame.filename
        and "/jax/experimental/" not in frame.filename
        and "/torch_tpu/_internal/pallas/" not in frame.filename
    ):
      return f"{frame.filename}:{frame.lineno} ({frame.name})"
  return None


def _normalize_memory_space(ms: Any, *, default: str = "vmem") -> str:
  while hasattr(ms, "memory_space") and not isinstance(ms, (str, bytes)):
    nxt = getattr(ms, "memory_space")
    if nxt is ms:
      break
    ms = nxt
  # Compare values rather than members: `pl.MemorySpace` gained `DEFAULT`
  # after JAX 0.9.
  value = getattr(ms, "value", ms)
  if value is None or value == "default":
    return default
  if value in ("any", "dram"):
    return "hbm"
  if isinstance(value, str):
    return value
  return str(ms).lower()


def _detect_semaphore_type(aval: Any) -> str | None:
  dtype = getattr(aval, "dtype", None)
  ms = getattr(aval, "memory_space", None)
  norm_ms = _normalize_memory_space(ms, default="") if ms is not None else ""
  dtype_str = str(dtype).lower()
  dtype_cls_name = type(dtype).__name__
  if dtype_cls_name == "DMASemaphore" or dtype_str.startswith("dma_sem"):
    return "dma"
  if dtype_cls_name == "BarrierSemaphore" or dtype_str.startswith(
      "barrier_sem"
  ):
    return "barrier"
  if (
      dtype_cls_name in ("Semaphore", "RegularSemaphore")
      or ms == pltpu.MemorySpace.SEMAPHORE
      or norm_ms == "semaphore_mem"
      or dtype_str in ("semaphore", "regular_semaphore")
      or dtype_str.startswith("sem<")
  ):
    return "regular"
  return None


def _is_sparse_core_type(core_type: Any) -> bool:
  if core_type is None:
    return False
  name = str(getattr(core_type, "name", core_type)).upper()
  val = str(getattr(core_type, "value", core_type)).lower()
  return name.startswith("SC_") or "sparse" in val or "subcore" in val


def _eval_atom(
    atom: Any, scalar_env: Mapping[Any, int | bool]
) -> int | bool | None:
  """Returns the concrete integer or Boolean value of an atom, if known."""
  if isinstance(atom, (bool, int)):
    return atom
  if isinstance(atom, (np.generic, np.ndarray)) and np.shape(atom) == ():
    if jnp.issubdtype(atom.dtype, jnp.bool_):
      return bool(atom.item())
    if jnp.issubdtype(atom.dtype, jnp.integer):
      return int(atom.item())
    return None
  if type(atom).__name__ == "Literal" and hasattr(atom, "val"):
    return _eval_atom(atom.val, scalar_env)
  if isinstance(atom, jax.Array) and atom.shape == ():
    if jnp.issubdtype(atom.dtype, jnp.bool_):
      return bool(atom.item())
    if jnp.issubdtype(atom.dtype, jnp.integer):
      return int(atom.item())
    return None
  if _is_jax_var(atom):
    return scalar_env.get(atom)
  return None


def _int_bit_width(dtype: Any) -> int:
  try:
    if jnp.issubdtype(dtype, jnp.integer):
      return int(jax.dtypes.itemsize_bits(dtype))
  except (TypeError, ValueError):
    pass
  return 64


def _wrap_int_to_dtype(val: int, dtype: Any) -> int:
  """Wraps `val` to the fixed-width two's-complement range of `dtype`."""
  try:
    if not jnp.issubdtype(dtype, jnp.integer):
      return int(val)
    bits = int(jax.dtypes.itemsize_bits(dtype))
    is_signed = jnp.issubdtype(dtype, jnp.signedinteger)
  except (TypeError, ValueError):
    return int(val)
  mask = (1 << bits) - 1
  uval = int(val) & mask
  if is_signed and (uval & (1 << (bits - 1))):
    return uval - (1 << bits)
  return uval


def _int_binary(name: str, a: int, b: int, *, dtype: Any = None) -> int | None:
  """Integer semantics of `lax` binary ops on scalars (C-style division)."""
  if name in ("add", "add_any"):
    return a + b
  if name == "sub":
    return a - b
  if name == "mul":
    return a * b
  if name in ("div", "rem"):
    if b == 0:
      return None
    quotient = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
    return quotient if name == "div" else a - quotient * b
  if name == "min":
    return min(a, b)
  if name == "max":
    return max(a, b)
  bits = _int_bit_width(dtype)
  if name == "shift_left":
    if b < 0 or b >= bits:
      return 0
    return a << b
  if name == "shift_right_logical":
    if b < 0 or b >= bits:
      return 0
    ua = int(a) & ((1 << bits) - 1)
    return ua >> b
  if name == "shift_right_arithmetic":
    sa = _wrap_int_to_dtype(a, dtype) if dtype is not None else a
    if b < 0 or b >= bits:
      return -1 if sa < 0 else 0
    return sa >> b
  return None


_COMPARISONS: dict[str, Callable[[int, int], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}
_PASSTHROUGH = frozenset({
    "broadcast_in_dim",
    "convert_element_type",
    "copy",
    "copy_p",
    "multiple_of",
    "reshape",
    "squeeze",
})


def _eval_scalar_op(
    name: str,
    vals: Sequence[int | bool],
    is_bool: bool,
    params: Any,
    *,
    out_dtype: Any = None,
) -> int | bool | None:
  """Evaluates a scalar `lax` primitive on concrete operands."""
  if name in _COMPARISONS and len(vals) == 2:
    return _COMPARISONS[name](int(vals[0]), int(vals[1]))
  res: int | bool | None = None
  if name in _PASSTHROUGH:
    res = bool(vals[0]) if is_bool else int(vals[0])
  elif name in ("and", "or", "xor"):
    if is_bool:
      a, b = bool(vals[0]), bool(vals[1])
      return {"and": a and b, "or": a | b, "xor": a != b}[name]
    a, b = int(vals[0]), int(vals[1])
    res = {"and": a & b, "or": a | b, "xor": a ^ b}[name]
  elif name == "not":
    if is_bool:
      return not bool(vals[0])
    res = ~int(vals[0])
  elif name == "neg":
    res = -int(vals[0])
  elif name == "abs":
    res = abs(int(vals[0]))
  elif name == "sign":
    v = int(vals[0])
    res = (v > 0) - (v < 0)
  elif name == "integer_pow":
    exp = int(params["y"])
    if exp < 0:
      return None
    res = int(vals[0]) ** exp
  elif name == "clamp" and len(vals) == 3:
    res = min(max(int(vals[1]), int(vals[0])), int(vals[2]))
  elif name == "select_n":
    which = int(vals[0])
    choices = vals[1:]
    res = choices[which] if 0 <= which < len(choices) else None
  elif len(vals) == 2:
    res = _int_binary(name, int(vals[0]), int(vals[1]), dtype=out_dtype)
  if res is None:
    return None
  if is_bool:
    return bool(res)
  return (
      _wrap_int_to_dtype(int(res), out_dtype)
      if out_dtype is not None
      else int(res)
  )


def _sym_atom(
    atom: Any,
    scalar_env: Mapping[Any, int | bool],
    sym_env: Mapping[Any, Any],
) -> Any:
  val = _eval_atom(atom, scalar_env)
  if val is not None:
    return ("const", int(val))
  if _is_jax_var(atom):
    return sym_env.get(atom, ("var", id(atom)))
  return ("atom", id(atom))


def _negate_sym_pred(expr: Any) -> Any:
  if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == "not":
    return expr[1]
  return ("not", expr)


def _record_symbolic_scalar_eqn(
    eqn: Any,
    scalar_env: Mapping[Any, int | bool],
    sym_env: MutableMapping[Any, Any] | None,
) -> None:
  """Tracks canonical symbolic boolean/comparison expressions for `cond`."""
  if sym_env is None or len(eqn.outvars) != 1:
    return
  out_var = eqn.outvars[0]
  if _is_drop_var(out_var):
    return
  name = eqn.primitive.name
  out_aval = getattr(out_var, "aval", None)
  is_bool = isinstance(out_aval, jax.core.ShapedArray) and (
      out_aval.dtype == np.bool_
  )
  if name in _PASSTHROUGH and len(eqn.invars) == 1:
    sym_env[out_var] = _sym_atom(eqn.invars[0], scalar_env, sym_env)
    return
  if name == "not" and len(eqn.invars) == 1 and is_bool:
    sym_env[out_var] = _negate_sym_pred(
        _sym_atom(eqn.invars[0], scalar_env, sym_env)
    )
    return
  if name in _COMPARISONS and len(eqn.invars) == 2:
    lhs = _sym_atom(eqn.invars[0], scalar_env, sym_env)
    rhs = _sym_atom(eqn.invars[1], scalar_env, sym_env)
    in0_aval = getattr(eqn.invars[0], "aval", None)
    in0_bool = isinstance(in0_aval, jax.core.ShapedArray) and (
        in0_aval.dtype == np.bool_
    )
    if in0_bool and rhs == ("const", 0) and name in ("eq", "ne"):
      sym_env[out_var] = _negate_sym_pred(lhs) if name == "eq" else lhs
      return
    if in0_bool and rhs == ("const", 1) and name in ("eq", "ne"):
      sym_env[out_var] = lhs if name == "eq" else _negate_sym_pred(lhs)
      return
    if name == "lt":
      sym_env[out_var] = ("lt", lhs, rhs)
    elif name == "ge":
      sym_env[out_var] = _negate_sym_pred(("lt", lhs, rhs))
    elif name == "gt":
      sym_env[out_var] = ("lt", rhs, lhs)
    elif name == "le":
      sym_env[out_var] = _negate_sym_pred(("lt", rhs, lhs))
    elif name == "eq":
      pair = (lhs, rhs) if str(lhs) <= str(rhs) else (rhs, lhs)
      sym_env[out_var] = ("eq", pair[0], pair[1])
    elif name == "ne":
      pair = (lhs, rhs) if str(lhs) <= str(rhs) else (rhs, lhs)
      sym_env[out_var] = _negate_sym_pred(("eq", pair[0], pair[1]))


def _try_eval_scalar_eqn(
    eqn: Any,
    scalar_env: MutableMapping[Any, int | bool],
    *,
    grid_coords: Sequence[int],
    grid_sizes: Sequence[int | None],
    axis_env: Mapping[Any, int],
    sym_env: MutableMapping[Any, Any] | None = None,
) -> bool:
  """Evaluates a scalar integer/Boolean equation into `scalar_env`."""
  name = eqn.primitive.name
  if name in ("program_id", "num_programs"):
    axis = int(eqn.params["axis"])
    source = grid_coords if name == "program_id" else grid_sizes
    if 0 <= axis < len(source):
      val = source[axis]
      if val is not None:
        scalar_env[eqn.outvars[0]] = val
        return True
    return False
  if name == "axis_index":
    ax_name = eqn.params["axis_name"]
    if ax_name in axis_env:
      scalar_env[eqn.outvars[0]] = axis_env[ax_name]
      return True
    return False
  if name in _CALL_PRIMITIVES:
    return _try_eval_scalar_call(
        eqn,
        scalar_env,
        grid_coords=grid_coords,
        grid_sizes=grid_sizes,
        axis_env=axis_env,
    )
  if len(eqn.outvars) != 1:
    return False
  out_aval = getattr(eqn.outvars[0], "aval", None)
  if not isinstance(out_aval, jax.core.ShapedArray) or out_aval.shape != ():
    return False
  if not (
      out_aval.dtype == np.bool_ or jnp.issubdtype(out_aval.dtype, jnp.integer)
  ):
    return False
  for v in eqn.invars:
    in_aval = getattr(v, "aval", None)
    if not isinstance(in_aval, jax.core.ShapedArray) or in_aval.shape != ():
      return False
    if not (
        in_aval.dtype == np.bool_ or jnp.issubdtype(in_aval.dtype, jnp.integer)
    ):
      return False
  vals = [_eval_atom(v, scalar_env) for v in eqn.invars]
  is_bool = out_aval.dtype == np.bool_
  # Short-circuit boolean and selection ops when one operand determines the
  # result even if the other operand depends on runtime data.
  if name == "and" and is_bool and any(v is not None and not v for v in vals):
    scalar_env[eqn.outvars[0]] = False
    return True
  if name == "or" and is_bool and any(v is not None and v == 1 for v in vals):
    scalar_env[eqn.outvars[0]] = True
    return True
  if (
      name == "mul"
      and not is_bool
      and any(v == 0 for v in vals if v is not None)
  ):
    scalar_env[eqn.outvars[0]] = 0
    return True
  if name == "select_n" and vals and vals[0] is not None:
    which = int(vals[0])
    choices = vals[1:]
    if 0 <= which < len(choices):
      chosen = choices[which]
      if chosen is not None:
        scalar_env[eqn.outvars[0]] = (
            bool(chosen)
            if is_bool
            else _wrap_int_to_dtype(int(chosen), out_aval.dtype)
        )
        return True
  concrete = [v for v in vals if v is not None]
  if not vals or len(concrete) != len(vals):
    _record_symbolic_scalar_eqn(eqn, scalar_env, sym_env)
    return False
  res = _eval_scalar_op(
      name, concrete, is_bool, eqn.params, out_dtype=out_aval.dtype
  )
  if res is None:
    return False
  scalar_env[eqn.outvars[0]] = res
  return True


def _try_eval_scalar_call(
    eqn: Any,
    scalar_env: MutableMapping[Any, int | bool],
    *,
    grid_coords: Sequence[int],
    grid_sizes: Sequence[int | None],
    axis_env: Mapping[Any, int],
) -> bool:
  """Evaluates a nested call, such as `jnp.floor_divide`, on scalars.

  Only calls with known scalar operands whose equations are all scalar
  arithmetic are evaluated, so calls that access refs are still walked.

  Args:
    eqn: The call equation.
    scalar_env: Known scalar values; receives the call's results.
    grid_coords: Grid point being walked.
    grid_sizes: Grid sizes (`None` for runtime-sized axes).
    axis_env: Device coordinates by mesh axis name.

  Returns:
    Whether every result of the call was evaluated.
  """
  vals = [_eval_atom(v, scalar_env) for v in eqn.invars]
  known = [v for v in vals if v is not None]
  if not eqn.outvars or len(known) != len(vals):
    return False
  for param in eqn.params.values():
    inner = _unwrap_closed_jaxpr(param)
    if not hasattr(inner, "eqns") or len(inner.invars) != len(known):
      continue
    inner_env: dict[Any, int | bool] = dict(zip(inner.invars, known))
    for var, const in zip(inner.constvars, getattr(param, "consts", ())):
      value = _eval_atom(const, {})
      if value is not None:
        inner_env[var] = value
    for inner_eqn in inner.eqns:
      if not _try_eval_scalar_eqn(
          inner_eqn,
          inner_env,
          grid_coords=grid_coords,
          grid_sizes=grid_sizes,
          axis_env=axis_env,
      ):
        return False
    outs = [_eval_atom(v, inner_env) for v in inner.outvars]
    if len(outs) != len(eqn.outvars) or any(o is None for o in outs):
      return False
    for outer, value in zip(eqn.outvars, outs):
      if value is not None and not _is_drop_var(outer):
        scalar_env[outer] = value
    return True
  return False


def _apply_opaque_transform_type(
    tr: Any,
    logical_shape: Sequence[int],
    dtype: Any,
    *,
    is_semaphore: bool,
) -> tuple[list[int], Any]:
  """Updates `(logical_shape, dtype)` through an opaque ref transform."""
  if not is_semaphore and hasattr(tr, "transform_type"):
    try:
      transformed = tr.transform_type(
          jax.core.ShapedArray(tuple(int(s) for s in logical_shape), dtype)
      )
      t_shape = getattr(transformed, "shape", None)
      t_dtype = getattr(transformed, "dtype", None)
      if t_shape is not None and all(
          isinstance(s, (int, np.integer)) for s in t_shape
      ):
        return [int(s) for s in t_shape], (
            t_dtype if t_dtype is not None else dtype
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
      # Stand-in `ShapedArray`s omit Mosaic-internal aval metadata on some
      # transforms; falling back to `tr.shape`/`tr.dtype` below is safe.
      pass
  tr_dtype = getattr(tr, "dtype", None)
  new_dtype = tr_dtype if tr_dtype is not None and not is_semaphore else dtype
  tr_shape = getattr(tr, "shape", None)
  if tr_shape is not None and all(
      isinstance(s, (int, np.integer)) for s in tr_shape
  ):
    return [int(s) for s in tr_shape], new_dtype
  return list(logical_shape), new_dtype


def _apply_transforms(
    base: _RefView,
    transforms: Sequence[Any],
    scalar_env: Mapping[Any, int | bool],
    *,
    access_kind: _SliceAccessKind = _SliceAccessKind.WRITE,
) -> tuple[_RefView, str | None]:
  """Composes indexing transforms onto `base`.

  Args:
    base: Base reference view before transforms.
    transforms: Sequence of Pallas ref transforms (`NDIndexer`, transpose,
      etc.).
    scalar_env: Known scalar integer/boolean values for evaluating index atoms.
    access_kind: Kind of access (`READ`, `WRITE`, or `DMA`); Mosaic
      `StridedLoadOp` allows `stride == 0` for sublane broadcast on `READ`.

  Returns:
    The view and, if a statically known index is out of bounds, a description
    of the violation.
  """
  dims = list(base.dims)
  active_axes = list(base.active_axes)
  axis_strides: list[int | None] = (
      list(base.axis_strides)
      if len(base.axis_strides) == len(dims)
      else [1 if d.exact else None for d in dims]
  )
  logical_shape = list(
      base.logical_shape
      if base.logical_shape is not None
      else (dims[ax].size for ax in active_axes)
  )
  dtype = base.dtype
  memory_space = base.memory_space
  opaque = base.opaque
  covers_full_buffer = base.covers_full_buffer
  violation: str | None = None
  for tr in transforms:
    tr_name = type(tr).__name__
    if tr_name == "AccessAssumptionTransform":
      continue
    if tr_name == "MemorySpaceCastTransform" and hasattr(tr, "memory_space"):
      memory_space = _normalize_memory_space(
          tr.memory_space, default=memory_space
      )
      continue
    perm_attr = getattr(tr, "permutation", getattr(tr, "perm", None))
    if tr_name == "TransposeTransform" and perm_attr is not None:
      perm = tuple(int(p) for p in perm_attr)
      if not opaque and len(perm) == len(active_axes):
        active_axes = [active_axes[p] for p in perm]
      if len(perm) == len(logical_shape):
        logical_shape = [logical_shape[p] for p in perm]
      continue
    if tr_name != "NDIndexer" or not hasattr(tr, "indices"):
      opaque = True  # Reshapes and bitcasts change the indexed shape.
      logical_shape, dtype = _apply_opaque_transform_type(
          tr, logical_shape, dtype, is_semaphore=base.sem_type is not None
      )
      continue
    if opaque:
      covers_full_buffer = False
      logical_shape, dtype = _apply_opaque_transform_type(
          tr, logical_shape, dtype, is_semaphore=base.sem_type is not None
      )
      continue
    new_active: list[int] = []
    new_logical: list[int] = []
    int_indexer_inserted = False
    int_indexer_shape = getattr(tr, "int_indexer_shape", None)
    num_indexed = min(len(active_axes), len(tr.indices))
    for view_pos, (ax_pos, idx_entry) in enumerate(
        zip(active_axes[:num_indexed], tr.indices[:num_indexed])
    ):
      parent = dims[ax_pos]
      parent_stride = axis_strides[ax_pos]
      axis_size = (
          logical_shape[view_pos]
          if view_pos < len(logical_shape)
          else parent.size
      )
      if isinstance(idx_entry, pl.Slice):
        start = _eval_atom(idx_entry.start, scalar_env)
        size = _eval_atom(idx_entry.size, scalar_env)
        raw_stride = getattr(idx_entry, "stride", None)
        stride_val = (
            1 if raw_stride is None else _eval_atom(raw_stride, scalar_env)
        )
        stride = 1 if stride_val is None else int(stride_val)
        min_stride = 0 if access_kind is _SliceAccessKind.READ else 1
        if stride < min_stride:
          bound_desc = "non-negative" if min_stride == 0 else "positive"
          violation = violation or (
              f"slice stride {stride} along axis {ax_pos} must be {bound_desc}"
          )
          dims[ax_pos] = SliceDim(parent.start, 0, exact=False)
          axis_strides[ax_pos] = None
          covers_full_buffer = False
        elif size is not None and int(size) < 0:
          violation = violation or (
              f"slice size {int(size)} along axis {ax_pos} is negative"
          )
          dims[ax_pos] = SliceDim(parent.start, 0, exact=False)
          axis_strides[ax_pos] = None
          covers_full_buffer = False
        elif size is not None and int(size) == 0:
          if start is not None and not 0 <= int(start) <= axis_size:
            violation = violation or (
                f"slice {int(start)}:{int(start)} is outside axis {ax_pos} of"
                f" size {axis_size}"
            )
          if parent_stride is not None and start is not None:
            dims[ax_pos] = SliceDim(
                parent.start + int(start) * parent_stride,
                0,
                exact=parent.exact,
            )
          else:
            dims[ax_pos] = SliceDim(parent.start, 0, exact=False)
            axis_strides[ax_pos] = None
          covers_full_buffer = False
        elif start is not None and size is not None:
          span = (int(size) - 1) * stride + 1
          if int(start) < 0 or int(start) + span > axis_size:
            violation = violation or (
                f"slice {int(start)}:{int(start) + span} is outside axis"
                f" {ax_pos} of size {axis_size}"
            )
            dims[ax_pos] = SliceDim(parent.start, 0, exact=False)
            axis_strides[ax_pos] = None
          elif parent_stride is None or stride_val is None:
            dims[ax_pos] = SliceDim(parent.start, parent.size, exact=False)
            axis_strides[ax_pos] = None
          else:
            new_stride = parent_stride * stride
            root_start = parent.start + int(start) * parent_stride
            root_span = (int(size) - 1) * new_stride + 1
            dims[ax_pos] = SliceDim(
                root_start,
                root_span,
                exact=parent.exact and new_stride in (0, 1),
            )
            axis_strides[ax_pos] = new_stride
          if (
              int(start) != 0
              or int(size) != axis_size
              or stride != 1
              or stride_val is None
          ):
            covers_full_buffer = False
        else:
          if size is not None:
            span = (int(size) - 1) * stride + 1
            if span > axis_size:
              violation = violation or (
                  f"slice of span {span} exceeds axis {ax_pos} of size"
                  f" {axis_size}"
              )
          if start is not None and not 0 <= int(start) < axis_size:
            violation = violation or (
                f"slice start {int(start)} is outside axis {ax_pos} of size"
                f" {axis_size}"
            )
          dims[ax_pos] = SliceDim(parent.start, parent.size, exact=False)
          axis_strides[ax_pos] = None
          covers_full_buffer = False
        new_active.append(ax_pos)
        new_logical.append(max(0, int(size)) if size is not None else axis_size)
        continue
      idx_aval = getattr(idx_entry, "aval", None)
      if isinstance(idx_aval, jax.core.ShapedArray) and idx_aval.shape:
        dims[ax_pos] = SliceDim(parent.start, parent.size, exact=False)
        axis_strides[ax_pos] = None
        covers_full_buffer = False
        new_active.append(ax_pos)  # Gather: the whole axis may be read.
        if not int_indexer_inserted:
          int_indexer_inserted = True
          bcast_shape = (
              int_indexer_shape
              if int_indexer_shape is not None
              else idx_aval.shape
          )
          new_logical.extend(int(s) for s in bcast_shape)
        continue
      k = _eval_atom(idx_entry, scalar_env)
      if k is not None:
        idx = int(k)
        if not 0 <= idx < axis_size:
          violation = violation or (
              f"index {idx} is outside axis {ax_pos} of size {axis_size}"
          )
          dims[ax_pos] = SliceDim(parent.start, 0, exact=False)
          axis_strides[ax_pos] = None
        elif parent_stride is None:
          dims[ax_pos] = SliceDim(parent.start, parent.size, exact=False)
        else:
          dims[ax_pos] = SliceDim(
              parent.start + idx * parent_stride,
              1,
              exact=parent.exact,
          )
        if axis_size != 1 or idx != 0:
          covers_full_buffer = False
      else:
        dims[ax_pos] = SliceDim(parent.start, parent.size, exact=False)
        axis_strides[ax_pos] = None
        covers_full_buffer = False
    if num_indexed < len(active_axes):
      new_active.extend(active_axes[num_indexed:])
      new_logical.extend(logical_shape[num_indexed:])
    active_axes = new_active
    logical_shape = new_logical
  view = dataclasses.replace(
      base,
      memory_space=memory_space,
      dtype=dtype,
      dims=tuple(dims),
      active_axes=tuple(active_axes),
      logical_shape=tuple(logical_shape),
      axis_strides=tuple(axis_strides),
      opaque=opaque,
      covers_full_buffer=covers_full_buffer,
  )
  return view, violation


def _extract_mosaic_compiler_params(compiler_params: Any) -> Any:
  if compiler_params is None:
    return None
  if isinstance(compiler_params, Mapping):
    return compiler_params.get("mosaic_tpu") or compiler_params.get("mosaic")
  return compiler_params


def _kernel_name(params: Mapping[str, Any]) -> str:
  for key in ("name", "name_and_src_info"):
    value = params.get(key)
    name = getattr(value, "name", value)
    if isinstance(name, str) and name:
      return name
  return "pallas_call"


class _KernelModelBuilder:
  """Accumulates events and static issues across devices and grid points."""

  def __init__(
      self,
      *,
      kernel_name: str,
      num_devices: int,
      grid: tuple[int, ...],
      dimension_semantics: tuple[str, ...],
      collective_id: int | None,
      max_unrolled_iterations: int,
      tpu_version: findings.TpuVersion = fx_verifier.DEFAULT_TPU_VERSION,
      is_sparse_core: bool = False,
      core_axis_names: tuple[str, ...] = (),
  ):
    self.kernel_name = kernel_name
    self.num_devices = num_devices
    self.grid = grid
    self.dimension_semantics = dimension_semantics
    self.collective_id = collective_id
    self.max_unrolled_iterations = max_unrolled_iterations
    self.tpu_version = tpu_version
    self.is_sparse_core = is_sparse_core
    self.core_axis_names = core_axis_names
    self.num_tpu_cores = int(
        getattr(fx_verifier.abstract_tpu_device(tpu_version), "num_cores", 1)
    )

    self.step_count = 0
    self.static_issues: list[StaticIssue] = []
    self._issue_keys: set[tuple[findings.FindingCategory, str]] = set()
    self._dedup_keys: set[tuple[Any, ...]] = set()
    self.incomplete_reasons: list[str] = []
    self.coverage_known = True
    self.uses_remote_operations = False
    self.has_revisit_any_output = False
    self.has_verified_full_overwrite_revisit = False
    self.uses_done_count_regular_dma = False
    # Monotone per-buffer mutation counter (never rolled back across branches).
    self._may_write_epoch: dict[str, int] = {}
    self._has_sync: dict[int, bool] = {}
    # (device id, output buffer id) -> regions written by stores or DMAs.
    self.output_writes: dict[tuple[int, str], list[HyperRectSlice]] = {}
    # (stream_actor, operand index) -> list of
    #   (grid_coords, window_key, region, was_read_before_full_write, writes).
    self.output_block_visits: dict[
        tuple[_ActorKey, int],
        list[
            tuple[
                tuple[int, ...],
                tuple[HyperRectSlice, tuple[int, ...]],
                HyperRectSlice,
                bool,
                tuple[HyperRectSlice, ...],
            ]
        ],
    ] = {}
    # (device id, operand index, block_region) ->
    #   list of (actor, grid_coords, tile_writes).
    self.parallel_output_blocks: dict[
        tuple[int, int, HyperRectSlice],
        list[
            tuple[
                _ActorKey,
                tuple[int, ...],
                list[HyperRectSlice],
            ]
        ],
    ] = {}
    # (device id, operand index) -> visited block regions for aliased VMEM
    # outputs.
    self.aliased_pipelined_blocks: dict[
        tuple[int, int], list[HyperRectSlice]
    ] = {}
    # Active output block and accumulated tile writes on each (actor, op_index)
    # across consecutive reduction steps targeting the same output block.
    self.active_output_tile_block: dict[
        tuple[_ActorKey, int], tuple[HyperRectSlice, tuple[int, ...]]
    ] = {}
    self.active_output_tile_writes: dict[
        tuple[_ActorKey, int], list[HyperRectSlice]
    ] = {}
    self.revisit_any_tile_writes: dict[
        tuple[_ActorKey, int, tuple[HyperRectSlice, tuple[int, ...]]],
        list[HyperRectSlice],
    ] = {}
    # (actor, canonical buffer id) written in an earlier grid or loop step.
    self.prior_step_written_buffers: set[tuple[_ActorKey, str]] = set()
    self._scope_counters: dict[tuple[Any, ...], int] = {}
    self._source_locations: dict[int, str | None] = {}

  def next_scope_index(self, key: tuple[Any, ...]) -> int:
    """Returns 0, 1, ... for successive calls with the same `key`."""
    index = self._scope_counters.get(key, -1) + 1
    self._scope_counters[key] = index
    return index

  def source_location(self, eqn: Any) -> str | None:
    """Returns the user source location of `eqn`, cached across grid points."""
    key = id(eqn)
    if key not in self._source_locations:
      self._source_locations[key] = _summarize_source_info(eqn)
    return self._source_locations[key]

  def may_write_epoch(self, buffer_id: str) -> int:
    """Returns a monotone version counter for writes targeting `buffer_id`."""
    return self._may_write_epoch.get(buffer_id, 0)

  def bump_may_write_epoch(self, buffer_id: str) -> None:
    """Records a potential write to `buffer_id` without branch rollback."""
    self._may_write_epoch[buffer_id] = (
        self._may_write_epoch.get(buffer_id, 0) + 1
    )

  def record_issue(
      self,
      category: findings.FindingCategory,
      message: str,
      loc: str | None = None,
      severity: findings.Severity = _Severity.ERROR,
      *,
      dedup_key: tuple[Any, ...] | None = None,
  ) -> None:
    if dedup_key is not None:
      if dedup_key in self._dedup_keys:
        return
      self._dedup_keys.add(dedup_key)
    if (category, message) not in self._issue_keys:
      self._issue_keys.add((category, message))
      self.static_issues.append(StaticIssue(category, severity, message, loc))

  def note_incomplete(
      self, reason: str, *, affects_coverage: bool = False
  ) -> None:
    if reason not in self.incomplete_reasons:
      self.incomplete_reasons.append(reason)
    if affects_coverage:
      self.coverage_known = False

  def has_sync(self, jaxpr: Any) -> bool:
    """Returns whether `jaxpr` or a nested jaxpr uses semaphores."""
    key = id(jaxpr)
    if key not in self._has_sync:
      self._has_sync[key] = any(
          eqn.primitive.name in _SYNC_PRIMITIVES
          or any(self.has_sync(sub) for sub in _sub_jaxprs(eqn))
          for eqn in jaxpr.eqns
      )
    return self._has_sync[key]

  def record_output_write(
      self, device_id: int, buffer_id: str, region: HyperRectSlice
  ) -> None:
    if not (
        buffer_id.startswith("arg_out_")
        or buffer_id.startswith("pipelined_out_")
    ):
      return
    if region.opaque or not all(d.exact for d in region.dims):
      self.note_incomplete(
          "an output write uses runtime-dependent or strided indices, so"
          " unwritten-output coverage is not checked",
          affects_coverage=True,
      )
      return
    self.output_writes.setdefault((device_id, buffer_id), []).append(region)

  def build(self) -> KernelVerificationModel:
    return KernelVerificationModel(
        kernel_name=self.kernel_name,
        num_devices=self.num_devices,
        grid=self.grid,
        dimension_semantics=self.dimension_semantics,
        collective_id=self.collective_id,
        static_issues=tuple(self.static_issues),
        incomplete_reasons=tuple(self.incomplete_reasons),
        uses_remote_operations=self.uses_remote_operations,
        uses_done_count_regular_dma=self.uses_done_count_regular_dma,
        has_revisit_any_output=self.has_revisit_any_output,
        has_verified_full_overwrite_revisit=(
            self.has_verified_full_overwrite_revisit
        ),
        core_axis_names=self.core_axis_names,
    )


@dataclasses.dataclass
class _Env:
  """Refs, concrete scalars, concolic scalars, and symbolic expressions."""

  refs: dict[Any, _RefView]
  scalars: dict[Any, int | bool]
  sym_exprs: dict[Any, Any] = dataclasses.field(default_factory=dict)
  concolic_scalars: dict[Any, int | bool] = dataclasses.field(
      default_factory=dict
  )

  def child(self) -> _Env:
    return _Env(
        dict(self.refs),
        dict(self.scalars),
        dict(self.sym_exprs),
        dict(self.concolic_scalars),
    )

  def bind(self, outer: Any, inner: Any, outer_env: _Env) -> None:
    """Binds `inner` to the ref or scalar value of `outer` in `outer_env`."""
    if _is_jax_var(outer) and outer in outer_env.refs:
      self.refs[inner] = outer_env.refs[outer]
    val = _eval_atom(outer, outer_env.scalars)
    if val is not None:
      self.scalars[inner] = val
      self.concolic_scalars.pop(inner, None)
      self.sym_exprs.pop(inner, None)
      return
    if _is_jax_var(outer):
      if outer in outer_env.concolic_scalars:
        self.concolic_scalars[inner] = outer_env.concolic_scalars[outer]
      if outer in outer_env.sym_exprs:
        self.sym_exprs[inner] = outer_env.sym_exprs[outer]


@dataclasses.dataclass
class _WalkContext:
  """Position of a jaxpr walk: device, grid point, actor, and scope."""

  builder: _KernelModelBuilder
  actor: _ActorKey
  device_id: int
  device_coords: tuple[int, ...]
  mesh_ctx: _MeshContext
  grid_coords: tuple[int, ...]
  grid_sizes: tuple[int | None, ...]
  scope_path: str
  touched_refs: set[str]
  # Buffer id -> regions written locally (by stores or local DMAs) at this
  # grid point.
  written: dict[str, list[HyperRectSlice]]
  extra_axis_env: Mapping[Any, int] = dataclasses.field(default_factory=dict)
  core_axis_names: tuple[str, ...] = ()
  all_core_axis_sizes: Mapping[str, int] = dataclasses.field(
      default_factory=dict
  )
  dimension_semantics: tuple[str, ...] | None = None
  is_sparse_core: bool | None = None
  remote_written: dict[tuple[int, str], list[HyperRectSlice]] = (
      dataclasses.field(default_factory=dict)
  )
  speculative: bool = False
  # Unaliased output buffer/tile id -> (op.describe(), op.index).
  unaliased_output_buffers: dict[str, tuple[str, int]] = dataclasses.field(
      default_factory=dict
  )
  # Buffers read anywhere during the current grid point.
  read_refs: set[str] = dataclasses.field(default_factory=set)
  # Buffers read before being completely overwritten in the current grid point.
  read_before_full_write: set[str] = dataclasses.field(default_factory=set)
  # Buffers read in the current step/iteration (for cross-step sub-32-bit float
  # accumulation checks).
  read_in_step: set[str] = dataclasses.field(default_factory=set)
  # Complementary runtime-predicate branch writes in the current scope:
  # (base_pred, truth_value) -> {buf_id: [regions]}.
  guarded_written: dict[tuple[Any, bool], dict[str, list[HyperRectSlice]]] = (
      dataclasses.field(default_factory=dict)
  )
  guarded_output_writes: dict[
      tuple[Any, bool], dict[tuple[int, str], list[HyperRectSlice]]
  ] = dataclasses.field(default_factory=dict)
  vmapped_dims: frozenset[int] = frozenset()
  prefetch_arrays: Mapping[int, _ConcreteArray] = dataclasses.field(
      default_factory=dict
  )

  @property
  def effective_dimension_semantics(self) -> tuple[str, ...]:
    if self.dimension_semantics is not None:
      return self.dimension_semantics
    return self.builder.dimension_semantics

  @property
  def effective_is_sparse_core(self) -> bool:
    if self.is_sparse_core is not None:
      return self.is_sparse_core
    return self.builder.is_sparse_core

  @property
  def program_grid_coords(self) -> tuple[int, ...]:
    if not self.vmapped_dims:
      return self.grid_coords
    return tuple(
        c
        for ax, c in enumerate(self.grid_coords)
        if ax not in self.vmapped_dims
    )

  @property
  def program_grid_sizes(self) -> tuple[int | None, ...]:
    if not self.vmapped_dims:
      return self.grid_sizes
    return tuple(
        s for ax, s in enumerate(self.grid_sizes) if ax not in self.vmapped_dims
    )

  @property
  def pipeline_stream_actor(self) -> _ActorKey:
    """Stream key for pipelined VMEM tile residency and revisit order."""
    if self.builder.num_tpu_cores == 1 and not self.core_axis_names:
      return _ActorKey(self.device_id, self.actor.core_id, ())
    return self.actor

  @property
  def axis_env(self) -> dict[Any, int]:
    env = self.mesh_ctx.coord_dict(self.device_coords)
    env.update(self.extra_axis_env)
    return env

  def scoped(self, suffix: str) -> _WalkContext:
    return dataclasses.replace(
        self,
        scope_path=f"{self.scope_path}.{suffix}",
        guarded_written={},
        guarded_output_writes={},
    )

  def grid_tag(self) -> str:
    return "g" + "_".join(str(c) for c in self.grid_coords)

  def arb_grid_tag(self) -> str:
    arb = [
        str(c)
        for c, sem in zip(self.grid_coords, self.effective_dimension_semantics)
        if sem != "parallel"
    ]
    return "g" + "_".join(arb)


def _canonical_accum_buffer_id_str(buf_id: str) -> str:
  if buf_id.startswith("tile_"):
    parts = buf_id.split("_", 2)
    if len(parts) >= 2:
      return f"tile_{parts[1]}"
  return buf_id


def _canonical_accum_buffer_id(view: _RefView) -> str:
  return _canonical_accum_buffer_id_str(view.buffer_id)


def _resolve_view(
    ctx: _WalkContext,
    env: _Env,
    ref_or_view: Any,
    extra_transforms: Sequence[Any],
    loc: str | None,
    *,
    access_kind: _SliceAccessKind = _SliceAccessKind.WRITE,
) -> _RefView | None:
  """Resolves a ref operand to a view and records out-of-bounds indexing."""
  root, pre = _unwrap_ref_and_transforms(ref_or_view)
  if not _is_jax_var(root) or root not in env.refs:
    return None
  transforms = (*pre, *extra_transforms)
  view, violation = _apply_transforms(
      env.refs[root],
      transforms,
      env.scalars,
      access_kind=access_kind,
  )
  if violation is None and env.concolic_scalars:
    concolic_view, c_violation = _apply_transforms(
        env.refs[root],
        transforms,
        {**env.scalars, **env.concolic_scalars},
        access_kind=access_kind,
    )
    if c_violation is not None:
      violation = f"{c_violation} (for the provided example inputs)"
    elif (
        view.sem_type is not None
        and view.sem_index is None
        and concolic_view.sem_index is not None
    ):
      view = concolic_view
      ctx.builder.note_incomplete(
          "a semaphore array index was evaluated at the provided example"
          " inputs",
          affects_coverage=False,
      )
  if violation is not None:
    suffix = (
        " (inside a runtime-dependent branch or loop)"
        if ctx.speculative
        else ""
    )
    ctx.builder.record_issue(
        _Category.OUT_OF_BOUNDS_ACCESS,
        f"access to {view.memory_space}:{view.buffer_id}: {violation}{suffix}",
        loc,
        severity=_Severity.WARNING if ctx.speculative else _Severity.ERROR,
        dedup_key=(
            "ref_oob",
            view.memory_space,
            _canonical_accum_buffer_id(view),
            violation,
            loc,
            ctx.speculative,
        ),
    )
  return view


def _resolve_sem_type(
    ctx: _WalkContext, view: _RefView, default_type: str
) -> str | None:
  if view.sem_index is None or view.opaque:
    ctx.builder.note_incomplete(
        f"the element of semaphore array {view.buffer_id} could not be"
        " determined statically"
    )
    return None
  return view.sem_type or default_type


def _record_buffer_read_before_step_write(
    ctx: _WalkContext, view: _RefView
) -> None:
  """Records whether `view` reads any element not yet written in this step."""
  if view.buffer_id in ctx.read_before_full_write:
    return
  prior_step_writes = [
      w
      for w in ctx.written.get(view.buffer_id, ())
      if not w.opaque and all(d.exact for d in w.dims)
  ]
  if not prior_step_writes:
    ctx.read_before_full_write.add(view.buffer_id)
    return
  if any(w.covers_full_buffer for w in prior_step_writes):
    return
  if view.region.opaque or not all(d.exact for d in view.region.dims):
    ctx.read_before_full_write.add(view.buffer_id)
    return
  shifted_writes: list[HyperRectSlice] = []
  for w in prior_step_writes:
    if len(w.dims) != len(view.region.dims):
      continue
    shifted_writes.append(
        HyperRectSlice(
            tuple(
                SliceDim(wd.start - vd.start, wd.size)
                for wd, vd in zip(w.dims, view.region.dims)
            )
        )
    )
  sub_shape = tuple(d.size for d in view.region.dims)
  if (
      _uncovered_region(sub_shape, shifted_writes, builder=ctx.builder)
      is not None
  ):
    ctx.read_before_full_write.add(view.buffer_id)


def _check_uninitialized_output_read(
    ctx: _WalkContext, view: _RefView, loc: str | None
) -> None:
  """Flags reads from an unaliased output buffer/tile before it is written."""
  if ctx.speculative or not ctx.builder.coverage_known:
    return
  out_info = ctx.unaliased_output_buffers.get(view.buffer_id)
  if out_info is None:
    return
  op_desc, op_idx = out_info
  prior_writes = list(
      ctx.builder.active_output_tile_writes.get(
          (ctx.pipeline_stream_actor, op_idx), ()
      )
  ) + list(ctx.written.get(view.buffer_id, ()))
  if not prior_writes:
    ctx.builder.record_issue(
        _Category.UNWRITTEN_OUTPUT,
        f"{op_desc} is read before being written at grid point"
        f" {ctx.grid_coords} (initialize the output tile on the first step or"
        " alias it via input_output_aliases)",
        loc,
        dedup_key=("uninit_out_read", op_idx, loc),
    )
    return
  if view.region.opaque or not all(d.exact for d in view.region.dims):
    return
  shifted_writes: list[HyperRectSlice] = []
  for w in prior_writes:
    if w.opaque or len(w.dims) != len(view.region.dims):
      return
    if not all(d.exact for d in w.dims):
      return
    shifted_writes.append(
        HyperRectSlice(
            tuple(
                SliceDim(wd.start - vd.start, wd.size)
                for wd, vd in zip(w.dims, view.region.dims)
            )
        )
    )
  sub_shape = tuple(d.size for d in view.region.dims)
  uncovered = _uncovered_region(sub_shape, shifted_writes, builder=ctx.builder)
  if isinstance(uncovered, HyperRectSlice):
    ctx.builder.record_issue(
        _Category.UNWRITTEN_OUTPUT,
        f"{op_desc} region {view.region} is read before being completely"
        f" written at grid point {ctx.grid_coords}",
        loc,
        dedup_key=("uninit_out_read", op_idx, loc),
    )


def _check_low_precision_ref_accumulation(
    ctx: _WalkContext,
    view: _RefView,
    *,
    is_additive_update: bool,
    loc: str | None,
) -> None:
  """Warns when cross-step accumulation writes into a <32-bit float Ref."""
  try:
    dt = jnp.dtype(view.dtype)
  except TypeError:
    return
  if not (jnp.issubdtype(dt, jnp.floating) and dt.itemsize < 4):
    return
  canon_id = _canonical_accum_buffer_id(view)
  if is_additive_update or (
      canon_id in ctx.read_in_step
      and (ctx.actor, canon_id) in ctx.builder.prior_step_written_buffers
  ):
    ctx.builder.record_issue(
        _Category.LOW_PRECISION_ACCUMULATION,
        f"accumulating across steps into {dt.name} buffer {canon_id} loses"
        f" precision; accumulate in float32 and cast to {dt.name} on the final"
        " write",
        loc,
        severity=_Severity.WARNING,
        dedup_key=("low_prec_ref_accum", canon_id, dt.name),
    )


def _record_access(
    ctx: _WalkContext,
    view: _RefView,
    *,
    is_write: bool,
    loc: str | None,
    is_additive_update: bool = False,
    event_region: HyperRectSlice | None = None,
    record_must_write: bool = True,
) -> None:
  ctx.touched_refs.add(view.buffer_id)
  canon_id = _canonical_accum_buffer_id(view)
  if not is_write:
    _check_uninitialized_output_read(ctx, view, loc)
    _record_buffer_read_before_step_write(ctx, view)
    ctx.read_refs.add(view.buffer_id)
    ctx.read_in_step.add(canon_id)
  if is_write:
    ctx.builder.bump_may_write_epoch(view.buffer_id)
    _check_low_precision_ref_accumulation(
        ctx, view, is_additive_update=is_additive_update, loc=loc
    )
    if record_must_write:
      ctx.written.setdefault(view.buffer_id, []).append(view.region)
    ctx.builder.record_output_write(
        ctx.device_id,
        view.buffer_id,
        view.region if record_must_write else (event_region or view.region),
    )


def _eval_sync_scalar(
    ctx: _WalkContext, env: _Env, expr: Any, description: str
) -> int | bool | None:
  """Evaluates a sync/target scalar statically, falling back to concolic."""
  val = _eval_atom(expr, env.scalars)
  if val is not None:
    return val
  if env.concolic_scalars:
    c_val = _eval_atom(expr, {**env.scalars, **env.concolic_scalars})
    if c_val is not None:
      ctx.builder.note_incomplete(
          f"{description} was evaluated at the provided example inputs",
          affects_coverage=False,
      )
      return c_val
  return None


def _resolve_axis_mapping(
    ctx: _WalkContext,
    env: _Env,
    mapping_expr: Mapping[Any, Any],
    *,
    allow_outer_axes: bool,
    known_core_axes: Sequence[str],
    core_axis_sizes: Mapping[str, int | None],
    core_coords: MutableMapping[Any, int | bool | None],
    dev_coords: MutableMapping[Any, int | bool | None],
    loc: str | None,
    affects_coverage: bool,
) -> tuple[bool, bool, bool]:
  """Resolves an axis-name dictionary into `core_coords` and `dev_coords`.

  Args:
    ctx: Active walk context.
    env: Active lexical scalar/ref environment.
    mapping_expr: Axis-name dictionary expression passed to `device_id` or
      `core_index`.
    allow_outer_axes: Whether outer SPMD mesh axes are valid keys.
    known_core_axes: Valid core-mesh axis names for the target ref/mesh.
    core_axis_sizes: Dimension sizes by core-mesh axis name.
    core_coords: Output mapping populated with resolved core-axis coordinates.
    dev_coords: Output mapping populated with resolved outer-mesh coordinates.
    loc: Source location for diagnostic reporting.
    affects_coverage: Whether unresolved coordinates affect output coverage.

  Returns:
    `(ok, has_outer_axis, has_core_axis)`.
  """
  mesh = ctx.mesh_ctx
  core_axes = tuple(known_core_axes)
  known_axes = (*mesh.axis_names, *core_axes) if allow_outer_axes else core_axes
  has_outer = False
  has_core = False
  for key, value_expr in mapping_expr.items():
    axes = key if isinstance(key, tuple) else (key,)
    if not axes:
      continue
    for ax in axes:
      if ax in core_axes:
        has_core = True
      elif allow_outer_axes and ax in mesh.axis_names:
        has_outer = True
      else:
        ctx.builder.record_issue(
            _Category.INVALID_DEVICE_ID,
            f"remote operation targets unknown axis {ax!r} (known axes:"
            f" {known_axes})",
            loc,
        )
        ctx.builder.note_incomplete(
            f"a remote operation targeted unknown axis {ax!r}",
            affects_coverage=affects_coverage,
        )
        return False, has_outer, has_core
    val = _eval_sync_scalar(ctx, env, value_expr, "a remote target coordinate")
    if len(axes) == 1:
      ax = axes[0]
      if ax in core_axes:
        core_coords[ax] = val
      else:
        dev_coords[ax] = val
      continue
    sub_sizes = [
        core_axis_sizes.get(ax)
        if ax in core_axes
        else mesh.mesh_shape[mesh.axis_names.index(ax)]
        for ax in axes
    ]
    sub_size_ints = [int(s) for s in sub_sizes if s is not None]
    if val is None or len(sub_size_ints) != len(sub_sizes):
      for ax in axes:
        if ax in core_axes:
          core_coords[ax] = None
        else:
          dev_coords[ax] = None
      continue
    total_sub = int(math.prod(sub_size_ints)) if sub_size_ints else 1
    flat_v = int(val)
    if not 0 <= flat_v < total_sub:
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          f"remote operation targets index {flat_v} along joint axes"
          f" {axes!r}, which only has {total_sub} core(s)/device(s) (wrap the"
          f" index with lax.rem(idx + {total_sub}, {total_sub}))",
          loc,
      )
      ctx.builder.note_incomplete(
          "a remote operation targets a device or core outside the mesh",
          affects_coverage=affects_coverage,
      )
      return False, has_outer, has_core
    sub_coords = (
        np.unravel_index(flat_v, tuple(sub_size_ints)) if sub_size_ints else ()
    )
    for ax, c_val in zip(axes, sub_coords):
      if ax in core_axes:
        core_coords[ax] = int(c_val)
      else:
        dev_coords[ax] = int(c_val)
  return True, has_outer, has_core


def _resolve_target_location(
    ctx: _WalkContext,
    env: _Env,
    device_id_expr: Any,
    loc: str | None,
    *,
    core_index_expr: Any = None,
    device_id_type: Any = None,
    affects_coverage: bool = False,
    target_view: _RefView | None = None,
) -> tuple[int, tuple[int, ...], bool] | None:
  """Resolves `device_id` and `core_index` to a target device and core.

  Args:
    ctx: Active walk context.
    env: Active lexical scalar/ref environment.
    device_id_expr: Pallas `device_id` expression (scalar, tuple, or dict).
    loc: Source location for diagnostic reporting.
    core_index_expr: Optional explicit `core_index` expression.
    device_id_type: Optional `pltpu.DeviceIdType` (`MESH` or `LOGICAL`).
    affects_coverage: Whether unresolved targets affect output coverage.
    target_view: Optional target `_RefView` carrying destination core-mesh
      metadata in multi-mesh `mpmd_map`.

  Returns:
    `(target_device_id, target_parallel_coords, is_remote)`, or None if the
    target is invalid or unknown (both are recorded on the builder).
  """
  has_target_mesh = bool(
      target_view is not None and target_view.target_core_axes
  )
  if (
      device_id_expr is None
      and core_index_expr is None
      and (
          not has_target_mesh
          or target_view is None
          or target_view.target_core_axes == ctx.core_axis_names
      )
  ):
    return ctx.device_id, ctx.actor.parallel_coords, False

  mesh = ctx.mesh_ctx
  if has_target_mesh and target_view is not None:
    dest_core_axes = target_view.target_core_axes
    dest_core_sizes = target_view.target_core_sizes
    dest_core_sems = target_view.target_core_semantics
  else:
    dest_core_axes = ctx.core_axis_names
    dest_core_sizes = ctx.grid_sizes
    dest_core_sems = ctx.effective_dimension_semantics

  if ctx.all_core_axis_sizes and not has_target_mesh:
    known_core_axes: tuple[str, ...] = tuple(ctx.all_core_axis_sizes.keys())
  else:
    known_core_axes = dest_core_axes

  core_axis_sizes: dict[str, int | None] = dict(ctx.all_core_axis_sizes)
  for ax, sz in zip(ctx.core_axis_names, ctx.grid_sizes):
    core_axis_sizes.setdefault(ax, sz)
  for ax, sz in zip(dest_core_axes, dest_core_sizes):
    core_axis_sizes[ax] = sz

  id_type_str = (
      getattr(device_id_type, "value", str(device_id_type)).lower()
      if device_id_type is not None
      else "mesh"
  )
  if "logical" in id_type_str:
    id_type_str = "logical"
  else:
    id_type_str = "mesh"

  dev_coords: dict[Any, int | bool | None] = dict(
      zip(mesh.axis_names, ctx.device_coords)
  )
  caller_axis_env = ctx.axis_env
  core_coords: dict[Any, int | bool | None] = {
      ax: caller_axis_env.get(ax) for ax in known_core_axes
  }
  for ax, c_val in zip(ctx.core_axis_names, ctx.grid_coords):
    core_coords.setdefault(ax, c_val)
  if has_target_mesh:
    for ax, sz in zip(dest_core_axes, dest_core_sizes):
      if core_coords.get(ax) is None and sz == 1:
        core_coords[ax] = 0
  has_outer_axis = False
  has_core_axis = has_target_mesh and (dest_core_axes != ctx.core_axis_names)

  if isinstance(device_id_expr, Mapping):
    if id_type_str == "logical":
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          "device_id is an axis-name dictionary, which is incompatible with"
          " device_id_type=pltpu.DeviceIdType.LOGICAL (use"
          " pltpu.DeviceIdType.MESH)",
          loc,
      )
      ctx.builder.note_incomplete(
          "a remote operation used a dictionary device_id with LOGICAL mode",
          affects_coverage=affects_coverage,
      )
      return None
    ok, has_outer_axis, mapped_core = _resolve_axis_mapping(
        ctx,
        env,
        device_id_expr,
        allow_outer_axes=True,
        known_core_axes=known_core_axes,
        core_axis_sizes=core_axis_sizes,
        core_coords=core_coords,
        dev_coords=dev_coords,
        loc=loc,
        affects_coverage=affects_coverage,
    )
    if not ok:
      return None
    has_core_axis = has_core_axis or mapped_core

  if core_index_expr is not None:
    if not dest_core_axes:
      ctx.builder.note_incomplete(
          "a core_index was specified without an enclosing core mesh",
          affects_coverage=affects_coverage,
      )
      return None
    has_core_axis = True
    if isinstance(core_index_expr, Mapping):
      ok, _, _ = _resolve_axis_mapping(
          ctx,
          env,
          core_index_expr,
          allow_outer_axes=False,
          known_core_axes=known_core_axes,
          core_axis_sizes=core_axis_sizes,
          core_coords=core_coords,
          dev_coords=dev_coords,
          loc=loc,
          affects_coverage=affects_coverage,
      )
      if not ok:
        return None
    elif isinstance(core_index_expr, tuple):
      if len(core_index_expr) != len(dest_core_axes):
        ctx.builder.record_issue(
            _Category.INVALID_DEVICE_ID,
            f"core_index tuple has {len(core_index_expr)} coordinate(s), but"
            f" core mesh {dest_core_axes} has {len(dest_core_axes)} axis/axes",
            loc,
        )
        ctx.builder.note_incomplete(
            "a remote core_index tuple had the wrong rank",
            affects_coverage=affects_coverage,
        )
        return None
      for axis, value_expr in zip(dest_core_axes, core_index_expr):
        core_coords[axis] = _eval_sync_scalar(
            ctx, env, value_expr, "a remote core index"
        )
    else:
      if len(dest_core_axes) > 1:
        ctx.builder.record_issue(
            _Category.INVALID_DEVICE_ID,
            f"scalar core_index is ambiguous on {len(dest_core_axes)}D core"
            f" mesh {dest_core_axes}; pass a dict of axis names or a"
            f" {len(dest_core_axes)}-tuple",
            loc,
        )
        ctx.builder.note_incomplete(
            "a scalar core_index was used on a multi-axis core mesh",
            affects_coverage=affects_coverage,
        )
        return None
      core_coords[dest_core_axes[0]] = _eval_sync_scalar(
          ctx, env, core_index_expr, "a remote core index"
      )

  target_parallel_coords = ctx.actor.parallel_coords
  if has_core_axis:
    for name in known_core_axes:
      c_val = core_coords.get(name)
      c_sz = core_axis_sizes.get(name)
      if c_val is not None and c_sz is not None and not 0 <= int(c_val) < c_sz:
        ctx.builder.record_issue(
            _Category.INVALID_DEVICE_ID,
            f"remote operation targets index {int(c_val)} along {name!r}, which"
            f" only has {c_sz} core(s)/device(s) (wrap the index with"
            f" lax.rem(idx + {c_sz}, {c_sz}))",
            loc,
        )
        ctx.builder.note_incomplete(
            "a remote operation targets a core outside the mesh",
            affects_coverage=affects_coverage,
        )
        return None
    if not has_target_mesh and any(
        ax not in dest_core_axes and core_coords.get(ax) is not None
        for ax in known_core_axes
    ):
      ctx.builder.note_incomplete(
          "a remote operation targets another core mesh without a"
          " mesh-qualified ref",
          affects_coverage=affects_coverage,
      )
      return None
    c_values = [core_coords.get(a) for a in dest_core_axes]
    c_ints = [int(v) for v in c_values if v is not None]
    if len(c_ints) != len(c_values) or len(c_ints) != len(dest_core_sizes):
      ctx.builder.note_incomplete(
          "a remote core index could not be determined statically",
          affects_coverage=affects_coverage,
      )
      return None
    for name, value, size in zip(dest_core_axes, c_ints, dest_core_sizes):
      if size is None:
        ctx.builder.note_incomplete(
            "a remote core mesh dimension could not be determined statically",
            affects_coverage=affects_coverage,
        )
        return None
      if not 0 <= value < size:
        ctx.builder.record_issue(
            _Category.INVALID_DEVICE_ID,
            f"remote operation targets index {value} along {name!r}, which"
            f" only has {size} core(s)/device(s) (wrap the index with"
            f" lax.rem(idx + {size}, {size}))",
            loc,
        )
        ctx.builder.note_incomplete(
            "a remote operation targets a core outside the mesh",
            affects_coverage=affects_coverage,
        )
        return None
    target_parallel_coords = tuple(
        c_ints[ax] for ax, sem in enumerate(dest_core_sems) if sem == "parallel"
    )

  if device_id_expr is None or (
      isinstance(device_id_expr, Mapping) and not has_outer_axis
  ):
    is_remote = target_parallel_coords != ctx.actor.parallel_coords or (
        target_view is not None
        and target_view.owner_core_id is not None
        and target_view.owner_core_id != ctx.actor.core_id
    )
    return ctx.device_id, target_parallel_coords, is_remote

  if isinstance(device_id_expr, Mapping):
    names: Sequence[str] = mesh.axis_names
    sizes: Sequence[int] = mesh.mesh_shape
    values = [dev_coords.get(a) for a in mesh.axis_names]
  elif isinstance(device_id_expr, tuple):
    if id_type_str == "logical":
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          "device_id is a coordinate tuple, which is incompatible with"
          " device_id_type=pltpu.DeviceIdType.LOGICAL",
          loc,
      )
      ctx.builder.note_incomplete(
          "a tuple device_id was used with LOGICAL mode",
          affects_coverage=affects_coverage,
      )
      return None
    if len(device_id_expr) != len(mesh.mesh_shape):
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          f"remote operation device_id tuple has {len(device_id_expr)}"
          f" coordinate(s), but mesh {mesh.axis_names} has rank"
          f" {len(mesh.mesh_shape)}",
          loc,
      )
      ctx.builder.note_incomplete(
          "a remote device_id tuple had the wrong rank",
          affects_coverage=affects_coverage,
      )
      return None
    names, sizes = mesh.axis_names, mesh.mesh_shape
    values = [
        _eval_sync_scalar(ctx, env, e, "a remote device id")
        for e in device_id_expr
    ]
  else:
    if id_type_str == "mesh" and len(mesh.mesh_shape) > 1:
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          "scalar device_id with DeviceIdType.MESH is invalid on"
          f" {len(mesh.mesh_shape)}D mesh {mesh.axis_names}; pass a dict of"
          f" axis names, a {len(mesh.mesh_shape)}-tuple of coordinates, or"
          " device_id_type=pltpu.DeviceIdType.LOGICAL",
          loc,
      )
      ctx.builder.note_incomplete(
          "a scalar device_id was used in MESH mode on a multi-axis mesh",
          affects_coverage=affects_coverage,
      )
      return None
    names, sizes = ("logical device id",), (mesh.num_devices,)
    values = [_eval_sync_scalar(ctx, env, device_id_expr, "a remote device id")]
  ints = [int(v) for v in values if v is not None]
  if len(ints) != len(values) or len(ints) != len(sizes):
    if mesh.num_devices > 1:
      ctx.builder.uses_remote_operations = True
    ctx.builder.note_incomplete(
        "a remote device id could not be determined statically",
        affects_coverage=affects_coverage,
    )
    return None
  for name, value, size in zip(names, ints, sizes):
    if not 0 <= value < size:
      if mesh.num_devices > 1:
        ctx.builder.uses_remote_operations = True
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          f"remote operation targets index {value} along {name!r}, which"
          f" only has {size} device(s) (wrap the index with"
          f" lax.rem(idx + {size}, {size}))",
          loc,
      )
      ctx.builder.note_incomplete(
          "a remote operation targets a device outside the mesh",
          affects_coverage=affects_coverage,
      )
      return None
  if not sizes:
    target_dev = ctx.device_id
  elif len(ints) == 1 and len(mesh.mesh_shape) != 1:
    target_dev = ints[0]  # A logical id on a multi-axis mesh.
  else:
    target_dev = int(np.ravel_multi_index(tuple(ints), tuple(sizes)))
  if target_dev != ctx.device_id:
    ctx.builder.uses_remote_operations = True
  is_remote = (
      target_dev != ctx.device_id
      or target_parallel_coords != ctx.actor.parallel_coords
      or (
          target_view is not None
          and target_view.owner_core_id is not None
          and target_view.owner_core_id != ctx.actor.core_id
      )
  )
  return target_dev, target_parallel_coords, is_remote


_UNRESOLVED_ACCESS = (
    "a ref access could not be resolved statically, so it is not checked"
)
_UNRESOLVED_SYNC = "a DMA or semaphore operand could not be resolved statically"


def _handle_get(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  view = None
  transforms: Any = ()
  if "tree" in eqn.params and eqn.invars:
    try:
      transforms = jax.tree_util.tree_unflatten(
          eqn.params["tree"], eqn.invars[1:]
      )
    except (TypeError, ValueError):
      transforms = None
    if transforms is not None:
      view = _resolve_view(
          ctx,
          env,
          eqn.invars[0],
          transforms,
          loc,
          access_kind=_SliceAccessKind.READ,
      )
  if view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_ACCESS, affects_coverage=False)
    return
  _record_access(ctx, view, is_write=False, loc=loc)
  if len(eqn.outvars) != 1 or _is_drop_var(eqn.outvars[0]):
    return
  outvar = eqn.outvars[0]
  if view.sem_index is not None:
    env.sym_exprs[outvar] = (
        "load",
        view.buffer_id,
        view.sem_index,
        ctx.builder.may_write_epoch(view.buffer_id),
    )
  if (
      view.buffer_id.startswith("prefetch_")
      and ctx.builder.may_write_epoch(view.buffer_id) == 0
      and not ctx.written.get(view.buffer_id)
      and ctx.prefetch_arrays
  ):
    pref_idx = int(view.buffer_id.split("_", 1)[1])
    pref_entry = ctx.prefetch_arrays.get(pref_idx)
    if pref_entry is None:
      return
    elem_idx = view.sem_index
    used_concolic_idx = False
    if elem_idx is None and env.concolic_scalars and "tree" in eqn.params:
      root, pre = _unwrap_ref_and_transforms(eqn.invars[0])
      if _is_jax_var(root) and root in env.refs:
        c_view, c_viol = _apply_transforms(
            env.refs[root],
            (*pre, *transforms),
            {**env.scalars, **env.concolic_scalars},
            access_kind=_SliceAccessKind.READ,
        )
        if c_viol is None and c_view.sem_index is not None:
          elem_idx = c_view.sem_index
          used_concolic_idx = True
    arr = pref_entry.values
    if (
        elem_idx is not None
        and len(elem_idx) == arr.ndim
        and all(0 <= idx < dim for idx, dim in zip(elem_idx, arr.shape))
    ):
      elem = arr[elem_idx]
      scalar_val: int | bool | None = None
      if np.issubdtype(arr.dtype, np.bool_):
        scalar_val = bool(elem.item())
      elif np.issubdtype(arr.dtype, np.integer):
        scalar_val = int(elem.item())
      if scalar_val is not None:
        if pref_entry.is_runtime or used_concolic_idx:
          env.concolic_scalars[outvar] = scalar_val
        else:
          env.scalars[outvar] = scalar_val
          env.sym_exprs.pop(outvar, None)


def _handle_swap(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  has_mask = bool(eqn.params.get("has_mask", False))
  sc_add = bool(eqn.params.get("add", False))
  view = None
  if "tree" in eqn.params and len(eqn.invars) >= (3 if has_mask else 2):
    leaf_slice = eqn.invars[2:-1] if has_mask else eqn.invars[2:]
    try:
      transforms = jax.tree_util.tree_unflatten(eqn.params["tree"], leaf_slice)
    except (TypeError, ValueError):
      transforms = None
    if transforms is not None:
      view = _resolve_view(
          ctx,
          env,
          eqn.invars[0],
          transforms,
          loc,
          access_kind=_SliceAccessKind.WRITE,
      )
  if view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_ACCESS, affects_coverage=True)
    return
  is_add = eqn.primitive.name == "addupdate" or sc_add
  event_region = (
      HyperRectSlice(
          tuple(SliceDim(d.start, d.size, exact=False) for d in view.dims),
          opaque=view.opaque or not view.dims,
          covers_full_buffer=False,
      )
      if (has_mask or sc_add)
      else None
  )
  if is_add:
    _record_access(
        ctx, view, is_write=False, loc=loc, event_region=event_region
    )
  _record_access(
      ctx,
      view,
      is_write=True,
      loc=loc,
      is_additive_update=is_add,
      event_region=event_region,
      record_must_write=not has_mask,
  )


def _handle_masked(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  is_write = eqn.primitive.name == "masked_swap"
  view = None
  if "args_tree" in eqn.params:
    try:
      ref_arg, transforms, *_ = eqn.params["args_tree"].unflatten(eqn.invars)
    except (TypeError, ValueError):
      ref_arg = transforms = None
    if ref_arg is not None and transforms is not None:
      view = _resolve_view(
          ctx,
          env,
          ref_arg,
          transforms,
          loc,
          access_kind=(
              _SliceAccessKind.WRITE if is_write else _SliceAccessKind.READ
          ),
      )
  if view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_ACCESS, affects_coverage=is_write)
    return
  # Masked stores are treated as writing their whole region, which can only
  # hide unwritten-output findings, never create false ones.
  _record_access(ctx, view, is_write=is_write, loc=loc)


def _check_dma_sem_view(
    ctx: _WalkContext,
    view: _RefView,
    loc: str | None,
    *,
    allow_done_count_regular: bool = False,
    is_smem_dma: bool = False,
) -> bool:
  if view.sem_type is None:
    ctx.builder.note_incomplete(
        "an asynchronous copy used a data buffer ref as a semaphore"
    )
    return False
  if not view.opaque and view.view_shape != ():
    ctx.builder.note_incomplete(
        "an asynchronous copy used a non-0D semaphore ref"
    )
    return False
  if view.sem_type == "dma":
    return True
  if view.sem_type == "barrier":
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        "asynchronous copy cannot use a barrier semaphore as a DMA completion"
        " semaphore; use pltpu.SemaphoreType.DMA",
        loc,
        dedup_key=("dma_barrier_sem", loc),
    )
    ctx.builder.note_incomplete("an asynchronous copy used a barrier semaphore")
    return False
  if view.sem_type == "regular" and allow_done_count_regular:
    ctx.builder.uses_done_count_regular_dma = True
    if is_smem_dma:
      # `tpu_ops.cc` checks `HasMemorySpace(..., kSmem, CoreType::kTc)`, whose
      # `core_type` is unset until `libtpu` runs `canonicalize-memory-space`.
      ctx.builder.record_issue(
          _Category.HARDWARE_CONSTRAINT_VIOLATION,
          f"asynchronous SMEM copy uses regular semaphore {view.buffer_id};"
          " SMEM DMA copies require pltpu.SemaphoreType.DMA",
          loc,
          dedup_key=("dma_smem_reg_sem", loc),
      )
      ctx.builder.note_incomplete(
          "an asynchronous SMEM copy used a regular semaphore"
      )
      return False
    if findings.supports_dma_done_count_mode(ctx.builder.tpu_version):
      return True
    # `jax.export` serializes `tpu.enqueue_dma` with a regular semaphore before
    # hardware-generation-specific `lower-to-llo` runs, so reject TPU v5e here.
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        f"asynchronous copy uses regular semaphore {view.buffer_id};"
        f" {ctx.builder.tpu_version.value} DMA copies require"
        " pltpu.SemaphoreType.DMA",
        loc,
        dedup_key=("dma_v5e_reg_sem", loc),
    )
  ctx.builder.note_incomplete("an asynchronous copy used a non-DMA semaphore")
  return False


def _check_non_dma_sem_view(
    ctx: _WalkContext, view: _RefView, op_desc: str
) -> bool:
  if view.sem_type is None:
    ctx.builder.note_incomplete(
        f"explicit semaphore {op_desc} targeted a data buffer ref"
    )
    return False
  if not view.opaque and view.view_shape != ():
    ctx.builder.note_incomplete(
        f"explicit semaphore {op_desc} targeted a non-0D semaphore ref"
    )
    return False
  if view.sem_type == "dma":
    ctx.builder.note_incomplete(
        f"explicit semaphore {op_desc} targeted a DMA semaphore"
    )
    return False
  return True


def _handle_dma_start(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  if "tree" not in eqn.params:
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC, affects_coverage=True)
    return
  try:
    ops = decode_dma_operands(eqn.params["tree"].unflatten(eqn.invars))
  except (TypeError, ValueError) as exc:
    ctx.builder.note_incomplete(
        f"could not decode DMA start operands ({exc})",
        affects_coverage=True,
    )
    return
  src = _resolve_view(
      ctx,
      env,
      ops.src.ref,
      ops.src.transforms,
      loc,
      access_kind=_SliceAccessKind.DMA,
  )
  dst = _resolve_view(
      ctx,
      env,
      ops.dst.ref,
      ops.dst.transforms,
      loc,
      access_kind=_SliceAccessKind.DMA,
  )
  if dst is not None:
    ctx.builder.bump_may_write_epoch(dst.buffer_id)
  if src is None or dst is None:
    affects_cov = dst is None or (
        dst.buffer_id.startswith("arg_out_")
        or dst.buffer_id.startswith("tile_")
    )
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC, affects_coverage=affects_cov)
    return
  resolved = _resolve_target_location(
      ctx,
      env,
      ops.device_id,
      loc,
      device_id_type=eqn.params.get("device_id_type"),
      affects_coverage=(
          dst.buffer_id.startswith("arg_out_")
          or dst.buffer_id.startswith("tile_")
      ),
      target_view=dst,
  )
  if resolved is None:
    return
  target, target_par, _ = resolved
  ctx.touched_refs.add(src.buffer_id)
  _record_buffer_read_before_step_write(ctx, src)
  ctx.read_refs.add(src.buffer_id)
  is_add = bool(eqn.params.get("add", False))
  if (
      is_add
      and target == ctx.device_id
      and target_par == ctx.actor.parallel_coords
  ):
    _record_buffer_read_before_step_write(ctx, dst)
    ctx.read_refs.add(dst.buffer_id)
  if not ctx.effective_is_sparse_core and ops.dst_sem is None:
    ctx.builder.note_incomplete(
        "a TensorCore asynchronous copy omitted its destination semaphore"
    )
  if (
      is_add
      and target == ctx.device_id
      and target_par == ctx.actor.parallel_coords
  ):
    _check_uninitialized_output_read(ctx, dst, loc)
    _check_low_precision_ref_accumulation(
        ctx, dst, is_additive_update=True, loc=loc
    )
  ctx.builder.record_output_write(target, dst.buffer_id, dst.region)
  if target == ctx.device_id and target_par == ctx.actor.parallel_coords:
    ctx.written.setdefault(dst.buffer_id, []).append(dst.region)
  else:
    ctx.remote_written.setdefault((target, dst.buffer_id), []).append(
        dst.region
    )
  is_smem_dma = not ctx.effective_is_sparse_core and (
      src.memory_space == "smem" or dst.memory_space == "smem"
  )
  dst_sem = _operand_sem(
      ctx,
      env,
      ops.dst_sem,
      loc,
      allow_done_count_regular=True,
      is_smem_dma=is_smem_dma,
  )
  src_sem = _operand_sem(
      ctx,
      env,
      ops.src_sem,
      loc,
      allow_done_count_regular=True,
      is_smem_dma=is_smem_dma,
  )
  if dst_sem is not None and src_sem is not None and dst_sem[1] != src_sem[1]:
    ctx.builder.note_incomplete(
        "a remote asynchronous copy used mismatched send/recv semaphore types"
    )
    return
  if src.byte_count is None or dst.byte_count is None:
    ctx.builder.note_incomplete(
        "a DMA transfer used an unsupported element dtype"
    )


def _operand_sem(
    ctx: _WalkContext,
    env: _Env,
    operand: RefOperand | None,
    loc: str | None,
    *,
    allow_done_count_regular: bool = False,
    is_smem_dma: bool = False,
) -> tuple[str, str] | None:
  if operand is None:
    return None
  view = _resolve_view(
      ctx,
      env,
      operand.ref,
      operand.transforms,
      loc,
      access_kind=_SliceAccessKind.DMA,
  )
  if view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC)
    return None
  if not _check_dma_sem_view(
      ctx,
      view,
      loc,
      allow_done_count_regular=allow_done_count_regular,
      is_smem_dma=is_smem_dma,
  ):
    return None
  sem_type = _resolve_sem_type(ctx, view, "dma")
  if sem_type is None:
    return None
  return view.buffer_id, sem_type


def _handle_dma_wait(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  if "tree" not in eqn.params:
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC)
    return
  try:
    ops = decode_dma_operands(eqn.params["tree"].unflatten(eqn.invars))
  except (TypeError, ValueError) as exc:
    ctx.builder.note_incomplete(f"could not decode DMA wait operands ({exc})")
    return
  # `wait_send` swaps the operands, so the waited semaphore is always in the
  # destination-semaphore slot and is always local.
  src = _resolve_view(
      ctx,
      env,
      ops.src.ref,
      ops.src.transforms,
      loc,
      access_kind=_SliceAccessKind.DMA,
  )
  dst = _resolve_view(
      ctx,
      env,
      ops.dst.ref,
      ops.dst.transforms,
      loc,
      access_kind=_SliceAccessKind.DMA,
  )
  sem_view = (
      None
      if ops.dst_sem is None
      else _resolve_view(
          ctx,
          env,
          ops.dst_sem.ref,
          ops.dst_sem.transforms,
          loc,
          access_kind=_SliceAccessKind.DMA,
      )
  )
  if src is None or dst is None or sem_view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC)
    return
  is_smem_dma = not ctx.effective_is_sparse_core and (
      src.memory_space == "smem" or dst.memory_space == "smem"
  )
  # JAX Pallas requires `pl.semaphore_wait` (not `dma_wait`) when waiting on a
  # done-count REGULAR semaphore used by `dma_start`.
  if not _check_dma_sem_view(
      ctx,
      sem_view,
      loc,
      allow_done_count_regular=False,
      is_smem_dma=is_smem_dma,
  ):
    return
  if src.byte_count is None or dst.byte_count is None:
    ctx.builder.note_incomplete("a DMA wait used an unsupported element dtype")
    return
  _resolve_sem_type(ctx, sem_view, "dma")


def _handle_barrier(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  tag = eqn.params.get("tag")
  sem_id = (
      f"barrier_sem_{tag if tag is not None else ctx.builder.collective_id}"
  )
  env.refs[eqn.outvars[0]] = _full_view(
      sem_id, "semaphore_mem", (), np.int32, sem_type="barrier"
  )


def _semaphore_amount(
    ctx: _WalkContext,
    env: _Env,
    amount_expr: Any,
    what: str,
    loc: str | None = None,
) -> int | None:
  amount = _eval_sync_scalar(
      ctx, env, amount_expr, f"a semaphore {what} amount"
  )
  if amount is None:
    ctx.builder.note_incomplete(
        f"a semaphore {what} amount could not be determined statically"
    )
    return None
  val = int(amount)
  if val < 0:
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        f"semaphore {what} amount {val} must be non-negative on TPU",
        loc,
        dedup_key=("neg_sem_amt", loc, what, val),
    )
    ctx.builder.note_incomplete(
        f"a semaphore {what} used a negative amount ({val})"
    )
    return None
  return val


def _handle_sem_signal(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  if "args_tree" not in eqn.params:
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC)
    return
  try:
    unflattened = eqn.params["args_tree"].unflatten(eqn.invars)
  except (TypeError, ValueError):
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC)
    return
  sem_arg, sem_tr, inc_arg, dev_id_arg, *rest = unflattened
  core_index_arg = rest[0] if rest else None
  view = _resolve_view(
      ctx, env, sem_arg, sem_tr, loc, access_kind=_SliceAccessKind.DMA
  )
  if view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC)
    return
  if not _check_non_dma_sem_view(ctx, view, "signal"):
    return
  amount = _semaphore_amount(ctx, env, inc_arg, "signal", loc)
  resolved = _resolve_target_location(
      ctx,
      env,
      dev_id_arg,
      loc,
      core_index_expr=core_index_arg,
      device_id_type=eqn.params.get("device_id_type"),
      target_view=view,
  )
  if amount is None or resolved is None:
    return
  _resolve_sem_type(ctx, view, "regular")


def _handle_sem_wait(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  if "args_tree" not in eqn.params:
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC)
    return
  try:
    sem_arg, sem_tr, val_arg, *rest = eqn.params["args_tree"].unflatten(
        eqn.invars
    )
  except (TypeError, ValueError):
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC)
    return
  decrement_arg = rest[0] if rest else True
  view = _resolve_view(
      ctx, env, sem_arg, sem_tr, loc, access_kind=_SliceAccessKind.DMA
  )
  if view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_SYNC)
    return
  if not _check_non_dma_sem_view(ctx, view, "wait"):
    return
  decrement = _eval_sync_scalar(
      ctx, env, decrement_arg, "a semaphore_wait decrement flag"
  )
  if decrement is None:
    # Upstream `_semaphore_wait_lowering_rule` checks `if not decrement:` on an
    # MLIR `ir.Value` (always truthy), silently ignoring dynamic `decrement`.
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        "pl.semaphore_wait requires a static decrement=True on TPU; dynamic"
        " decrement flags are ignored by Mosaic lowering",
        loc,
        dedup_key=("sem_wait_dec", loc),
    )
    ctx.builder.note_incomplete(
        "pl.semaphore_wait used a non-static decrement flag"
    )
    return
  if not decrement:
    # Mosaic lowering already rejects literal `decrement=False`; flag it here
    # too so standalone static model checking agrees without running lowering.
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        "pl.semaphore_wait with decrement=False is not supported on TPU",
        loc,
        dedup_key=("sem_wait_dec", loc),
    )
    ctx.builder.note_incomplete(
        "pl.semaphore_wait used unsupported decrement=False"
    )
    return
  _semaphore_amount(ctx, env, val_arg, "wait", loc)
  _resolve_sem_type(ctx, view, "regular")


def _handle_run_scoped(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  sub_jaxpr = _unwrap_closed_jaxpr(eqn.params["jaxpr"])
  ref_transforms = eqn.params.get("ref_transforms", ())
  shared = bool(eqn.params.get("collective_axes", ()))
  nested = env.child()
  for outer_v, const_v in zip(eqn.invars, sub_jaxpr.constvars):
    nested.bind(outer_v, const_v, env)
  scope = ctx.scoped(f"s{_next_scope_index(ctx)}")
  for alloc_idx, alloc_var in enumerate(sub_jaxpr.invars):
    aval = alloc_var.aval
    sem_type = _detect_semaphore_type(aval)
    # Parallel identity is carried on `SemaphoreKey.parallel_coords`, while data
    # buffers include `grid_tag()` so `_buffer_key` can remap cross-core DMAs.
    if shared:
      prefix = scope.scope_path
    elif sem_type is not None:
      prefix = f"{ctx.arb_grid_tag()}{scope.scope_path}"
    else:
      prefix = f"{ctx.grid_tag()}{scope.scope_path}"
    base = _full_view(
        f"{prefix}.b{alloc_idx}",
        _normalize_memory_space(getattr(aval, "memory_space", None)),
        getattr(aval, "shape", ()),
        getattr(aval, "dtype", np.float32),
        sem_type=sem_type,
        shared=shared,
    )
    init_tr = (
        ref_transforms[alloc_idx] if alloc_idx < len(ref_transforms) else ()
    )
    view, violation = _apply_transforms(
        base, init_tr, nested.scalars, access_kind=_SliceAccessKind.WRITE
    )
    if violation is not None:
      ctx.builder.record_issue(
          _Category.OUT_OF_BOUNDS_ACCESS,
          f"access to {view.memory_space}:{view.buffer_id}: {violation}",
          loc,
      )
    nested.refs[alloc_var] = view
  _walk(sub_jaxpr, scope, nested)
  _bind_outputs(sub_jaxpr.outvars, eqn.outvars, nested, env)


def _handle_empty_ref(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  scope_idx = _next_scope_index(ctx)
  for alloc_idx, alloc_var in enumerate(eqn.outvars):
    if _is_drop_var(alloc_var):
      continue
    aval = getattr(alloc_var, "aval", None)
    sem_type = _detect_semaphore_type(aval)
    prefix = (
        f"{ctx.arb_grid_tag()}{ctx.scope_path}.er{scope_idx}"
        if sem_type is not None
        else f"{ctx.grid_tag()}{ctx.scope_path}.er{scope_idx}"
    )
    env.refs[alloc_var] = _full_view(
        f"{prefix}.b{alloc_idx}",
        _normalize_memory_space(getattr(aval, "memory_space", None)),
        getattr(aval, "shape", ()),
        getattr(aval, "dtype", np.float32),
        sem_type=sem_type,
    )


def _next_scope_index(ctx: _WalkContext) -> int:
  """Returns a scope index unique within the current scope path."""
  return ctx.builder.next_scope_index(
      (ctx.device_id, ctx.grid_coords, ctx.scope_path)
  )


def _bind_outputs(
    inner_outs: Sequence[Any],
    outer_outs: Sequence[Any],
    inner: _Env,
    outer: _Env,
    *,
    force_concolic: bool = False,
) -> None:
  for inner_out, outer_out in zip(inner_outs, outer_outs):
    if _is_drop_var(outer_out):
      continue
    if not force_concolic:
      val = _eval_atom(inner_out, inner.scalars)
      if val is not None:
        outer.scalars[outer_out] = val
        continue
    c_env = (
        {**inner.scalars, **inner.concolic_scalars}
        if force_concolic or inner.concolic_scalars
        else None
    )
    if c_env is not None:
      c_val = _eval_atom(inner_out, c_env)
      if c_val is not None:
        outer.concolic_scalars[outer_out] = c_val
    if (
        not force_concolic
        and _is_jax_var(inner_out)
        and inner_out in inner.sym_exprs
    ):
      outer.sym_exprs[outer_out] = inner.sym_exprs[inner_out]


def _record_step_written_buffers(ctx: _WalkContext) -> None:
  for buf_id, regions in ctx.written.items():
    if not regions:
      continue
    ctx.builder.prior_step_written_buffers.add(
        (ctx.actor, _canonical_accum_buffer_id_str(buf_id))
    )


def _scan_layout(params: Mapping[str, Any]) -> tuple[int, int]:
  """Returns `(num_consts, num_carry)` of a scan equation."""
  if "num_consts" in params:  # Older JAX releases.
    return int(params["num_consts"]), int(params["num_carry"])
  consts, carry, _ = params["ft_in"].unpack()
  return len(consts), len(carry)


def _handle_scan(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  body = _unwrap_closed_jaxpr(eqn.params["jaxpr"])
  num_consts, num_carry = _scan_layout(eqn.params)
  length = int(eqn.params["length"])
  limit = ctx.builder.max_unrolled_iterations
  truncated = length > limit
  if truncated:
    ctx.builder.note_incomplete(
        f"only the first {limit} of {length} loop iterations are checked",
        affects_coverage=True,
    )
  consts = eqn.invars[:num_consts]
  init_carry = list(eqn.invars[num_consts : num_consts + num_carry])
  carry = list(init_carry)
  carry_env = env
  for step in range(min(length, limit)):
    nested = env.child()
    for outer_v, inner_v in zip(consts, body.invars[:num_consts]):
      nested.bind(outer_v, inner_v, env)
    for outer_v, inner_v in zip(carry, body.invars[num_consts:]):
      nested.bind(outer_v, inner_v, carry_env)
    step_ctx = dataclasses.replace(ctx.scoped(f"lp{step}"), read_in_step=set())
    _walk(body, step_ctx, nested)
    _record_step_written_buffers(step_ctx)
    carry = list(body.outvars[:num_carry])
    carry_env = nested
  if not truncated:
    _bind_outputs(carry, eqn.outvars[:num_carry], carry_env, env)
  else:
    body_carry_in = body.invars[num_consts : num_consts + num_carry]
    body_carry_out = body.outvars[:num_carry]
    for init_v, body_in_v, body_out_v, out_v in zip(
        init_carry, body_carry_in, body_carry_out, eqn.outvars[:num_carry]
    ):
      if _is_drop_var(out_v):
        continue
      init_val = _eval_atom(init_v, env.scalars)
      is_invariant = body_out_v == body_in_v or (
          not _is_jax_var(body_out_v) and _eval_atom(body_out_v, {}) == init_val
      )
      if init_val is not None and is_invariant:
        env.scalars[out_v] = init_val
      elif (
          body_out_v == body_in_v
          and _is_jax_var(init_v)
          and init_v in env.concolic_scalars
      ):
        env.concolic_scalars[out_v] = env.concolic_scalars[init_v]


def _try_eval_prefetch_get_eqn(
    eqn: Any,
    ctx: _WalkContext,
    refs: Mapping[Any, _RefView],
    scalar_env: MutableMapping[Any, int | bool],
    *,
    allow_runtime: bool,
) -> bool:
  """Evaluates an unwritten `prefetch_*` scalar `get` inside a predicate."""
  if (
      not ctx.prefetch_arrays
      or eqn.primitive.name != "get"
      or "tree" not in eqn.params
      or len(eqn.outvars) != 1
      or _is_drop_var(eqn.outvars[0])
  ):
    return False
  root, pre = _unwrap_ref_and_transforms(eqn.invars[0])
  if not _is_jax_var(root) or root not in refs:
    return False
  base_view = refs[root]
  if (
      not base_view.buffer_id.startswith("prefetch_")
      or ctx.builder.may_write_epoch(base_view.buffer_id) != 0
      or ctx.written.get(base_view.buffer_id)
  ):
    return False
  pref_idx = int(base_view.buffer_id.split("_", 1)[1])
  pref_entry = ctx.prefetch_arrays.get(pref_idx)
  if pref_entry is None or (pref_entry.is_runtime and not allow_runtime):
    return False
  try:
    transforms = jax.tree_util.tree_unflatten(
        eqn.params["tree"], eqn.invars[1:]
    )
  except (TypeError, ValueError):
    return False
  view, violation = _apply_transforms(
      base_view,
      (*pre, *transforms),
      scalar_env,
      access_kind=_SliceAccessKind.READ,
  )
  arr = pref_entry.values
  if (
      violation is not None
      or view.sem_index is None
      or len(view.sem_index) != arr.ndim
      or not all(0 <= idx < dim for idx, dim in zip(view.sem_index, arr.shape))
  ):
    return False
  elem = arr[view.sem_index]
  if np.issubdtype(arr.dtype, np.bool_):
    scalar_env[eqn.outvars[0]] = bool(elem.item())
    return True
  if np.issubdtype(arr.dtype, np.integer):
    scalar_env[eqn.outvars[0]] = int(elem.item())
    return True
  return False


def _eval_predicate(
    jaxpr: Any, ctx: _WalkContext, env: _Env
) -> tuple[bool | None, bool]:
  """Evaluates `jaxpr` predicate; returns `(value, used_concolic)`."""
  scalars = dict(env.scalars)
  for eqn in jaxpr.eqns:
    if _try_eval_prefetch_get_eqn(
        eqn, ctx, env.refs, scalars, allow_runtime=False
    ):
      continue
    _try_eval_scalar_eqn(
        eqn,
        scalars,
        grid_coords=ctx.program_grid_coords,
        grid_sizes=ctx.program_grid_sizes,
        axis_env=ctx.axis_env,
    )
  value = _eval_atom(jaxpr.outvars[0], scalars)
  if value is not None:
    return bool(value), False
  if not env.concolic_scalars and not ctx.prefetch_arrays:
    return None, False
  merged = {**env.scalars, **env.concolic_scalars}
  for eqn in jaxpr.eqns:
    if _try_eval_prefetch_get_eqn(
        eqn, ctx, env.refs, merged, allow_runtime=True
    ):
      continue
    _try_eval_scalar_eqn(
        eqn,
        merged,
        grid_coords=ctx.program_grid_coords,
        grid_sizes=ctx.program_grid_sizes,
        axis_env=ctx.axis_env,
    )
  c_val = _eval_atom(jaxpr.outvars[0], merged)
  if c_val is not None:
    return bool(c_val), True
  return None, False


def _handle_while(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  cond = _unwrap_closed_jaxpr(eqn.params["cond_jaxpr"])
  body = _unwrap_closed_jaxpr(eqn.params["body_jaxpr"])
  cond_nconsts = int(eqn.params["cond_nconsts"])
  body_nconsts = int(eqn.params["body_nconsts"])
  cond_consts = eqn.invars[:cond_nconsts]
  body_consts = eqn.invars[cond_nconsts : cond_nconsts + body_nconsts]
  init_carry = list(eqn.invars[cond_nconsts + body_nconsts :])
  carry = list(init_carry)
  carry_env = env
  limit = ctx.builder.max_unrolled_iterations
  exited = False
  used_concolic = False
  for step in range(limit + 1):
    cond_env = env.child()
    for outer_v, inner_v in zip(cond_consts, cond.invars[:cond_nconsts]):
      cond_env.bind(outer_v, inner_v, env)
    for outer_v, inner_v in zip(carry, cond.invars[cond_nconsts:]):
      cond_env.bind(outer_v, inner_v, carry_env)
    keep_going, pred_concolic = _eval_predicate(cond, ctx, cond_env)
    if pred_concolic:
      used_concolic = True
      ctx.builder.note_incomplete(
          "a while loop's trip count was evaluated at the provided example"
          " inputs",
          affects_coverage=False,
      )
    if keep_going is not None and not keep_going:
      exited = True
      break
    if step == limit:
      ctx.builder.note_incomplete(
          f"only the first {limit} iterations of a while loop are checked",
          affects_coverage=True,
      )
      break
    if keep_going is None:
      ctx.builder.note_incomplete(
          "a while loop's trip count depends on runtime data, so its body is"
          " checked once",
          affects_coverage=True,
      )
    nested = env.child()
    for outer_v, inner_v in zip(body_consts, body.invars[:body_nconsts]):
      nested.bind(outer_v, inner_v, env)
    for outer_v, inner_v in zip(carry, body.invars[body_nconsts:]):
      nested.bind(outer_v, inner_v, carry_env)
    step_ctx = dataclasses.replace(
        ctx.scoped(f"wh{step}"),
        speculative=ctx.speculative or (keep_going is None),
        read_in_step=set(),
    )
    _walk(body, step_ctx, nested)
    _record_step_written_buffers(step_ctx)
    carry = list(body.outvars)
    carry_env = nested
    if keep_going is None:
      break
  if exited:
    _bind_outputs(
        carry, eqn.outvars, carry_env, env, force_concolic=used_concolic
    )
  else:
    body_carry_in = body.invars[body_nconsts:]
    for init_v, body_in_v, body_out_v, out_v in zip(
        init_carry, body_carry_in, body.outvars, eqn.outvars
    ):
      if _is_drop_var(out_v):
        continue
      init_val = _eval_atom(init_v, env.scalars)
      is_invariant = body_out_v == body_in_v or (
          not _is_jax_var(body_out_v) and _eval_atom(body_out_v, {}) == init_val
      )
      if init_val is not None and is_invariant:
        env.scalars[out_v] = init_val
      elif (
          body_out_v == body_in_v
          and _is_jax_var(init_v)
          and init_v in env.concolic_scalars
      ):
        env.concolic_scalars[out_v] = env.concolic_scalars[init_v]


def _intersect_branch_regions(
    branch_regions: Sequence[Sequence[HyperRectSlice]],
) -> list[HyperRectSlice]:
  """Returns exact sub-regions covered along all `branch_regions`."""
  if not branch_regions or any(not br for br in branch_regions):
    return []
  exact_branches = [
      [
          r
          for r in br
          if not r.opaque and all(d.exact and d.size > 0 for d in r.dims)
      ]
      for br in branch_regions
  ]
  if any(not br for br in exact_branches):
    return []
  rank = len(exact_branches[0][0].dims)
  for br in exact_branches:
    for r in br:
      if len(r.dims) != rank:
        return [
            cand
            for cand in branch_regions[0]
            if all(cand in other for other in branch_regions[1:])
        ]
  if rank == 0:
    return [HyperRectSlice(())]
  cuts: list[list[int]] = []
  for axis in range(rank):
    points: set[int] = set()
    for br in exact_branches:
      for r in br:
        points.add(r.dims[axis].start)
        points.add(r.dims[axis].stop)
    cuts.append(sorted(points))
  cell_shape = tuple(max(0, len(c) - 1) for c in cuts)
  if any(s == 0 for s in cell_shape):
    return []
  if math.prod(cell_shape) > _MAX_COVERAGE_CELLS:
    return [
        r
        for r in branch_regions[0]
        if all(r in other for other in branch_regions[1:])
    ]
  common_mask = np.ones(cell_shape, dtype=bool)
  for br in exact_branches:
    br_mask = np.zeros(cell_shape, dtype=bool)
    for r in br:
      slices = [
          slice(
              int(np.searchsorted(cuts[a], r.dims[a].start)),
              int(np.searchsorted(cuts[a], r.dims[a].stop)),
          )
          for a in range(rank)
      ]
      br_mask[tuple(slices)] = True
    common_mask &= br_mask
    if not bool(common_mask.any()):
      return []
  if bool(common_mask.all()):
    return [HyperRectSlice(tuple(SliceDim(c[0], c[-1] - c[0]) for c in cuts))]
  common: list[HyperRectSlice] = []
  for cell in np.argwhere(common_mask):
    common.append(
        HyperRectSlice(
            tuple(
                SliceDim(cuts[a][int(i)], cuts[a][int(i) + 1] - cuts[a][int(i)])
                for a, i in enumerate(cell)
            )
        )
    )
  return common


def _canonical_cond_predicate(
    pred_atom: Any, env: _Env
) -> tuple[Any, bool] | None:
  """Returns `(base_pred, polarity)` for a 2-branch `cond` selector atom."""
  if not _is_jax_var(pred_atom) or pred_atom not in env.sym_exprs:
    return None
  sym = env.sym_exprs[pred_atom]
  polarity = True
  while isinstance(sym, tuple) and len(sym) == 2 and sym[0] == "not":
    polarity = not polarity
    sym = sym[1]
  return sym, polarity


def _handle_cond(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  branches = eqn.params["branches"]
  index = _eval_atom(eqn.invars[0], env.scalars)
  used_concolic = False
  c_index: int | bool | None = None
  if index is not None and 0 <= int(index) < len(branches):
    selected = [int(index)]
  else:
    has_branch_sync = any(
        ctx.builder.has_sync(_unwrap_closed_jaxpr(b)) for b in branches
    )
    c_index = (
        _eval_atom(eqn.invars[0], {**env.scalars, **env.concolic_scalars})
        if env.concolic_scalars
        else None
    )
    if (
        has_branch_sync
        and c_index is not None
        and 0 <= int(c_index) < len(branches)
    ):
      selected = [int(c_index)]
      used_concolic = True
      ctx.builder.note_incomplete(
          "a branch condition over synchronization was evaluated at the"
          " provided example inputs, so other runtime branches were not"
          " model checked",
          affects_coverage=False,
      )
    else:
      selected = list(range(len(branches)))
      reason = (
          "a branch condition depends on runtime data, so all branches are"
          " checked in sequence"
      )
      if has_branch_sync:
        # Semaphore operations of branches that exclude each other would be
        # counted together.
        ctx.builder.note_incomplete(reason)
      else:
        ctx.builder.note_incomplete(reason, affects_coverage=False)
  if len(selected) == 1:
    br_idx = selected[0]
    branch = _unwrap_closed_jaxpr(branches[br_idx])
    nested = env.child()
    for outer_v, inner_v in zip(eqn.invars[1:], branch.invars):
      nested.bind(outer_v, inner_v, env)
    _walk(branch, ctx.scoped(f"br{br_idx}"), nested)
    _bind_outputs(
        branch.outvars,
        eqn.outvars,
        nested,
        env,
        force_concolic=used_concolic,
    )
    return

  base_written = {k: list(v) for k, v in ctx.written.items()}
  base_output_writes = {
      k: list(v) for k, v in ctx.builder.output_writes.items()
  }
  base_remote_written = {k: list(v) for k, v in ctx.remote_written.items()}
  branch_added_written: list[dict[str, list[HyperRectSlice]]] = []
  branch_added_outputs: list[dict[tuple[int, str], list[HyperRectSlice]]] = []
  branch_added_remote: list[dict[tuple[int, str], list[HyperRectSlice]]] = []
  for br_idx in selected:
    ctx.written.clear()
    ctx.written.update({k: list(v) for k, v in base_written.items()})
    ctx.builder.output_writes.clear()
    ctx.builder.output_writes.update(
        {k: list(v) for k, v in base_output_writes.items()}
    )
    ctx.remote_written.clear()
    ctx.remote_written.update(
        {k: list(v) for k, v in base_remote_written.items()}
    )
    branch = _unwrap_closed_jaxpr(branches[br_idx])
    nested = env.child()
    for outer_v, inner_v in zip(eqn.invars[1:], branch.invars):
      nested.bind(outer_v, inner_v, env)
    br_ctx = dataclasses.replace(ctx.scoped(f"br{br_idx}"), speculative=True)
    _walk(branch, br_ctx, nested)
    if c_index is not None and br_idx == int(c_index):
      _bind_outputs(
          branch.outvars,
          eqn.outvars,
          nested,
          env,
          force_concolic=True,
      )
    branch_added_written.append({
        k: v[len(base_written.get(k, ())) :]
        for k, v in ctx.written.items()
        if len(v) > len(base_written.get(k, ()))
    })
    branch_added_outputs.append({
        k: v[len(base_output_writes.get(k, ())) :]
        for k, v in ctx.builder.output_writes.items()
        if len(v) > len(base_output_writes.get(k, ()))
    })
    branch_added_remote.append({
        k: v[len(base_remote_written.get(k, ())) :]
        for k, v in ctx.remote_written.items()
        if len(v) > len(base_remote_written.get(k, ()))
    })

  # Only regions written along every branch are guaranteed to be initialized
  # after the conditional completes.
  ctx.written.clear()
  ctx.written.update({k: list(v) for k, v in base_written.items()})
  if branch_added_written:
    for buf_id in branch_added_written[0]:
      common = _intersect_branch_regions(
          [br.get(buf_id, ()) for br in branch_added_written]
      )
      if common:
        ctx.written.setdefault(buf_id, []).extend(common)

  ctx.builder.output_writes.clear()
  ctx.builder.output_writes.update(
      {k: list(v) for k, v in base_output_writes.items()}
  )
  if branch_added_outputs:
    for key in branch_added_outputs[0]:
      common = _intersect_branch_regions(
          [br.get(key, ()) for br in branch_added_outputs]
      )
      if common:
        ctx.builder.output_writes.setdefault(key, []).extend(common)

  ctx.remote_written.clear()
  ctx.remote_written.update(
      {k: list(v) for k, v in base_remote_written.items()}
  )
  if branch_added_remote:
    for key in branch_added_remote[0]:
      common = _intersect_branch_regions(
          [br.get(key, ()) for br in branch_added_remote]
      )
      if common:
        ctx.remote_written.setdefault(key, []).extend(common)

  # Track complementary 2-branch predicates (such as `@pl.when(p)` followed by
  # `@pl.when(~p)`) so writes covering both truth values of `p` are combined.
  if len(branches) == 2:
    canon = _canonical_cond_predicate(eqn.invars[0], env)
    if canon is not None:
      base_pred, polarity = canon
      # In `lax.cond`, branch 0 is `index == 0` (False) and branch 1 is True.
      br_truths = (not polarity, polarity)
      for br_idx, truth in enumerate(br_truths):
        g_written = ctx.guarded_written.setdefault((base_pred, truth), {})
        for buf_id, regs in branch_added_written[br_idx].items():
          g_written.setdefault(buf_id, []).extend(regs)
        g_out = ctx.guarded_output_writes.setdefault((base_pred, truth), {})
        for key, regs in branch_added_outputs[br_idx].items():
          g_out.setdefault(key, []).extend(regs)
      false_w = ctx.guarded_written.get((base_pred, False), {})
      true_w = ctx.guarded_written.get((base_pred, True), {})
      for buf_id in set(false_w).intersection(true_w):
        common = _intersect_branch_regions([false_w[buf_id], true_w[buf_id]])
        if common:
          ctx.written.setdefault(buf_id, []).extend(common)
      false_out = ctx.guarded_output_writes.get((base_pred, False), {})
      true_out = ctx.guarded_output_writes.get((base_pred, True), {})
      for key in set(false_out).intersection(true_out):
        common = _intersect_branch_regions([false_out[key], true_out[key]])
        if common:
          ctx.builder.output_writes.setdefault(key, []).extend(common)


def _block_aval(bm: Any) -> Any:
  """Returns `bm`'s block aval across JAX versions."""
  aval = getattr(bm, "transformed_block_aval", None)
  return bm.block_aval if aval is None else aval


def _sub_jaxprs(eqn: Any) -> Iterator[Any]:
  for value in eqn.params.values():
    candidates = value if isinstance(value, (tuple, list)) else (value,)
    for candidate in candidates:
      inner = _unwrap_closed_jaxpr(candidate)
      if hasattr(inner, "eqns") and hasattr(inner, "invars"):
        yield inner


def _handle_call(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  for value in eqn.params.values():
    candidates = value if isinstance(value, (tuple, list)) else (value,)
    for candidate in candidates:
      inner = _unwrap_closed_jaxpr(candidate)
      if not (hasattr(inner, "eqns") and hasattr(inner, "invars")):
        continue
      if len(inner.invars) != len(eqn.invars):
        continue
      nested = env.child()
      for const_v, const_val in zip(
          inner.constvars, getattr(candidate, "consts", ())
      ):
        nested.bind(const_val, const_v, env)
      for outer_v, inner_v in zip(eqn.invars, inner.invars):
        nested.bind(outer_v, inner_v, env)
      _walk(inner, ctx, nested)
      _bind_outputs(inner.outvars, eqn.outvars, nested, env)
      return
  _handle_unknown(eqn, ctx, env, loc)


def _handle_unknown(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  prim_name = eqn.primitive.name
  subs = list(_sub_jaxprs(eqn))
  if not subs:
    touches_sem = prim_name == "barrier"
    touches_ref = False
    for invar in eqn.invars:
      root, _ = _unwrap_ref_and_transforms(invar)
      aval = getattr(root, "aval", getattr(invar, "aval", None))
      view = env.refs.get(root) if _is_jax_var(root) else None
      if view is not None:
        if view.sem_type is not None:
          touches_sem = True
        else:
          touches_ref = True
          ctx.touched_refs.add(view.buffer_id)
          ctx.builder.bump_may_write_epoch(view.buffer_id)
      elif _detect_semaphore_type(aval) is not None:
        touches_sem = True
      elif _is_abstract_ref(aval):
        touches_ref = True
    if touches_sem:
      ctx.builder.note_incomplete(
          f"unhandled synchronization primitive '{prim_name}'",
          affects_coverage=touches_ref,
      )
    elif touches_ref:
      ctx.builder.note_incomplete(
          f"unhandled ref primitive '{prim_name}'",
          affects_coverage=True,
      )
    return
  reason = (
      f"`{prim_name}` is not interpreted, so accesses inside it are not checked"
  )
  if any(ctx.builder.has_sync(sub) for sub in subs):
    # Its semaphore operations would be missing from the model.
    ctx.builder.note_incomplete(reason, affects_coverage=True)
  else:
    ctx.builder.note_incomplete(reason, affects_coverage=True)


_HANDLERS: dict[str, Callable[[Any, _WalkContext, _Env, str | None], None]] = {
    "addupdate": _handle_swap,
    "cond": _handle_cond,
    "dma_start": _handle_dma_start,
    "dma_wait": _handle_dma_wait,
    "empty_ref": _handle_empty_ref,
    "get": _handle_get,
    "get_barrier_semaphore": _handle_barrier,
    "masked_load": _handle_masked,
    "masked_swap": _handle_masked,
    "run_scoped": _handle_run_scoped,
    "scan": _handle_scan,
    "semaphore_signal": _handle_sem_signal,
    "semaphore_wait": _handle_sem_wait,
    "swap": _handle_swap,
    "while": _handle_while,
}


def _walk(jaxpr: Any, ctx: _WalkContext, env: _Env) -> None:
  """Symbolically executes `jaxpr` for one device and grid point."""
  for eqn in jaxpr.eqns:
    ctx.builder.step_count += 1
    if ctx.builder.step_count > _MAX_TOTAL_EQUATION_STEPS:
      ctx.builder.note_incomplete(
          f"stopped walking after {_MAX_TOTAL_EQUATION_STEPS} equations",
          affects_coverage=True,
      )
      return
    if _try_eval_scalar_eqn(
        eqn,
        env.scalars,
        grid_coords=ctx.program_grid_coords,
        grid_sizes=ctx.program_grid_sizes,
        axis_env=ctx.axis_env,
        sym_env=env.sym_exprs,
    ):
      continue
    name = eqn.primitive.name
    if env.concolic_scalars and name not in _HANDLERS:
      merged = {**env.scalars, **env.concolic_scalars}
      if _try_eval_scalar_eqn(
          eqn,
          merged,
          grid_coords=ctx.program_grid_coords,
          grid_sizes=ctx.program_grid_sizes,
          axis_env=ctx.axis_env,
      ):
        for out_v in eqn.outvars:
          if (
              not _is_drop_var(out_v)
              and out_v not in env.scalars
              and out_v in merged
          ):
            env.concolic_scalars[out_v] = merged[out_v]
        continue
    handler = _HANDLERS.get(name)
    if handler is None:
      handler = _handle_call if name in _CALL_PRIMITIVES else _handle_unknown
    try:
      handler(eqn, ctx, env, ctx.builder.source_location(eqn))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      reason = (
          f"could not analyze `{name}` equation ({type(exc).__name__}: {exc})"
      )
      if name in _SYNC_PRIMITIVES:
        ctx.builder.note_incomplete(reason, affects_coverage=True)
      else:
        ctx.builder.note_incomplete(reason, affects_coverage=True)


def _is_invalid_scratch_memory_space(
    ms: Any, *, allow_hbm: bool = False
) -> str | None:
  """Returns the normalized space name if `ms` is invalid for TPU scratch."""
  if ms is None:
    return None
  unwrapped = ms
  while hasattr(unwrapped, "memory_space") and not isinstance(
      unwrapped, (str, bytes)
  ):
    nxt = getattr(unwrapped, "memory_space")
    if nxt is unwrapped:
      break
    unwrapped = nxt
  raw_val = getattr(unwrapped, "value", unwrapped)
  if isinstance(raw_val, str) and raw_val.lower() == "any":
    return "ANY"
  norm = _normalize_memory_space(ms, default="")
  if norm == "hbm":
    return None if allow_hbm else "HBM"
  if norm == "cmem":
    return "CMEM"
  return None


def _static_operand_memory_space(op: RefOperand) -> str:
  """Returns the statically annotated memory space of a `RefOperand`."""
  ms = _normalize_memory_space(
      getattr(getattr(op.ref, "aval", None), "memory_space", None),
      default="vmem",
  )
  for tr in op.transforms:
    if type(tr).__name__ == "MemorySpaceCastTransform" and hasattr(
        tr, "memory_space"
    ):
      ms = _normalize_memory_space(tr.memory_space, default=ms)
  return ms


def _is_raw_any_memory_space(aval: Any, transforms: Sequence[Any] = ()) -> bool:
  """Returns whether `aval` (after transforms) explicitly uses `pl.ANY`."""
  ms = getattr(aval, "memory_space", None)
  for tr in transforms:
    if type(tr).__name__ == "MemorySpaceCastTransform" and hasattr(
        tr, "memory_space"
    ):
      ms = tr.memory_space
  raw_val = getattr(ms, "value", ms)
  return isinstance(raw_val, str) and raw_val.lower() == "any"


def _check_static_hardware_rules(
    jaxpr: Any,
    builder: _KernelModelBuilder,
    *,
    is_sparse_core: bool | None = None,
) -> None:
  """Flow-insensitively checks hardware constraints and precision lints."""
  effective_sc = (
      builder.is_sparse_core if is_sparse_core is None else is_sparse_core
  )
  for eqn in jaxpr.eqns:
    loc = _summarize_source_info(eqn)
    name = eqn.primitive.name
    if name == "dot_general":
      out_dtype = getattr(eqn.outvars[0].aval, "dtype", None)
      if (
          out_dtype is not None
          and jnp.issubdtype(out_dtype, jnp.floating)
          and jnp.dtype(out_dtype).itemsize < 4
      ):
        dt_name = jnp.dtype(out_dtype).name
        builder.record_issue(
            _Category.LOW_PRECISION_ACCUMULATION,
            f"dot_general produces a {dt_name} result; accumulating"
            " low-precision matmul results (for example across grid steps)"
            " loses precision, so pass preferred_element_type=jnp.float32",
            loc,
            severity=_Severity.WARNING,
        )
    elif name == "run_scoped":
      # `jax.export` lowers `run_scoped` HBM/CMEM/ANY allocations to
      # `memref.alloca` without error; only `libtpu` LLO lowering rejects them.
      for var in _unwrap_closed_jaxpr(eqn.params["jaxpr"]).invars:
        bad_ms = _is_invalid_scratch_memory_space(
            getattr(var.aval, "memory_space", None)
        )
        if bad_ms is not None:
          builder.record_issue(
              _Category.HARDWARE_CONSTRAINT_VIOLATION,
              f"pl.run_scoped cannot allocate {bad_ms} buffers on TPU",
              loc,
              dedup_key=("bad_scratch_ms", loc, bad_ms),
          )
    elif name == "empty_ref":
      for var in eqn.outvars:
        bad_ms = _is_invalid_scratch_memory_space(
            getattr(getattr(var, "aval", None), "memory_space", None)
        )
        if bad_ms is not None:
          builder.record_issue(
              _Category.HARDWARE_CONSTRAINT_VIOLATION,
              f"empty_ref cannot allocate {bad_ms} buffers inside a TPU kernel",
              loc,
              dedup_key=("bad_scratch_ms", loc, bad_ms),
          )
    elif name == "dma_start" and "tree" in eqn.params:
      try:
        ops = decode_dma_operands(eqn.params["tree"].unflatten(eqn.invars))
      except (TypeError, ValueError):
        ops = None
      if ops is not None:
        is_tc_smem = not effective_sc and (
            _static_operand_memory_space(ops.src) == "smem"
            or _static_operand_memory_space(ops.dst) == "smem"
        )
        for sem_op in (ops.dst_sem, ops.src_sem):
          if sem_op is None:
            continue
          sem_type = _detect_semaphore_type(getattr(sem_op.ref, "aval", None))
          if sem_type == "barrier":
            builder.record_issue(
                _Category.HARDWARE_CONSTRAINT_VIOLATION,
                "asynchronous copy cannot use a barrier semaphore as a DMA"
                " completion semaphore; use pltpu.SemaphoreType.DMA",
                loc,
                dedup_key=("dma_barrier_sem", loc),
            )
          elif sem_type == "regular":
            builder.uses_done_count_regular_dma = True
            if is_tc_smem:
              builder.record_issue(
                  _Category.HARDWARE_CONSTRAINT_VIOLATION,
                  "asynchronous SMEM copy uses a regular semaphore; SMEM DMA"
                  " copies require pltpu.SemaphoreType.DMA",
                  loc,
                  dedup_key=("dma_smem_reg_sem", loc),
              )
            elif not findings.supports_dma_done_count_mode(builder.tpu_version):
              builder.record_issue(
                  _Category.HARDWARE_CONSTRAINT_VIOLATION,
                  "asynchronous copy uses a regular semaphore;"
                  f" {builder.tpu_version.value} DMA copies require"
                  " pltpu.SemaphoreType.DMA",
                  loc,
                  dedup_key=("dma_v5e_reg_sem", loc),
              )
    elif name in ("semaphore_signal", "semaphore_wait") and (
        "args_tree" in eqn.params
    ):
      try:
        unflattened = eqn.params["args_tree"].unflatten(eqn.invars)
      except (TypeError, ValueError):
        unflattened = ()
      if len(unflattened) >= 3:
        amt_arg = unflattened[2]
        what = "signal" if name == "semaphore_signal" else "wait"
        amt_val = _eval_atom(amt_arg, {})
        if amt_val is not None and int(amt_val) < 0:
          builder.record_issue(
              _Category.HARDWARE_CONSTRAINT_VIOLATION,
              f"semaphore {what} amount {int(amt_val)} must be non-negative on"
              " TPU",
              loc,
              dedup_key=("neg_sem_amt", loc, what, int(amt_val)),
          )
        if name == "semaphore_wait":
          dec_arg = unflattened[3] if len(unflattened) > 3 else True
          if _is_jax_var(dec_arg):
            builder.record_issue(
                _Category.HARDWARE_CONSTRAINT_VIOLATION,
                "pl.semaphore_wait requires a static decrement=True on TPU;"
                " dynamic decrement flags are ignored by Mosaic lowering",
                loc,
                dedup_key=("sem_wait_dec", loc),
            )
          else:
            dec_val = _eval_atom(dec_arg, {})
            if dec_val is not None and not bool(dec_val):
              builder.record_issue(
                  _Category.HARDWARE_CONSTRAINT_VIOLATION,
                  "pl.semaphore_wait with decrement=False is not supported on"
                  " TPU",
                  loc,
                  dedup_key=("sem_wait_dec", loc),
              )
    elif (
        effective_sc
        and name in ("get", "swap", "load", "masked_load", "masked_swap")
        and eqn.invars
    ):
      # SparseCore `sc_lowering.py` only rejects synchronous `pl.ANY` accesses
      # during layout passes; flag them directly during static model extraction.
      ref_var = eqn.invars[0]
      transforms: Sequence[Any] = ()
      if "args_tree" in eqn.params:
        try:
          ref_var, unflat_tr, *_ = eqn.params["args_tree"].unflatten(eqn.invars)
          if isinstance(unflat_tr, Sequence):
            transforms = unflat_tr
        except (TypeError, ValueError):
          transforms = ()
      elif "tree" in eqn.params:
        has_mask = bool(eqn.params.get("has_mask", False))
        leaf_start = 2 if name == "swap" else 1
        leaf_slice = (
            eqn.invars[leaf_start:-1] if has_mask else eqn.invars[leaf_start:]
        )
        try:
          unflat_tr = jax.tree_util.tree_unflatten(
              eqn.params["tree"], leaf_slice
          )
          if isinstance(unflat_tr, Sequence):
            transforms = unflat_tr
        except (TypeError, ValueError):
          transforms = ()
      if _is_raw_any_memory_space(getattr(ref_var, "aval", None), transforms):
        builder.record_issue(
            _Category.HARDWARE_CONSTRAINT_VIOLATION,
            "SparseCore cannot perform direct synchronous load/store on pl.ANY"
            " memory space; use VMEM or asynchronous DMA",
            loc,
            dedup_key=("sc_any_sync_access", loc),
        )
    for sub in _sub_jaxprs(eqn):
      _check_static_hardware_rules(sub, builder, is_sparse_core=effective_sc)


@dataclasses.dataclass(frozen=True)
class _BlockSlice:
  """Region of the full array that a block covers at one grid point."""

  region: HyperRectSlice
  active_axes: tuple[int, ...]
  violation: str | None = None
  used_runtime_prefetch: bool = False
  unpadded_starts: tuple[int, ...] = ()
  padded_axes: frozenset[int] = frozenset()

  @property
  def window_key(self) -> tuple[HyperRectSlice, tuple[int, ...]]:
    """Distinct pipeline window identity (including padded `Element` starts)."""
    return (
        (self.region, self.unpadded_starts)
        if self.padded_axes
        else (self.region, ())
    )


def _uses_any(jaxpr: Any, variables: Sequence[Any]) -> bool:
  targets = set(variables)
  if any(v in targets for v in jaxpr.outvars if _is_jax_var(v)):
    return True
  for eqn in jaxpr.eqns:
    for v in eqn.invars:
      if _is_jax_var(v) and v in targets:
        return True
  return False


def _eval_index_map(
    bm: Any,
    grid_coords: Sequence[int],
    *,
    axis_env: Mapping[Any, int] | None = None,
    const_env: Mapping[Any, int | bool] | None = None,
    prefetch_arrays: Mapping[int, _ConcreteArray] | None = None,
    used_runtime_out: MutableSequence[bool] | None = None,
) -> list[Any] | None:
  """Evaluates a block index map; None if it depends on runtime data.

  The map is evaluated symbolically rather than executed, which would
  initialize JAX backends.

  Args:
    bm: The block mapping.
    grid_coords: The grid point.
    axis_env: Optional mesh/core axis coordinates for `lax.axis_index`.
    const_env: Optional pre-bound constants for `index_map_jaxpr.constvars`.
    prefetch_arrays: Optional concrete small integer/bool scalar-prefetch
      arrays.
    used_runtime_out: Appended with `True` when a runtime scalar-prefetch array
      element is read.

  Returns:
    The block indices, or None if they are not statically known.
  """
  closed = bm.index_map_jaxpr
  jaxpr = closed.jaxpr
  grid_vars = jaxpr.invars[: len(grid_coords)]
  extra_vars = jaxpr.invars[len(grid_coords) :]
  pref_refs: dict[Any, tuple[int, _RefView]] = {}
  if extra_vars:
    if not prefetch_arrays:
      if _uses_any(jaxpr, extra_vars):
        return None  # Depends on scalar-prefetch operands.
    else:
      for idx, var in enumerate(extra_vars):
        if idx in prefetch_arrays:
          pref_refs[var] = (
              idx,
              _full_view(
                  f"prefetch_{idx}",
                  "smem",
                  getattr(var.aval, "shape", ()),
                  getattr(var.aval, "dtype", np.int32),
              ),
          )
      unresolved_extra = [v for v in extra_vars if v not in pref_refs]
      if unresolved_extra and _uses_any(jaxpr, unresolved_extra):
        return None
  scalars: dict[Any, int | bool] = dict(zip(grid_vars, grid_coords))
  if const_env:
    scalars.update(const_env)
  for var, const in zip(jaxpr.constvars, closed.consts):
    value = _eval_atom(const, scalars)
    if value is not None:
      scalars[var] = value
  eff_axis_env = dict(axis_env or {})
  for eqn in jaxpr.eqns:
    if (
        pref_refs
        and eqn.primitive.name == "get"
        and "tree" in eqn.params
        and len(eqn.outvars) == 1
    ):
      root, pre = _unwrap_ref_and_transforms(eqn.invars[0])
      if _is_jax_var(root) and root in pref_refs:
        pref_idx, base_view = pref_refs[root]
        try:
          transforms = jax.tree_util.tree_unflatten(
              eqn.params["tree"], eqn.invars[1:]
          )
        except (TypeError, ValueError):
          return None
        view, violation = _apply_transforms(
            base_view,
            (*pre, *transforms),
            scalars,
            access_kind=_SliceAccessKind.READ,
        )
        pref_entry = (
            prefetch_arrays.get(pref_idx)
            if prefetch_arrays is not None
            else None
        )
        arr = pref_entry.values if pref_entry is not None else None
        if (
            violation is None
            and pref_entry is not None
            and arr is not None
            and view.sem_index is not None
            and len(view.sem_index) == arr.ndim
            and all(0 <= c < dim for c, dim in zip(view.sem_index, arr.shape))
        ):
          if pref_entry.is_runtime and used_runtime_out is not None:
            used_runtime_out.append(True)
          elem = arr[view.sem_index]
          if np.issubdtype(arr.dtype, np.bool_):
            scalars[eqn.outvars[0]] = bool(elem.item())
            continue
          if np.issubdtype(arr.dtype, np.integer):
            scalars[eqn.outvars[0]] = int(elem.item())
            continue
        return None
    _try_eval_scalar_eqn(
        eqn,
        scalars,
        grid_coords=grid_coords,
        grid_sizes=(),
        axis_env=eff_axis_env,
    )
  outs = [_eval_atom(v, scalars) for v in jaxpr.outvars]
  if any(o is None for o in outs):
    return None
  return bm.index_map_out_tree.unflatten(outs)


def _block_slice(
    bm: Any,
    grid_coords: Sequence[int],
    *,
    axis_env: Mapping[Any, int] | None = None,
    const_env: Mapping[Any, int | bool] | None = None,
    prefetch_arrays: Mapping[int, _ConcreteArray] | None = None,
) -> _BlockSlice | None:
  """Returns the array region of a block at a grid point, if computable."""
  used_runtime: list[bool] = []
  indices = _eval_index_map(
      bm,
      grid_coords,
      axis_env=axis_env,
      const_env=const_env,
      prefetch_arrays=prefetch_arrays,
      used_runtime_out=used_runtime,
  )
  if indices is None:
    return None
  array_shape = tuple(int(s) for s in bm.array_aval.shape)
  dims: list[SliceDim] = []
  unpadded_starts: list[int] = []
  padded_axes: set[int] = set()
  active: list[int] = []
  violation: str | None = None
  for axis, (value, block_dim) in enumerate(zip(indices, bm.block_shape)):
    size = array_shape[axis]
    if isinstance(block_dim, pl.BoundedSlice) and isinstance(value, pl.Slice):
      if not isinstance(
          value.start, (int, np.integer, np.ndarray)
      ) or not isinstance(value.size, (int, np.integer, np.ndarray)):
        return None
      start = int(np.asarray(value.start).item())
      length = int(np.asarray(value.size).item())
      max_block = int(block_dim.block_size)
      if start < 0 or length < 0 or start + length > size:
        violation = violation or (
            f"slice {start}:{start + length} is outside axis {axis} of size"
            f" {size}"
        )
      elif length > max_block:
        violation = violation or (
            f"BoundedSlice length {length} exceeds maximum block_size"
            f" {max_block} along axis {axis}"
        )
      dims.append(SliceDim(start, max(length, 0)))
      unpadded_starts.append(start)
      active.append(axis)
      continue
    if not isinstance(value, (int, np.integer, np.ndarray)) or (
        isinstance(value, np.ndarray) and value.ndim != 0
    ):
      return None
    index = int(np.asarray(value).item())
    if isinstance(block_dim, pl.Squeezed) or block_dim is None:
      if not 0 <= index < size:
        violation = violation or (
            f"index {index} is outside axis {axis} of size {size}"
        )
      dims.append(SliceDim(index, 1))
      unpadded_starts.append(index)
      continue
    if isinstance(block_dim, pl.Element):
      block_size = int(block_dim.block_size)
      raw_pad = getattr(block_dim, "padding", None) or (0, 0)
      pad_lo, pad_hi = int(raw_pad[0]), int(raw_pad[1])
      padded_size = size + pad_lo + pad_hi
      if (
          pad_lo < 0
          or pad_hi < 0
          or block_size < 0
          or index < 0
          or index + block_size > padded_size
      ):
        pad_note = (
            f" (array size {size} with padding ({pad_lo}, {pad_hi}))"
            if (pad_lo, pad_hi) != (0, 0)
            else ""
        )
        violation = violation or (
            f"element slice {index}:{index + max(block_size, 0)} is outside"
            f" axis {axis} of padded size {padded_size}{pad_note}"
        )
      raw_start = index - pad_lo
      raw_stop = raw_start + max(block_size, 0)
      clipped_lo = min(max(raw_start, 0), size)
      clipped_hi = min(max(raw_stop, 0), size)
      dims.append(SliceDim(clipped_lo, max(clipped_hi - clipped_lo, 0)))
      unpadded_starts.append(raw_start)
      if (pad_lo, pad_hi) != (0, 0):
        padded_axes.add(axis)
      active.append(axis)
      continue
    block = int(getattr(block_dim, "block_size", block_dim))
    if size == 0 or index < 0 or index * block >= size:
      violation = violation or (
          f"block index {index} along axis {axis} starts at element"
          f" {index * block}, past the array size {size}"
      )
    dims.append(SliceDim(index * block, block))
    unpadded_starts.append(index * block)
    active.append(axis)
  return _BlockSlice(
      HyperRectSlice(tuple(dims)),
      tuple(active),
      violation,
      used_runtime_prefetch=bool(used_runtime),
      unpadded_starts=tuple(unpadded_starts),
      padded_axes=frozenset(padded_axes),
  )


def _tile_region_to_array(
    tile_region: HyperRectSlice, block: _BlockSlice
) -> HyperRectSlice | None:
  """Maps a region of a VMEM tile to the array region it is written back to."""
  if len(tile_region.dims) != len(block.active_axes):
    return None
  dims = list(block.region.dims)
  for tile_dim, axis in zip(tile_region.dims, block.active_axes):
    base_start = (
        block.unpadded_starts[axis]
        if axis < len(block.unpadded_starts)
        else dims[axis].start
    )
    raw_lo = base_start + tile_dim.start
    raw_hi = base_start + tile_dim.stop
    if axis in block.padded_axes:
      blk_lo = dims[axis].start
      blk_hi = dims[axis].stop
      c_lo = min(max(raw_lo, blk_lo), blk_hi)
      c_hi = min(max(raw_hi, blk_lo), blk_hi)
      dims[axis] = SliceDim(
          c_lo,
          max(c_hi - c_lo, 0),
          exact=dims[axis].exact and tile_dim.exact,
      )
    else:
      dims[axis] = SliceDim(
          raw_lo,
          tile_dim.size,
          exact=dims[axis].exact and tile_dim.exact,
      )
  return HyperRectSlice(
      tuple(dims), opaque=tile_region.opaque or block.region.opaque
  )


def _uncovered_region(
    shape: Sequence[int],
    regions: Sequence[HyperRectSlice],
    *,
    builder: _KernelModelBuilder | None = None,
) -> HyperRectSlice | _CoverageOverflow | None:
  """Returns the bounding box of the part of `shape` no region covers.

  Args:
    shape: Target buffer or block shape along each axis.
    regions: Written `HyperRectSlice` regions to union over `shape`.
    builder: Optional model builder to record incompleteness on cell overflow.

  Returns:
    Bounding `HyperRectSlice` of uncovered elements, `None` when everything is
    covered, or `_COVERAGE_OVERFLOW` when the cell decomposition exceeds
    `_MAX_COVERAGE_CELLS`.
  """
  rank = len(shape)
  usable = [r for r in regions if len(r.dims) == rank]
  if rank == 0:
    return None if usable else HyperRectSlice(())
  if any(s == 0 for s in shape):
    return None
  if not usable:
    return HyperRectSlice(tuple(SliceDim(0, int(s)) for s in shape))
  cuts: list[np.ndarray] = []
  for axis, size in enumerate(shape):
    points = {0, int(size)}
    for region in usable:
      points.add(min(max(region.dims[axis].start, 0), int(size)))
      points.add(min(max(region.dims[axis].stop, 0), int(size)))
    cuts.append(np.asarray(sorted(points), dtype=np.int64))
  cell_shape = tuple(len(c) - 1 for c in cuts)
  if any(n <= 0 for n in cell_shape):
    return None
  if math.prod(cell_shape) > _MAX_COVERAGE_CELLS:
    if builder is not None:
      builder.note_incomplete(
          "block coverage grid exceeded cell budget",
          affects_coverage=True,
      )
    return _COVERAGE_OVERFLOW
  covered = np.zeros(cell_shape, dtype=bool)
  for region in usable:
    slices = []
    valid = True
    for axis, c in enumerate(cuts):
      lo = min(max(region.dims[axis].start, 0), int(shape[axis]))
      hi = min(max(region.dims[axis].stop, 0), int(shape[axis]))
      if hi <= lo:
        valid = False
        break
      i0 = int(np.searchsorted(c, lo, side="left"))
      i1 = int(np.searchsorted(c, hi, side="right")) - 1
      if i1 <= i0:
        valid = False
        break
      slices.append(slice(i0, i1))
    if valid:
      covered[tuple(slices)] = True
  if covered.all():
    return None
  uncovered = ~covered
  dims: list[SliceDim] = []
  for axis, c in enumerate(cuts):
    other_axes = tuple(i for i in range(rank) if i != axis)
    axis_uncov = uncovered.any(axis=other_axes) if other_axes else uncovered
    idx = np.flatnonzero(axis_uncov)
    lo = int(c[int(idx[0])])
    hi = int(c[int(idx[-1]) + 1])
    dims.append(SliceDim(lo, hi - lo))
  return HyperRectSlice(tuple(dims))


@dataclasses.dataclass(frozen=True)
class _Operand:
  """A pipelined or directly accessed `pallas_call` or `core_map` operand."""

  index: int  # Position among all block operands.
  io_index: int  # Position among the inputs or among the outputs.
  var: Any
  block_mapping: Any
  buffer_id: str
  is_output: bool
  memory_space: str
  array_shape: tuple[int, ...] = ()
  sem_type: str | None = None
  revisit_any: bool = False
  buffer_count: int = 1

  def describe(self) -> str:
    return f"{'output' if self.is_output else 'input'} {self.io_index}"

  def full_shape(self) -> tuple[int, ...]:
    if self.array_shape:
      return self.array_shape
    if self.block_mapping is not None:
      return tuple(int(s) for s in self.block_mapping.array_aval.shape)
    return ()


def _operands(eqn: Any, kernel_jaxpr: Any, grid_mapping: Any) -> list[_Operand]:
  io_aliases = dict(eqn.params.get("input_output_aliases", ()))
  num_prefetch = int(getattr(grid_mapping, "num_index_operands", 0))
  out_to_in = {out_i: in_i for in_i, out_i in io_aliases.items()}
  block_vars = kernel_jaxpr.invars[grid_mapping.slice_block_ops]
  num_inputs = grid_mapping.num_inputs
  operands = []
  for b_idx, (b_var, bm) in enumerate(
      zip(block_vars, grid_mapping.block_mappings)
  ):
    is_output = b_idx >= num_inputs
    io_index = b_idx - num_inputs if is_output else b_idx
    if not is_output:
      buffer_id = f"arg_in_{b_idx}"
    elif io_index in out_to_in:
      raw_in = out_to_in[io_index]
      block_in = raw_in - num_prefetch
      buffer_id = (
          f"arg_in_{block_in}" if block_in >= 0 else f"prefetch_{raw_in}"
      )
    else:
      buffer_id = f"arg_out_{io_index}"
    block_aval = _block_aval(bm)
    sem_type = _detect_semaphore_type(block_aval)
    memory_space = _normalize_memory_space(
        getattr(block_aval, "memory_space", None),
        default="semaphore_mem" if sem_type is not None else "vmem",
    )
    pm = getattr(bm, "pipeline_mode", None)
    revisit = getattr(pm, "revisit", None)
    revisit_any = getattr(revisit, "value", revisit) == "any"
    raw_buf_cnt = getattr(pm, "buffer_count", 1)
    buffer_count = 1 if raw_buf_cnt is None else int(raw_buf_cnt)
    operands.append(
        _Operand(
            b_idx,
            io_index,
            b_var,
            bm,
            buffer_id,
            is_output,
            memory_space,
            tuple(int(s) for s in bm.array_aval.shape),
            sem_type=sem_type,
            revisit_any=revisit_any,
            buffer_count=buffer_count,
        )
    )
  return operands


def _bind_operands(
    ctx: _WalkContext,
    env: _Env,
    operands: Sequence[_Operand],
    pipelined: bool,
) -> list[tuple[_Operand, str, _BlockSlice | None]]:
  """Binds operand refs at a grid point; returns pipelined tiles."""
  tiles = []
  for op in operands:
    bm = op.block_mapping
    block_aval = _block_aval(bm)
    dtype = getattr(block_aval, "dtype", np.float32)
    array_shape = op.full_shape()
    if op.sem_type is not None:
      env.refs[op.var] = _full_view(
          op.buffer_id,
          "semaphore_mem",
          array_shape,
          np.int32,
          sem_type=op.sem_type,
      )
      continue
    block = (
        _block_slice(
            bm,
            ctx.grid_coords,
            axis_env=ctx.axis_env,
            prefetch_arrays=ctx.prefetch_arrays,
        )
        if pipelined
        else None
    )
    if block is not None and block.violation is not None:
      viol_msg = (
          f"{block.violation} (for the provided example inputs)"
          if block.used_runtime_prefetch
          else block.violation
      )
      ctx.builder.record_issue(
          _Category.OUT_OF_BOUNDS_ACCESS,
          f"the BlockSpec index map of {op.describe()} at grid point"
          f" {ctx.grid_coords}: {viol_msg}",
          dedup_key=("block_oob", op.index),
      )
    if op.memory_space == "hbm" or not pipelined:
      env.refs[op.var] = _full_view(
          op.buffer_id, op.memory_space, array_shape, dtype
      )
      if op.is_output and op.buffer_id.startswith("arg_out_"):
        ctx.unaliased_output_buffers[op.buffer_id] = (op.describe(), op.index)
      continue
    tile_shape = tuple(int(s) for s in block_aval.shape)
    tile_id = f"tile_{op.index}_{ctx.grid_tag()}"
    env.refs[op.var] = _full_view(tile_id, op.memory_space, tile_shape, dtype)
    if op.is_output and op.buffer_id.startswith("arg_out_"):
      ctx.unaliased_output_buffers[tile_id] = (op.describe(), op.index)
      stream_actor = ctx.pipeline_stream_actor
      tile_key = (stream_actor, op.index)
      if block is not None:
        w_key = block.window_key
        if ctx.builder.active_output_tile_block.get(tile_key) != w_key:
          ctx.builder.active_output_tile_block[tile_key] = w_key
          ctx.builder.active_output_tile_writes[tile_key] = (
              list(
                  ctx.builder.revisit_any_tile_writes.get(
                      (stream_actor, op.index, w_key), ()
                  )
              )
              if op.revisit_any
              else []
          )
      else:
        ctx.builder.active_output_tile_block.pop(tile_key, None)
        ctx.builder.active_output_tile_writes.pop(tile_key, None)
    if block is None:
      ctx.builder.note_incomplete(
          f"the block index map of {op.describe()} depends on runtime data,"
          " so its pipelined accesses are not checked",
          affects_coverage=op.is_output,
      )
    elif block.used_runtime_prefetch:
      ctx.builder.note_incomplete(
          f"the block index map of {op.describe()} was evaluated at the"
          " provided example inputs",
          affects_coverage=False,
      )
    tiles.append((op, tile_id, block))
  return tiles


def _record_pipelined_tile_writes(
    ctx: _WalkContext,
    tiles: Sequence[tuple[_Operand, str, _BlockSlice | None]],
) -> None:
  """Records pipelined output block visits and HBM output coverage of tiles."""
  builder = ctx.builder
  _record_step_written_buffers(ctx)
  for op, tile_id, block in tiles:
    if block is None:
      continue
    if op.is_output:
      stream_actor = ctx.pipeline_stream_actor
      tile_writes = list(ctx.written.get(tile_id, ()))
      if ctx.grid_sizes:
        exact_tile_writes = tuple(
            r
            for r in tile_writes
            if not r.opaque and all(d.exact for d in r.dims)
        )
        builder.output_block_visits.setdefault(
            (stream_actor, op.index), []
        ).append((
            ctx.grid_coords,
            block.window_key,
            block.region,
            tile_id in ctx.read_before_full_write,
            exact_tile_writes,
        ))
        builder.parallel_output_blocks.setdefault(
            (ctx.device_id, op.index, block.region), []
        ).append((ctx.actor, ctx.grid_coords, tile_writes))
      if op.buffer_id.startswith("arg_out_"):
        writes_for_tile = builder.active_output_tile_writes.setdefault(
            (stream_actor, op.index), []
        )
        writes_for_tile.extend(tile_writes)
        if op.revisit_any:
          builder.revisit_any_tile_writes[
              (stream_actor, op.index, block.window_key)
          ] = list(writes_for_tile)
      elif op.buffer_id.startswith("arg_in_"):
        builder.aliased_pipelined_blocks.setdefault(
            (ctx.device_id, op.index), []
        ).append(block.region)
      for region in tile_writes:
        array_region = _tile_region_to_array(region, block)
        if array_region is None:
          builder.note_incomplete(
              f"the block layout of {op.describe()} is not understood",
              affects_coverage=True,
          )
        else:
          builder.record_output_write(ctx.device_id, op.buffer_id, array_region)
          if op.buffer_id.startswith("arg_in_"):
            builder.record_output_write(
                ctx.device_id, f"pipelined_out_{op.io_index}", array_region
            )
      for (target_dev, w_tile_id), regions in ctx.remote_written.items():
        if w_tile_id != tile_id:
          continue
        for region in regions:
          array_region = _tile_region_to_array(region, block)
          if array_region is None:
            builder.note_incomplete(
                f"the block layout of {op.describe()} is not understood",
                affects_coverage=True,
            )
          else:
            builder.record_output_write(target_dev, op.buffer_id, array_region)
            if op.buffer_id.startswith("arg_in_"):
              builder.record_output_write(
                  target_dev, f"pipelined_out_{op.io_index}", array_region
              )


def _check_output_coverage(
    builder: _KernelModelBuilder,
    operands: Sequence[_Operand],
    mesh_ctx: _MeshContext,
) -> None:
  """Checks non-consecutive output block revisits and unwritten outputs."""
  op_by_index = {op.index: op for op in operands}
  saw_revisit_warn = False
  saw_revisit_err = False
  for (_, op_idx), visits in builder.output_block_visits.items():
    op = op_by_index.get(op_idx)
    if op is None or op.revisit_any or not op.buffer_id.startswith("arg_out_"):
      continue
    tile_shape = (
        tuple(int(s) for s in _block_aval(op.block_mapping).shape)
        if op.block_mapping is not None
        else ()
    )
    # Coalesce consecutive iterations targeting the same pipeline window into a
    # single resident-VMEM visit run before checking non-consecutive revisits.
    runs: list[
        tuple[
            tuple[int, ...],
            tuple[HyperRectSlice, tuple[int, ...]],
            HyperRectSlice,
            bool,
            list[HyperRectSlice],
        ]
    ] = []
    overflowed_revisit = False
    for g_coords, w_key, region, was_read, step_writes in sorted(
        visits, key=lambda item: item[0]
    ):
      if runs and runs[-1][1] == w_key:
        prev_g, prev_k, prev_reg, prev_read, prev_writes = runs[-1]
        # Reading on step > 0 of a consecutive run reads the resident VMEM tile
        # already initialized by earlier steps in the same consecutive run if
        # `prev_writes` covered the tile; `_check_uninitialized_output_read`
        # handles uninitialized reads on the first step.
        prev_uncov = (
            _uncovered_region(tile_shape, prev_writes, builder=builder)
            if prev_writes
            else HyperRectSlice(tuple(SliceDim(0, s) for s in tile_shape))
        )
        if prev_uncov is _COVERAGE_OVERFLOW:
          overflowed_revisit = True
          still_reads_stale = prev_read
        else:
          still_reads_stale = prev_read or (was_read and prev_uncov is not None)
        prev_writes.extend(step_writes)
        runs[-1] = (prev_g, prev_k, prev_reg, still_reads_stale, prev_writes)
      else:
        runs.append((g_coords, w_key, region, was_read, list(step_writes)))

    if overflowed_revisit:
      continue

    seen_windows: set[tuple[HyperRectSlice, tuple[int, ...]]] = set()
    last_region: HyperRectSlice | None = None
    first_warn: (
        tuple[tuple[int, ...], HyperRectSlice, HyperRectSlice] | None
    ) = None
    emitted_error = False
    for g_coords, w_key, region, run_read_stale, run_writes in runs:
      if w_key in seen_windows and last_region is not None:
        run_uncov = (
            _uncovered_region(tile_shape, run_writes, builder=builder)
            if run_writes
            else HyperRectSlice(tuple(SliceDim(0, s) for s in tile_shape))
        )
        if run_uncov is _COVERAGE_OVERFLOW:
          first_warn = None
          break
        run_fully_written = run_uncov is None
        if run_read_stale or not run_fully_written:
          builder.record_issue(
              _Category.HARDWARE_CONSTRAINT_VIOLATION,
              f"the BlockSpec of {op.describe()} at grid point {g_coords}"
              f" revisits output block {region} after visiting {last_region}"
              " without overwriting the full tile from scratch; unaliased"
              " output tiles do not reload previously written HBM values across"
              " non-consecutive visits",
              dedup_key=("non_consec_revisit_err", op_idx),
          )
          emitted_error = True
          saw_revisit_err = True
          break
        if first_warn is None:
          first_warn = (g_coords, region, last_region)
      last_region = region
      seen_windows.add(w_key)
    if not emitted_error and first_warn is not None:
      saw_revisit_warn = True
      w_coords, w_region, w_last = first_warn
      builder.record_issue(
          _Category.HARDWARE_CONSTRAINT_VIOLATION,
          f"the BlockSpec of {op.describe()} at grid point {w_coords}"
          f" revisits output block {w_region} after visiting {w_last};"
          " full-tile overwrites remain defined, but the TPU output pipeline"
          " performs redundant HBM writebacks unless iterations targeting the"
          " same output block are consecutive",
          severity=_Severity.WARNING,
          dedup_key=("non_consec_revisit_warn", op_idx),
      )

  if saw_revisit_warn and not saw_revisit_err:
    builder.has_verified_full_overwrite_revisit = True

  if not builder.coverage_known:
    return

  for (
      device_id,
      op_idx,
      region,
  ), entries in builder.parallel_output_blocks.items():
    op = op_by_index.get(op_idx)
    if op is None:
      continue
    tile_shape = (
        tuple(int(s) for s in _block_aval(op.block_mapping).shape)
        if op.block_mapping is not None
        else ()
    )
    by_actor: dict[_ActorKey, tuple[tuple[int, ...], list[HyperRectSlice]]] = {}
    for act, g, regs in entries:
      prev = by_actor.get(act)
      if prev is None:
        by_actor[act] = (g, list(regs))
      else:
        prev[1].extend(regs)
    if len(by_actor) < 2:
      continue
    writers = [g for _, (g, regs) in by_actor.items() if regs]
    incomplete_actors: list[tuple[_ActorKey, tuple[int, ...]]] = []
    for act, (g, regs) in by_actor.items():
      if not any(
          other_act != act and o_regs
          for other_act, (_, o_regs) in by_actor.items()
      ):
        continue
      if not regs:
        incomplete_actors.append((act, g))
      elif tile_shape:
        uncov = _uncovered_region(tile_shape, regs, builder=builder)
        if isinstance(uncov, HyperRectSlice):
          incomplete_actors.append((act, g))
    if writers and incomplete_actors:
      where = f" on device {device_id}" if mesh_ctx.num_devices > 1 else ""
      builder.record_issue(
          _Category.UNWRITTEN_OUTPUT,
          f"{op.describe()} block {region} is mapped by multiple parallel grid"
          f" points ({writers[0]} and {incomplete_actors[0][1]}){where}, and"
          f" grid point {incomplete_actors[0][1]} leaves its VMEM tile"
          " unwritten (or partially unwritten) so its output pipeline"
          " writeback clobbers the block with uninitialized memory",
      )

  for op in operands:
    if not op.buffer_id.startswith("arg_out_") or op.sem_type is not None:
      continue
    shape = op.full_shape()
    for device_id, _ in mesh_ctx.iter_devices():
      regions = builder.output_writes.get((device_id, op.buffer_id), [])
      missing = _uncovered_region(shape, regions, builder=builder)
      if not isinstance(missing, HyperRectSlice):
        continue
      where = f" on device {device_id}" if mesh_ctx.num_devices > 1 else ""
      builder.record_issue(
          _Category.UNWRITTEN_OUTPUT,
          f"{op.describe()} (shape {shape}) region {missing} is never"
          f" written{where}, so it returns uninitialized memory",
      )

  for (
      device_id,
      op_idx,
  ), visited_blocks in builder.aliased_pipelined_blocks.items():
    op = op_by_index.get(op_idx)
    if op is None:
      continue
    pipe_key = (device_id, f"pipelined_out_{op.io_index}")
    writes = builder.output_writes.get(pipe_key, [])
    seen_blocks: set[HyperRectSlice] = set()
    for block_reg in visited_blocks:
      if block_reg in seen_blocks:
        continue
      seen_blocks.add(block_reg)
      block_shape = tuple(
          max(min(d.stop, full_s) - max(d.start, 0), 0)
          for d, full_s in zip(block_reg.dims, op.full_shape())
      )
      if any(s == 0 for s in block_shape):
        continue
      rel_writes: list[HyperRectSlice] = []
      for w in writes:
        if len(w.dims) != len(block_reg.dims):
          continue
        rel_dims = tuple(
            SliceDim(
                wd.start - max(bd.start, 0),
                wd.size,
                exact=wd.exact and bd.exact,
            )
            for wd, bd in zip(w.dims, block_reg.dims)
        )
        rel_writes.append(HyperRectSlice(rel_dims, opaque=w.opaque))
      missing_rel = _uncovered_region(block_shape, rel_writes, builder=builder)
      if isinstance(missing_rel, HyperRectSlice):
        missing_abs = HyperRectSlice(
            tuple(
                SliceDim(max(bd.start, 0) + md.start, md.size)
                for bd, md in zip(block_reg.dims, missing_rel.dims)
            )
        )
        where = f" on device {device_id}" if mesh_ctx.num_devices > 1 else ""
        builder.record_issue(
            _Category.UNWRITTEN_OUTPUT,
            f"{op.describe()} (aliased to input) pipelined VMEM block"
            f" {block_reg} leaves region {missing_abs} unwritten{where};"
            " output pipeline writeback overwrites the entire block from VMEM",
            dedup_key=("aliased_pipe_unwritten", device_id, op_idx),
        )
        break


def _grid_and_semantics(
    grid_mapping: Any, mosaic_params: Any, builder_note: Callable[[str], None]
) -> tuple[tuple[int, ...], tuple[int | None, ...], tuple[str, ...]]:
  grid: list[int] = []
  grid_sizes: list[int | None] = []
  for d in grid_mapping.grid:
    if isinstance(d, (int, np.integer)):
      grid.append(int(d))
      grid_sizes.append(int(d))
    else:
      grid.append(1)
      grid_sizes.append(None)
      builder_note(
          "a grid axis is only sized at run time, so one step of it is checked"
      )
  vmapped_dims = frozenset(getattr(grid_mapping, "vmapped_dims", None) or ())
  raw_sems = iter(getattr(mosaic_params, "dimension_semantics", None) or ())
  semantics: list[str] = []
  for axis in range(len(grid)):
    if axis in vmapped_dims:
      semantics.append("parallel")
      continue
    sem = next(raw_sems, "arbitrary")
    sem_val = getattr(sem, "value", str(sem))
    is_parallel = sem_val in ("parallel", "core_parallel", "subcore_parallel")
    semantics.append("parallel" if is_parallel else "arbitrary")
  return tuple(grid), tuple(grid_sizes), tuple(semantics)


def _extract_pallas_call_model(
    eqn: Any,
    mesh_ctx: _MeshContext,
    *,
    max_unrolled_iterations: int,
    tpu_version: findings.TpuVersion = fx_verifier.DEFAULT_TPU_VERSION,
    concrete_arrays: Mapping[Any, _ConcreteArray] | None = None,
) -> KernelVerificationModel:
  """Extracts a `KernelVerificationModel` from one `pallas_call` equation."""
  kernel_jaxpr = _unwrap_closed_jaxpr(eqn.params["jaxpr"])
  grid_mapping: Any = eqn.params["grid_mapping"]
  mosaic_params = _extract_mosaic_compiler_params(
      eqn.params.get("compiler_params")
  )
  collective_id = getattr(mosaic_params, "collective_id", None)
  is_sparse_core = _is_sparse_core_type(
      getattr(mosaic_params, "kernel_type", None)
  ) or _is_sparse_core_type(getattr(eqn.params.get("mesh"), "core_type", None))
  notes: list[str] = []
  grid, grid_sizes, semantics = _grid_and_semantics(
      grid_mapping, mosaic_params, notes.append
  )
  vmapped_dims = frozenset(getattr(grid_mapping, "vmapped_dims", None) or ())
  grid_names = tuple(getattr(grid_mapping, "grid_names", None) or ())
  builder = _KernelModelBuilder(
      kernel_name=_kernel_name(eqn.params),
      num_devices=mesh_ctx.num_devices,
      grid=grid,
      dimension_semantics=semantics,
      collective_id=collective_id,
      max_unrolled_iterations=max_unrolled_iterations,
      tpu_version=tpu_version,
      is_sparse_core=is_sparse_core,
  )
  for note in notes:
    builder.note_incomplete(note, affects_coverage=True)
  _check_static_hardware_rules(
      kernel_jaxpr, builder, is_sparse_core=is_sparse_core
  )
  for idx, var in enumerate(
      kernel_jaxpr.invars[grid_mapping.slice_scratch_ops]
  ):
    bad_ms = _is_invalid_scratch_memory_space(
        getattr(var.aval, "memory_space", None),
        allow_hbm=is_sparse_core,
    )
    if bad_ms is not None:
      builder.record_issue(
          _Category.HARDWARE_CONSTRAINT_VIOLATION,
          f"scratch_shapes[{idx}] cannot allocate {bad_ms} buffers on TPU",
      )

  operands = _operands(eqn, kernel_jaxpr, grid_mapping)
  for op in operands:
    if op.is_output and op.revisit_any:
      builder.has_revisit_any_output = True

  prefetch_arrays: dict[int, _ConcreteArray] = {}
  num_index = int(getattr(grid_mapping, "num_index_operands", 0))
  if num_index > 0:
    idx_offset = len(kernel_jaxpr.constvars) + int(
        getattr(grid_mapping, "num_dynamic_grid_bounds", 0)
    )
    for pref_idx in range(num_index):
      outer_pos = idx_offset + pref_idx
      if outer_pos >= len(eqn.invars):
        break
      invar = eqn.invars[outer_pos]
      entry: _ConcreteArray | None = None
      if concrete_arrays and _is_jax_var(invar):
        entry = concrete_arrays.get(invar)
      elif hasattr(invar, "val"):
        arr = _extract_small_int_cpu_array(invar.val)
        if arr is not None:
          entry = _ConcreteArray(arr, is_runtime=False)
      if entry is not None:
        prefetch_arrays[pref_idx] = entry

  num_points = math.prod(grid) if grid else 1
  if num_points > _MAX_COVERAGE_GRID_POINTS:
    builder.note_incomplete(
        f"only the first {_MAX_COVERAGE_GRID_POINTS} of {num_points} grid"
        " points are checked for out-of-bounds and unwritten outputs",
        affects_coverage=True,
    )
  points = list(
      itertools.islice(
          itertools.product(*(range(s) for s in grid)),
          _MAX_COVERAGE_GRID_POINTS,
      )
  )
  for dev_id, dev_coords in mesh_ctx.iter_devices():
    for g_coords in points:
      _walk_grid_point(
          eqn,
          kernel_jaxpr,
          grid_mapping,
          operands,
          _WalkContext(
              builder=builder,
              actor=_ActorKey(
                  dev_id,
                  0,
                  tuple(
                      g_coords[ax]
                      for ax, sem in enumerate(semantics)
                      if sem == "parallel"
                  ),
              ),
              device_id=dev_id,
              device_coords=dev_coords,
              mesh_ctx=mesh_ctx,
              grid_coords=g_coords,
              grid_sizes=grid_sizes,
              vmapped_dims=vmapped_dims,
              scope_path="",
              touched_refs=set(),
              written={},
              extra_axis_env=(
                  {
                      name: coord
                      for name, coord in zip(grid_names, g_coords)
                      if name is not None
                  }
                  if grid_names
                  else {}
              ),
              prefetch_arrays=prefetch_arrays,
          ),
      )
  _check_output_coverage(builder, operands, mesh_ctx)
  return builder.build()


def _walk_grid_point(
    eqn: Any,
    kernel_jaxpr: Any,
    grid_mapping: Any,
    operands: Sequence[_Operand],
    ctx: _WalkContext,
) -> None:
  """Walks the kernel body for one device and grid point."""
  env = _Env({}, {})
  for outer_v, const_v in zip(
      eqn.invars[: len(kernel_jaxpr.constvars)], kernel_jaxpr.constvars
  ):
    value = _eval_atom(outer_v, {})
    if value is not None:
      env.scalars[const_v] = value
  for idx, var in enumerate(kernel_jaxpr.invars[grid_mapping.slice_index_ops]):
    env.refs[var] = _full_view(
        f"prefetch_{idx}",
        "smem",
        getattr(var.aval, "shape", ()),
        getattr(var.aval, "dtype", np.int32),
    )
  for idx, var in enumerate(
      kernel_jaxpr.invars[grid_mapping.slice_scratch_ops]
  ):
    aval = var.aval
    sem_type = _detect_semaphore_type(aval)
    env.refs[var] = _full_view(
        f"scratch_{idx}",
        _normalize_memory_space(
            getattr(aval, "memory_space", None),
            default="semaphore_mem" if sem_type is not None else "vmem",
        ),
        getattr(var.aval, "shape", ()),
        getattr(var.aval, "dtype", np.float32),
        sem_type=sem_type,
    )
  tiles = _bind_operands(ctx, env, operands, pipelined=True)
  _walk(kernel_jaxpr, ctx, env)
  _record_pipelined_tile_writes(ctx, tiles)


def _find_outer_empty_ref_outputs(jaxpr: Any) -> set[Any]:
  """Returns `empty_ref` variables in `jaxpr` that flow to `outvars`."""
  empty_refs: set[Any] = set()
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "empty_ref":
      for v in eqn.outvars:
        if _is_jax_var(v):
          empty_refs.add(v)
  if not empty_refs:
    return set()
  reachable: set[Any] = {v for v in jaxpr.outvars if _is_jax_var(v)}
  for eqn in reversed(jaxpr.eqns):
    if eqn.primitive.name in ("mpmd_map", "pallas_call", "empty_ref"):
      continue
    if any(_is_jax_var(ov) and ov in reachable for ov in eqn.outvars):
      for iv in eqn.invars:
        root_iv, _ = _unwrap_ref_and_transforms(iv)
        if _is_jax_var(root_iv):
          reachable.add(root_iv)
  return empty_refs & reachable


def _mesh_axes_grid_semantics(
    pallas_mesh: Any,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
  """Extracts `(axis_names, grid, semantics)` from a Pallas `Mesh`."""
  mesh_shape_dict = dict(getattr(pallas_mesh, "shape", {}) or {})
  axis_names = tuple(str(k) for k in mesh_shape_dict.keys())
  grid = tuple(int(s) for s in mesh_shape_dict.values())
  raw_sems = getattr(pallas_mesh, "dimension_semantics", None) or ()
  semantics = tuple(
      "parallel"
      if getattr(
          raw_sems[ax] if ax < len(raw_sems) else "parallel",
          "value",
          str(raw_sems[ax] if ax < len(raw_sems) else "parallel"),
      )
      in ("parallel", "core_parallel", "subcore_parallel")
      else "arbitrary"
      for ax in range(len(grid))
  )
  return axis_names, grid, semantics


def _extract_aval_mesh_info(
    aval: Any,
    meshes: Sequence[Any],
) -> tuple[
    tuple[str, ...],
    tuple[int | None, ...],
    tuple[str, ...],
    int | None,
]:
  """Extracts destination core-mesh metadata from a `CoreMemorySpace` aval."""
  ms = getattr(aval, "memory_space", None)
  target_mesh = getattr(ms, "mesh", None)
  if target_mesh is None:
    return (), (), (), None
  axis_names, grid, semantics = _mesh_axes_grid_semantics(target_mesh)
  owner_core_id: int | None = None
  for idx, candidate in enumerate(meshes):
    if candidate == target_mesh:
      owner_core_id = idx
      break
  return axis_names, grid, semantics, owner_core_id


def _extract_mpmd_map_models(
    eqn: Any,
    mesh_ctx: _MeshContext,
    *,
    max_unrolled_iterations: int,
    tpu_version: findings.TpuVersion = fx_verifier.DEFAULT_TPU_VERSION,
    outer_output_vars: AbstractSet[Any] = frozenset(),
) -> list[KernelVerificationModel]:
  """Extracts kernel models from a `pl.core_map` (`mpmd_map`) equation."""
  meshes = tuple(eqn.params.get("meshes", ()) or ())
  external_meshes = tuple(eqn.params.get("external_meshes", ()) or ())
  all_meshes = (*meshes, *external_meshes)
  any_sparse_core = any(
      _is_sparse_core_type(getattr(m, "core_type", None)) for m in all_meshes
  )
  all_core_axis_sizes: dict[str, int] = {}
  for m in all_meshes:
    m_axes, m_grid, _ = _mesh_axes_grid_semantics(m)
    for ax_name, ax_size in zip(m_axes, m_grid):
      all_core_axis_sizes.setdefault(ax_name, ax_size)
  jaxprs = eqn.params.get("jaxprs", ())
  mosaic_params = _extract_mosaic_compiler_params(
      eqn.params.get("compiler_params")
  )
  collective_id = getattr(mosaic_params, "collective_id", None)
  io_aliases = dict(eqn.params.get("input_output_aliases", ()))
  out_to_in = {out_i: in_i for in_i, out_i in io_aliases.items()}
  num_in = len(eqn.invars)
  out_avals = eqn.params.get("out_avals", eqn.outvars)
  num_out = len(out_avals)
  outer_out_indices: dict[int, int] = {}
  next_outer_out = num_out
  for idx, invar in enumerate(eqn.invars):
    if _is_jax_var(invar) and invar in outer_output_vars:
      outer_out_indices[idx] = next_outer_out
      next_outer_out += 1
  shared_builder: _KernelModelBuilder | None = None
  all_mpmd_operands: list[_Operand] = []
  seen_op_indices: set[int] = set()
  for mesh_idx, (pallas_mesh, raw_jaxpr) in enumerate(zip(meshes, jaxprs)):
    kernel_jaxpr = _unwrap_closed_jaxpr(raw_jaxpr)
    mesh_default_ms = _normalize_memory_space(
        getattr(pallas_mesh, "default_memory_space", None),
        default="hbm",
    )
    is_sparse_core = _is_sparse_core_type(
        getattr(pallas_mesh, "core_type", None)
    )
    axis_names, grid, semantics = _mesh_axes_grid_semantics(pallas_mesh)
    if shared_builder is None:
      shared_builder = _KernelModelBuilder(
          kernel_name=_kernel_name(eqn.params),
          num_devices=mesh_ctx.num_devices,
          grid=grid,
          dimension_semantics=semantics,
          collective_id=collective_id,
          max_unrolled_iterations=max_unrolled_iterations,
          tpu_version=tpu_version,
          is_sparse_core=is_sparse_core or any_sparse_core,
          core_axis_names=tuple(all_core_axis_sizes.keys()) or axis_names,
      )
    elif axis_names and not shared_builder.core_axis_names:
      shared_builder.core_axis_names = axis_names
    builder = shared_builder
    _check_static_hardware_rules(
        kernel_jaxpr, builder, is_sparse_core=is_sparse_core
    )
    allow_hbm_scratch = len(meshes) > 1 or is_sparse_core or any_sparse_core
    for scr_idx, var in enumerate(kernel_jaxpr.invars[num_in + num_out :]):
      bad_ms = _is_invalid_scratch_memory_space(
          getattr(getattr(var, "aval", None), "memory_space", None),
          allow_hbm=allow_hbm_scratch,
      )
      if bad_ms is not None:
        builder.record_issue(
            _Category.HARDWARE_CONSTRAINT_VIOLATION,
            f"scratch[{scr_idx}] cannot allocate {bad_ms} buffers on TPU",
            dedup_key=("mpmd_scratch_ms", mesh_idx, scr_idx, bad_ms),
        )
    num_points = math.prod(grid) if grid else 1
    if num_points > _MAX_COVERAGE_GRID_POINTS:
      builder.note_incomplete(
          f"only the first {_MAX_COVERAGE_GRID_POINTS} of {num_points} grid"
          " points are checked for out-of-bounds and unwritten outputs",
          affects_coverage=True,
      )
    mpmd_operands: list[_Operand] = []
    for idx, out_i in outer_out_indices.items():
      if idx >= len(kernel_jaxpr.invars):
        continue
      var = kernel_jaxpr.invars[idx]
      aval = getattr(var, "aval", None)
      sem_type = _detect_semaphore_type(aval)
      mpmd_operands.append(
          _Operand(
              index=idx,
              io_index=out_i,
              var=var,
              block_mapping=None,
              buffer_id=f"arg_out_{out_i}",
              is_output=True,
              memory_space=_normalize_memory_space(
                  getattr(aval, "memory_space", None),
                  default="semaphore_mem"
                  if sem_type is not None
                  else mesh_default_ms,
              ),
              array_shape=tuple(int(s) for s in getattr(aval, "shape", ())),
              sem_type=sem_type,
          )
      )
    for idx in range(num_in, min(len(kernel_jaxpr.invars), num_in + num_out)):
      var = kernel_jaxpr.invars[idx]
      aval = getattr(var, "aval", None)
      sem_type = _detect_semaphore_type(aval)
      out_i = idx - num_in
      buf_id = (
          f"arg_in_{out_to_in[out_i]}"
          if out_i in out_to_in
          else f"arg_out_{out_i}"
      )
      mpmd_operands.append(
          _Operand(
              index=idx,
              io_index=out_i,
              var=var,
              block_mapping=None,
              buffer_id=buf_id,
              is_output=True,
              memory_space=_normalize_memory_space(
                  getattr(aval, "memory_space", None),
                  default="semaphore_mem"
                  if sem_type is not None
                  else mesh_default_ms,
              ),
              array_shape=tuple(int(s) for s in getattr(aval, "shape", ())),
              sem_type=sem_type,
          )
      )
    for op in mpmd_operands:
      if op.index not in seen_op_indices:
        seen_op_indices.add(op.index)
        all_mpmd_operands.append(op)
    points = list(
        itertools.islice(
            itertools.product(*(range(s) for s in grid)),
            _MAX_COVERAGE_GRID_POINTS,
        )
    )
    for dev_id, dev_coords in mesh_ctx.iter_devices():
      for g_coords in points:
        env = _Env({}, {})
        unaliased_out: dict[str, tuple[str, int]] = {}
        for idx, var in enumerate(kernel_jaxpr.invars):
          aval = getattr(var, "aval", None)
          sem_type = _detect_semaphore_type(aval)
          t_axes, t_sizes, t_sems, t_owner = _extract_aval_mesh_info(
              aval, meshes
          )
          if idx < num_in:
            val = _eval_atom(eqn.invars[idx], {})
            if val is not None:
              env.scalars[var] = val
            if idx in outer_out_indices:
              out_i = outer_out_indices[idx]
              buf_id = f"arg_out_{out_i}"
              unaliased_out[buf_id] = (f"output {out_i}", idx)
            else:
              buf_id = f"arg_in_{idx}"
            env.refs[var] = _full_view(
                buf_id,
                _normalize_memory_space(
                    getattr(aval, "memory_space", None),
                    default="semaphore_mem"
                    if sem_type is not None
                    else mesh_default_ms,
                ),
                getattr(aval, "shape", ()),
                getattr(aval, "dtype", np.float32),
                sem_type=sem_type,
                target_core_axes=t_axes,
                target_core_sizes=t_sizes,
                target_core_semantics=t_sems,
                owner_core_id=t_owner,
            )
          elif idx < num_in + num_out:
            out_i = idx - num_in
            buf_id = (
                f"arg_in_{out_to_in[out_i]}"
                if out_i in out_to_in
                else f"arg_out_{out_i}"
            )
            if buf_id.startswith("arg_out_"):
              unaliased_out[buf_id] = (f"output {out_i}", idx)
            env.refs[var] = _full_view(
                buf_id,
                _normalize_memory_space(
                    getattr(aval, "memory_space", None),
                    default="semaphore_mem"
                    if sem_type is not None
                    else mesh_default_ms,
                ),
                getattr(aval, "shape", ()),
                getattr(aval, "dtype", np.float32),
                sem_type=sem_type,
                target_core_axes=t_axes,
                target_core_sizes=t_sizes,
                target_core_semantics=t_sems,
                owner_core_id=t_owner,
            )
          else:
            scr_i = idx - (num_in + num_out)
            # `mpmd_map` allocates one shared scratch tuple passed to every
            # sub-mesh, so cross-mesh HBM scratch and semaphores share IDs.
            scr_id = f"scratch_{scr_i}"
            env.refs[var] = _full_view(
                scr_id,
                _normalize_memory_space(
                    getattr(aval, "memory_space", None),
                    default="semaphore_mem" if sem_type is not None else "vmem",
                ),
                getattr(aval, "shape", ()),
                getattr(aval, "dtype", np.float32),
                sem_type=sem_type,
                target_core_axes=t_axes,
                target_core_sizes=t_sizes,
                target_core_semantics=t_sems,
                owner_core_id=t_owner,
            )
        ctx = _WalkContext(
            builder=builder,
            actor=_ActorKey(
                dev_id,
                mesh_idx,
                tuple(
                    g_coords[ax]
                    for ax, sem in enumerate(semantics)
                    if sem == "parallel"
                ),
            ),
            device_id=dev_id,
            device_coords=dev_coords,
            mesh_ctx=mesh_ctx,
            grid_coords=g_coords,
            grid_sizes=grid,
            scope_path=f"m{mesh_idx}" if len(meshes) > 1 else "",
            touched_refs=set(),
            written={},
            extra_axis_env=dict(zip(axis_names, g_coords)),
            core_axis_names=axis_names,
            all_core_axis_sizes=all_core_axis_sizes,
            dimension_semantics=semantics,
            is_sparse_core=is_sparse_core,
            unaliased_output_buffers=unaliased_out,
        )
        _walk(kernel_jaxpr, ctx, env)
        _record_step_written_buffers(ctx)

  if shared_builder is None:
    return []
  _check_output_coverage(shared_builder, all_mpmd_operands, mesh_ctx)
  return [shared_builder.build()]


def _collect_models(
    jaxpr: Any,
    mesh_ctx: _MeshContext,
    out_models: MutableSequence[KernelVerificationModel],
    *,
    max_unrolled_iterations: int,
    tpu_version: findings.TpuVersion = fx_verifier.DEFAULT_TPU_VERSION,
    concrete_arrays: Mapping[Any, _ConcreteArray] | None = None,
) -> None:
  """Locates `shard_map`, `pallas_call`, and `mpmd_map` equations."""
  outer_empty_outputs = _find_outer_empty_ref_outputs(jaxpr)
  for eqn in jaxpr.eqns:
    name = eqn.primitive.name
    if name == "pallas_call":
      out_models.append(
          _extract_pallas_call_model(
              eqn,
              mesh_ctx,
              max_unrolled_iterations=max_unrolled_iterations,
              tpu_version=tpu_version,
              concrete_arrays=concrete_arrays,
          )
      )
      continue
    if name == "mpmd_map":
      out_models.extend(
          _extract_mpmd_map_models(
              eqn,
              mesh_ctx,
              max_unrolled_iterations=max_unrolled_iterations,
              tpu_version=tpu_version,
              outer_output_vars=outer_empty_outputs,
          )
      )
      continue
    sub_ctx = mesh_ctx
    in_specs: Any = None
    if name == "shard_map":
      mesh = eqn.params["mesh"]
      sub_ctx = _MeshContext(
          axis_names=tuple(mesh.axis_names),
          mesh_shape=tuple(int(s) for s in dict(mesh.shape).values()),
      )
      in_specs = eqn.params.get("in_specs")
    can_forward_invars = name in _CALL_PRIMITIVES or name == "shard_map"
    for sub in _sub_jaxprs(eqn):
      sub_arrays: dict[Any, _ConcreteArray] = {}
      for param_val in eqn.params.values():
        candidates = (
            param_val if isinstance(param_val, (tuple, list)) else (param_val,)
        )
        for cand in candidates:
          if hasattr(cand, "jaxpr") and hasattr(cand, "consts"):
            if cand.jaxpr is sub:
              for cv, cval in zip(sub.constvars, cand.consts):
                arr = _extract_small_int_cpu_array(cval)
                if arr is not None:
                  sub_arrays[cv] = _ConcreteArray(arr, is_runtime=False)
      if (
          concrete_arrays
          and can_forward_invars
          and len(eqn.invars) == len(sub.invars)
      ):
        for idx, (outer_v, inner_v) in enumerate(zip(eqn.invars, sub.invars)):
          if name == "shard_map":
            spec = (
                in_specs[idx]
                if in_specs is not None and idx < len(in_specs)
                else None
            )
            if spec is not None and any(s is not None for s in spec):
              continue
          if _is_jax_var(outer_v):
            entry = concrete_arrays.get(outer_v)
          else:
            lit_arr = _extract_small_int_cpu_array(
                getattr(outer_v, "val", None)
            )
            entry = (
                _ConcreteArray(lit_arr, is_runtime=False)
                if lit_arr is not None
                else None
            )
          if entry is not None:
            sub_arrays[inner_v] = entry
      _collect_models(
          sub,
          sub_ctx,
          out_models,
          max_unrolled_iterations=max_unrolled_iterations,
          tpu_version=tpu_version,
          concrete_arrays=sub_arrays or None,
      )


@dataclasses.dataclass(frozen=True)
class _ConcreteArray:
  """Small CPU integer/bool array and whether it came from runtime args."""

  values: np.ndarray
  is_runtime: bool


def _extract_small_int_cpu_array(
    val: Any,
    *,
    concrete_arg_ids: AbstractSet[int] | None = None,
) -> np.ndarray | None:
  """Returns a NumPy copy of a small CPU integer/bool input, or None."""
  if isinstance(val, torch.Tensor):
    if (
        fx_verifier.is_fake_tensor(val)
        or val.device.type != "cpu"
        or val.dtype not in fx_verifier.INT_OR_BOOL_TORCH_DTYPES
        or int(val.numel()) > 4096
        or (concrete_arg_ids is not None and id(val) not in concrete_arg_ids)
    ):
      return None
    return val.detach().cpu().numpy().copy()
  if isinstance(val, np.ndarray):
    if val.size > 4096:
      return None
    try:
      is_int_or_bool = (
          np.issubdtype(val.dtype, np.integer)
          or np.issubdtype(val.dtype, np.bool_)
          or jnp.issubdtype(val.dtype, jnp.integer)
          or jnp.issubdtype(val.dtype, jnp.bool_)
      )
    except TypeError:
      return None
    if not is_int_or_bool:
      return None
    if np.issubdtype(val.dtype, np.integer) or np.issubdtype(
        val.dtype, np.bool_
    ):
      return val.copy()
    return val.astype(np.int32)
  if isinstance(val, jax.Array) and not isinstance(val, jax.core.Tracer):
    if int(val.size) > 4096:
      return None
    try:
      is_int_or_bool = jnp.issubdtype(val.dtype, jnp.integer) or jnp.issubdtype(
          val.dtype, jnp.bool_
      )
    except TypeError:
      return None
    if not is_int_or_bool:
      return None
    try:
      arr = np.asarray(val).copy()
    except TypeError:
      return None
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(
        arr.dtype, np.bool_
    ):
      return arr
    return arr.astype(np.int32)
  return None


def _append_placeholder_slot(
    val: Any,
    dyn_placeholders: MutableSequence[Any],
    *,
    is_static: bool = False,
    mesh: Any = None,
    partition_spec: Any = None,
    dyn_concrete_arrays: MutableSequence[np.ndarray | None] | None = None,
    concrete_arg_ids: AbstractSet[int] | None = None,
) -> tuple[tuple[bool, Any], bool]:
  """Appends a placeholder or static slot and returns `(slot, is_sharded)`."""
  if is_static or not isinstance(
      val, (torch.Tensor, jax.Array, jax.ShapeDtypeStruct)
  ):
    return (True, val), False
  slot = (False, len(dyn_placeholders))
  is_replicated = partition_spec is None or all(
      s is None for s in partition_spec
  )
  if dyn_concrete_arrays is not None:
    dyn_concrete_arrays.append(
        _extract_small_int_cpu_array(val, concrete_arg_ids=concrete_arg_ids)
        if is_replicated
        else None
    )
  if isinstance(val, torch.Tensor):
    dyn_placeholders.append(
        pallas_lib.jax_placeholder(
            val, mesh=mesh, partition_spec=partition_spec
        )
    )
    return slot, False
  sharding = getattr(val, "sharding", None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    dyn_placeholders.append(
        jax.ShapeDtypeStruct(val.shape, val.dtype, sharding=sharding)
    )
    return slot, True
  dyn_placeholders.append(jax.ShapeDtypeStruct(val.shape, val.dtype))
  return slot, False


def _bind_closed_jaxpr_concrete_arrays(
    closed: Any,
    dyn_concrete_arrays: Sequence[np.ndarray | None],
) -> dict[Any, _ConcreteArray]:
  """Maps `closed.jaxpr` invars and constvars to known small int/bool arrays."""
  concrete: dict[Any, _ConcreteArray] = {}
  for invar, arr in zip(closed.jaxpr.invars, dyn_concrete_arrays):
    if arr is not None:
      concrete[invar] = _ConcreteArray(arr, is_runtime=True)
  for constvar, cval in zip(closed.jaxpr.constvars, closed.consts):
    arr = _extract_small_int_cpu_array(cval)
    if arr is not None:
      concrete[constvar] = _ConcreteArray(arr, is_runtime=False)
  return concrete


def _trace_op_with_concrete_arrays(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any],
    *,
    tpu_version: findings.TpuVersion = fx_verifier.DEFAULT_TPU_VERSION,
    concrete_arg_ids: AbstractSet[int] | None = None,
) -> tuple[Any, dict[Any, _ConcreteArray]]:
  """Traces `op_or_fn` to `(ClosedJaxpr, concrete_arrays)`."""
  jax_callable = fx_verifier.extract_jax_callable(op_or_fn)
  if jax_callable is not None:
    # Tracing needs no devices, so trace with the op's own (possibly
    # compile-only) mesh like TorchTPU does when it lowers the op. Unlike
    # building a CPU mesh, this does not initialize any JAX backend, which on
    # a TPU host would try to open the TPU.
    raw_fn, active_mesh = fx_verifier.unwrap_raw_jax_fn(jax_callable)
    # pylint: disable=protected-access
    norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
        example_args, dict(example_kwargs)
    )
    # pylint: enable=protected-access
    in_specs = jax_callable.input_partition_specs
    dyn_placeholders: list[Any] = []
    dyn_concrete_arrays: list[np.ndarray | None] = []
    arg_slots: list[tuple[bool, Any]] = []
    for idx, val in enumerate(norm_args):
      spec = (
          in_specs[idx]
          if in_specs is not None and idx < len(in_specs)
          else None
      )
      slot, _ = _append_placeholder_slot(
          val,
          dyn_placeholders,
          is_static=fx_verifier.is_static_positional_arg(jax_callable, idx),
          mesh=active_mesh,
          partition_spec=spec,
          dyn_concrete_arrays=dyn_concrete_arrays,
          concrete_arg_ids=concrete_arg_ids,
      )
      arg_slots.append(slot)

    kw_slots: dict[str, tuple[bool, Any]] = {}
    for k, val in norm_kwargs.items():
      slot, _ = _append_placeholder_slot(
          val,
          dyn_placeholders,
          mesh=active_mesh,
          dyn_concrete_arrays=dyn_concrete_arrays,
          concrete_arg_ids=concrete_arg_ids,
      )
      kw_slots[k] = slot

    def _make_target(fn_to_call):
      def _trace_target(*dyn_args):
        rebuilt_args = [
            slot_val if is_static else dyn_args[slot_val]
            for is_static, slot_val in arg_slots
        ]
        rebuilt_kwargs = {
            k: slot_val if is_static else dyn_args[slot_val]
            for k, (is_static, slot_val) in kw_slots.items()
        }
        return fn_to_call(*rebuilt_args, **rebuilt_kwargs)

      return _trace_target

    try:
      closed = jax.make_jaxpr(_make_target(raw_fn))(*dyn_placeholders)
    except ValueError as exc:
      if "Unsupported TPU device kind" not in str(exc):
        raise
      if active_mesh is None:
        with fx_verifier.override_abstract_mesh(
            fx_verifier.abstract_tpu_mesh(tpu_version)
        ):
          closed = jax.make_jaxpr(_make_target(raw_fn))(*dyn_placeholders)
      elif (
          isinstance(active_mesh, jax.sharding.AbstractMesh)
          and active_mesh.abstract_device is None
      ):
        tpu_mesh = active_mesh.update(
            abstract_device=fx_verifier.abstract_tpu_device(tpu_version)
        )
        rebound_fn = fx_verifier.rebind_function_meshes(
            raw_fn, tpu_mesh, clone_closures=True
        )
        rebound_phs = fx_verifier.replace_mock_meshes_in_pytree(
            dyn_placeholders, tpu_mesh
        )
        with fx_verifier.override_abstract_mesh(tpu_mesh):
          closed = jax.make_jaxpr(_make_target(rebound_fn))(*rebound_phs)
      else:
        raise
    return closed, _bind_closed_jaxpr_concrete_arrays(
        closed, dyn_concrete_arrays
    )

  # Raw JAX function path.
  preserve_outer_jit = fx_verifier.has_jit_named_sharding(op_or_fn)
  raw_fn = (
      op_or_fn
      if preserve_outer_jit
      else fx_verifier.unwrap_raw_jax_fn(op_or_fn)[0]
  )
  dyn_phs: list[Any] = []
  raw_concrete_arrays: list[np.ndarray | None] = []
  pos_map: list[tuple[bool, Any]] = []
  has_named_sharding = preserve_outer_jit
  for val in example_args:
    slot, named = _append_placeholder_slot(
        val,
        dyn_phs,
        dyn_concrete_arrays=raw_concrete_arrays,
        concrete_arg_ids=concrete_arg_ids,
    )
    pos_map.append(slot)
    has_named_sharding = has_named_sharding or named

  kw_map: dict[str, tuple[bool, Any]] = {}
  for k, val in example_kwargs.items():
    slot, named = _append_placeholder_slot(
        val,
        dyn_phs,
        dyn_concrete_arrays=raw_concrete_arrays,
        concrete_arg_ids=concrete_arg_ids,
    )
    kw_map[k] = slot
    has_named_sharding = has_named_sharding or named

  trace_ctx: Any = contextlib.nullcontext()
  if not has_named_sharding:
    fn_mesh = fx_verifier.find_function_mesh(raw_fn)
    if isinstance(fn_mesh, jax.sharding.AbstractMesh) and any(
        at == jax.sharding.AxisType.Explicit
        for at in getattr(fn_mesh, "axis_types", ())
    ):
      auto_mesh = fn_mesh.update(
          axis_types=(jax.sharding.AxisType.Auto,) * len(fn_mesh.axis_names)
      )
      raw_fn = fx_verifier.rebind_function_meshes(
          raw_fn, auto_mesh, clone_closures=True
      )
      if not jax.sharding.get_abstract_mesh().empty:
        trace_ctx = fx_verifier.override_abstract_mesh(auto_mesh)

  def _raw_target(*dyn_args):
    args = [v if is_st else dyn_args[v] for is_st, v in pos_map]
    kwargs = {
        k: v if is_st else dyn_args[v] for k, (is_st, v) in kw_map.items()
    }
    return raw_fn(*args, **kwargs)

  try:
    with trace_ctx:
      closed = jax.make_jaxpr(_raw_target)(*dyn_phs)
  except ValueError as exc:
    if "Unsupported TPU device kind" not in str(exc):
      raise
    with fx_verifier.override_abstract_mesh(
        fx_verifier.abstract_tpu_mesh(tpu_version)
    ):
      closed = jax.make_jaxpr(_raw_target)(*dyn_phs)
  return closed, _bind_closed_jaxpr_concrete_arrays(closed, raw_concrete_arrays)


def display_name(op_or_fn: Any) -> str:
  """Returns the name used for `op_or_fn` in findings."""
  return fx_verifier.get_op_display_name(op_or_fn)


def static_model_findings(
    model: KernelVerificationModel,
) -> list[findings.VerificationFinding]:
  """Converts static issues and incompleteness notes into findings."""
  op_name = model.kernel_name
  detected = [
      findings.VerificationFinding(
          category=issue.category,
          severity=issue.severity,
          phase=findings.Phase.MODEL_CHECK,
          op_name=op_name,
          message=issue.message,
          source_location=issue.source_location,
      )
      for issue in model.static_issues
  ]
  if model.incomplete_reasons:
    detected.append(
        findings.VerificationFinding(
            category=_Category.VERIFICATION_INCOMPLETE,
            severity=_Severity.INFO,
            phase=findings.Phase.MODEL_CHECK,
            op_name=op_name,
            message=(
                "the model is bounded or approximate, so findings may be"
                " missed: "
                + "; ".join(model.incomplete_reasons)
            ),
        )
    )
  return detected


def _classify_trace_exception(
    exc: Exception,
) -> tuple[_Category, _Severity]:
  """Maps trace-time bounds and device-id errors to finding categories."""
  lower = str(exc).lower()
  if (
      isinstance(exc, (IndexError, ValueError))
      and "for grid" not in lower
      and any(
          pat in lower
          for pat in (
              "out of bound slice",
              "out of bound indexer",
              "indexer out of bounds",
              "index out of bounds",
              "out of bounds",
          )
      )
  ):
    return _Category.OUT_OF_BOUNDS_ACCESS, _Severity.ERROR
  if isinstance(exc, ValueError) and any(
      pat in lower
      for pat in (
          "unrecognized axes in device_id",
          "duplicate axis key in device_id",
          "semaphore_signal with device_id",
          "`device_id`s must be an int32 value",
          "`device_id_type` must be mesh",
          "number of device ids must match",
      )
  ):
    return _Category.INVALID_DEVICE_ID, _Severity.ERROR
  return _Category.VERIFICATION_INCOMPLETE, _Severity.WARNING


def is_trace_incomplete_finding(finding: findings.VerificationFinding) -> bool:
  """Returns whether `finding` is an unclassified trace-time failure warning."""
  return (
      finding.category == _Category.VERIFICATION_INCOMPLETE
      and finding.message.startswith(TRACE_INCOMPLETE_PREFIX)
  )


def is_trace_validation_error_finding(
    finding: findings.VerificationFinding,
) -> bool:
  """Returns whether `finding` is a classified Pallas trace-time error."""
  return finding.severity == _Severity.ERROR and finding.message.startswith(
      TRACE_VALIDATION_ERROR_PREFIX
  )


def is_trace_failure_report(report: findings.VerificationReport) -> bool:
  """Returns whether `report` came from a trace failure before extraction."""
  return any(
      is_trace_incomplete_finding(f) or is_trace_validation_error_finding(f)
      for f in report.findings
  )


def check_kernel_static_models(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    max_unrolled_iterations: int = DEFAULT_MAX_UNROLLED_ITERATIONS,
    tpu_version: findings.TpuVersion | None = None,
    concrete_arg_ids: AbstractSet[int] | None = None,
) -> tuple[
    findings.VerificationReport,
    tuple[KernelVerificationModel, ...] | None,
]:
  """Traces `op_or_fn`, extracts kernel models, and checks static invariants.

  Args:
    op_or_fn: `@jax_op` handle (`CustomOpDef` / `JaxCallable`) or raw JAX
      callable containing `pl.pallas_call`, `pl.core_map`, or `pl.kernel`.
    example_args: Positional example arguments (`torch.Tensor`, `jax.Array`,
      `jax.ShapeDtypeStruct`, or Python scalars/pytrees) supplying shapes,
      dtypes, shardings, and optional small CPU integer/boolean index inputs.
    example_kwargs: Optional keyword example arguments.
    max_unrolled_iterations: Maximum number of grid points, loop iterations, and
      scan steps unrolled for concurrency event recording.
    tpu_version: Target TPU generation used for hardware and sublane rules; if
      `None`, inferred from `op_or_fn` or `example_args` when possible.
    concrete_arg_ids: Optional allowlist of `id(tensor)` values for CPU integer
      or boolean `torch.Tensor` inputs whose concrete contents are known (used
      by FX program verification to exclude placeholder tensors materialized
      from `FakeTensor` metadata); when `None`, all small CPU integer/boolean
      inputs are eligible.

  Returns:
    `(report, models)` where `report` contains `Phase.MODEL_CHECK` static
    findings and `models` is:
      - a non-empty tuple of extracted `KernelVerificationModel`s when at least
        one `pallas_call` or `mpmd_map` equation was found;
      - `()` when tracing succeeded without any `pallas_call` or `mpmd_map`; or
      - `None` when tracing or model extraction failed.
  """
  effective_version = fx_verifier.infer_tpu_version(
      op_or_fn, tpu_version, example_args, example_kwargs
  )
  try:
    closed, concrete_arrays = _trace_op_with_concrete_arrays(
        op_or_fn,
        example_args,
        dict(example_kwargs or {}),
        tpu_version=effective_version,
        concrete_arg_ids=concrete_arg_ids,
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    cat, sev = _classify_trace_exception(exc)
    prefix = (
        TRACE_INCOMPLETE_PREFIX
        if cat == _Category.VERIFICATION_INCOMPLETE
        else TRACE_VALIDATION_ERROR_PREFIX
    )
    report = findings.report_of(
        findings.Phase.MODEL_CHECK,
        [
            findings.VerificationFinding(
                category=cat,
                severity=sev,
                phase=findings.Phase.MODEL_CHECK,
                op_name=display_name(op_or_fn),
                message=f"{prefix} ({type(exc).__name__}: {exc})",
                source_location=_summarize_exception_traceback(exc),
            )
        ],
    )
    return report, None
  name = display_name(op_or_fn)
  models_list: list[KernelVerificationModel] = []
  try:
    _collect_models(
        closed.jaxpr,
        _MeshContext(),
        models_list,
        max_unrolled_iterations=max_unrolled_iterations,
        tpu_version=effective_version,
        concrete_arrays=concrete_arrays or None,
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    report = findings.report_of(
        findings.Phase.MODEL_CHECK,
        [
            findings.VerificationFinding(
                category=_Category.VERIFICATION_INCOMPLETE,
                severity=_Severity.WARNING,
                phase=findings.Phase.MODEL_CHECK,
                op_name=name,
                message=(
                    "could not extract static kernel model"
                    f" ({type(exc).__name__}: {exc})"
                ),
                source_location=_summarize_exception_traceback(exc),
            )
        ],
    )
    return report, None
  models = tuple(
      dataclasses.replace(
          model,
          kernel_name=name
          if len(models_list) == 1
          else f"{name} ({model.kernel_name} #{idx})",
      )
      for idx, model in enumerate(models_list)
  )
  if not models:
    report = findings.report_of(
        findings.Phase.MODEL_CHECK,
        [
            findings.VerificationFinding(
                category=_Category.VERIFICATION_INCOMPLETE,
                severity=_Severity.INFO,
                phase=findings.Phase.MODEL_CHECK,
                op_name=name,
                message=(
                    "no pallas_call or core_map was found, so there was nothing"
                    " to model check"
                ),
            )
        ],
    )
    return report, ()
  detected: list[findings.VerificationFinding] = []
  for model in models:
    detected.extend(static_model_findings(model))
  return findings.report_of(findings.Phase.MODEL_CHECK, detected), models
