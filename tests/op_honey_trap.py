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

"""Honey trap fault injection framework for TorchTPU operator test auditing.

This module provides HoneyTrapDispatchMode, a TorchDispatchMode context that
intercepts ATen operator dispatches targeting TPU devices. It operates in two
modes:
1. 'record': Captures schema-bound execution signatures of all executed TPU ops.
2. 'inject': Injects deterministic faults (condition raise, zero sentinel,
   scale offset, tight drift, broken writeback, skip out resize) when a call
   matches one of the target signatures.

Environment Variables:
- TT_HONEY_TRAP_MODE: Controls mode ('record' or 'inject'). If unset or empty,
  the honey trap is disabled.
- TT_HONEY_TRAP_SPEC: JSON string specifying fault rules and target signatures
  for 'inject' mode.
- TT_HONEY_TRAP_OP_HINT: Comma-separated operator name hints to prevent
  filtering out relevant ops (e.g. 'add,sum').
- TT_HONEY_TRAP_RECORD_FILE: Path to file where recorded signatures are written.
- TT_HONEY_TRAP_DEVICE: Target device to intercept ('tpu', 'cpu', etc.).
  Defaults to 'tpu'.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import copy
import json
import os
import sys
import threading
from typing import Any, Final
import torch
from torch._subclasses import fake_tensor
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

_THREAD_STATE = threading.local()

_ALWAYS_BOILERPLATE_OPS: Final[frozenset[str]] = frozenset({
    "aten::_to_copy",
    "aten::copy_",
    "aten::empty",
    "aten::empty_strided",
    "aten::empty_like",
    "aten::clone",
    "aten::detach",
    "aten::_local_scalar_dense",
    "aten::item",
    "aten::rand",
    "aten::randn",
    "aten::randint",
    "aten::lift_fresh",
    "aten::lift_fresh_copy",
    "aten::scalar_tensor",
})

_SETUP_VIEW_SEED_OPS: Final[frozenset[str]] = frozenset({
    "aten::zeros",
    "aten::zero_",
    "aten::zeros_like",
    "aten::ones",
    "aten::ones_like",
    "aten::full",
    "aten::full_like",
    "aten::linspace",
    "aten::arange",
    "aten::view",
    "aten::_unsafe_view",
    "aten::reshape",
    "aten::expand",
    "aten::as_strided",
    "aten::t",
    "aten::transpose",
    "aten::permute",
    "aten::contiguous",
    "aten::sum",
})


def _is_paused() -> bool:
  """Returns True if the current thread has paused honey trap dispatch."""
  return getattr(_THREAD_STATE, "paused_depth", 0) > 0


def _is_in_dispatch() -> bool:
  """Returns True if the current thread is currently in dispatch."""
  return getattr(_THREAD_STATE, "in_dispatch", False)


@contextlib.contextmanager
def pause_honey_trap() -> Iterator[None]:
  """Context manager to pause honey trap interception thread-locally."""
  _THREAD_STATE.paused_depth = getattr(_THREAD_STATE, "paused_depth", 0) + 1
  try:
    yield
  finally:
    _THREAD_STATE.paused_depth -= 1


def _is_matching_device(
    device: torch.device | None, target_device: str
) -> bool:
  """Returns True if the device matches the target interception device."""
  if device is None:
    return False
  if target_device == "cpu":
    return device.type == "cpu"
  return device.type in ("tpu", "privateuseone")


def _has_matching_tensor(tree: Any, target_device: str) -> bool:
  """Returns True if any non-fake tensor in tree resides on target device."""
  for leaf in _pytree.tree_leaves(tree):
    if fake_tensor.is_fake(leaf):
      continue
    if isinstance(leaf, torch.Tensor) and _is_matching_device(
        leaf.device, target_device
    ):
      return True
  return False


def _targets_device(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    result: Any,
    target_device: str,
) -> bool:
  """Returns True if the op invocation targets the selected device."""
  if _has_matching_tensor(args, target_device) or _has_matching_tensor(
      kwargs, target_device
  ):
    return True
  dev = kwargs.get("device")
  if dev is not None:
    if isinstance(dev, str):
      try:
        parsed_dev = torch.device(dev)
        if _is_matching_device(parsed_dev, target_device):
          return True
      except RuntimeError:
        if target_device == "cpu" and dev == "cpu":
          return True
        if target_device != "cpu" and ("tpu" in dev or "privateuseone" in dev):
          return True
    elif isinstance(dev, torch.device) and _is_matching_device(
        dev, target_device
    ):
      return True
  if _has_matching_tensor(result, target_device):
    return True
  return False


def _should_ignore_op(func: Any, op_hint: str | None) -> bool:
  """Checks if an op is in boilerplate/setup sets without matching op_hint."""
  schema = getattr(func, "_schema", None)
  if schema is None:
    return False
  op_name = schema.name
  str_func = str(func)
  if op_hint:
    hints = [h.strip().lower() for h in op_hint.split(",") if h.strip()]
    op_name_lower = op_name.lower()
    str_func_lower = str_func.lower()
    for h in hints:
      if h in op_name_lower or h in str_func_lower:
        return False
  if op_name in _ALWAYS_BOILERPLATE_OPS or op_name in _SETUP_VIEW_SEED_OPS:
    return True
  return False


def _bind_arguments(
    schema: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[tuple[Any, Any]]:
  """Binds positional and keyword args to schema arguments in order."""
  bound: list[tuple[Any, Any]] = []
  pos_idx = 0
  for arg in schema.arguments:
    if arg.name in kwargs:
      bound.append((arg, kwargs[arg.name]))
    elif pos_idx < len(args):
      bound.append((arg, args[pos_idx]))
      pos_idx += 1
    elif arg.default_value is not None:
      bound.append((arg, arg.default_value))
    else:
      bound.append((arg, None))
  return bound


def _default_drift_for_dtype(dtype: torch.dtype) -> float:
  """Returns default numerical drift tolerance for the given dtype."""
  if dtype in (torch.float64, torch.complex128):
    return 1e-4
  if dtype in (torch.float32, torch.complex64):
    return 1e-3
  if dtype == torch.float16:
    return 1e-2
  if dtype == torch.bfloat16:
    return 5e-2
  return 1e-3


def _corrupt_zero_or_sentinel(t: torch.Tensor) -> torch.Tensor:
  """Applies ZERO_OR_SENTINEL corruption to a single tensor."""
  with torch.no_grad():
    if t.dtype == torch.bool:
      if t.numel() > 0 and t.any().item():
        return torch.zeros_like(t)
      return ~t
    if t.is_floating_point() or t.is_complex():
      finite = torch.isfinite(t)
      if (
          t.numel() > 0
          and finite.any().item()
          and t.abs()[finite].max().item() >= 1e-3
      ):
        return torch.zeros_like(t)
      return torch.full_like(t, -999.0)
    is_signed = t.is_signed() if hasattr(t, "is_signed") else True
    if not is_signed:
      has_non_zero = t.numel() > 0 and (t.to(torch.int64) >= 1).any().item()
    else:
      has_non_zero = t.numel() > 0 and t.abs().max().item() >= 1
    if has_non_zero:
      return torch.zeros_like(t)
    return torch.full_like(t, 99)


def _corrupt_scale_or_offset(t: torch.Tensor) -> torch.Tensor:
  """Applies SCALE_OR_OFFSET corruption to a single tensor."""
  with torch.no_grad():
    if t.dtype == torch.bool:
      return ~t
    if t.is_floating_point() or t.is_complex():
      finite = torch.isfinite(t)
      has_finite = finite.any().item() if t.numel() > 0 else False
      if has_finite and (
          t.abs()[finite].max().item() > 1e-4 if finite.any() else False
      ):
        val = t * 0.5
      else:
        val = t + 2.0
      return torch.where(finite, val, torch.zeros_like(t))
    return (t.to(torch.int64) + 1).to(t.dtype)


def _corrupt_tight_drift(t: torch.Tensor, drift: float | None) -> torch.Tensor:
  """Applies TIGHT_DRIFT corruption to a single tensor."""
  with torch.no_grad():
    if t.dtype == torch.bool:
      t_flat = t.clone().reshape(-1)
      if t_flat.numel() > 0:
        t_flat[0] = ~t_flat[0]
      return t_flat.reshape(t.shape)
    if t.is_floating_point() or t.is_complex():
      delta = drift if drift is not None else _default_drift_for_dtype(t.dtype)
      val = t + delta
      finite = torch.isfinite(t)
      return torch.where(finite, val, torch.zeros_like(t))
    return (t.to(torch.int64) + 2).to(t.dtype)


def signature_matches(
    target_sig: Mapping[str, Any], call_sig: Mapping[str, Any]
) -> bool:
  """Checks whether a recorded call signature matches a target signature."""
  if target_sig.get("op") != call_sig.get("op"):
    return False
  if target_sig.get("read_dtypes") is not None:
    if target_sig["read_dtypes"] != call_sig.get("read_dtypes"):
      return False
  if target_sig.get("out_dtypes") is not None:
    if target_sig["out_dtypes"] != call_sig.get("out_dtypes"):
      return False
  if target_sig.get("dtype_kwargs") is not None:
    if target_sig["dtype_kwargs"] != call_sig.get("dtype_kwargs"):
      return False
  if target_sig.get("write_args") is not None:
    target_writes = target_sig["write_args"]
    call_writes = call_sig.get("write_args", [])
    if len(target_writes) != len(call_writes):
      return False
    for tw, cw in zip(target_writes, call_writes):
      if tw.get("name") != cw.get("name"):
        return False
      if tw.get("dtypes") is not None and tw["dtypes"] != cw.get("dtypes"):
        return False
      if "nonempty_resized" in tw and tw["nonempty_resized"] != cw.get(
          "nonempty_resized"
      ):
        return False
  return True


def _capture_writeable_snapshots(
    bound_args: Sequence[tuple[Any, Any]],
) -> list[dict[str, Any]]:
  """Captures pre-execution shape and metadata of writeable tensor arguments."""
  writeable_snapshots: list[dict[str, Any]] = []
  for arg_meta, val in bound_args:
    if arg_meta.alias_info and arg_meta.alias_info.is_write:
      if isinstance(val, torch.Tensor):
        writeable_snapshots.append({
            "name": arg_meta.name,
            "tensor": val,
            "pre_numel": val.numel(),
            "pre_shape": tuple(val.shape),
            "dtype": str(val.dtype),
        })
      elif isinstance(val, (list, tuple)):
        for idx, item in enumerate(val):
          if isinstance(item, torch.Tensor):
            writeable_snapshots.append({
                "name": f"{arg_meta.name}[{idx}]",
                "tensor": item,
                "pre_numel": item.numel(),
                "pre_shape": tuple(item.shape),
                "dtype": str(item.dtype),
            })
  return writeable_snapshots


def _build_call_signature(
    func: Any,
    bound_args: Sequence[tuple[Any, Any]],
    writeable_snapshots: Sequence[dict[str, Any]],
    result: Any,
) -> dict[str, Any]:
  """Constructs schema-bound execution signature for an operator call."""
  write_args = []
  for snap in writeable_snapshots:
    post_shape = tuple(snap["tensor"].shape)
    nonempty_resized = snap["pre_numel"] > 0 and snap["pre_shape"] != post_shape
    write_args.append({
        "name": snap["name"],
        "dtypes": [snap["dtype"]],
        "nonempty_resized": nonempty_resized,
    })

  read_dtypes = []
  for arg_meta, val in bound_args:
    if not (arg_meta.alias_info and arg_meta.alias_info.is_write):
      for leaf in _pytree.tree_leaves(val):
        if isinstance(leaf, torch.Tensor):
          read_dtypes.append(str(leaf.dtype))

  out_dtypes = [
      str(leaf.dtype)
      for leaf in _pytree.tree_leaves(result)
      if isinstance(leaf, torch.Tensor)
  ]

  dtype_kwargs = {}
  for arg_meta, val in bound_args:
    if "ScalarType" in str(arg_meta.type):
      dtype_kwargs[arg_meta.name] = str(val)

  return {
      "op": str(func),
      "read_dtypes": read_dtypes,
      "write_args": write_args,
      "out_dtypes": out_dtypes,
      "dtype_kwargs": dtype_kwargs,
  }


def _copy_corrupted_to_snapshots(
    corrupted: torch.Tensor,
    writeable_snapshots: Sequence[dict[str, Any]],
    target_tensor: torch.Tensor | None = None,
    used_indices: set[int] | None = None,
) -> None:
  """Propagates corrupted tensor values back into matched snapshots."""
  if used_indices is None:
    used_indices = set()
  for idx, snap in enumerate(writeable_snapshots):
    if idx in used_indices:
      continue
    if target_tensor is not None and snap["tensor"] is target_tensor:
      snap["tensor"].copy_(corrupted)
      used_indices.add(idx)
      return
  for idx, snap in enumerate(writeable_snapshots):
    if idx in used_indices:
      continue
    if (
        snap["tensor"].shape == corrupted.shape
        and snap["tensor"].dtype == corrupted.dtype
    ):
      snap["tensor"].copy_(corrupted)
      used_indices.add(idx)
      return


def _apply_value_corruption(
    result: Any,
    corrupt_fn: Callable[[torch.Tensor], torch.Tensor],
    writeable_snapshots: Sequence[dict[str, Any]],
    output_index: int | None = None,
) -> Any:
  """Applies value corruption to result and updates writeable snapshots."""
  if result is None:
    for snap in writeable_snapshots:
      snap["tensor"].copy_(corrupt_fn(snap["tensor"]))
    return None

  used_indices: set[int] = set()
  if isinstance(result, torch.Tensor):
    corrupted = corrupt_fn(result)
    _copy_corrupted_to_snapshots(
        corrupted, writeable_snapshots, result, used_indices
    )
    return corrupted

  if isinstance(result, (tuple, list)):
    mutated_list = list(result)
    if output_index is not None and 0 <= output_index < len(mutated_list):
      val = mutated_list[output_index]
      if isinstance(val, torch.Tensor):
        corrupted = corrupt_fn(val)
        _copy_corrupted_to_snapshots(
            corrupted, writeable_snapshots, val, used_indices
        )
        mutated_list[output_index] = corrupted
    else:
      has_float = any(
          isinstance(x, torch.Tensor)
          and (x.is_floating_point() or x.is_complex())
          for x in mutated_list
      )
      for idx, val in enumerate(mutated_list):
        if isinstance(val, torch.Tensor):
          if has_float and not (val.is_floating_point() or val.is_complex()):
            continue
          corrupted = corrupt_fn(val)
          _copy_corrupted_to_snapshots(
              corrupted, writeable_snapshots, val, used_indices
          )
          mutated_list[idx] = corrupted
    return tuple(mutated_list) if isinstance(result, tuple) else mutated_list

  return result


def _inject_fault(
    spec: Mapping[str, Any],
    call_sig: Mapping[str, Any],
    result: Any,
    writeable_snapshots: Sequence[dict[str, Any]],
) -> Any:
  """Injects configured fault into operator execution or snapshots."""
  rule = spec.get("rule", "CONDITION_RAISE")
  if rule == "CONDITION_RAISE":
    raise RuntimeError(
        f"HoneyTrap CONDITION_RAISE triggered on {call_sig['op']}"
    )

  if rule == "SKIP_OUT_RESIZE":
    for snap in writeable_snapshots:
      if (
          snap["pre_numel"] > 0
          and tuple(snap["tensor"].shape) != snap["pre_shape"]
      ):
        snap["tensor"].resize_(snap["pre_shape"])
    return result

  if rule == "BROKEN_WRITEBACK":
    if isinstance(result, torch.Tensor):
      clean_return = result.clone()
    elif isinstance(result, tuple):
      clean_return = tuple(
          x.clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x)
          for x in result
      )
    else:
      clean_return = copy.deepcopy(result)

    for snap in writeable_snapshots:
      w_tensor = snap["tensor"]
      with torch.no_grad():
        if w_tensor.dtype == torch.bool:
          w_tensor.logical_not_()
        elif w_tensor.is_floating_point() or w_tensor.is_complex():
          if (
              w_tensor.numel() > 0
              and (w_tensor - (-999.0)).abs().min().item() < 1e-1
          ):
            w_tensor.fill_(999.0)
          else:
            w_tensor.fill_(-999.0)
        else:
          if w_tensor.numel() > 0 and (w_tensor == 99).any().item():
            w_tensor.fill_(77)
          else:
            w_tensor.fill_(99)
    return clean_return

  corrupt_fn: Callable[[torch.Tensor], torch.Tensor]
  if rule == "ZERO_OR_SENTINEL":
    corrupt_fn = _corrupt_zero_or_sentinel
  elif rule == "SCALE_OR_OFFSET":
    corrupt_fn = _corrupt_scale_or_offset
  elif rule == "TIGHT_DRIFT":
    drift_val = spec.get("drift")
    corrupt_fn = lambda t: _corrupt_tight_drift(t, drift_val)
  else:
    return result

  return _apply_value_corruption(
      result, corrupt_fn, writeable_snapshots, spec.get("output_index")
  )


class HoneyTrapDispatchMode(TorchDispatchMode):
  """TorchDispatchMode for recording and injecting deterministic faults."""

  def __init__(
      self,
      mode: str = "record",
      spec_json: str = "",
      op_hint: str = "",
      record_file: str = "",
      target_device: str = "tpu",
      test_id: str | None = None,
  ) -> None:
    """Initializes the HoneyTrapDispatchMode instance.

    Args:
      mode: Operating mode, either 'record' to log signatures or 'inject' to
        inject deterministic faults.
      spec_json: JSON string describing fault injection rules and targets.
      op_hint: Comma-separated hints for ops that should not be filtered.
      record_file: Path to file where recorded signatures should be saved.
      target_device: Target device name ('tpu', 'cpu', etc.).
      test_id: Optional test identifier for logs and signatures.

    Raises:
      ValueError: If spec_json is non-empty but contains invalid JSON.
    """
    super().__init__()
    self._mode: str = mode.strip().lower()
    self._spec: dict[str, Any] = {}
    if spec_json:
      try:
        self._spec = json.loads(spec_json)
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid spec_json: {e}") from e
    self._op_hint: str = op_hint.strip()
    self._record_file: str = record_file.strip()
    self._target_device: str = (
        os.environ.get("TT_HONEY_TRAP_DEVICE", target_device).strip().lower()
    )
    self._test_id: str | None = test_id
    self._recorded_signatures: list[dict[str, Any]] = []
    self._orig_assert_close: Callable[..., Any] | None = None

  def __enter__(self) -> "HoneyTrapDispatchMode":
    super().__enter__()
    self._orig_assert_close = torch.testing.assert_close

    def _safe_assert_close(*args: Any, **kwargs: Any) -> Any:
      with pause_honey_trap():
        assert self._orig_assert_close is not None
        return self._orig_assert_close(*args, **kwargs)

    torch.testing.assert_close = _safe_assert_close
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    try:
      if self._orig_assert_close is not None:
        torch.testing.assert_close = self._orig_assert_close
    finally:
      self._orig_assert_close = None
      super().__exit__(exc_type, exc_val, exc_tb)

    if self._mode == "record":
      tag = (
          f"TT_HONEY_TRAP_RECORD_JSON|{self._test_id}|: "
          if self._test_id
          else "TT_HONEY_TRAP_RECORD_JSON: "
      )
      output_line = f"{tag}{json.dumps(self.get_deduped_signatures())}"
      sys.stderr.write(output_line + "\n")
      sys.stderr.flush()
      if self._record_file:
        try:
          with open(self._record_file, "a") as f:
            for sig in self.get_deduped_signatures():
              f.write(json.dumps(sig) + "\n")
        except OSError as e:
          sys.stderr.write(
              f"Failed to write record file {self._record_file}: {e}\n"
          )

  def get_deduped_signatures(self) -> list[dict[str, Any]]:
    """Returns deduplicated list of recorded signatures."""
    seen = set()
    result = []
    for sig in self._recorded_signatures:
      key = json.dumps(sig, sort_keys=True)
      if key not in seen:
        seen.add(key)
        result.append(sig)
    return result

  def __torch_dispatch__(
      self,
      func: Any,
      types: Sequence[type[Any]],
      args: tuple[Any, ...] = (),
      kwargs: dict[str, Any] | None = None,
  ) -> Any:
    if kwargs is None:
      kwargs = {}

    if _is_in_dispatch() or _is_paused():
      return func(*args, **kwargs)

    schema = getattr(func, "_schema", None)
    if schema is None:
      return func(*args, **kwargs)

    target_sigs = self._spec.get("signatures", [])
    str_func = str(func)
    is_direct_target = self._mode == "inject" and any(
        str_func == ts.get("op") for ts in target_sigs
    )
    if not is_direct_target and _should_ignore_op(func, self._op_hint):
      return func(*args, **kwargs)

    _THREAD_STATE.in_dispatch = True
    try:
      bound_args = _bind_arguments(schema, args, kwargs)
      writeable_snapshots = _capture_writeable_snapshots(bound_args)
      result = func(*args, **kwargs)

      if not _targets_device(args, kwargs, result, self._target_device):
        return result

      call_sig = _build_call_signature(
          func, bound_args, writeable_snapshots, result
      )

      if self._mode == "record":
        self._recorded_signatures.append(call_sig)
        return result

      if self._mode == "inject":
        matched = any(signature_matches(ts, call_sig) for ts in target_sigs)
        if not matched:
          return result
        return _inject_fault(self._spec, call_sig, result, writeable_snapshots)

      return result
    finally:
      _THREAD_STATE.in_dispatch = False


def create_honey_trap(
    test_id: str | None = None,
) -> HoneyTrapDispatchMode | None:
  """Creates HoneyTrapDispatchMode if configured via TT_HONEY_TRAP_MODE.

  Args:
    test_id: Optional test identifier attached to recorded signatures or logs.

  Returns:
    HoneyTrapDispatchMode instance if TT_HONEY_TRAP_MODE is set to 'record'
    or 'inject', otherwise None.
  """
  mode = os.environ.get("TT_HONEY_TRAP_MODE", "").strip().lower()
  if not mode or mode not in ("record", "inject"):
    return None
  return HoneyTrapDispatchMode(
      mode=mode,
      spec_json=os.environ.get("TT_HONEY_TRAP_SPEC", ""),
      op_hint=os.environ.get("TT_HONEY_TRAP_OP_HINT", "").strip(),
      record_file=os.environ.get("TT_HONEY_TRAP_RECORD_FILE", "").strip(),
      test_id=test_id,
  )
