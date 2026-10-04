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

"""TorchTPU API lifecycle stage annotations and decorators.

This module provides decorators (@experimental, @stable, @deprecated) to mark
the lifecycle stage of TorchTPU Python APIs. These decorators attach stage
metadata (__tt_api_stage__, __tt_api_stage_reason__,
__tt_api_deprecated_version__) to functions, methods, or classes for automatic
inspection by scanner tools, and emit standard Python runtime warnings
(UserWarning, DeprecationWarning) on the first invocation per process.

Warnings are suppressed when the caller is within the torch or torch_tpu
internal framework packages.

The @internal_or_experimental decorator marks a TorchTPU Python API as either
experimental or internal, depending on a condition (e.g. the PyTorch version).
An internal API is treated like one with a `_` prefix in its path. It's meant to
be called only by the PyTorch framework (torch and torch_tpu); calling it from
any other module emits a one-time warning per process that points to the
preferred alternative, if any.
"""

import dataclasses
import enum
import functools
import sys
from typing import Any, Callable, Mapping, Optional, TypeVar

import torch
from torch_tpu._internal import warnings
from torch_tpu._internal.utils import tt_warnings

_T = TypeVar("_T")

TT_API_STAGE = "__tt_api_stage__"
TT_API_STAGE_REASON = "__tt_api_stage_reason__"
TT_API_DEPRECATED_VERSION = "__tt_api_deprecated_version__"
TT_API_PREFER = "__tt_api_prefer__"
TT_API_STAGES_MAP = "__tt_api_stages__"
_FRAMEWORK_PACKAGES: tuple[str, ...] = ("torch", "torch_tpu")


def parse_version(version: str) -> tuple[int, ...]:
  """Parses leading numeric components from a version string into a tuple."""
  components = []
  for part in version.split("."):
    digits = ""
    for ch in part:
      if not ch.isdigit():
        break
      digits += ch
    if not digits:
      break
    components.append(int(digits))
  return tuple(components)


@functools.lru_cache(maxsize=1)
def get_torch_version() -> tuple[int, ...]:
  """Returns PyTorch runtime version parsed into an integer tuple."""
  return parse_version(torch.__version__)


def _is_framework_caller_frame(frame: Any) -> bool:
  """Checks if a stack frame belongs to torch or torch_tpu framework packages."""
  if frame is None:
    return False
  mod = frame.f_globals.get("__name__") or ""
  if not isinstance(mod, str):
    return False
  # Do not treat tests as framework callers so warnings can be tested and verified.
  if mod.startswith("torch_tpu.tests.") or mod == "torch_tpu.tests":
    return False
  return (
      mod in _FRAMEWORK_PACKAGES
      or mod.startswith("torch.")
      or mod.startswith("torch_tpu.")
  )


class Stage(str, enum.Enum):
  """Lifecycle stages for TorchTPU APIs."""

  INTERNAL = "Internal"
  STABLE = "Stable"
  EXPERIMENTAL = "Experimental"
  DEPRECATED = "Deprecated"

  @property
  def warning_category(
      self,
  ) -> Optional[warnings.WarningCategory]:
    """The TorchTPU warning category for this stage."""
    if self == Stage.INTERNAL:
      return warnings.INTERNAL_API
    if self == Stage.EXPERIMENTAL:
      return warnings.EXPERIMENTAL_API
    if self == Stage.DEPRECATED:
      return warnings.DEPRECATED_API
    return None

  def format_warning_message(
      self, name: str, reason: str, version: Optional[str] = None
  ) -> str:
    """Formats the warning message for this stage."""
    if self == Stage.INTERNAL:
      msg = (
          f"{name} is internal to TorchTPU and may change or be removed without"
          " notice."
      )
      return f"{msg} {reason}" if reason else msg
    if self == Stage.EXPERIMENTAL:
      return f"{name} is experimental and subject to change: {reason}"
    if self == Stage.DEPRECATED:
      return (
          f"{name} is deprecated as of TorchTPU {version} and will be"
          f" removed in a future release: {reason}"
      )
    return ""


@dataclasses.dataclass(frozen=True)
class ApiStageInfo:
  """Stage metadata for dynamic module/class attributes.

  Attributes:
    stage: The API readiness Stage (INTERNAL, STABLE, EXPERIMENTAL, DEPRECATED).
    reason: Human-readable rationale for the lifecycle stage.
    value: Optional runtime implementation value or object returned by
      resolve_module_attribute.
    version: Deprecation version string (e.g. "2.13"), used if stage is
      DEPRECATED.
  """

  stage: Stage
  reason: str
  value: Any = None
  version: Optional[str] = None


def _safe_setattr(target: Any, name: str, value: Any) -> None:
  """Safely sets an attribute on target, swallowing AttributeError and TypeError.

  Args:
    target: The target object to modify.
    name: The attribute name to set.
    value: The attribute value to assign.
  """
  try:
    setattr(target, name, value)  # pylint: disable=protected-access
  except (AttributeError, TypeError):
    pass


def _attach_stage_metadata(
    target: Any,
    stage: str,
    reason: str,
    version: Optional[str] = None,
    prefer: Optional[str] = None,
) -> None:
  """Attaches stage metadata attributes to a target.

  If the target has a `__func__` attribute (e.g. a bound method), the
  metadata attributes are also attached to the underlying function object.

  Args:
    target: The target object to modify.
    stage: The lifecycle stage of the target.
    reason: Explanation for the stage.
    version: The version string when the target was deprecated (e.g. "2.13").
      Only used for deprecated targets.
    prefer: The preferred alternative to the target, if any. Only used for
      internal targets.
  """
  targets = [target]
  # If target is a method (e.g. bound classmethod), also attach metadata to the
  # underlying function object so it is discoverable when inspecting __func__.
  if hasattr(target, "__func__"):
    targets.append(target.__func__)
  for t in targets:
    _safe_setattr(t, TT_API_STAGE, stage)
    _safe_setattr(t, TT_API_STAGE_REASON, reason)
    if version:
      _safe_setattr(t, TT_API_DEPRECATED_VERSION, version)
    if prefer:
      _safe_setattr(t, TT_API_PREFER, prefer)


def _create_warning_wrapper(
    fn_to_wrap: Callable[..., Any],
    display_target: Any,
    stage: Stage,
    reason: str,
    version: Optional[str] = None,
) -> Callable[..., Any]:
  """Wraps a callable to emit a one-time warning per process before execution.

  Warnings are suppressed when called from torch or torch_tpu internal framework
  code, preserving the one-time warning for subsequent external user calls.

  Args:
    fn_to_wrap: The callable to wrap.
    display_target: The target object to display in the warning message.
    stage: The lifecycle stage of the target.
    reason: Explanation for the stage.
    version: The version string when the target was deprecated (e.g. "2.13").
      Only used for deprecated targets.

  Returns:
    The wrapped callable.
  """
  _warned = False

  @functools.wraps(fn_to_wrap)  # type: ignore[arg-type]
  def wrapper(*args: Any, **kwargs: Any) -> Any:
    nonlocal _warned
    # TorchDynamo cannot trace `sys._getframe()`, so skip the caller check while
    # compiling to let callers inside `torch.compile` regions trace through
    # without graph breaks. The warning is emitted on the first eager call
    # instead.
    if not _warned and not torch.compiler.is_compiling():
      try:
        caller_frame = sys._getframe(1)
      except (AttributeError, ValueError):
        caller_frame = None

      if not _is_framework_caller_frame(caller_frame):
        name = getattr(display_target, "__name__", str(display_target))
        msg = stage.format_warning_message(name, reason, version)
        cat = stage.warning_category
        if cat is not None:
          _warned = tt_warnings.warn(msg, category=cat, stacklevel=2)
    return fn_to_wrap(*args, **kwargs)

  return wrapper


def _annotate_target(
    target: _T,
    stage: Stage,
    reason: str,
    version: Optional[str] = None,
    prefer: Optional[str] = None,
) -> _T:
  """Attaches stage metadata and warning wrapper to a target function or class.

  Behavior by target type:
  1. Functions / Callables: Returns a wrapper function that emits a one-time
     warning per process before execution, and attaches stage metadata to the
     wrapper.
  2. Non-Enum Classes: Wraps the class `__init__` method to emit a one-time
     warning per process on instantiation. Class members (methods/properties)
     are not automatically decorated, allowing individual members to be
     explicitly annotated with their own stages if desired.
  3. Enum Classes: Attaches stage metadata to the Enum class itself. Enum
     members automatically inherit stage metadata via standard Python class
     attribute lookup rules.

  Args:
    target: The function, method, or class to annotate.
    stage: The lifecycle stage of the target.
    reason: Explanation for the stage. Required to be non-empty for experimental
      and deprecated stages.
    version: The version string when the target was deprecated (e.g. "2.13").
      Only used for deprecated targets.
    prefer: The preferred alternative to the target, if any. Only used for
      internal targets.

  Returns:
    The annotated target.
  """
  # Always attach stage metadata to the target itself
  _attach_stage_metadata(target, stage.value, reason, version, prefer)

  # 1. Handle Class
  if isinstance(target, type):
    # Attach warning wrapper to the `__init__` method
    if stage.warning_category is not None:
      target.__init__ = _create_warning_wrapper(
          target.__init__, target, stage, reason, version
      )  # pylint: disable=protected-access
    return target

  # 2. Handle Function / Callable
  if stage.warning_category is not None:
    wrapper = _create_warning_wrapper(
        target, target, stage, reason, version  # type: ignore[arg-type]
    )
    # Attach stage metadata to the wrapper as well, so external reflection and
    # scanner tools inspecting the returned wrapper can access the attributes.
    _attach_stage_metadata(wrapper, stage.value, reason, version, prefer)
    return wrapper  # type: ignore[return-value]

  return target


def experimental(reason: str) -> Callable[[_T], _T]:
  """Marks a TorchTPU Python API (function or class) as experimental.

  Experimental APIs are exposed under public namespaces (e.g. torch.tpu.*) for
  early trial, but their signature and behavior are subject to change.

  Calling an experimental function or instantiating an experimental class will
  trigger a one-time UserWarning per process.

  Args:
    reason: Non-empty explanation for why the API is experimental.

  Returns:
    The decorated function or class with experimental stage metadata attached.
  """
  if not isinstance(reason, str) or not reason.strip():
    raise ValueError("@experimental requires a non-empty reason string.")

  return lambda target: _annotate_target(
      target, stage=Stage.EXPERIMENTAL, reason=reason
  )


def stable(reason: str = "") -> Callable[[_T], _T]:
  """Marks a TorchTPU Python API (function or class) as stable.

  Stable APIs are production-ready and backward compatibility is guaranteed. No
  runtime warnings are triggered.

  Args:
    reason: Optional description or justification for stable status.

  Returns:
    The decorated function or class with stable stage metadata attached.
  """

  return lambda target: _annotate_target(
      target, stage=Stage.STABLE, reason=reason
  )


def deprecated(version: str, reason: str) -> Callable[[_T], _T]:
  """Marks a TorchTPU Python API (function or class) as deprecated.

  Deprecated APIs will be removed in a future release. Calling a deprecated API
  or instantiating a deprecated class will trigger a one-time DeprecationWarning
  per process, pointing users to the replacement API or migration guide.

  Args:
    version: The version string when the API was deprecated (e.g. "2.13").
    reason: Guidance on replacement APIs and planned removal timeline.

  Returns:
    The decorated function or class with deprecated stage metadata attached.
  """
  if not isinstance(version, str) or not version.strip():
    raise ValueError(
        "@deprecated requires a non-empty version string (e.g. version='2.13')."
    )
  if not isinstance(reason, str) or not reason.strip():
    raise ValueError("@deprecated requires a non-empty reason string.")

  return lambda target: _annotate_target(
      target, stage=Stage.DEPRECATED, reason=reason, version=version
  )


def internal_or_experimental(
    *,
    internal_when: bool,
    experimental_msg: str,
    prefer: Optional[str] = None,
) -> Callable[[_T], _T]:
  """Marks a TorchTPU Python API (function or class) as internal or experimental.

  This is for an experimental API that becomes internal once a replacement is
  available, e.g. a `torch.tpu` API whose `torch.accelerator` equivalent exists
  in the running PyTorch version.

  When `internal_when` is False, this is equivalent to
  `@experimental(experimental_msg)`.

  When `internal_when` is True, the API is internal: an implementation detail
  meant to be used only by the PyTorch framework (the torch and torch_tpu
  packages), even if it's exposed under a public namespace (e.g. torch.tpu.*)
  for PyTorch to find it. It's treated like an API with a `_` prefix in its
  path, e.g. it isn't listed as a public API. Like an experimental API, calling
  an internal function or instantiating an internal class from any other module
  (including user code and tests) triggers a one-time warning per process, which
  names `prefer` if it's given.

  Args:
    internal_when: Whether the API is internal.
    experimental_msg: Non-empty explanation for why the API is experimental.
    prefer: Optional dotted path to the preferred alternative to use while the
      API is internal, e.g. "torch.accelerator.synchronize".

  Returns:
    The decorated function or class with internal or experimental stage
    metadata attached.
  """
  if not isinstance(experimental_msg, str) or not experimental_msg.strip():
    raise ValueError(
        "@internal_or_experimental requires a non-empty experimental_msg"
        " string."
    )
  if prefer is not None and (not isinstance(prefer, str) or not prefer.strip()):
    raise ValueError(
        "@internal_or_experimental 'prefer' must be a non-empty string when"
        " provided."
    )
  if not internal_when:
    return experimental(experimental_msg)
  reason = f"Use {prefer} instead." if prefer else ""
  return lambda target: _annotate_target(
      target, stage=Stage.INTERNAL, reason=reason, prefer=prefer
  )


def _resolve_module_attribute(
    stages_map: Mapping[str, ApiStageInfo],
    name: str,
    module_name: str,
    tt_api_globals: Optional[dict[str, Any] | type],
) -> Any:
  """Resolves a dynamic module attribute from a stages map and emits warnings if applicable.

  Optionally caches the resolved value into `tt_api_globals` (a module
  `globals()` dict or a class object) when no warning is required or after a
  warning has been emitted to an external caller, so subsequent accesses hit
  the dict/class directly with zero warning overhead while internal accesses
  do not prematurely suppress warnings for future external callers.

  Args:
    stages_map: Dictionary mapping attribute names to ApiStageInfo objects.
    name: The name of the attribute to resolve.
    module_name: The display name of the module for error/warning messages.
    tt_api_globals: Optional `globals()` dictionary or class (`type`) on which
      to cache the resolved value.

  Raises:
    AttributeError: If name is not registered in stages_map.

  Returns:
    The attribute value stored in stages_map for the given name.
  """
  if name not in stages_map:
    raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

  info = stages_map[name]
  cat = info.stage.warning_category
  warned = False
  # TorchDynamo cannot trace `sys._getframe()`, so skip the caller lookup while
  # compiling. The value then stays uncached so that the next eager access still
  # warns.
  if cat is not None and not torch.compiler.is_compiling():
    # `_resolve_module_attribute` is called from a module's or metaclass's
    # `__getattr__` (frame 1). When the attribute is accessed through a virtual
    # public module wrapper (e.g. `torch.tpu.X` or `from torch.tpu import X`),
    # `public_api` and `_frozen_importlib` frames sit between that `__getattr__`
    # and the external caller. Walk past those frames starting at frame 2 to
    # locate the real caller frame for `_is_framework_caller_frame`, and
    # increment `stacklevel` for each `public_api` frame (`warnings.warn`
    # already skips `_frozen_importlib` frames automatically).
    stacklevel = 3
    try:
      frame = sys._getframe(2)
      while (
          frame is not None
          and isinstance(mod := frame.f_globals.get("__name__"), str)
          and mod.startswith((
              "torch_tpu._internal.public_api",
              "_frozen_importlib",
          ))
      ):
        if not mod.startswith("_frozen_importlib"):
          stacklevel += 1
        frame = frame.f_back
      caller_frame = frame
    except (AttributeError, ValueError):
      caller_frame = None

    if not _is_framework_caller_frame(caller_frame):
      msg = info.stage.format_warning_message(
          f"'{name}'", info.reason, info.version
      )
      warned = tt_warnings.warn(msg, category=cat, stacklevel=stacklevel)

  # Dynamically attach stage metadata to map-registered objects upon resolution.
  if info.value is not None:
    _attach_stage_metadata(
        info.value, info.stage.value, info.reason, info.version
    )

  # Only cache into tt_api_globals when no warning is required (cat is None) or
  # after the warning has been emitted to an external caller, so internal calls do
  # not prematurely cache and suppress warnings for subsequent user calls.
  if tt_api_globals is not None and (cat is None or warned):
    if isinstance(tt_api_globals, type):
      type.__setattr__(tt_api_globals, name, info.value)
    else:
      tt_api_globals[name] = info.value

  return info.value
