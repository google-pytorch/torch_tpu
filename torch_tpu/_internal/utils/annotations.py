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
"""

import enum
import functools
from typing import Any, Callable, Optional, TypeVar
import warnings

_T = TypeVar("_T")

TT_API_STAGE = "__tt_api_stage__"
TT_API_STAGE_REASON = "__tt_api_stage_reason__"
TT_API_DEPRECATED_VERSION = "__tt_api_deprecated_version__"


class Stage(str, enum.Enum):
  """Lifecycle stages for TorchTPU APIs."""

  STABLE = "Stable"
  EXPERIMENTAL = "Experimental"
  DEPRECATED = "Deprecated"

  @property
  def warning_category(self) -> Optional[type[Warning]]:
    """The Warning category for this stage."""
    if self == Stage.EXPERIMENTAL:
      return UserWarning
    if self == Stage.DEPRECATED:
      return DeprecationWarning
    return None

  def format_warning_message(
      self, name: str, reason: str, version: Optional[str] = None
  ) -> str:
    """Formats the warning message for this stage."""
    if self == Stage.EXPERIMENTAL:
      return f"{name} is experimental and subject to change: {reason}"
    if self == Stage.DEPRECATED:
      return (
          f"{name} is deprecated as of TorchTPU {version} and will be"
          f" removed in a future release: {reason}"
      )
    return ""


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
) -> None:
  """Attaches stage metadata attributes to a target.

  Args:
    target: The target object to modify.
    stage: The lifecycle stage of the target.
    reason: Explanation for the stage.
    version: The version string when the target was deprecated (e.g. "2.13").
      Only used for deprecated targets.
  """
  _safe_setattr(target, TT_API_STAGE, stage)
  _safe_setattr(target, TT_API_STAGE_REASON, reason)
  if version:
    _safe_setattr(target, TT_API_DEPRECATED_VERSION, version)


def _create_warning_wrapper(
    fn_to_wrap: Callable[..., Any],
    display_target: Any,
    stage: Stage,
    reason: str,
    version: Optional[str] = None,
) -> Callable[..., Any]:
  """Wraps a callable to emit a one-time warning per process before execution.

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
    if not _warned:
      name = getattr(display_target, "__name__", str(display_target))
      msg = stage.format_warning_message(name, reason, version)
      cat = stage.warning_category
      if cat:
        warnings.warn(msg, category=cat, stacklevel=2)
      _warned = True
    return fn_to_wrap(*args, **kwargs)

  return wrapper


class _EnumProxy:
  """Proxy wrapper for Enum classes to trigger runtime warnings upon access."""

  def __init__(
      self,
      enum_cls: Any,
      stage_obj: Stage,
      reason: str,
      version: Optional[str] = None,
  ):
    self._enum_cls = enum_cls
    self._stage_obj = stage_obj
    self._reason = reason
    self._version = version
    self._warned = False
    _attach_stage_metadata(self, stage_obj.value, reason, version)

  def _warn_if_needed(self) -> None:
    if not self._warned:
      msg = self._stage_obj.format_warning_message(
          self._enum_cls.__name__, self._reason, self._version
      )
      cat = self._stage_obj.warning_category
      if cat:
        warnings.warn(msg, category=cat, stacklevel=3)
      self._warned = True

  def __getattr__(self, name: str) -> Any:
    self._warn_if_needed()
    return getattr(self._enum_cls, name)

  def __getitem__(self, item: Any) -> Any:
    self._warn_if_needed()
    return self._enum_cls[item]

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    self._warn_if_needed()
    return self._enum_cls(*args, **kwargs)

  def __iter__(self) -> Any:
    self._warn_if_needed()
    return iter(self._enum_cls)

  def __len__(self) -> int:
    self._warn_if_needed()
    return len(self._enum_cls)

  def __contains__(self, member: Any) -> bool:
    self._warn_if_needed()
    return member in self._enum_cls

  def __repr__(self) -> str:
    return repr(self._enum_cls)


def _annotate_target(
    target: _T,
    stage: Stage,
    reason: str,
    version: Optional[str] = None,
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
  3. Enum Classes: Wraps the Enum class with `_EnumProxy` when a warning is
     required (e.g. @experimental or @deprecated), which emits a one-time
     warning on first member access/instantiation.

  Args:
    target: The function, method, or class to annotate.
    stage: The lifecycle stage of the target.
    reason: Explanation for the stage. Required to be non-empty for experimental
      and deprecated stages.
    version: The version string when the target was deprecated (e.g. "2.13").
      Only used for deprecated targets.

  Returns:
    The annotated target.
  """
  # Always attach stage metadata to the target itself
  _attach_stage_metadata(target, stage.value, reason, version)

  # Handle Class / Enum Class
  if isinstance(target, type):
    # Enum Class requiring warning -> wrap with _EnumProxy
    if issubclass(target, enum.Enum) and stage.warning_category:
      return _EnumProxy(target, stage, reason, version)  # type: ignore[return-value]

    # Non-Enum Class: Attach warning wrapper to `__init__`
    if stage.warning_category:
      target.__init__ = _create_warning_wrapper(
          target.__init__, target, stage, reason, version
      )  # pylint: disable=protected-access
    return target

  # Handle Function / Callable
  if stage.warning_category:
    wrapper = _create_warning_wrapper(
        target, target, stage, reason, version  # type: ignore[arg-type]
    )
    # Attach stage metadata to the wrapper as well, so external reflection and
    # scanner tools inspecting the returned wrapper can access the attributes.
    _attach_stage_metadata(wrapper, stage.value, reason, version)
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
