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

import dataclasses
import enum
import functools
from typing import Any, Callable, Mapping, Optional, TypeVar
import warnings

_T = TypeVar("_T")

TT_API_STAGE = "__tt_api_stage__"
TT_API_STAGE_REASON = "__tt_api_stage_reason__"
TT_API_DEPRECATED_VERSION = "__tt_api_deprecated_version__"
TT_API_STAGES_MAP = "__tt_api_stages__"


class Stage(str, enum.Enum):
  """Lifecycle stages for TorchTPU APIs."""
  INTERNAL = "Internal"
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

  Returns:
    The annotated target.
  """
  # Always attach stage metadata to the target itself
  _attach_stage_metadata(target, stage.value, reason, version)

  # 1. Handle Class
  if isinstance(target, type):
    # Attach warning wrapper to the `__init__` method
    if stage.warning_category:
      target.__init__ = _create_warning_wrapper(
          target.__init__, target, stage, reason, version
      )  # pylint: disable=protected-access
    return target

  # 2. Handle Function / Callable
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


def _resolve_module_attribute(
    stages_map: Mapping[str, ApiStageInfo],
    name: str,
    module_name: str,
    tt_api_globals: Optional[dict[str, Any]],
) -> Any:
  """Resolves a dynamic module attribute from a stages map and emits warnings if applicable.

  Optionally caches the resolved value into tt_api_globals dict so that
  subsequent accesses in the same process hit globals() directly with zero
  warning overhead.

  Args:
    stages_map: Dictionary mapping attribute names to ApiStageInfo objects.
    name: The name of the attribute to resolve.
    module_name: The display name of the module for error/warning messages.
    tt_api_globals: Optional globals() dictionary of the calling module to cache
      the resolved value.

  Returns:
    The attribute value stored in stages_map for the given name.

  Raises:
    AttributeError: If name is not registered in stages_map.
  """
  if name not in stages_map:
    raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

  info = stages_map[name]
  cat = info.stage.warning_category
  if cat:
    msg = info.stage.format_warning_message(
        f"'{name}'", info.reason, info.version
    )
    warnings.warn(msg, category=cat, stacklevel=3)

  # Dynamically attach stage metadata to map-registered objects upon resolution.
  if info.value is not None:
    _attach_stage_metadata(
        info.value, info.stage.value, info.reason, info.version
    )

  if tt_api_globals is not None:
    tt_api_globals[name] = info.value

  return info.value
