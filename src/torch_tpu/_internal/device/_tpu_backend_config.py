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

"""torch.backends.tpu configurations."""

from typing import Any

from torch_tpu._internal.device import _device_ops_backend
from torch_tpu._internal.utils import annotations

__tt_api_stages__: dict[str, annotations.ApiStageInfo] = {}


class _TpuBackendConfig:
  """Configurations bound to torch.backends.tpu."""

  __tt_api_stages__ = __tt_api_stages__

  # Dynamic attribute lookup for deprecated or fallback module attributes.
  # tt_api_globals is set to self.__dict__ so resolved attributes are cached
  # on the instance dict, bypassing __getattr__ on 2nd access.
  def __getattr__(self, name: str) -> Any:
    return annotations._resolve_module_attribute(
        self.__tt_api_stages__,
        name,
        "torch.backends.tpu",
        tt_api_globals=self.__dict__,
    )

  @property
  @annotations.experimental(
      "allow_excess_precision is experimental and may change or be removed "
      "without notice."
  )
  def allow_excess_precision(self) -> bool:
    """Whether XLA is allowed to use excess precision.

    This property reflects the effective setting used by the XLA compiler even
    when not explicitly set. First time calling this property triggers parsing
    and memoization of the XLA_FLAGS environment variable.
    """
    # pylint: disable=protected-access
    return _device_ops_backend._get_allow_excess_precision()

  @allow_excess_precision.setter
  @annotations.experimental(
      "allow_excess_precision is experimental and may change or be removed "
      "without notice."
  )
  def allow_excess_precision(self, value: bool):
    """Sets whether XLA is allowed to use excess precision.

    It sets the XLA flag of --allow_excess_precision globally for all future
    compilations. Changing its value frequently is not recommended as it will
    trigger recompilation for the new setting.
    File a feature request for a context manager for this flag if needed.

    The flag defaults to False if not set in the XLA_FLAGS environment variable
    nor explicitly set by this property.

    Args:
      value: Whether to allow excess precision. Can be True or False.
    """

    # pylint: disable=protected-access
    _device_ops_backend._set_allow_excess_precision(value)
