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

from torch_tpu._internal.device import _device_ops_backend


class _TpuBackendConfig:
  """Configurations bound to torch.backends.tpu."""

  @property
  def allow_excess_precision(self) -> bool:
    """Whether XLA is allowed to use excess precision.

    This property reflects the effective setting used by the compiler even when
    not explicitly set. First time calling this property triggers parsing and
    memoization of the XLA_FLAGS environment variable.
    """

    return _device_ops_backend._get_allow_excess_precision()  # pylint: disable=protected-access

  @allow_excess_precision.setter
  def allow_excess_precision(self, value: bool):
    """Sets whether XLA is allowed to use excess precision.

    This flag can only be set once per process. Attempting to set it a second
    time (even to the same value) will raise a RuntimeError.

    Args:
      value: Whether to allow excess precision. Can be True or False.
    """

    _device_ops_backend._set_allow_excess_precision(value)  # pylint: disable=protected-access
