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
    # pylint: disable=protected-access
    return _device_ops_backend._get_allow_excess_precision()

  @allow_excess_precision.setter
  def allow_excess_precision(self, value: bool):
    """Sets whether XLA is allowed to use excess precision.

    It sets the XLA flag of --allow_excess_precision globally for all future
    compilations. Changing its value would result in compile cache eviction.

    Currently, the compilation cache will be evicted if this flag is changed,
    which would impact the performance. Recommend not changing it frequently.
    File a feature request for a context manager for this flag if needed.

    Args:
      value: Whether to allow excess precision. Can be True or False.
    """
    # TODO: b/498591854 - Update the docstring when the cache is no longer
    # evicted.
    # TODO: b/502610173 - Set to False when XLA_FLAGS is not set. Change the
    # docstring accordingly.
    # pylint: disable=protected-access
    _device_ops_backend._set_allow_excess_precision(value)
