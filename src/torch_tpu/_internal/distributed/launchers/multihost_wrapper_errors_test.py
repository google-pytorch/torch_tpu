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

"""Error tests for multihost_wrapper."""

import contextlib
import os
from unittest import mock

from absl.testing import absltest
from torch_tpu._internal import env
from torch_tpu._internal.distributed.launchers import multihost_wrapper
from tests import seed_test_utils


@contextlib.contextmanager
def _mock_env(values: dict[str, str]):
  """Makes the environment look exactly like `values` to the wrapper."""

  def get_str_env_once(name, default_value=None):
    if name in os.environ:
      return os.environ[name]
    return values.get(name, default_value)

  def get_int_env_once(name, default_value=None):
    if name in os.environ:
      return int(os.environ[name])
    return int(values[name]) if name in values else default_value

  def get_bool_env_once(name, default_value=None):
    raw_value = os.environ.get(name, values.get(name))
    if raw_value is not None:
      val = str(raw_value).lower()
      if val in ("1", "true", "yes", "on"):
        return True
      if val in ("0", "false", "no", "off"):
        return False
      return default_value
    return default_value

  with contextlib.ExitStack() as stack:
    stack.enter_context(mock.patch.dict(os.environ, values, clear=True))
    stack.enter_context(
        mock.patch.object(env, "get_str_env_once", side_effect=get_str_env_once)
    )
    stack.enter_context(
        mock.patch.object(env, "get_int_env_once", side_effect=get_int_env_once)
    )
    stack.enter_context(
        mock.patch.object(
            env, "get_bool_env_once", side_effect=get_bool_env_once
        )
    )
    yield


class MultiHostWrapperErrorsTest(seed_test_utils.RepeatableTest):

  def test_parse_topology_empty_string(self):
    with self.assertRaises(ValueError):
      multihost_wrapper.parse_topology("")

  def test_calculate_topological_node_rank_none(self):
    with self.assertRaises(ValueError):
      multihost_wrapper.calculate_topological_node_rank(None, None)

  def test_calculate_topological_node_rank_out_of_bounds_fallback(self):
    with self.assertRaisesRegex(ValueError, "Cannot determine NODE_RANK"):
      multihost_wrapper.calculate_topological_node_rank(
          [10, 0, 0], "4x4x4", fallback_index=None
      )
    with self.assertRaisesRegex(ValueError, "Cannot determine NODE_RANK"):
      multihost_wrapper.calculate_topological_node_rank(
          [-1, 0, 0], "4x4x4", fallback_index=None
      )

  def test_prepare_tpu_environment_with_coordinates_missing_coords_and_completion_index(
      self,
  ):
    env_vars = {
        "TPU_WORKER_HOSTNAMES": "h1",
        "PJRT_DEVICE": "TPU",
    }
    with (
        _mock_env(env_vars),
        mock.patch(
            "torch_tpu._internal.distributed.tpu_distributed.local_device_coords",
            return_value=[],
        ),
    ):
      with self.assertRaisesRegex(ValueError, "Cannot determine NODE_RANK"):
        multihost_wrapper.prepare_tpu_environment()


if __name__ == "__main__":
  absltest.main()
