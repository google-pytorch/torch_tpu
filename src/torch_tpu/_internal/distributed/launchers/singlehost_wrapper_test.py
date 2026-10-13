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

"""Tests for the deprecated `singlehost_wrapper` compatibility shim."""

from typing import Any
from unittest import mock
import warnings

from absl.testing import absltest
from torch_tpu._internal.distributed import distributed
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import annotations
from tests import seed_test_utils


def _reset_warn_once(api: Any) -> None:
  fn = getattr(api, "__func__", api)
  if hasattr(fn, "__code__") and "_warned" in fn.__code__.co_freevars:
    idx = fn.__code__.co_freevars.index("_warned")
    fn.__closure__[idx].cell_contents = False


class SingleHostWrapperTest(seed_test_utils.RepeatableTest):

  def test_stage_metadata(self):
    self.assertEqual(
        getattr(
            singlehost_wrapper.prepare_tpu_environment, annotations.TT_API_STAGE
        ),
        "Deprecated",
    )

  def test_prepare_tpu_environment_delegates_and_warns(self):
    _reset_warn_once(singlehost_wrapper.prepare_tpu_environment)
    mock_set_env = self.enter_context(
        mock.patch.object(
            distributed, "set_tpu_launch_env", autospec=True, spec_set=True
        )
    )

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      singlehost_wrapper.prepare_tpu_environment(world_size=4)

    mock_set_env.assert_called_once_with(nproc_per_node=4)
    deprecation_msgs = [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
    ]
    self.assertTrue(
        any(
            "<TorchTpu/ApiStage/DeprecatedApi/>" in msg
            and "prepare_tpu_environment is deprecated as of TorchTPU 0.2.0"
            in msg
            for msg in deprecation_msgs
        ),
        msg=f"Expected deprecation warning in {deprecation_msgs}",
    )


if __name__ == "__main__":
  absltest.main()
