# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import shutil
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal.profiler.profiler_config import TpuProfilerConfig
from tests import seed_test_utils

# Set TMPDIR to TEST_TMPDIR so TorchTPU's profiler writes artifacts into the
# test's temporary directory instead of falling back to system /tmp.
if "TEST_TMPDIR" in os.environ:
  os.environ["TMPDIR"] = os.environ["TEST_TMPDIR"]


def _get_profile_dir() -> pathlib.Path:
  tmpdir = os.environ.get("TEST_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
  return pathlib.Path(tmpdir) / "plugins" / "profile"


def _cleanup_profile_dir() -> None:
  try:
    shutil.rmtree(_get_profile_dir())
  except FileNotFoundError:
    pass


class ProfilerIntegrationTest(seed_test_utils.RepeatableTest):

  def test_invalid_profiler_config(self):
    """Tests that invalid profiler configuration raises a Python exception."""
    device = torch.device("tpu")
    output_dir = self.create_tempdir("invalid_config").full_path

    _cleanup_profile_dir()

    bad_config = torch.profiler._ExperimentalConfig(
        custom_profiler_config="invalid_option_missing_colon"
    )

    with self.assertRaisesRegex(
        RuntimeError,
        r"expected the config item to be in the 'key:value' format, got"
        r" 'invalid_option_missing_colon'",
    ):
      with torch.profiler.profile(
          activities=[
              torch.profiler.ProfilerActivity.CPU,
              torch.profiler.ProfilerActivity.TPU,  # type: ignore
          ],
          on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
          experimental_config=bad_config,
      ):
        _ = torch.ones((16, 16)).to(device) @ torch.ones((16, 16)).to(device)

  @unittest.skip(
      "Skipped until new libtpu wheel containing BrokenProfiler is rolled into"
      " google3 (bootstrap dependency)."
  )
  def test_tpu_profiler_config_invalid_experimental_option(self):
    device = torch.device("tpu")
    output_dir = pathlib.Path(
        self.create_tempdir("invalid_experimental_option").full_path
    )

    config = TpuProfilerConfig(
        host_tracer_level=2,
        experimental_options={
            "invalid_option_key": "some_value",
        },
        run_dir=output_dir,
    )

    with self.assertRaisesRegex(
        RuntimeError,
        r"Parsing advanced_configuration failed\. The following keys "
        r"were not recognized: invalid_option_key",
    ):
      with torch.profiler.profile(
          activities=[
              torch.profiler.ProfilerActivity.CPU,
              torch.profiler.ProfilerActivity.TPU,  # type: ignore
          ],
          on_trace_ready=torch.profiler.tensorboard_trace_handler(
              str(output_dir)
          ),
          experimental_config=config,
      ):
        a = torch.ones((16, 16)).to(device)
        b = torch.ones((16, 16)).to(device)
        _ = a @ b

  @parameterized.named_parameters(
      dict(testcase_name="colon", key="invalid:key", value="value"),
      dict(testcase_name="comma", key="invalid,key", value="value"),
  )
  def test_tpu_profiler_config_invalid_characters_in_experimental_options(
      self, key, value
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"Experimental option keys cannot contain ':' or ','",
    ):
      TpuProfilerConfig(experimental_options={key: value})

  def test_tpu_profiler_config_worker_rank_validation(self):
    # Valid values should not raise errors:
    _ = TpuProfilerConfig(worker_rank="3")
    _ = TpuProfilerConfig(worker_rank="worker_A")
    _ = TpuProfilerConfig(worker_rank=None)

    # Invalid values should raise TypeError or ValueError:
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=0)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=42)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=-1)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=3.5)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=True)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank={"rank": 1})
    with self.assertRaises(ValueError):
      TpuProfilerConfig(worker_rank="worker:1")
    with self.assertRaises(ValueError):
      TpuProfilerConfig(worker_rank="worker,1")


if __name__ == "__main__":
  absltest.main()
