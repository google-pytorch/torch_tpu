# Copyright 2025 Google LLC
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

"""Tests for _internal.profiler.profile APIs."""

from absl.testing import absltest
from torch_tpu._internal import profiler
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.profiler import _impl as profiler_impl
from tests import seed_test_utils


class ProfilerApiTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    profiler_impl._profile_state.reset()
    # Clean up any previous server potentially left running from a crashed test
    if profiler_impl._profiler_server is not None:
      profiler_impl.stop_server()

  def test_profile_api_with_invalid_rank(self):
    output_dir = self.create_tempdir('profile_api_with_invalid_rank').full_path
    handler = profiler.xprof_trace_handler(
        dir_name=output_dir, worker_rank='../../invalid'
    )
    with self.assertRaisesRegex(
        ValueError, 'worker_rank cannot contain path separators'
    ):
      with profiler.profile(
          activities=[profiler.ProfilerActivity.CPU],
          on_trace_ready=handler,
      ):
        pass

  def test_profile_api_gpu_unsupported(self):
    output_dir = self.create_tempdir('profile_api_gpu_unsupported').full_path
    handler = profiler.xprof_trace_handler(dir_name=output_dir)
    with self.assertRaisesRegex(ValueError, 'GPU profiling is not supported.'):
      with profiler.profile(
          activities=[profiler.ProfilerActivity.GPU],
          on_trace_ready=handler,
      ):
        pass


if __name__ == '__main__':
  absltest.main()
