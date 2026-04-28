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

"""Tests that C++ error traces are included in error messages when requested."""

import os
import re
from absl.testing import absltest

# Enable C++ context in pytorch error messages. This must be done before
# importing torch.
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

import torch  # pylint: disable=g-import-not-at-top


class TestErrorsWithCppContext(absltest.TestCase):
  """Tests that C++ error traces are included in error messages when requested."""

  def setUp(self):
    super().setUp()
    self.device = torch.accelerator.current_accelerator()
    self.maxDiff = None  # Show long diffs in assertEqual.

  def test_error_has_cpp_error_trace(self):
    with self.assertRaises(RuntimeError) as cm:
      torch.ones(-1, device=self.device, dtype=torch.float32)
    # The error message should look like:
    #   empty(): dimension sizes must be >= 0, got [-1], which contains -1
    #
    #   C++ error trace (starting from the origin):
    #   third_party/py/torch_tpu/common/error_utils.cc:95: ValidateTensorByteSize()
    #   third_party/py/torch_tpu/device_buffer.cc:361: CreateEmpty()
    #   third_party/py/torch_tpu/ops/nullary_aten_kernels.cc:90: MakeEmptyBuffer()
    #
    #   Exception raised from operator() at third_party/py/torch_tpu/ops/nullary_aten_kernels.cc:155 (most recent call first):
    #   C++ CapturedTraceback:
    #   ...
    self.assertRegex(  # This does a substring match.
        str(cm.exception),
        re.compile(
            r"\n\nC\+\+ error trace \(starting from the origin\):\n"
            r".*torch_tpu/.*\.cc:\d+: \w+\(\)\n"
            r".*\n"
            r"Exception raised from \w+ at .*\.cc:\d+ .*\n"
            r"C\+\+ CapturedTraceback:\n"
            r"#.*",
            re.DOTALL | re.MULTILINE,
        ),
    )


if __name__ == "__main__":
  absltest.main()
