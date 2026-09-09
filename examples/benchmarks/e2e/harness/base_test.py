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

"""Base test class for benchmark tests."""

import os

from absl import logging
from absl.testing import absltest

_DRY_RUN_OUTPUT_FILE = "test_targets.txt"


def _get_output_dir(key: str) -> str:
  try:
    return os.environ[key]
  except KeyError:
    raise RuntimeError(
        f"Output directory not set using environment variable {key}"
    )


class BaseBenchmarkTest(
    absltest.TestCase  # ABSLTEST_OK=Avoid loading torch_tpu shared C++ libs.
):
  """Base test class for benchmark tests."""

  def setUp(self):
    super().setUp()
    log_counter = getattr(logging, "_log_counter_per_token", None)
    if log_counter is not None:
      log_counter.clear()

  def _dry_run_test(self) -> None:
    """Logs and writes dry run test details to a text file."""
    logging.info(
        "[DRY RUN]: Would have run benchmark test method name %s",
        self._testMethodName,
    )
    output_dir = _get_output_dir("TEST_UNDECLARED_OUTPUTS_DIR")
    txt_path = os.path.join(output_dir, _DRY_RUN_OUTPUT_FILE)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, mode="a", encoding="utf-8") as f:
      f.write(self._testMethodName + "\n")
