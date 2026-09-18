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

"""Upstream PyTorch Test Runner for TorchTPU.

This script currently serves as a Minimum Viable Product (MVP). It scaffolds
a single sample test as a Proof of Concept (PoC) to validate that the
end-to-end CI pipeline infrastructure (GitHub Actions, Bazel targets, and
hardware initialization) functions correctly.

TODO: b/523293280 - Update this runner to a more appropriate, comprehensive
solution utilizing `PrivateUse1TestBase` to properly execute the full upstream
device test suites.
"""

import os
import pathlib
import sys
import pytest

_UPSTREAM_TEST_FILE = "test_torch.py"


def main():
  target_device = os.environ.get(
      "TORCH_TPU_INTERNAL_TEST_DEVICE", "cpu"
  ).lower()

  pytorch_test_dir = os.environ.get("TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR")
  if not pytorch_test_dir:
    raise EnvironmentError("TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR is not set.")

  test_file_path = str(pathlib.Path(pytorch_test_dir) / _UPSTREAM_TEST_FILE)

  # Hardcodes a single test filter as a minimal example to demonstrate how
  # upstream tests are wired up and executed in the TorchTPU CI environment.
  pytest_args = [
      "-v",
      test_file_path,
      "-k",
      "test_cdist_empty_cpu"
      if target_device == "cpu"
      else "test_cdist_empty_tpu",
  ]

  # TODO: b/523384731 - Replace this direct pytest.main() invocation with a
  # native Bazel `pytest_test` rule once the upstream PyTorch repository is
  # migrated into Bazel's dependency graph via Bzlmod. Currently, this wrapper
  # is required because Bazel cannot natively track and wrap the out-of-band
  # test files downloaded via GitHub Actions.
  sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
  main()
