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

"""Checks which PyTorch the loaded torch_tpu extension was built against.

The multi-ABI wheel bundles one extension glue per supported PyTorch release
and `torch_tpu._versioned_so_loader` picks one at import time based on the
installed PyTorch. `env.PYTORCH_BUILD_VERSION` is the PyTorch release whose
headers the loaded glue was actually compiled against, so these tests confirm
that the glue the loader claims to have picked is the one that got built and
loaded. Against a source build (no versioned glues) the dispatch is disabled,
and only the format of the build version is checked.
"""

import pathlib

from absl import logging
from absl.testing import absltest
import torch
import torch_tpu
from torch_tpu import _versioned_so_loader as loader
from torch_tpu._internal import env


def _expected_glue_suffix() -> str | None:
  """The glue suffix the loader selects for this process, or None if disabled."""
  return loader.resolve_suffix(
      loader.torch_version_suffix(),
      loader.discover_built_suffixes(pathlib.Path(torch_tpu.__file__).parent),
  )


# Self-contained (no `torch_tpu.tests` imports) because the nox smoke sessions
# run this file against the installed wheel alone.
class PyTorchBuildVersionTest(absltest.TestCase):  # ABSLTEST_OK=smoke test

  def setUp(self) -> None:
    super().setUp()
    # Surface the versions in the test log even when everything passes, so a
    # CI run shows at a glance which PyTorch (and which glue) it exercised.
    logging.info("Installed PyTorch version: %s", torch.__version__)
    logging.info("Loaded extension: %s", env.__file__)
    logging.info(
        "TorchTPU built against PyTorch version: %s",
        env.PYTORCH_BUILD_VERSION,
    )

  def test_pytorch_build_version_format(self) -> None:
    self.assertRegex(env.PYTORCH_BUILD_VERSION, r"^\d+\.\d+\.\d+$")

  def test_build_version_matches_loaded_extension(self) -> None:
    build_suffix = loader.version_suffix(env.PYTORCH_BUILD_VERSION)
    installed_suffix = loader.torch_version_suffix()
    self.assertIsNotNone(installed_suffix, torch.__version__)

    expected_suffix = _expected_glue_suffix()
    if expected_suffix is None:
      # No versioned glues, so the extension was built against the one and
      # only PyTorch available, which must be the one it is running with.
      self.assertEqual(
          build_suffix,
          installed_suffix,
          f"torch_tpu was built against PyTorch {env.PYTORCH_BUILD_VERSION}, "
          f"but PyTorch {torch.__version__} is installed.",
      )
      return

    # Multi-ABI wheel: the loader really redirected `torch_tpu._internal.env`
    # to the glue it resolved for the installed PyTorch ...
    self.assertEndsWith(
        pathlib.Path(env.__file__).name,
        f"_{expected_suffix}.so",
        f"Loaded {env.__file__}, but the loader should have selected the "
        f"{expected_suffix} glue for PyTorch {torch.__version__}.",
    )
    # ... and that glue was compiled against the PyTorch it is named for.
    self.assertEqual(
        build_suffix,
        expected_suffix,
        f"The {expected_suffix} glue reports being built against PyTorch "
        f"{env.PYTORCH_BUILD_VERSION}.",
    )


if __name__ == "__main__":
  absltest.main()
