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

"""Tests for the tcmalloc startup check in torch_tpu.__init__."""

import os
import sys
import types
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import torch_tpu
from tests import seed_test_utils


class CheckTcmallocTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch_tpu._tcmalloc_warning_emitted = False

  def test_warns_once_when_tcmalloc_missing(self):
    empty_syms = types.SimpleNamespace()
    with (
        mock.patch.object(sys, "platform", "linux"),
        mock.patch("ctypes.CDLL", return_value=empty_syms),
        mock.patch(
            "builtins.open",
            mock.mock_open(read_data="7f000000 /usr/lib/libc.so.6\n"),
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
      warnings.simplefilter("always")
      torch_tpu._check_tcmalloc()
      torch_tpu._check_tcmalloc()
      tcmalloc_warnings = [
          w for w in caught if "tcmalloc is not loaded" in str(w.message)
      ]
      self.assertLen(tcmalloc_warnings, 1)

  @parameterized.named_parameters(
      ("disable_tcmalloc", {"TORCH_TPU_DISABLE_TCMALLOC": "1"}, "", False),
      (
          "disable_check",
          {"TORCH_TPU_DISABLE_TCMALLOC_CHECK": "1"},
          "",
          False,
      ),
      (
          "maps_has_tcmalloc",
          {},
          "7f000000 /workspace/libtcmalloc.so\n",
          False,
      ),
      (
          "maps_has_tsan",
          {},
          "7f000000 /usr/lib/libtsan.so.0\n",
          False,
      ),
      (
          "rtld_has_tcmalloc_symbol",
          {},
          "",
          True,
      ),
  )
  def test_succeeds_without_warning(
      self, env_overrides, maps_content, has_symbol
  ):
    syms = (
        types.SimpleNamespace(MallocExtension_GetAllocatedSize=object())
        if has_symbol
        else types.SimpleNamespace()
    )
    with (
        mock.patch.object(sys, "platform", "linux"),
        mock.patch.dict(os.environ, env_overrides),
        mock.patch("ctypes.CDLL", return_value=syms),
        mock.patch("builtins.open", mock.mock_open(read_data=maps_content)),
        warnings.catch_warnings(record=True) as caught,
    ):
      warnings.simplefilter("always")
      torch_tpu._check_tcmalloc()
      self.assertFalse(
          any("tcmalloc is not loaded" in str(w.message) for w in caught)
      )


if __name__ == "__main__":
  absltest.main()
