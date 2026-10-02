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

"""Tests for the `torch.compile`-friendly warning helper."""

import inspect
import warnings

from absl.testing import absltest
import torch
from torch_tpu._internal import warnings as warning_categories
from torch_tpu._internal.utils import tt_warnings
from tests import seed_test_utils


def _warn_from_library(message: str) -> None:
  """Simulates a library function warning about its caller's usage."""
  tt_warnings.warn(message, warning_categories.CORRECTNESS, stacklevel=2)


class TtWarningsTest(seed_test_utils.RepeatableTest):

  def test_eager_warns_with_category_prefix_at_call_site(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      line = inspect.currentframe().f_lineno + 1
      issued = tt_warnings.warn("eager message", warning_categories.TORCH_TPU)
    self.assertTrue(issued)
    self.assertLen(w, 1)
    self.assertIs(w[0].category, UserWarning)
    self.assertEqual(str(w[0].message), "<TorchTpu/> eager message")
    self.assertEqual(w[0].filename, __file__)
    self.assertEqual(w[0].lineno, line)

  def test_eager_stacklevel_is_relative_to_caller(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      line = inspect.currentframe().f_lineno + 1
      _warn_from_library("caller message")
    self.assertLen(w, 1)
    self.assertIs(w[0].category, UserWarning)
    self.assertEqual(
        str(w[0].message), "<TorchTpu/Correctness/> caller message"
    )
    self.assertEqual(w[0].filename, __file__)
    self.assertEqual(w[0].lineno, line)

  def test_deprecated_api_uses_deprecation_warning(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      tt_warnings.warn(
          "deprecated message",
          warning_categories.DEPRECATED_API,
      )
    self.assertLen(w, 1)
    self.assertIs(w[0].category, DeprecationWarning)
    self.assertEqual(
        str(w[0].message),
        "<TorchTpu/ApiStage/DeprecatedApi/> deprecated message",
    )

  def test_filter_by_category_prefix(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      warnings.filterwarnings("ignore", message="<TorchTpu/Performance/")
      tt_warnings.warn("slow", warning_categories.UNOPTIMIZED_BUILD)
      tt_warnings.warn("wrong", warning_categories.CORRECTNESS)
    self.assertEqual(
        [str(item.message) for item in w],
        ["<TorchTpu/Correctness/> wrong"],
    )

  def test_error_filter_raises_prefixed_builtin_warning(self):
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      with self.assertRaisesRegex(
          UserWarning, r"^<TorchTpu/Performance/> slow$"
      ):
        tt_warnings.warn("slow", warning_categories.PERFORMANCE)

  def test_compile_fullgraph_does_not_warn(self):
    def fn(x: torch.Tensor) -> torch.Tensor:
      tt_warnings.warn("compiled message", warning_categories.TORCH_TPU)
      return x + 1

    # `fullgraph=True` raises on any graph break.
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      compiled(torch.ones(3))
    self.assertEmpty(
        [item for item in w if "compiled message" in str(item.message)]
    )


if __name__ == "__main__":
  absltest.main()
