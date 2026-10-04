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

"""Tests that TT_WARN and TT_WARN_ONCE from C++ honor Python warning filters."""

import re
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal import warnings as warning_categories
from tests import seed_test_utils

# The prefix of each category. These are public API: users filter on them.
_EXPECTED_PREFIXES = {
    warning_categories.TORCH_TPU: "<TorchTpu/> ",
    warning_categories.PERFORMANCE: "<TorchTpu/Performance/> ",
    warning_categories.UNOPTIMIZED_BUILD: (
        "<TorchTpu/Performance/UnoptimizedBuild/> "
    ),
    warning_categories.TCMALLOC: "<TorchTpu/Performance/TcMalloc/> ",
    warning_categories.CORRECTNESS: "<TorchTpu/Correctness/> ",
    warning_categories.API_STAGE: "<TorchTpu/ApiStage/> ",
    warning_categories.INTERNAL_API: "<TorchTpu/ApiStage/InternalApi/> ",
    warning_categories.EXPERIMENTAL_API: (
        "<TorchTpu/ApiStage/ExperimentalApi/> "
    ),
    warning_categories.DEPRECATED_API: "<TorchTpu/ApiStage/DeprecatedApi/> ",
}


def _matching(
    recorded: list[warnings.WarningMessage], text: str
) -> list[warnings.WarningMessage]:
  return [w for w in recorded if text in str(w.message)]


class CppWarningsTest(seed_test_utils.RepeatableTest):

  def test_all_categories_have_expected_prefixes(self) -> None:
    """Verifies the prefixes of all categories, which are public API."""
    self.assertCountEqual(
        _EXPECTED_PREFIXES,
        warning_categories.WarningCategory.__members__.values(),
    )
    for category, prefix in _EXPECTED_PREFIXES.items():
      self.assertEqual(warning_categories.get_prefix(category), prefix)

  @parameterized.named_parameters(
      (category.name, category) for category in _EXPECTED_PREFIXES
  )
  def test_prefix_is_usable_as_filter_everywhere(
      self, category: warning_categories.WarningCategory
  ) -> None:
    """Verifies the filter for a prefix reads the same in every filter syntax."""
    filter_text = warning_categories.get_prefix(category).removesuffix(" ")
    # `-W` and `PYTHONWARNINGS` split on ":" and "," and strip whitespace.
    self.assertNotRegex(filter_text, r"[:,\s]")
    # Code and pytest ini filters are regexes; `-W` filters are literal text.
    self.assertEqual(re.escape(filter_text), filter_text)

  @parameterized.named_parameters(
      (category.name, category) for category in _EXPECTED_PREFIXES
  )
  def test_emit_tt_warn_uses_category_prefix(
      self, category: warning_categories.WarningCategory
  ) -> None:
    """Verifies TT_WARN emits the category prefix and built-in category."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      tt_testing.emit_tt_warn(category, "Categorized C++ warning")

    matching = _matching(recorded, "Categorized C++ warning")
    self.assertLen(matching, 1)
    self.assertIs(
        matching[0].category, warning_categories.get_builtin_category(category)
    )
    # PyTorch appends the C++ source location to the message.
    self.assertStartsWith(
        str(matching[0].message),
        _EXPECTED_PREFIXES[category] + "Categorized C++ warning",
    )

  def test_emit_tt_warn_once(self) -> None:
    """Verifies TT_WARN_ONCE fires once, with the category prefix."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      for _ in range(2):
        tt_testing.emit_tt_warn_once(
            warning_categories.EXPERIMENTAL_API,
            "Once experimental warning",
        )

    matching = _matching(recorded, "Once experimental warning")
    self.assertLen(matching, 1)
    self.assertIs(matching[0].category, UserWarning)
    self.assertStartsWith(
        str(matching[0].message),
        warning_categories.format_message(
            warning_categories.EXPERIMENTAL_API, "Once experimental warning"
        ),
    )

  def test_emit_tt_warn_once_with_warn_always(self) -> None:
    """Verifies TT_WARN_ONCE warns every time under torch.set_warn_always."""
    # Doesn't consume the once-per-process warning of the only TT_WARN_ONCE
    # call site in `emit_tt_warn_once`, so test order doesn't matter.
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      torch.set_warn_always(True)
      try:
        for _ in range(2):
          tt_testing.emit_tt_warn_once(
              warning_categories.DEPRECATED_API,
              "Always deprecated warning",
          )
      finally:
        torch.set_warn_always(False)

    matching = _matching(recorded, "Always deprecated warning")
    self.assertLen(matching, 2)
    for w in matching:
      self.assertIs(w.category, DeprecationWarning)
      self.assertStartsWith(
          str(w.message),
          warning_categories.format_message(
              warning_categories.DEPRECATED_API, "Always deprecated warning"
          ),
      )

  def test_plain_torch_warn_has_no_prefix(self) -> None:
    """Verifies a plain TORCH_WARN is not given a TorchTPU prefix."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      tt_testing.emit_torch_warn("Untagged C++ warning")

    matching = _matching(recorded, "Untagged C++ warning")
    self.assertLen(matching, 1)
    self.assertIs(matching[0].category, UserWarning)
    self.assertStartsWith(str(matching[0].message), "Untagged C++ warning")

  def test_error_filter_raises_prefixed_warning(self) -> None:
    """Verifies simplefilter("error") raises the prefixed built-in warning."""
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      with self.assertRaises(  # ASSERT_RAISES_OK=Verifies the error filter.
          UserWarning
      ) as cm:
        tt_testing.emit_tt_warn(
            warning_categories.PERFORMANCE,
            "Raised performance warning",
        )

    self.assertIs(type(cm.exception), UserWarning)
    self.assertStartsWith(
        str(cm.exception),
        warning_categories.format_message(
            warning_categories.PERFORMANCE, "Raised performance warning"
        ),
    )

  @parameterized.named_parameters(
      ("torch_tpu", "<TorchTpu/"),
      ("api_stage", "<TorchTpu/ApiStage/"),
      ("experimental_api", "<TorchTpu/ApiStage/ExperimentalApi/"),
      ("experimental_api_only", "<TorchTpu/ApiStage/ExperimentalApi/>"),
  )
  def test_prefix_ignore_filter_matches_category_and_subcategories(
      self, prefix: str
  ) -> None:
    """Verifies a prefix filter matches its category and subcategories."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      warnings.filterwarnings("ignore", message=re.escape(prefix))
      tt_testing.emit_tt_warn(
          warning_categories.EXPERIMENTAL_API,
          "Ignored experimental warning",
      )

    self.assertEmpty(_matching(recorded, "Ignored experimental warning"))

  @parameterized.named_parameters(
      ("sibling", "<TorchTpu/Performance/"),
      ("parent_only", "<TorchTpu/ApiStage/>"),
      ("torch_tpu_only", "<TorchTpu/>"),
      # A trailing "/" keeps a category from matching another category whose
      # name it is a prefix of.
      ("name_prefix", "<TorchTpu/ApiStage/Experimental/"),
  )
  def test_prefix_ignore_filter_does_not_match_other_categories(
      self, prefix: str
  ) -> None:
    """Verifies a prefix filter leaves other categories alone."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      warnings.filterwarnings("ignore", message=re.escape(prefix))
      tt_testing.emit_tt_warn(
          warning_categories.EXPERIMENTAL_API,
          "Shown experimental warning",
      )

    self.assertLen(_matching(recorded, "Shown experimental warning"), 1)

  def test_category_error_filter(self) -> None:
    """Verifies an "error" filter for one category leaves others as warnings."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      warnings.filterwarnings(
          "error", message=re.escape("<TorchTpu/Performance/")
      )
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Verifies the error filter.
          UserWarning, "Category error filter performance warning"
      ):
        tt_testing.emit_tt_warn(
            warning_categories.PERFORMANCE,
            "Category error filter performance warning",
        )
      tt_testing.emit_tt_warn(
          warning_categories.CORRECTNESS,
          "Category error filter correctness warning",
      )

    self.assertLen(
        _matching(recorded, "Category error filter correctness warning"), 1
    )

  def test_filterwarnings_ignore_user_warning(self) -> None:
    """Verifies an "ignore" filter for UserWarning suppresses TorchTPU warnings."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      warnings.filterwarnings("ignore", category=UserWarning)
      tt_testing.emit_tt_warn(
          warning_categories.PERFORMANCE,
          "UserWarning-ignored performance warning",
      )

    self.assertEmpty(
        _matching(recorded, "UserWarning-ignored performance warning")
    )

  def test_filterwarnings_once(self) -> None:
    """Verifies the "once" action deduplicates C++ warnings."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.filterwarnings("once", message="<TorchTpu/")
      for _ in range(2):
        tt_testing.emit_tt_warn(
            warning_categories.TORCH_TPU,
            "Deduplicated warning message",
        )

    self.assertLen(_matching(recorded, "Deduplicated warning message"), 1)

  def test_filterwarnings_by_module(self) -> None:
    """Verifies module filters match the module triggering the warning."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("always")
      warnings.filterwarnings("error", module="some_other_module")
      tt_testing.emit_tt_warn(
          warning_categories.PERFORMANCE,
          "Module filter performance warning",
      )
      warnings.filterwarnings("error", module=__name__)
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Verifies module filter.
          UserWarning, "Module filter performance warning"
      ):
        tt_testing.emit_tt_warn(
            warning_categories.PERFORMANCE,
            "Module filter performance warning",
        )

    self.assertLen(_matching(recorded, "Module filter performance warning"), 1)

  def test_default_action_deduplicates_same_location(self) -> None:
    """Verifies the "default" action shows a warning once per location."""
    with warnings.catch_warnings(record=True) as recorded:
      warnings.simplefilter("default")
      for _ in range(3):
        tt_testing.emit_tt_warn(
            warning_categories.PERFORMANCE,
            "Default action performance warning",
        )

    matching = _matching(recorded, "Default action performance warning")
    self.assertLen(matching, 1)
    self.assertIs(matching[0].category, UserWarning)


if __name__ == "__main__":
  absltest.main()
