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

"""Unit tests for TorchTPU API lifecycle stage annotations."""

import enum
import warnings

from absl.testing import absltest
from torch_tpu._internal.device._device_module import _DeviceModule
from torch_tpu._internal.precision.precision_impl import Precision
from torch_tpu._internal.utils import annotations

experimental = annotations.experimental
stable = annotations.stable
deprecated = annotations.deprecated


class AnnotationsTest(absltest.TestCase):

  # ---------------------------------------------------------------------------
  # 1. Infrastructure Tests
  # ---------------------------------------------------------------------------

  def test_stage_enum(self):
    """Verifies that Stage enum values are correctly defined."""
    self.assertEqual(annotations.Stage.STABLE, "Stable")
    self.assertEqual(annotations.Stage.EXPERIMENTAL, "Experimental")
    self.assertEqual(annotations.Stage.DEPRECATED, "Deprecated")

  # ---------------------------------------------------------------------------
  # 2. Function / Callable Decorator Tests (@experimental, @stable, @deprecated)
  # ---------------------------------------------------------------------------

  def test_experimental_first_call_triggers_warning(self):
    """Verifies that calling an experimental function triggers a UserWarning."""
    @experimental("Testing experimental feature.")
    def sample_func(a: int, b: int) -> int:
      return a + b

    self.assertEqual(
        getattr(sample_func, annotations.TT_API_STAGE), "Experimental"
    )
    self.assertEqual(
        getattr(sample_func, annotations.TT_API_STAGE_REASON),
        "Testing experimental feature.",
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      sample_func(1, 2)
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("sample_func is experimental", str(w[0].message))

  def test_experimental_subsequent_calls_suppress_warning(self):
    """Verifies that subsequent function calls suppress warnings."""
    @experimental("Testing experimental feature.")
    def sample_func(a: int, b: int) -> int:
      return a + b

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      sample_func(1, 2)
      self.assertLen(w, 1)

      # Second and third calls must NOT trigger additional warnings
      sample_func(2, 3)
      sample_func(3, 4)
      self.assertLen(w, 1)

  def test_experimental_multiple_functions_independent_warnings(self):
    """Verifies that multiple experimental functions trigger warnings independently."""
    @experimental("First experimental feature.")
    def func_one(x: int) -> int:
      return x + 1

    @experimental("Second experimental feature.")
    def func_two(x: int) -> int:
      return x + 2

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      func_one(1)
      self.assertLen(w, 1)
      self.assertIn("func_one is experimental", str(w[0].message))

      # Calling second experimental function must trigger its own warning
      func_two(1)
      self.assertLen(w, 2)
      self.assertIn("func_two is experimental", str(w[1].message))

  def test_stable_metadata_no_warning(self):
    """Verifies that stable functions attach metadata and trigger no warnings."""
    @stable("Production ready.")
    def stable_func(x: int) -> int:
      return x * 2

    self.assertEqual(getattr(stable_func, annotations.TT_API_STAGE), "Stable")
    self.assertEqual(
        getattr(stable_func, annotations.TT_API_STAGE_REASON),
        "Production ready.",
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      stable_func(5)
      self.assertEmpty(w)

  def test_deprecated_metadata_and_warning(self):
    """Verifies that calling a deprecated function triggers a DeprecationWarning."""
    @deprecated(version="2.13", reason="Use new_api() instead.")
    def legacy_func(x: int) -> int:
      return x + 10

    self.assertEqual(
        getattr(legacy_func, annotations.TT_API_STAGE), "Deprecated"
    )
    self.assertEqual(
        getattr(legacy_func, annotations.TT_API_DEPRECATED_VERSION), "2.13"
    )
    self.assertEqual(
        getattr(legacy_func, annotations.TT_API_STAGE_REASON),
        "Use new_api() instead.",
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      legacy_func(5)
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, DeprecationWarning))
      self.assertIn(
          "legacy_func is deprecated as of TorchTPU 2.13", str(w[0].message)
      )

      # Second invocation must NOT trigger warning
      legacy_func(6)
      self.assertLen(w, 1)

  def test_experimental_invalid_reason_raises(self):
    """Verifies that empty or whitespace reason raises ValueError for @experimental."""
    with self.assertRaises(ValueError):

      @experimental("")
      def dummy_one():
        pass

    with self.assertRaises(ValueError):

      @experimental("   ")
      def dummy_two():
        pass

  def test_deprecated_invalid_args_raises(self):
    """Verifies that empty version or reason raises ValueError for @deprecated."""
    with self.assertRaises(ValueError):

      @deprecated(version="", reason="Valid reason")
      def dummy_one():
        pass

    with self.assertRaises(ValueError):

      @deprecated(version="2.13", reason="")
      def dummy_two():
        pass

  # ---------------------------------------------------------------------------
  # 3. Class Decorator Tests (Class & Class Members)
  # ---------------------------------------------------------------------------

  def test_experimental_class_metadata_and_warning(self):
    """Verifies that instantiating an experimental class triggers a UserWarning."""

    @experimental("Testing experimental class.")
    class SampleClass:

      def __init__(self, val: int):
        self.val = val

      def unannotated_method(self) -> int:
        return self.val * 2

    self.assertEqual(
        getattr(SampleClass, annotations.TT_API_STAGE), "Experimental"
    )
    self.assertEqual(
        getattr(SampleClass, annotations.TT_API_STAGE_REASON),
        "Testing experimental class.",
    )
    # Class members are not automatically decorated
    self.assertIsNone(
        getattr(SampleClass.unannotated_method, annotations.TT_API_STAGE, None)
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      SampleClass(100)
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("SampleClass is experimental", str(w[0].message))

      # Subsequent instantiation should suppress warning
      SampleClass(100)
      self.assertLen(w, 1)

  def test_deprecated_class_metadata_and_warning(self):
    """Verifies that instantiating a deprecated class triggers a DeprecationWarning."""

    @deprecated(version="2.13", reason="Use NewClass instead.")
    class LegacyClass:

      def __init__(self):
        pass

    self.assertEqual(
        getattr(LegacyClass, annotations.TT_API_STAGE), "Deprecated"
    )
    self.assertEqual(
        getattr(LegacyClass, annotations.TT_API_DEPRECATED_VERSION), "2.13"
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      LegacyClass()
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, DeprecationWarning))
      self.assertIn(
          "LegacyClass is deprecated as of TorchTPU 2.13", str(w[0].message)
      )

  def test_explicit_class_member_decoration(self):
    """Verifies class-level stage and method-level stages operate independently."""

    @stable("Stable service class.")
    class ServiceClass:

      @experimental("Experimental method.")
      def experimental_method(self) -> str:
        return "experimental"

      @deprecated(version="2.13", reason="Use new_method.")
      def legacy_method(self) -> str:
        return "legacy"

      def unannotated_method(self) -> str:
        return "normal"

    # Class stage is Stable
    self.assertEqual(
        getattr(ServiceClass, annotations.TT_API_STAGE, None), "Stable"
    )
    # Member methods retain their explicit stages independently
    self.assertEqual(
        getattr(
            ServiceClass.experimental_method, annotations.TT_API_STAGE, None
        ),
        "Experimental",
    )
    self.assertEqual(
        getattr(ServiceClass.legacy_method, annotations.TT_API_STAGE, None),
        "Deprecated",
    )
    # Unannotated method does not get automatically decorated
    self.assertIsNone(
        getattr(ServiceClass.unannotated_method, annotations.TT_API_STAGE, None)
    )

  # ---------------------------------------------------------------------------
  # 4. Enum Class & Enum Member Tests
  # ---------------------------------------------------------------------------

  def test_experimental_enum_class_metadata_and_members(self):
    """Verifies Enum class gets decorated, triggers warnings on usage, and members inherit stage via class lookup."""

    @experimental("Testing experimental enum.")
    class SampleEnum(enum.Enum):
      FOO = 1
      BAR = 2

    # 1. Enum class itself is decorated
    self.assertEqual(
        getattr(SampleEnum, annotations.TT_API_STAGE), "Experimental"
    )

    # 2. Accessing members triggers UserWarning once
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      foo = SampleEnum.FOO
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("SampleEnum is experimental", str(w[0].message))

      # Subsequent member access suppresses warning
      _ = SampleEnum.BAR
      self.assertLen(w, 1)

    # 3. Enum members automatically inherit class stage via getattr lookup
    self.assertEqual(getattr(foo, annotations.TT_API_STAGE), "Experimental")

    # 4. Members' instance __dict__ does NOT contain stage attribute (inherited from class)
    self.assertNotIn(annotations.TT_API_STAGE, foo.__dict__)

  def test_deprecated_enum_class_metadata_and_warning(self):
    """Verifies that accessing a deprecated Enum class triggers a DeprecationWarning."""

    @deprecated(version="2.13", reason="Use NewEnum instead.")
    class LegacyEnum(enum.Enum):
      ALPHA = 1
      BETA = 2

    self.assertEqual(
        getattr(LegacyEnum, annotations.TT_API_STAGE), "Deprecated"
    )
    self.assertEqual(
        getattr(LegacyEnum, annotations.TT_API_DEPRECATED_VERSION), "2.13"
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      _ = LegacyEnum.ALPHA
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, DeprecationWarning))
      self.assertIn(
          "LegacyEnum is deprecated as of TorchTPU 2.13", str(w[0].message)
      )

  # ---------------------------------------------------------------------------
  # 5. Integration Verification Tests with Real TorchTPU APIs
  # ---------------------------------------------------------------------------

  def test_annotated_real_api_get_amp_supported_dtype(self):
    """Verifies that real API get_amp_supported_dtype is annotated."""
    fn = getattr(_DeviceModule, "get_amp_supported_dtype", None)
    self.assertIsNotNone(fn)
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE, None), "Experimental"
    )
    self.assertIn(
        "get_amp_supported_dtype",
        getattr(fn, annotations.TT_API_STAGE_REASON, ""),
    )

  def test_annotated_real_enum_class_api_precision(self):
    """Verifies that real Enum Class API Precision is annotated with @experimental."""
    cls = Precision
    self.assertIsNotNone(cls)
    self.assertEqual(
        getattr(cls, annotations.TT_API_STAGE, None), "Experimental"
    )
    self.assertIn(
        "StableHLO precision",
        getattr(cls, annotations.TT_API_STAGE_REASON, ""),
    )


if __name__ == "__main__":
  absltest.main()
