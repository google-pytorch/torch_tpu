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
from typing import Any
from unittest import mock
import warnings

from absl.testing import absltest
import torch
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from torch_tpu._internal.precision.precision_impl import Precision
from torch_tpu._internal.utils import annotations
from tests import seed_test_utils


experimental = annotations.experimental
stable = annotations.stable
deprecated = annotations.deprecated


class AnnotationsTest(seed_test_utils.RepeatableTest):

  # ---------------------------------------------------------------------------
  # 1. Infrastructure Tests
  # ---------------------------------------------------------------------------

  def test_stage_enum(self):
    """Verifies that Stage enum values are correctly defined."""
    self.assertEqual(annotations.Stage.INTERNAL, "Internal")
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
    """Verifies Enum class gets decorated and members inherit stage via class lookup."""

    @experimental("Testing experimental enum.")
    class SampleEnum(enum.Enum):
      FOO = 1
      BAR = 2

    # 1. Enum class itself is decorated
    self.assertEqual(
        getattr(SampleEnum, annotations.TT_API_STAGE), "Experimental"
    )
    # 2. Enum members automatically inherit class stage via getattr lookup
    self.assertEqual(
        getattr(SampleEnum.FOO, annotations.TT_API_STAGE), "Experimental"
    )
    self.assertEqual(
        getattr(SampleEnum.BAR, annotations.TT_API_STAGE), "Experimental"
    )
    # 3. Members' instance __dict__ does NOT contain stage attribute (inherited from class)
    self.assertNotIn(annotations.TT_API_STAGE, SampleEnum.FOO.__dict__)
    self.assertNotIn(annotations.TT_API_STAGE, SampleEnum.BAR.__dict__)

  # ---------------------------------------------------------------------------
  # 5. Integration Verification Tests with Real TorchTPU APIs
  # ---------------------------------------------------------------------------

  def test_annotated_real_api_get_amp_supported_dtype(self):
    """Verifies that real API get_amp_supported_dtype is annotated."""
    fn = getattr(torch.tpu, "get_amp_supported_dtype", None)
    self.assertIsNotNone(fn)
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE, None), "Experimental"
    )
    self.assertIn(
        "get_amp_supported_dtype",
        getattr(fn, annotations.TT_API_STAGE_REASON, ""),
    )

  def test_annotated_real_api_initial_seed(self):
    """Verifies that real API initial_seed is annotated."""
    fn = getattr(torch.tpu, "initial_seed", None)
    self.assertIsNotNone(fn)
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE, None), "Experimental"
    )
    self.assertIn(
        "initial_seed",
        getattr(fn, annotations.TT_API_STAGE_REASON, ""),
    )

  def test_annotated_real_api_manual_seed(self):
    """Verifies that real API manual_seed is annotated."""
    fn = getattr(torch.tpu, "manual_seed", None)
    self.assertIsNotNone(fn)
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE, None), "Experimental"
    )
    self.assertIn(
        "manual_seed",
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

  @mock.patch(
      "torch_tpu._internal.device._device_ops_backend.get_default_generator",
      return_value="mock_generator",
  )
  @mock.patch.object(torch.tpu, "current_device", return_value=0)
  @mock.patch.object(torch.tpu, "device_count", return_value=1)
  def test_annotated_real_api_property_default_generators(
      self, _mock_count, _mock_current, _mock_gen
  ):
    """Verifies default_generators emits UserWarning on 1st access and no additional warnings on 2nd access."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      # 1st access: emits 1 UserWarning and returns default generators
      gens1 = torch.tpu.default_generators
      self.assertEqual(gens1, ("mock_generator",))
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("'default_generators' is experimental", str(w[0].message))

      # 2nd access: cached on class, emits 0 additional warnings
      gens2 = torch.tpu.default_generators
      self.assertEqual(gens1, gens2)
      self.assertLen(w, 1)

  def test_static_class_attributes_bypass_getattr_and_warnings(self):
    """Verifies accessing static class attributes defined on torch.tpu does not trigger __getattr__ or warnings."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      # Accessing static attributes physically defined in _DeviceModule class body
      _ = torch.tpu.current_device
      _ = torch.tpu._autocast_enabled
      # 0 warnings emitted, proving __getattr__ was bypassed completely
      self.assertEmpty(w)

  def test_annotated_real_api_enum_class_precision(self):
    """Verifies Precision Enum class emits UserWarning on 1st access and no additional warnings on 2nd access."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      # 1st access: emits 1 UserWarning and returns Precision Enum class
      prec1 = torch.tpu.Precision
      self.assertEqual(prec1, Precision)
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("'Precision' is experimental", str(w[0].message))

      # Verify metadata tags on Enum class and its members (DEFAULT, HIGH, HIGHEST)
      self.assertEqual(
          getattr(prec1, annotations.TT_API_STAGE, None),
          annotations.Stage.EXPERIMENTAL.value,
      )

      # 2nd access: accessing an Enum member off `Precision` hits the class cache, emitting 0 additional warnings
      self.assertEqual(
          getattr(Precision.DEFAULT, annotations.TT_API_STAGE, None),
          annotations.Stage.EXPERIMENTAL.value,
      )
      self.assertEqual(
          getattr(Precision.HIGH, annotations.TT_API_STAGE, None),
          annotations.Stage.EXPERIMENTAL.value,
      )
      self.assertEqual(
          getattr(Precision.HIGHEST, annotations.TT_API_STAGE, None),
          annotations.Stage.EXPERIMENTAL.value,
      )

      self.assertLen(w, 1)

  def test_annotated_real_api_property_allow_excess_precision(self):
    """Verifies allow_excess_precision property emits UserWarning on 1st access and no additional warnings on 2nd access."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      # 1st access: emits 1 UserWarning
      _ = torch.backends.tpu.allow_excess_precision
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("allow_excess_precision is experimental", str(w[0].message))

      # 2nd access: emits 0 additional warnings
      _ = torch.backends.tpu.allow_excess_precision
      self.assertLen(w, 1)

  def test_pep562_module_getattr_resolution_and_suppression(self):
    """Verifies PEP 562 __getattr__ module attribute resolution, warnings, caching, and AttributeError."""
    mock_tt_api_stages = {
        "EXPERIMENTAL_CONST": annotations.ApiStageInfo(
            stage=annotations.Stage.EXPERIMENTAL,
            reason="Experimental constant.",
            value=42,
        ),
        "DEPRECATED_CONST": annotations.ApiStageInfo(
            stage=annotations.Stage.DEPRECATED,
            reason="Use NEW_CONST instead.",
            value="legacy_val",
            version="2.13",
        ),
    }
    mock_module_globals = {}

    def mock_module_getattr(name: str) -> Any:
      if name in mock_module_globals:
        return mock_module_globals[name]
      return annotations._resolve_module_attribute(
          mock_tt_api_stages,
          name,
          "mock_module",
          tt_api_globals=mock_module_globals,
      )

    # 1. Experimental attribute: warning on 1st access, 0 on 2nd access
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      val1 = mock_module_getattr("EXPERIMENTAL_CONST")
      self.assertEqual(val1, 42)
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("'EXPERIMENTAL_CONST' is experimental", str(w[0].message))

      # 2nd access hits mock_module_globals directly
      val2 = mock_module_getattr("EXPERIMENTAL_CONST")
      self.assertEqual(val2, 42)
      self.assertLen(w, 1)

    # 2. Deprecated attribute: DeprecationWarning on 1st access
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      val = mock_module_getattr("DEPRECATED_CONST")
      self.assertEqual(val, "legacy_val")
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, DeprecationWarning))
      self.assertIn(
          "'DEPRECATED_CONST' is deprecated as of TorchTPU 2.13",
          str(w[0].message),
      )

      # 2nd access hits mock_module_globals directly
      val2 = mock_module_getattr("DEPRECATED_CONST")
      self.assertEqual(val2, "legacy_val")
      self.assertLen(w, 1)

    # 3. Unregistered attribute raises AttributeError
    with self.assertRaises(AttributeError):
      mock_module_getattr("UNKNOWN_CONST")


if __name__ == "__main__":
  absltest.main()
