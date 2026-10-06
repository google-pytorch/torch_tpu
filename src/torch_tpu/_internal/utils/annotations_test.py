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
import types
from typing import Any
from unittest import mock
import warnings

from absl.testing import absltest
import torch
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from torch_tpu._internal.device import _device_module
from torch_tpu._internal.precision.precision_impl import Precision
from torch_tpu._internal.utils import annotations
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

experimental = annotations.experimental
stable = annotations.stable
deprecated = annotations.deprecated
internal_or_experimental = annotations.internal_or_experimental
forbid_in_torch_compile = annotations.forbid_in_torch_compile


def _reset_warn_once(api: Any) -> None:
  """Resets the per-process one-time warning flag of an annotated API.

  Other tests in the same process may have already consumed the one-time
  warning, so reset it to make warning-count assertions order-independent.

  Args:
    api: The annotated function or bound classmethod.
  """
  fn = getattr(api, "__func__", api)
  idx = fn.__code__.co_freevars.index("_warned")
  fn.__closure__[idx].cell_contents = False


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

  def test_experimental_function_traces_under_torch_compile(self):
    """Verifies experimental functions trace without graph breaks."""

    @experimental("Testing experimental feature.")
    def sample_func(x: torch.Tensor) -> torch.Tensor:
      return x * 2

    # `fullgraph=True` raises on any graph break in the warning wrapper.
    compiled = torch.compile(
        lambda x: sample_func(x) + 1, backend="eager", fullgraph=True
    )
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      compiled(torch.ones(3))
      self.assertEmpty([
          item
          for item in w
          if "sample_func is experimental" in str(item.message)
      ])

    # The warning is still emitted on the first eager call.
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      sample_func(torch.ones(3))
      self.assertLen(w, 1)
      self.assertIn("sample_func is experimental", str(w[0].message))

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
        getattr(fn, annotations.TT_API_STAGE, None),
        "Internal"
        if annotations.get_torch_version() >= (2, 13)
        else "Experimental",
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
  @mock.patch(
      "torch_tpu._internal.device._device_ops_backend._get_current_device_id",
      return_value=0,
  )
  @mock.patch.object(torch.tpu, "device_count", return_value=1)
  def test_annotated_real_api_property_default_generators(
      self, _mock_count, _mock_current, _mock_gen
  ):
    """Verifies default_generators emits UserWarning on 1st external access even after internal access."""
    if "default_generators" in _device_module.TpuDeviceModule.__dict__:
      delattr(_device_module.TpuDeviceModule, "default_generators")
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      # Internal access via _get_generator must not emit a warning or cache
      # default_generators onto TpuDeviceModule.__dict__.
      gen = _device_module.TpuDeviceModule._get_generator(  # pylint: disable=protected-access
          torch.device("tpu", 0)
      )
      self.assertEqual(gen, "mock_generator")
      self.assertEmpty(w)
      self.assertNotIn(
          "default_generators", _device_module.TpuDeviceModule.__dict__
      )

      # 1st external access: emits 1 UserWarning and returns default generators
      gens1 = torch.tpu.default_generators
      self.assertEqual(gens1, ("mock_generator",))
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("'default_generators' is experimental", str(w[0].message))

      # 2nd external access: cached on class, emits 0 additional warnings
      gens2 = torch.tpu.default_generators
      self.assertEqual(gens1, gens2)
      self.assertLen(w, 1)

  def test_static_class_attributes_bypass_getattr_and_warnings(self):
    """Verifies accessing static class attributes via torch.tpu does not trigger stage warnings."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      # Accessing static attributes physically defined in _DeviceModule class body
      _ = torch.tpu.precision
      # 0 warnings emitted, proving _DeviceModuleMeta.__getattr__ was bypassed
      self.assertEmpty(w)

  def test_annotated_real_api_enum_class_precision(self):
    """Verifies Precision Enum class emits UserWarning on 1st access and no additional warnings on 2nd access."""
    if "Precision" in _device_module.TpuDeviceModule.__dict__:
      delattr(_device_module.TpuDeviceModule, "Precision")
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

  def test_stage_warning_from_user_getattr(self):
    """Verifies accessing a staged torch.tpu attribute inside a user __getattr__ emits a warning at the user line."""
    if "Precision" in _device_module.TpuDeviceModule.__dict__:
      delattr(_device_module.TpuDeviceModule, "Precision")

    class UserModule(torch.nn.Module):

      def __getattr__(self, name: str) -> Any:
        if name == "tpu_precision":
          return torch.tpu.Precision
        return super().__getattr__(name)

    # Trigger attribute lookup from a simulated `torch.*` frame to ensure user
    # `__getattr__` frames are not skipped onto an outer `torch.*` frame.
    torch_caller = types.FunctionType(
        (lambda m: m.tpu_precision).__code__,
        {"__name__": "torch.nn.modules.module"},
    )
    mod = UserModule()
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      val = torch_caller(mod)
    self.assertIs(val, Precision)
    self.assertLen(w, 1)
    self.assertTrue(issubclass(w[0].category, UserWarning))
    self.assertIn("'Precision' is experimental", str(w[0].message))
    self.assertEqual(w[0].filename, __file__)
    self.assertEqual(
        w[0].lineno, UserModule.__getattr__.__code__.co_firstlineno + 2
    )

  def test_annotated_real_api_property_allow_excess_precision(self):
    """Verifies allow_excess_precision property emits UserWarning on 1st access and no additional warnings on 2nd access."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      # 1st access: emits 1 UserWarning
      _ = torch.tpu.allow_excess_precision
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("allow_excess_precision is experimental", str(w[0].message))

      # 2nd access: emits 0 additional warnings
      _ = torch.tpu.allow_excess_precision
      self.assertLen(w, 1)

  def test_real_api_stream_context_warns_once_without_internal_warnings(self):
    """Verifies repeated `with torch.tpu.stream(s)` warns once, only for `stream`.

    TpuStreamContext must not leak warnings from internal calls (e.g.
    current_stream) to the user.
    """
    _reset_warn_once(torch.tpu.stream)
    _reset_warn_once(torch.tpu.current_stream)
    new_stream = torch.tpu.Stream()

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      for _ in range(3):
        with torch.tpu.stream(new_stream):
          pass

    self.assertLen(w, 1, msg=[str(x.message) for x in w])
    self.assertTrue(issubclass(w[0].category, UserWarning))
    self.assertIn("stream is experimental", str(w[0].message))

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

  def test_attach_stage_metadata_with_func_target(self):
    """Verifies _attach_stage_metadata propagates attributes to target.__func__."""

    class SampleClass:

      @classmethod
      def sample_classmethod(cls) -> str:
        return "sample"

    bound_method = SampleClass.sample_classmethod
    self.assertTrue(hasattr(bound_method, "__func__"))

    annotations._attach_stage_metadata(
        bound_method,
        stage=annotations.Stage.DEPRECATED.value,
        reason="Testing classmethod func attachment.",
        version="2.14",
    )

    self.assertEqual(
        getattr(bound_method, annotations.TT_API_STAGE), "Deprecated"
    )
    self.assertEqual(
        getattr(bound_method.__func__, annotations.TT_API_STAGE), "Deprecated"
    )
    self.assertEqual(
        getattr(bound_method.__func__, annotations.TT_API_STAGE_REASON),
        "Testing classmethod func attachment.",
    )
    self.assertEqual(
        getattr(bound_method.__func__, annotations.TT_API_DEPRECATED_VERSION),
        "2.14",
    )

  def _call_from_module(self, module_name: str, fn: Any, *args: Any) -> Any:
    """Calls fn(*args) from a frame whose module __name__ is module_name."""
    caller = types.FunctionType(
        (lambda f, a: f(*a)).__code__, {"__name__": module_name}
    )
    return caller(fn, args)

  def test_warning_suppressed_for_framework_callers(self):
    """Verifies that framework callers in torch/torch_tpu packages do not trigger warnings."""

    @experimental("Testing experimental suppression.")
    def exp_func(x: int) -> int:
      return x * 2

    # Simulate a framework caller inside torch or torch_tpu
    fake_torch_globals = {"__name__": "torch.serialization"}
    framework_fn = types.FunctionType(
        (lambda fn: fn(5)).__code__, fake_torch_globals
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      res = framework_fn(exp_func)
      self.assertEqual(res, 10)
      self.assertEmpty(w)

      res_tt = self._call_from_module("torch_tpu._internal.device", exp_func, 5)
      self.assertEqual(res_tt, 10)
      self.assertEmpty(w)

      # Now, an external user call SHOULD trigger the warning
      res2 = exp_func(5)
      self.assertEqual(res2, 10)
      self.assertLen(w, 1)
      self.assertIn("exp_func is experimental", str(w[0].message))

  # ---------------------------------------------------------------------------
  # 5. @internal_or_experimental Decorator Tests
  # ---------------------------------------------------------------------------

  def test_internal_or_experimental_is_experimental_when_not_internal(self):
    """Verifies that it's @experimental(experimental_msg) if not internal."""

    @internal_or_experimental(
        internal_when=False,
        experimental_msg="Testing experimental feature.",
        prefer="torch.public_func",
    )
    def sample_func(a: int, b: int) -> int:
      return a + b

    self.assertEqual(
        getattr(sample_func, annotations.TT_API_STAGE), "Experimental"
    )
    self.assertEqual(
        getattr(sample_func, annotations.TT_API_STAGE_REASON),
        "Testing experimental feature.",
    )
    self.assertFalse(hasattr(sample_func, annotations.TT_API_PREFER))
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertEqual(sample_func(1, 2), 3)
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn(
          "sample_func is experimental and subject to change: Testing"
          " experimental feature.",
          str(w[0].message),
      )

  def test_internal_or_experimental_internal_function_metadata(self):
    """Verifies that an internal API has internal stage metadata."""

    @internal_or_experimental(
        internal_when=True,
        experimental_msg="Testing experimental feature.",
        prefer="torch.public_func",
    )
    def sample_func() -> None:
      pass

    self.assertEqual(getattr(sample_func, annotations.TT_API_STAGE), "Internal")
    self.assertEqual(
        getattr(sample_func, annotations.TT_API_STAGE_REASON),
        "Use torch.public_func instead.",
    )
    self.assertEqual(
        getattr(sample_func, annotations.TT_API_PREFER), "torch.public_func"
    )

  def test_internal_function_warns_once_for_user_caller(self):
    """Verifies that calling an internal function from user code warns once."""

    @internal_or_experimental(
        internal_when=True,
        experimental_msg="Testing experimental feature.",
        prefer="torch.public_func",
    )
    def sample_func(x: int) -> int:
      return x * 2

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertEqual(sample_func(1), 2)
      self.assertEqual(sample_func(2), 4)
    self.assertLen(w, 1)
    self.assertTrue(issubclass(w[0].category, UserWarning))
    self.assertEqual(
        str(w[0].message),
        "<TorchTpu/ApiStage/InternalApi/> sample_func is internal to TorchTPU"
        " and may change or be removed without notice. Use torch.public_func"
        " instead.",
    )
    # The warning points at the caller.
    self.assertEqual(w[0].filename, __file__)

  def test_internal_function_without_prefer_warns(self):
    """Verifies the warning of an internal function without `prefer`."""

    @internal_or_experimental(
        internal_when=True, experimental_msg="Testing experimental feature."
    )
    def sample_func() -> None:
      pass

    self.assertEqual(getattr(sample_func, annotations.TT_API_STAGE), "Internal")
    self.assertFalse(hasattr(sample_func, annotations.TT_API_PREFER))
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      sample_func()
    self.assertLen(w, 1)
    self.assertEqual(
        str(w[0].message),
        "<TorchTpu/ApiStage/InternalApi/> sample_func is internal to TorchTPU"
        " and may change or be removed without notice.",
    )

  def test_internal_function_silent_for_framework_callers(self):
    """Verifies that torch and torch_tpu modules can call internal functions."""

    @internal_or_experimental(
        internal_when=True,
        experimental_msg="Testing experimental feature.",
        prefer="torch.public_func",
    )
    def sample_func(x: int) -> int:
      return x * 2

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      for module_name in (
          "torch",
          "torch.accelerator",
          "torch_tpu._internal.x",
      ):
        framework_fn = types.FunctionType(
            (lambda fn: fn(5)).__code__, {"__name__": module_name}
        )
        self.assertEqual(framework_fn(sample_func), 10, msg=module_name)
      self.assertEmpty(w)

    # TorchTPU tests aren't part of the framework.
    test_fn = types.FunctionType(
        (lambda fn: fn(5)).__code__, {"__name__": "torch_tpu.tests.foo_test"}
    )
    with self.assertWarnsRegex(UserWarning, "is internal to TorchTPU"):
      self.assertEqual(test_fn(sample_func), 10)

  def test_internal_class_warns_for_user_caller(self):
    """Verifies that instantiating an internal class from user code warns."""

    @internal_or_experimental(
        internal_when=True,
        experimental_msg="Testing experimental feature.",
        prefer="torch.PublicClass",
    )
    class SampleClass:

      def __init__(self, x: int):
        self.x = x

    self.assertEqual(getattr(SampleClass, annotations.TT_API_STAGE), "Internal")
    self.assertEqual(
        getattr(SampleClass, annotations.TT_API_PREFER), "torch.PublicClass"
    )
    framework_fn = types.FunctionType(
        (lambda cls: cls(3)).__code__, {"__name__": "torch.nn"}
    )
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertEqual(framework_fn(SampleClass).x, 3)
      self.assertEmpty(w)
    with self.assertWarnsRegex(
        UserWarning,
        r"SampleClass is internal to TorchTPU .* Use torch\.PublicClass"
        r" instead\.$",
    ):
      self.assertEqual(SampleClass(1).x, 1)

  def test_internal_or_experimental_validates_arguments(self):
    """Verifies that @internal_or_experimental validates its arguments."""
    for internal_when in (False, True):
      with self.assertRaises(ValueError):
        internal_or_experimental(
            internal_when=internal_when, experimental_msg=" "
        )
      with self.assertRaises(ValueError):
        internal_or_experimental(
            internal_when=internal_when,
            experimental_msg="Testing experimental feature.",
            prefer=" ",
        )

  def test_internal_function_traces_under_torch_compile(self):
    """Verifies internal functions trace without graph breaks."""

    @internal_or_experimental(
        internal_when=True,
        experimental_msg="Testing experimental feature.",
        prefer="torch.public_func",
    )
    def sample_func(x: torch.Tensor) -> torch.Tensor:
      return x * 2

    # `fullgraph=True` raises on any graph break in the warning wrapper.
    compiled = torch.compile(
        lambda x: sample_func(x) + 1, backend="eager", fullgraph=True
    )
    self.assertEqual(compiled(torch.ones(3)).tolist(), [3.0, 3.0, 3.0])

  def test_parse_version(self):
    """Verifies parse_version and get_torch_version parse version tuples."""
    self.assertEqual(annotations.parse_version("2.15.0a0"), (2, 15, 0))
    self.assertEqual(annotations.parse_version("2.12.0"), (2, 12, 0))
    self.assertEqual(
        annotations.parse_version("2.11.0.dev20260101"), (2, 11, 0)
    )
    self.assertEqual(annotations.parse_version("2.11.0+cpu"), (2, 11, 0))
    self.assertIsInstance(annotations.get_torch_version(), tuple)
    self.assertGreaterEqual(len(annotations.get_torch_version()), 2)

  def test_device_module_apis_with_torch_accelerator_equivalents(self):
    """Verifies torch.tpu APIs are internal iff torch.accelerator replaces them."""
    replacements = {
        "current_device": ("torch.accelerator.current_device_index", (2, 11)),
        "device": ("torch.accelerator.device_index", (2, 11)),
        "set_device": ("torch.accelerator.set_device_index", (2, 11)),
        "is_available": ("torch.accelerator.is_available", (2, 11)),
        "device_count": ("torch.accelerator.device_count", (2, 11)),
        "synchronize": ("torch.accelerator.synchronize", (2, 11)),
        "current_stream": ("torch.accelerator.current_stream", (2, 11)),
        "set_stream": ("torch.accelerator.set_stream", (2, 11)),
        "initial_seed": ("torch.accelerator.random.initial_seed", (2, 13)),
        "get_rng_state": ("torch.accelerator.random.get_rng_state", (2, 13)),
    }
    for api, (replacement, version) in replacements.items():
      with self.subTest(api=api):
        fn = getattr(torch.tpu, api)
        if annotations.get_torch_version() >= version:
          self.assertEqual(getattr(fn, annotations.TT_API_STAGE), "Internal")
          self.assertEqual(getattr(fn, annotations.TT_API_PREFER), replacement)
        else:
          self.assertEqual(
              getattr(fn, annotations.TT_API_STAGE), "Experimental"
          )

  def test_real_api_internal_device_count_warns(self):
    """Verifies that calling internal torch.tpu.device_count() warns."""
    if annotations.get_torch_version() < (2, 11):
      self.skipTest("torch.tpu.device_count() is experimental.")
    _reset_warn_once(torch.tpu.device_count)
    with self.assertWarnsRegex(
        UserWarning,
        r"^<TorchTpu/ApiStage/InternalApi/> device_count is internal to"
        r" TorchTPU .* Use torch\.accelerator\.device_count instead\.$",
    ):
      self.assertEqual(
          torch.tpu.device_count(), torch.accelerator.device_count()
      )

  def test_real_api_torch_accelerator_set_stream_works(self):
    """Verifies that torch.accelerator.set_stream works on TPU."""
    new_stream = torch.tpu.Stream()
    default_stream = torch.accelerator.current_stream()
    try:
      torch.accelerator.set_stream(new_stream)
      self.assertEqual(torch.accelerator.current_stream(), new_stream)
    finally:
      torch.accelerator.set_stream(default_stream)

  # ---------------------------------------------------------------------------
  # 6. torch.compile Restriction Tests (@forbid_in_torch_compile)
  # ---------------------------------------------------------------------------

  def test_forbid_in_torch_compile_function_raises_only_when_compiled(self):
    """Verifies a forbidden function raises under TorchDynamo but works eagerly."""

    @forbid_in_torch_compile("__reason__")
    def forbidden(x: torch.Tensor) -> torch.Tensor:
      return x + 1

    utils.assert_close(forbidden(torch.ones(1)), torch.full((1,), 2.0))

    torch._dynamo.reset()
    compiled = torch.compile(lambda x: forbidden(x) + 1, backend="tpu")
    with self.assertRaisesRegex(
        AssertionError, r"Attempt to trace forbidden callable"
    ):
      compiled(torch.ones(1))

  def test_forbid_in_torch_compile_class_raises_only_when_compiled(self):
    """Verifies a forbidden class raises under TorchDynamo but works eagerly."""

    @forbid_in_torch_compile("__reason__")
    class Forbidden:

      def __enter__(self) -> None:
        pass

      def __exit__(self, *args: Any) -> None:
        pass

    def calculate(x: torch.Tensor) -> torch.Tensor:
      with Forbidden():
        return x + 1

    utils.assert_close(calculate(torch.ones(1)), torch.full((1,), 2.0))

    torch._dynamo.reset()
    with self.assertRaisesRegex(
        AssertionError, r"Attempt to trace forbidden callable"
    ):
      torch.compile(calculate, backend="tpu")(torch.ones(1))

  def test_forbid_in_torch_compile_appends_docstring(self):
    """Verifies the reason is appended to the API docstring."""

    @forbid_in_torch_compile("__reason__")
    def forbidden():
      """Summary line.

      Details.
      """

    self.assertIn("__reason__", forbidden.__doc__)
    self.assertIn(".. note::", forbidden.__doc__)
    self.assertIn(
        "Unsupported inside `torch.compile` regions",
        forbidden.__doc__,
    )
    self.assertStartsWith(forbidden.__doc__, "Summary line.\n\nDetails.")

  def test_forbid_in_torch_compile_without_docstring(self):
    """Verifies a docstring is created when the target has none."""

    @forbid_in_torch_compile("__reason__")
    def forbidden():
      pass

    self.assertIn("__reason__", forbidden.__doc__)
    self.assertStartsWith(
        forbidden.__doc__,
        ".. note::\n  Unsupported inside `torch.compile` regions",
    )

  def test_forbid_in_torch_compile_invalid_reason_raises(self):
    """Verifies that an empty reason raises ValueError."""
    with self.assertRaises(ValueError):
      forbid_in_torch_compile("")


if __name__ == "__main__":
  absltest.main()
