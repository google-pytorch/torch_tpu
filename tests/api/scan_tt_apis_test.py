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

"""Unit tests for scan_tt_apis.py API scanning and stage resolution."""

import __future__
import dataclasses
import importlib
import inspect
import os
import sys
import tempfile
import textwrap
import types
import typing
from unittest import mock

from absl.testing import absltest
import torch_tpu
from torch_tpu._internal.utils import annotations
from tests import seed_test_utils
from tests.api import scan_tt_apis
import typing_extensions

Stage = annotations.Stage


def _public_api(
    name: str, stage: typing.Optional[Stage] = None
) -> scan_tt_apis.ApiEntry:
  return scan_tt_apis.ApiEntry(
      name=name, category="Public Python API", stage=stage, defined_in="runtime"
  )


# A `public_apis.md` with stale generated API lists.
_PUBLIC_APIS_MD_TEMPLATE = """\
# Public APIs

## Stable APIs

Stable text.

<!-- BEGIN GENERATED: Stable APIs -->
<!-- END GENERATED: Stable APIs -->

## Experimental APIs

<!-- BEGIN GENERATED: Experimental APIs -->

*   `torch.tpu.removed_op`

<!-- END GENERATED: Experimental APIs -->

Experimental text.

## Deprecated APIs

<!-- BEGIN GENERATED: Deprecated APIs -->
<!-- END GENERATED: Deprecated APIs -->

## Internal APIs

<!-- BEGIN GENERATED: Internal APIs -->
<!-- END GENERATED: Internal APIs -->
"""

class ScanTtApisTest(seed_test_utils.RepeatableTest):

  def test_is_unsupported_message(self):
    """Verifies detection of unsupported/unimplemented API messages."""
    self.assertTrue(
        scan_tt_apis._is_unsupported_message(
            "TpuPinnedAllocator::empty_cache is not implemented."
        )
    )
    self.assertTrue(
        scan_tt_apis._is_unsupported_message("Graph is NOT SUPPORTED on TPU")
    )
    self.assertTrue(
        scan_tt_apis._is_unsupported_message("Backend doesn't support this")
    )
    self.assertFalse(
        scan_tt_apis._is_unsupported_message("Expected a positive integer.")
    )
    self.assertFalse(scan_tt_apis._is_unsupported_message(""))

  def test_is_excluded_python_api_skips_internal_and_test_packages(self):
    """Verifies that internal-only and test-only packages are excluded."""
    self.assertTrue(scan_tt_apis._is_excluded_python_api("torch_tpu.google"))
    self.assertTrue(
        scan_tt_apis._is_excluded_python_api(
            "torch_tpu.google.tools.scan_tt_apis.main"
        )
    )
    self.assertTrue(
        scan_tt_apis._is_excluded_python_api(
            "torch_tpu.tests.seed_test_utils.RepeatableTest"
        )
    )
    self.assertFalse(
        scan_tt_apis._is_excluded_python_api("torch_tpu.google_public_api")
    )
    self.assertFalse(scan_tt_apis._is_excluded_python_api("torch.tpu.ops"))

  def test_get_python_api_stage_prioritizes_decorator(self):
    """Verifies that the decorator stage is prioritized over the map stage."""

    @annotations.experimental("Experimental feature.")
    def my_func():
      pass

    class FakeModule:
      __tt_api_stages__ = {
          "my_func": types.SimpleNamespace(
              stage=Stage.STABLE,
              reason="In map as stable",
          )
      }

    stage = scan_tt_apis._get_python_api_stage("my_func", my_func, FakeModule)
    self.assertEqual(stage, Stage.EXPERIMENTAL)

  def test_get_python_api_stage_falls_back_to_map(self):
    """Verifies that the map stage is used if no decorator is present."""

    def unannotated_func():
      pass

    class FakeModule:
      __tt_api_stages__ = {
          "unannotated_func": types.SimpleNamespace(
              stage=Stage.EXPERIMENTAL,
              reason="In map as experimental",
          )
      }

    stage = scan_tt_apis._get_python_api_stage(
        "unannotated_func", unannotated_func, FakeModule
    )
    self.assertEqual(stage, Stage.EXPERIMENTAL)

  def test_get_python_api_stage_info_reads_decorator_version(self):
    """Verifies that the deprecation version comes from the decorator."""

    @annotations.deprecated("2.13", "Use something else.")
    def my_func():
      pass

    self.assertEqual(
        scan_tt_apis._get_python_api_stage_info("my_func", my_func, object()),
        (Stage.DEPRECATED, "2.13", None),
    )

  def test_get_python_api_stage_info_reads_map_version(self):
    """Verifies that the deprecation version comes from the stage map."""

    class FakeModule:
      __tt_api_stages__ = {
          "CONSTANT": annotations.ApiStageInfo(
              stage=Stage.DEPRECATED, reason="Old.", version="2.14"
          )
      }

    self.assertEqual(
        scan_tt_apis._get_python_api_stage_info("CONSTANT", 1, FakeModule),
        (Stage.DEPRECATED, "2.14", None),
    )

  def test_get_python_api_stage_resolves_property_decorator(self):
    """Verifies that the stage is resolved for property getters and setters separately."""

    class FakeModule:

      @property
      @annotations.experimental("Getter is experimental")
      def sample_prop(self) -> bool:
        return True

      @sample_prop.setter
      @annotations.stable("Stable setter")
      def sample_prop(self, val: bool) -> None:
        pass

    entries = scan_tt_apis._extract_python_apis([(FakeModule, "fake_pkg")])
    entry_names = {e.name: e.stage for e in entries}

    self.assertIn("fake_pkg.sample_prop getter", entry_names)
    self.assertIn("fake_pkg.sample_prop setter", entry_names)
    self.assertEqual(
        entry_names["fake_pkg.sample_prop getter"], Stage.EXPERIMENTAL
    )
    self.assertEqual(entry_names["fake_pkg.sample_prop setter"], Stage.STABLE)

  def test_extract_python_apis_lists_internal_apis_as_internal(self):
    """Verifies that internal APIs are listed with the Internal stage."""

    class FakeModule:

      @annotations.internal_or_experimental(
          internal_when=True,
          experimental_msg="Experimental function.",
          prefer="fake_pkg.public_func",
      )
      def guarded_func(self) -> None:
        pass

      @annotations.experimental("Experimental function.")
      def public_func(self) -> None:
        pass

      def _private_func(self) -> None:
        pass

      @property
      @annotations.internal_or_experimental(
          internal_when=True, experimental_msg="Experimental getter."
      )
      def guarded_prop(self) -> bool:
        return True

      @guarded_prop.setter
      @annotations.experimental("Experimental setter.")
      def guarded_prop(self, val: bool) -> None:
        pass

    entries = scan_tt_apis._extract_python_apis([(FakeModule, "fake_pkg")])
    entry_dict = {e.name: e.stage for e in entries}

    self.assertEqual(entry_dict.get("fake_pkg.public_func"), Stage.EXPERIMENTAL)
    self.assertEqual(entry_dict.get("fake_pkg.guarded_func"), Stage.INTERNAL)
    self.assertEqual(
        entry_dict.get("fake_pkg.guarded_prop getter"), Stage.INTERNAL
    )
    self.assertEqual(
        entry_dict.get("fake_pkg.guarded_prop setter"), Stage.EXPERIMENTAL
    )
    self.assertNotIn("fake_pkg._private_func", entry_dict)
    prefer_dict = {e.name: e.prefer for e in entries}
    self.assertEqual(
        prefer_dict.get("fake_pkg.guarded_func"), "fake_pkg.public_func"
    )
    self.assertIsNone(prefer_dict.get("fake_pkg.guarded_prop getter"))

  def test_extract_python_apis_scans_metaclass_properties(self):
    """Verifies that properties defined on custom metaclasses are discovered and builtin type attributes are excluded."""

    class FakeMeta(type):
      __module__ = "torch_tpu.fake"

      @property
      @annotations.experimental("Experimental metaclass property")
      def sample_meta_prop(cls) -> bool:
        return True

      @sample_meta_prop.setter
      @annotations.stable("Stable metaclass property setter")
      def sample_meta_prop(cls, val: bool) -> None:
        pass

    class FakeClass(metaclass=FakeMeta):
      pass

    class FakeNormalClass:
      pass

    entries = scan_tt_apis._extract_python_apis([(FakeClass, "fake_pkg")])
    entry_dict = {e.name: e.stage for e in entries}

    self.assertEqual(
        entry_dict.get("fake_pkg.sample_meta_prop getter"), Stage.EXPERIMENTAL
    )
    self.assertEqual(
        entry_dict.get("fake_pkg.sample_meta_prop setter"), Stage.STABLE
    )

    # Verify builtin type attributes (e.g. mro) are not scanned
    normal_entries = scan_tt_apis._extract_python_apis(
        [(FakeNormalClass, "fake_normal_pkg")]
    )
    normal_entry_names = [e.name for e in normal_entries]
    self.assertNotIn("fake_normal_pkg.mro", normal_entry_names)

  def test_get_python_api_stage_falls_back_to_naming_heuristic(self):
    """Verifies that the naming heuristic is used if no decorator or map is present."""

    def _private_func():
      pass

    def torch_tpu_internal_check():
      pass

    class FakeModule:
      pass

    stage1 = scan_tt_apis._get_python_api_stage(
        "_private_func", _private_func, FakeModule
    )
    self.assertEqual(stage1, Stage.INTERNAL)

    stage2 = scan_tt_apis._get_python_api_stage(
        "torch_tpu_internal_check", torch_tpu_internal_check, FakeModule
    )
    self.assertEqual(stage2, Stage.INTERNAL)

  def test_get_python_api_stage_defaults_to_none(self):
    """Verifies that the default stage is None if no decorator, map, or naming heuristic is present."""

    def unannotated_func():
      pass

    class FakeModule:
      pass

    stage = scan_tt_apis._get_python_api_stage(
        "unannotated_func", unannotated_func, FakeModule
    )
    self.assertIsNone(stage)

  def test_extract_python_apis_inspects_factory_return_type_hints(self):
    """Verifies that factory functions with return type annotations inspect their returned class methods without stage inheritance."""

    class FakeTpuEvent:
      __module__ = "torch_tpu.dummy_internal.streams"

      def record(self) -> None:
        pass

      @annotations.stable("Stable method override")
      def query(self) -> bool:
        return True

    class FakeModule:
      __module__ = "torch_tpu.dummy_pkg"

      @annotations.experimental("Experimental factory")
      def Event(self) -> FakeTpuEvent:
        return FakeTpuEvent()

    entries = scan_tt_apis._extract_python_apis(
        [(FakeModule, "torch_tpu.dummy_pkg")]
    )
    entry_dict = {e.name: e.stage for e in entries}

    self.assertIn("torch_tpu.dummy_pkg.Event", entry_dict)
    self.assertEqual(
        entry_dict["torch_tpu.dummy_pkg.Event"], scan_tt_apis.Stage.EXPERIMENTAL
    )

    # Unannotated method remains None (Unknown) to enforce per-method auditing
    self.assertIn("torch_tpu.dummy_pkg.Event.record", entry_dict)
    self.assertIsNone(entry_dict["torch_tpu.dummy_pkg.Event.record"])

    # query overrides with its own Stage.STABLE decorator
    self.assertIn("torch_tpu.dummy_pkg.Event.query", entry_dict)
    self.assertEqual(
        entry_dict["torch_tpu.dummy_pkg.Event.query"], scan_tt_apis.Stage.STABLE
    )

  def test_extract_python_apis_class_and_property_individual_annotations(self):
    """Verifies that methods and properties require explicit stage annotations and do not inherit from parent class."""

    @annotations.experimental("Experimental container class")
    class FakeAnnotatedClass:
      __module__ = "torch_tpu.dummy_pkg"

      def default_method(self) -> None:
        pass

      @annotations.stable("Independently stable method")
      def overridden_method(self) -> None:
        pass

      @property
      def config_prop(self) -> int:
        return 42

    sub_mod = types.ModuleType("torch_tpu.dummy_pkg.submodule")

    def submodule_unannotated_func():
      pass

    submodule_unannotated_func.__module__ = sub_mod.__name__
    sub_mod.submodule_unannotated_func = submodule_unannotated_func

    class FakeTopModule:
      __module__ = "torch_tpu.dummy_pkg"
      AnnotatedClass = FakeAnnotatedClass

    FakeTopModule.submodule = sub_mod

    entries = scan_tt_apis._extract_python_apis(
        [(FakeTopModule, "torch_tpu.dummy_pkg")]
    )
    entry_dict = {e.name: e.stage for e in entries}

    # 1. Class itself is Experimental
    self.assertEqual(
        entry_dict["torch_tpu.dummy_pkg.AnnotatedClass"], Stage.EXPERIMENTAL
    )
    # 2. Unannotated method remains None (Unknown)
    self.assertIn(
        "torch_tpu.dummy_pkg.AnnotatedClass.default_method", entry_dict
    )
    self.assertIsNone(
        entry_dict["torch_tpu.dummy_pkg.AnnotatedClass.default_method"]
    )
    # 3. Method with direct decorator resolves to Stable
    self.assertEqual(
        entry_dict["torch_tpu.dummy_pkg.AnnotatedClass.overridden_method"],
        Stage.STABLE,
    )
    # 4. Unannotated property getter remains None (Unknown)
    self.assertIn(
        "torch_tpu.dummy_pkg.AnnotatedClass.config_prop getter", entry_dict
    )
    self.assertIsNone(
        entry_dict["torch_tpu.dummy_pkg.AnnotatedClass.config_prop getter"]
    )
    # 5. Submodule items remain unannotated (None)
    self.assertIn(
        "torch_tpu.dummy_pkg.submodule.submodule_unannotated_func", entry_dict
    )
    self.assertIsNone(
        entry_dict["torch_tpu.dummy_pkg.submodule.submodule_unannotated_func"]
    )

  def test_extract_python_apis_infers_factory_return_type_from_ast(self):
    """Verifies that un-annotated factory functions infer their returned class via AST return statement analysis, ignoring nested closures and resolving bound methods."""

    class FakeNestedClass:
      __module__ = "torch_tpu.dummy_internal.streams"

      def nested_method(self) -> None:
        pass

    class FakeTpuStream:
      __module__ = "torch_tpu.dummy_internal.streams"

      def synchronize(self) -> None:
        pass

    class FakeModule:
      __module__ = "torch_tpu.dummy_pkg"

      @classmethod
      @annotations.experimental("Experimental stream factory")
      def Stream(cls):
        def _nested_helper():
          return FakeNestedClass()

        return FakeTpuStream()

    # Pass FakeTpuStream and FakeNestedClass into raw unwrapped func globals
    raw_stream = inspect.unwrap(FakeModule.Stream)
    raw_globals = getattr(raw_stream, "__globals__", None) or getattr(
        getattr(raw_stream, "__func__", None), "__globals__", {}
    )
    raw_globals["FakeTpuStream"] = FakeTpuStream
    raw_globals["FakeNestedClass"] = FakeNestedClass

    entries = scan_tt_apis._extract_python_apis(
        [(FakeModule, "torch_tpu.dummy_pkg")]
    )
    entry_dict = {e.name: e.stage for e in entries}

    self.assertIn("torch_tpu.dummy_pkg.Stream", entry_dict)
    self.assertIn("torch_tpu.dummy_pkg.Stream.synchronize", entry_dict)
    self.assertNotIn("torch_tpu.dummy_pkg.Stream.nested_method", entry_dict)

  def test_extract_python_apis_skips_builtin_enum_properties(self):
    """Verifies that built-in 'name' and 'value' properties on enums are ignored."""

    class FakeCppEnum:
      __members__ = {"CONST_A": 1, "CONST_B": 2}
      CONST_A = 1
      CONST_B = 2

      @property
      def name(self) -> str:
        return "CONST_A"

      @property
      def value(self) -> int:
        return 1

    entries = scan_tt_apis._extract_python_apis(
        [(FakeCppEnum, "fake_pkg.FakeCppEnum")]
    )
    entry_names = [e.name for e in entries]

    self.assertIn("fake_pkg.FakeCppEnum.CONST_A", entry_names)
    self.assertIn("fake_pkg.FakeCppEnum.CONST_B", entry_names)
    self.assertNotIn("fake_pkg.FakeCppEnum.name getter", entry_names)
    self.assertNotIn("fake_pkg.FakeCppEnum.value getter", entry_names)

  def test_extract_python_apis_skips_upstream_base_class_properties(self):
    """Verifies that properties/methods inherited from non-TorchTPU upstream base classes are skipped."""

    class UpstreamBaseConfig:
      __module__ = "torch.profiler._internal"

      @property
      def upstream_getter_only(self) -> int:
        return 1

      @property
      def upstream_read_write(self) -> str:
        return "config"

      @upstream_read_write.setter
      def upstream_read_write(self, value: str) -> None:
        pass

    @annotations.experimental("Experimental config class")
    class FakeTpuConfig(UpstreamBaseConfig):
      __module__ = "torch_tpu._internal.profiler.profiler_config"

      def tpu_custom_method(self) -> None:
        pass

    class FakeProfilerModule:
      __module__ = "torch_tpu.dummy_pkg.profiler"
      TpuProfilerConfig = FakeTpuConfig

    entries = scan_tt_apis._extract_python_apis(
        [(FakeProfilerModule, "torch_tpu.dummy_pkg.profiler")]
    )
    entry_names = [e.name for e in entries]
    # The TorchTPU class itself and its directly-defined method are discovered
    self.assertIn("torch_tpu.dummy_pkg.profiler.TpuProfilerConfig", entry_names)
    self.assertIn(
        "torch_tpu.dummy_pkg.profiler.TpuProfilerConfig.tpu_custom_method",
        entry_names,
    )
    # Inherited upstream properties and methods are skipped
    self.assertNotIn(
        "torch_tpu.dummy_pkg.profiler.TpuProfilerConfig.upstream_getter_only"
        " getter",
        entry_names,
    )
    self.assertNotIn(
        "torch_tpu.dummy_pkg.profiler.TpuProfilerConfig.upstream_read_write"
        " getter",
        entry_names,
    )
    self.assertNotIn(
        "torch_tpu.dummy_pkg.profiler.TpuProfilerConfig.upstream_read_write"
        " setter",
        entry_names,
    )

  def test_extract_python_apis_lists_dataclass_fields(self):
    """Verifies that dataclass fields are listed, with or without defaults."""

    @dataclasses.dataclass(frozen=True)
    class FakeProperties:
      __module__ = "torch_tpu.dummy_pkg.fake"

      name: str
      sizes: list[int] = dataclasses.field(default_factory=list)
      count: int = 1
      _private: int = 0

      __tt_api_stages__ = {
          "count": annotations.ApiStageInfo(
              stage=Stage.EXPERIMENTAL,
              reason="count is experimental.",
          ),
      }

    entries = scan_tt_apis._extract_python_apis(
        [(FakeProperties, "fake_pkg.FakeProperties")]
    )
    entry_dict = {e.name: e.stage for e in entries}

    self.assertIn("fake_pkg.FakeProperties.name", entry_dict)
    self.assertIn("fake_pkg.FakeProperties.sizes", entry_dict)
    self.assertEqual(
        entry_dict["fake_pkg.FakeProperties.count"], Stage.EXPERIMENTAL
    )
    self.assertNotIn("fake_pkg.FakeProperties._private", entry_dict)

  def test_import_public_submodules_skips_private_and_excluded_modules(self):
    """Verifies that only public, non-excluded submodules are imported."""
    with tempfile.TemporaryDirectory() as temp_dir:
      files = {
          "fake_tt_pkg/__init__.py": "",
          "fake_tt_pkg/public_mod.py": "VALUE = 1\n",
          "fake_tt_pkg/_private_mod.py": "raise ImportError('private')\n",
          "fake_tt_pkg/sub/__init__.py": "",
          "fake_tt_pkg/sub/leaf.py": "VALUE = 2\n",
          "fake_tt_pkg/excluded/__init__.py": "raise ImportError('excluded')\n",
      }
      for path, content in files.items():
        full_path = os.path.join(temp_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
          f.write(content)
      importlib.invalidate_caches()

      with mock.patch.object(
          sys, "path", [temp_dir, *sys.path]
      ), mock.patch.dict(sys.modules), mock.patch.object(
          scan_tt_apis,
          "_EXCLUDED_PYTHON_API_PREFIXES",
          ("fake_tt_pkg.excluded",),
      ):
        package = importlib.import_module("fake_tt_pkg")
        scan_tt_apis._import_public_submodules(package)

        self.assertEqual(package.public_mod.VALUE, 1)
        self.assertEqual(package.sub.leaf.VALUE, 2)
        self.assertNotIn("fake_tt_pkg._private_mod", sys.modules)
        self.assertNotIn("fake_tt_pkg.excluded", sys.modules)

  def test_extract_python_apis_skips_names_imported_from_other_libraries(self):
    """Verifies that objects imported from other libraries are skipped."""
    fake_module = types.ModuleType("torch_tpu.fake_imports")

    def owned_func():
      pass

    owned_func.__module__ = fake_module.__name__
    fake_module.owned_func = owned_func
    fake_module.CONSTANT = 3
    fake_module.annotations = __future__.annotations
    fake_module.join = os.path.join
    fake_module.Optional = typing.Optional

    entries = scan_tt_apis._extract_python_apis(
        [(fake_module, "torch_tpu.fake_imports")]
    )

    self.assertCountEqual(
        [e.name for e in entries],
        [
            "torch_tpu.fake_imports.CONSTANT",
            "torch_tpu.fake_imports.owned_func",
        ],
    )

  def test_extract_python_apis_lists_module_version_with_map_stage(self):
    """Verifies that a module's `__version__` is listed with its map stage."""

    class FakeVersionedModule(types.ModuleType):

      # Like `torch.tpu`, the module doesn't list `__version__` in `dir()`.
      def __dir__(self):
        return ["other_library"]

    fake_module = FakeVersionedModule("torch_tpu.fake_versioned")
    fake_module.__version__ = "1.2.3"
    fake_module.__tt_api_stages__ = {
        "__version__": annotations.ApiStageInfo(
            stage=Stage.STABLE, reason="The version."
        )
    }
    # The `__version__` of a module from another library isn't listed.
    other_library = types.ModuleType("other_library")
    other_library.__version__ = "4.5.6"
    fake_module.other_library = other_library

    entries = scan_tt_apis._extract_python_apis(
        [(fake_module, "torch_tpu.fake_versioned")]
    )

    self.assertEqual(
        {e.name: e.stage for e in entries},
        {"torch_tpu.fake_versioned.__version__": Stage.STABLE},
    )
    self.assertEmpty(scan_tt_apis.find_module_versions_without_stage(entries))

  def test_find_module_versions_without_stage_flags_unannotated_version(self):
    """Verifies that a module's `__version__` without a stage is flagged."""
    fake_module = types.ModuleType("torch_tpu.fake_versioned")
    fake_module.__version__ = "1.2.3"

    entries = scan_tt_apis._extract_python_apis(
        [(fake_module, "torch_tpu.fake_versioned")]
    )

    # The naming heuristic doesn't make `__version__` internal.
    self.assertEqual(
        {e.name: e.stage for e in entries},
        {"torch_tpu.fake_versioned.__version__": None},
    )
    self.assertEqual(
        scan_tt_apis.find_module_versions_without_stage(entries),
        ["torch_tpu.fake_versioned.__version__"],
    )

  def test_extract_python_apis_lists_members_once_at_home(self):
    """Verifies that members are listed at home and aliases get one entry."""
    home_module = types.ModuleType("torch_tpu.fake_home")
    alias_module = types.ModuleType("torch_tpu.fake_alias")

    class Widget:
      __module__ = home_module.__name__
      __qualname__ = "Widget"

      def spin(self) -> None:
        pass

    def make_widget():
      pass

    make_widget.__module__ = home_module.__name__
    home_module.Widget = Widget
    home_module.make_widget = make_widget
    alias_module.Widget = Widget
    alias_module.fake_home = home_module
    alias_module.make_widget = make_widget

    class FakeRoot:
      pass

    # `fake_alias` sorts first, so the aliases are reached before the home.
    FakeRoot.fake_alias = alias_module
    FakeRoot.fake_home = home_module

    with mock.patch.dict(
        sys.modules,
        {
            home_module.__name__: home_module,
            alias_module.__name__: alias_module,
        },
    ):
      entries = scan_tt_apis._extract_python_apis([(FakeRoot, "torch_tpu")])

    self.assertCountEqual(
        [e.name for e in entries],
        [
            "torch_tpu.fake_alias.Widget",
            "torch_tpu.fake_alias.fake_home",
            "torch_tpu.fake_alias.make_widget",
            "torch_tpu.fake_home.Widget",
            "torch_tpu.fake_home.Widget.spin",
            "torch_tpu.fake_home.make_widget",
        ],
    )

  def test_extract_python_apis_lists_returned_class_at_home(self):
    """Verifies that a class returned by a factory is listed at its home.

    The entry at the home takes the stage there, and an alias keeps its own.
    """
    home_module = types.ModuleType("torch_tpu.fake_home")
    alias_module = types.ModuleType("torch_tpu.fake_alias")
    factory_module = types.ModuleType("torch_tpu.factory")

    class Widget:
      __module__ = home_module.__name__
      __qualname__ = "Widget"

      def spin(self) -> None:
        pass

    def make() -> Widget:
      return Widget()

    make.__module__ = factory_module.__name__
    home_module.Widget = Widget
    home_module.__tt_api_stages__ = {
        "Widget": types.SimpleNamespace(stage=Stage.EXPERIMENTAL)
    }
    alias_module.Widget = Widget
    factory_module.make = make

    class FakeRoot:
      pass

    # `factory` sorts first, so the factory is reached before the home.
    FakeRoot.factory = factory_module
    FakeRoot.fake_alias = alias_module
    FakeRoot.fake_home = home_module

    with mock.patch.dict(
        sys.modules,
        {
            module.__name__: module
            for module in (home_module, alias_module, factory_module)
        },
    ):
      entries = scan_tt_apis._extract_python_apis([(FakeRoot, "torch_tpu")])

    self.assertEqual(
        {e.name: e.stage for e in entries},
        {
            "torch_tpu.factory.make": None,
            "torch_tpu.fake_alias.Widget": None,
            "torch_tpu.fake_home.Widget": Stage.EXPERIMENTAL,
            "torch_tpu.fake_home.Widget.spin": None,
        },
    )

  def test_extract_python_apis_walks_module_without_public_home_once(self):
    """Verifies that a module without a public home is walked once.

    Its members are listed at the first path where it's found.
    """
    internal_module = types.ModuleType("torch_tpu._internal.fake_profiler")

    def configure():
      pass

    configure.__module__ = internal_module.__name__
    internal_module.configure = configure

    class FakeDeviceModule:
      pass

    FakeDeviceModule.profiler = internal_module
    FakeDeviceModule.profiler_alias = internal_module

    entries = scan_tt_apis._extract_python_apis(
        [(FakeDeviceModule, "torch.tpu")]
    )

    self.assertCountEqual(
        [e.name for e in entries],
        ["torch.tpu.profiler.configure", "torch.tpu.profiler_alias"],
    )

  def test_extract_python_apis_shares_walked_objects_across_roots(self):
    """Verifies that an object reached from several roots is walked once."""

    class Widget:
      __module__ = "torch_tpu._internal.fake_widgets"

      def spin(self) -> None:
        pass

    class FakeDeviceModule:
      pass

    FakeDeviceModule.Widget = Widget
    fake_package = types.ModuleType("torch_tpu.fake_pkg")
    fake_package.Widget = Widget

    entries = scan_tt_apis._extract_python_apis([
        (FakeDeviceModule, "torch.tpu"),
        (fake_package, "torch_tpu.fake_pkg"),
    ])

    self.assertCountEqual(
        [e.name for e in entries],
        [
            "torch.tpu.Widget",
            "torch.tpu.Widget.spin",
            "torch_tpu.fake_pkg.Widget",
        ],
    )

  def test_find_duplicate_public_paths_groups_paths_to_the_same_object(self):
    """Verifies that module-level paths to the same object are grouped.

    The entries come from `_extract_python_apis`, which lists the members of a
    module only at its home, so a module counts at that path too. Class members
    don't count.
    """
    home_module = types.ModuleType("torch_tpu.fake_home")
    alias_module = types.ModuleType("torch_tpu.fake_alias")

    def make_widget():
      pass

    class Widget:
      __module__ = home_module.__name__
      __qualname__ = "Widget"

      make = staticmethod(make_widget)

    make_widget.__module__ = home_module.__name__
    home_module.Widget = Widget
    home_module.make_widget = make_widget
    alias_module.Widget = Widget
    alias_module.fake_home = home_module
    alias_module.make_widget = make_widget

    class FakeRoot:
      pass

    FakeRoot.fake_alias = alias_module
    FakeRoot.fake_home = home_module

    with mock.patch.dict(
        sys.modules,
        {module.__name__: module for module in (home_module, alias_module)},
    ):
      apis = scan_tt_apis._extract_python_apis([(FakeRoot, "torch_tpu")])
      duplicates = scan_tt_apis.find_duplicate_public_paths(apis)

    self.assertIn("torch_tpu.fake_home.Widget.make", [api.name for api in apis])
    self.assertEqual(
        duplicates,
        [
            ["torch_tpu.fake_alias.Widget", "torch_tpu.fake_home.Widget"],
            ["torch_tpu.fake_alias.fake_home", "torch_tpu.fake_home"],
            [
                "torch_tpu.fake_alias.make_widget",
                "torch_tpu.fake_home.make_widget",
            ],
        ],
    )

  def test_find_duplicate_public_paths_groups_bound_methods(self):
    """Verifies that bound classmethods exported on a module are grouped.

    `torch.tpu` exports classmethods from `TpuDeviceModule`. Each lookup of a
    classmethod creates a new bound method, so bound methods are grouped by
    their class and function.
    """

    def synchronize(cls) -> None:
      del cls

    def current_device(cls) -> int:
      del cls
      return 0

    class FakeDeviceModule:
      pass

    for method in (synchronize, current_device):
      method.__module__ = "torch_tpu._internal.fake_device"
      setattr(FakeDeviceModule, method.__name__, classmethod(method))

    class FakeSubDeviceModule(FakeDeviceModule):
      pass

    alias_module = types.ModuleType("torch_tpu.fake_alias")
    alias_module.synchronize = FakeDeviceModule.synchronize
    apis = [
        _public_api("torch_tpu.fake_alias.synchronize"),
        _public_api("torch_tpu.fake_device.current_device"),
        _public_api("torch_tpu.fake_device.synchronize"),
        _public_api("torch_tpu.fake_sub_device.synchronize"),
    ]

    with mock.patch.dict(
        sys.modules,
        {
            "torch_tpu.fake_device": FakeDeviceModule,
            "torch_tpu.fake_sub_device": FakeSubDeviceModule,
            alias_module.__name__: alias_module,
        },
    ):
      duplicates = scan_tt_apis.find_duplicate_public_paths(apis)

    self.assertEqual(
        duplicates,
        [[
            "torch_tpu.fake_alias.synchronize",
            "torch_tpu.fake_device.synchronize",
        ]],
    )

  def test_find_duplicate_public_paths_counts_members_of_roots(self):
    """Verifies that the members of a root count even if it isn't a module."""
    home_module = types.ModuleType("torch_tpu.fake_home")

    class Widget:
      __module__ = home_module.__name__

    home_module.Widget = Widget
    apis = [
        _public_api("torch_tpu.fake_backend.Widget"),
        _public_api("torch_tpu.fake_home.Widget"),
    ]

    with mock.patch.object(
        scan_tt_apis, "_PYTHON_API_ROOTS", ("torch_tpu.fake_backend",)
    ), mock.patch.object(
        torch_tpu,
        "fake_backend",
        types.SimpleNamespace(Widget=Widget),
        create=True,
    ), mock.patch.dict(
        sys.modules, {home_module.__name__: home_module}
    ):
      duplicates = scan_tt_apis.find_duplicate_public_paths(apis)

    self.assertEqual(
        duplicates,
        [["torch_tpu.fake_backend.Widget", "torch_tpu.fake_home.Widget"]],
    )

  def test_find_duplicate_public_paths_skips_deprecated_paths(self):
    """Verifies that deprecated paths don't count."""
    home_module = types.ModuleType("torch_tpu.fake_home")
    alias_module = types.ModuleType("torch_tpu.fake_alias")

    class Widget:
      __module__ = home_module.__name__

      def spin(self) -> None:
        pass

    def make_widget():
      pass

    make_widget.__module__ = home_module.__name__
    for module in (home_module, alias_module):
      module.Widget = Widget
      module.make_widget = make_widget
    apis = [
        _public_api("torch_tpu.fake_alias.Widget"),
        _public_api("torch_tpu.fake_alias.make_widget", Stage.DEPRECATED),
        # The members of a deprecated class are listed under its path.
        _public_api("torch_tpu.fake_home.Widget", Stage.DEPRECATED),
        _public_api("torch_tpu.fake_home.Widget.spin"),
        _public_api("torch_tpu.fake_home.make_widget"),
    ]

    with mock.patch.dict(
        sys.modules,
        {module.__name__: module for module in (home_module, alias_module)},
    ):
      duplicates = scan_tt_apis.find_duplicate_public_paths(apis)

    self.assertEqual(duplicates, [])

  def test_find_duplicate_public_paths_skips_property_accessors(self):
    """Verifies that property accessors don't count and aren't called."""
    home_module = types.ModuleType("torch_tpu.fake_home")

    def make_widget():
      pass

    make_widget.__module__ = home_module.__name__
    home_module.make_widget = make_widget
    getter = mock.Mock(return_value=make_widget)

    class FakeDeviceModuleMeta(type):
      widget_factory = property(getter)

    class FakeDeviceModule(metaclass=FakeDeviceModuleMeta):
      pass

    apis = [
        _public_api("torch_tpu.fake_device.widget_factory getter"),
        _public_api("torch_tpu.fake_device.widget_factory setter"),
        _public_api("torch_tpu.fake_home.make_widget"),
    ]

    with mock.patch.dict(
        sys.modules,
        {
            "torch_tpu.fake_device": FakeDeviceModule,
            home_module.__name__: home_module,
        },
    ):
      duplicates = scan_tt_apis.find_duplicate_public_paths(apis)

    self.assertEqual(duplicates, [])
    getter.assert_not_called()

  def test_find_duplicate_public_paths_skips_objects_from_other_libraries(self):
    """Verifies that only paths to TorchTPU objects are grouped."""
    home_module = types.ModuleType("torch_tpu.fake_home")
    alias_module = types.ModuleType("torch_tpu.fake_alias")
    for module in (home_module, alias_module):
      module.CONSTANT = 3
      module.Optional = typing.Optional
      module.join = os.path.join
    apis = [
        _public_api(f"{module.__name__}.{name}")
        for module in (home_module, alias_module)
        for name in ("CONSTANT", "Optional", "join")
    ]

    with mock.patch.dict(
        sys.modules,
        {module.__name__: module for module in (home_module, alias_module)},
    ):
      duplicates = scan_tt_apis.find_duplicate_public_paths(apis)

    self.assertEqual(duplicates, [])

  def test_discover_cpp_source_surfaces_env_vars(self):
    """Master unit test for discovering C++ environment variables and resolving stages via kEnvVarToStage."""
    with tempfile.TemporaryDirectory() as temp_dir:
      common_dir = os.path.join(temp_dir, "common")
      os.makedirs(common_dir)
      env_vars_header = os.path.join(common_dir, "env_vars.h")

      mock_header_content = """
      // 1. Stable env var
      inline constexpr char kStableVar[] = "TORCH_TPU_STABLE_VAR";

      // 2. Internal implementation env var (must be omitted from API list)
      inline constexpr char kInternalImplVar[] = "TORCH_TPU_INTERNAL_IMPL_VAR";

      // 3. Internal API env var
      inline constexpr char kInternalApiVar[] = "TORCH_TPU_INTERNAL_API_VAR";

      // 4. Multi-line declaration with inline comment before string (unknown stage)
      inline constexpr char kMultiLineCommentVar[] = // Inline comment
          "TORCH_TPU_MULTILINE_COMMENT_VAR";

      // 5. Multi-line declaration without comment (unknown stage)
      inline constexpr char kMultiLineVar[] =
          "TORCH_TPU_MULTILINE_VAR";

      // 6. Experimental env var
      inline constexpr char kExpVar[] = "TORCH_TPU_EXP_FEATURE";

      // 7. Deprecated env var
      inline constexpr char kDepVar[] = "TORCH_TPU_DEP_FEATURE";

      // 8. Env var string containing // (must not be truncated by comment stripper)
      inline constexpr char kUrlVar[] = "http://example.com/api//path"; // Trailing comment

      // 9. Env var where key and value are on separate lines in kEnvVarToStage
      inline constexpr char kSeparateLinesVar[] =
          "TORCH_TPU_SEPARATE_LINES_VAR";

      inline constexpr auto kEnvVarToStage =
          MakeConstexprMap<std::string_view, std::optional<SymbolStage>>({
              {kStableVar, SymbolStage::Stable()},
              {kInternalImplVar, SymbolStage::InternalImplementation()},
              {kInternalApiVar, SymbolStage::InternalApi()},
              {kMultiLineCommentVar, std::nullopt},
              {kExpVar, SymbolStage::Experimental()},
              {kDepVar, SymbolStage::Deprecated("0.5")},
              {
                  kSeparateLinesVar,
                  SymbolStage::Stable(),
              },
          });
      """
      with open(env_vars_header, "w", encoding="utf-8") as f:
        f.write(mock_header_content)

      env_vars, _, _ = scan_tt_apis.discover_cpp_source_surfaces(temp_dir)
      env_dict = {entry.name: entry.stage for entry in env_vars}

      self.assertEqual(env_dict.get("TORCH_TPU_STABLE_VAR"), Stage.STABLE)
      self.assertEqual(
          env_dict.get("TORCH_TPU_INTERNAL_API_VAR"), Stage.INTERNAL
      )
      self.assertNotIn("TORCH_TPU_INTERNAL_IMPL_VAR", env_dict)
      self.assertIn("TORCH_TPU_MULTILINE_COMMENT_VAR", env_dict)
      self.assertIsNone(env_dict.get("TORCH_TPU_MULTILINE_COMMENT_VAR"))
      self.assertIn("TORCH_TPU_MULTILINE_VAR", env_dict)
      self.assertIsNone(env_dict.get("TORCH_TPU_MULTILINE_VAR"))
      self.assertEqual(
          env_dict.get("TORCH_TPU_EXP_FEATURE"), Stage.EXPERIMENTAL
      )
      self.assertEqual(env_dict.get("TORCH_TPU_DEP_FEATURE"), Stage.DEPRECATED)
      self.assertIn("http://example.com/api//path", env_dict)
      self.assertIsNone(env_dict.get("http://example.com/api//path"))
      self.assertEqual(
          env_dict.get("TORCH_TPU_SEPARATE_LINES_VAR"), Stage.STABLE
      )
      deprecated_in = {entry.name: entry.deprecated_in for entry in env_vars}
      self.assertEqual(deprecated_in.get("TORCH_TPU_DEP_FEATURE"), "0.5")
      self.assertIsNone(deprecated_in.get("TORCH_TPU_STABLE_VAR"))

  def test_discover_cpp_source_surfaces_absl_flags(self):
    """Master unit test for discovering ABSL C++ flags across all declaration styles and stages."""
    with tempfile.TemporaryDirectory() as temp_dir:
      source_file = os.path.join(temp_dir, "flags.cc")

      mock_cpp_content = """
      // 1. Standard public flag
      ABSL_FLAG(bool, public_flag, false, "Public flag");

      // 2. Internal flag
      ABSL_FLAG(int32_t, torch_tpu_internal_threads, 0, "Internal thread count");

      // 3. Templated type with commas
      ABSL_FLAG(std::map<int, int>, map_flag, {}, "Public flag map");

      // 4. Multi-line flag definition
      ABSL_FLAG(
          std::string,
          multiline_flag,
          "default_val",
          "Multi-line flag"
      );

      // 5. Flag with inline comment
      ABSL_FLAG(int, commented_flag, 10, "Commented flag"); // Inline comment
      """
      with open(source_file, "w", encoding="utf-8") as f:
        f.write(mock_cpp_content)

      _, cpp_flags, _ = scan_tt_apis.discover_cpp_source_surfaces(temp_dir)
      flag_dict = {entry.name: entry.stage for entry in cpp_flags}

      self.assertIn("public_flag", flag_dict)
      self.assertIsNone(flag_dict.get("public_flag"))
      self.assertEqual(
          flag_dict.get("torch_tpu_internal_threads"), Stage.INTERNAL
      )
      self.assertIn("map_flag", flag_dict)
      self.assertIsNone(flag_dict.get("map_flag"))
      self.assertIn("multiline_flag", flag_dict)
      self.assertIsNone(flag_dict.get("multiline_flag"))
      self.assertIn("commented_flag", flag_dict)
      self.assertIsNone(flag_dict.get("commented_flag"))

  def test_discover_custom_ops(self):
    """Discovers custom Ops in C++ TORCH_LIBRARY_IMPL and stage resolution."""
    with tempfile.TemporaryDirectory() as temp_dir:
      kernels_file = os.path.join(temp_dir, "tpu_aten_kernels.cc")
      mock_cpp_content = """
      TORCH_LIBRARY_IMPL(tpu, PrivateUse1, m) {
        ImplExperimental<OpName::kRaggedDot>(m, AtenRaggedDot);
        ImplStable<OpName::kSparseDenseMatmul>(m, AtenSparseDenseMatmul);
        ImplDeprecated<OpName::kLegacyCustomOp>(m, LegacyKernel);
      }
      """
      with open(kernels_file, "w", encoding="utf-8") as f:
        f.write(mock_cpp_content)

      _, _, custom_ops = scan_tt_apis.discover_cpp_source_surfaces(temp_dir)
      op_dict = {entry.name: entry.stage for entry in custom_ops}

      self.assertEqual(
          op_dict.get("torch.ops.tpu.ragged_dot"), Stage.EXPERIMENTAL
      )
      self.assertEqual(
          op_dict.get("torch.ops.tpu.sparse_dense_matmul"), Stage.STABLE
      )
      self.assertEqual(
          op_dict.get("torch.ops.tpu.legacy_custom_op"), Stage.DEPRECATED
      )

  def test_discover_custom_ops_skips_internal_ops(self):
    """Skips underscore-prefixed custom ops, which are internal."""
    with tempfile.TemporaryDirectory() as temp_dir:
      op_names_file = os.path.join(temp_dir, "op_names.h")
      mock_op_names_content = """
      switch (op_name) {
        case OpName::kTorchTpuInternalOp:
          return "_internal_op";
      }
      """
      with open(op_names_file, "w", encoding="utf-8") as f:
        f.write(mock_op_names_content)

      kernels_file = os.path.join(temp_dir, "tpu_aten_kernels.cc")
      mock_cpp_content = """
      TORCH_LIBRARY_IMPL(tpu, PrivateUse1, m) {
        ImplExperimental<OpName::kRaggedDot>(m, AtenRaggedDot);
        ImplInternal<OpName::kTorchTpuInternalOp>(m, InternalKernel);
      }
      """
      with open(kernels_file, "w", encoding="utf-8") as f:
        f.write(mock_cpp_content)

      _, _, custom_ops = scan_tt_apis.discover_cpp_source_surfaces(temp_dir)
      op_names = [entry.name for entry in custom_ops]

      self.assertEqual(op_names, ["torch.ops.tpu.ragged_dot"])

  def test_discover_custom_ops_prioritizes_op_names_mapping(self):
    """Prioritizes explicit string mapping in op_names.h over normalize fallback."""
    with tempfile.TemporaryDirectory() as temp_dir:
      op_names_file = os.path.join(temp_dir, "op_names.h")
      mock_op_names_content = """
      switch (op_name) {
        case OpName::kNonStandardEnumName:
          return "explicit_custom_name";
      }
      """
      with open(op_names_file, "w", encoding="utf-8") as f:
        f.write(mock_op_names_content)

      kernels_file = os.path.join(temp_dir, "tpu_aten_kernels.cc")
      mock_cpp_content = """
      TORCH_LIBRARY_IMPL(tpu, PrivateUse1, m) {
        ImplExperimental<OpName::kNonStandardEnumName>(m, CustomKernel);
      }
      """
      with open(kernels_file, "w", encoding="utf-8") as f:
        f.write(mock_cpp_content)

      _, _, custom_ops = scan_tt_apis.discover_cpp_source_surfaces(temp_dir)
      op_dict = {entry.name: entry.stage for entry in custom_ops}

      self.assertEqual(
          op_dict.get("torch.ops.tpu.explicit_custom_name"), Stage.EXPERIMENTAL
      )
      self.assertNotIn("torch.ops.tpu.non_standard_enum_name", op_dict)

  def test_discover_tpu_allow_excess_precision_apis(self):
    """Verifies that public properties are discovered with expected stages."""

    class FakeTpuMeta(type):
      __module__ = "torch_tpu"

      @property
      @annotations.experimental("experimental property")
      def allow_excess_precision(cls) -> bool:
        return False

      @allow_excess_precision.setter
      @annotations.experimental("experimental property")
      def allow_excess_precision(cls, value: bool):
        pass

    class FakeTpu(metaclass=FakeTpuMeta):
      __module__ = "torch_tpu"

    tpu_apis = scan_tt_apis._extract_python_apis([(FakeTpu, "torch.tpu")])
    tpu_dict = {entry.name: entry for entry in tpu_apis}

    self.assertIn("torch.tpu.allow_excess_precision getter", tpu_dict)
    self.assertIn("torch.tpu.allow_excess_precision setter", tpu_dict)
    self.assertEqual(
        tpu_dict["torch.tpu.allow_excess_precision getter"].stage,
        Stage.EXPERIMENTAL,
    )
    self.assertEqual(
        tpu_dict["torch.tpu.allow_excess_precision setter"].stage,
        Stage.EXPERIMENTAL,
    )

  def test_discover_accelerator_apis_filters_unsupported_by_runtime_probing(
      self,
  ):
    """Verifies runtime probing discovers TPU-implemented torch.accelerator APIs and excludes unsupported ones."""
    import warnings  # pylint: disable=g-import-not-at-top

    fake_acc = types.ModuleType("torch.accelerator")
    fake_acc.__all__ = [
        "Graph",
        "current_accelerator",
        "device_index",
        "get_device_capability",
        "reset_accumulated_memory_stats",
        "set_device_index",
        "set_stream",
    ]

    def current_accelerator(check_available: bool = False):
      del check_available
      return "tpu"

    def set_device_index(device):
      del device

    def set_stream(stream):
      del stream

    class device_index:  # pylint: disable=invalid-name

      def __init__(self, device):
        self.idx = device

      def __enter__(self):
        return None

      def __exit__(self, *exc_info):
        del exc_info
        return None

    class Graph:

      def __init__(self, keep_graph: bool = False):
        del keep_graph
        raise RuntimeError("Graph is not supported on device type: tpu")

    def get_device_capability(device=None):
      del device
      raise RuntimeError("Backend doesn't support getting device capabilities.")

    def reset_accumulated_memory_stats(device_index=None):
      del device_index
      warnings.warn(
          "torch.accelerator.memory.reset_accumulated_memory_stats is not"
          " implemented for TPU.",
          UserWarning,
      )

    fake_acc.current_accelerator = current_accelerator
    fake_acc.set_device_index = set_device_index
    fake_acc.set_stream = set_stream
    fake_acc.device_index = device_index
    fake_acc.Graph = Graph
    fake_acc.get_device_capability = get_device_capability
    fake_acc.reset_accumulated_memory_stats = reset_accumulated_memory_stats

    fake_mem = types.ModuleType("torch.accelerator.memory")
    fake_mem.__all__ = [
        "memory_allocated",
        "reset_accumulated_memory_stats",
    ]

    def memory_allocated(device_index=None):
      del device_index
      return 0

    fake_mem.memory_allocated = memory_allocated
    fake_mem.reset_accumulated_memory_stats = reset_accumulated_memory_stats
    fake_acc.memory = fake_mem

    entries = scan_tt_apis.discover_accelerator_apis(fake_acc)
    entry_map = {entry.name: entry for entry in entries}

    self.assertIn("torch.accelerator.current_accelerator", entry_map)
    self.assertIn("torch.accelerator.device_index", entry_map)
    self.assertIn("torch.accelerator.set_device_index", entry_map)
    self.assertIn("torch.accelerator.set_stream", entry_map)
    self.assertIn("torch.accelerator.memory.memory_allocated", entry_map)

    for entry in entries:
      self.assertEqual(entry.category, "Public Python API")
      self.assertEqual(entry.stage, Stage.STABLE)
      self.assertEqual(entry.defined_in, "runtime")

    # Verify unsupported APIs are excluded
    self.assertNotIn("torch.accelerator.Graph", entry_map)
    self.assertNotIn("torch.accelerator.get_device_capability", entry_map)
    self.assertNotIn(
        "torch.accelerator.reset_accumulated_memory_stats", entry_map
    )
    self.assertNotIn(
        "torch.accelerator.memory.reset_accumulated_memory_stats", entry_map
    )

  def test_discover_accelerator_apis_marks_apis_deprecated_by_pytorch(self):
    """Verifies that APIs that PyTorch deprecated are marked deprecated."""
    fake_acc = types.ModuleType("torch.accelerator")
    fake_acc.__all__ = ["current_device_idx", "current_device_index"]

    def current_device_index():
      return 0

    # This is how PyTorch deprecates `torch.accelerator` APIs.
    fake_acc.current_device_index = current_device_index
    fake_acc.current_device_idx = typing_extensions.deprecated(
        "Use `current_device_index` instead.", category=FutureWarning
    )(current_device_index)

    entries = scan_tt_apis.discover_accelerator_apis(fake_acc)

    self.assertEqual(
        entries,
        [
            scan_tt_apis.ApiEntry(
                name="torch.accelerator.current_device_idx",
                category="Public Python API",
                stage=Stage.DEPRECATED,
                defined_in="runtime",
            ),
            scan_tt_apis.ApiEntry(
                name="torch.accelerator.current_device_index",
                category="Public Python API",
                stage=Stage.STABLE,
                defined_in="runtime",
            ),
        ],
    )

  def test_format_tt_apis_yaml_lists_preferred_alternatives(self):
    """Verifies that tt_apis.yaml lists the alternatives to internal APIs."""
    items = [
        scan_tt_apis.ApiEntry(
            name="torch.tpu.synchronize",
            category="Public Python API",
            stage=annotations.Stage.INTERNAL,
            defined_in="runtime",
            prefer="torch.accelerator.synchronize",
        ),
    ]
    yaml_text = scan_tt_apis.format_tt_apis_yaml(items)
    self.assertIn(
        "torch.tpu.synchronize: {category: Public Python API, defined_in:"
        " runtime, prefer: torch.accelerator.synchronize, stage: Internal}",
        yaml_text,
    )

  def test_format_tt_apis_yaml_lists_duplicated_exports(self):
    """Verifies that each duplicate group is keyed by its public path."""
    items = [
        scan_tt_apis.ApiEntry(
            name="torch.tpu.current_device",
            category="Public Python API",
            stage=annotations.Stage.EXPERIMENTAL,
            defined_in="runtime",
        ),
    ]
    yaml_text = scan_tt_apis.format_tt_apis_yaml(
        items, [["torch_tpu.foo.op", "torch.tpu.op", "torch_tpu.op"]]
    )
    self.assertTrue(
        yaml_text.endswith(
            "\n\n# --- Duplicated Export ---\n"
            "torch.tpu.op: {also_exported_as: [torch_tpu.foo.op,"
            " torch_tpu.op], category: Duplicated Export}\n"
        ),
        yaml_text,
    )

  def test_find_duplicated_exports_keeps_groups_with_public_paths(self):
    """Verifies that groups of only legacy paths are not reported."""
    apis = [
        scan_tt_apis.ApiEntry(
            name="torch.tpu.op",
            category="Public Python API",
            stage=annotations.Stage.EXPERIMENTAL,
            defined_in="runtime",
        ),
        scan_tt_apis.ApiEntry(
            name="LIBTPU_INIT_ARGS",
            category="Environment Variable",
            stage=annotations.Stage.STABLE,
            defined_in="csrc/common/env_vars.h",
        ),
    ]
    with mock.patch.object(
        scan_tt_apis,
        "find_duplicate_public_paths",
        autospec=True,
        return_value=[
            ["torch.tpu.op", "torch_tpu.op"],
            ["torch_tpu.a", "torch_tpu.b"],
        ],
    ) as find_duplicate_public_paths:
      groups = scan_tt_apis.find_duplicated_exports(apis)

    self.assertEqual(groups, [["torch.tpu.op", "torch_tpu.op"]])
    find_duplicate_public_paths.assert_called_once_with(apis[:1])

  def test_find_public_instance_attributes_of_class_finds_assignments(self):
    """Verifies that public attributes assigned on `self` are found."""

    class FakeBase:
      __module__ = "torch_tpu.dummy_pkg.base"

      def __init__(self):
        self.base_attr = 1

    class FakeClass(FakeBase):
      __module__ = "torch_tpu.dummy_pkg.fake"

      def __init__(self):
        super().__init__()
        self.plain = 1
        self.annotated: int = 2
        self.first, (self.second, *self.rest) = 1, (2, 3)
        self._private = 4
        self.read_write = 5

      def reset(self):
        self.plain += 1
        self.later = 6

        def callback():
          self.from_closure = 7

        callback()

      @property
      def read_write(self) -> int:
        return self._private

      @read_write.setter
      def read_write(self, value: int) -> None:
        self._private = value

    found = scan_tt_apis.find_public_instance_attributes_of_class(FakeClass)

    self.assertCountEqual(
        found,
        [
            "base_attr",
            "plain",
            "annotated",
            "first",
            "second",
            "rest",
            "later",
            "from_closure",
        ],
    )
    self.assertRegex(found["plain"], r"scan_tt_apis_test\.py:\d+$")

  def test_find_public_instance_attributes_of_class_skips_exempt_attributes(
      self,
  ):
    """Verifies that exempt assignments are not reported."""

    class UpstreamBase:
      __module__ = "torch.optim.optimizer"

      def __init__(self):
        self.upstream_attr = 1

    class FakeClass(UpstreamBase):
      __module__ = "torch_tpu.dummy_pkg.fake"

      @staticmethod
      def configure_instance(instance):
        instance.static_attr = 1

      @classmethod
      def configure_class(cls):
        cls.class_attr = 1

    @dataclasses.dataclass
    class FakeGroup:
      __module__ = "torch_tpu.dummy_pkg.fake"

      params: dict[str, int]

      def __post_init__(self):
        self.params = dict(self.params)

    self.assertEqual(
        scan_tt_apis.find_public_instance_attributes_of_class(FakeClass), {}
    )
    self.assertEqual(
        scan_tt_apis.find_public_instance_attributes_of_class(FakeGroup), {}
    )

  def test_find_public_instance_attributes_reports_api_paths(self):
    """Verifies that attributes are reported under the public class path."""

    class FakeClass:
      __module__ = "torch_tpu.dummy_instance_attrs"

      def __init__(self):
        self.leaked = 1

      def method(self) -> None:
        pass

    module = types.ModuleType("torch_tpu.dummy_instance_attrs")
    module.FakeClass = FakeClass
    with mock.patch.dict(sys.modules, {module.__name__: module}):
      found = scan_tt_apis.find_public_instance_attributes([
          _public_api("torch_tpu.dummy_instance_attrs.FakeClass"),
          _public_api("torch_tpu.dummy_instance_attrs.FakeClass.method"),
      ])

    self.assertEqual(
        [name for name, _ in found],
        ["torch_tpu.dummy_instance_attrs.FakeClass.leaked"],
    )

  def test_update_public_apis_md_lists_public_apis_by_stage(self):
    """Verifies that only the generated API lists of public_apis.md change."""
    items = [
        _public_api("torch.tpu.op", Stage.EXPERIMENTAL),
        _public_api("torch.tpu.mode getter", Stage.STABLE),
        _public_api("torch.tpu.old_op", Stage.DEPRECATED),
        dataclasses.replace(
            _public_api(
                "torch.tpu.get_device_properties.total_memory getter",
                Stage.DEPRECATED,
            ),
            deprecated_in="0.2.1",
        ),
        _public_api("torch.tpu.unknown_op"),
        _public_api("torch.tpu.internal_op", Stage.INTERNAL),
        dataclasses.replace(
            _public_api("torch.tpu.replaced_op", Stage.INTERNAL),
            prefer="torch.accelerator.op",
        ),
        dataclasses.replace(
            _public_api("torch.tpu.long_replaced_op", Stage.INTERNAL),
            prefer="torch.accelerator.some_very_long_module_name.op",
        ),
        _public_api("torch_tpu._private_op", Stage.INTERNAL),
        scan_tt_apis.ApiEntry(
            name="TORCH_TPU_INTERNAL_VAR",
            category="Environment Variable",
            stage=Stage.INTERNAL,
            defined_in="common/env_vars.h",
        ),
        _public_api("torch_tpu.ops.legacy_op", Stage.EXPERIMENTAL),
        scan_tt_apis.ApiEntry(
            name="TORCH_TPU_OLD_VAR",
            category="Environment Variable",
            stage=Stage.DEPRECATED,
            defined_in="common/env_vars.h",
            deprecated_in="2.13",
        ),
        scan_tt_apis.ApiEntry(
            name="torch.ops.tpu.my_op",
            category="Custom Op",
            stage=Stage.EXPERIMENTAL,
            defined_in="eager/tpu_aten_kernels.cc",
        ),
        scan_tt_apis.ApiEntry(
            name="torch.accelerator.synchronize",
            category="Public Python API",
            stage=Stage.STABLE,
            defined_in="runtime",
        ),
        scan_tt_apis.ApiEntry(
            name="torch.accelerator.current_device_idx",
            category="Public Python API",
            stage=Stage.DEPRECATED,
            defined_in="runtime",
        ),
    ]
    items[2] = dataclasses.replace(items[2], deprecated_in="2.14")

    md_text = scan_tt_apis.update_public_apis_md(
        _PUBLIC_APIS_MD_TEMPLATE, items
    )

    self.assertEqual(
        md_text,
        textwrap.dedent("""\
            # Public APIs

            ## Stable APIs

            Stable text.

            <!-- BEGIN GENERATED: Stable APIs -->

            ### Python APIs

            *   `torch.accelerator.synchronize`
            *   `torch.tpu.mode` (getter)

            <!-- END GENERATED: Stable APIs -->

            ## Experimental APIs

            <!-- BEGIN GENERATED: Experimental APIs -->

            ### Python APIs

            *   `torch.tpu.op`

            <!-- END GENERATED: Experimental APIs -->

            Experimental text.

            ## Deprecated APIs

            <!-- BEGIN GENERATED: Deprecated APIs -->

            ### Python APIs

            *   `torch.accelerator.current_device_idx` (deprecated by PyTorch)
            *   `torch.tpu.get_device_properties.total_memory` (getter) (deprecated in
                TorchTPU 0.2.1)
            *   `torch.tpu.old_op` (deprecated in TorchTPU 2.14)

            ### Environment Variables

            *   `TORCH_TPU_OLD_VAR` (deprecated in TorchTPU 2.13)

            <!-- END GENERATED: Deprecated APIs -->

            ## Internal APIs

            <!-- BEGIN GENERATED: Internal APIs -->

            *   `torch.tpu.internal_op`
            *   `torch.tpu.long_replaced_op`: use
                `torch.accelerator.some_very_long_module_name.op` instead
            *   `torch.tpu.replaced_op`: use `torch.accelerator.op` instead

            <!-- END GENERATED: Internal APIs -->
            """),
    )

  def test_update_public_apis_md_is_idempotent(self):
    """Verifies that updating an up-to-date public_apis.md doesn't change it."""
    items = [_public_api("torch.tpu.op", Stage.EXPERIMENTAL)]
    md_text = scan_tt_apis.update_public_apis_md(
        _PUBLIC_APIS_MD_TEMPLATE, items
    )

    self.assertEqual(
        scan_tt_apis.update_public_apis_md(md_text, items), md_text
    )

  def test_update_public_apis_md_notes_empty_stages(self):
    """Verifies that a stage without APIs says so."""
    md_text = scan_tt_apis.update_public_apis_md(
        _PUBLIC_APIS_MD_TEMPLATE,
        [_public_api("torch.tpu.op", Stage.EXPERIMENTAL)],
    )

    self.assertIn(
        "<!-- BEGIN GENERATED: Stable APIs -->\n\n"
        "There are no stable APIs at this time.\n\n"
        "<!-- END GENERATED: Stable APIs -->",
        md_text,
    )
    self.assertIn(
        "<!-- BEGIN GENERATED: Deprecated APIs -->\n\n"
        "There are no deprecated APIs at this time.\n\n"
        "<!-- END GENERATED: Deprecated APIs -->",
        md_text,
    )
    self.assertIn(
        "<!-- BEGIN GENERATED: Internal APIs -->\n\n"
        "There are no internal APIs in public namespaces at this time.\n\n"
        "<!-- END GENERATED: Internal APIs -->",
        md_text,
    )

  def test_update_public_apis_md_requires_deprecation_version(self):
    """Verifies that a deprecated API must have a known deprecation version."""
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Test tooling error.
        ValueError, "torch.tpu.old_op is deprecated"
    ):
      scan_tt_apis.update_public_apis_md(
          _PUBLIC_APIS_MD_TEMPLATE,
          [_public_api("torch.tpu.old_op", Stage.DEPRECATED)],
      )

  def test_update_public_apis_md_requires_generated_lists(self):
    """Verifies that each stage must have exactly one generated API list."""
    md_text = _PUBLIC_APIS_MD_TEMPLATE.replace(
        "<!-- END GENERATED: Experimental APIs -->\n", ""
    )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Test tooling error.
        ValueError, "BEGIN GENERATED: Experimental APIs.*found 0"
    ):
      scan_tt_apis.update_public_apis_md(md_text, [])


if __name__ == "__main__":
  absltest.main()
