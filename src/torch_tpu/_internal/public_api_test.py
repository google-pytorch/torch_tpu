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

"""Unit tests for torch_tpu._internal.public_api."""

from collections.abc import Mapping
import importlib
import inspect
import runpy
import sys
import types
from typing import Any
from unittest import mock
import warnings

from absl.testing import absltest
from torch_tpu._internal import public_api
from torch_tpu._internal.utils import annotations


class _FakeMeta(type):
  """Fake metaclass mimicking _DeviceModuleMeta."""

  _stripes: int = 4

  @property
  def window_num_stripes(cls) -> int:
    return _FakeMeta._stripes

  @window_num_stripes.setter
  def window_num_stripes(cls, value: int) -> None:
    _FakeMeta._stripes = value

  def __getattr__(cls, name: str) -> Any:
    stages_map = getattr(cls, "__tt_api_stages__", {})
    val = annotations._resolve_module_attribute(
        stages_map, name, "fake_root.tpu", tt_api_globals=cls
    )
    if hasattr(val, "__get__"):
      return val.__get__(None, cls)
    return val


def _make_fake_impl_module() -> types.ModuleType:
  mod = types.ModuleType("fake_impl")
  stages = {
      "ExpConst": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="ExpConst is experimental.",
          value="exp_value",
      ),
      "__version__": annotations.ApiStageInfo(
          stage=annotations.Stage.STABLE,
          reason="The version of the fake package.",
          value="1.2.3",
      ),
  }

  class FakeDeviceClass(metaclass=_FakeMeta):
    __tt_api_stages__ = stages
    _private_val = "initial_private"

    @classmethod
    def current_device(cls) -> int:
      return 0

    @classmethod
    def _private_hook(cls) -> str:
      return cls._private_val

  def exported_fn() -> str:
    return "ok"

  class ExportedClass:
    pass

  mod.__tt_api_stages__ = stages  # type: ignore[attr-defined]
  mod._FakeMeta = _FakeMeta  # type: ignore[attr-defined]
  mod.FakeDeviceClass = FakeDeviceClass  # type: ignore[attr-defined]
  mod.exported_fn = exported_fn  # type: ignore[attr-defined]
  mod.ExportedClass = ExportedClass  # type: ignore[attr-defined]
  return mod


class PublicApiTest(absltest.TestCase):  # ABSLTEST_OK=pure Python; no RNG

  def setUp(self) -> None:
    super().setUp()
    _FakeMeta._stripes = 4
    self.root_mod = types.ModuleType("fake_root")
    self.impl_mod = _make_fake_impl_module()
    self.enter_context(
        mock.patch.dict(
            sys.modules,
            {
                "fake_root": self.root_mod,
                "fake_impl": self.impl_mod,
            },
        )
    )
    self.enter_context(mock.patch.dict(public_api._FINDER._loaders))

  def _sample_table(self) -> Mapping[str, public_api.ModuleSpec]:
    return {
        "fake_root.tpu": public_api.ModuleSpec(
            exports={
                "ExpConst": "fake_impl:FakeDeviceClass.ExpConst",
                "__version__": "fake_impl:FakeDeviceClass.__version__",
                "current_device": "fake_impl:FakeDeviceClass.current_device",
                "window_num_stripes": public_api.Property(
                    "fake_impl:_FakeMeta.window_num_stripes"
                ),
            },
            private_fallback="fake_impl:FakeDeviceClass",
            stages="fake_impl:__tt_api_stages__",
            doc="Fake TPU module.",
        ),
        "fake_root.tpu.distributed": public_api.ModuleSpec(
            exports={"exported_fn": "fake_impl:exported_fn"},
            main=annotations.__name__,
        ),
        "fake_root.tpu.profiler": public_api.ModuleSpec(
            exports={"ExportedClass": "fake_impl:ExportedClass"}
        ),
    }

  def test_install_is_lazy(self) -> None:
    lazy_mod = types.ModuleType("fake_lazy_impl")
    lazy_mod.val = 123  # type: ignore[attr-defined]
    table = {
        "fake_root.tpu": public_api.ModuleSpec(
            exports={"val": "fake_lazy_impl:val"}
        ),
        "fake_root.tpu.child": public_api.ModuleSpec(
            exports={"val": "fake_lazy_impl:val"}
        ),
    }
    real_import = importlib.import_module
    with mock.patch.object(
        importlib, "import_module", wraps=real_import
    ) as spy:
      public_api.install(table)
      self.assertNotIn("fake_root.tpu.child", sys.modules)
      tpu_mod = sys.modules["fake_root.tpu"]
      _ = dir(tpu_mod)
      spy.assert_not_called()

      with mock.patch.dict(sys.modules, {"fake_lazy_impl": lazy_mod}):
        self.assertEqual(tpu_mod.val, 123)
      spy.assert_called_once_with("fake_lazy_impl")

  def test_cached_attr_imported_once_and_stored_in_dict(self) -> None:
    public_api.install(self._sample_table())
    prof = self.root_mod.tpu.profiler
    self.assertNotIn("ExportedClass", vars(prof))
    real_import = importlib.import_module
    with mock.patch.object(
        importlib, "import_module", wraps=real_import
    ) as spy:
      self.assertTrue(hasattr(prof, "ExportedClass"))
      cls1 = prof.ExportedClass
      self.assertIs(cls1, self.impl_mod.ExportedClass)
      self.assertIs(prof.ExportedClass, cls1)
      spy.assert_called_once_with("fake_impl")
    self.assertIn("ExportedClass", vars(prof))

  def test_module_attr_caching_and_patching(self) -> None:
    public_api.install(self._sample_table())
    tpu_mod = sys.modules["fake_root.tpu"]
    self.assertEqual(tpu_mod.current_device(), 0)
    self.assertIn("current_device", tpu_mod.__dict__)
    with mock.patch.object(tpu_mod, "current_device", return_value=7):
      self.assertEqual(tpu_mod.current_device(), 7)
    self.assertEqual(tpu_mod.current_device(), 0)

  def test_property_get_set_and_descriptor_identity(self) -> None:
    public_api.install(self._sample_table())
    tpu_mod: Any = sys.modules["fake_root.tpu"]
    prop = inspect.getattr_static(type(tpu_mod), "window_num_stripes")
    self.assertIs(prop, _FakeMeta.window_num_stripes)
    self.assertEqual(tpu_mod.window_num_stripes, 4)
    tpu_mod.window_num_stripes = 16
    self.assertEqual(tpu_mod.window_num_stripes, 16)
    self.assertEqual(_FakeMeta._stripes, 16)

  def test_all_and_dir_sorted_with_children_and_no_private_names(self) -> None:
    public_api.install(self._sample_table())
    tpu_mod = sys.modules["fake_root.tpu"]
    expected = [
        "ExpConst",
        "current_device",
        "distributed",
        "profiler",
        "window_num_stripes",
    ]
    self.assertEqual(tpu_mod.__all__, expected)
    self.assertEqual(dir(tpu_mod), expected)
    self.assertEqual(
        dir(tpu_mod.distributed),
        ["exported_fn"],
    )

  def test_dunder_export_resolves_but_is_not_listed(self) -> None:
    public_api.install(self._sample_table())
    tpu_mod = sys.modules["fake_root.tpu"]
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertEqual(tpu_mod.__version__, "1.2.3")
      self.assertEmpty(w)
    self.assertNotIn("__version__", tpu_mod.__all__)
    self.assertNotIn("__version__", dir(tpu_mod))

  def test_private_fallback_and_attribute_errors(self) -> None:
    public_api.install(self._sample_table())
    tpu_mod = sys.modules["fake_root.tpu"]
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertEqual(tpu_mod._private_hook(), "initial_private")
      self.impl_mod.FakeDeviceClass._private_val = "updated_private"
      self.assertEqual(tpu_mod._private_hook(), "updated_private")
      self.assertEmpty(w)

      for bad_name in (
          "unknown_attr",
          "_unknown_private",
          "__mro__",
          "__ unknown__",
      ):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=module getattr
            AttributeError,
            f"module 'fake_root.tpu' has no attribute '{bad_name}'",
        ):
          getattr(tpu_mod, bad_name)
      self.assertEmpty(w)

  def test_imports_and_module_metadata(self) -> None:
    public_api.install(self._sample_table())
    tpu_mod: Any = sys.modules["fake_root.tpu"]
    self.assertIs(self.root_mod.tpu, tpu_mod)
    self.assertIsInstance(tpu_mod, types.ModuleType)
    self.assertEqual(tpu_mod.__name__, "fake_root.tpu")
    self.assertEqual(tpu_mod.__package__, "fake_root.tpu")
    self.assertEqual(tpu_mod.__path__, [])
    self.assertIsNotNone(tpu_mod.__spec__)
    self.assertIs(tpu_mod.__tt_api_stages__, self.impl_mod.__tt_api_stages__)

    self.assertNotIn("fake_root.tpu.distributed", sys.modules)
    get_module_details = getattr(runpy, "_get_module_details")
    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      _, mod_spec, code = get_module_details("fake_root.tpu.distributed")
    self.assertEqual(mod_spec.name, "fake_root.tpu.distributed")
    self.assertIsInstance(code, types.CodeType)

    dist_mod = importlib.import_module("fake_root.tpu.distributed")
    self.assertIs(dist_mod, self.root_mod.tpu.distributed)
    self.assertEqual(dist_mod.__package__, "fake_root.tpu")
    self.assertIsNotNone(dist_mod.__spec__)
    self.assertIsNotNone(dist_mod.__spec__.loader)
    self.assertTrue(
        dist_mod.__spec__.loader.get_filename(
            "fake_root.tpu.distributed"
        ).endswith("annotations.py")
    )
    self.assertIsNotNone(
        dist_mod.__spec__.loader.get_source("fake_root.tpu.distributed")
    )
    self.assertEqual(dist_mod.exported_fn(), "ok")

    with self.assertRaises(ModuleNotFoundError):  # ASSERT_RAISES_OK=import
      importlib.import_module("fake_root.tpu.non_existent")

  def test_missing_parent_raises_value_error(self) -> None:
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=parent module check
        ValueError, "Parent module 'fake_root.missing'"
    ):
      public_api.install({"fake_root.missing.child": public_api.ModuleSpec()})

  def test_install_is_idempotent(self) -> None:
    table = self._sample_table()
    public_api.install(table)
    tpu_before = sys.modules["fake_root.tpu"]
    prof_before = tpu_before.profiler
    public_api.install(table)
    self.assertIs(sys.modules["fake_root.tpu"], tpu_before)
    self.assertIs(sys.modules["fake_root.tpu.profiler"], prof_before)
    self.assertIs(tpu_before.profiler, prof_before)

  def test_stage_warning_fires_once_and_points_to_caller(self) -> None:
    public_api.install(self._sample_table())
    tpu_mod = sys.modules["fake_root.tpu"]
    internal_fn = types.FunctionType(
        (lambda mod: mod.ExpConst).__code__,
        {"__name__": "torch_tpu._internal.some_helper"},
    )
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertEqual(internal_fn(tpu_mod), "exp_value")
      self.assertEmpty(w)
      self.assertNotIn("ExpConst", tpu_mod.__dict__)
      self.assertNotIn("ExpConst", self.impl_mod.FakeDeviceClass.__dict__)

      val1 = tpu_mod.ExpConst
      val2 = tpu_mod.ExpConst
    self.assertEqual(val1, "exp_value")
    self.assertEqual(val2, "exp_value")
    self.assertNotIn("ExpConst", tpu_mod.__dict__)
    self.assertLen(w, 1)
    self.assertTrue(issubclass(w[0].category, UserWarning))
    self.assertIn("'ExpConst' is experimental", str(w[0].message))
    self.assertEqual(w[0].filename, __file__)

    delattr(self.impl_mod.FakeDeviceClass, "ExpConst")
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      exec("from fake_root.tpu import ExpConst", {"__name__": __name__})  # pylint: disable=exec-used
    self.assertLen(w, 1)
    self.assertEqual(w[0].filename, "<string>")


if __name__ == "__main__":
  absltest.main()
