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

"""Tests for the version-dispatch selection and import hook logic."""

from collections.abc import Iterator
import contextlib
import importlib
import importlib.machinery
import importlib.util
import pathlib
import pkgutil
import sys
import threading
import types
from typing import Final
from unittest import mock
import zipfile

from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu import _versioned_so_loader as loader
from tests import seed_test_utils

_IMPORT_STATE_LOCK: Final[threading.Lock] = threading.Lock()


@contextlib.contextmanager
def _isolated_import_state() -> Iterator[None]:
  """Saves and restores process-global import state under a lock."""
  with (
      _IMPORT_STATE_LOCK,
      mock.patch.object(sys, "path", sys.path[:]),
      mock.patch.object(sys, "path_hooks", sys.path_hooks[:]),
      mock.patch.dict(sys.path_importer_cache),
      mock.patch.dict(sys.modules),
  ):
    yield


@contextlib.contextmanager
def _fake_installed_torch(version: str = "2.12.0") -> Iterator[None]:
  """Fakes the installed `torch` module in `sys.modules`."""
  fake_torch = types.SimpleNamespace(__version__=version)
  with mock.patch.dict(sys.modules, {"torch": fake_torch}):
    yield


class ResolveSuffixTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      # Unversioned build: dispatch is disabled regardless of running version.
      ("no_built_glues_disables_dispatch", "2_12_0", [], None),
      ("no_built_glues_and_no_version_disables_dispatch", None, [], None),
      (
          "exact_match_is_chosen",
          "2_12_0",
          {"2_11_0", "2_12_0", "2_13_0"},
          "2_12_0",
      ),
      # 2.12.5 postdates the 2.12.0 glue and predates the 2.13.0 one, so the
      # 2.12.0 glue (which serves everything up to the next ABI break) is
      # chosen.
      (
          "release_between_glues_gets_the_newest_older_glue",
          "2_12_5",
          {"2_11_0", "2_12_0", "2_13_0"},
          "2_12_0",
      ),
      (
          "release_newer_than_every_glue_gets_the_newest",
          "2_14_0",
          {"2_11_0", "2_12_0"},
          "2_12_0",
      ),
      # "2_11_10" > "2_11_9" numerically although not lexicographically.
      (
          "comparison_is_numeric_not_lexicographic",
          "2_11_10",
          {"2_11_9", "2_11_10"},
          "2_11_10",
      ),
      (
          "comparison_is_numeric_below_the_running_version",
          "2_11_12",
          {"2_11_2", "2_11_10"},
          "2_11_10",
      ),
  )
  def test_resolution(self, running, built, expected):
    self.assertEqual(loader.resolve_suffix(running, built), expected)

  @parameterized.named_parameters(
      ("release_older_than_every_glue", "2_9_0", "predates"),
      ("undetectable_running_version", None, "could not determine"),
  )
  def test_raises_import_error(self, running, message):
    with self.assertRaisesRegex(ImportError, message):
      loader.resolve_suffix(running, {"2_11_0", "2_12_0"})


class VersionSuffixTest(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      ("2.13.0", "2_13_0"),
      ("2.13.0+cpu", "2_13_0"),
      # A dev build's suffix is the release triple it leads up to, so it is
      # served by the glue built against the matching dev snapshot.
      ("2.14.0.dev20260711+cpu", "2_14_0"),
      # A from-source build of PyTorch main reports a PEP 440 pre-release
      # marker attached straight to the patch component.
      ("2.14.0a0", "2_14_0"),
      ("2.14.0a0+gitfdcd20c", "2_14_0"),
      ("2.14.0rc1", "2_14_0"),
      ("2.11.10", "2_11_10"),
      # Undeterminable versions disable rather than misdirect the dispatch.
      ("2.13", None),
      ("main", None),
      ("2.13.0garbage", None),
  )
  def test_version_suffix(self, version, expected):
    self.assertEqual(loader.version_suffix(version), expected)


class InstalledTorchVersionTest(seed_test_utils.RepeatableTest):

  def test_prefers_the_imported_torch_over_install_metadata(self):
    # The glue must match the torch actually loaded in this process, which the
    # install metadata need not reflect.
    fake_torch = types.SimpleNamespace(__version__="2.12.5+fake")
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
      self.assertEqual(loader.installed_torch_version(), "2.12.5+fake")

  def test_falls_back_to_install_metadata_when_torch_not_imported(self) -> None:
    site_dir = pathlib.Path(self.create_tempdir().full_path)
    dist_info = site_dir / "torch-2.12.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torch\nVersion: 2.12.0+cu124\n",
        encoding="utf-8",
    )
    with _isolated_import_state():
      sys.modules.pop("torch", None)
      sys.path[:] = [str(site_dir)]
      self.assertEqual(loader.installed_torch_version(), "2.12.0+cu124")

  def test_returns_none_when_torch_not_imported_or_installed(self) -> None:
    site_dir = pathlib.Path(self.create_tempdir().full_path)
    with _isolated_import_state():
      sys.modules.pop("torch", None)
      sys.path[:] = [str(site_dir)]
      self.assertIsNone(loader.installed_torch_version())


class DiscoverBuiltSuffixesTest(seed_test_utils.RepeatableTest):

  def _touch(self, root: pathlib.Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

  def test_finds_versioned_glues_recursively_and_ignores_others(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    self._touch(root, "_internal/env_2_11_0.so")
    self._touch(root, "_internal/env_2_12_0.so")
    self._touch(root, "_internal/sync/_tpu_torch_sync_2_11_0.so")
    self._touch(root, "_internal/env.py")  # not a .so
    self._touch(root, "common/libpywrap_torch_tpu_common.so")  # unversioned
    self._touch(root, "common/libpywrap_2_11_0_common.so")  # not a glue

    self.assertEqual(
        loader.discover_built_suffixes(root), frozenset({"2_11_0", "2_12_0"})
    )

  def test_no_versioned_glues_returns_empty(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    self._touch(root, "_internal/env.so")
    self.assertEqual(loader.discover_built_suffixes(root), frozenset())


class PathEntryFinderAndHookTest(seed_test_utils.RepeatableTest):

  def setUp(self) -> None:
    super().setUp()
    self.enter_context(_isolated_import_state())

  def _write_file(
      self, root: pathlib.Path, rel: str, content: str = ""
  ) -> pathlib.Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

  def _make_fake_package(self) -> tuple[pathlib.Path, pathlib.Path]:
    site_dir = pathlib.Path(self.create_tempdir().full_path)
    pkg_dir = site_dir / "fake_tt"
    self._write_file(pkg_dir, "__init__.py")
    self._write_file(pkg_dir, "helper.py", 'VALUE = "top"\n')
    self._write_file(pkg_dir, "_internal/__init__.py")
    self._write_file(pkg_dir, "_internal/env_2_11_0.so")
    self._write_file(pkg_dir, "_internal/env_2_12_0.so")
    self._write_file(pkg_dir, "_internal/env_utils.py", 'VALUE = "internal"\n')
    self._write_file(pkg_dir, "_internal/sub/__init__.py")
    self._write_file(pkg_dir, "_internal/sub/leaf_2_11_0.so")
    self._write_file(pkg_dir, "_internal/sub/leaf_2_12_0.so")
    # Non-package directory without __init__.py (holds shared C++ libraries).
    self._write_file(pkg_dir, "csrc/common/libxla_base.so")
    self._write_file(pkg_dir, "csrc/common/libpywrap_2_12_0_common.so")

    sys.path.insert(0, str(site_dir))
    sys.path_importer_cache.pop(str(site_dir), None)
    return site_dir, pkg_dir

  def test_walk_packages_discovers_all_subpackages_and_unversioned_modules(
      self,
  ) -> None:
    _, pkg_dir = self._make_fake_package()
    with _fake_installed_torch():
      loader.install_hook(pkg_dir)

    import fake_tt  # pytype: disable=import-error  # pyrefly: ignore[missing-import]

    discovered = [
        (mod.name, mod.ispkg)
        for mod in pkgutil.walk_packages(fake_tt.__path__, "fake_tt.")
    ]
    self.assertEqual(
        discovered,
        [
            ("fake_tt._internal", True),
            ("fake_tt._internal.env", False),
            ("fake_tt._internal.env_utils", False),
            ("fake_tt._internal.sub", True),
            ("fake_tt._internal.sub.leaf", False),
            ("fake_tt.helper", False),
        ],
    )

  def test_find_spec_routes_unversioned_name_to_resolved_glue(self) -> None:
    _, pkg_dir = self._make_fake_package()
    with _fake_installed_torch():
      loader.install_hook(pkg_dir)

    import fake_tt._internal  # pylint: disable=unused-import  # ruff: ignore[unused-import]  # pytype: disable=import-error  # pyrefly: ignore[missing-import]

    spec = importlib.util.find_spec("fake_tt._internal.env")
    self.assertIsNotNone(spec)
    assert spec is not None
    self.assertEqual(spec.name, "fake_tt._internal.env")
    self.assertEqual(spec.origin, str(pkg_dir / "_internal/env_2_12_0.so"))
    self.assertIsInstance(spec.loader, loader.ExtensionProxyLoader)

  def test_pure_python_module_imports_normally(self) -> None:
    _, pkg_dir = self._make_fake_package()
    with _fake_installed_torch():
      loader.install_hook(pkg_dir)

    from fake_tt._internal import env_utils  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error  # pyrefly: ignore[missing-import]

    self.assertEqual(env_utils.VALUE, "internal")

  @parameterized.parameters(
      "fake_tt._internal.env_2_12_0",
      "fake_tt._internal.env_2_11_0",
  )
  def test_version_suffixed_glue_names_are_hidden_from_import(
      self, modname: str
  ) -> None:
    _, pkg_dir = self._make_fake_package()
    with _fake_installed_torch():
      loader.install_hook(pkg_dir)

    import fake_tt._internal  # pylint: disable=unused-import  # ruff: ignore[unused-import]  # pytype: disable=import-error  # pyrefly: ignore[missing-import]

    self.assertIsNone(importlib.util.find_spec(modname))
    with self.assertRaises(ModuleNotFoundError):  # ASSERT_RAISES_OK=loader
      importlib.import_module(modname)

  def test_package_takes_precedence_over_versioned_glue(self) -> None:
    _, pkg_dir = self._make_fake_package()
    self._write_file(pkg_dir, "_internal/conflict/__init__.py")
    self._write_file(pkg_dir, "_internal/conflict_2_12_0.so")
    with _fake_installed_torch():
      loader.install_hook(pkg_dir)

    import fake_tt._internal  # pytype: disable=import-error  # pyrefly: ignore[missing-import]

    internal_modules = {
        m.name: m.ispkg
        for m in pkgutil.iter_modules(
            fake_tt._internal.__path__, "fake_tt._internal."
        )
    }
    self.assertEqual(internal_modules.get("fake_tt._internal.conflict"), True)

    spec = importlib.util.find_spec("fake_tt._internal.conflict")
    self.assertIsNotNone(spec)
    assert spec is not None
    self.assertIsNotNone(spec.submodule_search_locations)

  def test_install_hook_evicts_cached_file_finder_for_package_dir(self) -> None:
    _, pkg_dir = self._make_fake_package()
    paths = (pkg_dir, pkg_dir / "_internal")
    # Simulate torch_tpu/__init__.py importing _tcmalloc_check before calling
    # install_hook, which populates sys.path_importer_cache with FileFinder.
    for path in paths:
      with self.subTest(phase="before", path=path):
        self.assertIsInstance(
            pkgutil.get_importer(str(path)), importlib.machinery.FileFinder
        )

    with _fake_installed_torch():
      loader.install_hook(pkg_dir)

    for path in paths:
      with self.subTest(phase="after", path=path):
        self.assertIsInstance(
            pkgutil.get_importer(str(path)),
            loader.VersionDispatchPathEntryFinder,
        )

  def test_install_hook_is_idempotent_for_same_package_dir(self) -> None:
    _, pkg_dir = self._make_fake_package()
    hooks_before = sys.path_hooks[:]
    with _fake_installed_torch():
      loader.install_hook(pkg_dir)
      loader.install_hook(pkg_dir / "_internal" / "..")

    self.assertEqual(sys.path_hooks[1:], hooks_before)

  def test_install_hook_is_noop_when_no_versioned_glues_exist(self) -> None:
    unversioned_dir = pathlib.Path(self.create_tempdir().full_path) / "plain_tt"
    self._write_file(unversioned_dir, "__init__.py")
    self._write_file(unversioned_dir, "_internal/env.so")

    hooks_before = sys.path_hooks[:]
    cache_before = dict(sys.path_importer_cache)
    with _fake_installed_torch():
      loader.install_hook(unversioned_dir)

    self.assertEqual(sys.path_hooks, hooks_before)
    self.assertEqual(sys.path_importer_cache, cache_before)

  def test_non_directory_path_entry_falls_through_to_zipimporter(self) -> None:
    _, pkg_dir = self._make_fake_package()
    zip_path = pkg_dir / "nested.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
      zf.writestr("zipped_mod.py", "X = 1\n")

    default_zip_importer = pkgutil.get_importer(str(zip_path))
    self.assertIsNotNone(default_zip_importer)
    sys.path_importer_cache.pop(str(zip_path), None)

    with _fake_installed_torch():
      loader.install_hook(pkg_dir)

    importer = pkgutil.get_importer(str(zip_path))
    self.assertIsNotNone(importer)
    self.assertNotIsInstance(importer, loader.VersionDispatchPathEntryFinder)
    self.assertIsInstance(importer, type(default_zip_importer))

  def test_directories_outside_package_dir_use_default_finder(self) -> None:
    site_dir, pkg_dir = self._make_fake_package()
    other_dir = site_dir / "other_pkg"
    self._write_file(other_dir, "__init__.py")

    with _fake_installed_torch():
      loader.install_hook(pkg_dir)

    for entry in (str(other_dir), str(pkg_dir / ".." / "other_pkg")):
      with self.subTest(entry=entry):
        self.assertIsInstance(
            pkgutil.get_importer(entry),
            importlib.machinery.FileFinder,
        )

  def test_invalidate_caches_refreshes_underlying_file_finder(self) -> None:
    _, pkg_dir = self._make_fake_package()
    with _fake_installed_torch():
      loader.install_hook(pkg_dir)

    import fake_tt._internal  # pylint: disable=unused-import  # ruff: ignore[unused-import]  # pytype: disable=import-error  # pyrefly: ignore[missing-import]

    self.assertIsNone(importlib.util.find_spec("fake_tt._internal.late_mod"))
    self._write_file(pkg_dir, "_internal/late_mod.py", "X = 1\n")
    importlib.invalidate_caches()
    self.assertIsNotNone(importlib.util.find_spec("fake_tt._internal.late_mod"))

  def test_extension_proxy_loader_passes_unversioned_spec_name_to_create_module(
      self,
  ) -> None:
    class FakeRealLoader:

      def __init__(self) -> None:
        self.seen_name: str | None = None
        self.executed_module: types.ModuleType | None = None

      def create_module(
          self, spec: importlib.machinery.ModuleSpec
      ) -> types.ModuleType:
        self.seen_name = spec.name
        return types.ModuleType(spec.name)

      def exec_module(self, module: types.ModuleType) -> None:
        self.executed_module = module

    real_loader = FakeRealLoader()
    proxy = loader.ExtensionProxyLoader(
        real_loader, "fake_tt._internal.env", "fake_tt._internal.env_2_12_0"
    )
    spec = importlib.machinery.ModuleSpec("fake_tt._internal.env_2_12_0", proxy)
    mod = proxy.create_module(spec)
    self.assertEqual(real_loader.seen_name, "fake_tt._internal.env")
    self.assertEqual(spec.name, "fake_tt._internal.env_2_12_0")
    proxy.exec_module(mod)
    self.assertIs(real_loader.executed_module, mod)


if __name__ == "__main__":
  absltest.main()
