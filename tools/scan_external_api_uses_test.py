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

"""Unit tests for `scan_external_api_uses.py`."""

import os
import shutil
import subprocess
import textwrap

from absl.testing import absltest
from torch_tpu.oss.tools import scan_external_api_uses as scanner
import yaml

Category = scanner.Category

_TT_APIS = textwrap.dedent("""\
    # --- Public Python API ---
    torch.accelerator.empty_cache: {category: Public Python API, stage: Stable}
    torch.accelerator.synchronize: {category: Public Python API, stage: Stable}
    torch.accelerator.set_device_idx: {category: Public Python API, stage: Deprecated}
    torch.tpu.Event: {category: Public Python API, stage: Experimental}
    torch.tpu.Event.synchronize: {category: Public Python API, stage: Experimental}
    torch.tpu.get_device_name: {category: Public Python API, deprecated_in: 0.2.1, stage: Deprecated}
    torch.tpu.memory.max_memory_allocated: {category: Public Python API, stage: Stable}
    torch.tpu.synchronize: {category: Public Python API, prefer: torch.accelerator.synchronize, stage: Internal}
    torch.tpu.allow_excess_precision getter: {category: Public Python API, stage: Experimental}
    torch.tpu.internal_api: {category: Public Python API, stage: Internal}
    # --- Environment Variable ---
    TORCH_TPU_PROFILER_OUTPUT_DIR: {category: Environment Variable, stage: Experimental}
    JOB_COMPLETION_INDEX: {category: Environment Variable, stage: Internal}
    XLA_FLAGS: {category: Environment Variable, stage: Experimental}
    # --- ABSL C++ Flag ---
    torch_tpu_internal_flag: {category: ABSL C++ Flag, stage: Internal}
""")

_PYTHON_CODE = textwrap.dedent("""\
    import os
    import torch
    import torch_tpu as tt
    from torch import tpu
    from torch_tpu._internal import compile as tc
    tc.tpu_torch_compile.execute()
    tt._private_fn()
    getattr(tt, "_hidden")
    torch.tpu.synchronize()
    tpu.empty_cache()
    torch.tpu.Event().synchronize()
    torch.tpu.memory.max_memory_allocated()
    torch.tpu.get_device_name()
    torch.accelerator.empty_cache()
    torch.tpu.allow_excess_precision
    torch.tpu.internal_api()
    foo()._bar
    os.environ["TORCH_TPU_INTERNAL_FOO"] = "1"
    os.environ["TORCH_TPU_PROFILER_OUTPUT_DIR"] = "/tmp"
    os.environ["TPU_VISIBLE_CHIPS"] = "0"
    os.environ["TORCH_TPU_UNKNOWN"] = "0"
    importlib.import_module("torch_tpu._internal.pallas")
    rank = os.environ.get("JOB_COMPLETION_INDEX")
    os.environ["XLA_FLAGS"] = ""
    torch.ops.tpu.max_pool2d.default(x)
    torch.ops.tpu._private_op(x)
    torch.tpu.set_device_idx(0)
""")


def _write(root: str, rel_path: str, content: str) -> None:
  path = os.path.join(root, rel_path)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write(content)


class ScanExternalApiUsesTest(absltest.TestCase):  # ABSLTEST_OK=tool test

  def setUp(self):
    super().setUp()
    tt_apis_dir = self.create_tempdir().full_path
    _write(tt_apis_dir, "tt_apis.yaml", _TT_APIS)
    self.python_apis, self.env_vars, self.cpp_flags = scanner.load_tt_apis(
        os.path.join(tt_apis_dir, "tt_apis.yaml")
    )
    self.root = self.create_tempdir().full_path
    _write(self.root, "pkg/worker.py", _PYTHON_CODE)

  def _findings(self) -> dict[str, scanner.Finding]:
    result = scanner.scan_repo(
        self.root, self.python_apis, self.env_vars, self.cpp_flags
    )
    return {f.name: f for f in result.findings}

  def test_internal_apis(self):
    findings = self._findings()

    for name, reason in (
        ("torch_tpu._internal.compile", "`_internal` module"),
        ("torch_tpu._internal.compile.tpu_torch_compile.execute", None),
        ("torch_tpu._private_fn", "private name"),
        ("torch_tpu._hidden", "private name"),
        ("torch_tpu._internal.pallas", None),
        ("torch.tpu.internal_api", "Internal in tt_apis.yaml"),
        ("JOB_COMPLETION_INDEX", "Internal in tt_apis.yaml"),
    ):
      with self.subTest(name):
        self.assertEqual(findings[name].category, Category.INTERNAL)
        if reason:
          self.assertEqual(findings[name].reason, reason)
    self.assertEqual(
        findings["torch_tpu._internal.compile"].files, ("pkg/worker.py",)
    )

  def test_internal_apis_with_preferred_replacement(self):
    findings = self._findings()

    self.assertEqual(
        findings["torch.tpu.synchronize"].category, Category.INTERNAL
    )
    self.assertEqual(
        findings["torch.tpu.synchronize"].reason, "Internal in tt_apis.yaml"
    )
    self.assertEqual(
        findings["torch.tpu.synchronize"].prefer,
        "torch.accelerator.synchronize",
    )

  def test_deprecated_apis(self):
    findings = self._findings()

    self.assertEqual(
        findings["torch.tpu.get_device_name"].category, Category.DEPRECATED
    )
    self.assertEqual(
        findings["torch.tpu.get_device_name"].reason, "Deprecated in 0.2.1"
    )

  def test_unsupported_apis(self):
    findings = self._findings()

    self.assertEqual(
        findings["torch.tpu.empty_cache"].category, Category.UNSUPPORTED
    )
    # `torch.accelerator.empty_cache` is public, so it's suggested.
    self.assertEqual(
        findings["torch.tpu.empty_cache"].prefer,
        "torch.accelerator.empty_cache",
    )
    # `torch.accelerator.set_device_idx` is deprecated, so it isn't.
    self.assertIsNone(findings["torch.tpu.set_device_idx"].prefer)
    for name in (
        "torch.tpu.set_device_idx",
        "torch.ops.tpu.max_pool2d.default",
    ):
      with self.subTest(name):
        self.assertEqual(findings[name].reason, "not a public API")
    self.assertEqual(
        findings["torch.ops.tpu._private_op"].category, Category.INTERNAL
    )
    # TorchTPU removed or renamed TORCH_TPU_INTERNAL_FOO (it's not in
    # tt_apis.yaml), so setting it has no effect.
    self.assertEqual(
        findings["TORCH_TPU_INTERNAL_FOO"].category, Category.UNSUPPORTED
    )
    # TorchTPU doesn't read TORCH_TPU_UNKNOWN (it's not in tt_apis.yaml), so
    # it's the repository's own variable.
    self.assertNotIn("TORCH_TPU_UNKNOWN", findings)

  def test_experimental_and_stable_apis(self):
    findings = self._findings()

    self.assertEqual(
        findings["torch.tpu.Event"].category, Category.EXPERIMENTAL
    )
    self.assertEqual(
        findings["torch.tpu.allow_excess_precision"].category,
        Category.EXPERIMENTAL,
    )
    for name in ("TORCH_TPU_PROFILER_OUTPUT_DIR", "XLA_FLAGS"):
      with self.subTest(name):
        self.assertEqual(findings[name].category, Category.EXPERIMENTAL)
    # Stable APIs, modules that contain public APIs, other libraries' private
    # names and other TPU runtime variables aren't reported.
    for name in (
        "torch.tpu.memory.max_memory_allocated",
        "torch.accelerator.empty_cache",
        "torch.tpu.memory",
        "TPU_VISIBLE_CHIPS",
    ):
      with self.subTest(name):
        self.assertNotIn(name, findings)
    self.assertFalse(any("_bar" in name for name in findings))

  def test_device_module(self):
    _write(
        self.root,
        "pkg/generic.py",
        textwrap.dedent("""\
            import torch
            from sglang.srt.utils import get_device_module

            torch.get_device_module().synchronize()
            dev = torch.get_device_module(x.device)
            dev.empty_cache()
            getattr(dev, "_private")()
            mod: object = get_device_module()
            mod.Event()

            class Runner:

              def __init__(self, device):
                self.device_module = torch.get_device_module(device)

              def run(self):
                self.device_module.current_stream()
                self.other.manual_seed()
        """),
    )

    findings = self._findings()

    self.assertIn("pkg/generic.py", findings["torch.tpu.synchronize"].files)
    self.assertEqual(
        findings["torch.tpu.empty_cache"].files,
        ("pkg/generic.py", "pkg/worker.py"),
    )
    self.assertEqual(findings["torch.tpu._private"].files, ("pkg/generic.py",))
    self.assertEqual(
        findings["torch.tpu.Event"].category, Category.EXPERIMENTAL
    )
    self.assertIn("pkg/generic.py", findings["torch.tpu.Event"].files)
    self.assertEqual(
        findings["torch.tpu.current_stream"].files, ("pkg/generic.py",)
    )
    # `self.other` isn't the device module.
    self.assertNotIn("torch.tpu.manual_seed", findings)

  def test_device_module_helpers_and_devices(self):
    _write(
        self.root,
        "pkg/helpers.py",
        textwrap.dedent("""\
            import torch

            def _device_module():
              return torch.get_device_module(DEVICE)

            def _maybe_device_module():
              if DEVICE:
                return torch.get_device_module()
              return None

            _device_module().is_initialized()
            _maybe_device_module().is_available()
            torch.get_device_module("cpu").manual_seed(0)
            torch.get_device_module("tpu:0").device_count()
        """),
    )

    findings = self._findings()

    self.assertEqual(
        findings["torch.tpu.is_initialized"].files, ("pkg/helpers.py",)
    )
    self.assertEqual(
        findings["torch.tpu.device_count"].files, ("pkg/helpers.py",)
    )
    # `_maybe_device_module()` may return None; "cpu" is the CPU module.
    self.assertNotIn("torch.tpu.is_available", findings)
    self.assertNotIn("torch.tpu.manual_seed", findings)

  def test_getattr_of_torch_with_a_device_is_the_device_module(self):
    _write(
        self.root,
        "pkg/lib.py",
        textwrap.dedent("""\
            import torch

            device_lib = getattr(torch, device)
            device_lib.device_count()
            getattr(torch, dtype).finfo
        """),
    )

    findings = self._findings()

    self.assertEqual(findings["torch.tpu.device_count"].files, ("pkg/lib.py",))
    # `getattr(torch, dtype)` isn't a device module.
    self.assertNotIn("torch.tpu.finfo", findings)

  def test_device_module_aliases_are_scoped(self):
    _write(
        self.root,
        "pkg/scoped.py",
        textwrap.dedent("""\
            import torch

            def uses_device_module():
              module = torch.get_device_module()
              module.synchronize()

            def has_a_parameter(dev):
              dev.bias

            def reassigns():
              device = torch.get_device_module()
              device = "cpu"
              device.startswith("c")

            def loops(modules):
              for dev in modules:
                dev.children()

            class A:

              def __init__(self):
                self.mod = torch.get_device_module()

            class B:

              def run(self):
                self.mod.empty_cache()

            dev = torch.get_device_module()

            def uses_the_global():
              dev.is_available()

            def outer():
              mod = torch.get_device_module()
              def inner():
                mod.current_stream()
        """),
    )

    findings = self._findings()

    self.assertEqual(
        findings["torch.tpu.synchronize"].files,
        ("pkg/scoped.py", "pkg/worker.py"),
    )
    for name in ("torch.tpu.is_available", "torch.tpu.current_stream"):
      with self.subTest(name):
        self.assertEqual(findings[name].files, ("pkg/scoped.py",))
    # `self.mod` of another class isn't the device module.
    self.assertEqual(
        findings["torch.tpu.empty_cache"].files, ("pkg/worker.py",)
    )
    for name in (
        "torch.tpu.bias",
        "torch.tpu.startswith",
        "torch.tpu.children",
    ):
      self.assertNotIn(name, findings)

  def test_device_module_by_name(self):
    _write(
        self.root,
        "pkg/named.py",
        textwrap.dedent("""\
            import torch
            from sglang.srt.utils import get_device_module
            from torch_tpu._internal.device import _device_module

            device_module = None

            def _get_device_module():
              global device_module
              if device_module is None:
                device_module = get_device_module()
              return device_module

            _get_device_module().current_stream()
            device_module.Event()
            device_module.Event.assert_not_called()
            _device_module._DeviceModule

            def set_kv_buffer(device_module):
              device_module.synchronize()

            class Mixin:

              def run(self):
                self.device_module.is_available()
                self.runner.device_module.device_count()
        """),
    )

    findings = self._findings()

    for name in (
        "torch.tpu.current_stream",
        "torch.tpu.Event",
        "torch.tpu.synchronize",
        "torch.tpu.is_available",
        "torch.tpu.device_count",
    ):
      with self.subTest(name):
        self.assertIn("pkg/named.py", findings[name].files)
    # An import takes precedence.
    self.assertNotIn("torch.tpu._DeviceModule", findings)
    # An attribute of an API in tt_apis.yaml, e.g. of a mock, is a use of it.
    self.assertNotIn("torch.tpu.Event.assert_not_called", findings)

  def test_mocks_and_fakes_in_tests(self):
    _write(
        self.root,
        "tests/test_runner.py",
        textwrap.dedent("""\
            from types import SimpleNamespace
            from unittest import mock
            import torch

            def test_warmup():
              with mock.patch.object(torch, "tpu", create=True) as tpu:
                run()
              tpu.synchronize.assert_called_once_with()
              tpu.max_memory_reserved.return_value = 0

            def test_monkeypatch(monkeypatch):
              monkeypatch.setattr(
                  torch, "tpu", SimpleNamespace(current_device=lambda: 0)
              )

            def test_pool():
              pool = SimpleNamespace(
                  device_module=SimpleNamespace(is_available=lambda: True),
              )
              scheduler.device_module = SimpleNamespace(Event=mock.Mock())
              device_module = torch.get_device_module(device="tpu")
        """),
    )

    findings = self._findings()

    for name in (
        "torch.tpu.synchronize",
        "torch.tpu.is_available",
        "torch.tpu.Event",
        "torch.tpu.current_device",
    ):
      with self.subTest(name):
        self.assertIn("tests/test_runner.py", findings[name].files)
    # Only the keyword arguments of a fake are APIs.
    self.assertNotIn("torch.tpu.device", findings)
    # Attributes of a mock aren't, also of an API that isn't in tt_apis.yaml.
    self.assertIn("torch.tpu.max_memory_reserved", findings)
    self.assertNotIn("torch.tpu.max_memory_reserved.return_value", findings)

  def test_module_aliases(self):
    _write(
        self.root,
        "pkg/aliases.py",
        textwrap.dedent("""\
            import torch
            import torch_tpu

            tpu = torch.tpu
            tpu.synchronize()

            def run():
              sync = torch_tpu._internal.sync
              sync.synchronize()

            class Kernel:

              def __init__(self):
                self.pallas = torch_tpu._internal.pallas

              def run(self):
                self.pallas.jax_op()
        """),
    )

    findings = self._findings()

    self.assertIn("pkg/aliases.py", findings["torch.tpu.synchronize"].files)
    for name in (
        "torch_tpu._internal.sync.synchronize",
        "torch_tpu._internal.pallas.jax_op",
    ):
      with self.subTest(name):
        self.assertEqual(findings[name].files, ("pkg/aliases.py",))

  def test_assigning_an_attribute_doesnt_rebind_the_object(self):
    _write(
        self.root,
        "pkg/patch.py",
        textwrap.dedent("""\
            from torch_tpu._internal import execution_mode

            def configure():
              execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER
              execution_mode.modes[0] = None
        """),
    )

    findings = self._findings()

    for name in (
        "torch_tpu._internal.execution_mode.eager_mode",
        "torch_tpu._internal.execution_mode.modes",
    ):
      with self.subTest(name):
        self.assertEqual(findings[name].files, ("pkg/patch.py",))

  def test_api_symbol_drops_attributes_of_an_api(self):
    for name, symbol in (
        # An enum value.
        (
            "torch_tpu._internal.execution_mode.eager_mode.name",
            "torch_tpu._internal.execution_mode.eager_mode",
        ),
        # A member of a class.
        (
            "torch_tpu._internal.execution_mode.EagerMode.DEFER_AND_FUSE",
            "torch_tpu._internal.execution_mode.EagerMode.DEFER_AND_FUSE",
        ),
        # A dunder attribute.
        (
            "torch_tpu._internal.compile.compiler.StaticCompiler.__call__",
            "torch_tpu._internal.compile.compiler.StaticCompiler",
        ),
        ("torch.tpu.Event.__init__", "torch.tpu.Event"),
        # A name that TorchTPU doesn't have.
        ("torch_tpu._internal.removed.fn", "torch_tpu._internal.removed.fn"),
    ):
      with self.subTest(name):
        self.assertEqual(scanner.api_symbol(name), symbol)

  def test_docs_scripts_and_notebooks(self):
    _write(
        self.root,
        "docs/run.md",
        "Run `torch.tpu._clear_cache()` first.\nexport TORCH_TPU_INTERNAL_X=1 "
        " # Overrides `TORCH_TPU_INTERNAL_OLD*`.\npython run.py"
        " --torch_tpu_internal_flag=true\ngit clone"
        " https://github.com/google-pytorch/torch_tpu.git\n",
    )
    _write(
        self.root,
        "nb.ipynb",
        '{"cells": [{"cell_type": "markdown", "source": ["Use torch.tpu._a"]},'
        ' {"cell_type": "code", "source": ["%pip install x\\n",'
        ' "import torch_tpu._internal.utils\\n"]}]}',
    )

    findings = self._findings()

    for name in (
        "torch.tpu._clear_cache",
        "TORCH_TPU_INTERNAL_X",
        "torch_tpu_internal_flag",
    ):
      with self.subTest(name):
        self.assertEqual(findings[name].files, ("docs/run.md",))
    self.assertEqual(
        findings["torch_tpu_internal_flag"].category, Category.INTERNAL
    )
    # A wildcard isn't a variable.
    self.assertNotIn("TORCH_TPU_INTERNAL_OLD", findings)
    # A URL isn't an API use.
    self.assertNotIn("torch_tpu.git", findings)
    for name in ("torch.tpu._a", "torch_tpu._internal.utils"):
      with self.subTest(name):
        self.assertEqual(findings[name].files, ("nb.ipynb",))

  def test_parse_errors_and_excluded_dirs(self):
    _write(self.root, "broken.py", "import torch_tpu._internal.a\ndef f(:\n")
    _write(self.root, "broken.ipynb", '["torch_tpu._internal.b"]')
    _write(self.root, "run.sh", "export TORCH_TPU_INTERNAL_Y=1\n")
    _write(
        self.root, ".git/hooks/x.py", "import torch_tpu._internal.excluded\n"
    )

    result = scanner.scan_repo(
        self.root, self.python_apis, self.env_vars, self.cpp_flags
    )
    names = {f.name for f in result.findings}

    self.assertLen(result.parse_errors, 2)
    self.assertEqual(
        result.parse_errors[0],
        "broken.ipynb: ValueError: not a Jupyter notebook",
    )
    self.assertStartsWith(result.parse_errors[1], "broken.py: SyntaxError")
    # A file that can't be parsed is still scanned as text.
    self.assertIn("torch_tpu._internal.a", names)
    self.assertIn("torch_tpu._internal.b", names)
    self.assertNotIn("torch_tpu._internal.excluded", names)
    # pkg/worker.py, the broken files, and run.sh, which only sets a variable.
    self.assertEqual(result.num_files_using_non_stable_apis, 4)

  def test_checkout_local_directory(self):
    with scanner.checkout_repo(self.root) as (path, commit):
      self.assertEqual(path, self.root)
      self.assertIsNone(commit)
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=Test tooling error.
      with scanner.checkout_repo(self.root, ref="main"):
        pass

  def test_checkout_git_url_at_branch_and_commit(self):
    if shutil.which("git") is None:
      self.skipTest("git is not installed.")
    for args in (
        ["init", "-q", "-b", "main"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"],
    ):
      subprocess.run(["git", "-C", self.root, *args], check=True)
    sha = subprocess.run(
        ["git", "-C", self.root, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    for ref in ("main", sha):
      with scanner.checkout_repo(f"file://{self.root}", ref) as (path, commit):
        self.assertTrue(os.path.exists(os.path.join(path, "pkg/worker.py")))
        self.assertEqual(commit, sha)
      self.assertFalse(os.path.exists(path))

  def test_yaml_report(self):
    result = scanner.scan_repo(
        self.root,
        self.python_apis,
        self.env_vars,
        self.cpp_flags,
        repo="https://github.com/org/repo",
        commit="abc",
    )

    report = yaml.safe_load(scanner.format_yaml([result]))

    self.assertLen(report, 1)
    repo = report[0]
    self.assertEqual(repo["repo"], "https://github.com/org/repo")
    self.assertEqual(repo["commit"], "abc")
    self.assertEqual(repo["files_using_non_stable_apis"], 1)
    self.assertEqual(
        repo["summary"]["deprecated"],
        {
            "symbols": len(repo["deprecated"]),
            "files": len({
                file
                for a in repo["deprecated"].values()
                for file in a["used_in"]
            }),
        },
    )
    self.assertEqual(
        repo["deprecated"]["torch.tpu.get_device_name"],
        {
            "kind": "Python API",
            "reason": "Deprecated in 0.2.1",
            "used_in": ["pkg/worker.py"],
        },
    )
    self.assertEqual(
        repo["unsupported"]["torch.tpu.empty_cache"]["prefer"],
        "torch.accelerator.empty_cache",
    )
    self.assertEqual(repo["parse_errors"], [])


if __name__ == "__main__":
  absltest.main()
