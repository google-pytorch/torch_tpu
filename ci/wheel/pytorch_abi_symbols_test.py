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

"""Unit tests for ci/wheel/pytorch_abi_symbols.py."""

from collections.abc import Sequence
import pathlib
import subprocess
import tempfile
import unittest
import zipfile

from ci.wheel import pytorch_abi_symbols


def _compile_stub_so(
    out_path: pathlib.Path,
    undefined_symbols: Sequence[str] = (),
    defined_symbols: Sequence[str] = (),
) -> None:
  """Compiles a minimal real ELF shared library with the requested symbols."""
  lines = [f"extern void {sym}(void);" for sym in undefined_symbols]
  for sym in defined_symbols:
    lines.append(f"void {sym}(void) {{}}")
  lines.append("void _stub_entry(void) {")
  for sym in undefined_symbols:
    lines.append(f"  {sym}();")
  lines.append("}")

  with tempfile.NamedTemporaryFile(
      mode="w", suffix=".c", encoding="utf-8", delete=True
  ) as src_file:
    src_file.write("\n".join(lines) + "\n")
    src_file.flush()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cc", "-shared", "-fPIC", src_file.name, "-o", str(out_path)],
        check=True,
    )


class PytorchAbiSymbolsTest(
    unittest.TestCase  # UNITTEST_OK=Standalone wheel inspection utility test
):

  def setUp(self):
    super().setUp()
    self._temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self._temp_dir.cleanup)
    self.work_dir = pathlib.Path(self._temp_dir.name)

  def test_normalize_version_suffix_and_dotted(self):
    self.assertEqual(
        pytorch_abi_symbols.normalize_version_suffix("2.15.0"), "2_15_0"
    )
    self.assertEqual(
        pytorch_abi_symbols.normalize_version_suffix(" 2_15_0 "), "2_15_0"
    )
    self.assertEqual(
        pytorch_abi_symbols.normalize_dotted_version("2_15_0"), "2.15.0"
    )
    with self.assertRaises(ValueError):
      pytorch_abi_symbols.normalize_version_suffix("2.15")

  def test_is_versioned_so_for_version(self):
    self.assertTrue(
        pytorch_abi_symbols.is_versioned_so_for_version(
            "torch_tpu/_internal/env_2_15_0.so", "2.15.0"
        )
    )
    self.assertTrue(
        pytorch_abi_symbols.is_versioned_so_for_version(
            "torch_tpu/csrc/common/glue_2_15_0/libpywrap_2_15_0_common.so",
            "2.15.0",
        )
    )
    self.assertFalse(
        pytorch_abi_symbols.is_versioned_so_for_version(
            "torch_tpu/_internal/env_2_14_0.so", "2.15.0"
        )
    )
    self.assertFalse(
        pytorch_abi_symbols.is_versioned_so_for_version(
            "torch_tpu/csrc/common/libxla_base.so", "2.15.0"
        )
    )

  def test_is_pytorch_lib_so(self):
    self.assertTrue(
        pytorch_abi_symbols.is_pytorch_lib_so("torch/lib/libc10.so")
    )
    self.assertTrue(
        pytorch_abi_symbols.is_pytorch_lib_so("torch/lib/libtorch_cpu.so")
    )
    self.assertTrue(pytorch_abi_symbols.is_pytorch_lib_so("libtorch_python.so"))
    self.assertFalse(
        pytorch_abi_symbols.is_pytorch_lib_so("torch/_C.cpython-314.so")
    )
    self.assertFalse(
        pytorch_abi_symbols.is_pytorch_lib_so("torch/lib/libtorch.a")
    )

  def test_is_pytorch_symbol_classifies_pytorch_and_non_pytorch_symbols(self):
    pytorch_syms = [
        "_ZN3c105ErrorC1ENS_14SourceLocationE",
        "_ZN2at4_ops5zeros4callEN3c108ArrayRefINS2_6SymIntEEE",
        "_ZN5torch8autograd13_wrap_outputsEv",
        "_ZN4c10d9FileStoreC1Ev",
        "_ZTVN3c1012AutogradMetaE",
        "_ZTIN5torch8autograd4NodeE",
        "_ZTWN3c104impl26raw_local_dispatch_key_setE",
        "_ZN3c104impl26raw_local_dispatch_key_setE",
        "_ZThn16_N3c1010TensorImplD1Ev",
        "_ZN8pybind116detail11type_casterIN2at6ScalarEvE4loadENS_6handleEb",
        "_Z16THPVariable_WrapN2at10TensorBaseE",
        "THPVariableClass",
        "ParameterClass",
    ]
    for sym in pytorch_syms:
      with self.subTest(sym=sym):
        self.assertTrue(pytorch_abi_symbols.is_pytorch_symbol(sym))

    non_pytorch_syms = [
        "PyObject_GetAttrString",
        "_Py_Dealloc",
        "_ZN3xla9ShapeUtil8MakeShapeEv",
        "_ZN9torch_tpu19GetBaseTensorBufferERKN2at6TensorE",
        "_ZN4absl12lts_202501276StatusC1Ev",
        "_ZSt20__throw_length_errorPKc",
        "memcpy@GLIBC_2.14",
        # Optional compiler-synthesized TLS init hook (_ZTH) is excluded.
        "_ZTHN3c104impl26raw_local_dispatch_key_setE",
    ]
    for sym in non_pytorch_syms:
      with self.subTest(sym=sym):
        self.assertFalse(pytorch_abi_symbols.is_pytorch_symbol(sym))

  def test_parse_nm_undefined_and_defined_symbols(self):
    nm_undefined = "\n".join([
        "glue_2_15_0.so:",
        "                 U _ZN3c105ErrorC1Ev",
        "                 w _ZTVN3c1012AutogradMetaE",
        "                 U PyObject_GetAttrString",
        "                 U _ZN3xla9ShapeUtil4MakeEv",
        "                 U _ZN9torch_tpu19GetBaseTensorBufferERKN2at6TensorE",
        "0000000000001000 T _ZN3c1014DefinedLocallyEv",
    ])
    self.assertEqual(
        pytorch_abi_symbols.parse_nm_undefined_pytorch_symbols(nm_undefined),
        frozenset({"_ZN3c105ErrorC1Ev", "_ZTVN3c1012AutogradMetaE"}),
    )

    nm_defined = "\n".join([
        "libc10.so:",
        "0000000000001000 T _ZN3c105ErrorC1Ev",
        "0000000000002000 V _ZTVN3c1012AutogradMetaE",
        "0000000000003000 B _ZN3c104impl26raw_local_dispatch_key_setE",
        "                 U memcpy@GLIBC_2.14",
    ])
    self.assertEqual(
        pytorch_abi_symbols.parse_nm_defined_symbols(nm_defined),
        frozenset({
            "_ZN3c105ErrorC1Ev",
            "_ZTVN3c1012AutogradMetaE",
            "_ZN3c104impl26raw_local_dispatch_key_setE",
        }),
    )

  def test_extract_dir_and_wheel_pytorch_symbols(self):
    wheel_root = self.work_dir / "unpacked"
    _compile_stub_so(
        wheel_root / "torch_tpu/_internal/env_2_15_0.so",
        undefined_symbols=[
            "_ZN3c105ErrorC1Ev",
            "PyObject_GetAttrString",
            "_ZN3xla9ShapeUtil4MakeEv",
        ],
    )
    _compile_stub_so(
        wheel_root
        / "torch_tpu/csrc/common/glue_2_15_0/libpywrap_2_15_0_common.so",
        undefined_symbols=[
            "_ZN2at4_ops5zeros4callEv",
            "THPVariableClass",
        ],
        defined_symbols=[
            "_ZN9torch_tpu19GetBaseTensorBufferEv",
        ],
    )
    # A 2.14.0 glue library whose symbols must be ignored when querying 2.15.0.
    _compile_stub_so(
        wheel_root / "torch_tpu/_internal/env_2_14_0.so",
        undefined_symbols=["_ZN3c1015report_overflowEPKc"],
    )

    expected = frozenset({
        "_ZN3c105ErrorC1Ev",
        "_ZN2at4_ops5zeros4callEv",
        "THPVariableClass",
    })
    self.assertEqual(
        pytorch_abi_symbols.extract_dir_pytorch_symbols(wheel_root, "2.15.0"),
        expected,
    )

    wheel_path = self.work_dir / "torch_tpu-0.1.2-cp314-cp314-linux_x86_64.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
      for file_path in wheel_root.rglob("*"):
        if file_path.is_file():
          zf.write(file_path, file_path.relative_to(wheel_root).as_posix())

    self.assertEqual(
        pytorch_abi_symbols.extract_wheel_pytorch_symbols(wheel_path, "2.15.0"),
        expected,
    )

  def test_extract_pytorch_lib_dir_and_wheel_defined_symbols(self):
    pytorch_root = self.work_dir / "pytorch_pkg"
    _compile_stub_so(
        pytorch_root / "torch/lib/libc10.so",
        defined_symbols=["_ZN3c105ErrorC1Ev"],
    )
    _compile_stub_so(
        pytorch_root / "torch/lib/libtorch_cpu.so",
        defined_symbols=["_ZN2at4_ops5zeros4callEv"],
    )

    defined_from_dir = (
        pytorch_abi_symbols.extract_pytorch_lib_dir_defined_symbols(
            pytorch_root
        )
    )
    self.assertTrue(
        {"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"}.issubset(
            defined_from_dir
        )
    )

    pytorch_whl = self.work_dir / "torch-2.15.0+cpu-cp314-cp314-linux.whl"
    with zipfile.ZipFile(pytorch_whl, "w") as zf:
      for file_path in pytorch_root.rglob("*"):
        if file_path.is_file():
          zf.write(file_path, file_path.relative_to(pytorch_root).as_posix())

    defined_from_whl = (
        pytorch_abi_symbols.extract_pytorch_wheel_defined_symbols(pytorch_whl)
    )
    self.assertTrue(
        {"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"}.issubset(
            defined_from_whl
        )
    )

  def test_high_water_mark_semantics_and_report(self):
    high_water_mark = frozenset({
        "_ZN3c105ErrorC1Ev",
        "_ZN2at4_ops5zeros4callEv",
        "_ZN2at4_ops3sum4callEv",
    })
    # A wheel using a strict subset of the high-water mark is compatible.
    subset_result = pytorch_abi_symbols.check_symbols_compatible(
        high_water_mark,
        frozenset({"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"}),
        pytorch_defined_symbols=high_water_mark | {"_ZN3c105ExtraEv"},
    )
    self.assertTrue(subset_result.is_compatible)
    self.assertEqual(subset_result.unexpected_in_wheel, ())
    self.assertEqual(subset_result.missing_from_pytorch, ())
    self.assertEqual(subset_result.unused_in_wheel, ("_ZN2at4_ops3sum4callEv",))

    # A wheel referencing a symbol outside the high-water mark fails.
    unexpected_result = pytorch_abi_symbols.check_symbols_compatible(
        high_water_mark,
        frozenset({"_ZN3c105ErrorC1Ev", "_ZN5torch7Library4_defEv"}),
    )
    self.assertFalse(unexpected_result.is_compatible)
    self.assertEqual(
        unexpected_result.unexpected_in_wheel, ("_ZN5torch7Library4_defEv",)
    )
    self.assertIn("torch::Library::_def()", unexpected_result.format_report())

    # A PyTorch wheel missing a high-water mark symbol (even one no longer used
    # by the current torch_tpu wheel) fails.
    missing_pytorch_result = pytorch_abi_symbols.check_symbols_compatible(
        high_water_mark,
        frozenset({"_ZN3c105ErrorC1Ev"}),
        pytorch_defined_symbols=frozenset(
            {"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"}
        ),
    )
    self.assertFalse(missing_pytorch_result.is_compatible)
    self.assertEqual(
        missing_pytorch_result.missing_from_pytorch,
        ("_ZN2at4_ops3sum4callEv",),
    )
    self.assertIn(
        "at::_ops::sum::call()", missing_pytorch_result.format_report()
    )

  def test_find_pytorch_wheel_url_in_index_html(self):
    html = """
    <html><body>
      <a href="/whl/test/cpu/torch-2.15.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=aaa">torch-313</a>
      <a href="/whl/test/cpu/torch-2.15.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=bbb">torch-314</a>
      <a href="/whl/test/cpu/torch-2.15.0%2Bcpu-cp314-cp314t-manylinux_2_28_x86_64.whl#sha256=ccc">torch-314t</a>
    </body></html>
    """
    url = pytorch_abi_symbols.find_pytorch_wheel_url_in_index_html(
        html,
        "https://download.pytorch.org/whl/test/cpu/torch/",
        version="2.15.0",
        python_tag="cp314",
        platform_arch="x86_64",
    )
    self.assertEqual(
        url,
        "https://download.pytorch.org/whl/test/cpu/torch-2.15.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=bbb",
    )

  def test_cli_high_water_mark_update_and_check(self):
    wheel_root = self.work_dir / "wheel_dir"
    _compile_stub_so(
        wheel_root / "torch_tpu/_internal/env_2_15_0.so",
        undefined_symbols=["_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"],
    )
    symbols_file = self.work_dir / "symbols.txt"

    # Initial dump populates the high-water mark with 2 symbols.
    self.assertEqual(
        pytorch_abi_symbols.main([
            "--wheel-dir",
            str(wheel_root),
            "--version",
            "2.15.0",
            "--output",
            str(symbols_file),
        ]),
        0,
    )
    self.assertEqual(
        pytorch_abi_symbols.load_symbols(symbols_file),
        frozenset({"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"}),
    )

    # Second build drops `_ZN2at4_ops5zeros4callEv` and adds
    # `_ZN2at4_ops3sum4callEv`. Updating without `--reset` keeps the union (3).
    _compile_stub_so(
        wheel_root / "torch_tpu/_internal/env_2_15_0.so",
        undefined_symbols=["_ZN3c105ErrorC1Ev", "_ZN2at4_ops3sum4callEv"],
    )
    self.assertEqual(
        pytorch_abi_symbols.main([
            "--wheel-dir",
            str(wheel_root),
            "--version",
            "2.15.0",
            "--output",
            str(symbols_file),
        ]),
        0,
    )
    self.assertEqual(
        pytorch_abi_symbols.load_symbols(symbols_file),
        frozenset({
            "_ZN3c105ErrorC1Ev",
            "_ZN2at4_ops5zeros4callEv",
            "_ZN2at4_ops3sum4callEv",
        }),
    )

    # Checking the 2-symbol wheel against the 3-symbol high-water mark passes.
    self.assertEqual(
        pytorch_abi_symbols.main([
            "--wheel-dir",
            str(wheel_root),
            "--version",
            "2.15.0",
            "--check",
            str(symbols_file),
        ]),
        0,
    )

    # Checking against a PyTorch lib dir that defines all 3 high-water mark
    # symbols passes.
    pytorch_lib_dir = self.work_dir / "torch_lib"
    _compile_stub_so(
        pytorch_lib_dir / "libtorch_cpu.so",
        defined_symbols=[
            "_ZN3c105ErrorC1Ev",
            "_ZN2at4_ops5zeros4callEv",
            "_ZN2at4_ops3sum4callEv",
        ],
    )
    self.assertEqual(
        pytorch_abi_symbols.main([
            "--pytorch-lib-dir",
            str(pytorch_lib_dir),
            "--version",
            "2.15.0",
            "--check",
            str(symbols_file),
        ]),
        0,
    )

    # If PyTorch drops `_ZN2at4_ops5zeros4callEv` (a historical high-water mark
    # symbol no longer in the current torch_tpu wheel), `--check` fails!
    _compile_stub_so(
        pytorch_lib_dir / "libtorch_cpu.so",
        defined_symbols=[
            "_ZN3c105ErrorC1Ev",
            "_ZN2at4_ops3sum4callEv",
        ],
    )
    self.assertEqual(
        pytorch_abi_symbols.main([
            "--pytorch-lib-dir",
            str(pytorch_lib_dir),
            "--version",
            "2.15.0",
            "--check",
            str(symbols_file),
        ]),
        1,
    )


if __name__ == "__main__":
  unittest.main()
