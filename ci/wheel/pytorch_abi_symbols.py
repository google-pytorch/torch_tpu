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

"""Extracts and validates PyTorch ABI symbols for versioned wheel libraries.

The multi-ABI `torch_tpu` wheel bundles versioned C++ extension glue `.so`
files (`*_<version>.so`) and a per-version common library
(`libpywrap_<version>_common.so`) for each supported PyTorch release, without
bundling `libtorch` or `libc10`. At runtime, those versioned `.so` files resolve
their undefined PyTorch symbols (`c10::`, `at::`, `torch::`, `c10d::`, `THP*`,
etc.) against the user's installed PyTorch distribution.

The checked-in symbol file (`ci/wheel/pytorch_symbols_2_15_0.txt`) records the
high-water mark of all undefined dynamic PyTorch symbols that `torch_tpu` has
referenced across builds targeting PyTorch 2.15.0:
  * Updating the file (`--wheel ... --output ...`) unions newly extracted
    symbols with the existing symbols in the file so previously used symbols are
    preserved.
  * Checking a `torch_tpu` wheel (`--wheel ... --check ...`) verifies that every
    undefined PyTorch symbol in the wheel's versioned glues is recorded in the
    high-water mark (`wheel_symbols <= high_water_mark`).
  * Checking a PyTorch distribution (`--pytorch-wheel`, `--pytorch-lib-dir`, or
    `--pull-latest-pytorch` with `--check ...`) verifies that every symbol in
    the high-water mark remains defined and exported by PyTorch's shared
    libraries (`high_water_mark <= pytorch_defined_symbols`).
"""

import argparse
from collections.abc import Iterable, Mapping, Sequence, Set
import dataclasses
import hashlib
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Final
import urllib.parse
import urllib.request
import zipfile

DEFAULT_PYTORCH_VERSION: Final[str] = "2.15.0"
DEFAULT_PYTHON_ABI_TAG: Final[str] = "cp314"
DEFAULT_PLATFORM_ARCH: Final[str] = "x86_64"

# PyTorch wheel index directories checked by `--pull-latest-pytorch`. The release
# index is checked first (for GA releases), falling back to the RC test index.
DEFAULT_PYTORCH_INDEX_URLS: Final[tuple[str, ...]] = (
    "https://download.pytorch.org/whl/cpu/torch/",
    "https://download.pytorch.org/whl/test/cpu/torch/",
)

# Dynamic symbol table (`nm -D`) types that represent undefined references
# resolved from external shared libraries at runtime: strong undefined (`U`) and
# weak undefined (`w`).
_UNDEFINED_NM_TYPES: Final[frozenset[str]] = frozenset({"U", "w"})

# Matches Itanium C++ ABI mangled symbols (and C-linkage THP / ParameterClass
# symbols) owned by PyTorch's libraries (`libc10.so`, `libtorch_cpu.so`,
# `libtorch_python.so`).
#
# Specifically matches:
#   * Nested names (`_ZN...`, `_ZNK...`, vtables `_ZTVN...`, typeinfo
#     `_ZTIN...` / `_ZTSN...`, VTT `_ZTTN...`, guard variables `_ZGVN...`,
#     thread-local wrappers `_ZTWN...`, and covariant/non-virtual thunks
#     `_ZTh..._N...` / `_ZTv..._N...`) rooted in the top-level PyTorch
#     namespaces `c10`, `at`, `torch`, `c10d`, `caffe2`, or `libkineto`, or
#     `pybind11::detail::type_caster<at::...>`. Compiler-synthesized optional
#     `_ZTH` thread-local init hooks (emitted as weak references even when the
#     underlying `thread_local` variable uses constant initialization) are
#     intentionally excluded; the underlying `_ZN...` / `_ZTW...` symbols are
#     still matched.
#   * `libtorch_python` C/C++ symbols prefixed with `THP` (e.g. `THPVariable_*`)
#     and `ParameterClass`.
_MANGLED_PYTORCH_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(
    r"^("
    r"_Z(?:TV|TI|TS|TT|GV|TW|Th[^_]*_|Tv[^_]*_)?N[KRVrO]*"
    r"(?:3c10|2at|5torch|4c10d|6caffe2|9libkineto"
    r"|8pybind116detail11type_casterIN2at)"
    r"|_Z\d*THP"
    r"|THP"
    r"|ParameterClass$"
    r")"
)

_FILE_HEADER_LINES: Final[Sequence[str]] = (
    "# Copyright 2026 Google LLC",
    "#",
    '# Licensed under the Apache License, Version 2.0 (the "License");',
    "# you may not use this file except in compliance with the License.",
    "# You may obtain a copy of the License at",
    "#",
    "#     http://www.apache.org/licenses/LICENSE-2.0",
    "#",
    "# Unless required by applicable law or agreed to in writing, software",
    '# distributed under the License is distributed on an "AS IS" BASIS,',
    (
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or"
        " implied."
    ),
    "# See the License for the specific language governing permissions and",
    "# limitations under the License.",
    "#",
    "# High-water mark of undefined dynamic PyTorch symbols referenced by the",
    "# versioned 2.15.0 glue libraries in the torch_tpu wheel.",
    "#",
    "# To update the high-water mark from a built torch_tpu wheel:",
    "#   python3 ci/wheel/pytorch_abi_symbols.py \\",
    "#       --wheel path/to/torch_tpu-*.whl \\",
    "#       --version 2.15.0 \\",
    "#       --output ci/wheel/pytorch_symbols_2_15_0.txt",
    "#",
    "# To verify that the latest PyTorch 2.15.0 wheel still exports all",
    "# symbols in this high-water mark:",
    "#   python3 ci/wheel/pytorch_abi_symbols.py \\",
    "#       --pull-latest-pytorch \\",
    "#       --version 2.15.0 \\",
    "#       --check ci/wheel/pytorch_symbols_2_15_0.txt",
)


def normalize_version_suffix(version: str) -> str:
  """Normalizes a version string like '2.15.0' or '2_15_0' to '2_15_0'."""
  cleaned = version.strip().replace(".", "_")
  if not re.fullmatch(r"\d+_\d+_\d+", cleaned):
    raise ValueError(
        "Expected a 3-component version like '2.15.0' or '2_15_0', got"
        f" {version!r}."
    )
  return cleaned


def normalize_dotted_version(version: str) -> str:
  """Normalizes a version string like '2.15.0' or '2_15_0' to '2.15.0'."""
  return normalize_version_suffix(version).replace("_", ".")


def is_versioned_so_for_version(
    path_str: str, version: str = DEFAULT_PYTORCH_VERSION
) -> bool:
  """Returns True if `path_str` is a glue or common `.so` for `version`."""
  suffix = normalize_version_suffix(version)
  filename = pathlib.PurePosixPath(path_str).name
  return bool(re.search(rf"_{re.escape(suffix)}(?:_common)?\.so$", filename))


def is_pytorch_lib_so(path_str: str) -> bool:
  """Returns True if `path_str` is a shared library in `torch/lib/lib*.so*`."""
  posix_path = pathlib.PurePosixPath(path_str)
  name = posix_path.name
  if not (name.startswith("lib") and (".so" in name)):
    return False
  parts = posix_path.parts
  if len(parts) >= 2 and parts[-3:-1] == ("torch", "lib"):
    return True
  return len(parts) == 1


def is_pytorch_symbol(mangled_name: str) -> bool:
  """Returns True if `mangled_name` is a PyTorch symbol from libtorch/libc10."""
  base_symbol = mangled_name.split("@", 1)[0]
  return bool(_MANGLED_PYTORCH_SYMBOL_RE.search(base_symbol))


def parse_nm_undefined_pytorch_symbols(nm_output: str) -> frozenset[str]:
  """Parses `nm -D --undefined-only` output and returns PyTorch symbols."""
  symbols: set[str] = set()
  for line in nm_output.splitlines():
    parts = line.split()
    if len(parts) < 2:
      continue
    sym_type = parts[-2]
    sym_name = parts[-1].split("@", 1)[0]
    if sym_type in _UNDEFINED_NM_TYPES and is_pytorch_symbol(sym_name):
      symbols.add(sym_name)
  return frozenset(symbols)


def parse_nm_defined_symbols(nm_output: str) -> frozenset[str]:
  """Parses `nm -D --defined-only` output and returns defined symbol names."""
  symbols: set[str] = set()
  for line in nm_output.splitlines():
    parts = line.split()
    if len(parts) < 2:
      continue
    sym_type = parts[-2]
    sym_name = parts[-1].split("@", 1)[0]
    if sym_type not in _UNDEFINED_NM_TYPES and len(sym_type) == 1:
      symbols.add(sym_name)
  return frozenset(symbols)


def extract_so_pytorch_symbols(
    so_paths: Sequence[pathlib.Path],
) -> frozenset[str]:
  """Extracts undefined dynamic PyTorch symbols from `.so` files via `nm -D`."""
  if not so_paths:
    raise ValueError("so_paths must not be empty.")
  proc = subprocess.run(
      ["nm", "-D", "--undefined-only", *(str(p) for p in so_paths)],
      capture_output=True,
      text=True,
      check=True,
  )
  return parse_nm_undefined_pytorch_symbols(proc.stdout)


def extract_so_defined_symbols(
    so_paths: Sequence[pathlib.Path],
) -> frozenset[str]:
  """Extracts defined dynamic symbols from `.so` files via `nm -D`."""
  if not so_paths:
    raise ValueError("so_paths must not be empty.")
  proc = subprocess.run(
      ["nm", "-D", "--defined-only", *(str(p) for p in so_paths)],
      capture_output=True,
      text=True,
      check=True,
  )
  return parse_nm_defined_symbols(proc.stdout)


def extract_dir_pytorch_symbols(
    wheel_root: pathlib.Path,
    version: str = DEFAULT_PYTORCH_VERSION,
) -> frozenset[str]:
  """Extracts undefined PyTorch symbols for `version` from an unpacked wheel."""
  so_paths = sorted(
      p
      for p in wheel_root.rglob("*.so")
      if p.is_file()
      and is_versioned_so_for_version(
          p.relative_to(wheel_root).as_posix(), version
      )
  )
  if not so_paths:
    raise FileNotFoundError(
        f"No versioned .so files for PyTorch {version} found under"
        f" {wheel_root}."
    )
  return extract_so_pytorch_symbols(so_paths)


def extract_wheel_pytorch_symbols(
    wheel_path: pathlib.Path,
    version: str = DEFAULT_PYTORCH_VERSION,
) -> frozenset[str]:
  """Extracts undefined PyTorch symbols for `version` from a `.whl` archive."""
  with (
      zipfile.ZipFile(wheel_path) as zf,
      tempfile.TemporaryDirectory() as tmp_dir,
  ):
    matching_members = sorted(
        name
        for name in zf.namelist()
        if is_versioned_so_for_version(name, version)
    )
    if not matching_members:
      raise FileNotFoundError(
          f"No versioned .so files for PyTorch {version} found in {wheel_path}."
      )
    tmp_path = pathlib.Path(tmp_dir)
    extracted_paths = [
        pathlib.Path(zf.extract(member, tmp_path))
        for member in matching_members
    ]
    return extract_so_pytorch_symbols(extracted_paths)


def extract_pytorch_lib_dir_defined_symbols(
    lib_dir: pathlib.Path,
) -> frozenset[str]:
  """Extracts defined dynamic symbols from a PyTorch `torch/lib` directory."""
  search_root = (
      lib_dir / "torch" / "lib"
      if (lib_dir / "torch" / "lib").is_dir()
      else lib_dir
  )
  so_paths = sorted(
      p
      for p in search_root.iterdir()
      if p.is_file() and p.name.startswith("lib") and ".so" in p.name
  )
  if not so_paths:
    raise FileNotFoundError(
        f"No PyTorch shared libraries (lib*.so*) found under {lib_dir}."
    )
  return extract_so_defined_symbols(so_paths)


def extract_pytorch_wheel_defined_symbols(
    pytorch_wheel_path: pathlib.Path,
) -> frozenset[str]:
  """Extracts defined dynamic symbols from a PyTorch (`torch-*.whl`) archive."""
  with (
      zipfile.ZipFile(pytorch_wheel_path) as zf,
      tempfile.TemporaryDirectory() as tmp_dir,
  ):
    matching_members = sorted(
        name for name in zf.namelist() if is_pytorch_lib_so(name)
    )
    if not matching_members:
      raise FileNotFoundError(
          "No PyTorch shared libraries (torch/lib/lib*.so*) found in"
          f" {pytorch_wheel_path}."
      )
    tmp_path = pathlib.Path(tmp_dir)
    extracted_paths = [
        pathlib.Path(zf.extract(member, tmp_path))
        for member in matching_members
    ]
    return extract_so_defined_symbols(extracted_paths)


def find_pytorch_wheel_url_in_index_html(
    html: str,
    base_url: str,
    version: str = DEFAULT_PYTORCH_VERSION,
    python_tag: str = DEFAULT_PYTHON_ABI_TAG,
    platform_arch: str = DEFAULT_PLATFORM_ARCH,
) -> str | None:
  """Finds a matching PyTorch wheel URL in a PEP 503 simple index HTML page."""
  dotted = normalize_dotted_version(version)
  href_pattern = re.compile(
      r'href=["\']([^"\']*torch-'
      + re.escape(dotted)
      + r"(?:%2B|\+|-)"
      + r'[^"\']*\.whl(?:#[^"\']*)?)["\']'
  )
  candidates: list[str] = []
  abi_substr = f"-{python_tag}-{python_tag}-"
  for match in href_pattern.finditer(html):
    href = match.group(1)
    unquoted = urllib.parse.unquote(href)
    if abi_substr in unquoted and platform_arch in unquoted:
      candidates.append(urllib.parse.urljoin(base_url, href))
  if not candidates:
    return None
  return candidates[-1]


def download_latest_pytorch_wheel(
    dest_dir: pathlib.Path,
    version: str = DEFAULT_PYTORCH_VERSION,
    python_tag: str = DEFAULT_PYTHON_ABI_TAG,
    platform_arch: str = DEFAULT_PLATFORM_ARCH,
    index_urls: Sequence[str] = DEFAULT_PYTORCH_INDEX_URLS,
) -> pathlib.Path:
  """Downloads the latest matching PyTorch wheel from `index_urls`."""
  wheel_url: str | None = None
  for index_url in index_urls:
    normalized_index = index_url if index_url.endswith("/") else f"{index_url}/"
    req = urllib.request.Request(
        normalized_index, headers={"User-Agent": "pytorch_abi_symbols/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
      html = resp.read().decode("utf-8", errors="replace")
    wheel_url = find_pytorch_wheel_url_in_index_html(
        html,
        normalized_index,
        version=version,
        python_tag=python_tag,
        platform_arch=platform_arch,
    )
    if wheel_url is not None:
      break

  if wheel_url is None:
    raise FileNotFoundError(
        f"Could not find a PyTorch {version} wheel for {python_tag} /"
        f" {platform_arch} in indexes: {list(index_urls)}"
    )

  parsed = urllib.parse.urlsplit(wheel_url)
  download_url = urllib.parse.urlunsplit(parsed._replace(fragment=""))
  expected_sha256: str | None = None
  if parsed.fragment.startswith("sha256="):
    expected_sha256 = parsed.fragment.removeprefix("sha256=").lower()

  filename = pathlib.PurePosixPath(urllib.parse.unquote(parsed.path)).name
  dest_dir.mkdir(parents=True, exist_ok=True)
  dest_path = dest_dir / filename

  req = urllib.request.Request(
      download_url, headers={"User-Agent": "pytorch_abi_symbols/1.0"}
  )
  hasher = hashlib.sha256()
  with (
      urllib.request.urlopen(req, timeout=120) as resp,
      dest_path.open("wb") as out_file,
  ):
    while chunk := resp.read(1024 * 1024):
      out_file.write(chunk)
      hasher.update(chunk)

  if expected_sha256 is not None and hasher.hexdigest() != expected_sha256:
    raise ValueError(
        f"SHA-256 mismatch for downloaded wheel {download_url}: expected"
        f" {expected_sha256}, got {hasher.hexdigest()}."
    )
  return dest_path


def demangle_symbols(mangled_names: Sequence[str]) -> Mapping[str, str]:
  """Demangles `mangled_names` with `c++filt`, falling back to raw names."""
  if not mangled_names:
    return {}
  try:
    proc = subprocess.run(
        ["c++filt"],
        input="\n".join(mangled_names),
        capture_output=True,
        text=True,
        check=True,
    )
    demangled_lines = proc.stdout.splitlines()
    if len(demangled_lines) == len(mangled_names):
      return dict(zip(mangled_names, demangled_lines, strict=True))
  except (OSError, subprocess.SubprocessError):
    pass
  return {name: name for name in mangled_names}


def load_symbols(path: pathlib.Path) -> frozenset[str]:
  """Loads a newline-delimited symbol file, ignoring blank and `#` lines."""
  symbols: set[str] = set()
  for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    symbols.add(line)
  return frozenset(symbols)


def format_symbols(symbols: Iterable[str]) -> str:
  """Formats `symbols` into a canonical sorted text file with license header."""
  lines = [*_FILE_HEADER_LINES, *sorted(set(symbols))]
  return "\n".join(lines) + "\n"


def update_high_water_mark(
    output_path: pathlib.Path,
    new_symbols: Iterable[str],
    *,
    reset: bool = False,
) -> frozenset[str]:
  """Updates `output_path` with `new_symbols`, preserving existing ones.

  Unless `reset=True` is passed, any symbols already recorded in `output_path`
  are unioned with `new_symbols` so that `output_path` acts as a monotonic
  high-water mark of all PyTorch symbols `torch_tpu` has used.

  Args:
    output_path: Target symbol file path.
    new_symbols: Newly extracted undefined PyTorch symbols from a `torch_tpu`
      wheel.
    reset: If True, overwrite `output_path` with only `new_symbols` instead of
      unioning with existing symbols.

  Returns:
    The resulting set of symbols written to `output_path`.
  """
  merged: set[str] = set(new_symbols)
  if not reset and output_path.exists():
    merged |= load_symbols(output_path)
  result = frozenset(merged)
  output_path.write_text(format_symbols(result), encoding="utf-8")
  return result


@dataclasses.dataclass(frozen=True)
class SymbolCompatibilityResult:
  """Result of checking PyTorch ABI symbols against a high-water mark list.

  Attributes:
    unexpected_in_wheel: Undefined PyTorch symbols present in the `torch_tpu`
      wheel's versioned glues but absent from the high-water mark file (e.g. a
      PyTorch function's signature/mangling changed in a newer release
      candidate, or `torch_tpu` added a new PyTorch C++ call that has not yet
      been added to the high-water mark).
    missing_from_pytorch: Symbols present in the high-water mark file that are
      not defined and exported by PyTorch's shared libraries (`libc10.so`,
      `libtorch_cpu.so`, `libtorch_python.so`, etc.), indicating an ABI break in
      PyTorch.
    unused_in_wheel: Historical symbols present in the high-water mark file that
      are not referenced by the current `torch_tpu` wheel. Allowed under
      high-water mark semantics (`is_compatible` remains True).
  """

  unexpected_in_wheel: tuple[str, ...] = ()
  missing_from_pytorch: tuple[str, ...] = ()
  unused_in_wheel: tuple[str, ...] = ()

  @property
  def is_compatible(self) -> bool:
    """Returns True if no unexpected or missing symbols were detected."""
    return not self.unexpected_in_wheel and not self.missing_from_pytorch

  def format_report(self) -> str:
    """Formats a human-readable diff report with demangled symbol names."""
    if self.is_compatible:
      return "PyTorch ABI symbols are compatible with the high-water mark."

    all_diff = sorted({*self.unexpected_in_wheel, *self.missing_from_pytorch})
    demangled = demangle_symbols(all_diff)
    lines = [
        (
            "PyTorch ABI symbol incompatibility detected against the high-water"
            " mark."
        ),
    ]
    if self.unexpected_in_wheel:
      lines.append(
          "\nUnexpected symbols in torch_tpu wheel"
          f" ({len(self.unexpected_in_wheel)} present in wheel but not in"
          " high-water mark; indicates a new PyTorch C++ call in torch_tpu or"
          " a PyTorch ABI signature change):"
      )
      for sym in self.unexpected_in_wheel:
        lines.append(f"  + {sym}  [{demangled.get(sym, sym)}]")
      lines.append(
          "\nIf this is an intentional torch_tpu C++ change that uses"
          " additional PyTorch symbols, update the high-water mark file with:\n"
          "  python3 ci/wheel/pytorch_abi_symbols.py"
          " --wheel <path/to/torch_tpu.whl> --version 2.15.0"
          " --output ci/wheel/pytorch_symbols_2_15_0.txt"
      )
    if self.missing_from_pytorch:
      lines.append(
          "\nHigh-water mark symbols missing from PyTorch shared libraries"
          f" ({len(self.missing_from_pytorch)} present in high-water mark but"
          " not defined in PyTorch libtorch/libc10; indicates an incompatible"
          " PyTorch ABI change):"
      )
      for sym in self.missing_from_pytorch:
        lines.append(f"  - {sym}  [{demangled.get(sym, sym)}]")
    return "\n".join(lines)


def check_symbols_compatible(
    high_water_mark_symbols: Set[str],
    wheel_symbols: Set[str] | None = None,
    *,
    pytorch_defined_symbols: Set[str] | None = None,
) -> SymbolCompatibilityResult:
  """Checks `wheel_symbols` and/or `pytorch_defined_symbols` vs high-water mark.

  Args:
    high_water_mark_symbols: The checked-in high-water mark symbol set.
    wheel_symbols: Undefined PyTorch symbols extracted from a `torch_tpu`
      wheel's versioned glues. When provided, verifies `wheel_symbols <=
      high_water_mark_symbols`.
    pytorch_defined_symbols: Defined dynamic symbols extracted from PyTorch's
      shared libraries (`torch/lib/lib*.so*`). When provided, verifies
      `high_water_mark_symbols <= pytorch_defined_symbols`.

  Returns:
    A `SymbolCompatibilityResult` describing any incompatibilities.
  """
  if wheel_symbols is None and pytorch_defined_symbols is None:
    raise ValueError(
        "At least one of wheel_symbols or pytorch_defined_symbols must be"
        " provided."
    )
  unexpected_in_wheel: tuple[str, ...] = ()
  unused_in_wheel: tuple[str, ...] = ()
  if wheel_symbols is not None:
    unexpected_in_wheel = tuple(sorted(wheel_symbols - high_water_mark_symbols))
    unused_in_wheel = tuple(sorted(high_water_mark_symbols - wheel_symbols))

  missing_from_pytorch: tuple[str, ...] = ()
  if pytorch_defined_symbols is not None:
    missing_from_pytorch = tuple(
        sorted(high_water_mark_symbols - pytorch_defined_symbols)
    )

  return SymbolCompatibilityResult(
      unexpected_in_wheel=unexpected_in_wheel,
      missing_from_pytorch=missing_from_pytorch,
      unused_in_wheel=unused_in_wheel,
  )


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description=(
          "Dump, update, or verify the high-water mark of PyTorch ABI symbols"
          " used by versioned torch_tpu wheel libraries."
      )
  )
  wheel_group = parser.add_mutually_exclusive_group()
  wheel_group.add_argument(
      "--wheel",
      type=pathlib.Path,
      help="Path to a built torch_tpu .whl file.",
  )
  wheel_group.add_argument(
      "--wheel-dir",
      type=pathlib.Path,
      help="Path to an unpacked torch_tpu wheel directory.",
  )

  pytorch_group = parser.add_mutually_exclusive_group()
  pytorch_group.add_argument(
      "--pytorch-wheel",
      type=pathlib.Path,
      help=(
          "Path to a PyTorch (torch-*.whl) wheel file to verify that all"
          " high-water mark symbols are defined in its shared libraries."
      ),
  )
  pytorch_group.add_argument(
      "--pytorch-lib-dir",
      type=pathlib.Path,
      help=(
          "Path to a PyTorch torch/lib directory (or unpacked torch wheel"
          " root) to verify that all high-water mark symbols are defined."
      ),
  )
  pytorch_group.add_argument(
      "--pull-latest-pytorch",
      action="store_true",
      help=(
          "Download the latest PyTorch wheel for --version from PyTorch's"
          " wheel index and verify that all high-water mark symbols are"
          " defined in it."
      ),
  )

  parser.add_argument(
      "--version",
      default=DEFAULT_PYTORCH_VERSION,
      help=(
          "PyTorch version to inspect (e.g. '2.15.0' or '2_15_0')."
          f" Defaults to {DEFAULT_PYTORCH_VERSION!r}."
      ),
  )
  parser.add_argument(
      "--python-tag",
      default=DEFAULT_PYTHON_ABI_TAG,
      help=(
          "Python ABI tag when downloading with --pull-latest-pytorch"
          f" (defaults to {DEFAULT_PYTHON_ABI_TAG!r})."
      ),
  )
  parser.add_argument(
      "--reset",
      action="store_true",
      help=(
          "When used with --output, overwrite the symbol file with only the"
          " given wheel's symbols instead of unioning with existing symbols."
      ),
  )
  action_group = parser.add_mutually_exclusive_group()
  action_group.add_argument(
      "--output",
      type=pathlib.Path,
      help=(
          "Update the high-water mark symbol file at this path (unions with"
          " existing symbols unless --reset is passed)."
      ),
  )
  action_group.add_argument(
      "--check",
      type=pathlib.Path,
      help="Check ABI compatibility against this high-water mark symbol file.",
  )
  args = parser.parse_args(argv)

  has_tpu_wheel = args.wheel is not None or args.wheel_dir is not None
  has_pytorch_source = (
      args.pytorch_wheel is not None
      or args.pytorch_lib_dir is not None
      or args.pull_latest_pytorch
  )

  if not has_tpu_wheel and not has_pytorch_source:
    parser.error(
        "At least one source (--wheel, --wheel-dir, --pytorch-wheel,"
        " --pytorch-lib-dir, or --pull-latest-pytorch) is required."
    )
  if has_pytorch_source and args.check is None:
    parser.error(
        "--check is required when using --pytorch-wheel, --pytorch-lib-dir,"
        " or --pull-latest-pytorch."
    )

  wheel_symbols: frozenset[str] | None = None
  if args.wheel is not None:
    wheel_symbols = extract_wheel_pytorch_symbols(args.wheel, args.version)
  elif args.wheel_dir is not None:
    wheel_symbols = extract_dir_pytorch_symbols(args.wheel_dir, args.version)

  pytorch_defined_symbols: frozenset[str] | None = None
  if args.pytorch_wheel is not None:
    pytorch_defined_symbols = extract_pytorch_wheel_defined_symbols(
        args.pytorch_wheel
    )
  elif args.pytorch_lib_dir is not None:
    pytorch_defined_symbols = extract_pytorch_lib_dir_defined_symbols(
        args.pytorch_lib_dir
    )
  elif args.pull_latest_pytorch:
    with tempfile.TemporaryDirectory() as tmp_dir:
      downloaded_whl = download_latest_pytorch_wheel(
          pathlib.Path(tmp_dir),
          version=args.version,
          python_tag=args.python_tag,
      )
      print(f"Downloaded {downloaded_whl.name}.")
      pytorch_defined_symbols = extract_pytorch_wheel_defined_symbols(
          downloaded_whl
      )

  if args.check is not None:
    high_water_mark = load_symbols(args.check)
    result = check_symbols_compatible(
        high_water_mark,
        wheel_symbols,
        pytorch_defined_symbols=pytorch_defined_symbols,
    )
    if not result.is_compatible:
      print(result.format_report(), file=sys.stderr)
      return 1
    if wheel_symbols is not None:
      print(
          f"Verified {len(wheel_symbols)} torch_tpu wheel symbols are within"
          f" the {len(high_water_mark)}-symbol high-water mark in"
          f" {args.check}."
      )
    if pytorch_defined_symbols is not None:
      print(
          f"Verified all {len(high_water_mark)} high-water mark symbols in"
          f" {args.check} are defined in PyTorch {args.version}"
          f" ({len(pytorch_defined_symbols)} exported symbols)."
      )
    return 0

  assert wheel_symbols is not None
  if args.output is not None:
    merged = update_high_water_mark(
        args.output, wheel_symbols, reset=args.reset
    )
    print(
        f"Updated high-water mark in {args.output} ({len(merged)} total"
        f" PyTorch {args.version} ABI symbols; {len(wheel_symbols)} in current"
        " wheel)."
    )
  else:
    sys.stdout.write(format_symbols(wheel_symbols))
  return 0


if __name__ == "__main__":
  sys.exit(main())
