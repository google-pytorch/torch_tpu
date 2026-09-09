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

"""Probes which shared objects torch_tpu's Python extensions map in.

`C10_REGISTER_GUARD_IMPL(PrivateUse1, TpuDeviceGuardImpl)` in
torch_tpu/csrc/eager/tpu_hooks.cc is a namespace-scope static, so it registers
at dlopen -- not when `_loader.load()` runs -- and claims the single PrivateUse1
slot that TorchAX also wants.

tpu_hooks is a BUILD dep of `_device_ops_backend` alone, but that says nothing
about which `.so` carries it. Under pywrap the per-extension `.so` files are
thin glue holding little more than `PyInit_<name>`; the object code of every
extension in bazel/pywrap_extensions.bzl and its cc deps is linked once into
`libpywrap_torch_tpu_common.so`, which the first extension import maps (see
shims/pybind11/pybind.bzl -- "The glue links two common libs"). So the question
is not which import maps a `.so` named after the extension, but which import
first maps that common library: tpu_hooks.o rides inside it.

This walks the module-scope torch_tpu imports of torch_benchmark_test.py's
import graph one at a time, printing what each newly maps with its size so the
glue-vs-common split is visible, and reports which import pulled in the common
library.

Run with:
  bazel run //examples/benchmarks/e2e/harness:guard_impl_probe
"""

import os
import re

# Mirror the harness (torch_benchmark_test.py:28). torch is imported inside
# main(), so this lands before torch reads the variable. Autoload is orthogonal
# to what is being measured here -- it gates `_loader.load()`, not dlopen -- but
# leaving it on would add torch_tpu mappings that no import in this file asked
# for, and hide which import is responsible.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

_SO_RE = re.compile(r"\S+\.so[^\s]*")


def _mapped() -> set[str]:
  """Returns the torch_tpu / pywrap shared objects mapped into this process."""
  with open("/proc/self/maps", encoding="utf-8") as f:
    maps = f.read()
  return {
      path
      for path in _SO_RE.findall(maps)
      if "torch_tpu" in path or "pywrap" in path
  }


def _common_lib(paths: set[str]) -> str | None:
  """Returns the pywrap common library among `paths`, or None if unmapped."""
  for path in paths:
    base = os.path.basename(path)
    if base.startswith("libpywrap") and "common" in base:
      return path
  return None


def _size(path: str) -> str:
  """Returns `path`'s size, formatted; glue and common differ by orders."""
  try:
    return f"{os.path.getsize(path) / 1e6:.1f} MB"
  except OSError:
    return "size unknown"


def _report(label: str, seen: set[str], state: dict[str, str]) -> set[str]:
  """Prints what `label` newly mapped and returns everything mapped so far.

  Args:
    label: The import being reported on.
    seen: The objects already mapped before it ran.
    state: Accumulator; gains a "common_lib_import" key naming the first
      import to map the pywrap common library.

  Returns:
    Every torch_tpu / pywrap object mapped after `label` ran.
  """
  now = _mapped()
  print(f"\n{label}")
  new = now - seen
  if new:
    for path in sorted(new):
      print(f"  + [{_size(path)}] {path}")
  else:
    print("  (nothing new mapped)")
  if "common_lib_import" not in state and _common_lib(now):
    state["common_lib_import"] = label
  return now


def main() -> None:
  """Imports the harness's extensions in order and reports what each maps."""
  state: dict[str, str] = {}

  # pylint: disable=g-import-not-at-top,unused-import
  import torch  # noqa: F401

  seen = _report("import torch", set(), state)

  # export.py:23
  from torch_tpu._internal.benchmarks import export_adapter  # noqa: F401

  seen = _report(
      "from torch_tpu._internal.benchmarks import export_adapter", seen, state
  )

  # measure.py:29. xprof_adapter now lives in the benchmarks namespace
  # package, so there is no __init__ barrel to run: expect nothing mapped.
  from torch_tpu._internal.benchmarks import xprof_adapter  # noqa: F401
  
  seen = _report(
      "from torch_tpu._internal.benchmarks import xprof_adapter", seen, state
  )

  # torch_benchmark_test.py:35
  from torch_tpu._internal import testing  # noqa: F401
  
  seen = _report("from torch_tpu._internal import testing", seen, state)

  # The extension that names tpu_hooks as a BUILD dep. Expect only its thin
  # glue here -- the object code came in with the common library above.
  from torch_tpu._internal.device import _device_ops_backend  # noqa: F401

  seen = _report(
      "from torch_tpu._internal.device import _device_ops_backend", seen, state
  )
  # pylint: enable=g-import-not-at-top,unused-import
  

  print("\n" + "-" * 72)
  culprit = state.get("common_lib_import")
  if culprit is None:
    print(
        "VERDICT: the pywrap common library was never mapped, so"
        " C10_REGISTER_GUARD_IMPL never ran and PrivateUse1 is unclaimed."
    )
  else:
    print(
        "VERDICT: the pywrap common library carrying tpu_hooks.o was mapped by"
        f" `{culprit}`, so TpuDeviceGuardImpl was registered there -- not at the"
        " _device_ops_backend import, whose glue holds no object code. Any"
        " torch_tpu extension import claims PrivateUse1."
    )
  print(
      "Confirm the symbol's home with:\n"
      "  nm -C <libpywrap_torch_tpu_common.so> | grep TpuDeviceGuardImpl"
  )


if __name__ == "__main__":
  main()
