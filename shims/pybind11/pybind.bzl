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

"""Stub for the pybind11 extension.

In addition to the plain extension `name` (used by bazel-only builds and by the
handful of non-pywrap label references), this macro fans out one extra glue per
supported PyTorch version so the separated pip wheel can ship a thin `.so` per
version. Each versioned glue is compiled against that version's torch headers
via a `//:_torch_version` transition and keeps the unversioned `PyInit_<name>`
symbol so the runtime dispatch loader (torch_tpu/_versioned_so_loader.py) can
rename the module at import time.

The glues for a given version are linked by a per-version `pywrap_library` (one
per PyTorch version, in its own package under torch_tpu/common/glue_<suffix>).
Each such library factors the version-independent XLA/MLIR/absl backend into a
single shared `libxla_base.so` (via a `common_lib_filters` entry fed by the
`_backend_probe` targets emitted here) and keeps that version's torch-touching
code in its own default common. All versions' `libxla_base.so` are byte-identical
and map to the same wheel path, so the wheel ships one copy. Bazel-only builds
are unaffected (they use `//torch_tpu/common:pywrap_torch_tpu`).
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", _pybind_extension = "pybind_extension")
load(
    "//shims/build_files:torch_version.bzl",
    "GLUE_TORCH_VERSIONS",
    "pin_glue_backend_deps",
    "torch_version_glue",
    "version_suffix",
)

# Package of the shared XLA base common library (libxla_base.so). Every
# per-version pywrap_library factors the backend into this one wheel location.
XLA_BASE_PACKAGE = "torch_tpu/common"

def glue_common_package(version):
    """Package (and wheel location) of a version's torch-touching common lib."""
    return "torch_tpu/common/glue_{}".format(version_suffix(version))

def pybind_extension(name, **kwargs):
    # On OSS we have not needed any python dependencies for these. If that
    # changes we will need to incorporate these dependencies.
    kwargs.pop("py_deps", None)

    # Force wrap_py_init off so the init function stays `PyInit_<name>` (from the
    # source's PYBIND11_MODULE) in the thin per-version glue rather than being
    # renamed after the target `name`. The dispatch loader relies on the
    # unversioned symbol being present in each versioned `.so`.
    kwargs["wrap_py_init"] = False

    # Base extension: the version-agnostic target every existing reference
    # (bazel tests via extra_pywrap_deps, and the ~6 direct label refs) resolves.
    _pybind_extension(name = name, **kwargs)

    raw_deps = kwargs.get("deps", [])
    testonly = kwargs.get("testonly")
    visibility = kwargs.get("visibility")

    # A plain cc_library mirroring this extension's deps, used only to feed the
    # shared-XLA-base filter. Under the filter's sentinel config its backend deps
    # (pinned via reset_torch_config) share linker-input identity with the
    # glues' backend, so aggregating these probes captures exactly the shared
    # XLA/MLIR to factor out of every per-version common. Its torch_tpu object
    # code is compiled under the sentinel torch_version and matches no glue, so
    # it is ignored.
    cc_library(
        name = "{}_backend_probe".format(name),
        deps = pin_glue_backend_deps("{}_probe".format(name), raw_deps),
        testonly = testonly,
        visibility = ["//visibility:public"],
    )

    # One transitioned glue per supported PyTorch version for the wheel. These
    # link real torch (via //shims/torch), so the glue's torch symbols resolve
    # through its per-version common library's DT_NEEDED on libtorch at load;
    # -z now surfaces any unresolved symbol immediately.
    versioned_kwargs = dict(kwargs)
    versioned_kwargs["linkopts"] = kwargs.get("linkopts", []) + [
        "-Wl,--allow-shlib-undefined",
        "-Wl,-z,now",
    ]
    for version in GLUE_TORCH_VERSIONS:
        versioned_name = "{}_{}".format(name, version_suffix(version))
        per_version_kwargs = dict(versioned_kwargs)

        # Pin the glue's own external backend deps to the canonical config so the
        # shared XLA base is built once (see pin_glue_backend_deps). torch_python
        # is added for the libtorch_python symbols (e.g. THPDtypeType) some
        # extensions use; under the version transition it links this version's
        # libtorch_python into the per-version common.
        per_version_kwargs["deps"] = pin_glue_backend_deps(
            versioned_name,
            raw_deps,
        ) + ["//shims/torch:torch_python"]

        # The glue links two common libs: the shared XLA base and this version's
        # torch common. common_lib_packages drives the glue's rpath to both.
        per_version_kwargs["common_lib_packages"] = [
            XLA_BASE_PACKAGE,
            glue_common_package(version),
        ]
        _pybind_extension(name = versioned_name, **per_version_kwargs)
        torch_version_glue(
            name = "{}_glue".format(versioned_name),
            dep = ":{}".format(versioned_name),
            torch_version = version,
            visibility = visibility,
            testonly = testonly,
        )

def backend_probe_labels(extension_labels):
    """Maps base pybind_extension labels to their `_backend_probe` labels."""
    return ["{}_backend_probe".format(label) for label in extension_labels]

def versioned_glue_labels_for(extension_labels, version):
    """Glue labels for a single PyTorch version.

    e.g. (["//torch_tpu/_internal:env"], "2.11.0") ->
        ["//torch_tpu/_internal:env_2_11_0_glue"]
    """
    return [
        "{}_{}_glue".format(label, version_suffix(version))
        for label in extension_labels
    ]

def versioned_glue_labels(extension_labels):
    """All per-version glue labels across every supported PyTorch version."""
    return [
        glue
        for version in GLUE_TORCH_VERSIONS
        for glue in versioned_glue_labels_for(extension_labels, version)
    ]
