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

"""Stub for the pybind11 extension."""

load("@rules_ml_toolchain//py/rules_pywrap:pywrap.impl.bzl", "CollectedPywrapInfo", "PywrapInfo")
load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", _pybind_extension = "pybind_extension")

# In Bazel, a "transition" allows you to change the configuration (e.g. build flags)
# for a specific dependency or part of the build graph. Here, we use it to
# force the `torch_version` flag to a specific version (like "2.10.0")
# for a specific sub-graph, regardless of what the user specified on the CLI.
# This enables us to build the exact same C++ code against multiple PyTorch
# versions simultaneously within a single `blaze build` invocation.
_TORCH_VERSION_FLAG = "//build_files:torch_version"

def _torch_version_transition_impl(settings, attr):
    """Transition implementation: maps the torch_version attribute to the internal flag."""
    return {_TORCH_VERSION_FLAG: attr.torch_version}

_torch_version_transition = transition(
    implementation = _torch_version_transition_impl,
    inputs = [],
    outputs = [_TORCH_VERSION_FLAG],
)

def _transitioned_pywrap_impl(ctx):
    """
    A wrapper rule that applies the configuration transition and forwards the
    underlying providers. By pointing `actual` to the base extension under a
    transition, the extension is compiled against the requested PyTorch version.
    """

    # Simply forward the DefaultInfo and PywrapLibraryInfo from the actual pybind_extension
    actual = ctx.attr.actual[0]
    providers = [actual[DefaultInfo]]
    if PywrapInfo in actual:
        providers.append(actual[PywrapInfo])
    if CollectedPywrapInfo in actual:
        providers.append(actual[CollectedPywrapInfo])

    if CcInfo in actual:
        providers.append(actual[CcInfo])

    if PyInfo in actual:
        providers.append(actual[PyInfo])

    return providers

_transitioned_pywrap = rule(
    implementation = _transitioned_pywrap_impl,
    attrs = {
        "actual": attr.label(cfg = _torch_version_transition),
        "torch_version": attr.string(),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

PYTORCH_VERSIONS = ["2.10.0", "2.11.0", "2.12.0"]

def _pybind_aggregator_impl(ctx):
    """
    Aggregates CcInfo and PywrapLibraryInfo across multiple transitioned extensions.
    This is necessary so that `pywrap_library` can successfully analyze a single
    target and traverse its dependencies to automatically extract the shared XLA/MLIR
    dependencies (common base factorization).
    """
    cc_infos = []
    pywrap_info = None
    py_info = None

    for dep in ctx.attr.deps:
        if CcInfo in dep:
            cc_infos.append(dep[CcInfo])
        if PywrapInfo in dep and not pywrap_info:
            pywrap_info = dep[PywrapInfo]
        if CollectedPywrapInfo in dep and not pywrap_info:
            pywrap_info = dep[CollectedPywrapInfo]
        if PyInfo in dep and not py_info:
            py_info = dep[PyInfo]

    files = depset(transitive = [dep[DefaultInfo].files for dep in ctx.attr.deps])
    runfiles = ctx.runfiles()
    for dep in ctx.attr.deps:
        runfiles = runfiles.merge(dep[DefaultInfo].default_runfiles)

    providers = [DefaultInfo(files = files, runfiles = runfiles)]

    if cc_infos:
        providers.append(cc_common.merge_cc_infos(cc_infos = cc_infos))
    if pywrap_info:
        providers.append(pywrap_info)
    if py_info:
        providers.append(py_info)

    return providers

pybind_aggregator = rule(
    implementation = _pybind_aggregator_impl,
    attrs = {
        "deps": attr.label_list(),
    },
)

def pybind_extension(name, **kwargs):
    """
    Generates a suite of PyTorch C++ bindings for multiple PyTorch ABI versions.

    Instead of building a single extension, this macro:
    1. Iterates over PYTORCH_VERSIONS.
    2. Builds the underlying `_pybind_extension` for each version under a config transition.
    3. Aggregates all the versioned extensions into a single target so downstream
       rules (like `pywrap_library`) can bundle them all at once.
    """

    # On OSS we have not needed any python dependencies for these. If that
    # changes we will need to incorporate these dependencies.
    kwargs.pop("py_deps", None)

    # Force wrap_py_init = False to ensure C++ extension code (which depends on PyTorch)
    # is kept in the versioned .so files instead of being merged into the common library.
    kwargs["wrap_py_init"] = False

    # We create the underlying extensions for each PyTorch version
    versioned_targets = []
    for version in PYTORCH_VERSIONS:
        suffix = "_" + version.replace(".", "_")
        actual_name = name + "_actual" + suffix
        _pybind_extension(name = actual_name, **kwargs)

        transitioned_name = name + suffix
        _transitioned_pywrap(
            name = transitioned_name,
            actual = ":" + actual_name,
            torch_version = version,
        )
        versioned_targets.append(":" + transitioned_name)

    # Bundle them all together so the user can just depend on `name`
    # and so that pywrap_library sees the combined CcInfo and PywrapLibraryInfo
    pybind_aggregator(
        name = name,
        deps = versioned_targets,
    )
