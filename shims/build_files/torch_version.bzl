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

"""Transitions for building torch_tpu against multiple PyTorch versions.

The `//:_torch_version` flag reconfigures the whole build graph, including the
XLA/MLIR backend that does not actually depend on PyTorch. To keep the shared
backend in a single configuration (so pywrap factors it into one common library
instead of one copy per PyTorch version), `reset_torch_config` re-exports a
target under a transition that pins `//:_torch_version` to a canonical value.

`torch_version_glue` is the dual: it re-exports a `pybind_extension` under a
transition that *sets* `//:_torch_version` to a specific value, so the same
extension source is compiled once per supported PyTorch version. The versioned
`.so`s (built by the wheel `pywrap_library`) share the single XLA/MLIR common
library produced under the pinned config above.
"""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:cc_shared_library_info.bzl", "CcSharedLibraryInfo")
load("@rules_ml_toolchain//py/rules_pywrap:pywrap.impl.bzl", "PywrapInfo")
load("@rules_python//python:py_info.bzl", "PyInfo")
load(
    "//bazel:pytorch_versions.bzl",
    "DEFAULT_TORCH_VERSION",
    "SENTINEL_TORCH_VERSION",
    _GLUE_TORCH_VERSIONS = "GLUE_TORCH_VERSIONS",
    _WHEEL_TORCH_VERSIONS = "WHEEL_TORCH_VERSIONS",
)

_TORCH_VERSION = "//:_torch_version"
_TORCH_HEADERS_ONLY = "//:_torch_headers_only"

# The PyTorch versions the separated pip wheel is built against (one thin glue
# `.so` per version, all sharing one XLA/MLIR common library) are the single
# source of truth in //bazel:pytorch_versions.bzl. They are re-exported here --
# Starlark does not re-export a loaded symbol, so we load it under a private
# alias and bind a real public global -- because shims/pybind11/pybind.bzl and
# torch_tpu/common/BUILD already load WHEEL_TORCH_VERSIONS from this module.
WHEEL_TORCH_VERSIONS = _WHEEL_TORCH_VERSIONS

# All buildable glues: the released versions plus the nightly channel's, which
# flows through the same machinery under the release version its dev snapshot
# leads up to. This is what the per-glue fan-out iterates.
GLUE_TORCH_VERSIONS = _GLUE_TORCH_VERSIONS

# Canonical values the version-independent backend is always built under. Both
# axes are pinned so the backend has one configuration no matter which glue (or
# the xla_base filter) reaches it: torch_version because the backend does not
# depend on PyTorch, and torch_headers_only so a glue that links real torch and
# the headers-only xla_base filter still produce byte-identical backend linker
# inputs. headers_only is pinned to True specifically -- the shared base must
# never pull real torch, or any stray torch reference in it would bind to one
# version's ABI and break against a different installed PyTorch. (The
# per-version torch code, which does link real torch, is not part of the backend
# and is not routed through here.)
#
# torch_version is pinned to the sentinel -- the flag's default, so the pin is
# a reset to the top-level configuration's value rather than a divergence from
# it. In particular, bumping DEFAULT_TORCH_VERSION does not change this
# configuration, so it does not force the version-independent backend to
# rebuild.
_FIXED_TORCH_VERSION = SENTINEL_TORCH_VERSION
_FIXED_TORCH_HEADERS_ONLY = True

def version_suffix(version):
    """Turns a PyTorch version like "2.12.1" into a target/file suffix "2_12_1"."""
    return version.replace(".", "_")

def _fix_torch_version_impl(_settings, _attr):
    return {
        _TORCH_VERSION: _FIXED_TORCH_VERSION,
        _TORCH_HEADERS_ONLY: _FIXED_TORCH_HEADERS_ONLY,
    }

_fix_torch_version = transition(
    implementation = _fix_torch_version_impl,
    inputs = [],
    outputs = [_TORCH_VERSION, _TORCH_HEADERS_ONLY],
)

# Every provider a cc_library can carry that a consumer might rely on:
# compile/link inputs (CcInfo), coverage and sanitizer instrumentation
# (InstrumentedFilesInfo), secondary/debug outputs (OutputGroupInfo), and
# shared-library metadata (CcSharedLibraryInfo). Aspect-provided providers
# cannot be re-exported from a rule and are still dropped.
_FORWARDED_PROVIDERS = [
    CcInfo,
    InstrumentedFilesInfo,
    OutputGroupInfo,
    CcSharedLibraryInfo,
]

def _reset_torch_config_impl(ctx):
    dep = ctx.attr.dep[0]
    return [dep[DefaultInfo]] + [dep[p] for p in _FORWARDED_PROVIDERS if p in dep]

reset_torch_config = rule(
    implementation = _reset_torch_config_impl,
    attrs = {
        "dep": attr.label(cfg = _fix_torch_version, mandatory = True),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

# The repositories torch_tpu gets PyTorch from. Deps from these must stay
# per-PyTorch-version, so they are the only @-deps kept out of the shared,
# version-independent backend. The default torch lives in the shared @pypi hub
# (which also holds non-torch backend deps like numpy), so its torch *package* is
# matched specifically rather than the whole repo. Keep in sync with the torch
# pip hubs declared in MODULE.bazel.
_TORCH_DEP_PREFIXES = (
    "@pypi_torch_",
    "@local_torch//",
)

def is_backend_dep(dep):
    """Whether a dep is part of the version-independent (non-PyTorch) backend.

    These are external repositories -- XLA, MLIR, StableHLO, Eigen, Abseil,
    protobuf, gRPC, TSL, RE2, Python/numpy headers, etc. -- that compile the same
    regardless of the PyTorch version, so they can be pinned to one config. Only
    @-deps qualify, and our known PyTorch sources (_TORCH_DEP_PREFIXES) are
    Excluded so they stay per-version. torch_tpu's own code is not an @-dep,
    so it is never pinned.

    Args:
      dep: String representing a bazel label.

    Returns:
      True if the dep is a version-independent backend dep, False otherwise.
    """
    if type(dep) != "string" or not dep.startswith("@"):
        return False
    for torch_prefix in _TORCH_DEP_PREFIXES:
        if dep.startswith(torch_prefix):
            return False
    return True

def _set_torch_version_impl(_settings, attr):
    # Only the torch_version is set: the glue links real torch (headers_only stays
    # at its default False), so its per-version common carries a DT_NEEDED on
    # libc10/libtorch_cpu resolved from the user's install at load. Just the
    # shared xla_base filter opts into headers-only, to stay torch-free.
    #
    # The default version maps to the sentinel -- its torch is the default pip
    # hub's, exactly what every select() serves at the flag's default value, so
    # its glue builds in the top-level configuration instead of forking an
    # identical one. Only the non-default versions need a config of their own.
    if attr.torch_version == DEFAULT_TORCH_VERSION:
        return {_TORCH_VERSION: SENTINEL_TORCH_VERSION}
    return {_TORCH_VERSION: attr.torch_version}

_set_torch_version = transition(
    implementation = _set_torch_version_impl,
    inputs = [],
    outputs = [_TORCH_VERSION],
)

def _torch_version_glue_impl(ctx):
    # Re-export the transitioned extension's providers unchanged. Keeping the
    # child's PywrapInfo (rather than reconstructing it) preserves its `owner`,
    # which pywrap_binaries uses to name the emitted `.so` -- so a child named
    # `env_2_12_1` yields `env_2_12_1.so` with the unversioned `PyInit_env`.
    child = ctx.attr.dep[0]
    return [
        child[PywrapInfo],
        child[PyInfo],
        child[DefaultInfo],
    ]

torch_version_glue = rule(
    implementation = _torch_version_glue_impl,
    doc = "Re-exports a pybind_extension compiled against a specific PyTorch version.",
    attrs = {
        "dep": attr.label(
            cfg = _set_torch_version,
            providers = [PywrapInfo],
            mandatory = True,
        ),
        "torch_version": attr.string(mandatory = True),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def pin_glue_backend_deps(name, deps):
    """Pins a versioned glue's external backend deps to the canonical config.

    A per-version glue is analyzed under a `//:_torch_version` transition, which
    reconfigures its *entire* dep graph -- including version-independent backend
    code that a glue depends on directly (e.g. `@com_google_absl//...`). Left
    alone, each version's copy lands in the shared common library under a
    distinct config and collides at link time ("duplicate symbol ... built in a
    different configuration"). Routing those deps through `reset_torch_config`
    resets them to the canonical config so pywrap factors them into one common
    library.

    torch_tpu_cc_library deps already self-route; this covers the glue's own
    direct backend deps. `//`-deps and torch headers are left untouched.

    Args:
      name: The name of the target being defined.
      deps: The original list of dependencies.

    Returns:
      A list of dependencies with backend deps routed through the fixed
      torch_version transition.
    """
    if type(deps) != "list":
        # A non-list deps value -- a bare select(), or a `[...] + select({...})`
        # SelectorList -- is opaque to Starlark: it cannot be iterated or
        # decomposed, so a backend @-dep hidden inside it cannot be discovered
        # and routed through reset_torch_config. Passing it through would let
        # such a dep silently escape pinning and fragment the shared XLA base,
        # so refuse the shape outright rather than miscompile. Keep every glue
        # dep in a plain list; select() is not supported here.
        fail(
            ("pin_glue_backend_deps {}: `deps` must be a plain list so backend " +
             "@-deps can be routed to the shared XLA base, but got a select(). " +
             "Backend deps hidden in a select cannot be pinned; keep deps a " +
             "plain list.").format(name),
        )
    result = []
    fixed_index = 0
    for dep in deps:
        if is_backend_dep(dep):
            fixed_name = "_{}_glue_backend_fixed_{}".format(name, fixed_index)
            reset_torch_config(
                name = fixed_name,
                dep = dep,
                visibility = ["//visibility:private"],
            )
            result.append(":" + fixed_name)
            fixed_index += 1
        else:
            result.append(dep)
    return result

def _xla_base_config_impl(_settings, _attr):
    # The filter builds under the canonical pinned config: its backend deps --
    # routed through reset_torch_config, exactly like the glues' -- share
    # linker-input identity with the glues' pinned backend, so the shared
    # XLA/MLIR is captured. Its own torch_tpu object code must match NO glue, so pywrap
    # drops it: the extra versions differ on torch_version (the sentinel never
    # appears in WHEEL_TORCH_VERSIONS), and the default version's glue -- which
    # also builds at the sentinel -- differs on torch_headers_only, which is
    # True here and always False for a glue.
    return {
        _TORCH_VERSION: SENTINEL_TORCH_VERSION,
        # headers-only so no real torch is pulled into the shared XLA base. The
        # glues themselves link real torch (headers_only=False), but
        # reset_torch_config canonicalizes both torch_version and
        # torch_headers_only, so the backend linker inputs captured here still
        # match the glues' pinned backend.
        _TORCH_HEADERS_ONLY: True,
    }

_xla_base_config = transition(
    implementation = _xla_base_config_impl,
    inputs = [],
    outputs = [_TORCH_VERSION, _TORCH_HEADERS_ONLY],
)

def _xla_base_filter_impl(ctx):
    dep = ctx.attr.dep[0]
    return [dep[CcInfo], dep[DefaultInfo]]

xla_base_filter = rule(
    implementation = _xla_base_filter_impl,
    doc = """Re-exports a cc dep under the shared-XLA-base config.

    Used as a pywrap `common_lib_filters` entry: its transitive linker inputs
    that match the glues' (the pinned XLA/MLIR/absl) are factored into the shared
    `libxla_base.so`; its torch_tpu object code, compiled under the sentinel
    torch_version, matches no glue and is dropped.
    """,
    attrs = {
        "dep": attr.label(
            cfg = _xla_base_config,
            providers = [CcInfo],
            mandatory = True,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
