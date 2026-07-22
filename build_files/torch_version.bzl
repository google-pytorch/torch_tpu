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
instead of one copy per PyTorch version), `xla_fixed` re-exports a target under
a transition that pins `//:_torch_version` to a canonical value.

`torch_versioned` is the dual: it re-exports a target under a transition that
*sets* `//:_torch_version` to a specific value, so the same source is compiled
once per supported PyTorch version. The per-version glue extensions of the
multi-ABI wheel will ride on the same transition; today its consumer is the
torch-version probe test (//tests:torch_version_probe_test), which proves each
configured hub really delivers the PyTorch it claims.
"""

load(
    "//bazel:pytorch_versions.bzl",
    "DEFAULT_TORCH_VERSION",
    "SENTINEL_TORCH_VERSION",
    _WHEEL_TORCH_VERSIONS = "WHEEL_TORCH_VERSIONS",
)

_TORCH_VERSION = "//:_torch_version"
_TORCH_HEADERS_ONLY = "//:_torch_headers_only"

# The PyTorch versions the separated pip wheel is built against (one thin glue
# `.so` per version, all sharing one XLA/MLIR common library) are the single
# source of truth in //bazel:pytorch_versions.bzl. They are re-exported here --
# Starlark does not re-export a loaded symbol, so we load it under a private
# alias and bind a real public global -- so consumers of the version machinery
# (the probe test today, the per-version glue build later) have one module to
# load it from.
WHEEL_TORCH_VERSIONS = _WHEEL_TORCH_VERSIONS

# Canonical values the version-independent backend is always built under. Both
# axes are pinned so the backend has one configuration no matter which
# versioned consumer reaches it: torch_version because the backend does not
# depend on PyTorch, and torch_headers_only so consumers that link real torch
# and headers-only ones still produce byte-identical backend linker
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

def _fix_torch_version_impl(settings, attr):
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

def _xla_fixed_impl(ctx):
    dep = ctx.attr.dep[0]
    return [dep[DefaultInfo]] + [dep[p] for p in _FORWARDED_PROVIDERS if p in dep]

xla_fixed = rule(
    implementation = _xla_fixed_impl,
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
    "@pypi//torch:",
    "@pypi//torch/",
    "@pypi_torch_",
    "@local_torch//",
)

def is_backend_dep(dep):
    """Whether a dep is part of the version-independent (non-PyTorch) backend.

    These are external repositories -- XLA, MLIR, StableHLO, Eigen, Abseil,
    protobuf, gRPC, TSL, RE2, Python/numpy headers, etc. -- that compile the same
    regardless of the PyTorch version, so they can be pinned to one config. Only
    @-deps qualify, and our known PyTorch sources (_TORCH_DEP_PREFIXES) are
    excluded so they stay per-version. torch_tpu's own code and PyTorch reached
    through //shims/torch are not @-deps, so they are never pinned.
    """
    if type(dep) != "string" or not dep.startswith("@"):
        return False
    for torch_prefix in _TORCH_DEP_PREFIXES:
        if dep.startswith(torch_prefix):
            return False
    return True

def _set_torch_version_impl(settings, attr):
    # Only the torch_version is set: a versioned target links real torch
    # (headers_only stays at its default False), so its output carries a
    # DT_NEEDED on libc10/libtorch_cpu resolved from the user's install at
    # load.
    #
    # The default version maps to the sentinel -- its torch is the default pip
    # hub's, exactly what every select() serves at the flag's default value, so
    # it builds in the top-level configuration instead of forking an identical
    # one. Only the non-default versions need a config of their own.
    if attr.torch_version == DEFAULT_TORCH_VERSION:
        return {_TORCH_VERSION: SENTINEL_TORCH_VERSION}
    return {_TORCH_VERSION: attr.torch_version}

_set_torch_version = transition(
    implementation = _set_torch_version_impl,
    inputs = [],
    outputs = [_TORCH_VERSION],
)

def _torch_versioned_impl(ctx):
    dep = ctx.attr.dep[0]
    return [dep[DefaultInfo]] + [dep[p] for p in _FORWARDED_PROVIDERS if p in dep]

torch_versioned = rule(
    implementation = _torch_versioned_impl,
    doc = "Re-exports a target built against a specific PyTorch version.",
    attrs = {
        "dep": attr.label(cfg = _set_torch_version, mandatory = True),
        "torch_version": attr.string(mandatory = True),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
