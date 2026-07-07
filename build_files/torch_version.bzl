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

The `//:torch_version` flag reconfigures the whole build graph, including the
XLA/MLIR backend that does not actually depend on PyTorch. To keep the shared
backend in a single configuration (so pywrap factors it into one common library
instead of one copy per PyTorch version), `xla_fixed` re-exports a target under
a transition that pins `//:torch_version` to a canonical value.
"""

_TORCH_VERSION = "//:torch_version"

# Canonical value the version-independent backend is always built under.
_FIXED_TORCH_VERSION = "2.11.0"

def _fix_torch_version_impl(settings, attr):
    return {_TORCH_VERSION: _FIXED_TORCH_VERSION}

_fix_torch_version = transition(
    implementation = _fix_torch_version_impl,
    inputs = [],
    outputs = [_TORCH_VERSION],
)

def _xla_fixed_impl(ctx):
    # Only DefaultInfo and CcInfo are forwarded -- enough for the cc_library
    # backend deps this routes (XLA, MLIR, ...). Any other provider a dep carries
    # (OutputGroupInfo, CcSharedLibraryInfo, aspect providers, ...) is dropped, so
    # do not route deps that are consumed for anything beyond compile/link inputs.
    dep = ctx.attr.dep[0]
    providers = [dep[DefaultInfo]]
    if CcInfo in dep:
        providers.append(dep[CcInfo])
    return providers

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
