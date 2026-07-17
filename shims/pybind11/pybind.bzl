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

Besides the extension itself, the macro emits a `<name>_backend_probe`
cc_library mirroring the extension's external backend deps (XLA, MLIR, absl,
... -- everything but PyTorch). The wheel's pywrap_library aggregates the
probes of all its extensions into a `common_lib_filters` entry, factoring the
shared, PyTorch-free backend into a single `libxla_base.so` while the
PyTorch-touching code stays in the default common library (see
//torch_tpu/common:BUILD).
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", _pybind_extension = "pybind_extension")

# The repositories torch_tpu gets PyTorch from. Deps from these must stay out
# of the shared, PyTorch-free backend. The default torch lives in the shared
# @pypi hub (which also holds non-torch backend deps like numpy), so its torch
# *package* is matched specifically rather than the whole repo. Keep in sync
# with the torch pip hubs declared in MODULE.bazel.
_TORCH_DEP_PREFIXES = (
    "@pypi//torch:",
    "@pypi//torch/",
    "@local_torch//",
)

def _is_backend_dep(dep):
    """Whether a dep is part of the shared (non-PyTorch) backend.

    These are external repositories -- XLA, MLIR, StableHLO, Eigen, Abseil,
    protobuf, gRPC, TSL, RE2, Python/numpy headers, etc. -- that compile
    independently of PyTorch. Only @-deps qualify, and our known PyTorch
    sources (_TORCH_DEP_PREFIXES) are excluded. torch_tpu's own code and
    PyTorch reached through //shims/torch are not @-deps, so they never
    land in the backend.
    """
    if type(dep) != "string" or not dep.startswith("@"):
        return False
    for torch_prefix in _TORCH_DEP_PREFIXES:
        if dep.startswith(torch_prefix):
            return False
    return True

def pybind_extension(name, **kwargs):
    # On OSS we have not needed any python dependencies for these. If that
    # changes we will need to incorporate these dependencies.
    kwargs.pop("py_deps", None)
    _pybind_extension(name = name, **kwargs)

    deps = kwargs.get("deps", [])
    if type(deps) != "list":
        # A bare select() (or `[...] + select({...})`) is opaque at macro
        # expansion time: a backend dep hidden inside it cannot be discovered,
        # so it would silently escape the probe and leave its backend code out
        # of the shared base. Refuse the shape rather than mis-split.
        fail(
            "pybind_extension {}: deps must be a plain list (no select()), ".format(name) +
            "so the backend probe can mirror the extension's backend deps.",
        )

    # A plain cc_library mirroring this extension's backend deps, aggregated by
    # the wheel pywrap_library's shared-base filter. Its transitive linker
    # inputs identify exactly the backend code the extensions share, so pywrap
    # factors that (and nothing PyTorch-flavored) into libxla_base.so.
    cc_library(
        name = "{}_backend_probe".format(name),
        deps = [dep for dep in deps if _is_backend_dep(dep)],
        testonly = kwargs.get("testonly"),
        visibility = ["//visibility:public"],
    )

def backend_probe_labels(extension_labels):
    """Maps pybind_extension labels to their `_backend_probe` labels."""
    return ["{}_backend_probe".format(label) for label in extension_labels]
