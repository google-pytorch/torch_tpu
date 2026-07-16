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

"""Per-version wheel pywrap_library for one PyTorch version.

The multi-ABI wheel ships one thin glue per version in GLUE_TORCH_VERSIONS
(//bazel:pytorch_versions.bzl), each in its own package `glue_<suffix>` so its
default common lib gets a unique basename (the wheel mapper keys on basename)
while the shared xla_base common keeps the same target name in every version.
Those packages are generated into `@torch_tpu_glues` by
//bazel:glue_packages.bzl -- adding a version to GLUE_TORCH_VERSIONS is all it
takes. Because this macro is instantiated from that generated repository, every
main-repo dep below is anchored with Label() (a plain "//..." string would
resolve inside `@torch_tpu_glues`). See `//shims/pybind11:pybind.bzl` for the
surrounding design.
"""

load(
    "@xla//third_party/py/rules_pywrap:pywrap.default.bzl",
    "pywrap_binaries",
    "pywrap_library",
)
load("//bazel:pytorch_versions.bzl", "GLUE_TORCH_VERSIONS")
load("//bazel:pywrap_extensions.bzl", "PYWRAP_EXTENSIONS")
load("//shims/pybind11:pybind.bzl", "versioned_glue_labels_for")

_GLUE_PACKAGE_PREFIX = "glue_"

def versioned_glue_library():
    """Emits the pywrap_library + binaries for the glue package it is called from.

    Call once, with no arguments, from a `//torch_tpu/common/glue_<suffix>/BUILD`.
    The PyTorch version is the `<suffix>` in the package name (e.g. `glue_2_12_1`
    -> `2.12.1`), checked against GLUE_TORCH_VERSIONS.
    """
    leaf = native.package_name().rsplit("/", 1)[-1]
    if not leaf.startswith(_GLUE_PACKAGE_PREFIX):
        fail("versioned_glue_library() must be called from a '{}<version>' package, but got //{}".format(
            _GLUE_PACKAGE_PREFIX,
            native.package_name(),
        ))
    suffix = leaf[len(_GLUE_PACKAGE_PREFIX):]
    version = suffix.replace("_", ".")
    if version not in GLUE_TORCH_VERSIONS:
        fail("glue package //{} selects PyTorch {}, which is not in GLUE_TORCH_VERSIONS {}".format(
            native.package_name(),
            version,
            GLUE_TORCH_VERSIONS,
        ))

    # Named per-version so the default common lib basename (lib<name>_common.so)
    # is unique across versions -- the wheel mapper keys on basename, and the
    # package path is stripped from the common binary's name.
    pywrap_library(
        name = "pywrap_" + suffix,
        common_lib_filters = {
            "torch_tpu/common/xla_base": Label("//torch_tpu/common:xla_base_filter"),
        },
        extra_deps = [Label("//torch_tpu/common:torch_rpath_flags")],
        pywrap_count = len(PYWRAP_EXTENSIONS),
        deps = [
            Label(glue)
            for glue in versioned_glue_labels_for(PYWRAP_EXTENSIONS, version)
        ],
    )
    pywrap_binaries(
        name = "binaries",
        dep = ":pywrap_" + suffix,
    )
