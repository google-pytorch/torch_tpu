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

"""select() builders and rules for consuming PyTorch wheels from rules_python hubs.

The multi-ABI wheel builds one glue per PyTorch version (see
//shims/pybind11:pybind.bzl). For each extra (non-default) version, torch deps
must resolve to that version's wheel; the default version falls through to the
caller-supplied base rows (local torch, CUDA torch, or default PyPI torch).

`_torch_whl_cc_headers` and `_torch_whl_cc_lib` wrap each hub's standard
`:extracted_whl_files` filegroup directly from `//shims/torch` so wheel hubs do
not need custom BUILD modifications or per-target hub aliases.

The extra versions come from //bazel:pytorch_versions.bzl, the single source of
truth, so adding a version updates every torch shim at once.
"""

load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//bazel:pytorch_versions.bzl", "EXTRA_PYTORCH_VERSIONS", "NIGHTLY_TORCH_VERSION")

# Every non-default version needing its own select() row: the extra release
# versions plus the nightly channel's, whose dev-snapshot wheel comes from its
# own @pypi_torch_<version> hub exactly like a release wheel.
_EXTRA_GLUE_VERSIONS = EXTRA_PYTORCH_VERSIONS + [NIGHTLY_TORCH_VERSION]

_HEADER_EXTENSIONS = (
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
    ".cuh",
    ".inc",
    ".inl",
    ".tcc",
)

_TORCH_INCLUDE_SUBDIRS = (
    "site-packages/torch/include",
    "site-packages/torch/include/torch/csrc/api/include",
    "site-packages/torch/include/kineto",
)

_WHEEL_LIB_NAMES = (
    "libc10",
    "libtorch_cpu",
    "libtorch_python",
)

def _is_torch_header(rel_under_torch):
    # Exclude PyTorch's bundled google/protobuf headers from declared inputs so
    # they do not shadow @com_google_protobuf in sandboxed/remote builds.
    return (
        not rel_under_torch.startswith("include/google/protobuf/") and
        rel_under_torch.endswith(_HEADER_EXTENSIONS)
    )

def _torch_whl_cc_headers_impl(ctx):
    hdrs = []
    repo_root = None

    # :extracted_whl_files is a glob(["**"]) of files in a single wheel
    # repository, so every matching file shares the same repository root.
    for f in ctx.files.whl_files:
        root, sep, rel_under_torch = f.path.partition("/site-packages/torch/")
        if not sep:
            continue
        if repo_root == None:
            repo_root = root
        if _is_torch_header(rel_under_torch):
            hdrs.append(f)

    if not hdrs:
        fail("No PyTorch header files found under site-packages/torch/ in {}".format(
            ctx.attr.whl_files.label,
        ))

    system_includes = [
        "{}/{}".format(repo_root, subdir)
        for subdir in _TORCH_INCLUDE_SUBDIRS
    ]

    compilation_context = cc_common.create_compilation_context(
        headers = depset(hdrs),
        system_includes = depset(system_includes),
    )
    return [
        DefaultInfo(files = depset(hdrs)),
        CcInfo(compilation_context = compilation_context),
    ]

_torch_whl_cc_headers = rule(
    implementation = _torch_whl_cc_headers_impl,
    doc = "Exposes C/C++ headers from a rules_python torch wheel's extracted_whl_files.",
    attrs = {
        "whl_files": attr.label(
            mandatory = True,
            doc = "The extracted_whl_files target from a torch wheel repository.",
        ),
    },
)

torch_whl_cc_headers_for_testing = _torch_whl_cc_headers

def _torch_whl_cc_lib_impl(ctx):
    so_suffix = "/site-packages/torch/lib/{}.so".format(ctx.attr.lib_name)
    matched = [
        f
        for f in ctx.files.whl_files
        if f.path.endswith(so_suffix)
    ]
    if len(matched) != 1:
        fail("Expected exactly 1 shared library for {} in {}, found: {}".format(
            ctx.attr.lib_name,
            ctx.attr.whl_files.label,
            [f.path for f in matched],
        ))
    lib_file = matched[0]

    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    library_to_link = cc_common.create_library_to_link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        dynamic_library = lib_file,
    )
    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset([library_to_link]),
    )
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([linker_input]),
    )
    return [
        DefaultInfo(
            files = depset([lib_file]),
            runfiles = ctx.runfiles(files = [lib_file]),
        ),
        CcInfo(linking_context = linking_context),
    ]

_torch_whl_cc_lib = rule(
    implementation = _torch_whl_cc_lib_impl,
    doc = "Exposes a shared library from a rules_python torch wheel's extracted_whl_files.",
    attrs = {
        "lib_name": attr.string(
            mandatory = True,
            doc = "Shared library stem under site-packages/torch/lib/, e.g. 'libc10'.",
        ),
        "whl_files": attr.label(
            mandatory = True,
            doc = "The extracted_whl_files target from a torch wheel repository.",
        ),
        "_cc_toolchain": attr.label(
            default = Label("@rules_cc//cc:current_cc_toolchain"),
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
)

torch_whl_cc_lib_for_testing = _torch_whl_cc_lib

def _hub_target_name(prefix, target):
    return "{}_{}".format(prefix, target)

def _define_hub_torch_targets(prefix, whl_files):
    # Mark hub wrapper targets manual so wildcard builds (e.g. `bazel build //...`
    # or `//shims/torch:all`) only fetch the wheel hub selected by the active
    # configuration rather than every extra-version, CUDA, and nightly wheel.
    _torch_whl_cc_headers(
        name = _hub_target_name(prefix, "torch_headers"),
        tags = ["manual"],
        whl_files = whl_files,
    )
    for lib_name in _WHEEL_LIB_NAMES:
        _torch_whl_cc_lib(
            name = _hub_target_name(prefix, lib_name),
            lib_name = lib_name,
            tags = ["manual"],
            whl_files = whl_files,
        )

# buildifier: disable=unnamed-macro
def define_wheel_torch_targets():
    """Defines C++ header and shared-library targets over each torch wheel hub."""
    _define_hub_torch_targets("pypi", "@pypi//torch:extracted_whl_files")
    _define_hub_torch_targets("pypi_cuda", "@pypi_cuda//torch:extracted_whl_files")
    for v in _EXTRA_GLUE_VERSIONS:
        ver = v.replace(".", "_")
        _define_hub_torch_targets(
            "pypi_torch_" + ver,
            "@pypi_torch_{}//torch:extracted_whl_files".format(ver),
        )

def _torch_version_setting(version):
    return "//:torch_" + version.replace(".", "_")

def _extra_wheel(version, target):
    prefix = "pypi_torch_" + version.replace(".", "_")
    return "//shims/torch:" + _hub_target_name(prefix, target)

def torch_lib_select(target, base):
    """A select() dict choosing `target` from each extra version's wheel.

    For single-label attributes (e.g. an alias `actual`). `base` supplies the
    rows for the default torch version and is returned unchanged for it.

    Args:
      target: the wheel target name to pick, e.g. "libc10".
      base: the select() dict for the default torch version.
    """
    rows = {
        _torch_version_setting(v): _extra_wheel(v, target)
        for v in _EXTRA_GLUE_VERSIONS
    }
    rows.update(base)
    return rows

def torch_headers_select(base, extra = []):
    """A select() dict choosing torch_headers from each extra version's wheel.

    For list-valued header `deps`. Each extra-version row is
    ["//shims/torch:pypi_torch_<ver>_torch_headers"] + `extra`; `base` supplies
    the default-version rows and is returned unchanged.

    Args:
      base: the select() dict for the default torch version.
      extra: labels appended to every extra-version row (e.g. the Python C
        headers that accompany torch_headers).
    """
    rows = {
        _torch_version_setting(v): [_extra_wheel(v, "torch_headers")] + extra
        for v in _EXTRA_GLUE_VERSIONS
    }
    rows.update(base)
    return rows
