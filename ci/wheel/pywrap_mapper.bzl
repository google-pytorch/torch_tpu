# Copyright 2025 Google LLC
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

"""This module provides a custom Starlark rule to remap build output files for wheels.

This is used to adjust the layout of files created by pywrap_torch_tpu_binaries.
The pywrap binaries provides a .json file that contains a mapping of the binary names and
their original file directories. This rule is used to call a python script to the correct
the location of these files to resolve the original file directory.
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

# Shown when someone reaches the execution phase without --//:wheel_build=True.
# This is deliberately an action that fails rather than a fail() in the rule
# implementation: a fail() runs during analysis, which takes down every
# repo-wide `cquery` -- including the one bazel-diff runs to hash the graph on
# presubmit -- even though nobody asked to build a wheel.
_WHEEL_BUILD_GUARD_MESSAGE = """
================================================================================
The torch_tpu wheel must be built with --//:wheel_build=True.

    bazel build -c opt --config=wheel_common //ci/wheel:torch_tpu_wheel

If you do not have RBE credentials, append --config=no_rbe *after*
--config=wheel_common to strip the remote cache and executor it turns on:

    bazel build -c opt --config=wheel_common //ci/wheel:torch_tpu_wheel \\
        --config=no_rbe

--//:wheel_build=True routes the shared XLA/MLIR backend through the
torch_version-reset transition, so pywrap factors it into a single
libxla_base.so. Built without it, every per-version common ships its own full
copy of the backend and the wheel aborts on `import torch` with duplicate
static registrations. The result is a silently broken wheel, not merely a
differently-laid-out one, which is why this is refused rather than warned about.
================================================================================
"""

_WHEEL_OPT_GUARD_MESSAGE = """
================================================================================
The torch_tpu wheel must be built with compilation mode "-c opt".

Unoptimized builds (-c fastbuild, -c dbg, etc.) can degrade performance and are
not intended for distribution wheels.

If you are developing or debugging and explicitly want an unoptimized wheel,
pass --//:allow_unoptimized_wheel=True on your command line:

    bazel build -c dbg --config=wheel_common //ci/wheel:torch_tpu_wheel \\
        --//:allow_unoptimized_wheel=True
================================================================================
"""

def _fail_action(ctx, out_dir, message, mnemonic, progress_message):
    ctx.actions.run_shell(
        outputs = [out_dir],
        command = "cat >&2 <<'_GUARD_EOF_'{}_GUARD_EOF_\nexit 1\n".format(message),
        mnemonic = mnemonic,
        progress_message = progress_message,
    )
    return [DefaultInfo(files = depset([out_dir]))]

def _remapper_impl(ctx):
    manifests = []
    binaries = []

    for f in ctx.files.srcs:
        if f.extension == "json":
            manifests.append(f)
        else:
            binaries.append(f)

    if not manifests:
        fail("Could not find a .json manifest in srcs. Ensure pywrap_binaries has JSON output.")

    out_dir = ctx.actions.declare_directory(ctx.attr.name + "_pkg")

    # Every wheel build funnels through this rule, so failing the action that
    # produces out_dir makes a mis-factored wheel unbuildable no matter which
    # wheel target was requested.
    if not ctx.attr._wheel_build[BuildSettingInfo].value:
        return _fail_action(
            ctx,
            out_dir,
            _WHEEL_BUILD_GUARD_MESSAGE,
            mnemonic = "PywrapWheelBuildGuard",
            progress_message = "Checking the torch_tpu wheel build configuration",
        )

    if ctx.var.get("COMPILATION_MODE") != "opt" and not ctx.attr._allow_unoptimized_wheel[BuildSettingInfo].value:
        return _fail_action(
            ctx,
            out_dir,
            _WHEEL_OPT_GUARD_MESSAGE,
            mnemonic = "PywrapWheelOptGuard",
            progress_message = "Checking the torch_tpu compilation mode configuration",
        )

    # 3. Construct arguments for the python script. Multiple pywrap_binaries
    # (one per PyTorch version) each contribute a manifest; the per-version
    # common libraries all reference the single shared libxla_base.so.
    args = ctx.actions.args()
    args.add_all(manifests, before_each = "--manifest")
    args.add("--out_dir", out_dir.path)
    args.add_all(binaries)

    # 4. Run the script
    ctx.actions.run(
        inputs = binaries + manifests,
        outputs = [out_dir],
        executable = ctx.executable._mapper_script,
        arguments = [args],
        mnemonic = "PywrapWheelRemap",
        progress_message = "Restructuring Pywrap binaries for Wheel...",
    )

    return [DefaultInfo(files = depset([out_dir]))]

remap_pywrap_binaries = rule(
    implementation = _remapper_impl,
    attrs = {
        "srcs": attr.label_list(mandatory = True, allow_files = True),
        "_wheel_build": attr.label(default = Label("//:wheel_build")),
        "_allow_unoptimized_wheel": attr.label(
            default = Label("//:allow_unoptimized_wheel"),
        ),
        "_mapper_script": attr.label(
            default = Label("//ci/wheel:wheel_mapper_bin"),
            executable = True,
            cfg = "exec",
        ),
    },
)
