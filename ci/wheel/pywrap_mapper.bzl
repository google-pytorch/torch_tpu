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
        "_mapper_script": attr.label(
            default = Label("//ci/wheel:wheel_mapper_bin"),
            executable = True,
            cfg = "exec",
        ),
    },
)
