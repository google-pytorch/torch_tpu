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

"""Build rule to split monolithic GPU golden files into a directory artifact."""

def _split_gpu_goldens_impl(ctx):
    out_dir = ctx.actions.declare_directory(ctx.attr.name)

    args = ctx.actions.args()
    args.add("--output_dir", out_dir.path)
    args.add_all("--base_input_files", ctx.files.base_srcs)
    if ctx.files.hopper_srcs:
        args.add_all("--hopper_input_files", ctx.files.hopper_srcs)

    ctx.actions.run(
        inputs = ctx.files.base_srcs + ctx.files.hopper_srcs,
        outputs = [out_dir],
        executable = ctx.executable._tool,
        arguments = [args],
        mnemonic = "SplitGpuGoldens",
        progress_message = "Splitting goldens into %s" % ctx.attr.name,
    )
    return [
        DefaultInfo(
            files = depset([out_dir]),
            runfiles = ctx.runfiles(files = [out_dir]),
        ),
    ]

split_gpu_goldens = rule(
    implementation = _split_gpu_goldens_impl,
    attrs = {
        "base_srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = "Monolithic golden .gz files recorded on the base GPU.",
        ),
        "hopper_srcs": attr.label_list(
            allow_files = True,
            doc = "Monolithic golden .gz files recorded on Hopper GPUs.",
        ),
        "_tool": attr.label(
            default = "//tests:split_gpu_goldens_bin",
            executable = True,
            cfg = "exec",
        ),
    },
    doc = """Splits monolithic GPU golden .gz files into per-test-case plists.

Outputs a directory artifact named after the target, containing one binary
plist file per test case:
  <name>/
    <test_case_name_1>.plist
    <test_case_name_2>.plist
    ...

Each `<test_case_name>.plist` file contains the golden dictionary for that
specific test method:
  `variant_name` -> `dtype_key` -> list of samples.
""",
)
