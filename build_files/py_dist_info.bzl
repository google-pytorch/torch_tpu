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

"""Rule for generating Python package metadata."""

load("//shims/build_cleaner:build_defs.bzl", "register_extension_info")

METADATA_TEMPLATE = """\
Metadata-Version: 2.1
Name: {name}
Version: {version}
"""

def _py_dist_info_rule_impl(ctx):
    """Implementation of the py_dist_info rule."""
    package_name = ctx.attr.package_name
    if not package_name:
        fail("package_name attribute is required")
    version = ctx.attr.version
    if not version:
        fail("version attribute is required")

    # Prepare entry_points.txt content
    entry_points_lines = []
    for group, points in ctx.attr.entry_points.items():
        entry_points_lines.append("[%s]" % group)
        entry_points_lines.extend(points)
        entry_points_lines.append("")
    entry_points_content = "\n".join(entry_points_lines)

    # Use unique names for staging files to avoid clashes
    entry_points_staging_file = ctx.actions.declare_file(ctx.label.name + "_entry_points.txt")
    ctx.actions.write(entry_points_staging_file, entry_points_content)

    # Prepare METADATA content
    metadata_content = METADATA_TEMPLATE.format(
        name = package_name,
        version = version,
    )
    metadata_staging_file = ctx.actions.declare_file(ctx.label.name + "_METADATA")
    ctx.actions.write(metadata_staging_file, metadata_content)

    # Create runfiles with the .dist-info directory structure
    target_path = ctx.attr.target_path
    path_components = [ctx.workspace_name]
    if target_path:
        path_components.append(target_path)
    path_components.append(package_name + ".dist-info")
    dist_info_dirname = "/".join(path_components)

    metadata_symlink_path = dist_info_dirname + "/METADATA"
    entry_points_symlink_path = dist_info_dirname + "/entry_points.txt"

    runfiles = ctx.runfiles(
        root_symlinks = {
            metadata_symlink_path: metadata_staging_file,
            entry_points_symlink_path: entry_points_staging_file,
        },
    )

    return [
        DefaultInfo(
            # We don't set `files` (default outputs) because the files only
            # make sense to show up where the root symlinks put them. Other
            # locations (ones not on sys.path) wouldn't work.
            runfiles = runfiles,  # Add the runfiles to the DefaultInfo
        ),
    ]

py_dist_info_rule = rule(
    implementation = _py_dist_info_rule_impl,
    attrs = {
        "package_name": attr.string(mandatory = True),
        "version": attr.string(mandatory = True),
        "entry_points": attr.string_list_dict(
            doc = "A dictionary of entry point groups to lists of entry points.",
        ),
        "target_path": attr.string(
            default = "",
            doc = "The target path for the distribution info directory, relative to the workspace root.",
        ),
    },
)

# TODO(pganssle): Pull this instead from a bazel / blaze config option?
_DIST_INFO_TARGET_PATH = ""

def py_dist_info(
        name,
        target_path = _DIST_INFO_TARGET_PATH,
        **kwargs):
    """Declares Python package metadata for importlib.metadata.

    This rule generates the necessary files for Python's `importlib.metadata` to
    discover package information, including entry points. This is particularly
    useful for libraries that rely on `importlib.metadata` for plugin systems or
    to get package information at runtime, avoiding the need to patch them.

    There should be one `py_dist_info` target for each Python package that
    requires metadata to be available at runtime.

    Args:
      name: The name of the target.
      target_path: The target path for the distribution info directory, relative to the workspace root.
      **kwargs: Additional arguments to pass to the rule.
    """
    py_dist_info_rule(
        name = name,
        target_path = target_path,
        **kwargs
    )

register_extension_info(
    extension = py_dist_info,
    label_regex_for_dep = "{extension_name}",
)
