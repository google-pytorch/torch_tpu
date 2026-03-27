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

"""Integrity check for wheel dependencies."""

InternalDepsInfo = provider(
    "Provider to collect transitive internal labels",
    fields = {"labels": "depset of labels"},
)

def _collect_internal_deps_aspect_impl(target, ctx):
    labels = []

    # Check if the target itself is an internal library
    # In Google3: torch_tpu/...
    # In OSS: root-level (empty package or starting with common/tests/etc)
    pkg = target.label.package
    is_internal = pkg.startswith("torch_tpu") or (not pkg.startswith("external/") and not pkg.startswith("@"))

    if is_internal:
        # We want to avoid collecting the test itself or the wheel/shim targets
        if not target.label.name.endswith("_test") and not "_validation" in target.label.name:
            labels.append(target.label)

    # If the target has a private attribute where we store original deps (see build_defs.bzl)
    # this allows us to see deps that were removed by the macro for wheel testing.
    transitive = []

    # Traverse standard deps and srcs
    for attr in ["deps", "srcs", "_original_deps"]:
        if hasattr(ctx.rule.attr, attr):
            for dep in getattr(ctx.rule.attr, attr):
                if InternalDepsInfo in dep:
                    transitive.append(dep[InternalDepsInfo].labels)

    return [InternalDepsInfo(
        labels = depset(labels, transitive = transitive),
    )]

collect_internal_deps_aspect = aspect(
    implementation = _collect_internal_deps_aspect_impl,
    attr_aspects = ["deps", "srcs", "_original_deps"],
)

def _wheel_integrity_test_impl(ctx):
    # Get all allowed labels (Wheel + Extra)
    allowed_labels = {}
    for target in ctx.attr.allowed_targets:
        if InternalDepsInfo in target:
            for label in target[InternalDepsInfo].labels.to_list():
                allowed_labels[str(label)] = True

    # Collect all required labels from tests that are marked for validation
    all_required = []
    for test_target in ctx.attr.tests:
        # We only care about the dependencies of our 'validation' filegroups
        # which were created by the torch_tpu_py_test macro.
        if "wheel_validation" in test_target.label.name or "wheel_validation" in ctx.rule.attr.tags:
            if InternalDepsInfo in test_target:
                all_required.append(test_target[InternalDepsInfo].labels)

    required_depset = depset(transitive = all_required)

    missing = []
    for label in required_depset.to_list():
        # Only check internal torch_tpu targets
        label_str = str(label)
        if label_str.startswith("//torch_tpu") or label_str.startswith("@//torch_tpu"):
            # Exclude some known non-library targets if necessary
            if label_str.endswith(":license") or label_str.endswith(":pyproject") or ":test_fixtures" in label_str:
                continue

            # Normalize label string if needed (some may have @ prefix or different formatting)
            # Standardize on //torch_tpu/...
            normalized = label_str
            if normalized.startswith("@//"):
                normalized = normalized[1:]

            if normalized not in allowed_labels:
                missing.append(normalized)

    if not missing:
        script_content = "#!/bin/bash\nexit 0\n"
    else:
        # Sort and unique missing
        missing = sorted({m: None for m in missing}.keys())

        error_msg = "\\n[!] WHEEL INTEGRITY ERROR: Missing dependencies for OSS tests!\\n"
        error_msg += "The following internal libraries are required by oss_ready tests but are NOT in the wheel:\\n"
        for m in missing:
            error_msg += "  - %s\\n" % m
        error_msg += "\\nACTION REQUIRED:\\n"
        error_msg += "1. If it's core library code -> Add to 'all_torch_tpu_files' in //ci/wheel/BUILD.bazel\\n"
        error_msg += "2. If it's a test helper -> Ensure it is a 'torch_tpu_py_library' so it redirects to the wheel.\\n"

        script_content = "#!/bin/bash\necho -e \"%s\"\nexit 1\n" % error_msg

    test_runner = ctx.actions.declare_file(ctx.label.name + "_runner.sh")
    ctx.actions.write(
        output = test_runner,
        content = script_content,
        is_executable = True,
    )

    return [DefaultInfo(
        executable = test_runner,
    )]

wheel_integrity_test = rule(
    implementation = _wheel_integrity_test_impl,
    attrs = {
        "tests": attr.label_list(
            aspects = [collect_internal_deps_aspect],
            mandatory = True,
        ),
        "allowed_targets": attr.label_list(
            aspects = [collect_internal_deps_aspect],
            mandatory = True,
            doc = "Targets that provide files allowed to be missing from the test environment (Wheel + Extras)",
        ),
    },
    test = True,
)
