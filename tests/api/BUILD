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

"""Tests and tools that guard the TorchTPU public API surface."""

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)
load("//build_files:wheel.bzl", "WHEEL_PY_TARGETS")
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

# The API baseline, which the internal exposed internal API scanner also reads.
exports_files(
    ["tt_apis.yaml"],
    # copybara:uncomment visibility = ["//src/torch_tpu/google/tools:__pkg__"],
)

# Everything that the wheel ships, so that scan_tt_apis can import all of its
# public modules.
torch_tpu_py_library(
    name = "wheel_py_targets",
    deps = WHEEL_PY_TARGETS,
)

torch_tpu_py_library(
    name = "scan_tt_apis_lib",
    srcs = ["scan_tt_apis.py"],
    deps = [
        # Imported at run time.
        ":wheel_py_targets",  # build_cleaner: keep
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests:repo_root",
        "@pypi//pyyaml",
    ],
)

# Updates tt_apis.yaml and docs/markdown/public_apis.md in the workspace. Run it
# on a TPU host with `blaze run` in google3 or `bazel run` in OSS.
pytype_strict_binary(
    name = "scan_tt_apis",
    srcs = ["scan_tt_apis.py"],
    data = [
        "tt_apis.yaml",
        "//csrc:all_cpp_files_recursive",
        "//docs/markdown:public_apis.md",
    ],
    main = "scan_tt_apis.py",
    deps = [
        ":scan_tt_apis_lib",
    ],
)

torch_tpu_py_test(
    name = "scan_tt_apis_test",
    srcs = ["scan_tt_apis_test.py"],
    deps = [
        ":scan_tt_apis_lib",
        "//shims/absl_py:app",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests:repo_root",
        "//tests:seed_test_utils",
        "@pypi//typing_extensions",
    ],
)

# Fails if tt_apis.yaml is out of sync with the discovered TorchTPU public APIs. Any change to
# tt_apis.yaml requires approval from the owners listed in OWNERS.
torch_tpu_py_test(
    name = "tt_apis_test",
    srcs = ["tt_apis_test.py"],
    data = [
        "tt_apis.yaml",
        "//csrc:all_cpp_files_recursive",
        "//docs/markdown:public_apis.md",
    ],
    tags = [
        # Must run on TPU hardware for C++ extension to be loaded into memory, enabling memory
        # reflection to discover all public runtime APIs.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":scan_tt_apis_lib",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:repo_root",
        "//tests:seed_test_utils",
        "@pypi//pyyaml",
    ],
)

torch_tpu_package_end()
