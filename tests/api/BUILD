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
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
        "@pypi//pyyaml",
    ],
)

# TODO(b/570277416): Run the tests below in OSS only. They are only built in OSS
# for now, because the scanner only finds the google3 repository root.
torch_tpu_py_test(
    name = "scan_tt_apis_test",
    srcs = ["scan_tt_apis_test.py"],
    notest_oss = True,  # NOTEST_OSS_OK=The scanner only finds the google3 repository root.
    deps = [
        ":scan_tt_apis_lib",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
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
    notest_oss = True,  # NOTEST_OSS_OK=The scanner only finds the google3 repository root.
    tags = [
        # Must run on TPU hardware for C++ extension to be loaded into memory, enabling memory
        # reflection to discover all public runtime APIs.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":scan_tt_apis_lib",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//pyyaml",
    ],
)

# The baselines and the files that scan_exposed_internal_apis scans. The
# scanned directories must match DEFAULT_SCAN_SUBDIRS in
# scan_exposed_internal_apis.py. They are listed per directory because in OSS
# docs/ and examples/ are outside the torch_tpu package, so the root
# all_*_files_recursive filegroups don't include them.
filegroup(
    name = "exposed_internal_apis_data",
    srcs = [
        "exposed_internal_apis.yaml",
        "tt_apis.yaml",
        "//docs/notebooks:all_doc_files_recursive",
        "//docs/notebooks:all_ipynb_files_recursive",
        "//docs/notebooks:all_py_files_recursive",
        "//docs/source:all_doc_files_recursive",
        "//docs/source:all_ipynb_files_recursive",
        "//docs/source:all_py_files_recursive",
        "//examples:all_doc_files_recursive",
        "//examples:all_ipynb_files_recursive",
        "//examples:all_py_files_recursive",
    ],
)

torch_tpu_py_library(
    name = "scan_exposed_internal_apis_lib",
    srcs = ["scan_exposed_internal_apis.py"],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//src/torch_tpu",
        "@pypi//pyyaml",
    ],
)

pytype_strict_binary(
    name = "scan_exposed_internal_apis",
    srcs = ["scan_exposed_internal_apis.py"],
    data = [":exposed_internal_apis_data"],
    main = "scan_exposed_internal_apis.py",
    deps = [
        ":scan_exposed_internal_apis_lib",
    ],
)

# Fails if exposed_internal_apis.yaml is out of sync with the internal APIs and
# environment variables that the docs and examples use. It scans
# docs/notebooks, docs/source and examples.
#
# TODO(b/570277416): Run the tests below in OSS. They are only built in OSS for
# now, because the scanner only finds the google3 repository root.
torch_tpu_py_test(
    name = "exposed_internal_apis_test",
    srcs = ["exposed_internal_apis_test.py"],
    data = [":exposed_internal_apis_data"],
    notest_oss = True,  # NOTEST_OSS_OK=The scanner only finds the google3 repository root.
    deps = [
        ":scan_exposed_internal_apis_lib",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//pyyaml",
    ],
)

torch_tpu_py_test(
    name = "scan_exposed_internal_apis_test",
    srcs = ["scan_exposed_internal_apis_test.py"],
    data = [":exposed_internal_apis_data"],
    notest_oss = True,  # NOTEST_OSS_OK=The scanner only finds the google3 repository root.
    deps = [
        ":scan_exposed_internal_apis_lib",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
    ],
)

torch_tpu_package_end()
