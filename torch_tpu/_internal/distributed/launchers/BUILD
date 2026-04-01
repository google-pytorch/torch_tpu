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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_strict_library(
    name = "singlehost_wrapper",
    srcs = ["singlehost_wrapper.py"],
    deps = [
        "//shims/absl_py/logging",
        "//torch_tpu/_internal/distributed:tpu_topology",
        "@pypi//portpicker",
    ],
)

pytype_strict_library(
    name = "multihost_wrapper",
    srcs = ["multihost_wrapper.py"],
    deps = ["//shims/absl_py/logging"],
)

torch_tpu_py_test(
    name = "singlehost_wrapper_v7x_test",
    srcs = ["singlehost_wrapper_test.py"],
    main = "singlehost_wrapper_test.py",
    nopresubmit = "long Forge queue",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    strict = True,
    tags = [
        "requires-tpu-v7x:4",  # Must use for testing v7x.
    ],
    deps = [
        ":singlehost_wrapper",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "singlehost_wrapper_v5e_test",
    srcs = ["singlehost_wrapper_test.py"],
    main = "singlehost_wrapper_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    strict = True,
    tags = ["requires-tpu-v5lite:8"],
    deps = [
        ":singlehost_wrapper",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "multihost_wrapper_test",
    srcs = ["multihost_wrapper_test.py"],
    strict = True,
    deps = [
        ":multihost_wrapper",
        "//shims/absl_py/testing:absltest",
    ],
)

torch_tpu_package_end()
