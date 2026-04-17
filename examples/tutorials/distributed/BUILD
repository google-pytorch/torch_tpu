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
    name = "hello_world_distributed",
    srcs = ["hello_world_distributed.py"],
    deps = ["//shims/torch:pytorch"],
)

torch_tpu_py_test(
    name = "run_hello_world_distributed",
    srcs = ["run_hello_world_distributed.py"],
    env = {
        "WORLD_SIZE": "8",
    },
    notap = "this is a manual test",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":hello_world_distributed",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_package_end()
