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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "launch",
    srcs = ["launch.py"],
    env = {
        "JOB_COMPLETION_INDEX": "0",
        "TORCH_TPU_TOPOLOGY": "2,2,2",
        "TPU_WORKER_HOSTNAMES": "localhost",
    },
    main = "launch.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "noasan",  # OOMs even with the highest memory limit.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":tp_worker",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:multihost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

pytype_strict_library(
    name = "tp_worker",
    srcs = ["tp_worker.py"],
    deps = [
        ":model",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

pytype_strict_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_package_end()
