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

# Domain experts for this package:
# leehowes
# zhanyong-wan

load(
    "//build_files:accelerator_specification.bzl",
    "tpus",
)
load(
    "//build_files:build_defs.bzl",
    "torch_tpu_oss_doc_test",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "launch",
    srcs = ["launch.py"],
    compatible_accelerators = tpus(min_chips = 4),
    env = {
        "JOB_COMPLETION_INDEX": "0",
        "TPU_WORKER_HOSTNAMES": "localhost",
    },
    main = "launch.py",
    nopresubmit = "slow example",
    tags = [
        "noasan",  # OOMs even with the highest memory limit.
        "requires-mem:28g",
    ],
    deps = [
        ":dtp_worker",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "dtp_worker",
    testonly = True,
    srcs = ["dtp_worker.py"],
    deps = [
        ":model",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

# OSS only: run the documented command for the test above as a user would; see
# torch_tpu_oss_doc_test.
torch_tpu_oss_doc_test(
    name = "launch_doc_test",
    doc = "README.md",
    deps = [
        ":dtp_worker",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_package_end()
