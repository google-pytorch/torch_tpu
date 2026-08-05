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

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "tpu_visualizer",
    testonly = True,
    srcs = ["tpu_visualizer.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/utils:hardware",
    ],
)

torch_tpu_py_test(
    name = "tpu_visualizer_test",
    srcs = ["tpu_visualizer_test.py"],
    notest_oss = "Topology differences",
    tags = [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":tpu_visualizer",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed:multiprocessing",
    ],
)

torch_tpu_package_end()
