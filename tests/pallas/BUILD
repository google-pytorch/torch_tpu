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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "pallas_test",
    srcs = ["pallas_test.py"],
    # Not supported on vl chips.
    env = {"TPU_PREMAPPED_BUFFER_SIZE": "0"},
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/jax:pallas",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/pallas",
        "//torch_tpu/_internal/utils",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "pallas_communication_test",
    srcs = ["pallas_communication_test.py"],
    env = {"TPU_PREMAPPED_BUFFER_SIZE": "0"},
    # TODO(b/497890629): Enable in OSS
    notest_oss = "fix hang in OSS",  # TODO
    tags = [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/pallas",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "@pypi//jax",
    ],
)

torch_tpu_package_end()
