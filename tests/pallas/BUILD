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
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/jax:pallas",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/pallas",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "pallas_communication_test",
    srcs = ["pallas_communication_test.py"],
    env = {
        "ACCELERATOR_TYPE": "v5litepod-8",  # Needed for OSS.
    },
    shard_count = 1,
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/distributed:tpu_env",
        "//torch_tpu/_internal/pallas",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "@pypi//jax",
    ],
)

torch_tpu_package_end()
