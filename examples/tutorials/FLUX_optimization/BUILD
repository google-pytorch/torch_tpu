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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "flux",
    srcs = ["flux.py"],
    deps = [
        "//shims/torch:pytorch",
        "@pypi//einops",
    ],
)

torch_tpu_py_test(
    name = "flux_test",
    size = "large",
    srcs = ["flux_test.py"],
    nopresubmit = "Slow tutorial",
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v6e",
    ],
    deps = [
        ":flux",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "flux_rope_parity_test",
    size = "large",
    srcs = ["flux_rope_parity_test.py"],
    nopresubmit = "Requires TPU hardware",
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v6e",
    ],
    deps = [
        ":flux",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_package_end()
