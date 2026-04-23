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
load("//shims/pynball:build_defs.bzl", "ipynb_deps")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "hello_world_llm_eager_test",
    srcs = ["hello_world_llm_eager_test.py"],
    notap = "this is a manual test",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "no-sandbox",  # Bazel's linux-sandbox restricts write access to your home directory for downloading weights.
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//accelerate",  # buildcleaner: keep
        "@pypi//transformers",
    ],
)

ipynb_deps(
    name = "hello_world_llm_deps",
    srcs = ["hello_world_llm.ipynb"],
    pytype = False,
    deps = [
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

torch_tpu_package_end()
