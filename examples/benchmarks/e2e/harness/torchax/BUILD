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
load("//shims/mpm:genmpm.bzl", "genmpm")
load("//shims/pkg:pkg.bzl", "pkg_binary_with_runfiles")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "torchax_device_ops",
    srcs = ["torchax_device_ops.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:target",
        "@pypi//jax",
    ],
)

torch_tpu_py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    deps = [
        "//shims/torch:pytorch",
        "@pypi//optax",
    ],
)

torch_tpu_py_library(
    name = "torchax_step",
    srcs = ["torchax_step.py"],
    deps = [
        ":optimizer",
        "//examples/benchmarks/e2e/harness:compile",
        "//examples/benchmarks/e2e/harness:context",
        "//examples/benchmarks/e2e/harness:registry",
        "//examples/benchmarks/e2e/harness:step_lib",
        "//shims/torch:pytorch",
        "@pypi//jax",
        "@pypi//optax",
        "@pypi//torchax",
    ],
)

torch_tpu_py_test(
    name = "torchax_benchmark_test",
    timeout = "long",
    srcs = ["torchax_benchmark_test.py"],
    args = [
        "--run_scope=presubmit",
        "--min_warmup_steps=1",
        "--max_warmup_steps=1",
        "--post_warmup_steps=0",
    ],
    autoload = False,
    env = {
        "BENCHMARK_PLATFORM": "v5e_1x1",
    },
    nolocal = "user requested to run this manually",
    notest_oss = "requires huggingface configs from internal infra",
    shard_count = 50,
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        ":torchax_device_ops",
        ":torchax_step",
        "//examples/benchmarks/e2e:common",
        "//examples/benchmarks/e2e/harness:base_test",
        "//examples/benchmarks/e2e/harness:cases",
        "//examples/benchmarks/e2e/harness:compile",
        "//examples/benchmarks/e2e/harness:context",
        "//examples/benchmarks/e2e/harness:discovery",
        "//examples/benchmarks/e2e/harness:export",
        "//examples/benchmarks/e2e/harness:flags",
        "//examples/benchmarks/e2e/harness:measure",
        "//examples/benchmarks/e2e/harness:metrics",
        "//examples/benchmarks/e2e/harness:mode",
        "//examples/benchmarks/e2e/harness:registry",
        "//examples/benchmarks/e2e/harness:step_lib",
        "//examples/benchmarks/e2e/harness:target",
        "//examples/benchmarks/e2e/harness/models",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//torch_tpu",
        "@pypi//jax",
        "@pypi//torchax",
    ],
)

pkg_binary_with_runfiles(
    name = "torchax_benchmark_test_with_runfiles",
    testonly = 1,
    binary = ":torchax_benchmark_test",
)

# Using a temporal MPM instead of a durable MPM as these are only used for testing. It also allows
# testing changes from a local workspace.
genmpm(
    name = "torchax_benchmark_test_mpm",
    testonly = 1,
    temporal = True,
    deps = [":torchax_benchmark_test_with_runfiles"],
)

torch_tpu_package_end()
