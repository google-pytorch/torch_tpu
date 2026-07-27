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
    name = "metrics",
    srcs = ["metrics.py"],
)

torch_tpu_py_library(
    name = "device_ops",
    srcs = ["device_ops.py"],
)

torch_tpu_py_library(
    name = "discovery",
    srcs = ["discovery.py"],
)

torch_tpu_py_library(
    name = "measure",
    srcs = ["measure.py"],
    deps = [
        ":device_ops",
        ":metrics",
        ":step_lib",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "mode",
    srcs = ["mode.py"],
    deps = [
        ":target",
        "//examples/benchmarks/e2e:common",
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
    ],
)

torch_tpu_py_library(
    name = "step_lib",
    srcs = ["step_lib.py"],
)

torch_tpu_py_library(
    name = "loss",
    srcs = ["loss.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "registry",
    srcs = ["registry.py"],
    deps = [
        ":compile",
        ":step_lib",
        ":target",
    ],
)

torch_tpu_py_library(
    name = "target",
    srcs = ["target.py"],
)

torch_tpu_py_library(
    name = "torch_device_ops",
    srcs = ["torch_device_ops.py"],
    deps = [
        ":target",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_library(
    name = "compile",
    srcs = ["compile.py"],
    deps = [
        ":target",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile",
    ],
)

torch_tpu_py_library(
    name = "context",
    srcs = ["context.py"],
    deps = [
        ":target",
        "//shims/absl_py/flags",
    ],
)

torch_tpu_py_library(
    name = "cases",
    srcs = ["cases.py"],
    deps = [
        ":mode",
        ":registry",
        ":target",
        "//examples/benchmarks/e2e:common",
    ],
)

torch_tpu_py_test(
    name = "torch_benchmark_test",
    timeout = "long",
    srcs = ["torch_benchmark_test.py"],
    args = [
        "--run_scope=presubmit",
        "--min_warmup_steps=1",
        "--max_warmup_steps=1",
        "--post_warmup_steps=0",
    ],
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
        ":cases",
        ":compile",
        ":context",
        ":discovery",
        ":measure",
        ":metrics",
        ":mode",
        ":registry",
        ":step_lib",
        ":target",
        ":torch_device_ops",
        "//examples/benchmarks/e2e:common",
        "//examples/benchmarks/e2e/harness/models",
        "//examples/benchmarks/e2e/harness/steps",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//torch_tpu",
    ],
)

torch_tpu_package_end()
