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
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "base_test",
    testonly = True,
    srcs = ["base_test.py"],
    deps = [
        "//shims/absl_py/logging",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "flags",
    srcs = ["flags.py"],
    deps = [
        "//shims/absl_py/flags",
    ],
)

torch_tpu_py_library(
    name = "export",
    srcs = ["export.py"],
    deps = [
        ":metrics",
        ":mode",
        ":target",
        "//examples/benchmarks/e2e:common",
        "//torch_tpu/_internal/benchmarks:export_adapter",
    ],
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
        "//torch_tpu/_internal/profiler:xprof_adapter",
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
        ":base_test",
        ":cases",
        ":compile",
        ":context",
        ":discovery",
        ":export",
        ":flags",
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
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//torch_tpu",
    ],
)

pkg_binary_with_runfiles(
    name = "torch_benchmark_test_with_runfiles",
    testonly = 1,
    binary = ":torch_benchmark_test",
)

# Using a temporal MPM instead of a durable MPM as these are only used for testing. It also allows
# testing changes from a local workspace.
genmpm(
    name = "torch_benchmark_test_mpm",
    testonly = 1,
    temporal = True,
    deps = [":torch_benchmark_test_with_runfiles"],
)

torch_tpu_py_library(
    name = "multihost_runner_lib",
    testonly = 1,
    srcs = ["multihost_runner.py"],
    deps = [
        ":compile",
        ":context",
        ":discovery",
        ":flags",
        ":measure",
        ":mode",
        ":registry",
        ":step_lib",
        ":target",
        ":torch_device_ops",
        "//examples/benchmarks/e2e:common",
        "//examples/benchmarks/e2e/harness/models",
        "//examples/benchmarks/e2e/harness/steps",
        "//shims/absl_py:app",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed:multiprocessing",
        "@pypi//fairscale",
    ],
)

pytype_strict_binary(
    name = "multihost_runner_bin",
    testonly = 1,
    srcs = ["multihost_runner.py"],
    main = "multihost_runner.py",
    deps = [
        ":multihost_runner_lib",
    ],
)

alias(
    name = "multihost_runner",
    actual = ":multihost_runner_bin",
)

torch_tpu_package_end()
