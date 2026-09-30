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
load(
    "//shims/py_rules:pytype.bzl",
    "pytype_strict_binary",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_library(
    name = "device_utils",
    srcs = ["device_utils.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/sync",
        "@pypi//jax",
        "@pypi//psutil",
    ],
)

torch_tpu_py_test(
    name = "device_utils_test",
    srcs = ["device_utils_test.py"],
    args = ["--torch_tpu_internal_enable_compilation_container=true"],
    notest_oss = True,  # Flags don't work in OSS.  # NOTEST_OSS_OK=benchmarks run nightly
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":device_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "single_trace_trainer_adapter",
    srcs = ["single_trace_trainer_adapter.py"],
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/extras/optimizers:adamw",
        "//src/torch_tpu/extras/optimizers:muon",
        "//src/torch_tpu/extras/optimizers:optim",
        "//src/torch_tpu/extras/optimizers:sgd",
        "//src/torch_tpu/extras/optimizers:single_trace_trainer",
    ],
)

torch_tpu_py_library(
    name = "base_test",
    testonly = True,
    srcs = ["base_test.py"],
    deps = [
        ":common",
        ":export",
        ":flags",
        ":metrics",
        ":mode",
        ":registry",
        ":target",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
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
        ":common",
        ":metrics",
        ":mode",
        ":target",
        "//src/torch_tpu/_internal/benchmarks:export_adapter",
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
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "mode",
    srcs = ["mode.py"],
    deps = [
        ":common",
        ":target",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
    ],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Decouple from torch_tpu to avoid C++ dependency in coverage tests.
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

pytype_strict_library(
    # PY_LIBRARY_OK=Decouple from torch_tpu to avoid C++ dependency in coverage tests.
    name = "target",
    srcs = ["target.py"],
)

torch_tpu_py_library(
    name = "torch_device_ops",
    srcs = ["torch_device_ops.py"],
    deps = [
        ":device_utils",
        ":target",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
    ],
)

torch_tpu_py_library(
    name = "compile",
    srcs = ["compile.py"],
    deps = [
        ":device_utils",
        ":target",
        "//perflab/steps:common",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/compile",
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
        ":common",
        ":mode",
        ":registry",
        ":target",
    ],
)

torch_tpu_py_library(
    name = "model_helpers",
    srcs = ["model_helpers.py"],
    deps = [
        "//shims/torch:pytorch",
        "//tests:module_registry",
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
    data = [
        "//perflab/huggingface_transformers:model_configs_data",
        "//shims/model_configs:google_model_configs_data",
    ],
    env = {
        "BENCHMARK_PLATFORM": "v5e_1x1",
    },
    nolocal = "user requested to run this manually",
    nopresubmit = "Benchmark.",
    notest_oss = True,  # requires huggingface configs from internal infra  # NOTEST_OSS_OK=benchmarks run nightly
    shard_count = 50,
    tags = [
        "nofastbuild",  # Benchmark.
        "nosan",  # *san tests provide no benefit for benchmarks.
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    use_tcmalloc = True,
    deps = [
        ":base_test",
        ":cases",
        ":common",
        ":compile",
        ":context",
        ":discovery",
        ":flags",
        ":measure",
        ":metrics",
        ":mode",
        ":registry",
        ":step_lib",
        ":target",
        ":torch_device_ops",
        "//perflab/models",
        "//perflab/steps",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
    ],
)

# Benchmark target intended for manually running for dev or experimental purposes
# instead of CI. Sets all the flags for expected use.
torch_tpu_py_test(
    name = "dev_benchmarks",
    timeout = "eternal",
    srcs = ["torch_benchmark_test.py"],
    args = [
        "--run_scope=full",
        "--min_warmup_steps=3",
        "--max_warmup_steps=10",
        "--post_warmup_steps=10",
        "--xprof_steps=2",
        "--enable_xprof=true",
    ],
    data = [
        "//perflab/huggingface_transformers:model_configs_data",
    ],
    env = {
        "BENCHMARK_PLATFORM": "v7_1x1x1",
    },
    main = "torch_benchmark_test.py",
    nolocal = "user requested to run this manually",
    # Cannot run in OSS either: needs HuggingFace configs from internal infra.
    notap = "Development benchmark target; run manually on demand.",
    shard_count = 50,
    tags = [
        "nofastbuild",
        "nosan",  # *san tests provide no benefit for benchmarks.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        ":base_test",
        ":cases",
        ":common",
        ":compile",
        ":context",
        ":discovery",
        ":flags",
        ":measure",
        ":metrics",
        ":mode",
        ":registry",
        ":step_lib",
        ":target",
        ":torch_device_ops",
        "//perflab/models",
        "//perflab/steps",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
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
    name = "distributed_runner_lib",
    testonly = 1,
    srcs = ["distributed_runner.py"],
    deps = [
        ":common",
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
        "//perflab/distributed_models",
        "//perflab/steps",
        "//shims/absl_py:app",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests/distributed:distributed_utils",
        "@pypi//fairscale",
    ],
)

torch_tpu_py_test(
    name = "torch_benchmark_full",
    timeout = "eternal",
    srcs = ["torch_benchmark_test.py"],
    data = [
        "//perflab/huggingface_transformers:model_configs_data",
        "//shims/model_configs:google_model_configs_data",
    ],
    env = {
        "BENCHMARK_PLATFORM": "v7_1x1x1",
    },
    main = "torch_benchmark_test.py",
    notap = "Intended to be run with nightly benchmark suite",
    shard_count = 50,
    tags = [
        "nofastbuild",  # Benchmark.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    use_tcmalloc = True,
    deps = [
        ":base_test",
        ":cases",
        ":common",
        ":compile",
        ":context",
        ":discovery",
        ":flags",
        ":measure",
        ":metrics",
        ":mode",
        ":registry",
        ":step_lib",
        ":target",
        ":torch_device_ops",
        "//perflab/models",
        "//perflab/steps",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
    ],
)

# buildozer: disable=forbidden-tag-requires-gpu
torch_tpu_py_test(
    name = "torch_benchmark_full_cuda",
    timeout = "long",
    srcs = ["torch_benchmark_test.py"],
    data = [
        "//perflab/huggingface_transformers:model_configs_data",
    ],
    env = {
        "BENCHMARK_PLATFORM": "b200_1",
    },
    exec_properties = {
        "cpp_link.mem": "16g",
    },
    main = "torch_benchmark_test.py",
    notap = "Intended to be run with nightly benchmark suite",
    shard_count = 50,
    tags = [
        "nofastbuild",  # Benchmark.
        "requires-accel-hbm",
        "requires-gpu-sm100-full",
        "requires-net:external",
    ],
    deps = [
        ":base_test",
        ":cases",
        ":common",
        ":compile",
        ":context",
        ":discovery",
        ":flags",
        ":measure",
        ":metrics",
        ":mode",
        ":registry",
        ":step_lib",
        ":target",
        ":torch_device_ops",
        "//perflab/models",
        "//perflab/steps",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
    ],
)

pytype_strict_binary(
    name = "distributed_runner",
    testonly = 1,
    srcs = ["distributed_runner.py"],
    data = [
        "//perflab/huggingface_transformers:model_configs_data",
        "//shims/model_configs:google_model_configs_data",
    ],
    deps = [
        ":distributed_runner_lib",
    ],
)

alias(
    name = "distributed_runner_bin",
    actual = ":distributed_runner",
)

pkg_binary_with_runfiles(
    name = "distributed_runner_with_runfiles",
    testonly = 1,
    binary = ":distributed_runner",
)

genmpm(
    name = "distributed_runner_mpm",
    testonly = 1,
    temporal = True,
    deps = [":distributed_runner_with_runfiles"],
)

torch_tpu_py_library(
    name = "init",
    srcs = ["__init__.py"],
)

torch_tpu_package_end()
