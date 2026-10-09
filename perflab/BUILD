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
    "if_oss",
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
    args = if_oss(
        [],
        ["--torch_tpu_internal_enable_compilation_container=true"],
    ),
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":device_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//tests:oss_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "single_trace_trainer_adapter",
    srcs = ["single_trace_trainer_adapter.py"],
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/optimizers:adamw",
        "//src/torch_tpu/_internal/optimizers:muon",
        "//src/torch_tpu/_internal/optimizers:sgd",
    ],
)

torch_tpu_py_library(
    name = "base_test",
    testonly = True,
    srcs = ["base_test.py"],
    deps = [
        ":cases",
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
        ":known_failures",
        ":mode",
        ":registry",
        ":target",
    ],
)

torch_tpu_py_library(
    name = "known_failures",
    srcs = ["known_failures.py"],
    deps = [
        ":common",
        ":mode",
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

# v5e benchmark smoke test: runs every presubmit-scope case once. The
# benchmark-smoke tag wires it into presubmit, where it blocks CL submission.
torch_tpu_py_test(
    name = "torch_benchmark_test",
    timeout = "long",
    srcs = ["torch_benchmark_test.py"],
    args = [
        "--run_scope=presubmit",
        # Deliberate, although it is the default: each case must keep running
        # in its own process to guard multi-process setup (b/552911303).
        "--run_in_subprocess",
        "--noenable_xprof",
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
    shard_count = 10,
    tags = [
        "benchmark-smoke",
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
    # The torchtitan dep (via models:qwen3_8) exceeds the Forge input file limit
    # with precompiled .pyc inputs.
    use_precompile = False,
    deps = if_oss(
        [],
        [
            "//perflab/models:muse_glimmer",
            "//perflab/models:qwen3_8",
        ],
    ) + [
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
        ":cases",
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
    # The torchtitan dep (via models:qwen3_8) exceeds the Forge input file limit
    # with precompiled .pyc inputs.
    use_precompile = False,
    use_tcmalloc = True,
    deps = if_oss(
        [],
        [
            "//perflab/models:muse_glimmer",
            "//perflab/models:qwen3_8",
        ],
    ) + [
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
    # The torchtitan dep (via models:qwen3_8) exceeds the Forge input file limit
    # with precompiled .pyc inputs.
    use_precompile = False,
    deps = if_oss(
        [],
        [
            "//perflab/models:muse_glimmer",
            "//perflab/models:qwen3_8",
        ],
    ) + [
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

_DISTRIBUTED_RUNNER_DATA = [
    "//shims/model_configs:google_model_configs_data",
    "//perflab/huggingface_transformers:model_configs_data",
]

# The Muse Glimmer (glimmer prototype), Qwen3.8 and synthetic (torchtitan)
# benchmarks are not available in OSS, so they are added to the runner binaries
# rather than to `distributed_runner_lib`.
_DISTRIBUTED_RUNNER_MODEL_DEPS = if_oss(
    [],
    [
        "//perflab/distributed_models:muse_glimmer",
        "//perflab/distributed_models:qwen3_8",
        "//perflab/distributed_models:synthetic",
    ],
)

pytype_strict_binary(
    name = "distributed_runner",
    testonly = 1,
    srcs = ["distributed_runner.py"],
    data = _DISTRIBUTED_RUNNER_DATA,
    deps = _DISTRIBUTED_RUNNER_MODEL_DEPS + [
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

# Distributed counterparts of `dev_benchmarks`: manual dev targets that run
# `distributed_runner` under blaze test on a single TPU host. Select
# benchmarks with --test_filter (full target names, e.g.
# `--test_filter=qwen3_8_27b_4_layers_simple_fsdp_training_compiled`).
# `distributed_runner` splits the runnable targets round-robin across
# `shard_count` shards, each on its own host.
# Every shard still holds a full TPU host even when the filter leaves it nothing
# to run, so add `--test_sharding_strategy=disabled` when filtering to fewer
# targets than shards.
_DISTRIBUTED_DEV_BENCHMARKS_ARGS = [
    "--min_warmup_steps=3",
    "--max_warmup_steps=10",
    "--enable_xprof=true",
    # Surfaces the per-rank absl logs (incl. the "Metrics for ..." line) in the
    # test log.
    "--alsologtostderr",
]

_DISTRIBUTED_DEV_BENCHMARKS_DEPS = _DISTRIBUTED_RUNNER_MODEL_DEPS + [
    ":distributed_runner_lib",
    "//src/torch_tpu",
    "//src/torch_tpu/_internal/distributed:multiprocessing",
]

# world_size=2: one TPU v7 chip, one process per TensorCore.
torch_tpu_py_test(
    name = "distributed_dev_benchmarks_ws2",
    timeout = "eternal",
    srcs = ["distributed_runner.py"],
    args = _DISTRIBUTED_DEV_BENCHMARKS_ARGS,
    data = _DISTRIBUTED_RUNNER_DATA,
    env = {
        "BENCHMARK_PLATFORM": "v7_1x1x1_2tc",
    },
    main = "distributed_runner.py",
    nolocal = "user requested to run this manually",
    notap = "Development benchmark target; run manually on demand.",
    shard_count = 8,
    tags = [
        "nofastbuild",
        "nosan",  # *san tests provide no benefit for benchmarks.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    # The torchtitan dep (via distributed_models:qwen3_8) exceeds the Forge input
    # file limit with precompiled .pyc inputs.
    use_precompile = False,
    deps = _DISTRIBUTED_DEV_BENCHMARKS_DEPS,
)

# world_size=4: two TPU v7 chips x 2 TensorCores. Forge only allocates a
# single chip or a full 4-chip host, so this takes the full host and runs on a
# 1x2x1 sub-slice of it.
torch_tpu_py_test(
    name = "distributed_dev_benchmarks_ws4",
    timeout = "eternal",
    srcs = ["distributed_runner.py"],
    args = _DISTRIBUTED_DEV_BENCHMARKS_ARGS,
    data = _DISTRIBUTED_RUNNER_DATA,
    env = {
        "BENCHMARK_PLATFORM": "v7_1x2x1",
    },
    main = "distributed_runner.py",
    nolocal = "user requested to run this manually",
    notap = "Development benchmark target; run manually on demand.",
    shard_count = 8,
    tags = [
        "nofastbuild",
        "nosan",  # *san tests provide no benefit for benchmarks.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x:4",
    ],
    # The torchtitan dep (via distributed_models:qwen3_8) exceeds the Forge input
    # file limit with precompiled .pyc inputs.
    use_precompile = False,
    deps = _DISTRIBUTED_DEV_BENCHMARKS_DEPS,
)

# world_size=8: one TPU v7 host, 4 chips x 2 TensorCores.
torch_tpu_py_test(
    name = "distributed_dev_benchmarks_ws8",
    timeout = "eternal",
    srcs = ["distributed_runner.py"],
    args = _DISTRIBUTED_DEV_BENCHMARKS_ARGS,
    data = _DISTRIBUTED_RUNNER_DATA,
    env = {
        "BENCHMARK_PLATFORM": "v7_2x2x1",
    },
    main = "distributed_runner.py",
    nolocal = "user requested to run this manually",
    notap = "Development benchmark target; run manually on demand.",
    shard_count = 8,
    tags = [
        "nofastbuild",
        "nosan",  # *san tests provide no benefit for benchmarks.
        # 8 processes compiling concurrently exceed the 28g `requires-mem` cap
        # (Forge OOM). `requires-accel-hbm` raises the host RAM limit to 28 GiB
        # plus the allocated chips' HBM (go/forge-requires-mem#accel-hbm).
        # Forge rejects `requires-mem` combined with `requires-accel-hbm`.
        "requires-accel-hbm",
        # A bare "requires-tpu-v7x" allocates one chip; ":4" allocates the
        # full 4-chip host (8 TensorCores).
        "requires-tpu-v7x:4",
        "requires-net:external",
    ],
    # The torchtitan dep (via distributed_models:qwen3_8) exceeds the Forge input
    # file limit with precompiled .pyc inputs.
    use_precompile = False,
    deps = _DISTRIBUTED_DEV_BENCHMARKS_DEPS,
)

# GPU counterpart of `distributed_dev_benchmarks_ws2`: 2 B200s, one process per
# GPU. Run with --config=cuda. Forge allocates at most 2 whole B200s, so the
# 4- and 8-GPU counterparts (b200_4 / b200_8) run through the multi-host
# benchmark launcher instead.
torch_tpu_py_test(
    name = "distributed_dev_benchmarks_cuda_ws2",
    timeout = "eternal",
    srcs = ["distributed_runner.py"],
    # No XProf: it interferes with GPU collectives (b/517613809).
    args = [
        "--min_warmup_steps=3",
        "--max_warmup_steps=10",
        "--alsologtostderr",
    ],
    # Keeps the TPU backend from autoloading on the CUDA workers.
    autoload = False,
    data = _DISTRIBUTED_RUNNER_DATA,
    env = {
        "BENCHMARK_PLATFORM": "b200_2",
    },
    exec_properties = {
        "cpp_link.mem": "16g",
    },
    main = "distributed_runner.py",
    nolocal = "user requested to run this manually",
    notap = "Development benchmark target; run manually on demand.",
    platform = "//buildenv/platforms/forge/gpu/b200_full_only:2",
    shard_count = 4,
    tags = [
        "nofastbuild",
        "nosan",  # *san tests provide no benefit for benchmarks.
        "requires-net:external",
    ],
    # The torchtitan dep (via distributed_models:qwen3_8) exceeds the Forge input
    # file limit with precompiled .pyc inputs.
    use_precompile = False,
    deps = _DISTRIBUTED_DEV_BENCHMARKS_DEPS,
)

torch_tpu_py_library(
    name = "init",
    srcs = ["__init__.py"],
)

torch_tpu_package_end()
