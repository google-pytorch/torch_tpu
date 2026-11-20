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

# Domain experts for this package:
# zhanyong-wan
# lukeboyer

load(
    "//build_files:build_defs.bzl",
    "internal_target",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/compile:compiler",
    ],
)

torch_tpu_py_test(
    name = "adamw_test",
    srcs = ["adamw_test.py"],
    main = "adamw_test.py",
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/optimizers:adamw",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "sgd_test",
    srcs = ["sgd_test.py"],
    main = "sgd_test.py",
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/optimizers:sgd",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "muon_test",
    srcs = ["muon_test.py"],
    main = "muon_test.py",
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/optimizers:muon",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "single_trace_trainer_test",
    srcs = ["single_trace_trainer_test.py"],
    main = "single_trace_trainer_test.py",
    nopresubmit = "slow example",
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/optimizers:adamw",
        "//src/torch_tpu/_internal/optimizers:sgd",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

_DISTRIBUTED_SINGLE_TRACE_TRAINER_TEST_DEPS = [
    "//shims/absl_py/flags",
    "//shims/absl_py/testing:absltest",
    "//shims/absl_py/testing:parameterized",
    "//shims/torch:pytorch",
    "//shims/torch:testing_internal",
    "//src/torch_tpu",
    "//src/torch_tpu/_internal/compile:compiler",
    "//src/torch_tpu/_internal/optimizers:adamw",
    "//src/torch_tpu/_internal/optimizers:muon",
    "//src/torch_tpu/_internal/optimizers:sgd",
    "//src/torch_tpu/_internal/optimizers:single_trace_trainer",
    "//src/torch_tpu/_internal/utils:test_utils",
    "//src/torch_tpu/_internal/distributed:multiprocessing",
    "//tests:seed_test_utils",
    "//tests/distributed:continuous_test_utils",
    "//shims/torchtitan/experiments/graph_trainer:simple_fsdp",
]

# SingleTraceTrainer on SimpleFSDP (DTensor) models: 2 gloo ranks on CPU with
# the traced step uncompiled. torchtitan (SimpleFSDP) is not available in OSS.
internal_target(
    torch_tpu_py_test,
    name = "distributed_single_trace_trainer_test",
    timeout = "long",
    srcs = ["distributed_single_trace_trainer_test.py"],
    main = "distributed_single_trace_trainer_test.py",
    tags = ["requires-mem:28g"],
    # Avoids TOO_MANY_INPUT_FILES from torchtitan's sources.
    use_precompile = False,
    deps = _DISTRIBUTED_SINGLE_TRACE_TRAINER_TEST_DEPS,
)

# The same tests on one TPU v7 chip (one rank per TensorCore), compiled with
# StaticCompiler.
internal_target(
    torch_tpu_py_test,
    name = "distributed_single_trace_trainer_tpu_test",
    srcs = ["distributed_single_trace_trainer_test.py"],
    args = ["--device=tpu"],
    main = "distributed_single_trace_trainer_test.py",
    tags = [
        "nosan",  # Spawns TPU processes.
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    use_precompile = False,
    deps = _DISTRIBUTED_SINGLE_TRACE_TRAINER_TEST_DEPS,
)

torch_tpu_package_end()
