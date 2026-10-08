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
    "internal_target",
    "torch_tpu_package_end",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "base_test_test",
    srcs = ["base_test_test.py"],
    tags = [
        # `perflab/base_test.py` matches the `*_test.py` exclude of
        # `wheel_test_site_packages`, so OSS wheel tests cannot import it.
        "wheel_test_excluded",
    ],
    deps = [
        "//perflab:base_test",
        "//perflab:common",
        "//perflab:flags",
        "//perflab:known_failures",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "export_test",
    srcs = ["export_test.py"],
    deps = [
        "//perflab:common",
        "//perflab:export",
        "//perflab:metrics",
        "//perflab:mode",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:export_adapter",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "measure_test",
    srcs = ["measure_test.py"],
    nonightly_oss = "TODO:(b/553018817) Re-enable once OSS backend loading is resolved",
    deps = [
        "//perflab:measure",
        "//perflab:metrics",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "mode_test",
    srcs = ["mode_test.py"],
    deps = [
        "//perflab:common",
        "//perflab:mode",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "registry_test",
    srcs = ["registry_test.py"],
    deps = [
        "//perflab:compile",
        "//perflab:registry",
        "//perflab:step_lib",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "step_test",
    srcs = ["step_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//perflab:compile",
        "//perflab:discovery",
        "//perflab:measure",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//perflab/steps",
        "//perflab/steps:decode",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/optimizers:sgd",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "decode_stepper_coverage_test",
    srcs = ["decode_stepper_coverage_test.py"],
    # Resolves model configs from module_registry's weights mirror, which is not
    # hermetic for TAP.
    notap = "reads model configs from external storage; run manually",
    shard_count = 4,
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//perflab/steps:decode",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:module_registry",
        "//tests:seed_test_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "target_test",
    srcs = ["target_test.py"],
    deps = [
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "torch_device_ops_test",
    srcs = ["torch_device_ops_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "compile_test",
    srcs = ["compile_test.py"],
    deps = [
        "//perflab:compile",
        "//perflab:device_utils",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "context_test",
    srcs = ["context_test.py"],
    deps = [
        "//perflab:context",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "discovery_test",
    srcs = ["discovery_test.py"],
    deps = [
        "//perflab:discovery",
        "//perflab:registry",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "cases_test",
    srcs = ["cases_test.py"],
    deps = [
        "//perflab:cases",
        "//perflab:common",
        "//perflab:known_failures",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "known_failures_test",
    srcs = ["known_failures_test.py"],
    deps = [
        "//perflab:common",
        "//perflab:known_failures",
        "//perflab:mode",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "loss_test",
    srcs = ["loss_test.py"],
    deps = [
        "//perflab:loss",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "distributed_runner_test",
    srcs = ["distributed_runner_test.py"],
    deps = [
        "//perflab:common",
        "//perflab:discovery",
        "//perflab:distributed_runner_lib",
        "//perflab:export",
        "//perflab:known_failures",
        "//perflab:measure",
        "//perflab:metrics",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//perflab/steps",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "distributed_llama_test",
    srcs = ["distributed_llama_test.py"],
    deps = [
        "//perflab:context",
        "//perflab:registry",
        "//perflab:target",
        "//perflab/distributed_models:llama",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

# torchtitan is not available in OSS.
internal_target(
    torch_tpu_py_test,
    name = "distributed_qwen3_8_test",
    srcs = ["distributed_qwen3_8_test.py"],
    # Fix to alleviate too many input files error (transformers + torchtitan).
    use_precompile = False,
    deps = [
        "//perflab:context",
        "//perflab:target",
        "//perflab/distributed_models:qwen3_8",
        "//perflab/model_utils:torchtitan_qwen3_5",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "//tests/distributed:distributed_utils",
    ],
)

# torchtitan (SimpleFSDP) is not available in OSS.
internal_target(
    torch_tpu_py_test,
    name = "distributed_synthetic_test",
    srcs = ["distributed_synthetic_test.py"],
    # Avoids TOO_MANY_INPUT_FILES from torchtitan's sources.
    use_precompile = False,
    deps = [
        "//perflab:context",
        "//perflab:registry",
        "//perflab:target",
        "//perflab/distributed_models:synthetic",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "//tests/distributed:distributed_utils",
    ],
)

torch_tpu_package_end()
