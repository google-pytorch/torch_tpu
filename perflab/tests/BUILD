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
# gjh8
# cbasile-g
# lukeboyer
# ghpvnist
# lizlian

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

torch_tpu_py_test(
    name = "base_test_test",
    srcs = ["base_test_test.py"],
    tags = [
        # perflab/base_test.py matches *_test.py, so the wheel test site
        # packages leave it out.
        "wheel_test_excluded",
    ],
    deps = [
        "//perflab:base_test",
        "//perflab:common",
        "//perflab:device_ops",
        "//perflab:discovery",
        "//perflab:export",
        "//perflab:flags",
        "//perflab:known_failures",
        "//perflab:measure",
        "//perflab:metrics",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "benchmark_registry_test",
    srcs = ["benchmark_registry_test.py"],
    data = ["//perflab:benchmark_registry.pbtxt"],
    deps = [
        "//perflab:export",
        "//perflab:measure",
        "//perflab:target",
        "//perflab:workload",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//src/torch_tpu",
        "//tests:repo_paths",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "batch_size_override_test",
    srcs = ["batch_size_override_test.py"],
    data = [
        "//perflab/huggingface_transformers:model_configs_data",
        "//shims/model_configs:google_model_configs_data",
    ],
    notest_oss = True,  # NOTEST_OSS_OK=builds models from the internal in-tree model configs, which OSS does not export
    deps = [
        "//perflab:common",
        "//perflab:context",
        "//perflab:discovery",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:target",
        "//perflab:workload",
        "//perflab/models",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "benchmark_contract_test",
    srcs = ["benchmark_contract_test.py"],
    data = [
        "//perflab/huggingface_transformers:model_configs_data",
        "//shims/model_configs:google_model_configs_data",
    ],
    notest_oss = True,  # NOTEST_OSS_OK=checks the internal in-tree model configs, which OSS does not export
    deps = [
        "//perflab:common",
        "//perflab:mode",
        "//perflab:target",
        "//perflab/gen_utils",
        "//perflab/models",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        "//perflab:common",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "conformer_test",
    srcs = ["conformer_test.py"],
    tags = [
        "nomsan",  # 3p 'py/torch' error: use-of-uninitialized-value in reduced_float_type_copy
    ],
    deps = [
        "//perflab:context",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:target",
        "//perflab/models",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
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
        "//perflab:known_failures",
        "//perflab:metrics",
        "//perflab:mode",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:export_adapter",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "fakes",
    testonly = True,
    srcs = ["fakes.py"],
    deps = [
        "@tsl//tsl/profiler/protobuf:xplane_proto_py_proto",
    ],
)

torch_tpu_py_test(
    name = "flags_test",
    srcs = ["flags_test.py"],
    deps = [
        "//perflab:context",
        "//perflab:flags",
        "//perflab:target",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "fake_memory_test",
    srcs = ["fake_memory_test.py"],
    tags = [
        "nomsan",  # 3p 'py/torch' error: use-of-uninitialized-value in reduced_float_type_copy
    ],
    deps = [
        "//perflab:fake_memory",
        "//perflab:mode",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "flops_test",
    srcs = ["flops_test.py"],
    tags = [
        "nomsan",  # 3p 'py/torch' error: use-of-uninitialized-value in reduced_float_type_copy
    ],
    deps = [
        "//perflab:context",
        "//perflab:discovery",
        "//perflab:flops",
        "//perflab:metrics",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//perflab:workload",
        "//perflab/steps",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "measure_test",
    srcs = ["measure_test.py"],
    deps = [
        ":fakes",
        "//perflab:device_utils",
        "//perflab:measure",
        "//perflab:metrics",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "model_helpers_test",
    srcs = ["model_helpers_test.py"],
    deps = [
        "//perflab:context",
        "//perflab:model_helpers",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "mode_test",
    srcs = ["mode_test.py"],
    tags = [
        "nomsan",  # 3p 'py/torch' error: use-of-uninitialized-value in reduced_float_type_copy
    ],
    deps = [
        "//perflab:common",
        "//perflab:mode",
        "//perflab:target",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "models_presubmit_scope_test",
    srcs = ["models_presubmit_scope_test.py"],
    deps = [
        "//perflab:context",
        "//perflab:registry",
        "//perflab:target",
        "//perflab/models",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "registry_test",
    srcs = ["registry_test.py"],
    deps = [
        "//perflab:compile",
        "//perflab:context",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:workload",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "gen_benchmarks_presubmit_scope_test",
    srcs = ["gen_benchmarks_presubmit_scope_test.py"],
    data = [
        "//perflab/huggingface_transformers:model_configs_data",
        "//shims/model_configs:google_model_configs_data",
    ],
    notest_oss = True,  # NOTEST_OSS_OK=checks the internal in-tree model configs, which OSS does not export
    deps = [
        "//perflab:cases",
        "//perflab:context",
        "//perflab:flags",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:target",
        "//perflab/models",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
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
        "//perflab:device_utils",
        "//perflab:mode",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "torch_base_test_test",
    srcs = ["torch_base_test_test.py"],
    tags = [
        # perflab/torch_base_test.py depends on perflab/base_test.py, which
        # matches *_test.py, so the wheel test site packages leave it out.
        "wheel_test_excluded",
    ],
    deps = [
        "//perflab:common",
        "//perflab:context",
        "//perflab:discovery",
        "//perflab:export",
        "//perflab:flags",
        "//perflab:measure",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_base_test",
        "//perflab/steps",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
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
        "//perflab:workload",
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
        "//perflab:context",
        "//perflab:known_failures",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:workload",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "workload_test",
    srcs = ["workload_test.py"],
    deps = [
        "//perflab:metrics",
        "//perflab:step_lib",
        "//perflab:workload",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
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
    name = "layer_utils_test",
    srcs = ["layer_utils_test.py"],
    deps = [
        "//perflab/layer_utils:builders",
        "//perflab/layer_utils:configs",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "layer_registration_test",
    srcs = ["layer_registration_test.py"],
    deps = [
        "//perflab:common",
        "//perflab:compile",
        "//perflab:context",
        "//perflab:flags",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab/layer_utils:builders",
        "//perflab/layer_utils:configs",
        "//perflab/layer_utils:registration",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "layers_contract_test",
    srcs = ["layers_contract_test.py"],
    deps = [
        "//perflab:context",
        "//perflab:discovery",
        "//perflab:mode",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab/layer_utils:builders",
        "//perflab/layer_utils:configs",
        "//perflab/layer_utils:registration",
        "//perflab/layers",
        "//perflab/models",
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
    timeout = "long",
    srcs = ["distributed_qwen3_8_test.py"],
    tags = ["requires-mem:28g"],
    # Fix to alleviate too many input files error (transformers + torchtitan).
    use_precompile = False,
    deps = [
        "//perflab:context",
        "//perflab:target",
        "//perflab/distributed_models:qwen3_8",
        "//perflab/model_utils:torchtitan_batching",
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

internal_target(
    torch_tpu_py_test,
    name = "distributed_muse_glimmer_test",
    timeout = "long",
    srcs = ["distributed_muse_glimmer_test.py"],
    tags = ["requires-mem:28g"],
    use_precompile = False,
    deps = [
        "//perflab:context",
        "//perflab:target",
        "//perflab/distributed_models:muse_glimmer",
        "//perflab/model_utils:torchtitan_batching",
        "//perflab/model_utils:torchtitan_muse_glimmer",
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
    timeout = "long",
    srcs = ["distributed_synthetic_test.py"],
    shard_count = 3,
    tags = ["requires-mem:28g"],
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
