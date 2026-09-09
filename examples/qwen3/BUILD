# Copyright 2025 Google LLC
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
    "internal_target",
    "torch_tpu_package_end",
    "torch_tpu_py_test",
)
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary")
load("//shims/pynball:build_defs.bzl", "ipynb_deps")

package(
    default_applicable_licenses = ["//:license"],
)

torch_tpu_py_test(
    name = "qwen3_example",
    srcs = ["qwen3_example.py"],
    nonightly_oss = "Disabled during switch from opt-in to opt-out",
    nopresubmit = "slow example",
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples/qwen3/impl:qwen3",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "qwen_kv_cache",
    srcs = ["qwen_kv_cache.py"],
    nonightly_oss = "Disabled during switch from opt-in to opt-out",
    nopresubmit = "slow example",
    tags = [
        "cpu:8",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_kv_cache_bd",
    srcs = ["qwen3_kv_cache_bd.py"],
    args = [
        "--device=tpu",
    ],
    data = [
        "//examples/huggingface_transformers:model_configs_data",
    ],
    notap = "this is a manual test",
    strict = True,
    tags = [
        "cpu:8",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/benchmarks:xprof_adapter",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/utils:test_utils",
        "@pypi//transformers",
        "@rules_python//python/runfiles",
    ],
)

torch_tpu_py_test(
    name = "qwen3_kv_cache_bd_b200",
    srcs = ["qwen3_kv_cache_bd.py"],
    args = [
        "--device=cuda",
    ],
    data = [
        "//examples/huggingface_transformers:model_configs_data",
    ],
    main = "qwen3_kv_cache_bd.py",
    notap = "manual test",
    platform = if_oss(None, "//buildenv/platforms/forge/gpu/gb200"),
    tags = [
        "cpu:1",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/gpu_cuda:cuda_runtime",
        "//shims/gpu_profiler:gpuprof_lib",  # buildcleaner: keep
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/benchmarks:xprof_adapter",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:test_utils",
        "@pypi//transformers",
        "@rules_python//python/runfiles",
    ],
)

internal_target(
    pytype_strict_binary,
    name = "xm_launch_bd",
    srcs = ["xm_launch_bd.py"],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "@pypi//xmanager/xm",
        "@pypi//xmanager/xm_abc",
    ],
)

internal_target(
    pytype_strict_binary,
    name = "xm_launch_compile_bd",
    srcs = ["xm_launch_compile_bd.py"],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "@pypi//xmanager/xm",
        "@pypi//xmanager/xm_abc",
    ],
)

torch_tpu_py_test(
    name = "qwen3_hf_example",
    # The test finishes fast most of the time. The timeout is for the 15% chance
    # of high latency when accessing remote storage.
    timeout = "long",  # 15 minutes
    srcs = ["qwen3_hf_example.py"],
    nolocal = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "nofastbuild",  # Benchmark.
        "requires-net:external",  # Needed to access remote files.
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/absl_py/logging",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_dynamic_kv_cache_presubmit",
    timeout = "long",
    srcs = ["qwen3_dynamic_kv_cache.py"],
    args = [
        "--batch_size=1",
        "--prefill_seq_len=128",
        "--hf_data_path=\"\"",
    ],
    main = "qwen3_dynamic_kv_cache.py",
    nolocal = "slow example",
    shuffle_tests = False,
    tags = [
        "cpu:4",
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/etils/epath",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/benchmarks:xprof_adapter",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_dynamic_kv_cache",
    timeout = "long",  # 15 minutes
    srcs = ["qwen3_dynamic_kv_cache.py"],
    nolocal = "slow example",
    notap = "manual test for dynamic performance evaluation",
    notest_oss = "manual test for dynamic performance evaluation",
    tags = [
        "cpu:4",  # Match b200 variant for fairness.
        "nofastbuild",  # Benchmark.
        "requires-accel-hbm",
        "requires-net:external",  # Needed to access remote files.
        "requires-tpu-v7x",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/etils/epath",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/benchmarks:xprof_adapter",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_dynamic_kv_cache_multi",
    timeout = "eternal",
    srcs = ["qwen3_multi_wrapper.py"],
    args = [
        "--benchmark_configs=1:1024,16:1024,16:2048",
        "--use_random_weights=true",
    ],
    data = [
        ":qwen3_dynamic_kv_cache",
    ],
    main = "qwen3_multi_wrapper.py",
    nolocal = "slow example",
    notap = "manual test for dynamic performance evaluation",
    notest_oss = "manual test for dynamic performance evaluation",
    tags = [
        "cpu:4",
        "nofastbuild",
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//torch_tpu",  # buildcleaner: keep
    ],
)

# buildifier: disable=forbidden-tag-requires-gpu
torch_tpu_py_test(
    name = "qwen3_dynamic_kv_cache_b200",
    timeout = "long",  # 15 minutes
    srcs = ["qwen3_dynamic_kv_cache.py"],
    args = [
        "--device=cuda",
    ],
    exec_properties = {"cpp_link.mem": "16g"},
    main = "qwen3_dynamic_kv_cache.py",
    notap = "manual test for dynamic performance evaluation",
    notest_oss = "manual test for dynamic performance evaluation",
    tags = [
        "cpu:4",
        "nofastbuild",  # Benchmark.
        "requires-accel-hbm",
        "requires-gpu-sm100-full",
        "requires-net:external",  # Needed to access remote files.
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/etils/epath",
        "//shims/gpu_cuda:cuda_runtime",
        "//shims/gpu_profiler:gpuprof_lib",  # buildcleaner: keep
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/benchmarks:xprof_adapter",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_hf_pipe_example",
    timeout = "long",  # 15 minutes
    srcs = ["qwen3_hf_pipe_example.py"],
    nonightly_oss = "Disabled during switch from opt-in to opt-out",
    nopresubmit = "slow example",
    tags = [
        "nomsan",  # 3p 'rust/tempfile' error: use-of-uninitialized-value in tempfile::file::imp::platform::reopen
        "requires-net:external",  # Needed to access remote files.
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compiler_options",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_0_6b_forward_test_vl",
    srcs = ["qwen3_0_6b_forward_test.py"],
    args = ["--device=tpu"],
    main = "qwen3_0_6b_forward_test.py",
    nonightly_oss = "Disabled during switch from opt-in to opt-out",
    nopresubmit = "slow example",
    tags = [
        "requires-mem:28g",  # For the msan build
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples/huggingface_transformers:model_configs",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_0_6b_forward_test_a100",
    srcs = ["qwen3_0_6b_forward_test.py"],
    args = ["--device=cuda"],
    main = "qwen3_0_6b_forward_test.py",
    notap = "slow example",
    tags = [
        "requires-gpu-sm80",
    ],
    deps = [
        "//examples/huggingface_transformers:model_configs",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_0_6b_forward_test_a100_xla_cuda",
    srcs = ["qwen3_0_6b_forward_test.py"],
    args = ["--device=xla_cuda"],
    main = "qwen3_0_6b_forward_test.py",
    notap = "slow example",
    tags = [
        "requires-gpu-sm80",
    ],
    deps = [
        "//examples/huggingface_transformers:model_configs",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/stream_executor:cuda_platform",
    ],
)

torch_tpu_py_test(
    name = "qwen3_0_6b_custom_sampling_test_tpu",
    timeout = "long",  # 15 minutes
    srcs = ["qwen3_0_6b_custom_sampling_test.py"],
    args = ["--device=tpu"],
    main = "qwen3_0_6b_custom_sampling_test.py",
    nonightly_oss = "Disabled during switch from opt-in to opt-out",
    nopresubmit = "slow example",
    tags = [
        "nomsan",  # 3p 'rust/tempfile' error: use-of-uninitialized-value in tempfile::file::imp::platform::reopen
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_0_6b_custom_sampling_test_cuda",
    srcs = ["qwen3_0_6b_custom_sampling_test.py"],
    args = ["--device=cuda"],
    main = "qwen3_0_6b_custom_sampling_test.py",
    notap = "slow example",
    tags = [
        "local",
        "requires-gpu-sm80",  # A100 or above
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

ipynb_deps(
    name = "qwen3_0_6b_custom_sampling_deps",
    srcs = ["qwen3_0_6b_custom_sampling.ipynb"],
    pytype = False,
    deps = [
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

ipynb_deps(
    name = "qwen3_0_6b_deps",
    srcs = ["qwen3_0_6b.ipynb"],
    pytype = False,
    deps = [
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "test_qwen3_vl_load",
    srcs = ["test_qwen3_vl_load.py"],
    data = [
        "//examples/huggingface_transformers:model_configs_data",
    ],
    notap = "manual test for dynamic performance evaluation",
    notest_oss = "manual test for dynamic performance evaluation",
    tags = [
        "nomsan",  # 3p 'torchvision' is uninstrumented in msan mode.
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@pypi//pillow",
        "@pypi//torchvision",  # build_cleaner: keep
        "@pypi//transformers",
        "@rules_python//python/runfiles",
    ],
)

torch_tpu_py_test(
    name = "qwen3_vl_dynamic_kv_cache",
    timeout = "eternal",
    srcs = ["qwen3_vl_dynamic_kv_cache.py"],
    data = [
        "//examples/huggingface_transformers:model_configs_data",
    ],
    nolocal = "slow example",
    notap = "manual test for dynamic performance evaluation",
    notest_oss = "manual test for dynamic performance evaluation",
    tags = [
        "cpu:4",
        "nofastbuild",  # Benchmark.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/benchmarks:xprof_adapter",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//pillow",
        "@pypi//torchvision",  # build_cleaner: keep
        "@pypi//transformers",
        "@rules_python//python/runfiles",
    ],
)

torch_tpu_package_end()
