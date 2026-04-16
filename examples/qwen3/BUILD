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
    nopresubmit = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples/qwen3/impl:qwen3",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "qwen_kv_cache",
    srcs = ["qwen_kv_cache.py"],
    nopresubmit = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "cpu:8",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/profile_cpu:python_tracer",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
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
    notest_oss = "Disabled during switch from opt-in to opt-out",
    strict = True,
    tags = [
        "cpu:8",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/profile_cpu:python_tracer",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
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
    notest_oss = "Disabled during switch from opt-in to opt-out",
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
        "//shims/profile_cpu:python_tracer",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "@pypi//transformers",
        "@rules_python//python/runfiles",
    ],
)

pytype_strict_binary(
    name = "xm_launch_bd",
    srcs = ["xm_launch_bd.py"],
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
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_hf_pipe_example",
    timeout = "long",  # 15 minutes
    srcs = ["qwen3_hf_pipe_example.py"],
    nopresubmit = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-net:external",  # Needed to access remote files.
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compiler_options",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_0_6b_forward_test_vl",
    srcs = ["qwen3_0_6b_forward_test.py"],
    args = ["--device=tpu"],
    main = "qwen3_0_6b_forward_test.py",
    nopresubmit = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples/huggingface_transformers:model_configs",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_0_6b_forward_test_v100",
    srcs = ["qwen3_0_6b_forward_test.py"],
    args = ["--device=cuda"],
    main = "qwen3_0_6b_forward_test.py",
    notap = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "ignore_for_dep=third_party.torch_tpu.api",
        "requires-gpu-sm70-only",  # v100 has 16 or 32 GB HBM.
    ],
    deps = [
        "//shims/absl_py/logging",
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu/_internal/utils:log_utils",
        "//examples/huggingface_transformers:model_configs",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "@pypi//transformers",
        # Do not add "//torch_tpu/api" here.
    ],
)

torch_tpu_py_test(
    name = "qwen3_0_6b_forward_test_a100_xla_cuda",
    srcs = ["qwen3_0_6b_forward_test.py"],
    args = ["--device=xla_cuda"],
    main = "qwen3_0_6b_forward_test.py",
    notap = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-gpu-sm80-only",
    ],
    deps = [
        "//examples/huggingface_transformers:model_configs",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
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
    nopresubmit = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
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
        "//torch_tpu",  # build_cleaner: keep
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
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "ignore_for_dep=third_party.torch_tpu.api",
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
        "//torch_tpu",  # build_cleaner: keep
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

torch_tpu_package_end()
