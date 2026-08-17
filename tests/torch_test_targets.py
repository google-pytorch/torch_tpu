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

"""Per-module customization for the generated upstream test targets.

gen_torch_tests.py discovers the upstream modules that instantiate
device-generic tests for the tpu backend and wires every one of them with
default kwargs (collect everything, then exclude). This file holds the only
hand-maintained target state:

  TARGET_KWARGS: extra `torch_tpu_oss_torch_test` kwargs for modules whose
    defaults do not fit — "shard_count" (op_db-parametrized suites explode
    to thousands of cases per device), "extra_deps" (guarded module-level
    imports upstream would otherwise skip meaningful test groups on), and
    "args" (forwarded to the upstream unittest main). Which tests run
    inside a module is not configured here: see tests/torch_test_configs.py.

  OMITTED_TEST_MODULES: modules deliberately NOT wired even though (or in
    case) they instantiate tpu tests, each with a reason. The deny-list
    overrules discovery. Revisit when bumping the tested PyTorch version.

Module ids are paths under pytorch/test/ without ".py": "test_torch",
"nn/test_pooling".
"""

TARGET_KWARGS = {
    # Not eligible under v2.13.0 (no device-generic call yet); kept for
    # nightly-tracking runs, where it is.
    "distributions/test_distributions": {"shard_count": 8},
    # jax: two debugging-utility tests import it directly.
    "dynamo/test_debug_utils": {"extra_deps": ["@pypi//jax"]},
    "dynamo/test_misc": {"shard_count": 8},
    # pyyaml: imported at module level; without it every shard aborts.
    "inductor/test_compiled_autograd": {
        "extra_deps": ["@pypi//pyyaml"],
        "shard_count": 8,
    },
    "inductor/test_compiled_optimizers": {"shard_count": 8},
    "inductor/test_torchinductor_dynamic_shapes": {"shard_count": 8},
    # op_db-parametrized: thousands of cases per device.
    "inductor/test_torchinductor_opinfo": {"shard_count": 16},
    "inductor/test_torchinductor_opinfo_properties": {"shard_count": 8},
    "nn/test_convolution": {"shard_count": 8},
    "nn/test_embedding": {"shard_count": 4},
    "nn/test_pooling": {"shard_count": 4},
    "profiler/test_profiler": {"shard_count": 4},
    "test_binary_ufuncs": {
        "extra_deps": ["@pypi//scipy"],
        "shard_count": 8,
    },
    # op_db-parametrized: thousands of cases per device.
    "test_decomp": {"shard_count": 16},
    "test_foreach": {"shard_count": 4},
    "test_indexing": {"shard_count": 4},
    "test_linalg": {
        "extra_deps": ["@pypi//packaging", "@pypi//scipy"],
        "shard_count": 8,
    },
    # op_db-parametrized: thousands of cases per device.
    "test_meta": {
        "extra_deps": ["@pypi//pyyaml"],
        "shard_count": 16,
    },
    # module_db-parametrized.
    "test_modules": {"shard_count": 8},
    "test_nestedtensor": {"shard_count": 8},
    "test_nn": {"shard_count": 8},
    # op_db-parametrized: the largest suite by far.
    "test_ops": {"shard_count": 16},
    "test_ops_fwd_gradients": {"shard_count": 8},
    "test_ops_gradients": {"shard_count": 8},
    "test_ops_jit": {"shard_count": 8},
    "test_optim": {"shard_count": 4},
    # Import touches device APIs the generator's fake backend cannot serve;
    # eligible under the real torch_tpu backend, so the entry force-wires it.
    "test_overrides": {},
    "test_prims": {"extra_deps": ["@pypi//scipy"]},
    "test_reductions": {"shard_count": 8},
    "test_serialization": {"extra_deps": ["@pypi//dill"]},
    "test_sort_and_select": {"shard_count": 4},
    "test_sparse": {
        "extra_deps": ["@pypi//packaging", "@pypi//scipy"],
        "shard_count": 8,
    },
    "test_sparse_csr": {
        "extra_deps": ["@pypi//scipy"],
        "shard_count": 8,
    },
    "test_spectral_ops": {
        "extra_deps": ["@pypi//packaging", "@pypi//scipy"],
        "shard_count": 4,
    },
    "test_tensor_creation_ops": {"shard_count": 8},
    "test_testing": {"shard_count": 4},
    "test_torch": {"shard_count": 8},
    "test_type_promotion": {"shard_count": 4},
    # 32 shards: coarser shards accumulate enough compiled programs over
    # thousands of shape/dtype variants that the OOM killer takes them.
    "test_unary_ufuncs": {
        "extra_deps": ["@pypi//scipy"],
        "shard_count": 32,
    },
    "test_view_ops": {"shard_count": 4},
}

OMITTED_TEST_MODULES = {
    "cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing": (
        "OpenReg extension's own self-test; exercises the openreg backend,"
        " not tpu"
    ),
    "cpp_extensions/test_libtorch_agnostic": (
        "compiles a C++ extension at import; not hermetically runnable"
    ),
    "onnx/test_op_consistency": (
        "requires onnxruntime, not in the test environment"
    ),
    "onnx/torchlib/test_ops": (
        "requires onnxruntime, not in the test environment"
    ),
    "profiler/test_cpp_thread": (
        "compiles a C++ extension at import; not hermetically runnable"
    ),
    "test_cuda": "CUDA-gated classes; zero tests instantiate for tpu",
    "test_custom_ops": "only_for=(cpu, cuda, xpu)",
    "test_fake_tensor": "only_for=(cpu, cuda)",
    "test_functionalization_of_rng_ops": (
        "only_for=(cuda,) and only_for=(cpu,)"
    ),
    "test_jit_fuser_te": "CPU/CUDA fuser-specific",
    "test_jiterator": "CUDA jiterator-specific",
    "test_maskedtensor": "only_for=(cpu, cuda)",
    "test_mkldnn": "oneDNN (CPU-only); needs torchvision",
    "test_mps": "MPS-specific",
    "test_ops_unbacked": "only_for=(cpu,)",
    "test_proxy_tensor": "device-generic call is cpu-only",
    "test_schema_check": "device-generic call is cpu-only",
    "test_sparse_semi_structured": "CUDA sparse semi-structured kernels",
    "test_stateless_rng": "only_for=(cuda,) on all three classes",
    "test_transformers": (
        "SDPA classes are cuda-gated; needs fairseq for meaningful subset"
    ),
    "test_varlen_attention": "CUDA varlen attention",
}
