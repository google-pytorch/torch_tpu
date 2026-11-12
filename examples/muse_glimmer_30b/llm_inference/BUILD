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
    "torch_tpu_oss_doc_test",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

exports_files([
    "single_tpu_inference_dynamic_test.py",
    "single_tpu_inference_static_test.py",
])

torch_tpu_py_library(
    name = "kv_cache",
    srcs = [
        "kv_cache.py",
    ],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "rope",
    srcs = [
        "rope.py",
    ],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "model",
    srcs = [
        "model.py",
    ],
    deps = [
        ":kv_cache",
        ":rope",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "checkpoint_utils",
    srcs = [
        "checkpoint_utils.py",
    ],
    deps = [
        ":model",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

torch_tpu_py_library(
    name = "summary_utils",
    srcs = [
        "summary_utils.py",
    ],
)

# The cache tests pick the `tpu` compile backend when a TPU is available and
# `aot_eager` otherwise, so the same source runs on CPU and on TPU.
torch_tpu_py_test(
    name = "swa_cache_test",
    srcs = ["swa_cache_test.py"],
    tags = ["nofastbuild"],  # Slow example.
    deps = [
        ":kv_cache",
        ":model",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "swa_cache_tpu_test",
    srcs = ["swa_cache_test.py"],
    main = "swa_cache_test.py",
    notest_oss = True,  # NOTEST_OSS_OK=Requires Cloud TPU hardware not available on OSS CI runners.
    tags = [
        "nofastbuild",  # Slow example.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":kv_cache",
        ":model",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_library(
    name = "runner_dynamic",
    srcs = [
        "runner_dynamic.py",
    ],
    deps = [
        ":checkpoint_utils",
        ":model",
        ":summary_utils",
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

torch_tpu_py_library(
    name = "runner_static",
    srcs = [
        "runner_static.py",
    ],
    deps = [
        ":checkpoint_utils",
        ":kv_cache",
        ":model",
        ":summary_utils",
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "single_tpu_inference_dynamic_test",
    size = "enormous",
    srcs = ["single_tpu_inference_dynamic_test.py"],
    args = [
        "--xla_tpu_dvfs_p_state=7",  # Required to get 2307 TFLOP/s on Ironwood
    ],
    notap = "manual performance run",
    tags = [
        "cpu:8",
        "nofastbuild",  # Measures performance.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",  # Ironwood
    ],
    deps = [
        ":runner_dynamic",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "single_tpu_inference_static_test",
    size = "enormous",
    srcs = ["single_tpu_inference_static_test.py"],
    args = [
        "--xla_tpu_dvfs_p_state=7",  # Required to get 2307 TFLOP/s on Ironwood
    ],
    notap = "manual performance run",
    tags = [
        "cpu:8",
        "nofastbuild",  # Measures performance.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",  # Ironwood
    ],
    deps = [
        ":runner_static",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "kv_cache_test",
    srcs = ["kv_cache_test.py"],
    # Restrict HWASan pointer tags on Arm to 5 bits (bits 56..60) so they do
    # not collide with PyTorch c10::SymInt's 3-bit tag packing in bits 61..63.
    env = {"HWASAN_OPTIONS": "tag_bits=5"},
    deps = [
        ":kv_cache",
        ":model",
        ":runner_dynamic",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

# OSS only: run the documented commands for the tests above as a user would;
# see torch_tpu_oss_doc_test. swa_cache_doc_test also covers
# swa_cache_tpu_test, which runs the same script.
torch_tpu_oss_doc_test(
    name = "swa_cache_doc_test",
    doc = "README.md",
    deps = [
        ":kv_cache",
        ":model",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_oss_doc_test(
    name = "single_tpu_inference_dynamic_doc_test",
    data = ["launch.sh"],
    doc = "README.md",
    notest_oss = True,  # NOTEST_OSS_OK=Mirrors :single_tpu_inference_dynamic_test, which is notap.
    deps = [
        ":runner_dynamic",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
    ],
)

torch_tpu_oss_doc_test(
    name = "single_tpu_inference_static_doc_test",
    data = ["launch.sh"],
    doc = "README.md",
    notest_oss = True,  # NOTEST_OSS_OK=Mirrors :single_tpu_inference_static_test, which is notap.
    deps = [
        ":runner_static",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
    ],
)

torch_tpu_oss_doc_test(
    name = "kv_cache_doc_test",
    doc = "README.md",
    deps = [
        ":kv_cache",
        ":model",
        ":runner_dynamic",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_package_end()
