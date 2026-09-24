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
        "//src/torch_tpu/_internal/utils:test_utils",
        "//src/torch_tpu/extras/optimizers:adamw",
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
        "//src/torch_tpu/_internal/utils:test_utils",
        "//src/torch_tpu/extras/optimizers:sgd",
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
        "//src/torch_tpu/_internal/utils:test_utils",
        "//src/torch_tpu/extras/optimizers:muon",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "single_trace_trainer_test",
    srcs = ["single_trace_trainer_test.py"],
    main = "single_trace_trainer_test.py",
    nopresubmit = "slow example",
    notest_oss = True,  # slow example
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/extras/optimizers:adamw",
        "//src/torch_tpu/extras/optimizers:sgd",
        "//src/torch_tpu/extras/optimizers:single_trace_trainer",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_package_end()
