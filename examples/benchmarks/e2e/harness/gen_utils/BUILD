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

# Workaround to avoid querying for data dependencies which doesn't play well in OSS.
# Just stick the csv contents in a python source file directly.
genrule(
    name = "models_csv_py",
    srcs = ["models.csv"],
    outs = ["models_csv.py"],
    cmd = "echo 'MODELS_CSV = \"\"\"' > $@ && cat $(location models.csv) >> $@ && echo '\"\"\"' >> $@",
)

torch_tpu_py_library(
    name = "models_csv",
    srcs = ["models_csv.py"],
)

torch_tpu_py_library(
    name = "gen_utils",
    srcs = ["gen_utils.py"],
    deps = [
        ":load_model",
        ":models_csv",
        "//examples/benchmarks/e2e/harness:context",
        "//examples/benchmarks/e2e/harness:loss",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "load_model",
    srcs = ["load_model.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:context",
        "//examples/benchmarks/e2e/harness:torch_device_ops",
        "//shims/torch:pytorch",
        "//tests:module_registry",
    ],
)

torch_tpu_py_test(
    name = "gen_utils_test",
    srcs = ["gen_utils_test.py"],
    deps = [
        ":gen_utils",
        "//shims/absl_py/testing:absltest",
        "//testing/pybase",
        "//torch_tpu",
    ],
)

torch_tpu_package_end()
