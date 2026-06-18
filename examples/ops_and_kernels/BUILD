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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
)

torch_tpu_py_test(
    name = "qat_linear",
    srcs = ["qat_linear.py"],
    main = "qat_linear.py",
    nopresubmit = "slow example",
    shuffle_tests = False,
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "call_linear",
    srcs = ["call_linear.py"],
    main = "call_linear.py",
    nopresubmit = "slow example",
    shuffle_tests = False,
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "pointwise_fusion_cuda",
    srcs = ["pointwise_fusion_cuda.py"],
    main = "pointwise_fusion_cuda.py",
    notap = "slow example",
    tags = ["requires-gpu-sm80"],  # Triton requires Ampere+
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "pointwise_fusion_xla",
    srcs = ["pointwise_fusion_xla.py"],
    args = ["--undefok=xla_jf_dump_to,xla_jf_dump_llo_text"],
    main = "pointwise_fusion_xla.py",
    nopresubmit = "slow example",
    shuffle_tests = False,
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "sigmoid_no_vanishing_grad",
    srcs = ["sigmoid_no_vanishing_grad.py"],
    main = "sigmoid_no_vanishing_grad.py",
    nopresubmit = "slow example",
    shuffle_tests = False,
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "quantized_sum",
    srcs = ["quantized_sum.py"],
    args = ["--undefok=xla_jf_dump_to,xla_jf_dump_llo_text"],
    main = "quantized_sum.py",
    nopresubmit = "slow example",
    shuffle_tests = False,
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/pallas",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "quant",
    srcs = ["quant.py"],
    args = ["--undefok=xla_jf_dump_to,xla_jf_dump_llo_text"],
    main = "quant.py",
    nopresubmit = "slow example",
    shuffle_tests = False,
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "quant_pallas",
    srcs = ["quant_pallas.py"],
    args = ["--undefok=xla_jf_dump_to,xla_jf_dump_llo_text"],
    main = "quant_pallas.py",
    nopresubmit = "slow example",
    shuffle_tests = False,
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/pallas",
        "@pypi//jax",
    ],
)

torch_tpu_package_end()
