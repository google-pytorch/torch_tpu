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
"""Adding dependency for input_preprocessing library within jax_tpu_embedding.

Note: Despite residing within jax_tpu_embedding, the core C++ SparseCore
preprocessing library is framework-independent.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

JAX_TPU_EMBEDDING_COMMIT = "9772781da67c3d39d87f52d184f4f179043726bd"
JAX_TPU_EMBEDDING_SHA256 = "264047b40dcded934b91670c5d4ea08bead6e7f00d9b517746bc7fa8234be03d"

def jax_tpu_embedding_repo():
    http_archive(
        name = "jax_tpu_embedding",
        sha256 = JAX_TPU_EMBEDDING_SHA256,
        strip_prefix = "jax-tpu-embedding-{commit}".format(commit = JAX_TPU_EMBEDDING_COMMIT),
        urls = [
            "https://github.com/jax-ml/jax-tpu-embedding/archive/{commit}.tar.gz".format(commit = JAX_TPU_EMBEDDING_COMMIT),
        ],
    )
