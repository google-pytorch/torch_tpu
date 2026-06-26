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

# pyre-strict

"""Helpers for SparseCore CSR inputs used by benchmark harnesses."""

from typing import Any

import numpy as np


def sanitize_minibatched_row_pointers(
    row_pointers: Any,
    max_ids_per_partition: Any,
    num_minibatches: Any,
    *,
    row_pointers_per_sparsecore_per_minibatch: int = 8,
) -> np.ndarray:
    """Preserve valid JTE minibatched CSR row pointers and blank padded rows.

    JTE's minibatched preprocessing returns a physical row-pointer buffer whose
    valid prefix is `8 * num_sparsecores * num_minibatches` entries. The padded
    tail may contain uninitialized host memory. TorchTPU direct custom calls see
    the full physical tensor, so convert that tail to empty CSR rows by repeating
    the final valid pointer.
    """

    sanitized = np.asarray(row_pointers).copy()
    num_sparsecores = len(np.asarray(max_ids_per_partition).reshape(-1))
    max_num_minibatches = int(np.max(np.asarray(num_minibatches)))
    valid_row_pointers = (
        row_pointers_per_sparsecore_per_minibatch
        * num_sparsecores
        * max_num_minibatches
    )
    if valid_row_pointers <= 0:
        raise ValueError("minibatched CSR row pointers must have a valid prefix")
    if valid_row_pointers > sanitized.size:
        raise ValueError(
            "valid minibatched CSR row-pointer prefix exceeds physical buffer: "
            f"{valid_row_pointers} > {sanitized.size}"
        )
    if valid_row_pointers < sanitized.size:
        sanitized[valid_row_pointers:] = sanitized[valid_row_pointers - 1]
    return sanitized
