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

"""Implementation of `torch.tpu.pallas.get_pallas_mesh`.

This lives apart from `torch_tpu._internal.pallas.pallas` because that module
keeps a deprecated `get_pallas_mesh` alias for existing callers, and the public
API must be a distinct object so that its lifecycle stage is recorded
separately.
"""

from collections.abc import Sequence

import jax
import torch.distributed as dist
from torch_tpu._internal.pallas import pallas
from torch_tpu._internal.utils import annotations


@annotations.experimental(
    "torch.tpu.pallas.get_pallas_mesh is experimental and subject to change."
)
def get_pallas_mesh(
    axis_names: Sequence[str] | str,
    mesh_shape: Sequence[int] | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> jax.sharding.Mesh:
  """Constructs a rank-aligned JAX Mesh using TPU runtime device metadata.

  The mesh maps each position to the TPU device driven by the corresponding
  torch rank, so it can be passed to Pallas kernels that communicate across
  devices.

  .. warning::
      This API is experimental and subject to change in future releases.

  Args:
    axis_names: Names of the mesh axes.
    mesh_shape: Shape of the mesh. Required when there is more than one axis;
      its product must equal the size of `process_group` (the world size if
      `process_group` is None). Defaults to a 1D mesh over all ranks of the
      process group.
    process_group: The process group whose rank-to-device assignment to use.
      Defaults to the world process group.

  Returns:
    A `jax.sharding.Mesh` over the process group's TPU devices (in group-rank
    order).

  Raises:
    RuntimeError: If `torch.distributed` is not initialized.
    ValueError: If `axis_names` and `mesh_shape` are inconsistent with each
      other or with the process group size.
  """
  return pallas._get_pallas_mesh(  # pylint: disable=protected-access
      axis_names, mesh_shape=mesh_shape, process_group=process_group
  )


# DO NOT SUBMIT: fake API to check how the OSS newest-deps presubmit reports
# missing tt_apis.yaml entries.
@annotations.experimental(
    "torch.tpu.pallas.fake_api_for_ci_test is a DO-NOT-SUBMIT test API."
)
def fake_api_for_ci_test() -> int:
  """Returns 42. DO NOT SUBMIT."""
  return 42
