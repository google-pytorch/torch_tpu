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

"""Deprecated compatibility shim for multi-host TPU launch environment setup.

Use `torch.tpu.distributed.set_tpu_launch_env` instead.
"""

from collections.abc import Sequence

from absl import logging
from torch_tpu._internal.distributed import distributed
from torch_tpu._internal.utils import annotations


@annotations.deprecated(
    version="0.2.0",
    reason="Use torch.tpu.distributed.set_tpu_launch_env instead.",
)
def calculate_topological_node_rank(
    coords: Sequence[int] | None,
    topology_str: str | None,
    fallback_index: str | int | None = None,
    chips_per_host: tuple[int, int, int] = (2, 2, 1),
) -> int:
  """Deprecated shim; calculates the topologically sorted NODE_RANK."""
  return distributed._calculate_topological_node_rank(  # pylint: disable=protected-access
      coords,
      topology_str,
      fallback_index=fallback_index,
      chips_per_host=chips_per_host,
  )


@annotations.deprecated(
    version="0.2.0",
    reason="Use torch.tpu.distributed.set_tpu_launch_env instead.",
)
def prepare_tpu_environment() -> None:
  """Deprecated shim; use `torch.tpu.distributed.set_tpu_launch_env` instead."""
  distributed.set_tpu_launch_env()


if __name__ == "__main__":
  logging.get_absl_logger().propagate = False
  distributed._print_launch_env()  # pylint: disable=protected-access
