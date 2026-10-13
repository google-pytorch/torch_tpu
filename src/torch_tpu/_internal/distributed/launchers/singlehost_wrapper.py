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

"""Deprecated compatibility shim for single-host TPU launch environment setup.

Use `torch.tpu.distributed.set_tpu_launch_env` instead.
"""

from torch_tpu._internal.distributed import distributed
from torch_tpu._internal.utils import annotations


@annotations.deprecated(
    version="0.2.0",
    reason=(
        "Use torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=...)"
        " instead."
    ),
)
def prepare_tpu_environment(world_size: int | None = None) -> None:
  """Prepares the environment variables required for TPU distributed execution.

  Deprecated: use `torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=...)`
  instead.

  Args:
    world_size: If provided, use this as the target device count instead of
      auto-detecting from hardware.
  """
  distributed.set_tpu_launch_env(nproc_per_node=world_size)


if __name__ == "__main__":
  distributed._print_launch_env()  # pylint: disable=protected-access
