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

"""API for providing XLA specific annotations on torch tensors."""

import dataclasses
from typing import List
from torch_tpu._internal.device_utils import annotations_py

field = dataclasses.field
dataclass = dataclasses.dataclass


@dataclass
class TpuLayout:
  """Mimics layout class from xla/layout.h.

  Attributes:
    minor_to_major: The minor-to-major ordering of dimensions.
    tiles: The tiling of the layout. Each tile is a list of integers
      representing the tile dimensions.
    element_size_in_bits: The size of the element in bits, if it is a sub-byte
      type.
  """

  minor_to_major: List[int]
  tiles: List[List[int]] = field(default_factory=list)
  element_size_in_bits: int = 0


class LayoutContext:
  """Context manager to set the layout hint for TPU tensors.

  Usage:
    layout = TpuLayout(minor_to_major=[1, 0], tiles=[[16, 128], [2, 1]])
    with LayoutContext(layout):
      # All TPU tensors created in this block will be created with the
      # specified layout.
      ...
  """

  def __init__(self, layout: TpuLayout):
    """Initializes the LayoutContext.

    Usage:
      with LayoutContext(layout):
        ...

    Args:
      layout: The TpuLayout instance.
    """
    if not isinstance(layout, TpuLayout):
      raise TypeError("layout must be an instance of TpuLayout")
    self._layout = layout

  def __enter__(self):
    annotations_py.enter_layout_context(
        self._layout.minor_to_major,
        self._layout.tiles,
        self._layout.element_size_in_bits,
    )

  def __exit__(self, *args):
    annotations_py.exit_layout_context()
