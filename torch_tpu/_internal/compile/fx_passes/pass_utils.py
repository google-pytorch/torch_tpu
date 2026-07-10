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

"""Utilities for FX passes."""

import enum
import torch


class DeviceId(enum.Enum):
  """Enum representing device types supported in FX passes."""

  CPU = "cpu"
  TPU = "tpu"
  TRANSFER = "transfer"
  UNKNOWN = "unknown"

  @classmethod
  def from_device(cls, device: str | torch.device) -> "DeviceId":
    """Parses a device name or torch.device to DeviceId enum.

    Args:
      device: A string or torch.device object.

    Returns:
      The mapped DeviceId enum value.
    """
    if hasattr(device, "type"):
      device = device.type
    if device.lower() == "cpu":
      return DeviceId.CPU
    elif device.lower() == "tpu":
      return DeviceId.TPU
    return DeviceId.UNKNOWN


def has_dynamic_shape(tensor: torch.Tensor) -> bool:
  """Checks if a tensor has dynamic shape."""
  return any(isinstance(dim, torch.SymInt) for dim in tensor.size())


def get_node_device(node: torch.fx.Node) -> DeviceId:
  """Gets the device type of a node based on its meta value.

  Args:
    node: The FX node to get the device for.

  Returns:
    The DeviceId of the node.
  """

  def from_meta_val(node: torch.fx.Node) -> DeviceId | None:
    if "val" in node.meta:
      val = node.meta["val"]
      if isinstance(val, torch.Tensor):
        return DeviceId.from_device(val.device)
      if isinstance(val, (tuple, list)):
        devices = [
            DeviceId.from_device(v.device)
            for v in val
            if isinstance(v, torch.Tensor)
        ]
        return devices[0] if devices else None
    return None

  res = from_meta_val(node)

  if node.target == torch.ops.aten._to_copy.default:  # pylint: disable=protected-access
    if res:
      for arg in node.args:
        if isinstance(arg, torch.fx.Node):
          arg_res = from_meta_val(arg)
          if arg_res and arg_res != res:
            return DeviceId.TRANSFER

  if res:
    return res

  return DeviceId.UNKNOWN


def is_dependent(node: torch.fx.Node, maybe_ancestor: torch.fx.Node) -> bool:
  """Checks if a node is dependent on another node (ancestor) in the graph.

  Args:
    node: The candidate dependent node.
    maybe_ancestor: The candidate ancestor node.

  Returns:
    True if node depends on maybe_ancestor, False otherwise.
  """
  seen = set()
  stack = list(node.all_input_nodes)

  while stack:
    curr = stack.pop()

    if curr in stack:
      continue

    if curr == maybe_ancestor:
      return True

    seen.add(curr)
    stack.extend(curr.all_input_nodes)

  return False
