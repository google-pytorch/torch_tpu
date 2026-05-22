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

import ast
import logging
from typing import Any, Dict, List, Tuple
import torch


def deserialize_tensor(
    shape: List[int], dtype: torch.dtype, device: str = "tpu"
) -> torch.Tensor:
  """Creates a synthetic random tensor on the specified device."""
  # Ensure device is mapped correctly for PyTorch
  if device == "tpu":
    # In TorchTPU, device is accessed via 'tpu' after registration.
    torch_device = torch.device("tpu")
  else:
    torch_device = torch.device(device)

  # Generate random data based on dtype
  if dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
    return torch.randn(shape, dtype=dtype, device=torch_device)
  elif dtype in (
      torch.int64,
      torch.int32,
      torch.int16,
      torch.int8,
      torch.uint8,
  ):
    # Generate integers in a reasonable range
    return torch.randint(0, 100, shape, dtype=dtype, device=torch_device)
  elif dtype == torch.bool:
    return torch.randint(0, 2, shape, dtype=dtype, device=torch_device).to(
        torch.bool
    )
  else:
    # Fallback for other dtypes
    logging.warning(f"Fallback to zeros for dtype {dtype}")
    return torch.zeros(shape, dtype=dtype, device=torch_device)


def deserialize_args(
    inputs_str: str, device: str = "tpu"
) -> Tuple[List[Any], Dict[str, Any]]:
  """Deserializes the inputs string into actual arguments and keyword arguments,

  creating tensors on the specified device.
  Uses AST parsing for safety.
  """
  try:
    tree = ast.parse(inputs_str, mode="eval")
    if not isinstance(tree, ast.Expression):
      raise ValueError("Expected an expression")

    root = tree.body
    if not isinstance(root, ast.Tuple) or len(root.elts) != 2:
      raise ValueError("Expected a tuple of (args, kwargs)")

    args_node = root.elts[0]
    kwargs_node = root.elts[1]

    def eval_node(node):
      if isinstance(node, ast.Tuple):
        return tuple(eval_node(x) for x in node.elts)
      elif isinstance(node, ast.List):
        return [eval_node(x) for x in node.elts]
      elif isinstance(node, ast.Dict):
        return {
            eval_node(k): eval_node(v) for k, v in zip(node.keys, node.values)
        }
      elif isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, str):
          if val.startswith("torch."):
            parts = val.split(".")
            if len(parts) == 2:
              attr_name = parts[1]
              attr = getattr(torch, attr_name, None)
              if isinstance(attr, (torch.dtype, torch.memory_format)):
                return attr
          elif "tpu" in val and device != "tpu":
            return val.replace("tpu", device)
        return val
      elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
          return -eval_node(node.operand)
        else:
          raise ValueError(f"Unsupported unary operator: {type(node.op)}")
      elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "T":
          if len(node.args) != 2:
            raise ValueError("T expects 2 arguments")
          shape_node = node.args[0]
          dtype_node = node.args[1]

          # Evaluate shape
          if not isinstance(shape_node, ast.List):
            raise ValueError("T shape must be a list")
          shape = []
          for x in shape_node.elts:
            val = eval_node(x)
            if not isinstance(val, int):
              raise ValueError("All shape elements must be integers")
            shape.append(val)

          # Evaluate dtype
          if not isinstance(dtype_node, ast.Attribute):
            raise ValueError("T dtype must be a torch.dtype")
          if (
              not isinstance(dtype_node.value, ast.Name)
              or dtype_node.value.id != "torch"
          ):
            raise ValueError("T dtype must be accessed via torch.")
          dtype_str = dtype_node.attr
          dtype = getattr(torch, dtype_str, None)
          if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown or invalid dtype: {dtype_str}")

          return deserialize_tensor(shape, dtype, device=device)
        else:
          raise ValueError(
              "Unsupported function call:"
              f" {node.func.id if isinstance(node.func, ast.Name) else type(node.func)}"
          )
      else:
        raise ValueError(f"Unsupported AST node: {type(node)}")

    args = eval_node(args_node)
    kwargs = eval_node(kwargs_node)

    if not isinstance(args, list):
      raise ValueError("Args must be a list")
    if not isinstance(kwargs, dict):
      raise ValueError("Kwargs must be a dict")

    return args, kwargs

  except Exception as e:
    raise ValueError(
        f"Failed to deserialize inputs string: {inputs_str}. Error: {e}"
    ) from e
