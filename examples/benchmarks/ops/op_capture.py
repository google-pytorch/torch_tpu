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

from collections import Counter
import json
from typing import Any, Dict, List, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class OpCaptureMode(TorchDispatchMode):
  """A TorchDispatchMode that intercepts ATen operations and records their

  metadata and frequency. Supports basic clustering of operations.
  """

  def __init__(self, cluster_size: int = 0):
    super().__init__()
    # Key: (op_name, serialized_args, serialized_kwargs) or ("CLUSTER", tuple_of_keys)
    # Value: count
    self.op_counts = Counter()
    self.cluster_size = cluster_size
    self.current_cluster = []

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if kwargs is None:
      kwargs = {}

    op_name = str(func)

    def serialize_arg(arg):
      if isinstance(arg, torch.Tensor):
        return ("T", tuple(arg.shape), arg.dtype)
      elif isinstance(arg, (list, tuple)):
        return tuple(serialize_arg(a) for a in arg)
      elif isinstance(arg, dict):
        return (
            "DICT",
            tuple(sorted((k, serialize_arg(v)) for k, v in arg.items())),
        )
      elif isinstance(arg, (int, float, str, bool)) or arg is None:
        return arg
      else:
        return str(arg)

    serialized_args = tuple(serialize_arg(a) for a in args)
    serialized_kwargs = tuple(
        sorted((k, serialize_arg(v)) for k, v in kwargs.items())
    )

    op_data = (op_name, serialized_args, serialized_kwargs)

    # Handle clustering
    if self.cluster_size > 0:
      self.current_cluster.append(op_data)
      if len(self.current_cluster) >= self.cluster_size:
        cluster_key = ("CLUSTER", tuple(self.current_cluster))
        self.op_counts[cluster_key] += 1
        self.current_cluster = []  # Reset

    # Also record individual op (Intentional, to have both as requested by user)
    self.op_counts[op_data] += 1

    return func(*args, **kwargs)

  def __exit__(self, exc_type, exc_val, exc_tb):
    # Flush remaining cluster if any
    if self.cluster_size > 0 and self.current_cluster:
      cluster_key = ("CLUSTER", tuple(self.current_cluster))
      self.op_counts[cluster_key] += 1
      self.current_cluster = []
    super().__exit__(exc_type, exc_val, exc_tb)

  def save_to_json(self, file_path: str):
    """Saves the captured operator counts to a JSON file."""
    output = {}

    def to_code_str(val):
      if isinstance(val, tuple) and len(val) == 3 and val[0] == "T":
        shape = val[1]
        dtype = val[2]
        return f"T({list(shape)}, {dtype})"
      elif isinstance(val, tuple) and len(val) == 2 and val[0] == "DICT":
        items = [f"{k!r}: {to_code_str(v)}" for k, v in val[1]]
        return "{" + ", ".join(items) + "}"
      elif isinstance(val, tuple):
        items = [to_code_str(x) for x in val]
        return "[" + ", ".join(items) + "]"
      elif isinstance(val, str):
        return f"{val!r}"
      return str(val)

    def serialize_op_tuple(op_tuple):
      if isinstance(op_tuple, tuple) and len(op_tuple) == 3:
        op_name = op_tuple[0]
        args_tuple = op_tuple[1]
        kwargs_tuple = op_tuple[2]

        args_code = [to_code_str(a) for a in args_tuple]

        kwargs_code = {}
        for item in kwargs_tuple:
          if isinstance(item, tuple) and len(item) == 2:
            k = item[0]
            v = item[1]
            kwargs_code[k] = to_code_str(v)
          else:
            raise ValueError(f"Expected pair, got {type(item)}")

        args_str = "[" + ", ".join(args_code) + "]"
        kwargs_str = (
            "{" + ", ".join(f"{k!r}: {v}" for k, v in kwargs_code.items()) + "}"
        )
        return f"({args_str}, {kwargs_str})"
      else:
        raise ValueError(f"Expected 3-tuple, got {type(op_tuple)}")

    for key, count in self.op_counts.items():
      if isinstance(key, tuple) and len(key) == 2 and key[0] == "CLUSTER":
        # Handle cluster
        cluster_ops = key[1]
        if "CLUSTER" not in output:
          output["CLUSTER"] = []

        ops_list = []
        for op_tuple in cluster_ops:
          if isinstance(op_tuple, tuple) and len(op_tuple) == 3:
            op_name = op_tuple[0]
            ops_list.append(
                {"op": op_name, "inputs": serialize_op_tuple(op_tuple)}
            )
          else:
            raise ValueError(f"Expected 3-tuple, got {type(op_tuple)}")

        output["CLUSTER"].append({"ops": ops_list, "count": count})
      else:
        # Handle normal op
        if isinstance(key, tuple) and len(key) == 3:
          op_name = key[0]
          if op_name not in output:
            output[op_name] = []

          output[op_name].append(
              {"inputs": serialize_op_tuple(key), "count": count}
          )
        else:
          raise ValueError(f"Expected 3-tuple, got {type(key)}")

    with open(file_path, "w") as f:
      json.dump(output, f, indent=4)
