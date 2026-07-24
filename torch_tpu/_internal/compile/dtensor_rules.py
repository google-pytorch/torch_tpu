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

"""DTensor sharding propagation rules for TorchTPU custom ops."""

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor._op_schema import OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule


@register_prop_rule(torch.ops.tpu.sparse_dense_matmul.default)
def sparse_dense_matmul_rule(op_schema: OpSchema) -> OutputSharding:
  """Propagates DTensor sharding spec for sparse_dense_matmul."""
  row_pointers_spec = op_schema.args_schema[0]
  embedding_table_spec = op_schema.args_schema[4]

  world_size = row_pointers_spec.mesh.size()
  global_offsets_size = row_pointers_spec.tensor_meta.shape[0]
  global_batch_size = global_offsets_size - world_size
  embedding_dim = embedding_table_spec.tensor_meta.shape[1]

  output_shape = torch.Size([global_batch_size, embedding_dim])
  output_stride = (embedding_dim, 1)

  tensor_meta = TensorMeta(
      shape=output_shape,
      stride=output_stride,
      dtype=embedding_table_spec.tensor_meta.dtype,
  )

  output_spec = DTensorSpec.from_dim_map(
      mesh=row_pointers_spec.mesh,
      dim_map=row_pointers_spec.dim_map,
      sums=row_pointers_spec.sums,
      tensor_meta=tensor_meta,
  )
  return OutputSharding(output_spec)


@register_prop_rule(torch.ops.tpu.sparse_dense_matmul_grad_with_sgd.default)
def sparse_dense_matmul_grad_with_sgd_rule(
    op_schema: OpSchema,
) -> OutputSharding:
  """Propagates DTensor sharding spec for sparse_dense_matmul_grad_with_sgd."""
  return OutputSharding(None, op_schema)
