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

"""Tests for ragged_all_to_all in a distributed environment on TPU."""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def run_ragged_all_to_all_test() -> None:
  """Tests standard symmetric 1-element ragged_all_to_all across ranks.

  What this test does:
    1. Initializes an 8-rank TPU distributed process group.
    2. Each rank i holds world_size elements: [i*world_size .. (i+1)*world_size
    - 1].
    3. Each rank sends exactly 1 element to every destination peer j, sourced
       from local offset j in its operand buffer.
    4. Destination peer j places rank i's element at output offset i.

  Expected output:
    On rank r, output buffer receives element r from each source rank src,
    placed
    at index src. The resulting tensor is [r, r + world_size, ..., r +
    (world_size - 1) * world_size],
    representing a distributed transpose of the operand grid across ranks.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # Remote destination offsets: rank i places its slice on destination dst at offset rank.
  output_offsets = torch.tensor([rank] * world_size, dtype=torch.int32).tpu()

  # Symmetric 1-element communication: send 1 element to each peer, recv 1 from each.
  send_sizes = torch.tensor([1] * world_size, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(range(world_size), dtype=torch.int32).tpu()
  recv_sizes = torch.tensor([1] * world_size, dtype=torch.int32).tpu()

  offset = rank * world_size
  operand = torch.arange(offset, offset + world_size, dtype=torch.int32).tpu()
  output = torch.zeros(world_size, dtype=torch.int32).tpu()

  result = torch.ops.tpu.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      dist.group.WORLD.group_name,
  )

  # Expected: distributed matrix transpose where rank r receives elements
  # with stride world_size starting from r.
  expected = torch.arange(
      rank,
      rank + world_size * world_size,
      world_size,
      dtype=torch.int32,
  )

  utils.assert_close(result.cpu(), expected)

  dist.barrier()
  dist.destroy_process_group()


def run_ragged_all_to_all_uneven_test() -> None:
  """Tests the lower-level torch.ops.tpu.ragged_all_to_all custom call with uneven splits.

  What this test does:
    1. Exercises the underlying TPU custom call directly with non-uniform
    message sizes.
    2. Uses a communication matrix where rank i sends 2 elements to
    (i+1)%world_size
       and 1 element to all other ranks.
    3. Manually calculates:
       - send_sizes and recv_sizes: non-uniform slice sizes.
       - input_offsets: starting index in local operand for each destination's
       outgoing slice.
       - output_offsets: exact destination buffer offset on the remote peer,
       accounting
         for cumulative sizes of slices received from preceding ranks (src <
         rank).

  Expected output:
    On destination rank r, output buffer receives slices from all source ranks
    src = 0..world_size - 1 placed contiguously in source-rank order. Each slice
    contains elements [src * 100 + offset .. src * 100 + offset + count] sent
    from src.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # Communication split matrix: matrix[src][dst] is the number of elements
  # sent from src to dst. Rank i sends 2 elements to (i+1)%world_size, 1 to others.
  matrix = [
      [(2 if j == (i + 1) % world_size else 1) for j in range(world_size)]
      for i in range(world_size)
  ]
  send_sizes_list = matrix[rank]
  recv_sizes_list = [matrix[src][rank] for src in range(world_size)]

  # Calculate local input offsets for each destination slice in operand.
  input_offsets_list = []
  curr = 0
  for s in send_sizes_list:
    input_offsets_list.append(curr)
    curr += s

  # Calculate remote output offsets: where each outgoing slice will be placed
  # on destination `dst`. On rank `dst`, slices from src=0..rank-1 arrive before `rank`.
  output_offsets_list = []
  for dst in range(world_size):
    dst_offset = sum(matrix[src][dst] for src in range(rank))
    output_offsets_list.append(dst_offset)

  send_sizes = torch.tensor(send_sizes_list, dtype=torch.int32).tpu()
  recv_sizes = torch.tensor(recv_sizes_list, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(input_offsets_list, dtype=torch.int32).tpu()
  output_offsets = torch.tensor(output_offsets_list, dtype=torch.int32).tpu()

  total_send = sum(send_sizes_list)
  total_recv = sum(recv_sizes_list)

  # Initialize unique input elements: rank 0 gets 0..N, rank 1 gets 100..100+N, etc.
  operand = torch.arange(
      rank * 100, rank * 100 + total_send, dtype=torch.int32
  ).tpu()
  output = torch.zeros(total_recv, dtype=torch.int32).tpu()

  # Invoke the ragged_all_to_all ATen custom call.
  result = torch.ops.tpu.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      dist.group.WORLD.group_name,
  )

  # Construct expected output buffer from received slices.
  expected_slices = []
  for src in range(world_size):
    count = matrix[src][rank]
    src_send_offset = sum(matrix[src][d] for d in range(rank))
    expected_slices.append(
        torch.arange(
            src * 100 + src_send_offset,
            src * 100 + src_send_offset + count,
            dtype=torch.int32,
        )
    )
  expected = torch.cat(expected_slices)
  utils.assert_close(result.cpu(), expected)

  dist.barrier()
  dist.destroy_process_group()


def run_dist_all_to_all_single_uneven_test() -> None:
  """Tests high-level torch.distributed.all_to_all_single with uneven split sizes.

  What this test does:
    1. Exercises the high-level PyTorch API torch.distributed.all_to_all_single
    with
       non-uniform split sizes (send_sizes_list and recv_sizes_list).
    2. Validates that ProcessGroupTpu automatically calculates input and output
    offsets
       and dispatches the lower-level TPU collective.
    3. Uses an uneven communication pattern: rank i sends 2 elements to
    (i+1)%world_size
       and 1 element to all other ranks.

  Expected output:
    The output buffer matches the reconstructed ground-truth tensor formed by
    concatenating the exact variable-length slices received from all source
    ranks
    src = 0..world_size - 1 in rank order.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # 1. Define the uneven communication pattern.
  # matrix[src][dst] defines the number of elements sent from `src` to `dst`.
  # Here, rank i sends 2 elements to peer (i+1)%world_size, and 1 element to all others.
  matrix = [
      [(2 if j == (i + 1) % world_size else 1) for j in range(world_size)]
      for i in range(world_size)
  ]
  # Slice sizes sent from this rank to all destination ranks.
  send_sizes_list = matrix[rank]
  # Slice sizes this rank expects to receive from all source ranks.
  recv_sizes_list = [matrix[src][rank] for src in range(world_size)]

  total_send = sum(send_sizes_list)
  total_recv = sum(recv_sizes_list)

  # 2. Populate unique, rank-identifiable input data.
  # For instance, rank 0 has values 0..total_send, rank 1 has 100..100+total_send, etc.
  operand = torch.arange(
      rank * 100, rank * 100 + total_send, dtype=torch.int32
  ).tpu()
  # Pre-allocate output buffer matching total received element count across all sources.
  output = torch.zeros(total_recv, dtype=torch.int32).tpu()

  # 3. Invoke standard PyTorch collective with non-uniform split sizes.
  # ProcessGroupTpu calculates local/remote offsets and lowers to ragged_all_to_all.
  torch.distributed.all_to_all_single(
      output,
      operand,
      recv_sizes_list,
      send_sizes_list,
  )

  # 4. Reconstruct ground-truth expected buffer from the split matrix.
  # For each source rank `src`, extract the exact slice that `src` sent to `rank`.
  expected_slices = []
  for src in range(world_size):
    count = matrix[src][rank]
    # Calculate offset in source rank's operand where the slice sent to `rank` begins.
    src_send_offset = sum(matrix[src][d] for d in range(rank))
    expected_slices.append(
        torch.arange(
            src * 100 + src_send_offset,
            src * 100 + src_send_offset + count,
            dtype=torch.int32,
        )
    )
  expected = torch.cat(expected_slices)

  # 5. Verify the received output matches the expected tensor values.
  utils.assert_close(output.cpu(), expected)

  dist.barrier()
  dist.destroy_process_group()


def run_ragged_all_to_all_autograd_test() -> None:
  """Tests the backward autograd pass of torch.ops.tpu.ragged_all_to_all with uneven splits.

  What this test does:
    1. Evaluates autograd backward gradient propagation through
    ragged_all_to_all.
    2. Uses an uneven split communication pattern (2 elements to peer
    (i+1)%world_size,
       1 element to others).
    3. Backpropagates a rank-identifiable grad_output:
       [(rank + 1) * 1000.0 .. (rank + 1) * 1000.0 + total_recv - 1].
    4. Exercises TPU-native offset transposition
    (ProcessGroupTpu::alltoall_base)
       which transposes destination offsets back to source positions across
       ranks.

  Expected output:
    operand.grad matches the ground-truth gradient slices where each destination
    dst routes its grad_output[dst_offset : dst_offset + length] back into
    operand.grad[input_offsets[dst] : input_offsets[dst] + length].
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # Communication split matrix: matrix[src][dst] is the number of elements
  # sent from src to dst. Rank i sends 2 elements to (i+1)%world_size, 1 to others.
  matrix = [
      [(2 if j == (i + 1) % world_size else 1) for j in range(world_size)]
      for i in range(world_size)
  ]
  send_sizes_list = matrix[rank]
  recv_sizes_list = [matrix[src][rank] for src in range(world_size)]

  # Calculate local input offsets for each destination slice in operand.
  input_offsets_list = []
  curr = 0
  for s in send_sizes_list:
    input_offsets_list.append(curr)
    curr += s

  # Calculate remote output offsets: where each outgoing slice will be placed
  # on destination `dst`. On rank `dst`, slices from src=0..rank-1 arrive before `rank`.
  output_offsets_list = []
  for dst in range(world_size):
    dst_offset = sum(matrix[src][dst] for src in range(rank))
    output_offsets_list.append(dst_offset)

  send_sizes = torch.tensor(send_sizes_list, dtype=torch.int32).tpu()
  recv_sizes = torch.tensor(recv_sizes_list, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(input_offsets_list, dtype=torch.int32).tpu()
  output_offsets = torch.tensor(output_offsets_list, dtype=torch.int32).tpu()

  total_send = sum(send_sizes_list)
  total_recv = sum(recv_sizes_list)

  # Initialize input operand with requires_grad=True.
  operand = (
      torch.arange(rank * 100, rank * 100 + total_send, dtype=torch.float32)
      .tpu()
      .requires_grad_(True)
  )
  output = torch.zeros(total_recv, dtype=torch.float32).tpu()

  # Invoke the ragged_all_to_all ATen custom call.
  result = torch.ops.tpu.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      dist.group.WORLD.group_name,
  )

  # Construct rank-identifiable grad_output.
  grad_output = torch.arange(
      (rank + 1) * 1000.0,
      (rank + 1) * 1000.0 + total_recv,
      dtype=torch.float32,
  ).tpu()

  result.backward(grad_output)

  # Verify operand.grad against analytical backward routing.
  # On destination `dst`, the slice received from `rank` starts at `dst_offset`
  # with length `matrix[rank][dst]`.
  # In backward, `dst` routes its `grad_output[dst_offset : dst_offset + length]`
  # back to `rank` into `operand.grad[input_offsets[dst] : input_offsets[dst] + length]`.
  expected_grad_slices = []
  for dst in range(world_size):
    count = matrix[rank][dst]
    dst_offset = sum(matrix[src][dst] for src in range(rank))
    expected_grad_slices.append(
        torch.arange(
            (dst + 1) * 1000.0 + dst_offset,
            (dst + 1) * 1000.0 + dst_offset + count,
            dtype=torch.float32,
        )
    )
  expected_grad = torch.cat(expected_grad_slices)
  utils.assert_close(operand.grad.cpu(), expected_grad)

  dist.barrier()
  dist.destroy_process_group()


def run_ragged_all_to_all_autograd_zero_sized_and_padding_test() -> None:
  """Tests ragged_all_to_all backward with zero-sized slices and unsent padding.

  What this test does:
    1. Introduces zero-sized communication slices between specific rank pairs
       (where (i + j) % 3 == 0) to verify edge-case handling of empty transfers.
    2. Adds 5 unsent trailing padding elements to the operand buffer.
    3. Executes forward ragged_all_to_all and backpropagates grad_output.

  Expected output:
    1. Sent slices (operand.grad[:total_send]) receive correctly routed
    gradients
       matching the analytical backward slice reconstruction.
    2. Zero-sized communication channels transfer zero data without errors or
    crashes.
    3. Unsent padding elements (operand.grad[total_send:]) receive strictly zero
       gradient (torch.zeros(pad_len)), ensuring no out-of-bounds accumulation
       occurs.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # Asymmetric split matrix containing non-uniform sizes and zero-sized slices.
  matrix = [
      [
          (3 if j == (i + 1) % world_size else (0 if (i + j) % 3 == 0 else 1))
          for j in range(world_size)
      ]
      for i in range(world_size)
  ]
  send_sizes_list = matrix[rank]
  recv_sizes_list = [matrix[src][rank] for src in range(world_size)]

  input_offsets_list = []
  curr = 0
  for s in send_sizes_list:
    input_offsets_list.append(curr)
    curr += s

  output_offsets_list = []
  for dst in range(world_size):
    dst_offset = sum(matrix[src][dst] for src in range(rank))
    output_offsets_list.append(dst_offset)

  send_sizes = torch.tensor(send_sizes_list, dtype=torch.int32).tpu()
  recv_sizes = torch.tensor(recv_sizes_list, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(input_offsets_list, dtype=torch.int32).tpu()
  output_offsets = torch.tensor(output_offsets_list, dtype=torch.int32).tpu()

  total_send = sum(send_sizes_list)
  total_recv = sum(recv_sizes_list)

  # Allocate operand with extra unsent padding elements at the end.
  pad_len = 5
  operand = (
      torch.arange(
          rank * 100.0,
          rank * 100.0 + total_send + pad_len,
          dtype=torch.float32,
      )
      .tpu()
      .requires_grad_(True)
  )
  output = torch.zeros(total_recv, dtype=torch.float32).tpu()

  result = torch.ops.tpu.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      dist.group.WORLD.group_name,
  )

  grad_output = torch.arange(
      (rank + 1) * 1000.0,
      (rank + 1) * 1000.0 + total_recv,
      dtype=torch.float32,
  ).tpu()

  result.backward(grad_output)

  # Verify sent slices in operand.grad.
  expected_grad_slices = []
  for dst in range(world_size):
    count = matrix[rank][dst]
    if count > 0:
      dst_offset = sum(matrix[src][dst] for src in range(rank))
      expected_grad_slices.append(
          torch.arange(
              (dst + 1) * 1000.0 + dst_offset,
              (dst + 1) * 1000.0 + dst_offset + count,
              dtype=torch.float32,
          )
      )
  expected_sent_grad = torch.cat(expected_grad_slices)
  utils.assert_close(operand.grad[:total_send].cpu(), expected_sent_grad)

  # Verify unsent padding elements receive strictly zero gradient.
  utils.assert_close(
      operand.grad[total_send:].cpu(),
      torch.zeros(pad_len, dtype=torch.float32),
  )

  dist.barrier()
  dist.destroy_process_group()


def run_ragged_all_to_all_autograd_bfloat16_and_chained_test() -> None:
  """Tests autograd backward with bfloat16 dtype and chained operations.

  What this test does:
    1. Tests backward pass with torch.bfloat16 data precision.
    2. Chains operations before and after ragged_all_to_all:
       - Pre-collective:  z = operand * 2.0
       - Collective:      res = ragged_all_to_all(z, ...)
       - Post-collective: loss = ((res + 1.0) ** 2).sum()
    3. Calls loss.backward() to propagate gradients through both math ops and
    collective.

  Expected output:
    1. operand.grad is preserved as torch.bfloat16.
    2. By the chain rule: d/dx (2x + 1)^2 = 4(2x + 1) = 8x + 4.
       operand.grad numerically matches 8.0 * operand + 4.0 within bfloat16
       tolerance (atol=1e-2, rtol=1e-2).
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # Cyclic communication pattern: rank i sends 3 tokens to (i+1)%world_size,
  # and 0 tokens to all other peers.
  matrix = [
      [(3 if j == (i + 1) % world_size else 0) for j in range(world_size)]
      for i in range(world_size)
  ]
  send_sizes_list = matrix[rank]
  recv_sizes_list = [matrix[src][rank] for src in range(world_size)]

  input_offsets_list = []
  curr = 0
  for s in send_sizes_list:
    input_offsets_list.append(curr)
    curr += s

  output_offsets_list = []
  for dst in range(world_size):
    dst_offset = sum(matrix[src][dst] for src in range(rank))
    output_offsets_list.append(dst_offset)

  send_sizes = torch.tensor(send_sizes_list, dtype=torch.int32).tpu()
  recv_sizes = torch.tensor(recv_sizes_list, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(input_offsets_list, dtype=torch.int32).tpu()
  output_offsets = torch.tensor(output_offsets_list, dtype=torch.int32).tpu()

  total_send = sum(send_sizes_list)
  total_recv = sum(recv_sizes_list)

  # Initialize input operand in bfloat16 with requires_grad=True.
  operand = (
      torch.arange(
          rank * 10.0,
          rank * 10.0 + total_send,
          dtype=torch.bfloat16,
      )
      .tpu()
      .requires_grad_(True)
  )
  output = torch.zeros(total_recv, dtype=torch.bfloat16).tpu()

  # Chain an operation before the collective: z = operand * 2.0
  z = operand * 2.0

  # Invoke ragged_all_to_all on intermediate tensor z
  res = torch.ops.tpu.ragged_all_to_all(
      z,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      dist.group.WORLD.group_name,
  )

  # Chain an operation after the collective: loss = ((res + 1.0) ** 2).sum()
  loss = ((res + 1.0) ** 2).sum()
  loss.backward()

  # Verify gradient dtype is bfloat16.
  assert (
      operand.grad.dtype == torch.bfloat16
  ), f"Expected bfloat16, got {operand.grad.dtype}"

  # Analytical gradient via chain rule: d/dx (2x + 1)^2 = 4(2x + 1) = 8x + 4
  expected_grad = 8.0 * operand.detach().cpu().float() + 4.0
  utils.assert_close(
      operand.grad.cpu().float(),
      expected_grad,
      atol=1e-2,
      rtol=1e-2,
  )

  dist.barrier()
  dist.destroy_process_group()


def run_ragged_all_to_all_autograd_2d_test() -> None:
  """Tests ragged_all_to_all autograd with 2D tensors [tokens, hidden_dim].

  What this test does:
    1. Tests autograd backward propagation on multi-dimensional tensors with
    shape
       [total_tokens, hidden_dim] (here hidden_dim = 16), which models
       Mixture-of-Experts
       (MoE) token embedding routing.
    2. Uses an uneven token split matrix where rank i sends variable numbers of
    tokens
       to each peer rank.
    3. Backpropagates a 2D grad_output tensor shaped [total_recv, hidden_dim].

  Expected output:
    operand.grad has shape [total_send, hidden_dim] and matches the analytically
    routed 2D gradient slices, verifying that multidimensional feature vectors
    are preserved and correctly routed back to source token positions.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  hidden_dim = 16
  # Uneven split matrix: rank i sends 2 tokens to (i+1)%world_size, 1 to others.
  matrix = [
      [(2 if j == (i + 1) % world_size else 1) for j in range(world_size)]
      for i in range(world_size)
  ]
  send_sizes_list = matrix[rank]
  recv_sizes_list = [matrix[src][rank] for src in range(world_size)]

  input_offsets_list = []
  curr = 0
  for s in send_sizes_list:
    input_offsets_list.append(curr)
    curr += s

  output_offsets_list = []
  for dst in range(world_size):
    dst_offset = sum(matrix[src][dst] for src in range(rank))
    output_offsets_list.append(dst_offset)

  send_sizes = torch.tensor(send_sizes_list, dtype=torch.int32).tpu()
  recv_sizes = torch.tensor(recv_sizes_list, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(input_offsets_list, dtype=torch.int32).tpu()
  output_offsets = torch.tensor(output_offsets_list, dtype=torch.int32).tpu()

  total_send = sum(send_sizes_list)
  total_recv = sum(recv_sizes_list)

  base_tokens = (
      torch.arange(
          rank * 100.0,
          rank * 100.0 + total_send,
          dtype=torch.float32,
      )
      .unsqueeze(1)
      .expand(-1, hidden_dim)
  )
  feature_offsets = torch.arange(hidden_dim, dtype=torch.float32).unsqueeze(0)
  operand = (base_tokens + feature_offsets).tpu().requires_grad_(True)
  output = torch.zeros((total_recv, hidden_dim), dtype=torch.float32).tpu()

  result = torch.ops.tpu.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      dist.group.WORLD.group_name,
  )

  grad_base = (
      torch.arange(
          (rank + 1) * 1000.0,
          (rank + 1) * 1000.0 + total_recv,
          dtype=torch.float32,
      )
      .unsqueeze(1)
      .expand(-1, hidden_dim)
  )
  grad_output = (grad_base + feature_offsets).tpu()

  result.backward(grad_output)

  expected_grad_slices = []
  for dst in range(world_size):
    count = matrix[rank][dst]
    dst_offset = sum(matrix[src][dst] for src in range(rank))
    dst_grad_base = (
        torch.arange(
            (dst + 1) * 1000.0 + dst_offset,
            (dst + 1) * 1000.0 + dst_offset + count,
            dtype=torch.float32,
        )
        .unsqueeze(1)
        .expand(-1, hidden_dim)
    )
    expected_grad_slices.append(dst_grad_base + feature_offsets)
  expected_grad = torch.cat(expected_grad_slices, dim=0)

  utils.assert_close(operand.grad.cpu(), expected_grad)

  dist.barrier()
  dist.destroy_process_group()


def run_ragged_all_to_all_autograd_multiple_iterations_test() -> None:
  """Tests multiple sequential forward+backward training iterations.

  What this test does:
    1. Executes 5 sequential training loop steps with fresh input operands and
       backpropagated gradients at each step.
    2. Exercises the process group and TPU collective autograd graph repeatedly
    to
       verify that memory buffers, sequence counters, and autograd contexts
       reset
       cleanly between training iterations.

  Expected output:
    At each step (step 0 through 4), operand.grad exactly matches the
    analytically
    derived step-scaled gradient slices without numerical drift, stale memory
    accumulation, or buffer corruption from previous iterations.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  matrix = [
      [(2 if j == (i + 1) % world_size else 1) for j in range(world_size)]
      for i in range(world_size)
  ]
  send_sizes_list = matrix[rank]
  recv_sizes_list = [matrix[src][rank] for src in range(world_size)]

  input_offsets_list = []
  curr = 0
  for s in send_sizes_list:
    input_offsets_list.append(curr)
    curr += s

  output_offsets_list = []
  for dst in range(world_size):
    dst_offset = sum(matrix[src][dst] for src in range(rank))
    output_offsets_list.append(dst_offset)

  send_sizes = torch.tensor(send_sizes_list, dtype=torch.int32).tpu()
  recv_sizes = torch.tensor(recv_sizes_list, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(input_offsets_list, dtype=torch.int32).tpu()
  output_offsets = torch.tensor(output_offsets_list, dtype=torch.int32).tpu()

  total_send = sum(send_sizes_list)
  total_recv = sum(recv_sizes_list)

  for step in range(5):
    step_scale = float(step + 1)
    operand = (
        torch.arange(
            rank * 100.0 + step * 10.0,
            rank * 100.0 + step * 10.0 + total_send,
            dtype=torch.float32,
        )
        .tpu()
        .requires_grad_(True)
    )
    output = torch.zeros(total_recv, dtype=torch.float32).tpu()

    result = torch.ops.tpu.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        dist.group.WORLD.group_name,
    )

    grad_output = torch.arange(
        (rank + 1) * 1000.0 * step_scale,
        (rank + 1) * 1000.0 * step_scale + total_recv,
        dtype=torch.float32,
    ).tpu()

    result.backward(grad_output)

    expected_grad_slices = []
    for dst in range(world_size):
      count = matrix[rank][dst]
      dst_offset = sum(matrix[src][dst] for src in range(rank))
      expected_grad_slices.append(
          torch.arange(
              (dst + 1) * 1000.0 * step_scale + dst_offset,
              (dst + 1) * 1000.0 * step_scale + dst_offset + count,
              dtype=torch.float32,
          )
      )
    expected_grad = torch.cat(expected_grad_slices)
    utils.assert_close(operand.grad.cpu(), expected_grad)

  dist.barrier()
  dist.destroy_process_group()


def run_ragged_all_to_all_no_grad_test() -> None:
  """Tests execution within torch.no_grad() context.

  What this test does:
    1. Executes torch.ops.tpu.ragged_all_to_all wrapped inside a `with
    torch.no_grad():` block.
    2. Verifies that the ATen autograd wrapper detects that gradient tracking is
    disabled
       and bypasses backward state construction.

  Expected output:
    1. result.requires_grad is False.
    2. result.grad_fn is None (tensor is completely detached from the autograd
    graph).
    3. The output tensor values correctly match the expected strided all-to-all
    transpose
       data [rank, rank + world_size, ..., rank + (world_size - 1) *
       world_size].
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  output_offsets = torch.tensor([rank] * world_size, dtype=torch.int32).tpu()
  send_sizes = torch.tensor([1] * world_size, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(range(world_size), dtype=torch.int32).tpu()
  recv_sizes = torch.tensor([1] * world_size, dtype=torch.int32).tpu()

  offset = rank * world_size
  operand = (
      torch.arange(offset, offset + world_size, dtype=torch.float32)
      .tpu()
      .requires_grad_(True)
  )
  output = torch.zeros(world_size, dtype=torch.float32).tpu()

  with torch.no_grad():
    result = torch.ops.tpu.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        dist.group.WORLD.group_name,
    )

  assert not result.requires_grad
  assert result.grad_fn is None

  expected = torch.arange(
      rank,
      rank + world_size * world_size,
      world_size,
      dtype=torch.float32,
  )
  utils.assert_close(result.cpu(), expected)

  dist.barrier()
  dist.destroy_process_group()


class RaggedAllToAllTest(seed_test_utils.MultiProcessRepeatableTest):
  """Tests the ragged_all_to_all TPU collective operation and autograd.

  Exercises ragged all-to-all communication and backward gradient routing across
  8 TPU ranks under various split geometries, datatypes, multidimensional
  tensors,
  and iteration sequences.
  """

  def test_ragged_all_to_all(self):
    """Tests symmetric 1-element ragged all-to-all resulting in a distributed transpose."""
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(run_ragged_all_to_all_test),
    )

  def test_ragged_all_to_all_uneven(self):
    """Tests ragged all-to-all custom call with non-uniform slice sizes and manual offsets."""
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_ragged_all_to_all_uneven_test
        ),
    )

  def test_ragged_all_to_all_autograd(self):
    """Tests backward autograd gradient routing across ranks with uneven splits."""
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_ragged_all_to_all_autograd_test
        ),
    )

  def test_ragged_all_to_all_autograd_zero_sized_and_padding(self):
    """Tests backward pass with zero-sized communication channels and unsent padding."""
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_ragged_all_to_all_autograd_zero_sized_and_padding_test
        ),
    )

  def test_ragged_all_to_all_autograd_bfloat16_and_chained(self):
    """Tests autograd backward with bfloat16 dtype and chained math operations."""
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_ragged_all_to_all_autograd_bfloat16_and_chained_test
        ),
    )

  def test_dist_all_to_all_single_uneven(self):
    """Tests high-level torch.distributed.all_to_all_single with non-uniform split sizes."""
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dist_all_to_all_single_uneven_test
        ),
    )

  def test_ragged_all_to_all_autograd_2d(self):
    """Tests autograd backward on 2D tensors [tokens, hidden_dim] for MoE routing."""
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_ragged_all_to_all_autograd_2d_test
        ),
    )

  def test_ragged_all_to_all_autograd_multiple_iterations(self):
    """Tests 5 consecutive training iterations verifying state cleanup and memory safety."""
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_ragged_all_to_all_autograd_multiple_iterations_test
        ),
    )

  def test_ragged_all_to_all_no_grad(self):
    """Tests execution inside torch.no_grad() ensuring detached output and no grad state."""
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_ragged_all_to_all_no_grad_test
        ),
    )


if __name__ == "__main__":
  torch.multiprocessing.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
