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

from absl.testing import absltest
import torch
from torch_tpu._internal import batch_transfer
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class BatchTransferTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.device('tpu')

  def test_d2h_full_copy(self):
    # Create a TPU tensor
    tpu_tensor = torch.arange(10, dtype=torch.float32, device=self.device)

    # Create a CPU pinned tensor for receiving data.
    # TPU device buffers are padded to hardware-specific boundaries. For small
    # tensors, allocation is typically padded to 4KB chunks.
    # Batch D2H transfer copies the entire padded buffer, so host buffer must be
    # large enough to hold it. We expect a 4KB buffer for tensors like
    # arange(10), but allocate 32KB (4096*2*sizeof(float32)) to be safe against
    # padding variations across hardware and tensor shapes.
    cpu_tensor_large = torch.empty(
        4096 * 2, dtype=torch.float32, pin_memory=True
    )

    # Trigger async transfer
    future = batch_transfer.batch_transfer_d2h([tpu_tensor], [cpu_tensor_large])
    self.assertIsInstance(future, batch_transfer.TransferFuture)

    # Wait for completion
    future.wait()

    # Check values
    utils.assert_close(
        actual=cpu_tensor_large[:10],
        expected=torch.arange(10, dtype=torch.float32),
    )

  def test_d2h_batch_copy(self):
    tpu_tensors = [
        torch.ones(100, dtype=torch.float32, device=self.device) * i
        for i in range(5)
    ]
    # TPU device buffers are padded to hardware-specific boundaries. For small
    # tensors, allocation is typically padded to 4KB chunks.
    # Batch D2H transfer copies the entire padded buffer, so host buffer must be
    # large enough to hold it. We expect a 4KB buffer for tensors like
    # ones(100), but allocate 32KB (4096*2*sizeof(float32)) to be safe against
    # padding variations across hardware and tensor shapes.
    cpu_tensors_large = [
        torch.empty(4096 * 2, dtype=torch.float32, pin_memory=True)
        for _ in range(5)
    ]

    future = batch_transfer.batch_transfer_d2h(tpu_tensors, cpu_tensors_large)
    future.wait()

    for i in range(5):
      utils.assert_close(
          actual=cpu_tensors_large[i][:100],
          expected=torch.ones(100, dtype=torch.float32) * i,
      )

  def test_d2h_h2d_partial_roundtrip(self):
    # Create a tiled shape, typically 3D helps to bypass certain
    # restrictions if tiling is applied
    tpu_tensor = torch.arange(
        128 * 128 * 8, dtype=torch.float32, device=self.device
    ).view(8, 128, 128)

    # We want to copy the slice tpu_tensor[2:4, :, :]
    # Each major dim slice is 128*128 elements = 16384 floats = 65536 bytes
    # (multiple of 4096)
    cpu_tensor = torch.empty(2, 128, 128, dtype=torch.float32, pin_memory=True)

    # Copy 2 slices, starting from index 2 on TPU and index 0 on CPU
    d2h_future = batch_transfer.batch_transfer_d2h(
        [tpu_tensor],
        [cpu_tensor],
        src_offsets_major_dim=[2],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[2],
    )
    d2h_future.wait()

    # Now copy back to TPU to validate logical values, rather than interpreting
    # physical layout on CPU buffer directly.
    tpu_tensor_dst = torch.empty(
        2, 128, 128, dtype=torch.float32, device=self.device
    )
    h2d_future = batch_transfer.batch_transfer_h2d(
        [cpu_tensor],
        [tpu_tensor_dst],
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[2],
    )
    h2d_future.wait()

    expected = torch.arange(128 * 128 * 8, dtype=torch.float32).view(
        8, 128, 128
    )[2:4, :, :]
    utils.assert_close(actual=tpu_tensor_dst.cpu(), expected=expected)

  def test_h2d_full_copy(self):
    cpu_data = torch.arange(10, dtype=torch.float32)
    # For H2D batch transfer, host buffer must be sized to TPU padded buffer
    # size. TPU device buffers are padded to hardware-specific boundaries. For
    # small tensors, allocation is typically padded to 4KB chunks. Batch H2D
    # transfer copies bytes for entire padded buffer from host, so host buffer
    # must be large enough. We expect a 4KB buffer for tensors like arange(10),
    # but allocate 32KB (4096*2*sizeof(float32)) to be safe against padding
    # variations across hardware and tensor shapes.
    cpu_tensor_large = torch.zeros(
        4096 * 2, dtype=torch.float32, pin_memory=True
    )
    cpu_tensor_large[:10] = cpu_data
    tpu_tensor = torch.empty(10, dtype=torch.float32, device=self.device)

    future = batch_transfer.batch_transfer_h2d([cpu_tensor_large], [tpu_tensor])
    future.wait()

    utils.assert_close(tpu_tensor.cpu(), cpu_data)

  def test_d2h_h2d_roundtrip(self):
    tpu_tensor_src = torch.arange(100, dtype=torch.float32, device=self.device)
    cpu_tensor_stage = torch.empty(
        4096 * 2, dtype=torch.float32, pin_memory=True
    )
    tpu_tensor_dst = torch.empty(100, dtype=torch.float32, device=self.device)

    d2h_future = batch_transfer.batch_transfer_d2h(
        [tpu_tensor_src], [cpu_tensor_stage]
    )
    d2h_future.wait()

    h2d_future = batch_transfer.batch_transfer_h2d(
        [cpu_tensor_stage], [tpu_tensor_dst]
    )
    h2d_future.wait()

    utils.assert_close(tpu_tensor_dst.cpu(), tpu_tensor_src.cpu())

  def test_d2h_h2d_batch_roundtrip(self):
    tpu_tensors_src = [
        torch.ones(100, dtype=torch.float32, device=self.device) * i
        for i in range(5)
    ]
    cpu_tensors_stage = [
        torch.empty(4096 * 2, dtype=torch.float32, pin_memory=True)
        for _ in range(5)
    ]
    tpu_tensors_dst = [
        torch.empty(100, dtype=torch.float32, device=self.device)
        for _ in range(5)
    ]

    d2h_future = batch_transfer.batch_transfer_d2h(
        tpu_tensors_src, cpu_tensors_stage
    )
    d2h_future.wait()

    h2d_future = batch_transfer.batch_transfer_h2d(
        cpu_tensors_stage, tpu_tensors_dst
    )
    h2d_future.wait()

    for i in range(5):
      utils.assert_close(tpu_tensors_dst[i].cpu(), tpu_tensors_src[i].cpu())

  def test_h2d_retains_dst_buffer_on_overwrite(self):
    cpu_tensor_large = torch.zeros(
        4096 * 2, dtype=torch.float32, pin_memory=True
    )
    cpu_tensor_large[:10] = torch.arange(10, dtype=torch.float32)
    tpu_tensor = torch.empty(10, dtype=torch.float32, device=self.device)

    future = batch_transfer.batch_transfer_h2d([cpu_tensor_large], [tpu_tensor])
    # Replace the tensor's underlying DeviceBufferRef before waiting on the
    # in-flight H2D DMA; TransferFuture must keep the original DeviceBufferRef
    # alive until wait() completes.
    tpu_tensor.zero_()
    future.wait()

  def test_h2d_rejects_non_base_view(self):
    cpu_tensor_large = torch.zeros(
        4096 * 2, dtype=torch.float32, pin_memory=True
    )
    tpu_base = torch.empty(20, dtype=torch.float32, device=self.device)
    tpu_slice = tpu_base[2:12]

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Requires TPU v5lite hardware
        RuntimeError, 'destination TPU tensor must be a contiguous base tensor'
    ):
      batch_transfer.batch_transfer_h2d([cpu_tensor_large], [tpu_slice])


if __name__ == '__main__':
  absltest.main()
