import os

from absl.testing import absltest
import torch
import torch_tpu
import torch_tpu.api as tm
from torch_tpu._internal.raw_transfer_impl import (
    transfer_d2h_async,
    transfer_d2h_batch_async,
    PjRtCopyFuture,
)

class RawTransferTest(absltest.TestCase):
    def setUp(self):
        self.device = tm.tpu_device()

    def test_d2h_full_copy(self):
        # Create a TPU tensor
        tpu_tensor = torch.arange(10, dtype=torch.float32, device=self.device)

        # Create a CPU pinned tensor. We allocate a large buffer because TPU raw transfers
        # transfer the entire physically padded buffer.
        cpu_tensor_large = torch.empty(4096 * 2, dtype=torch.float32, pin_memory=True)

        # Trigger async transfer
        future = transfer_d2h_async(tpu_tensor, cpu_tensor_large)
        self.assertTrue(isinstance(future, PjRtCopyFuture))

        # Wait for completion
        future.wait()

        # Check values
        torch.testing.assert_close(cpu_tensor_large[:10], torch.arange(10, dtype=torch.float32))

    def test_d2h_batch_copy(self):
        tpu_tensors = [torch.ones(100, dtype=torch.float32, device=self.device) * i for i in range(5)]
        cpu_tensors_large = [torch.empty(4096 * 2, dtype=torch.float32, pin_memory=True) for _ in range(5)]

        future = transfer_d2h_batch_async(tpu_tensors, cpu_tensors_large)
        future.wait()

        for i in range(5):
            torch.testing.assert_close(cpu_tensors_large[i][:100], torch.ones(100, dtype=torch.float32) * i)

    def test_d2h_partial_copy(self):
        # Create a tiled shape, typically 3D helps to bypass certain restrictions if tiling is applied
        tpu_tensor = torch.arange(128 * 128 * 8, dtype=torch.float32, device=self.device).view(8, 128, 128)

        # We want to copy the slice tpu_tensor[2:4, :, :]
        # Each major dim slice is 128*128 elements = 16384 floats = 65536 bytes (multiple of 4096)
        cpu_tensor = torch.empty(2, 128, 128, dtype=torch.float32, pin_memory=True)

        # Copy 2 slices, starting from index 2 on TPU and index 0 on CPU
        future = transfer_d2h_async(
            tpu_tensor, 
            cpu_tensor, 
            src_offsets_major_dim=[2], 
            dst_offsets_major_dim=[0], 
            copy_sizes_major_dim=[2]
        )
        future.wait()

        expected = torch.arange(128 * 128 * 8, dtype=torch.float32).view(8, 128, 128)[2:4, :, :]
        torch.testing.assert_close(cpu_tensor, expected)

if __name__ == '__main__':
    absltest.main()
