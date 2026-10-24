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

"""Multi-TPU and zero-copy DMA Window unit tests using standard c10d::Window."""

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch_tpu  # pylint: disable=unused-import  # ruff: ignore[unused-import]
from torch_tpu._internal.utils.test_utils import assert_close
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils


def _run_window_backend_support_test() -> None:
  """Verifies dist._supports_window() and backend.supports_window properties.

  Validates that TorchTPU's ProcessGroupTpu correctly advertises window
  RMA capability to PyTorch's distributed runtime.
  """
  assert dist._supports_window()
  backend = dist.get_backend_impl(device=torch.device("tpu:0"))
  assert backend.supports_window


def _run_window_local_h2d_dma_test() -> None:
  """Tests local zero-copy Host-to-Device (H2D) DMA put into registered TPU window.

  Validates that host CPU tensors can be directly transferred via zero-copy
  DMA into pre-registered TPU device memory windows using c10d::Window::put.
  """
  rank = dist.get_rank()
  num_elements = 1024 * 64
  half_elements = num_elements // 2

  # 1. Allocate and register destination TPU memory window.
  tpu_window_tensor = torch.zeros(
      (num_elements,), dtype=torch.bfloat16, device="tpu:0"
  )
  win_tpu = dist._new_window(tpu_window_tensor)

  # 2. Prepare CPU source data with rank-specific payload.
  cpu_source_data = torch.full(
      (half_elements,), float(100 + rank), dtype=torch.bfloat16, device="cpu"
  )

  # 3. Perform asynchronous Host-to-Device (H2D) DMA transfer into target offset.
  h2d_work = win_tpu.put(
      cpu_source_data,
      dst_rank=rank,
      target_offset_nelems=half_elements,
      async_op=True,
  )
  if h2d_work is not None:
    h2d_work.wait()

  # 4. Verify data integrity in the transferred window slice.
  assert_close(
      tpu_window_tensor[half_elements:].cpu(),
      cpu_source_data,
      rtol=1e-3,
      atol=1e-3,
  )

  # 5. Deregister window cleanly.
  win_tpu.tensor_deregister()
  del win_tpu


def _run_window_local_d2h_dma_test() -> None:
  """Tests local zero-copy Device-to-Host (D2H) DMA put from TPU to registered Host window.

  Validates that TPU device tensors are correctly transferred via zero-copy DMA
  into
  specified offsets of registered pinned host memory.
  """
  rank = dist.get_rank()
  num_elements = 1024 * 64
  half_elements = num_elements // 2

  # 1. Allocate page-locked (pinned) CPU host buffer and register host window.
  host_pool = torch.zeros(
      (num_elements,), dtype=torch.bfloat16, device="cpu"
  ).pin_memory()
  win_host = dist._new_window(host_pool)

  # 2. Prepare TPU source data on device with rank-specific payload.
  tpu_source_data = torch.full(
      (half_elements,), float(200 + rank), dtype=torch.bfloat16, device="tpu:0"
  )

  # 3. Perform asynchronous Device-to-Host (D2H) DMA transfer into host offset.
  d2h_work = win_host.put(
      tpu_source_data,
      dst_rank=rank,
      target_offset_nelems=half_elements,
      async_op=True,
  )
  if d2h_work is not None:
    d2h_work.wait()

  # 4. Verify host buffer slice content matches transferred TPU source data.
  expected_data = torch.full(
      (half_elements,), float(200 + rank), dtype=torch.bfloat16, device="cpu"
  )
  assert_close(host_pool[half_elements:], expected_data, rtol=1e-3, atol=1e-3)

  # 5. Deregister window cleanly.
  win_host.tensor_deregister()
  del win_host


def _run_window_async_compute_overlap_test() -> None:
  """Tests asynchronous DMA transfer overlapping with concurrent TPU compute.

  Demonstrates that zero-copy D2H DMA offload executes non-blockingly on
  background DMA engines while the main execution thread continues issuing TPU
  operations.
  """
  rank = dist.get_rank()
  # 8 MB payload (bfloat16): provides realistic DMA transfer volume while
  # keeping memory allocation and page-locking overhead minimal.
  num_elements = 1024 * 1024 * 4

  # 1. Initialize TPU source tensor and pinned host destination window.
  tpu_src = torch.full(
      (num_elements,), float(100 + rank), dtype=torch.bfloat16, device="tpu:0"
  )
  host_dst = torch.zeros(
      (num_elements,), dtype=torch.bfloat16, device="cpu"
  ).pin_memory()
  win_host = dist._new_window(host_dst)

  # 2. Launch asynchronous D2H DMA transfer to host.
  d2h_work = win_host.put(
      tpu_src, dst_rank=rank, target_offset_nelems=0, async_op=True
  )

  # 3. Concurrently perform compute on TPU TensorCores during active DMA transfer.
  # This proves that the host thread returned immediately from put() and the
  # device can execute subsequent matrix compute while DMA is in-flight.
  mat_a = torch.randn((64, 64), dtype=torch.bfloat16, device="tpu:0")
  mat_b = torch.randn((64, 64), dtype=torch.bfloat16, device="tpu:0")
  compute_result = torch.matmul(mat_a, mat_b)
  _ = compute_result.sum().item()

  # 4. Wait for asynchronous D2H DMA transfer completion.
  d2h_work.wait()

  # 5. Verify transferred host data matches source TPU data.
  expected = torch.full(
      (num_elements,), float(100 + rank), dtype=torch.bfloat16, device="cpu"
  )
  assert_close(host_dst, expected, rtol=1e-3, atol=1e-3)

  # 6. Deregister window cleanly.
  win_host.tensor_deregister()
  del win_host


def _run_window_multi_tpu_peer_transfer_test() -> None:
  """Tests multi-TPU peer-to-peer Window RMA transfers across processes.

  Verifies one-sided RMA put operations across distinct TPU ranks over
  Inter-Chip Interconnect (ICI) links synchronized with win.wait_signal().
  """
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  num_elements = 1024 * 32

  # 1. Allocate TPU tensor and register window on all ranks.
  tpu_tensor = torch.full(
      (num_elements,), float(rank * 10 + 1), dtype=torch.float32, device="tpu:0"
  )
  win = dist._new_window(tpu_tensor)

  send_work = None
  recv_work = None
  if rank < world_size - 1:
    # Forward payload to neighboring rank (rank + 1) via RMA put.
    send_work = win.put(
        tpu_tensor,
        dst_rank=rank + 1,
        target_offset_nelems=0,
        async_op=True,
    )
  if rank > 0:
    # Wait for RMA signal and payload arrival from predecessor rank (rank - 1).
    recv_work = win.wait_signal(peer_rank=rank - 1, async_op=True)
    expected = torch.full(
        (num_elements,), float((rank - 1) * 10 + 1), dtype=torch.float32
    )
    local_tensor = win.map_remote_tensor(rank)
    assert_close(local_tensor.cpu(), expected)

  if recv_work is not None:
    recv_work.wait()
  if send_work is not None:
    send_work.wait()

  # Deregister window cleanly.
  win.tensor_deregister()
  del win
  dist.barrier()


def _run_window_map_remote_tensor_test() -> None:
  """Tests win.map_remote_tensor(rank) returns a tensor view of the registered window.

  Validates that querying a rank's window tensor via map_remote_tensor provides
  a valid tensor view with identical shape, dtype, device location, and
  numerical values as the underlying registered TPU tensor.
  """
  rank = dist.get_rank()
  num_elements = 1024 * 64

  # 1. Allocate TPU tensor with rank-specific values and register window.
  tpu_tensor = torch.full(
      (num_elements,),
      float(rank * 100 + 42),
      dtype=torch.float32,
      device="tpu:0",
  )
  win = dist._new_window(tpu_tensor)

  # 2. Map the local rank's tensor view via map_remote_tensor.
  local_tensor = win.map_remote_tensor(rank)

  # 3. Verify that mapped tensor matches dtype, shape, device, and numerical contents.
  assert local_tensor.dtype == tpu_tensor.dtype
  assert local_tensor.shape == tpu_tensor.shape
  assert local_tensor.device == tpu_tensor.device
  assert_close(local_tensor.cpu(), tpu_tensor.cpu())

  # 4. Deregister window cleanly.
  win.tensor_deregister()
  del win


def _run_window_attributes_test() -> None:
  """Tests win.get_attr(peer_rank) returns WindowAttr with UNIFIED access type.

  Validates that querying window attributes for a peer rank returns a valid
  WindowAttr structure with WindowAccessType.UNIFIED memory coherence model.
  """
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # 1. Allocate TPU tensor and register window.
  tpu_tensor = torch.zeros((1024,), dtype=torch.float32, device="tpu:0")
  win = dist._new_window(tpu_tensor)

  # 2. Query attributes for neighboring rank.
  peer_rank = (rank + 1) % world_size
  attr = win.get_attr(peer_rank)
  from torch._C._distributed_c10d import WindowAccessType  # pylint: disable=g-import-not-at-top

  # 3. Assert attribute properties.
  assert attr is not None
  assert attr.access_type == WindowAccessType.UNIFIED

  win.tensor_deregister()
  del win


def _run_window_signal_wait_test() -> None:
  """Tests explicit win.signal(peer_rank) paired with win.wait_signal(peer_rank).

  Validates pure point-to-point signaling between peer ranks in pipeline order
  without accompanying RMA data transfers.
  """
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # 1. Allocate TPU tensor and register window.
  tpu_tensor = torch.zeros((1024,), dtype=torch.float32, device="tpu:0")
  win = dist._new_window(tpu_tensor)

  # 2. Coordinate pipeline signaling from rank 0 -> 1 -> 2 -> ... -> world_size - 1.
  sig_work = None
  wait_work = None
  if rank < world_size - 1:
    sig_work = win.signal(peer_rank=rank + 1, async_op=True)
  if rank > 0:
    wait_work = win.wait_signal(peer_rank=rank - 1, async_op=True)
  if wait_work is not None:
    wait_work.wait()
  if sig_work is not None:
    sig_work.wait()

  # 3. Clean up and synchronize across ranks.
  win.tensor_deregister()
  del win
  dist.barrier()


def _run_window_explicit_register_lifecycle_test() -> None:
  """Tests two-stage window lifecycle: empty creation, tensor_register, and tensor_deregister.

  Validates that an unattached window can be created with dist._new_window(),
  subsequently attached to a device tensor via tensor_register, used for
  RMA views, and cleanly deregistered with tensor_deregister.
  """
  rank = dist.get_rank()
  num_elements = 1024 * 64

  # 1. Create an unattached window without providing an initial tensor.
  win = dist._new_window()

  # 2. Register TPU device tensor into the existing window.
  tpu_tensor = torch.full(
      (num_elements,), float(rank + 1), dtype=torch.float32, device="tpu:0"
  )
  win.tensor_register(tpu_tensor)

  # 3. Verify mapped tensor view against original tensor.
  local_tensor = win.map_remote_tensor(rank)
  assert_close(local_tensor.cpu(), tpu_tensor.cpu())

  # 4. Deregister tensor and verify window cleanup.
  win.tensor_deregister()
  del win


def _run_window_sync_dma_test() -> None:
  """Tests synchronous (blocking) Host-to-Device and Device-to-Host DMA transfers with async_op=False.

  Validates that setting async_op=False causes put operations to synchronously
  block until the hardware DMA transfer completes before returning.
  """
  rank = dist.get_rank()
  num_elements = 1024 * 64

  # 1. Host-to-Device synchronous DMA put.
  tpu_tensor = torch.zeros(
      (num_elements,), dtype=torch.bfloat16, device="tpu:0"
  )
  win_tpu = dist._new_window(tpu_tensor)

  cpu_src = torch.full(
      (num_elements,), float(rank + 5), dtype=torch.bfloat16, device="cpu"
  )
  win_tpu.put(cpu_src, dst_rank=rank, target_offset_nelems=0, async_op=False)
  assert_close(tpu_tensor.cpu(), cpu_src)

  # 2. Device-to-Host synchronous DMA put.
  host_dst = torch.zeros(
      (num_elements,), dtype=torch.bfloat16, device="cpu"
  ).pin_memory()
  win_host = dist._new_window(host_dst)

  win_host.put(
      tpu_tensor, dst_rank=rank, target_offset_nelems=0, async_op=False
  )
  assert_close(host_dst, cpu_src)

  win_tpu.tensor_deregister()
  win_host.tensor_deregister()


class WindowTpuTest(continuous_test_utils.TpuMultiProcContinuousTest):
  """Unit test suite for TorchTPU native c10d::Window and zero-copy DMA."""

  world_size = 8

  @classmethod
  def setUpClass(cls):
    if not hasattr(dist, "_supports_window"):
      raise absltest.SkipTest(
          "c10d::Window RMA is not supported in this PyTorch version "
          f"({torch.__version__}); requires PyTorch >= 2.14."
      )
    super().setUpClass()

  def test_window_backend_support(self):
    """Verifies dist._supports_window() and backend.supports_window flags."""
    _run_window_backend_support_test()

  def test_window_local_h2d_dma(self):
    """Verifies local zero-copy Host-to-Device (H2D) DMA transfers into TPU windows."""
    _run_window_local_h2d_dma_test()

  def test_window_local_d2h_dma(self):
    """Verifies local zero-copy Device-to-Host (D2H) DMA transfers from TPU to host windows."""
    _run_window_local_d2h_dma_test()

  def test_window_sync_dma(self):
    """Verifies synchronous (blocking, async_op=False) DMA transfers."""
    _run_window_sync_dma_test()

  def test_window_async_compute_overlap(self):
    """Verifies non-blocking DMA execution concurrently overlapping with TPU matrix compute."""
    _run_window_async_compute_overlap_test()

  def test_window_multi_tpu_peer_transfer(self):
    """Verifies multi-TPU peer RMA transfers across distinct processes over ICI."""
    _run_window_multi_tpu_peer_transfer_test()

  def test_window_map_remote_tensor(self):
    """Verifies win.map_remote_tensor returns correct tensor view of registered window."""
    _run_window_map_remote_tensor_test()

  def test_window_attributes(self):
    """Verifies win.get_attr returns WindowAttr with UNIFIED memory access type."""
    _run_window_attributes_test()

  def test_window_explicit_register_lifecycle(self):
    """Verifies two-stage lifecycle: empty new_window, tensor_register, and tensor_deregister."""
    _run_window_explicit_register_lifecycle_test()

  def test_window_signal_wait(self):
    """Verifies win.signal paired with win.wait_signal across peer ranks."""
    _run_window_signal_wait_test()


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
